"""PySide6 diagnostics GUI for the NeuroPawn Knight IMU board."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .brainflow_adapter import BoardConnectionError, KnightBrainFlowAdapter
from .config import AppConfig, ImuConfig, MouseConfig, load_config
from .diagnostics import active_exg_summary_text, write_snapshot
from .jaw_calibration import GuidedJawCalibration
from .jaw_model import JawPrediction, JawPredictor, train_profile
from .jaw_eval import discover_usable_sessions, train_and_evaluate_profile
from .mouse_control import MouseVelocity, QtCursorController
from .orientation import OrientationEstimator
from .recording import SessionRecorder
from .rows import (
    ACCEL_ROWS,
    GYRO_ROWS,
    LOFF_STATN_ROW,
    LOFF_STATP_ROW,
    MAG_ROWS,
    NUM_ROWS,
    ROW_LABELS,
    SAMPLING_RATE_HZ,
    loff_bits,
    row_stats,
)

try:
    import pyqtgraph.opengl as gl
except Exception:  # pragma: no cover - depends on local OpenGL support
    gl = None


class StreamWorker(QObject):
    data_ready = Signal(object)
    connected = Signal(object)
    error = Signal(str)
    log = Signal(str)
    finished = Signal()

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self._running = False
        self.adapter: KnightBrainFlowAdapter | None = None

    @Slot()
    def run(self) -> None:
        self._running = True
        self.adapter = KnightBrainFlowAdapter(self.config, log=self.log.emit)
        try:
            self.adapter.connect()
            self.connected.emit(
                {
                    "port": self.adapter.port,
                    "brainflow_version": self.adapter.brainflow_version,
                    "board_descr": self.adapter.board_descr,
                }
            )
            while self._running:
                batch = self.adapter.drain()
                if batch.shape[1] > 0:
                    self.data_ready.emit(batch)
                time.sleep(0.02)
        except BoardConnectionError as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(f"stream worker failed: {exc}")
        finally:
            if self.adapter is not None:
                self.adapter.close()
            self.finished.emit()

    def stop(self) -> None:
        self._running = False


class DiagnosticsWindow(QMainWindow):
    def __init__(self, config_path: str) -> None:
        super().__init__()
        self.config_path = config_path
        self.config = load_config(config_path)
        self.estimator = OrientationEstimator(self.config.imu)
        self.cursor_controller = QtCursorController(self.config.mouse)
        self.recorder = SessionRecorder()
        self.worker: StreamWorker | None = None
        self.thread: QThread | None = None
        self.buffer = np.zeros((NUM_ROWS, 0), dtype=float)
        self.max_buffer_samples = SAMPLING_RATE_HZ * 12
        self.total_packets = 0
        self.started_at: float | None = None
        self.cursor_armed = False
        self.last_cursor_step: float | None = None
        self.latest_cursor_velocity = MouseVelocity(0.0, 0.0)
        self.latest_orientation = None
        self.connection_info: dict[str, object] = {}
        self._block_item = None
        self.jaw_calibration: GuidedJawCalibration | None = None
        self.jaw_predictor: JawPredictor | None = None
        self.latest_jaw_prediction = JawPrediction(0.0, "no_model")
        self.jaw_event_count = 0
        self.last_jaw_event_sample = -10_000_000
        self.last_calibration_session: Path | None = None
        self.channel_active_checks: dict[int, QCheckBox] = {}
        self.channel_rld_checks: dict[int, QCheckBox] = {}
        self.channel_gain_boxes: dict[int, QComboBox] = {}
        self.jaw_exg_rows = tuple(int(channel) for channel in (self.config.jaw.channels or [2]))

        self.setWindowTitle("NeuroPawn Knight IMU Diagnostics")
        self.resize(1480, 980)
        self._load_jaw_predictor()
        self._build_ui()
        self._apply_style()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_ui)
        self.timer.start(100)

        self.cursor_timer = QTimer(self)
        self.cursor_timer.timeout.connect(self._drive_cursor)
        self.cursor_timer.start(20)

        self.escape_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.escape_shortcut.activated.connect(self._escape_disarm_cursor)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        root_layout.addWidget(self._build_top_bar())

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(10)
        self.main_splitter.addWidget(self._build_left_panel())
        self.main_splitter.addWidget(self._build_right_scroll_panel())
        self.main_splitter.setStretchFactor(0, 5)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([1120, 520])
        root_layout.addWidget(self.main_splitter, 1)

        self.setCentralWidget(root)

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.start_button = QPushButton("Start Stream")
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_stream)
        self.stop_button.setEnabled(False)
        self.record_button = QPushButton("Start Jaw Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        self.record_label = QComboBox()
        self.record_label.addItems(["neutral", "jaw_clench", "test"])
        self.record_label.setCurrentText("jaw_clench")
        self.calibration_button = QPushButton("Start Jaw Calibration")
        self.calibration_button.clicked.connect(self.start_jaw_calibration)
        self.calibration_button.setEnabled(False)
        self.train_jaw_button = QPushButton("Train Jaw Model")
        self.train_jaw_button.clicked.connect(self.train_jaw_model)
        self.qa_jaw_button = QPushButton("Jaw Model QA")
        self.qa_jaw_button.clicked.connect(self.run_jaw_model_qa)
        self.capture_button = QPushButton("Capture Snapshot")
        self.capture_button.clicked.connect(self.capture_snapshot)
        self.capture_button.setEnabled(False)
        self.zero_button = QPushButton("Zero (level)")
        self.zero_button.clicked.connect(self.zero_orientation)

        self.mag_checkbox = QCheckBox("Use magnetometer")
        self.mag_checkbox.setChecked(self.config.imu.use_magnetometer)
        self.mag_checkbox.stateChanged.connect(self._orientation_controls_changed)

        self.gyro_units = QComboBox()
        self.gyro_units.addItems(["auto", "rad_s", "deg_s"])
        self.gyro_units.setCurrentText(self.config.imu.gyro_units)
        self.gyro_units.currentTextChanged.connect(self._orientation_controls_changed)

        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setDecimals(3)
        self.beta_spin.setSingleStep(0.01)
        self.beta_spin.setRange(0.001, 2.0)
        self.beta_spin.setValue(self.config.imu.beta)
        self.beta_spin.valueChanged.connect(self._orientation_controls_changed)

        self.pivot_spin = QDoubleSpinBox()
        self.pivot_spin.setDecimals(3)
        self.pivot_spin.setSingleStep(0.01)
        self.pivot_spin.setRange(-2.0, 2.0)
        self.pivot_spin.setValue(self.config.imu.pivot_z)
        self.pivot_spin.valueChanged.connect(self._orientation_controls_changed)

        self.swap_checkbox = QCheckBox("Swap pitch/roll")
        self.swap_checkbox.setChecked(self.config.imu.swap_pitch_roll)
        self.swap_checkbox.stateChanged.connect(self._orientation_controls_changed)

        self.invert_checkbox = QCheckBox("Invert roll")
        self.invert_checkbox.setChecked(self.config.imu.invert_roll)
        self.invert_checkbox.stateChanged.connect(self._orientation_controls_changed)

        self.arm_cursor_button = QPushButton("Arm Cursor")
        self.arm_cursor_button.clicked.connect(self.toggle_cursor_arm)

        self.cursor_dead_zone_spin = QDoubleSpinBox()
        self.cursor_dead_zone_spin.setDecimals(1)
        self.cursor_dead_zone_spin.setSingleStep(0.5)
        self.cursor_dead_zone_spin.setRange(0.0, 45.0)
        self.cursor_dead_zone_spin.setValue(self.config.mouse.dead_zone_degrees)
        self.cursor_dead_zone_spin.valueChanged.connect(self._mouse_controls_changed)

        self.cursor_speed_spin = QDoubleSpinBox()
        self.cursor_speed_spin.setDecimals(0)
        self.cursor_speed_spin.setSingleStep(5.0)
        self.cursor_speed_spin.setRange(5.0, 250.0)
        self.cursor_speed_spin.setValue(self.config.mouse.speed_px_per_second_per_degree)
        self.cursor_speed_spin.valueChanged.connect(self._mouse_controls_changed)

        self.cursor_invert_x_checkbox = QCheckBox("Invert X")
        self.cursor_invert_x_checkbox.setChecked(self.config.mouse.invert_x)
        self.cursor_invert_x_checkbox.stateChanged.connect(self._mouse_controls_changed)

        self.cursor_invert_y_checkbox = QCheckBox("Invert Y")
        self.cursor_invert_y_checkbox.setChecked(self.config.mouse.invert_y)
        self.cursor_invert_y_checkbox.stateChanged.connect(self._mouse_controls_changed)

        for widget in (
            self.start_button,
            self.stop_button,
            self.capture_button,
            self.zero_button,
        ):
            layout.addWidget(widget)
        layout.addStretch(1)
        return bar

    def _build_controls_box(self) -> QGroupBox:
        box = QGroupBox("Controls")
        layout = QGridLayout(box)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        layout.addWidget(QLabel("Jaw label"), 0, 0)
        layout.addWidget(self.record_label, 0, 1)
        layout.addWidget(self.record_button, 1, 0, 1, 2)
        layout.addWidget(self.calibration_button, 2, 0)
        layout.addWidget(self.train_jaw_button, 2, 1)
        layout.addWidget(self.qa_jaw_button, 3, 0, 1, 2)

        layout.addWidget(self.mag_checkbox, 4, 0, 1, 2)
        layout.addWidget(QLabel("Gyro units"), 5, 0)
        layout.addWidget(self.gyro_units, 5, 1)
        layout.addWidget(QLabel("Beta"), 6, 0)
        layout.addWidget(self.beta_spin, 6, 1)
        layout.addWidget(QLabel("Pivot Z"), 7, 0)
        layout.addWidget(self.pivot_spin, 7, 1)
        layout.addWidget(self.swap_checkbox, 8, 0)
        layout.addWidget(self.invert_checkbox, 8, 1)

        layout.addWidget(self.arm_cursor_button, 9, 0, 1, 2)
        layout.addWidget(QLabel("Cursor DZ"), 10, 0)
        layout.addWidget(self.cursor_dead_zone_spin, 10, 1)
        layout.addWidget(QLabel("Cursor speed"), 11, 0)
        layout.addWidget(self.cursor_speed_spin, 11, 1)
        layout.addWidget(self.cursor_invert_x_checkbox, 12, 0)
        layout.addWidget(self.cursor_invert_y_checkbox, 12, 1)
        for widget in (
            self.record_button,
            self.calibration_button,
            self.train_jaw_button,
            self.qa_jaw_button,
            self.arm_cursor_button,
            self.record_label,
            self.gyro_units,
            self.beta_spin,
            self.pivot_spin,
            self.cursor_dead_zone_spin,
            self.cursor_speed_spin,
        ):
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return box

    def _build_calibration_prompt_box(self) -> QGroupBox:
        box = QGroupBox("Jaw Calibration Prompt")
        layout = QVBoxLayout(box)
        self.calibration_prompt_label = QLabel("Start a jaw calibration when the stream is running.")
        self.calibration_prompt_label.setWordWrap(True)
        self.calibration_prompt_label.setAlignment(Qt.AlignCenter)
        self.calibration_prompt_label.setMinimumHeight(86)
        self.calibration_prompt_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.calibration_prompt_label.setObjectName("calibrationPrompt")
        layout.addWidget(self.calibration_prompt_label)
        return box

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._build_status_box())
        layout.addWidget(self._build_calibration_prompt_box())
        layout.addWidget(self._build_plot_box("EXG Data", "exg"), 3)
        layout.addWidget(self._build_frequency_box(), 2)
        layout.addWidget(self._build_plot_box("Accel / Gyro / Mag", "imu"), 1)
        return panel

    def _build_status_box(self) -> QGroupBox:
        box = QGroupBox("Stream Status")
        layout = QGridLayout(box)
        self.state_label = QLabel("idle")
        self.port_label = QLabel(self.config.board.serial_port)
        self.rate_label = QLabel("0.0 Hz")
        self.packet_label = QLabel("0")
        self.loff_label = QLabel("P: --  N: --")
        self.exg_health_label = QLabel("EXG --")
        self.euler_label = QLabel("roll 0.0  pitch 0.0  yaw 0.0")
        self.raw_euler_label = QLabel("raw roll 0.0  pitch 0.0  yaw 0.0")
        self.cursor_label = QLabel("disarmed")
        self.jaw_preview_label = QLabel("jaw model not loaded")
        self.calibration_label = QLabel("calibration idle")
        self.backend_label = QLabel(self.estimator.backend)

        rows = [
            ("State", self.state_label),
            ("Port", self.port_label),
            ("Packets", self.packet_label),
            ("Rate", self.rate_label),
            ("LOFF", self.loff_label),
            ("EXG", self.exg_health_label),
            ("Corrected", self.euler_label),
            ("Raw", self.raw_euler_label),
            ("Cursor", self.cursor_label),
            ("Jaw", self.jaw_preview_label),
            ("Calibration", self.calibration_label),
            ("Orientation", self.backend_label),
        ]
        for index, (label, widget) in enumerate(rows):
            layout.addWidget(QLabel(label), index // 4, (index % 4) * 2)
            layout.addWidget(widget, index // 4, (index % 4) * 2 + 1)
        return box

    def _build_plot_box(self, title: str, kind: str) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        if kind == "exg":
            self.exg_plot = pg.PlotWidget()
            self.exg_plot.setLabel("bottom", "seconds")
            self.exg_plot.setLabel("left", "EEG 2")
            self.exg_plot.showGrid(x=True, y=False, alpha=0.16)
            self.exg_plot.setMouseEnabled(x=False, y=False)
            self.exg_plot.hideButtons()
            self.exg_curves = []
            rows = self.jaw_exg_rows
            self.exg_offsets = {row: float(len(rows) - idx) for idx, row in enumerate(rows)}
            ticks = [(offset, f"EEG {row}") for row, offset in self.exg_offsets.items()]
            self.exg_plot.getAxis("left").setTicks([ticks])
            self.exg_plot.setYRange(0.35, len(rows) + 0.65)
            for row in rows:
                curve = self.exg_plot.plot(pen=pg.mkPen("#d7dee3", width=1.2), name=f"EEG {row}")
                self.exg_curves.append((row, curve))
            layout.addWidget(self.exg_plot)
        else:
            self.accel_plot = self._make_imu_plot("Accel rows 11-13")
            self.gyro_plot = self._make_imu_plot("Gyro rows 14-16")
            self.mag_plot = self._make_imu_plot("Mag rows 17-19")
            layout.addWidget(self.accel_plot)
            layout.addWidget(self.gyro_plot)
            layout.addWidget(self.mag_plot)
            self.imu_curves = {
                "accel": self._add_xyz_curves(self.accel_plot, ACCEL_ROWS),
                "gyro": self._add_xyz_curves(self.gyro_plot, GYRO_ROWS),
                "mag": self._add_xyz_curves(self.mag_plot, MAG_ROWS),
            }
        return box

    def _build_frequency_box(self) -> QGroupBox:
        box = QGroupBox("Frequency Analysis of Active Channels")
        layout = QVBoxLayout(box)
        self.freq_plot = pg.PlotWidget()
        self.freq_plot.setLabel("bottom", "frequency", "Hz")
        self.freq_plot.setLabel("left", "amplitude")
        self.freq_plot.showGrid(x=True, y=True, alpha=0.18)
        self.freq_plot.setMouseEnabled(x=False, y=False)
        self.freq_plot.hideButtons()
        self.freq_plot.setXRange(0, 60)
        colors = [
            "#d7dee3",
            "#9ca7ae",
            "#7f8b93",
            "#b8c1c7",
            "#e1e6ea",
            "#8d979e",
            "#c6ced3",
            "#aab4ba",
        ]
        self.freq_curves = {
            row: self.freq_plot.plot(pen=pg.mkPen(colors[idx % len(colors)], width=1.2), name=f"EEG {row}")
            for idx, row in enumerate(self.jaw_exg_rows)
        }
        layout.addWidget(self.freq_plot)
        return box

    def _make_imu_plot(self, title: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(title=title)
        plot.setMaximumHeight(175)
        plot.setLabel("bottom", "seconds")
        plot.showGrid(x=True, y=True, alpha=0.25)
        return plot

    def _add_xyz_curves(self, plot: pg.PlotWidget, rows: tuple[int, int, int]) -> list[tuple[int, object]]:
        colors = ["#e15b64", "#7bc96f", "#4aa3df"]
        labels = ["x", "y", "z"]
        curves = []
        for idx, row in enumerate(rows):
            curves.append((row, plot.plot(pen=pg.mkPen(colors[idx], width=1.2), name=labels[idx])))
        return curves

    def _build_right_scroll_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(440)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        scroll.setWidget(self._build_right_panel())
        return scroll

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(420)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._build_gl_box(), 2)
        layout.addWidget(self._build_controls_box(), 1)
        layout.addWidget(self._build_channel_controls_box(), 4)
        layout.addWidget(self._build_table_box(), 3)
        layout.addWidget(self._build_log_box(), 2)
        return panel

    def _build_channel_controls_box(self) -> QGroupBox:
        box = QGroupBox("Jaw EEG Channel")
        layout = QGridLayout(box)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        channel = self.jaw_exg_rows[0]
        active = set(self.config.board.active_exg_channels)
        rld = set(self.config.board.rld_channels)
        active_check = QCheckBox("Active")
        active_check.setChecked(channel in active)
        active_check.setEnabled(False)
        active_check.stateChanged.connect(self._channel_controls_changed)
        rld_check = QCheckBox("Route RLD")
        rld_check.setChecked(channel in rld)
        rld_check.stateChanged.connect(self._channel_controls_changed)
        gain_box = QComboBox()
        gain_box.addItems(["1", "2", "3", "4", "6", "8", "12"])
        gain_box.setCurrentText(str(self.config.board.channel_gains.get(channel, self.config.board.gain)))
        gain_box.currentTextChanged.connect(self._channel_controls_changed)

        self.channel_active_checks[channel] = active_check
        self.channel_rld_checks[channel] = rld_check
        self.channel_gain_boxes[channel] = gain_box

        channel_label = QLabel(f"EEG channel {channel} feeds jaw clench detection.")
        channel_label.setWordWrap(True)
        layout.addWidget(channel_label, 0, 0, 1, 3)
        layout.addWidget(active_check, 1, 0)
        layout.addWidget(rld_check, 1, 1)
        layout.addWidget(QLabel("Gain"), 2, 0)
        layout.addWidget(gain_box, 2, 1)

        self.apply_channels_button = QPushButton("Apply On Restart")
        self.apply_channels_button.setEnabled(False)
        layout.addWidget(self.apply_channels_button, 3, 0, 1, 3)
        return box

    def _build_gl_box(self) -> QGroupBox:
        box = QGroupBox("Orientation")
        layout = QVBoxLayout(box)
        if gl is None:
            self.gl_view = None
            layout.addWidget(QLabel("OpenGL view unavailable"))
            return box

        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=2.8, elevation=20, azimuth=45)
        grid = gl.GLGridItem()
        grid.setSize(3.0, 3.0)
        grid.setSpacing(0.25, 0.25)
        self.gl_view.addItem(grid)
        points = self._wireframe_points(np.array([1.0, 0.0, 0.0, 0.0]))
        self._block_item = gl.GLLinePlotItem(
            pos=points,
            color=(1.0, 0.15, 0.12, 1.0),
            width=2.0,
            mode="lines",
            antialias=True,
        )
        self.gl_view.addItem(self._block_item)
        layout.addWidget(self.gl_view)
        return box

    def _build_table_box(self) -> QGroupBox:
        box = QGroupBox("Raw Rows")
        layout = QVBoxLayout(box)
        self.row_table = QTableWidget(NUM_ROWS, 5)
        self.row_table.setHorizontalHeaderLabels(["Row", "Name", "Latest", "Mean", "Spread"])
        self.row_table.verticalHeader().setVisible(False)
        self.row_table.setAlternatingRowColors(True)
        for row in range(NUM_ROWS):
            self.row_table.setItem(row, 0, QTableWidgetItem(str(row)))
            self.row_table.setItem(row, 1, QTableWidgetItem(ROW_LABELS[row]))
        self.row_table.resizeColumnsToContents()
        layout.addWidget(self.row_table)
        return box

    def _build_log_box(self) -> QGroupBox:
        box = QGroupBox("Stream Logs")
        layout = QVBoxLayout(box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        layout.addWidget(self.log_view)
        return box

    def _apply_style(self) -> None:
        pg.setConfigOptions(antialias=True, background="#111416", foreground="#d7dee3")
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111416; color: #d7dee3; }
            QGroupBox {
                border: 1px solid #2a3035;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton {
                background: #242b30;
                border: 1px solid #374149;
                border-radius: 5px;
                padding: 6px 10px;
            }
            QPushButton:hover { background: #2e373d; }
            QPushButton:disabled { color: #67717a; }
            QLabel#calibrationPrompt {
                background: #0d1012;
                border: 1px solid #374149;
                border-radius: 6px;
                color: #f4f7f8;
                font-size: 24px;
                font-weight: 700;
                padding: 16px;
            }
            QTableWidget {
                background: #151a1d;
                alternate-background-color: #101416;
                gridline-color: #2a3035;
            }
            QHeaderView::section {
                background: #20272c;
                color: #d7dee3;
                border: 1px solid #2a3035;
                padding: 4px;
            }
            QPlainTextEdit {
                background: #0d1012;
                border: 1px solid #2a3035;
                border-radius: 5px;
                font-family: Menlo, Consolas, monospace;
            }
            """
        )

    @Slot()
    def start_stream(self) -> None:
        if self.thread is not None:
            return
        self.buffer = np.zeros((NUM_ROWS, 0), dtype=float)
        self.total_packets = 0
        self.started_at = time.monotonic()
        self.state_label.setText("connecting")
        self._log("Starting stream")

        self.thread = QThread(self)
        self.worker = StreamWorker(self.config)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.data_ready.connect(self._handle_data)
        self.worker.connected.connect(self._handle_connected)
        self.worker.error.connect(self._handle_error)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._handle_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.calibration_button.setEnabled(True)
        self.capture_button.setEnabled(True)

    @Slot()
    def stop_stream(self) -> None:
        self._disarm_cursor("stream stopping")
        if self.worker is not None:
            self.worker.stop()
        if self.recorder.is_recording:
            if self.jaw_calibration is not None:
                self._finish_jaw_calibration(cancelled=True)
            else:
                self.toggle_recording()
        self.state_label.setText("stopping")

    @Slot(object)
    def _handle_connected(self, info: dict[str, object]) -> None:
        self.connection_info = info
        self.port_label.setText(str(info.get("port", "")))
        self.state_label.setText("streaming")
        self._log(f"Connected: {info}")

    @Slot(str)
    def _handle_error(self, message: str) -> None:
        self._disarm_cursor("stream error")
        self.state_label.setText("error")
        self._log(f"ERROR: {message}")

    @Slot()
    def _handle_finished(self) -> None:
        self._disarm_cursor("stream finished")
        self._log("Stream worker finished")
        self.state_label.setText("idle")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.calibration_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.thread = None
        self.worker = None

    @Slot(object)
    def _handle_data(self, batch: np.ndarray) -> None:
        self.total_packets += batch.shape[1]
        self.buffer = np.concatenate([self.buffer, batch], axis=1)
        if self.buffer.shape[1] > self.max_buffer_samples:
            self.buffer = self.buffer[:, -self.max_buffer_samples :]

        orientation_samples = []
        for idx in range(batch.shape[1]):
            orientation_samples.append(self.estimator.update(batch[:, idx]))
        if orientation_samples:
            self.latest_orientation = orientation_samples[-1]
        self.recorder.append(batch, orientation_samples)
        self._update_jaw_calibration()

    @Slot()
    def toggle_recording(self) -> None:
        if not self.recorder.is_recording:
            metadata = {
                "config": self.config.to_dict(),
                "config_path": self.config_path,
                "connection": self.connection_info,
                "row_map": ROW_LABELS,
                "gesture": "jaw",
                "label": self.record_label.currentText(),
                "active_exg_channels": list(self.config.board.active_exg_channels),
            }
            path = self.recorder.start(metadata)
            self.record_button.setText("Stop Jaw Recording")
            self.record_label.setEnabled(False)
            self.calibration_button.setEnabled(False)
            self._log(f"Jaw recording started: {path} label={self.record_label.currentText()}")
            return

        path = self.recorder.stop()
        self.record_button.setText("Start Jaw Recording")
        self.record_label.setEnabled(True)
        self.calibration_button.setEnabled(True)
        self._log(f"Jaw recording saved: {path}")

    @Slot()
    def start_jaw_calibration(self) -> None:
        if self.thread is None or self.state_label.text() != "streaming":
            self._log("Jaw calibration requires an active stream")
            return
        if self.recorder.is_recording:
            self._log("Stop the current recording before starting jaw calibration")
            return
        self.jaw_calibration = GuidedJawCalibration(self.config.jaw)
        self.jaw_calibration.start(0)
        metadata = {
            "config": self.config.to_dict(),
            "config_path": self.config_path,
            "connection": self.connection_info,
            "row_map": ROW_LABELS,
            "gesture": "jaw",
            "label": "jaw_guided_calibration",
            "active_exg_channels": list(self.config.jaw.channels),
            "prompt_schedule": [phase.__dict__ for phase in self.jaw_calibration.schedule],
        }
        path = self.recorder.start(metadata)
        self.record_button.setEnabled(False)
        self.record_label.setEnabled(False)
        self.calibration_button.setEnabled(False)
        self.calibration_label.setText(self.jaw_calibration.prompt_text(0))
        self._set_calibration_prompt(self.jaw_calibration.prompt_text(0), active=True)
        self._log(f"Jaw calibration started: {path}")

    @Slot()
    def train_jaw_model(self) -> None:
        session = self.last_calibration_session
        if session is None:
            self._log("No guided calibration session available to train")
            return
        self._train_and_load_jaw_model(session)

    @Slot()
    def capture_snapshot(self) -> None:
        if self.buffer.shape[1] == 0:
            self._log("Snapshot skipped: no stream samples yet")
            return
        metadata = {
            "config": self.config.to_dict(),
            "config_path": self.config_path,
            "connection": self.connection_info,
            "row_map": ROW_LABELS,
        }
        path = write_snapshot(
            self.buffer[:, -250:],
            self.config.board.active_exg_channels,
            metadata=metadata,
        )
        self._log(f"Snapshot saved: {path}")

    def _update_jaw_calibration(self) -> None:
        if self.jaw_calibration is None or not self.recorder.is_recording:
            return
        sample_count = self.recorder.sample_count
        labels = [label.to_dict() for label in self.jaw_calibration.update(sample_count)]
        if labels:
            self.recorder.add_labels(labels)
            for label in labels:
                if label.get("type") == "event":
                    self._log(f"Jaw label: {label['label']} sample={label.get('sample')}")
        prompt = self.jaw_calibration.prompt_text(sample_count)
        self.calibration_label.setText(prompt)
        self._set_calibration_prompt(prompt, active=True)
        if self.jaw_calibration.finished:
            self._finish_jaw_calibration(cancelled=False)

    def _finish_jaw_calibration(self, cancelled: bool) -> None:
        if not self.recorder.is_recording:
            self.jaw_calibration = None
            return
        path = self.recorder.stop()
        self.last_calibration_session = path
        self.jaw_calibration = None
        self.record_button.setEnabled(True)
        self.record_label.setEnabled(True)
        self.calibration_button.setEnabled(True)
        message = "calibration cancelled" if cancelled else "calibration complete"
        self.calibration_label.setText(message)
        self._set_calibration_prompt(message, active=False)
        self._log(f"Jaw calibration saved: {path}")
        if not cancelled and path is not None:
            self._train_and_load_jaw_model(path)

    def _train_and_load_jaw_model(self, session: Path) -> None:
        try:
            profile = train_profile([session], self.config)
            self.jaw_predictor = JawPredictor.load(profile, self.config.jaw)
            self.jaw_event_count = 0
            self.last_jaw_event_sample = -10_000_000
            self._log(f"Jaw clench model trained but not validated: {profile}")
        except Exception as exc:
            self._log(f"Jaw model training failed: {exc}")

    @Slot()
    def run_jaw_model_qa(self) -> None:
        sessions_root = Path("data/sessions")
        sessions, skipped = discover_usable_sessions(sessions_root, self.config)
        if skipped:
            self._log(f"Jaw QA skipped {len(skipped)} unusable session(s)")
        if len(sessions) < 2:
            self._log("Jaw QA requires at least two complete guided sessions")
            return
        train_sessions = sessions[:-1]
        validation_sessions = [sessions[-1]]
        try:
            report = train_and_evaluate_profile(train_sessions, validation_sessions, self.config)
            self.jaw_predictor = JawPredictor.load(Path(report["profile_dir"]), self.config.jaw)
            self.jaw_event_count = 0
            self.last_jaw_event_sample = -10_000_000
            metrics = report["metrics"]
            self._log(
                "Jaw QA "
                f"{report['validation_status']}: report={report['report_dir']} "
                f"threshold={report['selected_threshold']:.2f} "
                f"precision={metrics['precision']:.2f} recall={metrics['recall']:.2f} "
                f"fp/min={metrics['false_positives_per_minute']:.2f}"
            )
        except Exception as exc:
            self._log(f"Jaw QA failed: {exc}")

    def _load_jaw_predictor(self) -> None:
        profile = Path("models") / self.config.jaw.profile_name
        try:
            self.jaw_predictor = JawPredictor.load(profile, self.config.jaw)
            self.latest_jaw_prediction = JawPrediction(0.0, "relaxed")
        except Exception:
            self.jaw_predictor = None

    @Slot()
    def zero_orientation(self) -> None:
        self.estimator.zero_level()
        self._log("Orientation zeroed")

    @Slot()
    def _orientation_controls_changed(self) -> None:
        self.config.imu = ImuConfig(
            use_magnetometer=self.mag_checkbox.isChecked(),
            gyro_units=self.gyro_units.currentText(),
            beta=float(self.beta_spin.value()),
            pivot_z=float(self.pivot_spin.value()),
            swap_pitch_roll=self.swap_checkbox.isChecked(),
            invert_roll=self.invert_checkbox.isChecked(),
        )
        self.estimator.update_config(self.config.imu)
        self.backend_label.setText(self.estimator.backend)

    @Slot()
    def _channel_controls_changed(self) -> None:
        active = [
            channel
            for channel, checkbox in self.channel_active_checks.items()
            if checkbox.isChecked()
        ]
        rld = [
            channel
            for channel, checkbox in self.channel_rld_checks.items()
            if checkbox.isChecked()
        ]
        gains = {
            channel: int(box.currentText()) for channel, box in self.channel_gain_boxes.items()
        }
        self.config.board.active_exg_channels = active
        self.config.board.rld_channels = rld
        self.config.board.channel_gains = gains
        if not active:
            self.apply_channels_button.setText("Select at least one channel")
            self.apply_channels_button.setEnabled(False)
            return
        if self.thread is not None:
            self.apply_channels_button.setText("Restart stream to apply")
            self.apply_channels_button.setEnabled(True)
        else:
            self.apply_channels_button.setText("Ready for next stream")
            self.apply_channels_button.setEnabled(False)

    @Slot()
    def _mouse_controls_changed(self) -> None:
        self.config.mouse = MouseConfig(
            dead_zone_degrees=float(self.cursor_dead_zone_spin.value()),
            speed_px_per_second_per_degree=float(self.cursor_speed_spin.value()),
            max_speed_px_per_second=self.config.mouse.max_speed_px_per_second,
            smoothing=self.config.mouse.smoothing,
            invert_x=self.cursor_invert_x_checkbox.isChecked(),
            invert_y=self.cursor_invert_y_checkbox.isChecked(),
        )
        self.cursor_controller.update_config(self.config.mouse)

    @Slot()
    def toggle_cursor_arm(self) -> None:
        if self.cursor_armed:
            self._disarm_cursor("user disarmed")
            return
        self._arm_cursor()

    def _arm_cursor(self) -> None:
        if self.latest_orientation is None:
            self._log("Cursor arm blocked: start streaming and wait for orientation first")
            return
        if self.state_label.text() != "streaming":
            self._log("Cursor arm blocked: stream is not active")
            return
        self.cursor_controller.reset()
        self.cursor_armed = True
        self.last_cursor_step = time.monotonic()
        self.latest_cursor_velocity = MouseVelocity(0.0, 0.0)
        self.arm_cursor_button.setText("Disarm Cursor")
        self.cursor_label.setText("armed  vx 0  vy 0")
        self._log("Cursor armed. Zero (level) is the neutral pose; press Esc in this window to stop.")

    def _disarm_cursor(self, reason: str) -> None:
        if not self.cursor_armed:
            return
        self.cursor_armed = False
        self.last_cursor_step = None
        self.cursor_controller.reset()
        self.latest_cursor_velocity = MouseVelocity(0.0, 0.0)
        self.arm_cursor_button.setText("Arm Cursor")
        self.cursor_label.setText("disarmed")
        self._log(f"Cursor disarmed: {reason}")

    @Slot()
    def _escape_disarm_cursor(self) -> None:
        self._disarm_cursor("Esc")

    @Slot()
    def _drive_cursor(self) -> None:
        if not self.cursor_armed:
            return
        if self.latest_orientation is None:
            self._disarm_cursor("no orientation samples")
            return
        now = time.monotonic()
        if self.last_cursor_step is None:
            self.last_cursor_step = now
            return
        dt = max(0.005, min(0.050, now - self.last_cursor_step))
        self.last_cursor_step = now
        self.latest_cursor_velocity = self.cursor_controller.step(
            self.latest_orientation.roll,
            self.latest_orientation.pitch,
            dt,
        )

    @Slot()
    def _refresh_ui(self) -> None:
        if self.started_at is not None:
            elapsed = max(time.monotonic() - self.started_at, 0.001)
            self.rate_label.setText(f"{self.total_packets / elapsed:.1f} Hz")
        self.packet_label.setText(str(self.total_packets))
        if self.cursor_armed:
            self.cursor_label.setText(
                f"armed  vx {self.latest_cursor_velocity.vx:5.0f}  vy {self.latest_cursor_velocity.vy:5.0f}"
            )
        else:
            self.cursor_label.setText("disarmed")
        self._refresh_jaw_preview()

        if self.buffer.shape[1] == 0:
            return

        latest = self.buffer[:, -1]
        p_bits = loff_bits(latest[LOFF_STATP_ROW])
        n_bits = loff_bits(latest[LOFF_STATN_ROW])
        self.loff_label.setText(f"P: {p_bits or 'clear'}  N: {n_bits or 'clear'}")
        self.exg_health_label.setText(
            active_exg_summary_text(self.buffer, self.config.board.active_exg_channels)
        )

        if self.latest_orientation is not None:
            sample = self.latest_orientation
            self.euler_label.setText(
                f"roll {sample.roll:7.2f}  pitch {sample.pitch:7.2f}  yaw {sample.yaw:7.2f}"
            )
            self.raw_euler_label.setText(
                f"raw roll {sample.raw_roll:7.2f}  pitch {sample.raw_pitch:7.2f}  yaw {sample.raw_yaw:7.2f}"
            )
            self._update_wireframe(np.array(sample.quaternion, dtype=float))

        self._refresh_table()
        self._refresh_plots()

    def _refresh_jaw_preview(self) -> None:
        if self.jaw_predictor is None:
            self.jaw_preview_label.setText("model not loaded")
            return
        needed = max(2, int(round(self.config.jaw.short_clench_seconds * self.config.jaw.sampling_rate)))
        if self.buffer.shape[1] < needed:
            self.jaw_preview_label.setText("waiting for jaw samples")
            return
        self.latest_jaw_prediction = self.jaw_predictor.predict(self.buffer)
        current_sample = self.total_packets
        refractory = int(round(self.config.jaw.sampling_rate))
        if (
            self.latest_jaw_prediction.event_confidence >= self.jaw_predictor.threshold
            and current_sample - self.last_jaw_event_sample >= refractory
        ):
            self.jaw_event_count += 1
            self.last_jaw_event_sample = current_sample
        status = "validated" if self.jaw_predictor.validated else "unvalidated"
        self.jaw_preview_label.setText(
            f"{self.latest_jaw_prediction.state}  "
            f"clench {self.latest_jaw_prediction.event_confidence:.2f}  "
            f"thr {self.jaw_predictor.threshold:.2f}  "
            f"count {self.jaw_event_count}  {status}"
        )

    def _refresh_table(self) -> None:
        for stat in row_stats(self.buffer[:, -250:]):
            for col, value in enumerate(
                [
                    f"{stat.latest:.4f}",
                    f"{stat.mean:.4f}",
                    f"{stat.spread:.4f}",
                ],
                start=2,
            ):
                item = self.row_table.item(stat.row, col)
                if item is None:
                    item = QTableWidgetItem()
                    self.row_table.setItem(stat.row, col, item)
                item.setText(value)
                if stat.row in (*ACCEL_ROWS, *GYRO_ROWS, *MAG_ROWS):
                    item.setForeground(QColor("#d7dee3"))
        self.row_table.resizeColumnsToContents()

    def _refresh_plots(self) -> None:
        data = self.buffer[:, -self.max_buffer_samples :]
        n = data.shape[1]
        x = (np.arange(n) - n + 1) / SAMPLING_RATE_HZ
        active = set(self.config.board.active_exg_channels)
        for row, curve in self.exg_curves:
            curve.setPen(pg.mkPen("#d7dee3" if row in active else "#4f5559", width=0.9))
            curve.setData(x, self._stacked_exg_trace(data[row], self.exg_offsets[row]))
        for group in self.imu_curves.values():
            for row, curve in group:
                curve.setData(x, data[row])
        self._refresh_frequency_plot(data)

    def _stacked_exg_trace(self, values: np.ndarray, offset: float) -> np.ndarray:
        if values.size == 0:
            return values
        centered = values - np.median(values)
        scale = float(np.percentile(np.abs(centered), 95)) if centered.size else 1.0
        scale = max(scale, 1.0)
        return centered / scale * 0.36 + offset

    def _refresh_frequency_plot(self, data: np.ndarray) -> None:
        n = data.shape[1]
        if n < 16:
            for curve in self.freq_curves.values():
                curve.setData([], [])
            return
        active = set(self.config.board.active_exg_channels)
        window_size = min(n, SAMPLING_RATE_HZ * 4)
        segment = data[:, -window_size:]
        freq = np.fft.rfftfreq(window_size, d=1.0 / SAMPLING_RATE_HZ)
        keep = freq <= 60.0
        taper = np.hanning(window_size)
        for row, curve in self.freq_curves.items():
            if row not in active:
                curve.setData([], [])
                continue
            values = segment[row] - np.mean(segment[row])
            spectrum = np.abs(np.fft.rfft(values * taper)) / max(window_size, 1)
            curve.setData(freq[keep], spectrum[keep] / 1_000_000.0)

    def _wireframe_points(self, q: np.ndarray) -> np.ndarray:
        vertices = np.array(
            [
                [-0.35, -0.20, -0.12],
                [0.35, -0.20, -0.12],
                [0.35, 0.20, -0.12],
                [-0.35, 0.20, -0.12],
                [-0.35, -0.20, 0.12],
                [0.35, -0.20, 0.12],
                [0.35, 0.20, 0.12],
                [-0.35, 0.20, 0.12],
            ],
            dtype=float,
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        rot = self._rotation_matrix(q)
        translated = vertices.copy()
        translated[:, 2] += self.config.imu.pivot_z
        rotated = translated @ rot.T
        points = []
        for a, b in edges:
            points.append(rotated[a])
            points.append(rotated[b])
        return np.array(points, dtype=float)

    def _rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def _update_wireframe(self, q: np.ndarray) -> None:
        if self._block_item is not None:
            self._block_item.setData(pos=self._wireframe_points(q))

    @Slot(str)
    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"{timestamp} {message}")

    def _set_calibration_prompt(self, message: str, active: bool) -> None:
        if not hasattr(self, "calibration_prompt_label"):
            return
        self.calibration_prompt_label.setText(message)
        border = "#f0b84f" if active else "#374149"
        background = "#1b1710" if active else "#0d1012"
        self.calibration_prompt_label.setStyleSheet(
            "QLabel#calibrationPrompt {"
            f"background: {background};"
            f"border: 1px solid {border};"
            "border-radius: 6px;"
            "color: #f4f7f8;"
            "font-size: 24px;"
            "font-weight: 700;"
            "padding: 16px;"
            "}"
        )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._disarm_cursor("app closing")
        self.stop_stream()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(2000)
        event.accept()


def run_gui(config_path: str) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = DiagnosticsWindow(config_path)
    window.show()
    return int(app.exec())
