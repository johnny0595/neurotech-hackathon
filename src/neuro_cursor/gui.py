"""PySide6 diagnostics GUI for the NeuroPawn Knight IMU board."""

from __future__ import annotations

import sys
import time

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
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .brainflow_adapter import BoardConnectionError, KnightBrainFlowAdapter
from .config import AppConfig, ImuConfig, MouseConfig, load_config
from .mouse_control import MouseVelocity, QtCursorController
from .orientation import OrientationEstimator
from .recording import SessionRecorder
from .rows import (
    ACCEL_ROWS,
    EXG_ROWS,
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

        self.setWindowTitle("NeuroPawn Knight IMU Diagnostics")
        self.resize(1480, 980)
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

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter, 1)

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
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
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
            self.record_button,
            self.zero_button,
            self.mag_checkbox,
            QLabel("Gyro units"),
            self.gyro_units,
            QLabel("Beta"),
            self.beta_spin,
            QLabel("Pivot Z"),
            self.pivot_spin,
            self.swap_checkbox,
            self.invert_checkbox,
            self.arm_cursor_button,
            QLabel("Cursor DZ"),
            self.cursor_dead_zone_spin,
            QLabel("Cursor speed"),
            self.cursor_speed_spin,
            self.cursor_invert_x_checkbox,
            self.cursor_invert_y_checkbox,
        ):
            layout.addWidget(widget)
        layout.addStretch(1)
        return bar

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._build_status_box())
        layout.addWidget(self._build_plot_box("EXG channels 1-4", "exg"))
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
        self.euler_label = QLabel("roll 0.0  pitch 0.0  yaw 0.0")
        self.raw_euler_label = QLabel("raw roll 0.0  pitch 0.0  yaw 0.0")
        self.cursor_label = QLabel("disarmed")
        self.backend_label = QLabel(self.estimator.backend)

        rows = [
            ("State", self.state_label),
            ("Port", self.port_label),
            ("Packets", self.packet_label),
            ("Rate", self.rate_label),
            ("LOFF", self.loff_label),
            ("Corrected", self.euler_label),
            ("Raw", self.raw_euler_label),
            ("Cursor", self.cursor_label),
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
            self.exg_plot.setLabel("left", "uV")
            self.exg_plot.addLegend(offset=(8, 8))
            colors = ["#e15b64", "#4aa3df", "#7bc96f", "#d7b84f"]
            self.exg_curves = []
            for idx, row in enumerate(EXG_ROWS[:4]):
                curve = self.exg_plot.plot(pen=pg.mkPen(colors[idx], width=1.4), name=f"EXG {row}")
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

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._build_gl_box(), 2)
        layout.addWidget(self._build_table_box(), 3)
        layout.addWidget(self._build_log_box(), 2)
        return panel

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

    @Slot()
    def stop_stream(self) -> None:
        self._disarm_cursor("stream stopping")
        if self.worker is not None:
            self.worker.stop()
        if self.recorder.is_recording:
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

    @Slot()
    def toggle_recording(self) -> None:
        if not self.recorder.is_recording:
            metadata = {
                "config": self.config.to_dict(),
                "config_path": self.config_path,
                "connection": self.connection_info,
                "row_map": ROW_LABELS,
            }
            path = self.recorder.start(metadata)
            self.record_button.setText("Stop Recording")
            self._log(f"Recording started: {path}")
            return

        path = self.recorder.stop()
        self.record_button.setText("Start Recording")
        self._log(f"Recording saved: {path}")

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

        if self.buffer.shape[1] == 0:
            return

        latest = self.buffer[:, -1]
        p_bits = loff_bits(latest[LOFF_STATP_ROW])
        n_bits = loff_bits(latest[LOFF_STATN_ROW])
        self.loff_label.setText(f"P: {p_bits or 'clear'}  N: {n_bits or 'clear'}")

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
        for row, curve in self.exg_curves:
            curve.setData(x, data[row])
        for group in self.imu_curves.values():
            for row, curve in group:
                curve.setData(x, data[row])

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
