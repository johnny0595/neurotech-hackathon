"""User-editable YAML config for the diagnostics app."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from .rows import BOARD_ID_NAME

DEFAULT_CONFIG_PATH = Path("config/neuro_cursor.yaml")
ALLOWED_GAINS = {1, 2, 3, 4, 6, 8, 12}
ALLOWED_GYRO_UNITS = {"auto", "rad_s", "deg_s"}


@dataclass
class BoardConfig:
    id: str = BOARD_ID_NAME
    serial_port: str = "/dev/cu.usbserial-A5069RR4"
    active_exg_channels: list[int] = field(default_factory=lambda: [2])
    rld_channels: list[int] = field(default_factory=lambda: [2])
    channel_gains: dict[int, int] = field(
        default_factory=lambda: {channel: 12 for channel in range(1, 9)}
    )
    gain: int = 12
    buffer_size: int = 450000
    configure_channels: bool = True
    stream_settle_seconds: float = 2.0
    config_command_delay_seconds: float = 0.5


@dataclass
class ImuConfig:
    use_magnetometer: bool = False
    gyro_units: str = "auto"
    beta: float = 0.2
    pivot_z: float = 0.25
    swap_pitch_roll: bool = False
    invert_roll: bool = False


@dataclass
class MouseConfig:
    dead_zone_degrees: float = 1.25
    speed_px_per_second_per_degree: float = 70.0
    max_speed_px_per_second: float = 1400.0
    smoothing: float = 0.35
    response_curve: float = 1.35
    full_tilt_degrees: float = 18.0
    motion_boost_px_per_second_per_degree: float = 8.0
    invert_x: bool = False
    invert_y: bool = False
    center_roll_degrees: float = 0.0
    center_pitch_degrees: float = 0.0
    x_axis_roll: float = 1.0
    x_axis_pitch: float = 0.0
    y_axis_roll: float = 0.0
    y_axis_pitch: float = 1.0


@dataclass
class JawConfig:
    channels: list[int] = field(default_factory=lambda: [2])
    sampling_rate: float = 125.0
    short_clench_seconds: float = 0.5
    relax_seconds: float = 2.0
    hold_seconds: float = 3.0
    pre_event_seconds: float = 0.5
    post_event_seconds: float = 0.8
    short_clench_reps: int = 20
    hold_reps: int = 0
    initial_neutral_seconds: float = 8.0
    final_neutral_seconds: float = 6.0
    event_min_seconds: float = 0.35
    event_max_seconds: float = 0.50
    relax_min_seconds: float = 1.2
    relax_max_seconds: float = 2.5
    hard_negative_reps: int = 5
    hard_negative_seconds: float = 0.75
    window_seconds: list[float] = field(default_factory=lambda: [0.25, 0.35, 0.50])
    window_step_seconds: float = 0.05
    positive_label: str = "eyebrow_raise"
    protocol_version: str = "short_event_v2"
    prompt_seed: int = 7
    profile_name: str = "eyebrow"
    event_threshold: float = 0.70
    hold_threshold: float = 0.70


@dataclass
class AppConfig:
    board: BoardConfig = field(default_factory=BoardConfig)
    channel_map: dict[int, str] = field(
        default_factory=lambda: {
            1: "unused",
            2: "eyebrow",
            3: "unused",
            4: "unused",
        }
    )
    imu: ImuConfig = field(default_factory=ImuConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    jaw: JawConfig = field(default_factory=JawConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        config = AppConfig()
        validate_config(config)
        return config

    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    board_raw = dict(raw.get("board") or {})
    if "channel_gains" in board_raw:
        board_raw["channel_gains"] = {
            int(channel): int(gain) for channel, gain in (board_raw["channel_gains"] or {}).items()
        }
    board = BoardConfig(**board_raw)
    channel_map_raw = raw.get("channel_map") or {}
    channel_map = {int(key): str(value) for key, value in channel_map_raw.items()}
    imu = ImuConfig(**(raw.get("imu") or {}))
    mouse = MouseConfig(**(raw.get("mouse") or {}))
    jaw = JawConfig(**(raw.get("jaw") or {}))
    config = AppConfig(
        board=board,
        channel_map=channel_map or AppConfig().channel_map,
        imu=imu,
        mouse=mouse,
        jaw=jaw,
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    if config.board.id != BOARD_ID_NAME:
        raise ValueError(f"Phase 1 only supports {BOARD_ID_NAME}, got {config.board.id}")
    if config.board.gain not in ALLOWED_GAINS:
        raise ValueError(f"gain must be one of {sorted(ALLOWED_GAINS)}, got {config.board.gain}")
    for channel, gain in config.board.channel_gains.items():
        if int(channel) < 1 or int(channel) > 8:
            raise ValueError(f"channel_gains keys must be in 1..8, got {channel}")
        if int(gain) not in ALLOWED_GAINS:
            raise ValueError(
                f"channel {channel} gain must be one of {sorted(ALLOWED_GAINS)}, got {gain}"
            )
    if config.imu.gyro_units not in ALLOWED_GYRO_UNITS:
        raise ValueError(
            f"gyro_units must be one of {sorted(ALLOWED_GYRO_UNITS)}, got {config.imu.gyro_units}"
        )
    if config.board.buffer_size <= 0:
        raise ValueError("buffer_size must be positive")
    if config.board.stream_settle_seconds < 0:
        raise ValueError("stream_settle_seconds must be non-negative")
    if config.board.config_command_delay_seconds < 0:
        raise ValueError("config_command_delay_seconds must be non-negative")
    if config.imu.beta <= 0:
        raise ValueError("Madgwick beta must be positive")
    if config.mouse.dead_zone_degrees < 0:
        raise ValueError("mouse.dead_zone_degrees must be non-negative")
    if config.mouse.speed_px_per_second_per_degree <= 0:
        raise ValueError("mouse.speed_px_per_second_per_degree must be positive")
    if config.mouse.max_speed_px_per_second <= 0:
        raise ValueError("mouse.max_speed_px_per_second must be positive")
    if not 0.0 <= config.mouse.smoothing <= 0.95:
        raise ValueError("mouse.smoothing must be between 0.0 and 0.95")
    if config.mouse.response_curve < 1.0:
        raise ValueError("mouse.response_curve must be at least 1.0")
    if config.mouse.full_tilt_degrees <= config.mouse.dead_zone_degrees:
        raise ValueError("mouse.full_tilt_degrees must be larger than dead_zone_degrees")
    if config.mouse.motion_boost_px_per_second_per_degree < 0:
        raise ValueError("mouse.motion_boost_px_per_second_per_degree must be non-negative")
    mouse_axis_values = (
        config.mouse.center_roll_degrees,
        config.mouse.center_pitch_degrees,
        config.mouse.x_axis_roll,
        config.mouse.x_axis_pitch,
        config.mouse.y_axis_roll,
        config.mouse.y_axis_pitch,
    )
    if any(not isfinite(value) for value in mouse_axis_values):
        raise ValueError("mouse cursor-axis calibration values must be finite")
    determinant = (
        config.mouse.x_axis_roll * config.mouse.y_axis_pitch
        - config.mouse.x_axis_pitch * config.mouse.y_axis_roll
    )
    if abs(determinant) <= 1e-6:
        raise ValueError("mouse cursor-axis calibration must not be degenerate")
    if not config.jaw.channels:
        raise ValueError("jaw.channels must contain at least one EXG channel")
    invalid_jaw = [channel for channel in config.jaw.channels if channel < 1 or channel > 8]
    if invalid_jaw:
        raise ValueError(f"jaw.channels must be in 1..8, got {invalid_jaw}")
    for name in (
        "sampling_rate",
        "short_clench_seconds",
        "relax_seconds",
        "hold_seconds",
        "pre_event_seconds",
        "post_event_seconds",
    ):
        if getattr(config.jaw, name) <= 0:
            raise ValueError(f"jaw.{name} must be positive")
    if config.jaw.short_clench_reps <= 0:
        raise ValueError("jaw.short_clench_reps must be positive")
    if config.jaw.hold_reps < 0:
        raise ValueError("jaw.hold_reps must be non-negative")
    if config.jaw.hard_negative_reps < 0:
        raise ValueError("jaw.hard_negative_reps must be non-negative")
    for name in (
        "initial_neutral_seconds",
        "final_neutral_seconds",
        "event_min_seconds",
        "event_max_seconds",
        "relax_min_seconds",
        "relax_max_seconds",
        "hard_negative_seconds",
        "window_step_seconds",
    ):
        if getattr(config.jaw, name) <= 0:
            raise ValueError(f"jaw.{name} must be positive")
    if config.jaw.event_max_seconds < config.jaw.event_min_seconds:
        raise ValueError("jaw.event_max_seconds must be >= jaw.event_min_seconds")
    if config.jaw.relax_max_seconds < config.jaw.relax_min_seconds:
        raise ValueError("jaw.relax_max_seconds must be >= jaw.relax_min_seconds")
    if not config.jaw.window_seconds:
        raise ValueError("jaw.window_seconds must contain at least one value")
    if any(value <= 0 for value in config.jaw.window_seconds):
        raise ValueError("jaw.window_seconds values must be positive")
    if not config.jaw.positive_label:
        raise ValueError("jaw.positive_label must be non-empty")
    if not 0.0 < config.jaw.event_threshold < 1.0:
        raise ValueError("jaw.event_threshold must be between 0 and 1")
    if not 0.0 < config.jaw.hold_threshold < 1.0:
        raise ValueError("jaw.hold_threshold must be between 0 and 1")
    if not config.board.active_exg_channels:
        raise ValueError("at least one active EXG channel is required")
    invalid = [channel for channel in config.board.active_exg_channels if channel < 1 or channel > 8]
    if invalid:
        raise ValueError(f"active EXG channels must be in 1..8, got {invalid}")
    invalid_rld = [channel for channel in config.board.rld_channels if channel < 1 or channel > 8]
    if invalid_rld:
        raise ValueError(f"rld_channels must be in 1..8, got {invalid_rld}")
