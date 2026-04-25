"""User-editable YAML config for the diagnostics app."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    active_exg_channels: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
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
    dead_zone_degrees: float = 3.0
    speed_px_per_second_per_degree: float = 70.0
    max_speed_px_per_second: float = 1400.0
    smoothing: float = 0.35
    invert_x: bool = False
    invert_y: bool = False


@dataclass
class AppConfig:
    board: BoardConfig = field(default_factory=BoardConfig)
    channel_map: dict[int, str] = field(
        default_factory=lambda: {
            1: "frontal_left",
            2: "frontal_right",
            3: "jaw_left",
            4: "jaw_right",
        }
    )
    imu: ImuConfig = field(default_factory=ImuConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)

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

    board = BoardConfig(**(raw.get("board") or {}))
    channel_map_raw = raw.get("channel_map") or {}
    channel_map = {int(key): str(value) for key, value in channel_map_raw.items()}
    imu = ImuConfig(**(raw.get("imu") or {}))
    mouse = MouseConfig(**(raw.get("mouse") or {}))
    config = AppConfig(
        board=board,
        channel_map=channel_map or AppConfig().channel_map,
        imu=imu,
        mouse=mouse,
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    if config.board.id != BOARD_ID_NAME:
        raise ValueError(f"Phase 1 only supports {BOARD_ID_NAME}, got {config.board.id}")
    if config.board.gain not in ALLOWED_GAINS:
        raise ValueError(f"gain must be one of {sorted(ALLOWED_GAINS)}, got {config.board.gain}")
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
    if not config.board.active_exg_channels:
        raise ValueError("at least one active EXG channel is required")
    invalid = [channel for channel in config.board.active_exg_channels if channel < 1 or channel > 8]
    if invalid:
        raise ValueError(f"active EXG channels must be in 1..8, got {invalid}")
