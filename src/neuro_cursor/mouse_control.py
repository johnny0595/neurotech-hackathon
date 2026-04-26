"""Explicitly armed cursor velocity control from corrected head tilt."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QCursor, QGuiApplication

from .config import MouseConfig


@dataclass(frozen=True)
class MouseVelocity:
    vx: float
    vy: float


@dataclass(frozen=True)
class CursorAxisCalibration:
    center_roll_degrees: float
    center_pitch_degrees: float
    x_axis_roll: float
    x_axis_pitch: float
    y_axis_roll: float
    y_axis_pitch: float


def _unit_axis(roll: float, pitch: float) -> tuple[float, float]:
    length = hypot(roll, pitch)
    if length <= 1e-6:
        raise ValueError("cursor calibration tilt is too small")
    return roll / length, pitch / length


def calibrated_axes_from_samples(
    neutral: tuple[float, float],
    left_tilt: tuple[float, float],
    right_tilt: tuple[float, float],
    up_tilt: tuple[float, float],
    down_tilt: tuple[float, float],
) -> CursorAxisCalibration:
    """Build a screen-axis basis from neutral plus four cardinal head poses."""

    center_roll, center_pitch = neutral
    x_axis_roll, x_axis_pitch = _unit_axis(
        right_tilt[0] - left_tilt[0],
        right_tilt[1] - left_tilt[1],
    )
    y_axis_roll, y_axis_pitch = _unit_axis(
        down_tilt[0] - up_tilt[0],
        down_tilt[1] - up_tilt[1],
    )
    determinant = x_axis_roll * y_axis_pitch - x_axis_pitch * y_axis_roll
    if abs(determinant) <= 0.15:
        raise ValueError("cursor calibration tilts are too similar")
    return CursorAxisCalibration(
        center_roll_degrees=center_roll,
        center_pitch_degrees=center_pitch,
        x_axis_roll=x_axis_roll,
        x_axis_pitch=x_axis_pitch,
        y_axis_roll=y_axis_roll,
        y_axis_pitch=y_axis_pitch,
    )


def with_cursor_axis_calibration(
    config: MouseConfig,
    calibration: CursorAxisCalibration,
) -> MouseConfig:
    return MouseConfig(
        dead_zone_degrees=config.dead_zone_degrees,
        speed_px_per_second_per_degree=config.speed_px_per_second_per_degree,
        max_speed_px_per_second=config.max_speed_px_per_second,
        smoothing=config.smoothing,
        response_curve=config.response_curve,
        full_tilt_degrees=config.full_tilt_degrees,
        motion_boost_px_per_second_per_degree=config.motion_boost_px_per_second_per_degree,
        invert_x=config.invert_x,
        invert_y=config.invert_y,
        center_roll_degrees=calibration.center_roll_degrees,
        center_pitch_degrees=calibration.center_pitch_degrees,
        x_axis_roll=calibration.x_axis_roll,
        x_axis_pitch=calibration.x_axis_pitch,
        y_axis_roll=calibration.y_axis_roll,
        y_axis_pitch=calibration.y_axis_pitch,
    )


def calibrated_tilt(roll: float, pitch: float, config: MouseConfig) -> tuple[float, float]:
    roll_delta = roll - config.center_roll_degrees
    pitch_delta = pitch - config.center_pitch_degrees
    determinant = (
        config.x_axis_roll * config.y_axis_pitch
        - config.x_axis_pitch * config.y_axis_roll
    )
    if abs(determinant) <= 1e-6:
        return roll_delta, pitch_delta
    x = (
        config.y_axis_pitch * roll_delta
        - config.y_axis_roll * pitch_delta
    ) / determinant
    y = (
        -config.x_axis_pitch * roll_delta
        + config.x_axis_roll * pitch_delta
    ) / determinant
    return x, y


def _apply_tilt_response(x: float, y: float, config: MouseConfig) -> tuple[float, float]:
    magnitude = hypot(x, y)
    dead_zone = config.dead_zone_degrees
    if magnitude <= dead_zone:
        return 0.0, 0.0

    active = magnitude - dead_zone
    reference = max(config.full_tilt_degrees - dead_zone, 1e-6)
    if active <= reference:
        adjusted = reference * (active / reference) ** config.response_curve
    else:
        adjusted = active
    scale = adjusted / magnitude
    return x * scale, y * scale


def _clamp_velocity(vx: float, vy: float, config: MouseConfig) -> MouseVelocity:
    max_speed = config.max_speed_px_per_second
    return MouseVelocity(
        vx=max(-max_speed, min(max_speed, vx)),
        vy=max(-max_speed, min(max_speed, vy)),
    )


def tilt_to_velocity(roll: float, pitch: float, config: MouseConfig) -> MouseVelocity:
    calibrated_x, calibrated_y = calibrated_tilt(roll, pitch, config)
    x, y = _apply_tilt_response(calibrated_x, calibrated_y, config)
    vx = x * config.speed_px_per_second_per_degree
    vy = y * config.speed_px_per_second_per_degree
    if config.invert_x:
        vx = -vx
    if config.invert_y:
        vy = -vy

    return _clamp_velocity(vx, vy, config)


class QtCursorController:
    """Moves the global cursor. It never arms itself."""

    def __init__(self, config: MouseConfig) -> None:
        self.config = config
        self._vx = 0.0
        self._vy = 0.0
        self._rem_x = 0.0
        self._rem_y = 0.0
        self._last_tilt: tuple[float, float] | None = None

    def update_config(self, config: MouseConfig) -> None:
        self.config = config

    def reset(self) -> None:
        self._vx = 0.0
        self._vy = 0.0
        self._rem_x = 0.0
        self._rem_y = 0.0
        self._last_tilt = None

    def step(self, roll: float, pitch: float, dt: float) -> MouseVelocity:
        target = tilt_to_velocity(roll, pitch, self.config)
        current_tilt = calibrated_tilt(roll, pitch, self.config)
        if self._last_tilt is not None:
            target = self._add_motion_boost(target, current_tilt, dt)
        self._last_tilt = current_tilt

        smoothing = self.config.smoothing
        self._vx = smoothing * self._vx + (1.0 - smoothing) * target.vx
        self._vy = smoothing * self._vy + (1.0 - smoothing) * target.vy

        dx = self._vx * dt + self._rem_x
        dy = self._vy * dt + self._rem_y
        move_x = int(round(dx))
        move_y = int(round(dy))
        self._rem_x = dx - move_x
        self._rem_y = dy - move_y

        if move_x or move_y:
            self._move_by(move_x, move_y)
        return MouseVelocity(vx=self._vx, vy=self._vy)

    def _add_motion_boost(
        self,
        target: MouseVelocity,
        current_tilt: tuple[float, float],
        dt: float,
    ) -> MouseVelocity:
        boost_scale = self.config.motion_boost_px_per_second_per_degree
        if boost_scale <= 0.0 or dt <= 0.0:
            return target
        if hypot(*current_tilt) <= self.config.dead_zone_degrees:
            return target

        previous_x, previous_y = self._last_tilt or current_tilt
        boost_x = ((current_tilt[0] - previous_x) / dt) * boost_scale
        boost_y = ((current_tilt[1] - previous_y) / dt) * boost_scale
        if self.config.invert_x:
            boost_x = -boost_x
        if self.config.invert_y:
            boost_y = -boost_y

        max_boost = self.config.max_speed_px_per_second * 0.35
        boost_x = max(-max_boost, min(max_boost, boost_x))
        boost_y = max(-max_boost, min(max_boost, boost_y))
        return _clamp_velocity(target.vx + boost_x, target.vy + boost_y, self.config)

    def _move_by(self, dx: int, dy: int) -> None:
        position = QCursor.pos()
        bounds = _virtual_screen_geometry()
        x = max(bounds.left(), min(bounds.right(), position.x() + dx))
        y = max(bounds.top(), min(bounds.bottom(), position.y() + dy))
        if x in (bounds.left(), bounds.right()):
            self._rem_x = 0.0
        if y in (bounds.top(), bounds.bottom()):
            self._rem_y = 0.0
        QCursor.setPos(QPoint(x, y))


def _virtual_screen_geometry() -> QRect:
    screens = QGuiApplication.screens()
    if not screens:
        return QRect(0, 0, 1, 1)
    geometry = screens[0].geometry()
    for screen in screens[1:]:
        geometry = geometry.united(screen.geometry())
    return geometry
