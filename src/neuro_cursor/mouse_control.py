"""Explicitly armed cursor velocity control from corrected head tilt."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QCursor, QGuiApplication

from .config import MouseConfig


@dataclass(frozen=True)
class MouseVelocity:
    vx: float
    vy: float


def _dead_zone(value: float, threshold: float) -> float:
    magnitude = abs(value) - threshold
    if magnitude <= 0:
        return 0.0
    return magnitude if value > 0 else -magnitude


def tilt_to_velocity(roll: float, pitch: float, config: MouseConfig) -> MouseVelocity:
    x = _dead_zone(roll, config.dead_zone_degrees)
    y = _dead_zone(pitch, config.dead_zone_degrees)
    vx = x * config.speed_px_per_second_per_degree
    vy = y * config.speed_px_per_second_per_degree
    if config.invert_x:
        vx = -vx
    if config.invert_y:
        vy = -vy

    max_speed = config.max_speed_px_per_second
    vx = max(-max_speed, min(max_speed, vx))
    vy = max(-max_speed, min(max_speed, vy))
    return MouseVelocity(vx=vx, vy=vy)


class QtCursorController:
    """Moves the global cursor. It never arms itself."""

    def __init__(self, config: MouseConfig) -> None:
        self.config = config
        self._vx = 0.0
        self._vy = 0.0
        self._rem_x = 0.0
        self._rem_y = 0.0

    def update_config(self, config: MouseConfig) -> None:
        self.config = config

    def reset(self) -> None:
        self._vx = 0.0
        self._vy = 0.0
        self._rem_x = 0.0
        self._rem_y = 0.0

    def step(self, roll: float, pitch: float, dt: float) -> MouseVelocity:
        target = tilt_to_velocity(roll, pitch, self.config)
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
