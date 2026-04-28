import pytest

from neuro_cursor.config import MouseConfig
from neuro_cursor.mouse_control import (
    QtCursorController,
    calibrated_axes_from_samples,
    post_left_click,
    post_mouse_move,
    tilt_to_velocity,
    with_cursor_axis_calibration,
)


class FakeCursorPoint:
    def __init__(self, x: int, y: int) -> None:
        self._x = x
        self._y = y

    def x(self) -> int:
        return self._x

    def y(self) -> int:
        return self._y


class FakeScreenBounds:
    def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
        self._left = left
        self._top = top
        self._right = right
        self._bottom = bottom

    def left(self) -> int:
        return self._left

    def top(self) -> int:
        return self._top

    def right(self) -> int:
        return self._right

    def bottom(self) -> int:
        return self._bottom


def test_mouse_dead_zone_blocks_small_tilt():
    config = MouseConfig(dead_zone_degrees=3.0)
    velocity = tilt_to_velocity(roll=2.0, pitch=-2.0, config=config)
    assert velocity.vx == 0.0
    assert velocity.vy == 0.0


def test_mouse_velocity_uses_radial_dead_zone_for_cardinal_tilt():
    config = MouseConfig(
        dead_zone_degrees=3.0,
        speed_px_per_second_per_degree=70.0,
        max_speed_px_per_second=1400.0,
        response_curve=1.0,
    )
    velocity = tilt_to_velocity(roll=5.0, pitch=0.0, config=config)
    assert velocity.vx == 140.0
    assert velocity.vy == 0.0

    velocity = tilt_to_velocity(roll=0.0, pitch=-6.0, config=config)
    assert velocity.vx == 0.0
    assert velocity.vy == -210.0


def test_mouse_response_curve_softens_small_tilt():
    config = MouseConfig(
        dead_zone_degrees=1.25,
        speed_px_per_second_per_degree=70.0,
        max_speed_px_per_second=1400.0,
        response_curve=1.35,
        full_tilt_degrees=18.0,
    )
    linear = MouseConfig(
        dead_zone_degrees=1.25,
        speed_px_per_second_per_degree=70.0,
        max_speed_px_per_second=1400.0,
        response_curve=1.0,
        full_tilt_degrees=18.0,
    )
    curved = tilt_to_velocity(roll=3.0, pitch=0.0, config=config)
    straight = tilt_to_velocity(roll=3.0, pitch=0.0, config=linear)
    assert 0.0 < curved.vx < straight.vx
    assert curved.vy == 0.0


def test_mouse_velocity_clamps_and_inverts():
    config = MouseConfig(
        dead_zone_degrees=0.0,
        speed_px_per_second_per_degree=100.0,
        max_speed_px_per_second=300.0,
        response_curve=1.0,
        invert_x=True,
        invert_y=True,
    )
    velocity = tilt_to_velocity(roll=10.0, pitch=-10.0, config=config)
    assert velocity.vx == -300.0
    assert velocity.vy == 300.0


def test_mouse_velocity_uses_calibrated_cursor_axes():
    calibration = calibrated_axes_from_samples(
        neutral=(0.0, 0.0),
        left_tilt=(0.0, 12.0),
        right_tilt=(0.0, -12.0),
        up_tilt=(10.0, 0.0),
        down_tilt=(-10.0, 0.0),
    )
    config = with_cursor_axis_calibration(
        MouseConfig(
            dead_zone_degrees=3.0,
            speed_px_per_second_per_degree=70.0,
            max_speed_px_per_second=1400.0,
            response_curve=1.0,
        ),
        calibration,
    )

    right_velocity = tilt_to_velocity(roll=0.0, pitch=-5.0, config=config)
    up_velocity = tilt_to_velocity(roll=6.0, pitch=0.0, config=config)
    down_velocity = tilt_to_velocity(roll=-6.0, pitch=0.0, config=config)

    assert right_velocity.vx == 140.0
    assert right_velocity.vy == 0.0
    assert up_velocity.vx == 0.0
    assert up_velocity.vy == -210.0
    assert down_velocity.vx == 0.0
    assert down_velocity.vy == 210.0


def test_cursor_axis_calibration_rejects_degenerate_tilts():
    with pytest.raises(ValueError, match="too similar"):
        calibrated_axes_from_samples(
            neutral=(0.0, 0.0),
            left_tilt=(8.0, 0.0),
            right_tilt=(10.0, 0.0),
            up_tilt=(11.0, 0.0),
            down_tilt=(12.0, 0.0),
        )


def test_left_click_falls_back_to_osascript_on_macos(monkeypatch):
    prompted = []

    monkeypatch.setattr("neuro_cursor.mouse_control._post_left_click_quartz", lambda x, y: False)
    monkeypatch.setattr("neuro_cursor.mouse_control._post_left_click_coregraphics", lambda x, y: False)
    monkeypatch.setattr("neuro_cursor.mouse_control.sys.platform", "darwin")
    monkeypatch.setattr("neuro_cursor.mouse_control.request_accessibility_prompt", lambda: prompted.append(True) or True)

    assert post_left_click(12, 34) is False
    assert prompted == [True]


def test_left_click_returns_false_when_no_backend(monkeypatch):
    monkeypatch.setattr("neuro_cursor.mouse_control._post_left_click_quartz", lambda x, y: False)
    monkeypatch.setattr("neuro_cursor.mouse_control._post_left_click_coregraphics", lambda x, y: False)
    monkeypatch.setattr("neuro_cursor.mouse_control.sys.platform", "linux")

    assert post_left_click(12, 34) is False


def test_left_click_uses_coregraphics_before_prompt(monkeypatch):
    monkeypatch.setattr("neuro_cursor.mouse_control._post_left_click_quartz", lambda x, y: False)
    monkeypatch.setattr("neuro_cursor.mouse_control._post_left_click_coregraphics", lambda x, y: True)

    assert post_left_click(12, 34) is True


def test_mouse_move_uses_native_event_before_qt_fallback(monkeypatch):
    posted = []
    qt_positions = []
    controller = QtCursorController(MouseConfig())

    monkeypatch.setattr(
        "neuro_cursor.mouse_control.QCursor.pos",
        lambda: FakeCursorPoint(10, 20),
    )
    monkeypatch.setattr(
        "neuro_cursor.mouse_control._virtual_screen_geometry",
        lambda: FakeScreenBounds(0, 0, 100, 100),
    )
    monkeypatch.setattr(
        "neuro_cursor.mouse_control.post_mouse_move",
        lambda x, y, dx, dy: posted.append((x, y, dx, dy)) or True,
    )
    monkeypatch.setattr(
        "neuro_cursor.mouse_control.QCursor.setPos",
        lambda point: qt_positions.append((point.x(), point.y())),
    )

    controller._move_by(3, -4)

    assert posted == [(13, 16, 3, -4)]
    assert qt_positions == []


def test_mouse_move_falls_back_to_qt_when_native_event_fails(monkeypatch):
    qt_positions = []
    controller = QtCursorController(MouseConfig())

    monkeypatch.setattr(
        "neuro_cursor.mouse_control.QCursor.pos",
        lambda: FakeCursorPoint(10, 20),
    )
    monkeypatch.setattr(
        "neuro_cursor.mouse_control._virtual_screen_geometry",
        lambda: FakeScreenBounds(0, 0, 100, 100),
    )
    monkeypatch.setattr(
        "neuro_cursor.mouse_control.post_mouse_move",
        lambda *args: False,
    )
    monkeypatch.setattr(
        "neuro_cursor.mouse_control.QCursor.setPos",
        lambda point: qt_positions.append((point.x(), point.y())),
    )

    controller._move_by(3, -4)

    assert qt_positions == [(13, 16)]


def test_mouse_move_clamps_to_virtual_screen_bounds(monkeypatch):
    posted = []
    controller = QtCursorController(MouseConfig())
    controller._rem_x = 1.5
    controller._rem_y = -0.5

    monkeypatch.setattr(
        "neuro_cursor.mouse_control.QCursor.pos",
        lambda: FakeCursorPoint(95, 5),
    )
    monkeypatch.setattr(
        "neuro_cursor.mouse_control._virtual_screen_geometry",
        lambda: FakeScreenBounds(0, 0, 100, 100),
    )
    monkeypatch.setattr(
        "neuro_cursor.mouse_control.post_mouse_move",
        lambda x, y, dx, dy: posted.append((x, y, dx, dy)) or True,
    )

    controller._move_by(20, -10)

    assert posted == [(100, 0, 5, -5)]
    assert controller._rem_x == 0.0
    assert controller._rem_y == 0.0


def test_mouse_move_native_helper_returns_false_when_no_backend(monkeypatch):
    monkeypatch.setattr("neuro_cursor.mouse_control.sys.platform", "linux")

    assert post_mouse_move(12, 34) is False
