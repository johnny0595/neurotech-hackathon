import pytest

from neuro_cursor.config import MouseConfig
from neuro_cursor.mouse_control import (
    calibrated_axes_from_samples,
    tilt_to_velocity,
    with_cursor_axis_calibration,
)


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
