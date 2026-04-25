from neuro_cursor.config import MouseConfig
from neuro_cursor.mouse_control import tilt_to_velocity


def test_mouse_dead_zone_blocks_small_tilt():
    config = MouseConfig(dead_zone_degrees=3.0)
    velocity = tilt_to_velocity(roll=2.5, pitch=-2.5, config=config)
    assert velocity.vx == 0.0
    assert velocity.vy == 0.0


def test_mouse_velocity_uses_corrected_roll_and_pitch():
    config = MouseConfig(
        dead_zone_degrees=3.0,
        speed_px_per_second_per_degree=70.0,
        max_speed_px_per_second=1400.0,
    )
    velocity = tilt_to_velocity(roll=5.0, pitch=-6.0, config=config)
    assert velocity.vx == 140.0
    assert velocity.vy == -210.0


def test_mouse_velocity_clamps_and_inverts():
    config = MouseConfig(
        dead_zone_degrees=0.0,
        speed_px_per_second_per_degree=100.0,
        max_speed_px_per_second=300.0,
        invert_x=True,
        invert_y=True,
    )
    velocity = tilt_to_velocity(roll=10.0, pitch=-10.0, config=config)
    assert velocity.vx == -300.0
    assert velocity.vy == 300.0
