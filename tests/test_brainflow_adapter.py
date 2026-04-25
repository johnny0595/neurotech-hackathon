from neuro_cursor.brainflow_adapter import channel_config_commands
from neuro_cursor.config import AppConfig


def test_channel_config_commands_match_two_sensor_jaw_setup():
    config = AppConfig()
    config.board.active_exg_channels = [1, 2]
    config.board.rld_channels = [1, 2]

    commands = channel_config_commands(config)

    assert "chon_1_12" in commands
    assert "chon_2_12" in commands
    assert "rldadd_1" in commands
    assert "rldadd_2" in commands
    assert "choff_3" in commands
    assert "choff_5" in commands
    assert "choff_6" in commands
    assert "choff_8" in commands
    assert "chon_7_12" not in commands
