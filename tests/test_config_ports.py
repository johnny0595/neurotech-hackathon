from pathlib import Path

import pytest

from neuro_cursor.config import AppConfig, validate_config
from neuro_cursor.ports import SerialPortNotFound, resolve_serial_port


def test_default_config_validates():
    validate_config(AppConfig())


def test_invalid_gain_rejected():
    config = AppConfig()
    config.board.gain = 24
    with pytest.raises(ValueError):
        validate_config(config)


def test_invalid_per_channel_gain_rejected():
    config = AppConfig()
    config.board.channel_gains[7] = 24
    with pytest.raises(ValueError):
        validate_config(config)


def test_invalid_mouse_smoothing_rejected():
    config = AppConfig()
    config.mouse.smoothing = 1.0
    with pytest.raises(ValueError):
        validate_config(config)


def test_invalid_mouse_response_rejected():
    config = AppConfig()
    config.mouse.response_curve = 0.8
    with pytest.raises(ValueError):
        validate_config(config)


def test_invalid_mouse_full_tilt_rejected():
    config = AppConfig()
    config.mouse.dead_zone_degrees = 2.0
    config.mouse.full_tilt_degrees = 2.0
    with pytest.raises(ValueError):
        validate_config(config)


def test_degenerate_cursor_axis_calibration_rejected():
    config = AppConfig()
    config.mouse.x_axis_roll = 1.0
    config.mouse.x_axis_pitch = 0.0
    config.mouse.y_axis_roll = 2.0
    config.mouse.y_axis_pitch = 0.0
    with pytest.raises(ValueError):
        validate_config(config)


def test_invalid_jaw_channel_rejected():
    config = AppConfig()
    config.jaw.channels = [9]
    with pytest.raises(ValueError):
        validate_config(config)


def test_existing_preferred_port_wins(tmp_path, monkeypatch):
    fake = tmp_path / "cu.usbserial-TEST"
    fake.touch()
    monkeypatch.setattr("neuro_cursor.ports.list_usbserial_ports", lambda: ["/dev/cu.usbserial-OTHER"])
    result = resolve_serial_port(str(fake))
    assert result.port == str(fake)


def test_missing_ports_fail_clearly(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setattr("neuro_cursor.ports.list_usbserial_ports", lambda: [])
    with pytest.raises(SerialPortNotFound):
        resolve_serial_port("/dev/cu.usbserial-MISSING")
