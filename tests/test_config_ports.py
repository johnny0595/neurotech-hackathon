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
