"""Serial-port discovery for the Knight IMU board."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from glob import glob
from pathlib import Path


@dataclass(frozen=True)
class SerialPortResolution:
    port: str
    available_ports: list[str]
    message: str


class SerialPortNotFound(RuntimeError):
    pass


def _pyserial_ports() -> list[str]:
    try:
        from serial.tools import list_ports
    except Exception:
        return []
    ports: list[str] = []
    for port in list_ports.comports():
        device = port.device
        haystack = " ".join(
            str(value or "")
            for value in (device, port.description, port.hwid, port.manufacturer)
        ).upper()
        if os.name == "nt" or any(
            marker in haystack
            for marker in ("USB", "FTDI", "UART", "USBSERIAL", "USBMODEM", "TTYUSB", "TTYACM")
        ):
            ports.append(device)
    return sorted(ports)


def list_usbserial_ports() -> list[str]:
    """List likely serial ports on macOS, Linux, and Windows."""

    candidates = [
        *glob("/dev/cu.usbserial-*"),
        *glob("/dev/cu.SLAB_USBtoUART*"),
        *glob("/dev/cu.usbmodem*"),
        *glob("/dev/ttyUSB*"),
        *glob("/dev/ttyACM*"),
        *_pyserial_ports(),
    ]
    return sorted(dict.fromkeys(candidates))


def resolve_serial_port(preferred: str | None) -> SerialPortResolution:
    available = list_usbserial_ports()
    if preferred and (Path(preferred).exists() or (os.name == "nt" and _looks_like_com_port(preferred))):
        return SerialPortResolution(
            port=preferred,
            available_ports=available,
            message=f"Using configured serial port {preferred}",
        )

    if available:
        chosen = available[0]
        if preferred:
            message = (
                f"Configured serial port {preferred} was not found; "
                f"using discovered USB serial port {chosen}"
            )
        else:
            message = f"Using discovered USB serial port {chosen}"
        return SerialPortResolution(port=chosen, available_ports=available, message=message)

    detail = f"Configured port: {preferred}" if preferred else "No configured port."
    raise SerialPortNotFound(
        "No likely USB serial ports found. "
        f"{detail} Connect the Knight IMU board and retry. "
        "On macOS use /dev/cu.*; on Windows set a COM port such as COM3."
    )


def _looks_like_com_port(port: str) -> bool:
    return bool(re.fullmatch(r"COM\d+", port.upper()))
