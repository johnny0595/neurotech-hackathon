# NeuroPawn Head Cursor

Phase 1 is a hardware diagnostics app for the NeuroPawn Knight IMU board. The
GUI can also move the system cursor from IMU tilt, but only after you explicitly
click `Arm Cursor` each launch.

## Quick Start

```bash
uv sync --extra dev
uv run neuro-smoke --config config/neuro_cursor.yaml
uv run neuro-cursor --config config/neuro_cursor.yaml
```

The default config prefers `/dev/cu.usbserial-A5069RR4` for this Mac. If that
port is not present, the app searches likely USB serial ports and fails clearly
when no board is connected. Windows users should set `board.serial_port` to
their COM port, for example `COM3`.

## Cursor Control

Cursor movement is intentionally GUI-only and defaults to disarmed. Start the
stream, click `Zero (level)` with your head in a neutral pose, then click
`Arm Cursor`. Corrected roll controls horizontal cursor velocity and corrected
pitch controls vertical cursor velocity. Use `Cursor DZ`, `Cursor speed`,
`Invert X`, and `Invert Y` in the top bar if the direction or sensitivity feels
wrong.

Press `Esc` while the diagnostics window is focused, click `Disarm Cursor`, stop
the stream, or close the app to stop cursor motion. This implementation moves
the pointer only; it does not click or drag.

On macOS, grant the terminal or app used to launch `uv run neuro-cursor`
Accessibility permission in System Settings > Privacy & Security > Accessibility
if pointer control is blocked or unreliable.

## BrainFlow Knight IMU Rows

The app uses explicit row constants for `NEUROPAWN_KNIGHT_BOARD_IMU`:

- `0`: package counter
- `1-8`: EXG
- `9-10`: LOFF STATP/STATN
- `11-13`: accelerometer
- `14-16`: gyroscope
- `17-19`: magnetometer
- `20`: timestamp
- `21`: marker

BrainFlow exposes IMU rows as `other_channels` for this board, so the app does
not call `get_accel_channels()` or `get_gyro_channels()` for Knight IMU data.

## Scripts

- `debug_channels.py`: compatibility wrapper for the row-based smoke test.
- `head_mouse.py`: intentionally refuses direct pointer control; use the GUI
  `Arm Cursor` control instead.
