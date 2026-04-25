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

The current EXG default is the one-sensor jaw setup: channel `2` is active and
routed to RLD at gain `12`.

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

## EXG Visualizer

The GUI has a NeuroPawn-style EXG panel:

- stacked traces for rows `1-8`
- frequency analysis for the active channels
- per-channel `Active`, `Route RLD`, and `Gain` controls
- `Capture Snapshot`, which writes row stats and active-channel status to
  `data/snapshots/`

Channel-control changes apply on the next stream start. If you change channels
while streaming, stop and start the stream so the app can resend `chon_*`,
`rldadd_*` or `rldremove_*`, and `choff_*` commands.

## Jaw Data Recording

Use `Start Jaw Calibration` in the GUI for a guided session: neutral baseline,
five short jaw clenches, and three clench-holds. The GUI writes event-level
labels and trains a personal jaw model for live preview. `Start Jaw Recording`
is still available for manually labeled whole-session captures.

Each guided session includes:

- `raw.npz`: full 22-row BrainFlow data
- `exg.csv`: 8 EXG columns, compatible with NeuroPawn visualizer-style CSVs
- `labels.jsonl`: `jaw_clench_start/end` and `jaw_hold_start/end` event labels
- `clips.npz`: clipped windows around each clench and hold
- `features.json`: channel-2 jaw RMS, peak-to-peak, slope, bandpower, envelope stats
- `metadata.json`: config, row map, active EXG channels, and connection info

The live preview shows jaw event confidence, hold confidence, detected event
count, and state. It does not click or drag the mouse.

For terminal captures:

```bash
uv run neuro-capture --config config/neuro_cursor.yaml --seconds 10 --label jaw_clench
```

To import an 8-column EXG CSV:

```bash
uv run neuro-capture --config config/neuro_cursor.yaml --from-csv /path/to/recording.csv --duration-seconds 30 --label jaw_clench
```

For externally recorded files, add event labels with clench center times or hold
intervals:

```bash
uv run neuro-capture --config config/neuro_cursor.yaml --from-csv /path/to/recording.csv --duration-seconds 30 --label jaw_clench --event-times 6.5,10.2,14.0
uv run neuro-capture --config config/neuro_cursor.yaml --from-csv /path/to/recording.csv --duration-seconds 30 --label jaw_hold --hold-intervals 5.0:8.0,12.0:15.0
```

To train from one or more guided sessions:

```bash
uv run neuro-train-jaw data/sessions/<timestamp>
```

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
