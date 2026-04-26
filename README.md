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

The current EXG default is the one-sensor eyebrow setup: channel `2` is active,
routed to RLD at gain `12`, and used for `eyebrow_raise` detection.

## Cursor Control

Cursor movement is intentionally GUI-only and defaults to disarmed. Start the
stream, hold your head in a neutral pose, then click `Calibrate Cursor Axes`.
Hold neutral, tilt left, tilt right, tilt up, then tilt down when prompted; the
app learns how the current headset mounting maps corrected roll/pitch onto
screen X/Y. Click `Arm Cursor` after calibration. The cursor uses a small
radial deadzone, curved speed response, and a motion boost for quicker head
moves. Use `Cursor DZ`, `Cursor speed`, `Invert X`, and `Invert Y` in the top
bar if the direction or sensitivity feels wrong. When the head is still near
neutral for several seconds, the app auto-zeros orientation drift.

Press `Esc` while the diagnostics window is focused, click `Disarm Cursor`, stop
the stream, or close the app to stop cursor motion and eyebrow-click behavior.
When armed, a detected `eyebrow_raise` sends one left click with a refractory
delay so a single raise does not repeat rapidly. It does not drag.

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

## Short-Event Data Recording

Use `Start Eyebrow Calibration` in the GUI for a guided short-event session.
New guided sessions use only eyebrow prompts: 8 seconds of neutral baseline, 20
randomized eyebrow raises, 20 hard-negative prompts, and 6 seconds of neutral
baseline at the end. Hold prompts and jaw clench collection are disabled for the
current workflow.

Each guided session includes:

- `raw.npz`: full 22-row BrainFlow data
- `exg.csv`: 8 EXG columns, compatible with NeuroPawn visualizer-style CSVs
- `labels.jsonl`: event-level labels such as `eyebrow_raise_start/end` and
  `hard_negative_*`
- `clips.npz`: clipped windows around each short event and hard negative
- `features.json`: channel-2 RMS, peak-to-peak, slope, bandpower, envelope stats
- `metadata.json`: config, row map, active EXG channels, and connection info

The live preview shows one confidence value, detected event count, click count,
and state: `relaxed` or the active positive label. Clicks are sent only after
you explicitly arm cursor control.

For terminal captures:

```bash
uv run neuro-capture --config config/neuro_cursor.yaml --seconds 10 --label eyebrow_raise
```

To import an 8-column EXG CSV:

```bash
uv run neuro-capture --config config/neuro_cursor.yaml --from-csv /path/to/eyebrow.csv --duration-seconds 30 --label eyebrow_raise
```

For externally recorded files, add eyebrow event labels with center times.
`--hold-intervals` is kept only for importing older hold-labeled files.

```bash
uv run neuro-capture --config config/neuro_cursor.yaml --from-csv /path/to/eyebrow.csv --duration-seconds 30 --label eyebrow_raise --event-times 6.5,10.2,14.0
```

To validate with held-out sessions and run the model bakeoff:

```bash
uv run neuro-evaluate-jaw --config config/neuro_cursor.yaml --positive-label eyebrow_raise --profile eyebrow --profile-dir models/eyebrow --train-sessions data/sessions/<train1> data/sessions/<train2> --validation-sessions data/sessions/<validation>
```

For the full eyebrow-only collection sequence, follow
[docs/data_collection_protocol.md](docs/data_collection_protocol.md).

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
