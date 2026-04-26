# Eyebrow Raise Data Collection Protocol

This protocol is for the current one-sensor setup: EEG channel 2 only. The
active target is `eyebrow_raise`; jaw clench is not part of the current
collection or model selection workflow.

## Before Recording

1. Launch the GUI:

   ```bash
   uv run neuro-cursor --config config/neuro_cursor.yaml
   ```

2. Connect the board and click `Start Stream`.
3. Confirm the EXG panel shows `EEG 2` moving and the status panel reports live packets.
4. Keep cursor control disarmed. Do not use click/drag during data collection.
5. Sit in the same posture you expect during live use. Keep electrode contact stable.

## What One Guided Session Records

The GUI `Start Eyebrow Calibration` button records one short-event session:

- 8 seconds neutral baseline.
- 20 randomized eyebrow-raise prompts.
- 20 hard negatives:
  - blink naturally
  - swallow
  - small head movement with face relaxed
  - light teeth touch with no eyebrow raise
- 6 seconds final neutral baseline.

Each session takes about 2 minutes. The app writes the session to
`data/sessions/<timestamp>/` with `raw.npz`, `exg.csv`, `labels.jsonl`,
`clips.npz`, `features.json`, and `metadata.json`.

## Minimum Collection

Record 5 complete guided sessions:

1. Normal eyebrow raises.
2. Normal eyebrow raises.
3. Normal eyebrow raises.
4. Light eyebrow raises.
5. Natural relaxed face; raise eyebrows only when prompted.

This gives:

- 100 eyebrow-raise positives.
- 100 hard negatives total.
- 25 examples of each hard-negative type.

## Better Collection

If time allows, record 3 more complete guided sessions:

1. Normal eyebrow raises with small relaxed head motion between prompts.
2. Light eyebrow raises.
3. Strong eyebrow raises.

This brings the set to:

- 160 eyebrow-raise positives.
- 160 hard negatives total.
- 40 examples of each hard-negative type.

## GUI Steps For Each Session

1. Confirm the protocol selector says `Eyebrow raise`.
2. Click `Start Eyebrow Calibration`.
3. Follow only the big prompt text. Raise eyebrows only during the eyebrow prompt.
4. During `Relax`, return to neutral and avoid preparing early.
5. During hard-negative prompts, keep eyebrows relaxed:
   - Blink without raising eyebrows.
   - Swallow without raising eyebrows.
   - Move head lightly without raising eyebrows.
   - Touch teeth lightly without raising eyebrows.
6. Wait for the app to save after the session completes.
7. Write down the session timestamp and condition, for example:
   `20260425-221500 eyebrow normal`.

## Reject And Redo A Session If

- The stream drops or packet rate freezes.
- The electrode moves or loses contact.
- You raise eyebrows outside the prompt repeatedly.
- You miss more than 2 prompted events.
- The session ends early.
- The GUI logs a save or training error.

Move rejected sessions out of `data/sessions/` before running QA.

## Train And Validate

Use complete sessions only. Split by whole session, never random windows.

After 5 eyebrow sessions:

```bash
uv run neuro-evaluate-jaw \
  --config config/neuro_cursor.yaml \
  --positive-label eyebrow_raise \
  --profile eyebrow \
  --profile-dir models/eyebrow \
  --train-sessions data/sessions/<eyebrow1> data/sessions/<eyebrow2> data/sessions/<eyebrow3> data/sessions/<eyebrow4> \
  --validation-sessions data/sessions/<eyebrow5>
```

The model is usable only if relaxed false positives stay low. Prefer
`false_positives_per_minute <= 0.5`; lower is better, even if recall drops.
