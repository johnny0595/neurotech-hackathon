import json

import numpy as np

from neuro_cursor.config import AppConfig
from neuro_cursor.jaw_model import JawPredictor, train_profile
from neuro_cursor.rows import NUM_ROWS


def test_jaw_model_training_writes_artifacts(tmp_path):
    config = AppConfig()
    config.jaw.sampling_rate = 20.0
    raw = np.zeros((NUM_ROWS, 200), dtype=float)
    raw[2, 40:50] = 50.0
    session = tmp_path / "session"
    session.mkdir()
    np.savez_compressed(session / "raw.npz", raw=raw)
    labels = [
        {"label": "neutral", "type": "interval", "start_sample": 0, "end_sample": 30},
        {"label": "neutral", "type": "interval", "start_sample": 70, "end_sample": 110},
        {"label": "eyebrow_raise", "type": "interval", "start_sample": 40, "end_sample": 50},
    ]
    with (session / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label) + "\n")

    profile = train_profile([session], config, profile_dir=tmp_path / "model")
    predictor = JawPredictor.load(profile, config.jaw)
    prediction = predictor.predict(raw[:, 35:55])

    assert (profile / "jaw_event.joblib").exists()
    assert (profile / "metadata.json").exists()
    assert prediction.event_confidence >= 0.0
    assert prediction.state in {"relaxed", "eyebrow_raise"}


def test_jaw_model_uses_session_sampling_rate_metadata(tmp_path):
    config = AppConfig()
    raw = np.zeros((NUM_ROWS, 200), dtype=float)
    raw[2, 40:50] = 50.0
    session = tmp_path / "session"
    session.mkdir()
    np.savez_compressed(session / "raw.npz", raw=raw)
    labels = [
        {"label": "neutral", "type": "interval", "start_sample": 0, "end_sample": 30},
        {"label": "eyebrow_raise", "type": "interval", "start_sample": 40, "end_sample": 50},
    ]
    with (session / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label) + "\n")
    with (session / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": {"jaw": {"channels": [2], "sampling_rate": 1000.0}}}, handle)

    profile = train_profile([session], config, profile_dir=tmp_path / "model")
    metadata = json.loads((profile / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["sampling_rate_hz"] == 1000.0
