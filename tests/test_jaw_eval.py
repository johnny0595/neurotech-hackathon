import json

import numpy as np
import pytest

from neuro_cursor.config import AppConfig
from neuro_cursor.jaw_eval import (
    classification_metrics,
    session_quality,
    threshold_sweep,
    train_and_evaluate_profile,
)
from neuro_cursor.rows import NUM_ROWS


def _write_session(root, name, clench_amp=80.0, include_all_clenches=True):
    config = AppConfig()
    config.jaw.sampling_rate = 20.0
    config.jaw.short_clench_reps = 2
    config.jaw.short_clench_seconds = 0.5
    raw = np.zeros((NUM_ROWS, 240), dtype=float)
    raw[20] = np.arange(raw.shape[1], dtype=float) / config.jaw.sampling_rate
    labels = [
        {
            "label": "neutral",
            "type": "interval",
            "start_sample": 0,
            "end_sample": 40,
            "trial_type": "neutral",
        },
        {
            "label": "jaw_clench",
            "type": "interval",
            "start_sample": 50,
            "end_sample": 60,
            "trial_type": "short_clench",
        },
        {
            "label": "neutral",
            "type": "interval",
            "start_sample": 80,
            "end_sample": 130,
            "trial_type": "neutral",
        },
    ]
    raw[2, 50:60] = clench_amp * np.array([0, 1, -1, 0.8, -0.8, 1.2, -1.2, 0.6, -0.6, 0.0])
    if include_all_clenches:
        labels.append(
            {
                "label": "jaw_clench",
                "type": "interval",
                "start_sample": 150,
                "end_sample": 160,
                "trial_type": "short_clench",
            }
        )
        raw[2, 150:160] = clench_amp * np.array([0, 1.1, -1.1, 0.9, -0.9, 1.3, -1.3, 0.7, -0.7, 0.0])
    labels.append(
        {
            "label": "neutral",
            "type": "interval",
            "start_sample": 180,
            "end_sample": 230,
            "trial_type": "neutral",
        }
    )
    session = root / name
    session.mkdir()
    np.savez_compressed(session / "raw.npz", raw=raw)
    with (session / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label) + "\n")
    with (session / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {"jaw": config.jaw.__dict__},
                "metadata": {"label": "jaw_guided_calibration", "prompt_schedule": ["x"]},
            },
            handle,
        )
    return session, config


def test_threshold_sweep_selects_lowest_threshold_that_meets_false_positive_limit():
    rows = [
        {"true_label": 0, "probability": 0.10, "duration_seconds": 1.0},
        {"true_label": 0, "probability": 0.20, "duration_seconds": 1.0},
        {"true_label": 1, "probability": 0.80, "duration_seconds": 1.0},
    ]

    threshold, sweep = threshold_sweep(rows, max_false_positives_per_minute=0.0, min_precision=0.70)
    metrics = classification_metrics(rows, threshold)

    assert threshold == 0.21
    assert metrics["false_positive"] == 0
    assert sweep


def test_session_holdout_evaluation_writes_model_and_report(tmp_path):
    train, config = _write_session(tmp_path, "train")
    validation, _ = _write_session(tmp_path, "validation", clench_amp=90.0)

    report = train_and_evaluate_profile(
        [train],
        [validation],
        config,
        report_root=tmp_path / "reports",
        profile_dir=tmp_path / "model",
        max_false_positives_per_minute=10.0,
        min_precision=0.5,
    )

    assert report["metadata"]["train_sessions"] == [str(train)]
    assert report["metadata"]["validation_sessions"] == [str(validation)]
    assert (tmp_path / "model" / "jaw_event.joblib").exists()
    assert (tmp_path / "model" / "metadata.json").exists()
    assert (tmp_path / "reports").exists()
    assert report["metrics"]["true_positive"] >= 1


def test_session_holdout_rejects_overlapping_train_and_validation(tmp_path):
    session, config = _write_session(tmp_path, "session")

    with pytest.raises(ValueError, match="disjoint"):
        train_and_evaluate_profile([session], [session], config, report_root=tmp_path / "reports")


def test_quality_gate_rejects_incomplete_guided_session(tmp_path):
    session, config = _write_session(tmp_path, "incomplete", include_all_clenches=False)
    raw = np.load(session / "raw.npz")["raw"]
    labels = [json.loads(line) for line in (session / "labels.jsonl").read_text().splitlines()]

    quality = session_quality(session, raw, labels, config.jaw)

    assert quality["errors"]
    assert "incomplete guided clench intervals" in quality["errors"][0]
