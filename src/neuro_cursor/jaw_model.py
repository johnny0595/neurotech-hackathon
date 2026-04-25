"""Personal jaw clench model training and live inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig, JawConfig, load_config
from .jaw_features import (
    FEATURE_NAMES,
    interval_labels,
    jaw_feature_dict,
    jaw_feature_vector,
    jaw_signal,
    neutral_reference,
    read_labels,
)


@dataclass(frozen=True)
class JawPrediction:
    event_confidence: float
    hold_confidence: float
    state: str


def load_session(session_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    raw_path = session_dir / "raw.npz"
    labels_path = session_dir / "labels.jsonl"
    with np.load(raw_path) as archive:
        raw = np.asarray(archive["raw"], dtype=float)
    return raw, read_labels(labels_path)


def train_profile(
    session_dirs: list[Path],
    config: AppConfig,
    profile_dir: Path | None = None,
) -> Path:
    if not session_dirs:
        raise ValueError("at least one session directory is required")
    target_dir = profile_dir or Path("models") / config.jaw.profile_name
    target_dir.mkdir(parents=True, exist_ok=True)

    event_x, event_y, hold_x, hold_y = [], [], [], []
    references: list[dict[str, float]] = []
    for session_dir in session_dirs:
        raw, labels = load_session(session_dir)
        reference = neutral_reference(raw, labels, config.jaw)
        references.append(reference)
        exg = jaw_signal(raw, config.jaw)
        event_examples = _training_examples(exg, labels, config.jaw, "jaw_clench", reference)
        hold_examples = _training_examples(exg, labels, config.jaw, "jaw_hold", reference)
        event_x.extend([row[0] for row in event_examples])
        event_y.extend([row[1] for row in event_examples])
        hold_x.extend([row[0] for row in hold_examples])
        hold_y.extend([row[1] for row in hold_examples])

    event_model = _fit_classifier(event_x, event_y)
    hold_model = _fit_classifier(hold_x, hold_y)
    _dump_model(target_dir / "jaw_event.joblib", event_model)
    _dump_model(target_dir / "jaw_hold_state.joblib", hold_model)

    neutral = _median_reference(references)
    metadata = {
        "profile_name": config.jaw.profile_name,
        "jaw_channels": list(config.jaw.channels),
        "sampling_rate_hz": config.jaw.sampling_rate,
        "feature_names": list(FEATURE_NAMES),
        "neutral_reference": neutral,
        "sessions": [str(path) for path in session_dirs],
        "event_examples": len(event_y),
        "hold_examples": len(hold_y),
    }
    with (target_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return target_dir


class JawPredictor:
    def __init__(
        self,
        event_model: Any,
        hold_model: Any,
        neutral: dict[str, float],
        config: JawConfig,
    ) -> None:
        self.event_model = event_model
        self.hold_model = hold_model
        self.neutral = neutral
        self.config = config

    @classmethod
    def load(cls, profile_dir: Path, config: JawConfig) -> "JawPredictor":
        metadata_path = profile_dir / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return cls(
            event_model=_load_model(profile_dir / "jaw_event.joblib"),
            hold_model=_load_model(profile_dir / "jaw_hold_state.joblib"),
            neutral=dict(metadata.get("neutral_reference") or {}),
            config=config,
        )

    def predict(self, raw: np.ndarray) -> JawPrediction:
        exg = jaw_signal(raw, self.config)
        event_samples = max(2, int(round(self.config.short_clench_seconds * self.config.sampling_rate)))
        hold_samples = max(event_samples, int(round(min(self.config.hold_seconds, 1.0) * self.config.sampling_rate)))
        event_conf = _predict_probability(
            self.event_model,
            jaw_feature_vector(exg[-event_samples:], self.config.sampling_rate, self.neutral),
        )
        hold_conf = _predict_probability(
            self.hold_model,
            jaw_feature_vector(exg[-hold_samples:], self.config.sampling_rate, self.neutral),
        )
        state = "holding" if hold_conf >= self.config.hold_threshold else "relaxed"
        if event_conf >= self.config.event_threshold and state == "relaxed":
            state = "clench_event"
        return JawPrediction(event_confidence=event_conf, hold_confidence=hold_conf, state=state)


def train_from_args(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train personal jaw clench models from sessions")
    parser.add_argument("sessions", nargs="+", help="Session directories containing raw.npz and labels.jsonl")
    parser.add_argument("--config", default="config/neuro_cursor.yaml")
    parser.add_argument("--profile", default=None, help="Override profile name")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.profile:
        config.jaw.profile_name = args.profile
    path = train_profile([Path(session) for session in args.sessions], config)
    print(f"Jaw model saved: {path}")
    return 0


def _training_examples(
    exg: np.ndarray,
    labels: list[dict[str, Any]],
    config: JawConfig,
    positive_label: str,
    reference: dict[str, float],
) -> list[tuple[np.ndarray, int]]:
    rows: list[tuple[np.ndarray, int]] = []
    intervals = interval_labels(labels)
    positive_intervals = [row for row in intervals if row.get("label") == positive_label]
    neutral_intervals = [row for row in intervals if row.get("label") == "neutral"]
    for row in positive_intervals:
        start, end = int(row["start_sample"]), int(row["end_sample"])
        rows.append((jaw_feature_vector(exg[start:end], config.sampling_rate, reference), 1))
    window_seconds = config.short_clench_seconds if positive_label == "jaw_clench" else min(1.0, config.hold_seconds)
    window = max(2, int(round(window_seconds * config.sampling_rate)))
    for row in neutral_intervals:
        start, end = int(row["start_sample"]), int(row["end_sample"])
        for neg_start in range(start, max(start, end - window + 1), window):
            neg_end = min(end, neg_start + window)
            if neg_end - neg_start >= max(2, window // 2):
                rows.append((jaw_feature_vector(exg[neg_start:neg_end], config.sampling_rate, reference), 0))
    return rows


def _fit_classifier(features: list[np.ndarray], labels: list[int]) -> Any:
    if not features:
        return ConstantProbability(0.0)
    y = np.asarray(labels, dtype=int)
    if len(set(y.tolist())) < 2:
        return ConstantProbability(float(np.mean(y)))
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.vstack(features)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(x, y)


class ConstantProbability:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        return np.tile(np.array([[1.0 - self.probability, self.probability]], dtype=float), (n, 1))


def _predict_probability(model: Any, vector: np.ndarray) -> float:
    proba = model.predict_proba(vector.reshape(1, -1))
    return float(proba[0, 1])


def _dump_model(path: Path, model: Any) -> None:
    from joblib import dump

    dump(model, path)


def _load_model(path: Path) -> Any:
    from joblib import load

    return load(path)


def _median_reference(references: list[dict[str, float]]) -> dict[str, float]:
    if not references:
        return {}
    keys = sorted({key for reference in references for key in reference})
    return {key: float(np.median([reference.get(key, 0.0) for reference in references])) for key in keys}
