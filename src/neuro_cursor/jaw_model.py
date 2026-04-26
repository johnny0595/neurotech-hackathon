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

CLASSIFIER_NAMES = (
    "extra_trees",
    "random_forest",
    "hist_gradient_boosting",
    "logistic_regression",
    "emg_rms_threshold",
)


@dataclass(frozen=True)
class JawPrediction:
    event_confidence: float
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

    event_x, event_y = [], []
    references: list[dict[str, float]] = []
    session_rates: list[float] = []
    session_channels: list[list[int]] = []
    positive_labels: list[str] = []
    for session_dir in session_dirs:
        raw, labels = load_session(session_dir)
        jaw_config = _session_jaw_config(session_dir, config.jaw)
        session_rates.append(jaw_config.sampling_rate)
        session_channels.append(list(jaw_config.channels))
        positive_labels.append(jaw_config.positive_label)
        reference = neutral_reference(raw, labels, jaw_config)
        references.append(reference)
        exg = jaw_signal(raw, jaw_config)
        event_examples = jaw_clench_training_examples(exg, labels, jaw_config, reference)
        event_x.extend([row[0] for row in event_examples])
        event_y.extend([row[1] for row in event_examples])
    if len({round(rate, 6) for rate in session_rates}) > 1:
        raise ValueError(f"cannot train one jaw model from mixed sample rates: {session_rates}")
    if len({tuple(channels) for channels in session_channels}) > 1:
        raise ValueError(f"cannot train one jaw model from mixed jaw channels: {session_channels}")
    if len(set(positive_labels)) > 1:
        raise ValueError(f"cannot train one jaw model from mixed positive labels: {positive_labels}")
    if not any(event_y):
        raise ValueError(f"jaw model training requires at least one {positive_labels[0]} interval")
    if not any(label == 0 for label in event_y):
        raise ValueError("jaw model training requires at least one neutral interval")

    event_model = _fit_classifier(event_x, event_y)
    _dump_model(target_dir / "jaw_event.joblib", event_model)

    neutral = _median_reference(references)
    metadata = {
        "model_kind": "short_event_binary",
        "positive_label": positive_labels[0],
        "classifier_name": "random_forest",
        "profile_name": config.jaw.profile_name,
        "jaw_channels": session_channels[0],
        "sampling_rate_hz": session_rates[0],
        "feature_names": list(FEATURE_NAMES),
        "neutral_reference": neutral,
        "sessions": [str(path) for path in session_dirs],
        "event_examples": len(event_y),
        "positive_examples": int(sum(event_y)),
        "negative_examples": int(len(event_y) - sum(event_y)),
        "event_threshold": float(config.jaw.event_threshold),
        "validation_status": "unvalidated",
    }
    with (target_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return target_dir


class JawPredictor:
    def __init__(
        self,
        event_model: Any,
        neutral: dict[str, float],
        config: JawConfig,
        threshold: float | None = None,
        validated: bool = False,
        positive_label: str = "jaw_clench",
    ) -> None:
        self.event_model = event_model
        self.neutral = neutral
        self.config = config
        self.threshold = float(threshold if threshold is not None else config.event_threshold)
        self.validated = bool(validated)
        self.positive_label = positive_label

    @classmethod
    def load(cls, profile_dir: Path, config: JawConfig) -> "JawPredictor":
        metadata_path = profile_dir / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        stored_features = metadata.get("feature_names")
        if stored_features and list(stored_features) != list(FEATURE_NAMES):
            raise ValueError(
                "jaw model feature set is outdated; retrain the jaw model with the current app"
            )
        model_config = JawConfig(**config.__dict__)
        if metadata.get("jaw_channels"):
            model_config.channels = [int(channel) for channel in metadata["jaw_channels"]]
        if metadata.get("sampling_rate_hz"):
            model_config.sampling_rate = float(metadata["sampling_rate_hz"])
        positive_label = str(metadata.get("positive_label", model_config.positive_label))
        model_config.positive_label = positive_label
        return cls(
            event_model=_load_model(profile_dir / "jaw_event.joblib"),
            neutral=dict(metadata.get("neutral_reference") or {}),
            config=model_config,
            threshold=float(metadata.get("event_threshold", model_config.event_threshold)),
            validated=str(metadata.get("validation_status", "")).lower() == "passed",
            positive_label=positive_label,
        )

    def predict(self, raw: np.ndarray) -> JawPrediction:
        exg = jaw_signal(raw, self.config)
        probabilities: list[float] = []
        for seconds in self.config.window_seconds:
            event_samples = max(2, int(round(seconds * self.config.sampling_rate)))
            if exg.size >= 2:
                probabilities.append(
                    _predict_probability(
                        self.event_model,
                        jaw_feature_vector(exg[-event_samples:], self.config.sampling_rate, self.neutral),
                    )
                )
        if not probabilities:
            event_samples = max(2, int(round(self.config.short_clench_seconds * self.config.sampling_rate)))
            probabilities.append(
                _predict_probability(
                    self.event_model,
                    jaw_feature_vector(exg[-event_samples:], self.config.sampling_rate, self.neutral),
                )
            )
        event_conf = max(probabilities)
        state = self.positive_label if event_conf >= self.threshold else "relaxed"
        return JawPrediction(event_confidence=event_conf, state=state)


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


def jaw_clench_training_examples(
    exg: np.ndarray,
    labels: list[dict[str, Any]],
    config: JawConfig,
    reference: dict[str, float],
) -> list[tuple[np.ndarray, int]]:
    rows: list[tuple[np.ndarray, int]] = []
    intervals = interval_labels(labels)
    positive_intervals = [row for row in intervals if row.get("label") == config.positive_label]
    negative_intervals = [row for row in intervals if _is_negative_label(str(row.get("label")), config.positive_label)]
    for row in positive_intervals:
        start, end = int(row["start_sample"]), int(row["end_sample"])
        rows.append((jaw_feature_vector(exg[start:end], config.sampling_rate, reference), 1))
    window = max(2, int(round(config.short_clench_seconds * config.sampling_rate)))
    for row in negative_intervals:
        start, end = int(row["start_sample"]), int(row["end_sample"])
        if end - start < window:
            if end - start >= 2:
                rows.append((jaw_feature_vector(exg[start:end], config.sampling_rate, reference), 0))
            continue
        for neg_start in range(start, max(start, end - window + 1), window):
            neg_end = min(end, neg_start + window)
            if neg_end - neg_start >= max(2, window // 2):
                rows.append((jaw_feature_vector(exg[neg_start:neg_end], config.sampling_rate, reference), 0))
    return rows


def _is_negative_label(label: str, positive_label: str) -> bool:
    if label == positive_label:
        return False
    return (
        label == "neutral"
        or label.startswith("hard_negative")
        or label in {"jaw_clench", "eyebrow_raise"}
    )


def _session_jaw_config(session_dir: Path, fallback: JawConfig) -> JawConfig:
    raw: dict[str, Any] = {}
    for filename in ("summary.json", "metadata.json"):
        path = session_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw = dict((payload.get("config") or {}).get("jaw") or {})
        if raw:
            break
    if not raw:
        return JawConfig(**fallback.__dict__)
    merged = dict(fallback.__dict__)
    merged.update(raw)
    return JawConfig(**merged)


def _fit_classifier(features: list[np.ndarray], labels: list[int]) -> Any:
    return fit_classifier(features, labels, "random_forest")


def fit_classifier(features: list[np.ndarray], labels: list[int], classifier_name: str) -> Any:
    if not features:
        return ConstantProbability(0.0)
    y = np.asarray(labels, dtype=int)
    if len(set(y.tolist())) < 2:
        return ConstantProbability(float(np.mean(y)))
    x = np.vstack(features)
    if classifier_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=7,
        ).fit(x, y)
    if classifier_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=7,
        ).fit(x, y)
    if classifier_name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            random_state=7,
        ).fit(x, y)
    if classifier_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ).fit(x, y)
    if classifier_name == "emg_rms_threshold":
        return FeatureThresholdClassifier("emg_rms_delta_neutral").fit(x, y)
    raise ValueError(f"unknown jaw classifier {classifier_name!r}")


class ConstantProbability:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        return np.tile(np.array([[1.0 - self.probability, self.probability]], dtype=float), (n, 1))


class FeatureThresholdClassifier:
    def __init__(self, feature_name: str) -> None:
        self.feature_name = feature_name
        self.feature_index = FEATURE_NAMES.index(feature_name)
        self.threshold = 0.0
        self.scale = 1.0
        self.direction = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FeatureThresholdClassifier":
        values = np.asarray(x[:, self.feature_index], dtype=float)
        labels = np.asarray(y, dtype=int)
        positives = values[labels == 1]
        negatives = values[labels == 0]
        if positives.size == 0 or negatives.size == 0:
            self.threshold = float(np.median(values)) if values.size else 0.0
            self.scale = 1.0
            self.direction = 1.0
            return self
        pos_median = float(np.median(positives))
        neg_median = float(np.median(negatives))
        self.threshold = (pos_median + neg_median) * 0.5
        self.direction = 1.0 if pos_median >= neg_median else -1.0
        mad = float(np.median(np.abs(values - np.median(values))))
        self.scale = max(mad, float(np.std(values)), 1e-9)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x[:, self.feature_index], dtype=float)
        logits = self.direction * (values - self.threshold) / self.scale
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        return np.column_stack([1.0 - probabilities, probabilities])


def _predict_probability(model: Any, vector: np.ndarray) -> float:
    proba = model.predict_proba(vector.reshape(1, -1))
    return float(proba[0, 1])


def predict_probability(model: Any, vector: np.ndarray) -> float:
    return _predict_probability(model, vector)


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
