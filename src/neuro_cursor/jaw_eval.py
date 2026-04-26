"""Session-holdout evaluation for the jaw clench classifier."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .capture import timestamp_rate
from .config import AppConfig, JawConfig, load_config
from .jaw_features import (
    FEATURE_NAMES,
    interval_labels,
    jaw_feature_vector,
    jaw_signal,
    neutral_reference,
)
from .jaw_model import (
    CLASSIFIER_NAMES,
    _dump_model,
    _median_reference,
    _session_jaw_config,
    fit_classifier,
    load_session,
    predict_probability,
)


DEFAULT_MAX_FALSE_POSITIVES_PER_MINUTE = 0.5
DEFAULT_MIN_PRECISION = 0.70


@dataclass(frozen=True)
class JawWindow:
    session: str
    start_sample: int
    end_sample: int
    duration_seconds: float
    true_label: int
    features: np.ndarray


def evaluate_from_args(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and validate a channel-2 short-event model with session holdout")
    parser.add_argument("--config", default="config/neuro_cursor.yaml")
    parser.add_argument("--train-sessions", nargs="+", required=True)
    parser.add_argument("--validation-sessions", nargs="+", required=True)
    parser.add_argument("--profile", default=None, help="Override profile name")
    parser.add_argument("--profile-dir", default=None, help="Output model directory; defaults to models/<profile>")
    parser.add_argument("--positive-label", default=None, help="Positive event label, e.g. jaw_clench or eyebrow_raise")
    parser.add_argument(
        "--model-candidates",
        nargs="+",
        default=list(CLASSIFIER_NAMES),
        choices=list(CLASSIFIER_NAMES),
        help="Classifier candidates for the model bakeoff",
    )
    parser.add_argument("--report-root", default="reports/jaw_eval")
    parser.add_argument(
        "--max-false-positives-per-minute",
        type=float,
        default=DEFAULT_MAX_FALSE_POSITIVES_PER_MINUTE,
    )
    parser.add_argument("--min-precision", type=float, default=DEFAULT_MIN_PRECISION)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.profile:
        config.jaw.profile_name = args.profile
    if args.positive_label:
        config.jaw.positive_label = args.positive_label
    report = train_and_evaluate_profile(
        train_sessions=[Path(path) for path in args.train_sessions],
        validation_sessions=[Path(path) for path in args.validation_sessions],
        config=config,
        report_root=Path(args.report_root),
        profile_dir=Path(args.profile_dir) if args.profile_dir else None,
        model_candidates=list(args.model_candidates),
        max_false_positives_per_minute=args.max_false_positives_per_minute,
        min_precision=args.min_precision,
    )
    print(f"Jaw evaluation report: {report['report_dir']}")
    print(
        f"status={report['validation_status']} model={report['selected_model']} "
        f"label={report['metadata']['positive_label']} threshold={report['selected_threshold']:.2f} "
        f"precision={report['metrics']['precision']:.3f} recall={report['metrics']['recall']:.3f} "
        f"false_pos/min={report['metrics']['false_positives_per_minute']:.2f}"
    )
    return 0 if report["validation_status"] == "passed" else 1


def train_and_evaluate_profile(
    train_sessions: list[Path],
    validation_sessions: list[Path],
    config: AppConfig,
    report_root: Path = Path("reports/jaw_eval"),
    profile_dir: Path | None = None,
    model_candidates: list[str] | None = None,
    max_false_positives_per_minute: float = DEFAULT_MAX_FALSE_POSITIVES_PER_MINUTE,
    min_precision: float = DEFAULT_MIN_PRECISION,
) -> dict[str, Any]:
    if not train_sessions:
        raise ValueError("at least one training session is required")
    if not validation_sessions:
        raise ValueError("at least one validation session is required")
    _ensure_disjoint_sessions(train_sessions, validation_sessions)

    candidates = model_candidates or list(CLASSIFIER_NAMES)
    train_payloads = [_load_checked_session(path, config.jaw, require_positive=False) for path in train_sessions]
    validation_payloads = [_load_checked_session(path, config.jaw, require_positive=True) for path in validation_sessions]
    _ensure_compatible_configs(train_payloads + validation_payloads)

    neutral = _median_reference([payload["neutral_reference"] for payload in train_payloads])
    train_windows = _windows_from_payloads(train_payloads, neutral)
    validation_windows = _windows_from_payloads(validation_payloads, neutral)
    train_x = [window.features for window in train_windows]
    train_y = [window.true_label for window in train_windows]
    if len(set(train_y)) < 2:
        raise ValueError(f"training set must contain both relaxed and {config.jaw.positive_label} windows")
    if len({window.true_label for window in validation_windows}) < 2:
        raise ValueError(f"validation set must contain both relaxed and {config.jaw.positive_label} windows")

    selected = _select_candidate_model(
        train_x,
        train_y,
        validation_windows,
        candidates,
        max_false_positives_per_minute,
        min_precision,
    )
    model = selected["model"]
    scored = selected["scored"]
    selected_threshold = selected["selected_threshold"]
    sweep = selected["threshold_sweep"]
    metrics = classification_metrics(scored, selected_threshold)
    session_metrics = {
        session: classification_metrics(
            [row for row in scored if row["session"] == session],
            selected_threshold,
        )
        for session in sorted({row["session"] for row in scored})
    }
    feature_summary = feature_separation(validation_windows)
    validation_status = "passed" if _passes_quality(metrics, feature_summary, max_false_positives_per_minute, min_precision) else "failed"

    target_dir = profile_dir or Path("models") / config.jaw.profile_name
    target_dir.mkdir(parents=True, exist_ok=True)
    _dump_model(target_dir / "jaw_event.joblib", model)

    report_dir = report_root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    report_dir.mkdir(parents=True, exist_ok=False)
    _write_timeline(report_dir / "prediction_timeline.csv", scored, selected_threshold)
    _write_threshold_sweep(report_dir / "threshold_sweep.csv", sweep)

    metadata = {
        "model_kind": "short_event_binary",
        "positive_label": config.jaw.positive_label,
        "classifier_name": selected["classifier_name"],
        "model_candidates": candidates,
        "profile_name": config.jaw.profile_name,
        "jaw_channels": list(train_payloads[0]["jaw_config"].channels),
        "sampling_rate_hz": float(train_payloads[0]["jaw_config"].sampling_rate),
        "feature_names": list(FEATURE_NAMES),
        "neutral_reference": neutral,
        "train_sessions": [str(path) for path in train_sessions],
        "validation_sessions": [str(path) for path in validation_sessions],
        "event_threshold": float(selected_threshold),
        "validation_status": validation_status,
        "validation_report": str(report_dir),
        "event_examples": len(train_y),
        "positive_examples": int(sum(train_y)),
        "negative_examples": int(len(train_y) - sum(train_y)),
    }
    with (target_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_dir": str(report_dir),
        "profile_dir": str(target_dir),
        "validation_status": validation_status,
        "selected_threshold": float(selected_threshold),
        "selected_model": selected["classifier_name"],
        "metrics": metrics,
        "model_bakeoff": selected["bakeoff"],
        "session_metrics": session_metrics,
        "feature_separation": feature_summary,
        "threshold_sweep": sweep,
        "quality": {
            "max_false_positives_per_minute": max_false_positives_per_minute,
            "min_precision": min_precision,
        },
        "data_quality": {
            "train": [payload["quality"] for payload in train_payloads],
            "validation": [payload["quality"] for payload in validation_payloads],
        },
        "metadata": metadata,
    }
    with (report_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def discover_usable_sessions(root: Path, config: AppConfig) -> tuple[list[Path], list[dict[str, Any]]]:
    usable: list[Path] = []
    skipped: list[dict[str, Any]] = []
    for session_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        try:
            raw, labels = load_session(session_dir)
            jaw_config = _session_jaw_config(session_dir, config.jaw)
            quality = session_quality(session_dir, raw, labels, jaw_config, require_positive=True)
        except Exception as exc:
            skipped.append({"session": str(session_dir), "errors": [str(exc)], "warnings": []})
            continue
        if quality["errors"]:
            skipped.append(quality)
        else:
            usable.append(session_dir)
    return usable, skipped


def _select_candidate_model(
    train_x: list[np.ndarray],
    train_y: list[int],
    validation_windows: list[JawWindow],
    candidates: list[str],
    max_false_positives_per_minute: float,
    min_precision: float,
) -> dict[str, Any]:
    bakeoff: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for classifier_name in candidates:
        model = fit_classifier(train_x, train_y, classifier_name)
        scored = _score_windows(model, validation_windows)
        threshold, sweep = threshold_sweep(
            scored,
            max_false_positives_per_minute=max_false_positives_per_minute,
            min_precision=min_precision,
        )
        metrics = classification_metrics(scored, threshold)
        row = {
            "classifier_name": classifier_name,
            "selected_threshold": float(threshold),
            "metrics": metrics,
        }
        bakeoff.append(row)
        candidate = {
            "classifier_name": classifier_name,
            "model": model,
            "scored": scored,
            "selected_threshold": float(threshold),
            "threshold_sweep": sweep,
            "metrics": metrics,
            "bakeoff": bakeoff,
        }
        if selected is None or _candidate_sort_key(candidate) < _candidate_sort_key(selected):
            selected = candidate
    if selected is None:
        raise ValueError("at least one model candidate is required")
    selected["bakeoff"] = bakeoff
    return selected


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = candidate["metrics"]
    return (
        float(metrics["false_positive"]),
        float(metrics["false_positives_per_minute"]),
        -float(metrics["recall"]),
        -float(metrics["f1"]),
        -float(metrics["precision"]),
    )


def classification_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    negative_seconds = 0.0
    for row in rows:
        truth = int(row["true_label"])
        pred = 1 if float(row["probability"]) >= threshold else 0
        if truth == 0:
            negative_seconds += float(row["duration_seconds"])
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        elif truth == 1 and pred == 0:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positives_per_minute": float(fp / max(negative_seconds / 60.0, 1e-12)),
        "negative_minutes": float(negative_seconds / 60.0),
    }


def threshold_sweep(
    rows: list[dict[str, Any]],
    max_false_positives_per_minute: float = DEFAULT_MAX_FALSE_POSITIVES_PER_MINUTE,
    min_precision: float = DEFAULT_MIN_PRECISION,
) -> tuple[float, list[dict[str, Any]]]:
    thresholds = [round(value, 2) for value in np.linspace(0.05, 0.95, 91)]
    sweep = [classification_metrics(rows, threshold) for threshold in thresholds]
    candidates = [
        row
        for row in sweep
        if row["false_positives_per_minute"] <= max_false_positives_per_minute
        and row["precision"] >= min_precision
    ]
    if candidates:
        best = min(candidates, key=lambda row: row["threshold"])
    else:
        best = min(
            sweep,
            key=lambda row: (
                row["false_positives_per_minute"],
                -row["precision"],
                -row["recall"],
                row["threshold"],
            ),
        )
    return float(best["threshold"]), sweep


def feature_separation(windows: list[JawWindow]) -> dict[str, Any]:
    positives = [window.features for window in windows if window.true_label == 1]
    negatives = [window.features for window in windows if window.true_label == 0]
    if not positives or not negatives:
        return {"usable": False, "features": {}}
    pos = np.vstack(positives)
    neg = np.vstack(negatives)
    rows: dict[str, Any] = {}
    usable = False
    for index, name in enumerate(FEATURE_NAMES):
        pos_median = float(np.median(pos[:, index]))
        neg_median = float(np.median(neg[:, index]))
        neg_mad = float(np.median(np.abs(neg[:, index] - neg_median)))
        ratio = abs(pos_median - neg_median) / max(neg_mad, 1e-9)
        rows[name] = {
            "positive_median": pos_median,
            "negative_median": neg_median,
            "median_delta": pos_median - neg_median,
            "robust_separation": float(ratio),
        }
        usable = usable or ratio >= 1.0
    return {"usable": usable, "features": rows}


def _load_checked_session(
    session_dir: Path,
    fallback: JawConfig,
    require_positive: bool,
) -> dict[str, Any]:
    raw, labels = load_session(session_dir)
    jaw_config = _session_jaw_config(session_dir, fallback)
    quality = session_quality(session_dir, raw, labels, jaw_config, require_positive=require_positive)
    if quality["errors"]:
        raise ValueError(f"{session_dir} is not usable for jaw validation: {quality['errors']}")
    return {
        "session_dir": session_dir,
        "raw": raw,
        "labels": labels,
        "jaw_config": jaw_config,
        "neutral_reference": neutral_reference(raw, labels, jaw_config),
        "quality": quality,
    }


def session_quality(
    session_dir: Path,
    raw: np.ndarray,
    labels: list[dict[str, Any]],
    config: JawConfig,
    require_positive: bool = True,
) -> dict[str, Any]:
    intervals = interval_labels(labels)
    neutral_count = sum(1 for row in intervals if row.get("label") == "neutral")
    positive_count = sum(1 for row in intervals if row.get("label") == config.positive_label)
    hard_negative_count = sum(
        1 for row in intervals if str(row.get("label", "")).startswith("hard_negative")
    )
    guided = _looks_guided(session_dir, labels)
    errors: list[str] = []
    warnings: list[str] = []
    if neutral_count <= 0:
        errors.append("missing neutral intervals")
    if require_positive and positive_count <= 0:
        errors.append(f"missing {config.positive_label} intervals")
    if require_positive and guided and positive_count < config.short_clench_reps:
        errors.append(
            f"incomplete guided {config.positive_label} intervals: "
            f"expected {config.short_clench_reps}, got {positive_count}"
        )
    values = jaw_signal(raw, config)
    if values.size == 0 or np.count_nonzero(values) == 0:
        errors.append("channel 2 is empty")
    if values.size > 0 and float(np.ptp(values)) <= 1e-9:
        errors.append("channel 2 is flat")
    rate = timestamp_rate(raw)
    if rate > 0:
        expected = float(config.sampling_rate)
        if abs(rate - expected) / max(expected, 1e-9) > 0.25:
            warnings.append(f"timestamp rate {rate:.1f}Hz differs from configured {expected:.1f}Hz")
    return {
        "session": str(session_dir),
        "samples": int(raw.shape[1]),
        "neutral_intervals": neutral_count,
        "positive_label": config.positive_label,
        "positive_intervals": positive_count,
        "hard_negative_intervals": hard_negative_count,
        "jaw_clench_intervals": sum(1 for row in intervals if row.get("label") == "jaw_clench"),
        "eyebrow_raise_intervals": sum(1 for row in intervals if row.get("label") == "eyebrow_raise"),
        "guided": guided,
        "timestamp_rate_hz": rate,
        "errors": errors,
        "warnings": warnings,
    }


def _windows_from_payloads(payloads: list[dict[str, Any]], reference: dict[str, float]) -> list[JawWindow]:
    windows: list[JawWindow] = []
    for payload in payloads:
        windows.extend(_session_windows(payload["session_dir"], payload["raw"], payload["labels"], payload["jaw_config"], reference))
    return windows


def _session_windows(
    session_dir: Path,
    raw: np.ndarray,
    labels: list[dict[str, Any]],
    config: JawConfig,
    reference: dict[str, float],
) -> list[JawWindow]:
    exg = jaw_signal(raw, config)
    rows: list[JawWindow] = []
    for label in interval_labels(labels):
        start = max(0, int(label["start_sample"]))
        end = min(exg.size, int(label["end_sample"]))
        if end <= start:
            continue
        label_name = str(label.get("label", ""))
        if label_name == config.positive_label:
            for win_start, win_end in _positive_window_ranges(start, end, exg.size, config):
                rows.append(_make_window(session_dir, exg, win_start, win_end, 1, config, reference))
        elif _is_negative_label(label_name, config.positive_label):
            for win_start, win_end in _negative_window_ranges(start, end, config):
                rows.append(_make_window(session_dir, exg, win_start, win_end, 0, config, reference))
    return rows


def _positive_window_ranges(
    start: int,
    end: int,
    sample_count: int,
    config: JawConfig,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    event_len = max(1, end - start)
    for seconds in config.window_seconds:
        window = max(2, int(round(seconds * config.sampling_rate)))
        candidates = {
            (max(0, end - window), end),
            (max(0, start - max(0, window - event_len) // 2), min(sample_count, start - max(0, window - event_len) // 2 + window)),
            (start, min(sample_count, start + window)),
        }
        for win_start, win_end in candidates:
            if win_end - win_start >= max(2, window // 2):
                ranges.append((win_start, win_end))
    return sorted(set(ranges))


def _negative_window_ranges(start: int, end: int, config: JawConfig) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    step = max(1, int(round(config.window_step_seconds * config.sampling_rate)))
    for seconds in config.window_seconds:
        window = max(2, int(round(seconds * config.sampling_rate)))
        if end - start < window:
            if end - start >= max(2, window // 2):
                ranges.append((start, end))
            continue
        for win_start in range(start, end - window + 1, step):
            ranges.append((win_start, win_start + window))
    return sorted(set(ranges))


def _is_negative_label(label: str, positive_label: str) -> bool:
    if label == positive_label:
        return False
    return (
        label == "neutral"
        or label.startswith("hard_negative")
        or label in {"jaw_clench", "eyebrow_raise"}
    )


def _make_window(
    session_dir: Path,
    exg: np.ndarray,
    start: int,
    end: int,
    true_label: int,
    config: JawConfig,
    reference: dict[str, float],
) -> JawWindow:
    return JawWindow(
        session=str(session_dir),
        start_sample=start,
        end_sample=end,
        duration_seconds=float(end - start) / float(config.sampling_rate),
        true_label=true_label,
        features=jaw_feature_vector(exg[start:end], config.sampling_rate, reference),
    )


def _score_windows(model: Any, windows: list[JawWindow]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in windows:
        probability = predict_probability(model, window.features)
        rows.append(
            {
                "session": window.session,
                "start_sample": window.start_sample,
                "end_sample": window.end_sample,
                "duration_seconds": window.duration_seconds,
                "true_label": window.true_label,
                "probability": probability,
            }
        )
    return rows


def _write_timeline(path: Path, rows: list[dict[str, Any]], threshold: float) -> None:
    fieldnames = ["session", "start_sample", "end_sample", "true_label", "probability", "predicted_label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "session": row["session"],
                    "start_sample": row["start_sample"],
                    "end_sample": row["end_sample"],
                    "true_label": row["true_label"],
                    "probability": f"{float(row['probability']):.8f}",
                    "predicted_label": int(float(row["probability"]) >= threshold),
                }
            )


def _write_threshold_sweep(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "threshold",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "precision",
        "recall",
        "f1",
        "false_positives_per_minute",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _passes_quality(
    metrics: dict[str, Any],
    feature_summary: dict[str, Any],
    max_false_positives_per_minute: float,
    min_precision: float,
) -> bool:
    return (
        metrics["false_positives_per_minute"] <= max_false_positives_per_minute
        and metrics["precision"] >= min_precision
        and bool(feature_summary.get("usable"))
    )


def _ensure_disjoint_sessions(train_sessions: list[Path], validation_sessions: list[Path]) -> None:
    train = {path.resolve() for path in train_sessions}
    validation = {path.resolve() for path in validation_sessions}
    overlap = train & validation
    if overlap:
        raise ValueError(f"train and validation sessions must be disjoint: {sorted(str(path) for path in overlap)}")


def _ensure_compatible_configs(payloads: list[dict[str, Any]]) -> None:
    rates = {round(float(payload["jaw_config"].sampling_rate), 6) for payload in payloads}
    channels = {tuple(payload["jaw_config"].channels) for payload in payloads}
    if len(rates) > 1:
        raise ValueError(f"cannot evaluate mixed sampling rates: {sorted(rates)}")
    if len(channels) > 1:
        raise ValueError(f"cannot evaluate mixed jaw channels: {sorted(channels)}")


def _looks_guided(session_dir: Path, labels: list[dict[str, Any]]) -> bool:
    guided_trial_types = {"short_clench", "jaw_clench", "eyebrow_raise", "jaw_hold"}
    if any(
        label.get("trial_type") in guided_trial_types
        or str(label.get("trial_type", "")).startswith("hard_negative")
        for label in labels
    ):
        return True
    for filename in ("summary.json", "metadata.json"):
        path = session_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = payload.get("metadata") or payload
        label = str(metadata.get("label", ""))
        if label == "jaw_guided_calibration" or label.endswith("_guided_calibration") or metadata.get("prompt_schedule"):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    return evaluate_from_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
