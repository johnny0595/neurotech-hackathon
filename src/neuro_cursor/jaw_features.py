"""Channel-2 jaw feature extraction, clips, and labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import signal

from .config import JawConfig
from .rows import NUM_ROWS, validate_batch

FEATURE_NAMES = (
    "rms_centered",
    "peak_to_peak",
    "mean_abs_slope",
    "envelope_p95",
    "bandpower_20_45",
    "bandpower_45_60",
    "rms_delta_neutral",
    "p2p_delta_neutral",
)


@dataclass(frozen=True)
class JawClip:
    label: str
    start_sample: int
    end_sample: int
    event_start_sample: int
    event_end_sample: int


def read_labels(path: Path) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    if not path.exists():
        return labels
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                labels.append(json.loads(line))
    return labels


def write_labels(path: Path, labels: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label, sort_keys=True) + "\n")


def interval_labels(labels: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals = [
        label
        for label in labels
        if label.get("type") == "interval" and "start_sample" in label and "end_sample" in label
    ]
    return sorted(intervals, key=lambda row: int(row["start_sample"]))


def jaw_signal(raw: np.ndarray, config: JawConfig) -> np.ndarray:
    data = validate_batch(raw)
    channel = int(config.channels[0])
    return np.asarray(data[channel], dtype=float)


def jaw_feature_dict(
    values: np.ndarray,
    sampling_rate: float,
    neutral_reference: dict[str, float] | None = None,
) -> dict[str, float]:
    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        return {name: 0.0 for name in FEATURE_NAMES}
    centered = sample - float(np.median(sample))
    diff = np.diff(centered)
    envelope = _rms_envelope(centered, sampling_rate)
    power_20_45 = _bandpower(centered, sampling_rate, 20.0, 45.0)
    power_45_60 = _bandpower(centered, sampling_rate, 45.0, 60.0)
    rms = float(np.sqrt(np.mean(centered * centered)))
    p2p = float(np.max(sample) - np.min(sample))
    reference = neutral_reference or {}
    return {
        "rms_centered": rms,
        "peak_to_peak": p2p,
        "mean_abs_slope": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "envelope_p95": float(np.percentile(envelope, 95)) if envelope.size else 0.0,
        "bandpower_20_45": power_20_45,
        "bandpower_45_60": power_45_60,
        "rms_delta_neutral": rms - float(reference.get("rms_centered", 0.0)),
        "p2p_delta_neutral": p2p - float(reference.get("peak_to_peak", 0.0)),
    }


def jaw_feature_vector(
    values: np.ndarray,
    sampling_rate: float,
    neutral_reference: dict[str, float] | None = None,
) -> np.ndarray:
    features = jaw_feature_dict(values, sampling_rate, neutral_reference)
    return np.array([features[name] for name in FEATURE_NAMES], dtype=float)


def neutral_reference(raw: np.ndarray, labels: list[dict[str, Any]], config: JawConfig) -> dict[str, float]:
    signal_values = jaw_signal(raw, config)
    neutral_features: list[dict[str, float]] = []
    for label in interval_labels(labels):
        if label.get("label") != "neutral":
            continue
        start = max(0, int(label["start_sample"]))
        end = min(signal_values.size, int(label["end_sample"]))
        if end > start:
            neutral_features.append(jaw_feature_dict(signal_values[start:end], config.sampling_rate))
    if not neutral_features:
        return {name: 0.0 for name in FEATURE_NAMES}
    return {
        name: float(np.median([features[name] for features in neutral_features]))
        for name in FEATURE_NAMES
        if not name.endswith("_delta_neutral")
    }


def extract_labeled_clips(
    raw: np.ndarray,
    labels: list[dict[str, Any]],
    config: JawConfig,
) -> list[JawClip]:
    data = validate_batch(raw)
    pre = int(round(config.pre_event_seconds * config.sampling_rate))
    post = int(round(config.post_event_seconds * config.sampling_rate))
    clips: list[JawClip] = []
    for label in interval_labels(labels):
        if label.get("label") not in {"jaw_clench", "jaw_hold"}:
            continue
        event_start = int(label["start_sample"])
        event_end = int(label["end_sample"])
        start = max(0, event_start - pre)
        end = min(data.shape[1], event_end + post)
        if end <= start:
            continue
        clips.append(
            JawClip(
                label=str(label["label"]),
                start_sample=start,
                end_sample=end,
                event_start_sample=event_start,
                event_end_sample=event_end,
            )
        )
    return clips


def clip_feature_rows(
    raw: np.ndarray,
    labels: list[dict[str, Any]],
    config: JawConfig,
) -> list[dict[str, Any]]:
    data = validate_batch(raw)
    signal_values = jaw_signal(data, config)
    reference = neutral_reference(data, labels, config)
    rows: list[dict[str, Any]] = []
    for clip in extract_labeled_clips(data, labels, config):
        features = jaw_feature_dict(
            signal_values[clip.event_start_sample : clip.event_end_sample],
            config.sampling_rate,
            reference,
        )
        rows.append(
            {
                "label": clip.label,
                "start_sample": clip.start_sample,
                "end_sample": clip.end_sample,
                "event_start_sample": clip.event_start_sample,
                "event_end_sample": clip.event_end_sample,
                "features": features,
            }
        )
    return rows


def write_clips_npz(path: Path, raw: np.ndarray, labels: list[dict[str, Any]], config: JawConfig) -> None:
    data = validate_batch(raw)
    clips = extract_labeled_clips(data, labels, config)
    arrays: dict[str, np.ndarray] = {
        "labels": np.array([clip.label for clip in clips]),
        "start_samples": np.array([clip.start_sample for clip in clips], dtype=int),
        "end_samples": np.array([clip.end_sample for clip in clips], dtype=int),
        "event_start_samples": np.array([clip.event_start_sample for clip in clips], dtype=int),
        "event_end_samples": np.array([clip.event_end_sample for clip in clips], dtype=int),
    }
    for index, clip in enumerate(clips):
        arrays[f"clip_{index:03d}"] = data[:, clip.start_sample : clip.end_sample]
    np.savez_compressed(path, **arrays)


def session_feature_payload(
    raw: np.ndarray,
    labels: list[dict[str, Any]],
    config: JawConfig,
) -> dict[str, Any]:
    data = validate_batch(raw)
    reference = neutral_reference(data, labels, config)
    signal_values = jaw_signal(data, config)
    return {
        "jaw_channels": list(config.channels),
        "sampling_rate_hz": config.sampling_rate,
        "neutral_reference": reference,
        "whole_signal": jaw_feature_dict(signal_values, config.sampling_rate, reference),
        "clips": clip_feature_rows(data, labels, config),
    }


def _rms_envelope(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(1, int(round(0.12 * sampling_rate)))
    kernel = np.ones(window, dtype=float) / window
    return np.sqrt(signal.convolve(values * values, kernel, mode="same"))


def _bandpower(values: np.ndarray, sampling_rate: float, low: float, high: float) -> float:
    if values.size < 4 or sampling_rate <= 0:
        return 0.0
    nyquist = sampling_rate * 0.5
    if low >= nyquist:
        return 0.0
    high = min(high, nyquist - 1e-6)
    if high <= low:
        return 0.0
    freq, spectrum = signal.welch(
        values,
        fs=sampling_rate,
        nperseg=min(values.size, max(8, int(round(sampling_rate)))),
    )
    keep = (freq >= low) & (freq <= high)
    if not np.any(keep):
        return 0.0
    return float(np.trapezoid(spectrum[keep], freq[keep]))
