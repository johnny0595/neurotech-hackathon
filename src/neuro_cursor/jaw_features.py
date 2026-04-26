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
    "raw_rms_centered",
    "raw_peak_to_peak",
    "raw_mean_abs_slope",
    "emg_rms",
    "emg_peak_to_peak",
    "emg_mean_abs_slope",
    "emg_envelope_p95",
    "bandpower_20_45",
    "bandpower_45_60",
    "high_low_power_ratio",
    "raw_rms_delta_neutral",
    "raw_p2p_delta_neutral",
    "emg_rms_delta_neutral",
    "emg_p2p_delta_neutral",
    "emg_rms_recent_delta",
    "emg_p2p_recent_delta",
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
    filtered = _emg_filtered(centered, sampling_rate)
    diff = np.diff(centered)
    filtered_diff = np.diff(filtered)
    envelope = _rms_envelope(filtered, sampling_rate)
    power_20_45 = _bandpower(filtered, sampling_rate, 20.0, 45.0)
    power_45_60 = _bandpower(filtered, sampling_rate, 45.0, 60.0)
    raw_rms = float(np.sqrt(np.mean(centered * centered)))
    emg_rms = float(np.sqrt(np.mean(filtered * filtered)))
    raw_p2p = float(np.max(sample) - np.min(sample))
    emg_p2p = float(np.max(filtered) - np.min(filtered))
    emg_rms_recent_delta, emg_p2p_recent_delta = _recent_feature_deltas(filtered)
    reference = neutral_reference or {}
    return {
        "raw_rms_centered": raw_rms,
        "raw_peak_to_peak": raw_p2p,
        "raw_mean_abs_slope": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "emg_rms": emg_rms,
        "emg_peak_to_peak": emg_p2p,
        "emg_mean_abs_slope": float(np.mean(np.abs(filtered_diff))) if filtered_diff.size else 0.0,
        "emg_envelope_p95": float(np.percentile(envelope, 95)) if envelope.size else 0.0,
        "bandpower_20_45": power_20_45,
        "bandpower_45_60": power_45_60,
        "high_low_power_ratio": power_45_60 / max(power_20_45, 1e-9),
        "raw_rms_delta_neutral": raw_rms - float(reference.get("raw_rms_centered", 0.0)),
        "raw_p2p_delta_neutral": raw_p2p - float(reference.get("raw_peak_to_peak", 0.0)),
        "emg_rms_delta_neutral": emg_rms - float(reference.get("emg_rms", 0.0)),
        "emg_p2p_delta_neutral": emg_p2p - float(reference.get("emg_peak_to_peak", 0.0)),
        "emg_rms_recent_delta": emg_rms_recent_delta,
        "emg_p2p_recent_delta": emg_p2p_recent_delta,
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
    clip_labels = {"jaw_clench", "eyebrow_raise"}
    for label in interval_labels(labels):
        label_name = str(label.get("label", ""))
        if label_name not in clip_labels and not label_name.startswith("hard_negative"):
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


def _emg_filtered(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    sample = np.asarray(values, dtype=float)
    if sample.size == 0 or sampling_rate <= 0:
        return sample
    nyquist = sampling_rate * 0.5
    low = 5.0
    high = min(60.0, nyquist - 1e-6)
    if high <= low or sample.size < 24:
        return sample
    b, a = signal.butter(3, [low / nyquist, high / nyquist], btype="bandpass")
    try:
        return signal.filtfilt(b, a, sample, method="gust")
    except ValueError:
        return sample


def _recent_feature_deltas(values: np.ndarray) -> tuple[float, float]:
    sample = np.asarray(values, dtype=float)
    if sample.size < 4:
        return 0.0, 0.0
    midpoint = sample.size // 2
    early = sample[:midpoint]
    recent = sample[midpoint:]
    early_rms = float(np.sqrt(np.mean(early * early))) if early.size else 0.0
    recent_rms = float(np.sqrt(np.mean(recent * recent))) if recent.size else 0.0
    early_p2p = float(np.max(early) - np.min(early)) if early.size else 0.0
    recent_p2p = float(np.max(recent) - np.min(recent)) if recent.size else 0.0
    return recent_rms - early_rms, recent_p2p - early_p2p


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
