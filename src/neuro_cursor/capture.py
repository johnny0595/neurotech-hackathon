"""Capture raw Knight IMU/EXG data and summarize EXG features."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .brainflow_adapter import BoardConnectionError, KnightBrainFlowAdapter
from .config import JawConfig
from .config import DEFAULT_CONFIG_PATH, load_config
from .diagnostics import exg_channel_status, snapshot_payload
from .jaw_features import session_feature_payload, write_clips_npz, write_labels
from .rows import EXG_ROWS, NUM_ROWS, PACKAGE_ROW, ROW_LABELS, TIMESTAMP_ROW

DEFAULT_JAW_LABELS = {
    "neutral",
    "jaw_clench",
    "eyebrow_raise",
    "hard_negative",
    "jaw_hold",
    "jaw_release",
    "test",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config")
    parser.add_argument("--seconds", type=float, default=10.0, help="Capture duration")
    parser.add_argument("--label", default="test", help="Recording label, e.g. neutral or jaw_clench")
    parser.add_argument("--root", default="data/captures", help="Output directory")
    parser.add_argument("--from-csv", help="Import an 8-column EXG CSV instead of reading hardware")
    parser.add_argument("--sampling-rate", type=float, default=None, help="CSV sampling rate")
    parser.add_argument("--duration-seconds", type=float, default=30.0, help="CSV duration for rate inference")
    parser.add_argument("--positive-label", default=None, help="Positive event label for --event-times")
    parser.add_argument("--event-times", default="", help="Comma-separated positive event center times in seconds")
    parser.add_argument("--hold-intervals", default="", help="Legacy comma-separated hold intervals as start:end seconds")
    return parser


def sanitize_label(label: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in label.strip())
    return safe or "capture"


def exg_csv_matrix(data: np.ndarray) -> np.ndarray:
    return np.asarray(data[list(EXG_ROWS), :], dtype=float).T


def csv_to_raw(csv_data: np.ndarray, sampling_rate: float = 125.0) -> np.ndarray:
    values = np.asarray(csv_data, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != len(EXG_ROWS):
        raise ValueError(f"expected CSV shape (samples, 8), got {values.shape}")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")

    raw = np.zeros((NUM_ROWS, values.shape[0]), dtype=float)
    raw[PACKAGE_ROW] = np.arange(values.shape[0], dtype=float) % 256
    raw[list(EXG_ROWS)] = values.T
    raw[TIMESTAMP_ROW] = np.arange(values.shape[0], dtype=float) / sampling_rate
    return raw


def exg_features(data: np.ndarray) -> list[dict[str, float | int | str]]:
    features: list[dict[str, float | int | str]] = []
    for row in EXG_ROWS:
        values = np.asarray(data[row], dtype=float)
        if values.size == 0:
            centered = values
        else:
            centered = values - float(np.median(values))
        diff = np.diff(centered) if centered.size > 1 else np.zeros(0, dtype=float)
        features.append(
            {
                "channel": row,
                "row": row,
                "label": ROW_LABELS[row],
                "latest": float(values[-1]) if values.size else 0.0,
                "mean": float(np.mean(values)) if values.size else 0.0,
                "median": float(np.median(values)) if values.size else 0.0,
                "rms_centered": float(np.sqrt(np.mean(centered * centered))) if centered.size else 0.0,
                "peak_to_peak": float(np.max(values) - np.min(values)) if values.size else 0.0,
                "mean_abs_slope": float(np.mean(np.abs(diff))) if diff.size else 0.0,
                "nonzero_samples": int(np.count_nonzero(values)),
            }
        )
    return features


def counter_gap_count(data: np.ndarray) -> int:
    counters = data[PACKAGE_ROW].astype(int)
    if counters.size <= 1:
        return 0
    deltas = (np.diff(counters) % 256).astype(int)
    return int(np.sum(deltas != 1))


def timestamp_rate(data: np.ndarray) -> float:
    if data.shape[1] <= 1:
        return 0.0
    span = float(data[TIMESTAMP_ROW, -1] - data[TIMESTAMP_ROW, 0])
    if span <= 0:
        return 0.0
    return float((data.shape[1] - 1) / span)


def write_capture(
    data: np.ndarray,
    config_dict: dict[str, Any],
    label: str,
    root: Path,
    metadata: dict[str, Any],
    labels: list[dict[str, Any]] | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = sanitize_label(label)
    capture_dir = root / f"{timestamp}-{safe_label}"
    capture_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(capture_dir / "raw.npz", raw=data)
    np.savetxt(capture_dir / "exg.csv", exg_csv_matrix(data), delimiter=",", fmt="%.10f")
    label_rows = labels or [
        {
            "label": label,
            "type": "interval",
            "gesture": "jaw",
            "start_sample": 0,
            "end_sample": int(data.shape[1]),
            "active_exg_channels": config_dict["board"]["active_exg_channels"],
            "channel_map": config_dict.get("channel_map", {}),
        }
    ]
    write_labels(capture_dir / "labels.jsonl", label_rows)
    jaw_config = JawConfig(**(config_dict.get("jaw") or {"channels": [2]}))
    write_clips_npz(capture_dir / "clips.npz", data, label_rows, jaw_config)
    summary = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "samples": int(data.shape[1]),
        "rows": int(data.shape[0]),
        "counter_gaps": counter_gap_count(data),
        "timestamp_rate_hz": timestamp_rate(data),
        "config": config_dict,
        "metadata": metadata,
        "exg_features": exg_features(data),
        "jaw": session_feature_payload(data, label_rows, jaw_config),
        "snapshot": snapshot_payload(
            data,
            config_dict["board"]["active_exg_channels"],
            metadata=metadata,
        ),
    }
    with (capture_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return capture_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.positive_label:
        config.jaw.positive_label = args.positive_label
    if args.from_csv:
        try:
            csv_data = np.loadtxt(args.from_csv, delimiter=",")
            sampling_rate = _resolve_csv_sampling_rate(
                csv_data.shape[0],
                args.sampling_rate,
                args.duration_seconds,
            )
            config.jaw.sampling_rate = sampling_rate
            data = csv_to_raw(csv_data, sampling_rate=sampling_rate)
            labels = _manual_labels_from_args(args, config.jaw, data.shape[1])
        except Exception as exc:
            print(f"CSV import failed: {exc}", file=sys.stderr)
            return 2
        capture_dir = write_capture(
            data=data,
            config_dict=config.to_dict(),
            label=args.label,
            root=Path(args.root),
            metadata={
                "config_path": args.config,
                "source_csv": args.from_csv,
                "source_sampling_rate_hz": sampling_rate,
                "duration_seconds": args.duration_seconds,
                "wall_rate_hz": sampling_rate,
            },
            labels=labels,
        )
        print(f"Imported capture saved: {capture_dir}")
        print(f"Samples={data.shape[1]} source_rate={sampling_rate:.1f}Hz")
        _print_exg_feature_summary(data, config.board.active_exg_channels)
        return 0

    logs: list[str] = []

    def log(message: str) -> None:
        logs.append(message)
        print(f"[brainflow] {message}")

    adapter = KnightBrainFlowAdapter(config, log=log)
    batches: list[np.ndarray] = []
    started = 0.0
    ended = 0.0
    try:
        adapter.connect()
        print(f"Capturing {args.seconds:.1f}s for label={args.label!r}...")
        started = time.monotonic()
        while time.monotonic() - started < args.seconds:
            batch = adapter.drain()
            if batch.shape[1] > 0:
                batches.append(batch)
            time.sleep(0.02)
        ended = time.monotonic()
    except BoardConnectionError as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        return 2
    finally:
        adapter.close()

    if not batches:
        print("No data captured.", file=sys.stderr)
        return 3
    data = np.concatenate(batches, axis=1)
    if data.shape[0] != NUM_ROWS:
        print(f"Expected {NUM_ROWS} rows, got {data.shape[0]}", file=sys.stderr)
        return 4

    capture_dir = write_capture(
        data=data,
        config_dict=config.to_dict(),
        label=args.label,
        root=Path(args.root),
        metadata={
            "config_path": args.config,
            "board_descriptor": adapter.board_descr,
            "brainflow_version": adapter.brainflow_version,
            "logs": logs,
            "wall_rate_hz": data.shape[1] / max(ended - started, 0.001),
        },
        labels=_manual_labels_from_args(args, config.jaw, data.shape[1]),
    )

    print(f"Capture saved: {capture_dir}")
    print(
        f"Samples={data.shape[1]} wall_rate={data.shape[1] / max(ended - started, 0.001):.1f}Hz "
        f"timestamp_rate={timestamp_rate(data):.1f}Hz counter_gaps={counter_gap_count(data)}"
    )
    _print_exg_feature_summary(data, config.board.active_exg_channels)
    return 0


def _print_exg_feature_summary(data: np.ndarray, active_channels: list[int]) -> None:
    print("EXG feature summary:")
    statuses = {status.channel: status.status for status in exg_channel_status(data, active_channels)}
    for feature in exg_features(data):
        channel = int(feature["channel"])
        active_marker = "*" if channel in active_channels else " "
        status = statuses.get(channel, "inactive")
        print(
            f"  {active_marker}ch{channel} {status:>9} "
            f"p2p={feature['peak_to_peak']:>11.1f} "
            f"rms={feature['rms_centered']:>10.1f} "
            f"slope={feature['mean_abs_slope']:>9.1f} "
            f"nonzero={feature['nonzero_samples']}"
        )


def _resolve_csv_sampling_rate(
    samples: int,
    sampling_rate: float | None,
    duration_seconds: float | None,
) -> float:
    if sampling_rate is not None:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        return float(sampling_rate)
    if duration_seconds is None or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive when sampling_rate is omitted")
    return float(samples / duration_seconds)


def _manual_labels_from_args(args: argparse.Namespace, config: JawConfig, samples: int) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    positive_intervals: list[tuple[int, int]] = []
    positive_label = config.positive_label
    start_event, end_event = _event_names(positive_label)
    for index, center in enumerate(_parse_times(args.event_times), start=1):
        duration = max(1, int(round(config.short_clench_seconds * config.sampling_rate)))
        center_sample = int(round(center * config.sampling_rate))
        start = max(0, center_sample - duration // 2)
        end = min(samples, start + duration)
        if end > start:
            positive_intervals.append((start, end))
        labels.extend(_event_interval_labels(positive_label, start_event, end_event, start, end, index, config))
    for index, (start_s, end_s) in enumerate(_parse_intervals(args.hold_intervals), start=1):
        start = max(0, int(round(start_s * config.sampling_rate)))
        end = min(samples, int(round(end_s * config.sampling_rate)))
        if end > start:
            positive_intervals.append((start, end))
        labels.extend(_event_interval_labels("jaw_hold", "jaw_hold_start", "jaw_hold_end", start, end, index, config))
    return _neutral_complement_labels(positive_intervals, samples) + labels


def _neutral_complement_labels(positive_intervals: list[tuple[int, int]], samples: int) -> list[dict[str, Any]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(positive_intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    labels: list[dict[str, Any]] = []
    cursor = 0
    index = 0
    for start, end in merged:
        if start > cursor:
            index += 1
            labels.append(_neutral_label(cursor, start, index))
        cursor = max(cursor, end)
    if cursor < samples:
        index += 1
        labels.append(_neutral_label(cursor, samples, index))
    if not labels:
        labels.append(_neutral_label(0, samples, 1))
    return labels


def _neutral_label(start: int, end: int, index: int) -> dict[str, Any]:
    return {
        "label": "neutral",
        "type": "interval",
        "start_sample": start,
        "end_sample": end,
        "trial_type": "background",
        "trial_index": index,
    }


def _event_interval_labels(
    label: str,
    start_event: str,
    end_event: str,
    start: int,
    end: int,
    index: int,
    config: JawConfig,
) -> list[dict[str, Any]]:
    if end <= start:
        return []
    if label == "jaw_hold":
        trial_type = "jaw_hold"
    elif label == "jaw_clench":
        trial_type = "short_clench"
    else:
        trial_type = label
    return [
        {
            "label": start_event,
            "type": "event",
            "start_sample": start,
            "sample": start,
            "timestamp": start / config.sampling_rate,
            "trial_type": trial_type,
            "trial_index": index,
        },
        {
            "label": label,
            "type": "interval",
            "start_sample": start,
            "end_sample": end,
            "timestamp": start / config.sampling_rate,
            "trial_type": trial_type,
            "trial_index": index,
        },
        {
            "label": end_event,
            "type": "event",
            "start_sample": end,
            "sample": end,
            "timestamp": end / config.sampling_rate,
            "trial_type": trial_type,
            "trial_index": index,
        },
    ]


def _event_names(label: str) -> tuple[str, str]:
    if label == "jaw_clench":
        return "jaw_clench_start", "jaw_clench_end"
    if label == "eyebrow_raise":
        return "eyebrow_raise_start", "eyebrow_raise_end"
    return f"{label}_start", f"{label}_end"


def _parse_times(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_intervals(value: str) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    if not value.strip():
        return intervals
    for item in value.split(","):
        if not item.strip():
            continue
        start, end = item.split(":", 1)
        intervals.append((float(start), float(end)))
    return intervals


if __name__ == "__main__":
    raise SystemExit(main())
