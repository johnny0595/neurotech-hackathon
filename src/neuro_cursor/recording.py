"""Session recording for raw and orientation diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .capture import counter_gap_count, exg_csv_matrix, exg_features, timestamp_rate
from .config import JawConfig
from .jaw_features import session_feature_payload, write_clips_npz, write_labels
from .orientation import OrientationSample


class SessionRecorder:
    def __init__(self, root: Path = Path("data/sessions")) -> None:
        self.root = root
        self.session_dir: Path | None = None
        self.metadata: dict[str, Any] = {}
        self._raw_batches: list[np.ndarray] = []
        self._orientation: list[OrientationSample] = []
        self._labels: list[dict[str, Any]] = []

    @property
    def is_recording(self) -> bool:
        return self.session_dir is not None

    @property
    def sample_count(self) -> int:
        return int(sum(batch.shape[1] for batch in self._raw_batches))

    def start(self, metadata: dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = self.root / timestamp
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.metadata["started_at"] = timestamp
        self._raw_batches = []
        self._orientation = []
        self._labels = []
        return self.session_dir

    def append(self, raw_batch: np.ndarray, orientation: list[OrientationSample]) -> None:
        if not self.is_recording:
            return
        if raw_batch.shape[1] > 0:
            self._raw_batches.append(raw_batch.copy())
        self._orientation.extend(orientation)

    def add_labels(self, labels: list[dict[str, Any]]) -> None:
        if not self.is_recording:
            return
        self._labels.extend(labels)

    def stop(self) -> Path | None:
        if self.session_dir is None:
            return None
        session_dir = self.session_dir
        raw = (
            np.concatenate(self._raw_batches, axis=1)
            if self._raw_batches
            else np.zeros((22, 0), dtype=float)
        )
        np.savez_compressed(session_dir / "raw.npz", raw=raw)
        np.savetxt(session_dir / "exg.csv", exg_csv_matrix(raw), delimiter=",", fmt="%.10f")

        orientation_rows = [asdict(sample) for sample in self._orientation]
        if orientation_rows:
            keys = list(orientation_rows[0].keys())
            arrays = {key: np.array([row[key] for row in orientation_rows], dtype=float) for key in keys}
        else:
            arrays = {"quaternion": np.zeros((0, 4), dtype=float)}
        np.savez_compressed(session_dir / "orientation.npz", **arrays)

        self.metadata["samples"] = int(raw.shape[1])
        self.metadata["orientation_samples"] = len(self._orientation)
        label = str(self.metadata.get("label", "unlabeled"))
        labels = self._labels or [
            {
                "label": label,
                "type": "interval",
                "gesture": self.metadata.get("gesture", "jaw"),
                "start_sample": 0,
                "end_sample": int(raw.shape[1]),
                "active_exg_channels": self.metadata.get("active_exg_channels", []),
                "channel_map": self.metadata.get("config", {}).get("channel_map", {}),
            }
        ]
        write_labels(session_dir / "labels.jsonl", labels)
        jaw_config = _jaw_config_from_metadata(self.metadata)
        write_clips_npz(session_dir / "clips.npz", raw, labels, jaw_config)
        features = {
            "label": label,
            "samples": int(raw.shape[1]),
            "counter_gaps": counter_gap_count(raw),
            "timestamp_rate_hz": timestamp_rate(raw),
            "exg_features": exg_features(raw),
            "jaw": session_feature_payload(raw, labels, jaw_config),
        }
        with (session_dir / "features.json").open("w", encoding="utf-8") as handle:
            json.dump(features, handle, indent=2, sort_keys=True)
        with (session_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2, sort_keys=True)

        self.session_dir = None
        return session_dir


def _jaw_config_from_metadata(metadata: dict[str, Any]) -> JawConfig:
    raw = dict((metadata.get("config") or {}).get("jaw") or {})
    if not raw:
        raw = {
            "channels": metadata.get("active_exg_channels") or [2],
        }
    return JawConfig(**raw)
