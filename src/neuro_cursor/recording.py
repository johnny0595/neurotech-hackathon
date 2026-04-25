"""Session recording for raw and orientation diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .orientation import OrientationSample


class SessionRecorder:
    def __init__(self, root: Path = Path("data/sessions")) -> None:
        self.root = root
        self.session_dir: Path | None = None
        self.metadata: dict[str, Any] = {}
        self._raw_batches: list[np.ndarray] = []
        self._orientation: list[OrientationSample] = []

    @property
    def is_recording(self) -> bool:
        return self.session_dir is not None

    def start(self, metadata: dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = self.root / timestamp
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.metadata["started_at"] = timestamp
        self._raw_batches = []
        self._orientation = []
        return self.session_dir

    def append(self, raw_batch: np.ndarray, orientation: list[OrientationSample]) -> None:
        if not self.is_recording:
            return
        if raw_batch.shape[1] > 0:
            self._raw_batches.append(raw_batch.copy())
        self._orientation.extend(orientation)

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

        orientation_rows = [asdict(sample) for sample in self._orientation]
        if orientation_rows:
            keys = list(orientation_rows[0].keys())
            arrays = {key: np.array([row[key] for row in orientation_rows], dtype=float) for key in keys}
        else:
            arrays = {"quaternion": np.zeros((0, 4), dtype=float)}
        np.savez_compressed(session_dir / "orientation.npz", **arrays)

        self.metadata["samples"] = int(raw.shape[1])
        self.metadata["orientation_samples"] = len(self._orientation)
        with (session_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2, sort_keys=True)

        self.session_dir = None
        return session_dir

