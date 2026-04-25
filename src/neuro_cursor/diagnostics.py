"""Stream diagnostics and snapshot capture helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .rows import EXG_ROWS, ROW_LABELS, row_stats, validate_batch


@dataclass(frozen=True)
class ExgChannelStatus:
    channel: int
    row: int
    label: str
    active: bool
    latest: float
    mean: float
    spread: float
    status: str


def exg_channel_status(
    data: np.ndarray,
    active_channels: list[int] | tuple[int, ...],
    window_samples: int = 250,
) -> list[ExgChannelStatus]:
    batch = validate_batch(data)
    active = set(int(channel) for channel in active_channels)
    window = batch[:, -window_samples:] if batch.shape[1] else batch
    statuses: list[ExgChannelStatus] = []

    for row in EXG_ROWS:
        values = window[row] if window.shape[1] else np.zeros(0, dtype=float)
        latest = float(values[-1]) if values.size else 0.0
        mean = float(np.mean(values)) if values.size else 0.0
        spread = float(np.max(values) - np.min(values)) if values.size else 0.0
        active_row = row in active
        if not active_row:
            status = "inactive"
        elif values.size == 0:
            status = "no_data"
        elif abs(latest) < 1e-9 and abs(mean) < 1e-9 and spread < 1e-9:
            status = "flat_zero"
        elif spread < 1.0:
            status = "flat"
        else:
            status = "live"
        statuses.append(
            ExgChannelStatus(
                channel=row,
                row=row,
                label=ROW_LABELS[row],
                active=active_row,
                latest=latest,
                mean=mean,
                spread=spread,
                status=status,
            )
        )
    return statuses


def active_exg_summary_text(
    data: np.ndarray,
    active_channels: list[int] | tuple[int, ...],
    window_samples: int = 250,
) -> str:
    statuses = [s for s in exg_channel_status(data, active_channels, window_samples) if s.active]
    if not statuses:
        return "no active EXG channels"
    return "  ".join(f"{status.channel}:{status.status}" for status in statuses)


def snapshot_payload(
    data: np.ndarray,
    active_channels: list[int] | tuple[int, ...],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    batch = validate_batch(data)
    stats = row_stats(batch)
    return {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "samples": int(batch.shape[1]),
        "metadata": metadata or {},
        "active_exg_channels": [int(channel) for channel in active_channels],
        "active_exg_status": [
            asdict(status)
            for status in exg_channel_status(batch, active_channels)
            if status.active
        ],
        "rows": [asdict(stat) for stat in stats],
        "latest_by_label": {stat.label: stat.latest for stat in stats},
    }


def write_snapshot(
    data: np.ndarray,
    active_channels: list[int] | tuple[int, ...],
    metadata: dict[str, Any] | None = None,
    root: Path = Path("data/snapshots"),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / f"{timestamp}-raw-snapshot.json"
    payload = snapshot_payload(data, active_channels, metadata)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path
