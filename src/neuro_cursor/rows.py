"""Explicit row map for BrainFlow's NeuroPawn Knight IMU board."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

BOARD_ID_NAME = "NEUROPAWN_KNIGHT_BOARD_IMU"
BOARD_ID_VALUE = 66
SAMPLING_RATE_HZ = 125
NUM_ROWS = 22

PACKAGE_ROW = 0
EXG_ROWS = tuple(range(1, 9))
LOFF_STATP_ROW = 9
LOFF_STATN_ROW = 10
ACCEL_ROWS = (11, 12, 13)
GYRO_ROWS = (14, 15, 16)
MAG_ROWS = (17, 18, 19)
TIMESTAMP_ROW = 20
MARKER_ROW = 21

ROW_LABELS = {
    PACKAGE_ROW: "package_counter",
    **{row: f"exg_{row}" for row in EXG_ROWS},
    LOFF_STATP_ROW: "loff_statp",
    LOFF_STATN_ROW: "loff_statn",
    ACCEL_ROWS[0]: "accel_x",
    ACCEL_ROWS[1]: "accel_y",
    ACCEL_ROWS[2]: "accel_z",
    GYRO_ROWS[0]: "gyro_x",
    GYRO_ROWS[1]: "gyro_y",
    GYRO_ROWS[2]: "gyro_z",
    MAG_ROWS[0]: "mag_x",
    MAG_ROWS[1]: "mag_y",
    MAG_ROWS[2]: "mag_z",
    TIMESTAMP_ROW: "timestamp",
    MARKER_ROW: "marker",
}


@dataclass(frozen=True)
class BrainFlowSample:
    """Single 22-row Knight IMU package."""

    raw: np.ndarray
    timestamp: float


@dataclass(frozen=True)
class RowStats:
    row: int
    label: str
    latest: float
    mean: float
    spread: float


def validate_batch(data: np.ndarray) -> np.ndarray:
    """Validate and normalize a BrainFlow batch to shape ``(22, n)``."""

    batch = np.asarray(data, dtype=float)
    if batch.ndim != 2:
        raise ValueError(f"expected 2D BrainFlow batch, got shape {batch.shape}")
    if batch.shape[0] != NUM_ROWS:
        raise ValueError(f"expected {NUM_ROWS} rows for Knight IMU, got {batch.shape[0]}")
    return batch


def latest_sample(data: np.ndarray) -> BrainFlowSample:
    batch = validate_batch(data)
    if batch.shape[1] == 0:
        raise ValueError("cannot get latest sample from an empty batch")
    raw = batch[:, -1].copy()
    return BrainFlowSample(raw=raw, timestamp=float(raw[TIMESTAMP_ROW]))


def vector(raw: np.ndarray, rows: Iterable[int]) -> np.ndarray:
    sample = np.asarray(raw, dtype=float)
    return np.array([sample[row] for row in rows], dtype=float)


def accel(raw: np.ndarray) -> np.ndarray:
    return vector(raw, ACCEL_ROWS)


def gyro(raw: np.ndarray) -> np.ndarray:
    return vector(raw, GYRO_ROWS)


def mag(raw: np.ndarray) -> np.ndarray:
    return vector(raw, MAG_ROWS)


def loff_bits(value: float) -> list[int]:
    """Return set bit positions from a LOFF status byte."""

    byte = int(value) & 0xFF
    return [bit for bit in range(8) if byte & (1 << bit)]


def row_stats(data: np.ndarray) -> list[RowStats]:
    batch = validate_batch(data)
    if batch.shape[1] == 0:
        return [
            RowStats(row=row, label=ROW_LABELS[row], latest=0.0, mean=0.0, spread=0.0)
            for row in range(NUM_ROWS)
        ]
    stats: list[RowStats] = []
    for row in range(NUM_ROWS):
        values = batch[row]
        stats.append(
            RowStats(
                row=row,
                label=ROW_LABELS[row],
                latest=float(values[-1]),
                mean=float(np.mean(values)),
                spread=float(np.max(values) - np.min(values)),
            )
        )
    return stats


def board_descriptor_contract() -> dict[str, object]:
    return {
        "name": "KnightIMU",
        "sampling_rate": SAMPLING_RATE_HZ,
        "package_num_channel": PACKAGE_ROW,
        "timestamp_channel": TIMESTAMP_ROW,
        "marker_channel": MARKER_ROW,
        "num_rows": NUM_ROWS,
        "eeg_channels": list(EXG_ROWS),
        "other_channels": [
            LOFF_STATP_ROW,
            LOFF_STATN_ROW,
            *ACCEL_ROWS,
            *GYRO_ROWS,
            *MAG_ROWS,
        ],
    }

