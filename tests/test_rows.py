import numpy as np
import pytest

from neuro_cursor.rows import (
    ACCEL_ROWS,
    GYRO_ROWS,
    MAG_ROWS,
    NUM_ROWS,
    accel,
    gyro,
    latest_sample,
    loff_bits,
    mag,
    row_stats,
    validate_batch,
)


def test_validate_batch_requires_22_rows():
    data = np.zeros((NUM_ROWS, 3))
    assert validate_batch(data).shape == (NUM_ROWS, 3)
    with pytest.raises(ValueError):
        validate_batch(np.zeros((21, 3)))


def test_vectors_use_explicit_knight_imu_rows():
    raw = np.arange(NUM_ROWS, dtype=float)
    assert accel(raw).tolist() == list(ACCEL_ROWS)
    assert gyro(raw).tolist() == list(GYRO_ROWS)
    assert mag(raw).tolist() == list(MAG_ROWS)


def test_latest_sample_and_stats():
    data = np.tile(np.arange(NUM_ROWS, dtype=float).reshape(NUM_ROWS, 1), (1, 4))
    data[:, -1] += 1
    sample = latest_sample(data)
    assert sample.raw.shape == (NUM_ROWS,)
    stats = row_stats(data)
    assert len(stats) == NUM_ROWS
    assert stats[0].latest == 1.0
    assert stats[0].spread == 1.0


def test_loff_bits():
    assert loff_bits(0) == []
    assert loff_bits(0b1010) == [1, 3]

