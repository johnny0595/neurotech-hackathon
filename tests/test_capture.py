import numpy as np

from neuro_cursor.capture import (
    counter_gap_count,
    csv_to_raw,
    exg_csv_matrix,
    exg_features,
    _manual_labels_from_args,
    timestamp_rate,
    write_capture,
)
from neuro_cursor.config import JawConfig
from neuro_cursor.rows import NUM_ROWS, PACKAGE_ROW, TIMESTAMP_ROW


def test_exg_features_rank_large_channel_activity():
    data = np.zeros((NUM_ROWS, 5), dtype=float)
    data[1] = [0.0, 10.0, -10.0, 20.0, -20.0]

    features = {feature["channel"]: feature for feature in exg_features(data)}

    assert features[1]["peak_to_peak"] == 40.0
    assert features[1]["rms_centered"] > features[2]["rms_centered"]
    assert features[2]["nonzero_samples"] == 0


def test_counter_gap_count_uses_wrapping_package_counter():
    data = np.zeros((NUM_ROWS, 5), dtype=float)
    data[PACKAGE_ROW] = [254, 255, 0, 2, 3]

    assert counter_gap_count(data) == 1


def test_timestamp_rate_reports_samples_per_second():
    data = np.zeros((NUM_ROWS, 5), dtype=float)
    data[TIMESTAMP_ROW] = [10.0, 10.25, 10.5, 10.75, 11.0]

    assert timestamp_rate(data) == 4.0


def test_csv_to_raw_maps_eight_exg_columns_to_rows():
    csv_data = np.zeros((3, 8), dtype=float)
    csv_data[:, 0] = [1.0, 2.0, 3.0]
    csv_data[:, 1] = [4.0, 5.0, 6.0]

    raw = csv_to_raw(csv_data, sampling_rate=10.0)

    assert raw.shape == (NUM_ROWS, 3)
    assert raw[1].tolist() == [1.0, 2.0, 3.0]
    assert raw[2].tolist() == [4.0, 5.0, 6.0]
    assert raw[TIMESTAMP_ROW].tolist() == [0.0, 0.1, 0.2]


def test_exg_csv_matrix_exports_visualizer_shape():
    data = np.zeros((NUM_ROWS, 2), dtype=float)
    data[1] = [10.0, 20.0]
    data[8] = [80.0, 90.0]

    matrix = exg_csv_matrix(data)

    assert matrix.shape == (2, 8)
    assert matrix[:, 0].tolist() == [10.0, 20.0]
    assert matrix[:, 7].tolist() == [80.0, 90.0]


def test_write_capture_writes_raw_exg_csv_and_labels(tmp_path):
    data = np.zeros((NUM_ROWS, 2), dtype=float)
    config = {
        "board": {"active_exg_channels": [2]},
        "channel_map": {1: "unused", 2: "jaw"},
        "jaw": {"channels": [2]},
    }

    capture_dir = write_capture(data, config, "jaw_clench", tmp_path, metadata={})

    assert (capture_dir / "raw.npz").exists()
    assert (capture_dir / "exg.csv").exists()
    assert (capture_dir / "labels.jsonl").exists()
    assert (capture_dir / "clips.npz").exists()
    assert (capture_dir / "summary.json").exists()


def test_manual_csv_labels_make_neutral_complement():
    class Args:
        event_times = "2.0"
        hold_intervals = ""

    config = JawConfig(sampling_rate=10.0, short_clench_seconds=0.5)

    labels = _manual_labels_from_args(Args(), config, samples=50)
    neutral = [label for label in labels if label["label"] == "neutral"]
    clench = [label for label in labels if label["label"] == "jaw_clench"][0]

    assert clench["start_sample"] == 18
    assert clench["end_sample"] == 23
    assert neutral == [
        {
            "label": "neutral",
            "type": "interval",
            "start_sample": 0,
            "end_sample": 18,
            "trial_type": "background",
            "trial_index": 1,
        },
        {
            "label": "neutral",
            "type": "interval",
            "start_sample": 23,
            "end_sample": 50,
            "trial_type": "background",
            "trial_index": 2,
        },
    ]
