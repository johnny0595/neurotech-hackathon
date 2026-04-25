import numpy as np

from neuro_cursor.config import JawConfig
from neuro_cursor.jaw_features import (
    FEATURE_NAMES,
    extract_labeled_clips,
    jaw_feature_vector,
    jaw_signal,
    session_feature_payload,
)
from neuro_cursor.rows import NUM_ROWS


def test_jaw_features_use_channel_2_only():
    raw = np.zeros((NUM_ROWS, 20), dtype=float)
    raw[1] = 1000.0
    raw[2, 5:10] = [0, 10, -10, 20, -20]
    config = JawConfig(channels=[2], sampling_rate=10.0)

    assert np.all(jaw_signal(raw, config) == raw[2])
    vector = jaw_feature_vector(raw[2], config.sampling_rate)

    assert vector.shape == (len(FEATURE_NAMES),)
    assert vector[0] > 0


def test_clip_extraction_includes_pre_and_post_padding():
    raw = np.zeros((NUM_ROWS, 100), dtype=float)
    labels = [
        {
            "label": "jaw_clench",
            "type": "interval",
            "start_sample": 40,
            "end_sample": 45,
        }
    ]
    config = JawConfig(channels=[2], sampling_rate=10.0, pre_event_seconds=0.5, post_event_seconds=0.8)

    clips = extract_labeled_clips(raw, labels, config)

    assert len(clips) == 1
    assert clips[0].start_sample == 35
    assert clips[0].end_sample == 53


def test_session_feature_payload_contains_clip_features():
    raw = np.zeros((NUM_ROWS, 100), dtype=float)
    raw[2, 40:45] = [0, 10, -10, 20, -20]
    labels = [
        {"label": "neutral", "type": "interval", "start_sample": 0, "end_sample": 20},
        {"label": "jaw_clench", "type": "interval", "start_sample": 40, "end_sample": 45},
    ]
    config = JawConfig(channels=[2], sampling_rate=10.0)

    payload = session_feature_payload(raw, labels, config)

    assert payload["jaw_channels"] == [2]
    assert payload["clips"][0]["label"] == "jaw_clench"
    assert payload["clips"][0]["features"]["peak_to_peak"] == 40.0
