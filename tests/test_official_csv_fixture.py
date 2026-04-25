from pathlib import Path

import numpy as np
import pytest

from neuro_cursor.capture import csv_to_raw, exg_features


CSV_PATH = Path("/Users/jonathanduran-ortiz/Downloads/recording_1777151365.csv")


@pytest.mark.skipif(not CSV_PATH.exists(), reason="local official NeuroPawn CSV fixture is unavailable")
def test_official_csv_fixture_uses_channel_2_only():
    csv_data = np.loadtxt(CSV_PATH, delimiter=",")
    raw = csv_to_raw(csv_data, sampling_rate=csv_data.shape[0] / 30.0)
    nonzero = [int(np.count_nonzero(csv_data[:, index])) for index in range(csv_data.shape[1])]
    features = {feature["channel"]: feature for feature in exg_features(raw)}

    assert nonzero[1] > 0
    assert nonzero[:1] + nonzero[2:] == [0, 0, 0, 0, 0, 0, 0]
    assert features[2]["peak_to_peak"] > 0
    assert features[2]["peak_to_peak"] == max(feature["peak_to_peak"] for feature in features.values())
