import json

import numpy as np

from neuro_cursor.diagnostics import exg_channel_status, snapshot_payload, write_snapshot
from neuro_cursor.rows import NUM_ROWS, PACKAGE_ROW


def test_active_exg_status_identifies_live_and_flat_zero():
    data = np.zeros((NUM_ROWS, 4), dtype=float)
    data[1] = [10.0, 20.0, 30.0, 40.0]
    statuses = {status.channel: status for status in exg_channel_status(data, [1, 2, 3, 4])}

    assert statuses[1].status == "live"
    assert statuses[1].latest == 40.0
    assert statuses[2].status == "flat_zero"
    assert statuses[3].status == "flat_zero"
    assert statuses[4].status == "flat_zero"


def test_snapshot_payload_keeps_package_counter_and_exg_1_separate():
    data = np.zeros((NUM_ROWS, 2), dtype=float)
    data[PACKAGE_ROW] = [158.0, 159.0]
    data[1] = [100000.0, 117007.9609]

    payload = snapshot_payload(data, [1])

    assert payload["latest_by_label"]["package_counter"] == 159.0
    assert payload["latest_by_label"]["exg_1"] == 117007.9609
    assert payload["rows"][0]["label"] == "package_counter"
    assert payload["rows"][1]["label"] == "exg_1"


def test_write_snapshot_creates_json_file(tmp_path):
    data = np.zeros((NUM_ROWS, 1), dtype=float)
    data[1, 0] = 42.0

    path = write_snapshot(data, [1], root=tmp_path)

    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["latest_by_label"]["exg_1"] == 42.0
