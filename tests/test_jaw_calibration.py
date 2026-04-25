from neuro_cursor.config import JawConfig
from neuro_cursor.jaw_calibration import GuidedJawCalibration, build_guided_jaw_schedule


def test_guided_schedule_has_exact_event_counts():
    config = JawConfig(
        sampling_rate=10.0,
        short_clench_seconds=0.5,
        relax_seconds=1.0,
        hold_seconds=2.0,
        short_clench_reps=5,
        hold_reps=3,
    )
    schedule = build_guided_jaw_schedule(config)

    assert sum(1 for phase in schedule if phase.interval_label == "jaw_clench") == 5
    assert sum(1 for phase in schedule if phase.interval_label == "jaw_hold") == 3


def test_guided_calibration_emits_start_end_labels():
    config = JawConfig(
        sampling_rate=10.0,
        short_clench_seconds=0.5,
        relax_seconds=1.0,
        hold_seconds=2.0,
        short_clench_reps=1,
        hold_reps=1,
    )
    calibration = GuidedJawCalibration(config)
    calibration.start(0)

    labels = [label.to_dict() for label in calibration.update(1000)]
    names = [label["label"] for label in labels]

    assert "jaw_clench_start" in names
    assert "jaw_clench" in names
    assert "jaw_clench_end" in names
    assert "jaw_hold_start" in names
    assert "jaw_hold" in names
    assert "jaw_hold_end" in names
    assert calibration.finished
