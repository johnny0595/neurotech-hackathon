from neuro_cursor.config import JawConfig
from neuro_cursor.jaw_calibration import GuidedJawCalibration, build_guided_jaw_schedule


def test_guided_schedule_has_exact_event_counts():
    config = JawConfig(
        sampling_rate=10.0,
        initial_neutral_seconds=1.0,
        final_neutral_seconds=1.0,
        event_min_seconds=0.5,
        event_max_seconds=0.5,
        relax_min_seconds=1.0,
        relax_max_seconds=1.0,
        short_clench_reps=5,
        hold_reps=0,
        hard_negative_reps=1,
    )
    schedule = build_guided_jaw_schedule(config)

    assert sum(1 for phase in schedule if phase.interval_label == "eyebrow_raise") == 5
    assert sum(1 for phase in schedule if phase.interval_label == "jaw_hold") == 0
    assert sum(1 for phase in schedule if phase.interval_label.startswith("hard_negative")) == 4


def test_guided_calibration_emits_start_end_labels():
    config = JawConfig(
        sampling_rate=10.0,
        initial_neutral_seconds=1.0,
        final_neutral_seconds=1.0,
        event_min_seconds=0.5,
        event_max_seconds=0.5,
        relax_min_seconds=1.0,
        relax_max_seconds=1.0,
        short_clench_reps=1,
        hold_reps=0,
        hard_negative_reps=1,
    )
    calibration = GuidedJawCalibration(config)
    calibration.start(0)

    labels = [label.to_dict() for label in calibration.update(1000)]
    names = [label["label"] for label in labels]

    assert "eyebrow_raise_start" in names
    assert "eyebrow_raise" in names
    assert "eyebrow_raise_end" in names
    assert "jaw_hold" not in names
    assert "hard_negative_blink" in names
    assert calibration.finished


def test_guided_calibration_can_target_eyebrow_raise():
    config = JawConfig(
        sampling_rate=10.0,
        initial_neutral_seconds=1.0,
        final_neutral_seconds=1.0,
        event_min_seconds=0.5,
        event_max_seconds=0.5,
        relax_min_seconds=1.0,
        relax_max_seconds=1.0,
        short_clench_reps=1,
        hold_reps=0,
        hard_negative_reps=0,
        positive_label="eyebrow_raise",
    )
    calibration = GuidedJawCalibration(config)
    calibration.start(0)

    labels = [label.to_dict() for label in calibration.update(1000)]
    names = [label["label"] for label in labels]

    assert "eyebrow_raise_start" in names
    assert "eyebrow_raise" in names
    assert "eyebrow_raise_end" in names
