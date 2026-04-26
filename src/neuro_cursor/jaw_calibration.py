"""Guided jaw calibration schedule and event labels."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from .config import JawConfig


@dataclass(frozen=True)
class JawPhase:
    prompt: str
    duration_seconds: float
    interval_label: str
    start_event: str | None = None
    end_event: str | None = None
    trial_type: str = "neutral"
    trial_index: int = 0


EVENT_PROMPTS = {
    "jaw_clench": "Short jaw clench",
    "eyebrow_raise": "Eyebrow raise",
}

EVENT_START_LABELS = {
    "jaw_clench": "jaw_clench_start",
    "eyebrow_raise": "eyebrow_raise_start",
}

EVENT_END_LABELS = {
    "jaw_clench": "jaw_clench_end",
    "eyebrow_raise": "eyebrow_raise_end",
}

HARD_NEGATIVE_PHASES = (
    ("hard_negative_blink", "Blink naturally"),
    ("hard_negative_swallow", "Swallow"),
    ("hard_negative_head_motion", "Small head movement, jaw relaxed"),
    ("hard_negative_teeth_touch", "Light teeth touch, no clench"),
)


@dataclass(frozen=True)
class JawLabel:
    label: str
    type: str
    start_sample: int
    end_sample: int | None = None
    sample: int | None = None
    timestamp: float | None = None
    trial_type: str = ""
    trial_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def build_guided_jaw_schedule(config: JawConfig) -> list[JawPhase]:
    rng = random.Random(config.prompt_seed)
    event_label = config.positive_label
    event_prompt = EVENT_PROMPTS.get(event_label, event_label.replace("_", " ").title())
    start_event = EVENT_START_LABELS.get(event_label, f"{event_label}_start")
    end_event = EVENT_END_LABELS.get(event_label, f"{event_label}_end")
    phases: list[JawPhase] = [
        JawPhase(
            prompt="Neutral baseline",
            duration_seconds=config.initial_neutral_seconds,
            interval_label="neutral",
            trial_type="neutral",
            trial_index=0,
        )
    ]
    for index in range(1, config.short_clench_reps + 1):
        relax_seconds = rng.uniform(config.relax_min_seconds, config.relax_max_seconds)
        event_seconds = rng.uniform(config.event_min_seconds, config.event_max_seconds)
        phases.append(
            JawPhase(
                prompt=f"Relax before trial {index}",
                duration_seconds=relax_seconds,
                interval_label="neutral",
                trial_type="neutral",
                trial_index=index,
            )
        )
        phases.append(
            JawPhase(
                prompt=f"{event_prompt} {index}",
                duration_seconds=event_seconds,
                interval_label=event_label,
                start_event=start_event,
                end_event=end_event,
                trial_type=event_label,
                trial_index=index,
            )
        )
        phases.append(
            JawPhase(
                prompt="Release and relax",
                duration_seconds=config.post_event_seconds,
                interval_label="neutral",
                trial_type="neutral",
                trial_index=index,
            )
        )
    hard_index = 0
    for _ in range(config.hard_negative_reps):
        for label, prompt in HARD_NEGATIVE_PHASES:
            hard_index += 1
            phases.append(
                JawPhase(
                    prompt="Relax",
                    duration_seconds=rng.uniform(config.relax_min_seconds, config.relax_max_seconds),
                    interval_label="neutral",
                    trial_type="neutral",
                    trial_index=hard_index,
                )
            )
            phases.append(
                JawPhase(
                    prompt=prompt,
                    duration_seconds=config.hard_negative_seconds,
                    interval_label=label,
                    trial_type=label,
                    trial_index=hard_index,
                )
            )
    phases.append(
        JawPhase(
            prompt="Final neutral baseline",
            duration_seconds=config.final_neutral_seconds,
            interval_label="neutral",
            trial_type="neutral",
            trial_index=config.short_clench_reps + hard_index + 1,
        )
    )
    return phases


def labels_for_phase(
    phase: JawPhase,
    start_sample: int,
    end_sample: int,
    sampling_rate: float,
) -> list[JawLabel]:
    labels: list[JawLabel] = []
    if phase.start_event:
        labels.append(
            JawLabel(
                label=phase.start_event,
                type="event",
                start_sample=start_sample,
                sample=start_sample,
                timestamp=start_sample / sampling_rate,
                trial_type=phase.trial_type,
                trial_index=phase.trial_index,
            )
        )
    labels.append(
        JawLabel(
            label=phase.interval_label,
            type="interval",
            start_sample=start_sample,
            end_sample=end_sample,
            timestamp=start_sample / sampling_rate,
            trial_type=phase.trial_type,
            trial_index=phase.trial_index,
        )
    )
    if phase.end_event:
        labels.append(
            JawLabel(
                label=phase.end_event,
                type="event",
                start_sample=end_sample,
                sample=end_sample,
                timestamp=end_sample / sampling_rate,
                trial_type=phase.trial_type,
                trial_index=phase.trial_index,
            )
        )
    return labels


class GuidedJawCalibration:
    def __init__(self, config: JawConfig) -> None:
        self.config = config
        self.schedule = build_guided_jaw_schedule(config)
        self.index = 0
        self.phase_start_sample = 0
        self.finished = False

    @property
    def current_phase(self) -> JawPhase | None:
        if self.finished or self.index >= len(self.schedule):
            return None
        return self.schedule[self.index]

    def start(self, sample_count: int = 0) -> None:
        self.index = 0
        self.phase_start_sample = sample_count
        self.finished = False

    def update(self, sample_count: int) -> list[JawLabel]:
        if self.finished:
            return []
        labels: list[JawLabel] = []
        sampling_rate = self.config.sampling_rate
        while not self.finished:
            phase = self.current_phase
            if phase is None:
                self.finished = True
                break
            duration_samples = max(1, int(round(phase.duration_seconds * sampling_rate)))
            phase_end = self.phase_start_sample + duration_samples
            if sample_count < phase_end:
                break
            labels.extend(labels_for_phase(phase, self.phase_start_sample, phase_end, sampling_rate))
            self.index += 1
            self.phase_start_sample = phase_end
            if self.index >= len(self.schedule):
                self.finished = True
                break
        return labels

    def prompt_text(self, sample_count: int) -> str:
        phase = self.current_phase
        if phase is None:
            return "Calibration complete"
        elapsed = max(0, sample_count - self.phase_start_sample) / self.config.sampling_rate
        remaining = max(0.0, phase.duration_seconds - elapsed)
        return f"{phase.prompt}  {remaining:0.1f}s"
