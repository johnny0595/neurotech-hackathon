"""Guided jaw calibration schedule and event labels."""

from __future__ import annotations

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
    phases: list[JawPhase] = [
        JawPhase(
            prompt="Relax jaw",
            duration_seconds=config.relax_seconds,
            interval_label="neutral",
            trial_type="neutral",
            trial_index=0,
        )
    ]
    for index in range(1, config.short_clench_reps + 1):
        phases.append(
            JawPhase(
                prompt=f"Relax before short clench {index}",
                duration_seconds=config.relax_seconds,
                interval_label="neutral",
                trial_type="neutral",
                trial_index=index,
            )
        )
        phases.append(
            JawPhase(
                prompt=f"Short jaw clench {index}",
                duration_seconds=config.short_clench_seconds,
                interval_label="jaw_clench",
                start_event="jaw_clench_start",
                end_event="jaw_clench_end",
                trial_type="short_clench",
                trial_index=index,
            )
        )
    for index in range(1, config.hold_reps + 1):
        phases.append(
            JawPhase(
                prompt=f"Relax before jaw hold {index}",
                duration_seconds=config.relax_seconds,
                interval_label="neutral",
                trial_type="neutral",
                trial_index=index,
            )
        )
        phases.append(
            JawPhase(
                prompt=f"Hold jaw clench {index}",
                duration_seconds=config.hold_seconds,
                interval_label="jaw_hold",
                start_event="jaw_hold_start",
                end_event="jaw_hold_end",
                trial_type="jaw_hold",
                trial_index=index,
            )
        )
    phases.append(
        JawPhase(
            prompt="Relax jaw",
            duration_seconds=config.relax_seconds,
            interval_label="neutral",
            trial_type="neutral",
            trial_index=config.short_clench_reps + config.hold_reps + 1,
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
