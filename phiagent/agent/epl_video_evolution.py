"""EPL-conditioned parameter evolution for localized video replacement.

This module deliberately has no numerical or video dependencies.  The renderer
persists measured scorecards and asks this policy for one bounded repair at a
time; it is not presented as a learned policy or as official PhiZero inference.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

from phiagent.physical_language.schema import ManipulationPhase


@dataclass(frozen=True)
class ReplacementParameters:
    """One reproducible candidate configuration in camera pixel coordinates."""

    flow_strength: float = 0.0
    flow_blur_pixels: int = 1
    flow_clip_pixels: float = 28.0
    mask_dilation_pixels: int = 1
    mask_feather_pixels: float = 1.0
    protect_objects: bool = False
    object_dilation_pixels: int = 0
    coordinate_frame: str = "camera:source_pixels"

    def __post_init__(self) -> None:
        if not 0.0 <= self.flow_strength <= 1.0:
            raise ValueError("flow_strength must be in [0, 1]")
        if self.flow_blur_pixels < 1 or self.flow_blur_pixels % 2 == 0:
            raise ValueError("flow_blur_pixels must be a positive odd integer")
        if not math.isfinite(self.flow_clip_pixels) or self.flow_clip_pixels <= 0:
            raise ValueError("flow_clip_pixels must be finite and positive")
        if self.mask_dilation_pixels < 0 or self.object_dilation_pixels < 0:
            raise ValueError("mask dilation values must be non-negative")
        if not math.isfinite(self.mask_feather_pixels) or self.mask_feather_pixels < 0:
            raise ValueError("mask_feather_pixels must be finite and non-negative")
        if self.coordinate_frame != "camera:source_pixels":
            raise ValueError("replacement transforms must use camera:source_pixels")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseScore:
    phase: ManipulationPhase
    motion_preservation: float
    temporal_consistency: float
    samples: int

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("phase score requires at least one sample")
        for field in ("motion_preservation", "temporal_consistency"):
            value = getattr(self, field)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be finite and in [0, 1]")


@dataclass(frozen=True)
class ReplacementScorecard:
    background_lock: float
    object_lock: float
    subject_replacement: float
    robot_identity: float
    motion_preservation: float
    temporal_consistency: float
    phase_scores: tuple[PhaseScore, ...]

    def __post_init__(self) -> None:
        for field in (
            "background_lock",
            "object_lock",
            "subject_replacement",
            "robot_identity",
            "motion_preservation",
            "temporal_consistency",
        ):
            value = getattr(self, field)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be finite and in [0, 1]")
        if not self.phase_scores:
            raise ValueError("scorecard requires EPL phase scores")

    @property
    def epl_minimum(self) -> float:
        return min(
            min(score.motion_preservation, score.temporal_consistency)
            for score in self.phase_scores
        )

    @property
    def mean_score(self) -> float:
        values = (
            self.background_lock,
            self.object_lock,
            self.subject_replacement,
            self.robot_identity,
            self.motion_preservation,
            self.temporal_consistency,
            self.epl_minimum,
        )
        return sum(values) / len(values)

    def to_dict(self) -> dict[str, object]:
        return {
            "background_lock": self.background_lock,
            "object_lock": self.object_lock,
            "subject_replacement": self.subject_replacement,
            "robot_identity": self.robot_identity,
            "motion_preservation": self.motion_preservation,
            "temporal_consistency": self.temporal_consistency,
            "epl_minimum": self.epl_minimum,
            "mean_score": self.mean_score,
            "phase_scores": [
                {**asdict(score), "phase": score.phase.value}
                for score in self.phase_scores
            ],
        }


@dataclass(frozen=True)
class ReplacementThresholds:
    background_lock: float = 0.999
    object_lock: float = 0.98
    subject_replacement: float = 0.88
    robot_identity: float = 0.72
    motion_preservation: float = 0.72
    temporal_consistency: float = 0.72
    epl_minimum: float = 0.62

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be finite and in [0, 1]")

    def accepted(self, scorecard: ReplacementScorecard) -> bool:
        return all(
            getattr(scorecard, field) >= value
            for field, value in asdict(self).items()
        )

    def constraint_margin(self, scorecard: ReplacementScorecard) -> float:
        return min(
            getattr(scorecard, field) - value
            for field, value in asdict(self).items()
        )


@dataclass(frozen=True)
class EvolutionDecision:
    parameters: ReplacementParameters
    diagnoses: tuple[str, ...]
    actions: tuple[str, ...]


class EPLVideoEvolutionAgent:
    """Deterministic feedback policy for one candidate repair per round."""

    def propose(
        self,
        current: ReplacementParameters,
        scorecard: ReplacementScorecard,
        thresholds: ReplacementThresholds,
    ) -> EvolutionDecision:
        parameters = current
        diagnoses: list[str] = []
        actions: list[str] = []

        if scorecard.object_lock < thresholds.object_lock:
            diagnoses.append("protected flower/stem pixels changed")
            parameters = replace(
                parameters,
                protect_objects=True,
                object_dilation_pixels=min(7, parameters.object_dilation_pixels + 2),
            )
            actions.append("enable and dilate flower/stem source-pixel restoration")
            # Evolve one failure family at a time.  Object preservation is a
            # prerequisite for evaluating motion, so keep a clean protected
            # parent before attempting a higher-risk flow mutation.
            return EvolutionDecision(parameters, tuple(diagnoses), tuple(actions))

        weak_phases = tuple(
            item.phase.value
            for item in scorecard.phase_scores
            if item.motion_preservation < thresholds.epl_minimum
        )
        if scorecard.motion_preservation < thresholds.motion_preservation or weak_phases:
            diagnoses.append(
                "insufficient source motion transfer"
                + (f" during {', '.join(weak_phases)}" if weak_phases else "")
            )
            parameters = replace(
                parameters,
                flow_strength=min(1.0, parameters.flow_strength + 0.35),
            )
            actions.append("increase camera-frame optical-flow conditioning")

        unstable_phases = tuple(
            item.phase.value
            for item in scorecard.phase_scores
            if item.temporal_consistency < thresholds.epl_minimum
        )
        if (
            scorecard.temporal_consistency < thresholds.temporal_consistency
            or unstable_phases
        ):
            diagnoses.append(
                "temporal inconsistency"
                + (f" during {', '.join(unstable_phases)}" if unstable_phases else "")
            )
            parameters = replace(
                parameters,
                flow_blur_pixels=min(15, parameters.flow_blur_pixels + 4),
                flow_clip_pixels=max(10.0, parameters.flow_clip_pixels - 4.0),
                flow_strength=max(0.15, parameters.flow_strength - 0.08),
            )
            actions.append("smooth and clip flow while retaining EPL motion conditioning")

        if scorecard.subject_replacement < thresholds.subject_replacement:
            diagnoses.append("human appearance may remain at the replacement boundary")
            parameters = replace(
                parameters,
                mask_dilation_pixels=min(11, parameters.mask_dilation_pixels + 2),
                mask_feather_pixels=min(4.0, parameters.mask_feather_pixels + 0.5),
            )
            actions.append("expand and feather the localized replacement mask")

        if scorecard.robot_identity < thresholds.robot_identity:
            diagnoses.append("robot appearance deformed away from the identity anchor")
            parameters = replace(
                parameters,
                flow_blur_pixels=min(15, parameters.flow_blur_pixels + 4),
                flow_strength=max(0.0, parameters.flow_strength - 0.15),
            )
            actions.append("reduce and smooth motion transfer to preserve robot identity")

        if scorecard.background_lock < thresholds.background_lock:
            diagnoses.append("pixels outside the allowed subject region changed")
            parameters = replace(
                parameters,
                mask_feather_pixels=max(0.0, parameters.mask_feather_pixels - 0.5),
            )
            actions.append("tighten replacement-boundary feathering")

        if not actions:
            diagnoses.append("all hard constraints passed")
            actions.append("accept candidate")
        elif parameters == current:
            actions.append("parameter limits reached")
        return EvolutionDecision(parameters, tuple(diagnoses), tuple(actions))
