"""Strict end-to-end acceptance for embodiment-transfer evaluations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from phiagent.evaluation.acceptance import (
    AcceptanceContract,
    AcceptanceDecision,
    EvaluationRecord,
    ExperimentStatistics,
    GateRequirement,
    IndependentEvaluationUnit,
    evaluate_acceptance,
    summarize_experiment,
)
from phiagent.evaluation.embodiment import EmbodimentScorecard
from phiagent.evaluation.interaction import InteractionScorecard
from phiagent.evaluation.task_motion import TaskMotionScorecard
from phiagent.evaluation.video_quality import VideoQualityScorecard


_GATE_WEIGHTS = {
    "action_adherence": 3.0,
    "phase_agreement": 2.0,
    "contact_agreement": 2.5,
    "embodiment_consistency": 2.5,
    "object_interaction": 3.0,
    "interaction_contract": 1.0,
    "temporal_consistency": 1.5,
    "motion_physicality": 1.5,
    "background_consistency": 0.5,
    "visual_proxy_quality": 0.5,
}


def _unit_score(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")


@dataclass(frozen=True)
class PhysicalTransferThresholds:
    """Hard gates for a valid physical-state and embodiment transfer."""

    action_adherence: float = 0.75
    phase_agreement: float = 0.75
    contact_agreement: float = 0.75
    embodiment_consistency: float = 0.75
    object_interaction: float = 0.75
    interaction_contract: float = 1.0
    temporal_consistency: float = 0.75
    motion_physicality: float = 0.75
    background_consistency: float = 0.75
    minimum_visual_proxy_quality: float | None = None
    human_review_required: bool = True

    def __post_init__(self) -> None:
        for name in (
            "action_adherence",
            "phase_agreement",
            "contact_agreement",
            "embodiment_consistency",
            "object_interaction",
            "interaction_contract",
            "temporal_consistency",
            "motion_physicality",
            "background_consistency",
        ):
            _unit_score(getattr(self, name), f"{name} threshold")
        if self.minimum_visual_proxy_quality is not None:
            _unit_score(
                self.minimum_visual_proxy_quality,
                "minimum_visual_proxy_quality threshold",
            )
        if not isinstance(self.human_review_required, bool):
            raise ValueError("human_review_required must be a boolean")

    def contract(self) -> AcceptanceContract:
        thresholds = (
            ("action_adherence", self.action_adherence),
            ("phase_agreement", self.phase_agreement),
            ("contact_agreement", self.contact_agreement),
            ("embodiment_consistency", self.embodiment_consistency),
            ("object_interaction", self.object_interaction),
            ("interaction_contract", self.interaction_contract),
            ("temporal_consistency", self.temporal_consistency),
            ("motion_physicality", self.motion_physicality),
            ("background_consistency", self.background_consistency),
        )
        requirements = [
            GateRequirement(name, threshold, _GATE_WEIGHTS[name]) for name, threshold in thresholds
        ]
        if self.minimum_visual_proxy_quality is not None:
            requirements.append(
                GateRequirement(
                    "visual_proxy_quality",
                    self.minimum_visual_proxy_quality,
                    _GATE_WEIGHTS["visual_proxy_quality"],
                )
            )
        return AcceptanceContract(
            tuple(requirements),
            human_review_required=self.human_review_required,
        )


@dataclass(frozen=True)
class PhysicalTransferGateScores:
    """Normalized gate values derived from the four specialized evaluators."""

    action_adherence: float
    phase_agreement: float
    contact_agreement: float
    embodiment_consistency: float
    object_interaction: float
    interaction_contract: float
    temporal_consistency: float
    motion_physicality: float
    background_consistency: float
    visual_proxy_quality: float

    def __post_init__(self) -> None:
        for name in self.as_mapping():
            _unit_score(getattr(self, name), name)

    @classmethod
    def from_scorecards(
        cls,
        *,
        task_motion: TaskMotionScorecard,
        embodiment: EmbodimentScorecard,
        interaction: InteractionScorecard,
        video_quality: VideoQualityScorecard,
    ) -> "PhysicalTransferGateScores":
        if not isinstance(task_motion, TaskMotionScorecard):
            raise TypeError("task_motion must be a TaskMotionScorecard")
        if not isinstance(embodiment, EmbodimentScorecard):
            raise TypeError("embodiment must be an EmbodimentScorecard")
        if not isinstance(interaction, InteractionScorecard):
            raise TypeError("interaction must be an InteractionScorecard")
        if not isinstance(video_quality, VideoQualityScorecard):
            raise TypeError("video_quality must be a VideoQualityScorecard")

        object_interaction = min(
            interaction.identity_score,
            interaction.visibility_coverage,
            interaction.relative_trajectory_score,
            interaction.terminal_state_score,
            interaction.hand_object_distance_score,
            interaction.contact_agreement_score,
            interaction.contact_timing_score,
            interaction.motion_coupling_score,
            interaction.causal_order_score,
            interaction.continuity_score,
            interaction.manipulation_score,
        )
        motion_physicality = min(
            embodiment.articulation,
            video_quality.motion_requirement_score,
            video_quality.motion_smoothness_score,
        )
        return cls(
            action_adherence=task_motion.action_adherence,
            phase_agreement=task_motion.phase_agreement,
            contact_agreement=task_motion.contact_agreement,
            embodiment_consistency=embodiment.essential,
            object_interaction=object_interaction,
            interaction_contract=float(interaction.passed),
            temporal_consistency=video_quality.temporal_score,
            motion_physicality=motion_physicality,
            background_consistency=video_quality.background_preservation_score,
            visual_proxy_quality=video_quality.sharpness_score,
        )

    def as_mapping(self) -> dict[str, float]:
        return {
            "action_adherence": self.action_adherence,
            "phase_agreement": self.phase_agreement,
            "contact_agreement": self.contact_agreement,
            "embodiment_consistency": self.embodiment_consistency,
            "object_interaction": self.object_interaction,
            "interaction_contract": self.interaction_contract,
            "temporal_consistency": self.temporal_consistency,
            "motion_physicality": self.motion_physicality,
            "background_consistency": self.background_consistency,
            "visual_proxy_quality": self.visual_proxy_quality,
        }


@dataclass(frozen=True)
class PhysicalTransferAssessment:
    """One auditable all-gates decision and its source gate values."""

    scores: PhysicalTransferGateScores
    record: EvaluationRecord
    contract: AcceptanceContract
    decision: AcceptanceDecision

    def __post_init__(self) -> None:
        if self.record.unit != self.decision.unit:
            raise ValueError("assessment record and decision units must match")
        expected = evaluate_acceptance(self.contract, self.record)
        if self.decision != expected:
            raise ValueError("assessment decision does not match its bound acceptance contract")

    @property
    def accepted(self) -> bool:
        return self.decision.accepted


def evaluate_physical_transfer_scores(
    unit: IndependentEvaluationUnit,
    scores: PhysicalTransferGateScores,
    *,
    human_review: bool | None,
    thresholds: PhysicalTransferThresholds = PhysicalTransferThresholds(),
) -> PhysicalTransferAssessment:
    """Apply the strict transfer contract to already-computed specialized scores."""

    record = EvaluationRecord(
        unit=unit,
        gate_scores=scores.as_mapping(),
        human_review=human_review,
    )
    contract = thresholds.contract()
    decision = evaluate_acceptance(contract, record)
    return PhysicalTransferAssessment(
        scores=scores,
        record=record,
        contract=contract,
        decision=decision,
    )


def evaluate_physical_transfer(
    unit: IndependentEvaluationUnit,
    *,
    task_motion: TaskMotionScorecard,
    embodiment: EmbodimentScorecard,
    interaction: InteractionScorecard,
    video_quality: VideoQualityScorecard,
    human_review: bool | None,
    thresholds: PhysicalTransferThresholds = PhysicalTransferThresholds(),
) -> PhysicalTransferAssessment:
    """Combine specialized evidence without allowing one metric to mask another."""

    scores = PhysicalTransferGateScores.from_scorecards(
        task_motion=task_motion,
        embodiment=embodiment,
        interaction=interaction,
        video_quality=video_quality,
    )
    return evaluate_physical_transfer_scores(
        unit,
        scores,
        human_review=human_review,
        thresholds=thresholds,
    )


def summarize_physical_transfers(
    assessments: Iterable[PhysicalTransferAssessment],
    *,
    grouping_key: str,
    thresholds: PhysicalTransferThresholds | None = None,
) -> ExperimentStatistics:
    """Report VTR using the exact contract bound to every assessment."""

    assessments = tuple(assessments)
    if not assessments:
        raise ValueError("cannot summarize an empty physical-transfer experiment")
    bound_contract = assessments[0].contract
    if any(assessment.contract != bound_contract for assessment in assessments[1:]):
        raise ValueError("all assessments must use the same acceptance contract")
    if thresholds is not None and thresholds.contract() != bound_contract:
        raise ValueError("summary thresholds do not match the assessments' bound contract")
    return summarize_experiment(
        bound_contract,
        (assessment.record for assessment in assessments),
        grouping_key=grouping_key,
    )
