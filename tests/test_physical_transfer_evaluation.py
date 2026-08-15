from __future__ import annotations

from dataclasses import replace

import pytest

from phiagent.evaluation.acceptance import IndependentEvaluationUnit
from phiagent.evaluation.embodiment import EmbodimentDiagnostics, EmbodimentScorecard
from phiagent.evaluation.interaction import InteractionDiagnostics, InteractionScorecard
from phiagent.evaluation.physical_transfer import (
    PhysicalTransferGateScores,
    PhysicalTransferThresholds,
    evaluate_physical_transfer,
    evaluate_physical_transfer_scores,
    summarize_physical_transfers,
)
from phiagent.evaluation.task_motion import (
    ActionDiagnostics,
    ContactDiagnostics,
    PhaseDiagnostics,
    TaskMotionScorecard,
)
from phiagent.evaluation.video_quality import TrajectoryDiagnostics, VideoQualityScorecard


def _unit(action: str = "lift", embodiment: str = "sharpa", seed: int = 1):
    return IndependentEvaluationUnit(
        scene="scene-1",
        action=action,
        object="bowl",
        embodiment=embodiment,
        seed=seed,
    )


def _scores(value: float = 1.0) -> PhysicalTransferGateScores:
    return PhysicalTransferGateScores(
        action_adherence=value,
        phase_agreement=value,
        contact_agreement=value,
        embodiment_consistency=value,
        object_interaction=value,
        interaction_contract=value,
        temporal_consistency=value,
        motion_physicality=value,
        background_consistency=value,
        visual_proxy_quality=value,
    )


def _task_scorecard() -> TaskMotionScorecard:
    return TaskMotionScorecard(
        *(1.0 for _ in range(12)),
        action_diagnostics=ActionDiagnostics(0.0, 0.0, 1.0, 1.0, 1.0),
        phase_diagnostics=PhaseDiagnostics(0.0, ("grasp",)),
        contact_diagnostics=ContactDiagnostics(0.0, 1, 1, 1),
    )


def _embodiment_scorecard() -> EmbodimentScorecard:
    diagnostics = EmbodimentDiagnostics(
        component_counts=(1, 1),
        connected_component_counts=(1, 1),
        component_areas=(16, 16),
        missing_landmarks=((), ()),
        link_relative_drifts=(("finger", 0.0),),
        target_ids=("sharpa", "sharpa"),
        articulation_displacements=(0.2,),
        sustained_articulation_displacements=(0.2,),
    )
    return EmbodimentScorecard(*(1.0 for _ in range(7)), diagnostics=diagnostics)


def _interaction_scorecard(*, passed: bool = True) -> InteractionScorecard:
    diagnostics = InteractionDiagnostics(
        identity_valid_frames=3,
        visible_target_frames=3,
        missing_target_frames=(),
        duplicate_target_frames=(),
        candidate_hand_object_distances_m=(0.02, 0.01, 0.01),
        reference_hand_object_distances_m=(0.02, 0.01, 0.01),
        candidate_contact=(True, True, True),
        reference_contact=(True, True, True),
        candidate_contact_onset_s=0.0,
        candidate_contact_offset_s=0.2,
        reference_contact_onset_s=0.0,
        reference_contact_offset_s=0.2,
        object_motion_onset_s=0.1,
        coupling_errors_m=(0.0, 0.0),
        teleport_frames=(),
        reasons=(),
    )
    values = (1.0 for _ in range(11))
    return InteractionScorecard(*values, passed=passed, diagnostics=diagnostics)


def _video_scorecard() -> VideoQualityScorecard:
    trajectory = TrajectoryDiagnostics(
        scale=1.0,
        timestamp_unit_seconds=1.0,
        jerk_normalization=1.0,
        speeds=(1.0, 1.0),
        accelerations=(0.0,),
        jerks=(),
        smoothness_score=1.0,
    )
    return VideoQualityScorecard(
        *(1.0 for _ in range(13)),
        candidate_trajectory=trajectory,
        reference_trajectory=trajectory,
        mean_background_error=0.0,
        mean_activity=0.1,
        mean_articulation=0.1,
        estimated_translations=((0, 0),),
        candidate_temporal_errors=(0.0,),
        reference_temporal_errors=(0.0,),
        candidate_roi_temporal_errors=(0.0,),
        reference_roi_temporal_errors=(0.0,),
        background_errors=(0.0,),
    )


def test_all_gates_and_human_review_are_required() -> None:
    accepted = evaluate_physical_transfer_scores(_unit(), _scores(), human_review=True)
    pending = evaluate_physical_transfer_scores(_unit(), _scores(), human_review=None)
    weak_action = evaluate_physical_transfer_scores(
        _unit(),
        replace(_scores(), action_adherence=0.1),
        human_review=True,
    )

    assert accepted.accepted
    assert not pending.accepted
    assert not weak_action.accepted
    assert weak_action.decision.mean_score is not None
    assert weak_action.decision.mean_score > 0.8
    assert weak_action.decision.gate_failure_names == ("action_adherence",)


def test_visual_proxy_is_diagnostic_unless_explicitly_required() -> None:
    weak_visual = replace(_scores(), visual_proxy_quality=0.1)
    visual_thresholds = PhysicalTransferThresholds(minimum_visual_proxy_quality=0.75)

    diagnostic_only = evaluate_physical_transfer_scores(
        _unit(),
        weak_visual,
        human_review=True,
    )
    required = evaluate_physical_transfer_scores(
        _unit(),
        weak_visual,
        human_review=True,
        thresholds=visual_thresholds,
    )

    assert diagnostic_only.accepted
    assert not required.accepted
    assert required.decision.gate_failure_names == ("visual_proxy_quality",)
    statistics = summarize_physical_transfers((required,), grouping_key="action")
    assert statistics.passed == 0
    with pytest.raises(ValueError, match="do not match"):
        summarize_physical_transfers(
            (required,),
            grouping_key="action",
            thresholds=PhysicalTransferThresholds(),
        )


def test_specialized_scorecards_map_to_strict_transfer_gates() -> None:
    accepted = evaluate_physical_transfer(
        _unit(),
        task_motion=_task_scorecard(),
        embodiment=_embodiment_scorecard(),
        interaction=_interaction_scorecard(),
        video_quality=_video_scorecard(),
        human_review=True,
    )
    interaction_failed = evaluate_physical_transfer(
        _unit(),
        task_motion=_task_scorecard(),
        embodiment=_embodiment_scorecard(),
        interaction=_interaction_scorecard(passed=False),
        video_quality=_video_scorecard(),
        human_review=True,
    )

    assert accepted.accepted
    assert accepted.scores.motion_physicality == pytest.approx(1.0)
    assert not interaction_failed.accepted
    assert interaction_failed.decision.gate_failure_names == ("interaction_contract",)


def test_vtr_uses_unique_units_and_reports_worst_action() -> None:
    assessments = (
        evaluate_physical_transfer_scores(_unit("lift", seed=1), _scores(), human_review=True),
        evaluate_physical_transfer_scores(_unit("lift", seed=2), _scores(), human_review=True),
        evaluate_physical_transfer_scores(
            _unit("slide", seed=1),
            replace(_scores(), action_adherence=0.0),
            human_review=True,
        ),
    )

    statistics = summarize_physical_transfers(assessments, grouping_key="action")

    assert statistics.passed == 2
    assert statistics.total == 3
    assert statistics.worst_group.group == "slide"
    assert statistics.worst_group.valid_transfer_rate.rate == 0.0
    with pytest.raises(ValueError, match="duplicate independent"):
        summarize_physical_transfers(
            (assessments[0], assessments[0]),
            grouping_key="action",
        )
