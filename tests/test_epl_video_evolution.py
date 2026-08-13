from __future__ import annotations

import pytest

from phiagent.agent.epl_video_evolution import (
    EPLVideoEvolutionAgent,
    PhaseScore,
    ReplacementParameters,
    ReplacementScorecard,
    ReplacementThresholds,
)
from phiagent.physical_language.schema import ManipulationPhase


def _scorecard(**overrides: float) -> ReplacementScorecard:
    values = {
        "background_lock": 1.0,
        "object_lock": 1.0,
        "subject_replacement": 0.95,
        "robot_identity": 0.92,
        "motion_preservation": 0.85,
        "temporal_consistency": 0.85,
    }
    values.update(overrides)
    return ReplacementScorecard(
        **values,
        phase_scores=(
            PhaseScore(ManipulationPhase.APPROACH, 0.80, 0.82, 10),
            PhaseScore(ManipulationPhase.MANIPULATE, 0.78, 0.80, 20),
        ),
    )


def test_epl_agent_repairs_object_lock_before_motion_conditioning() -> None:
    initial = ReplacementParameters()
    decision = EPLVideoEvolutionAgent().propose(
        initial,
        _scorecard(object_lock=0.2, motion_preservation=0.3),
        ReplacementThresholds(),
    )

    assert decision.parameters.protect_objects
    assert decision.parameters.object_dilation_pixels == 2
    assert decision.parameters.flow_strength == pytest.approx(0.0)
    assert any("flower" in diagnosis for diagnosis in decision.diagnoses)
    assert len(decision.actions) == 1


def test_epl_agent_smooths_phase_local_temporal_failure() -> None:
    scorecard = ReplacementScorecard(
        background_lock=1.0,
        object_lock=1.0,
        subject_replacement=0.95,
        robot_identity=0.92,
        motion_preservation=0.8,
        temporal_consistency=0.8,
        phase_scores=(
            PhaseScore(ManipulationPhase.APPROACH, 0.8, 0.8, 10),
            PhaseScore(ManipulationPhase.GRASP, 0.8, 0.3, 10),
        ),
    )
    decision = EPLVideoEvolutionAgent().propose(
        ReplacementParameters(flow_strength=0.7),
        scorecard,
        ReplacementThresholds(),
    )

    assert decision.parameters.flow_blur_pixels == 5
    assert decision.parameters.flow_strength == pytest.approx(0.62)
    assert "grasp" in " ".join(decision.diagnoses)


def test_thresholds_use_phase_minimum_as_hard_gate() -> None:
    scorecard = ReplacementScorecard(
        background_lock=1.0,
        object_lock=1.0,
        subject_replacement=1.0,
        robot_identity=1.0,
        motion_preservation=1.0,
        temporal_consistency=1.0,
        phase_scores=(PhaseScore(ManipulationPhase.RELEASE, 0.4, 1.0, 1),),
    )

    thresholds = ReplacementThresholds()
    assert not thresholds.accepted(scorecard)
    assert thresholds.constraint_margin(scorecard) == pytest.approx(-0.22)


def test_epl_agent_reduces_flow_when_robot_identity_breaks() -> None:
    decision = EPLVideoEvolutionAgent().propose(
        ReplacementParameters(flow_strength=0.7),
        _scorecard(robot_identity=0.2),
        ReplacementThresholds(),
    )

    assert decision.parameters.flow_strength == pytest.approx(0.55)
    assert decision.parameters.flow_blur_pixels == 5
    assert "identity" in " ".join(decision.diagnoses)


def test_replacement_parameters_require_explicit_camera_pixel_frame() -> None:
    with pytest.raises(ValueError, match="camera:source_pixels"):
        ReplacementParameters(coordinate_frame="world")
