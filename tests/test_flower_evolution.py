from __future__ import annotations

import pytest

from phiagent.agent.flower_evolution import (
    GATE_NAMES,
    FlowerAcceptanceContract,
    FlowerCandidateEvaluation,
    FlowerEvolutionAgent,
    FlowerPipelineConfig,
    GateMeasurement,
    GateVerdict,
    PipelineFamily,
)


def _measurement(name: str, score: float = 1.0) -> GateMeasurement:
    return GateMeasurement(name, GateVerdict.PASS, score, (f"evidence:{name}",))


def _evaluation(**failures: GateVerdict) -> FlowerCandidateEvaluation:
    measurements = []
    for name in GATE_NAMES:
        verdict = failures.get(name, GateVerdict.PASS)
        score = 1.0 if verdict is GateVerdict.PASS else None
        measurements.append(GateMeasurement(name, verdict, score, (f"evidence:{name}",)))
    return FlowerCandidateEvaluation("candidate", tuple(measurements), 660, 660)


def test_unknown_contact_cannot_be_hidden_by_other_perfect_scores() -> None:
    evaluation = _evaluation(hand_flower_contact=GateVerdict.UNKNOWN)
    contract = FlowerAcceptanceContract.strict()

    assert not evaluation.accepted(contract)
    assert "hand_flower_contact" in evaluation.failed_gates(contract)


def test_user_preference_is_a_hard_gate() -> None:
    evaluation = _evaluation(full_video_human_preference=GateVerdict.FAIL)

    assert not evaluation.accepted(FlowerAcceptanceContract.strict())


def test_semantic_failure_switches_prompt_pipeline_to_explicit_geometry() -> None:
    decision = FlowerEvolutionAgent().propose(
        FlowerPipelineConfig(PipelineFamily.FULL_FRAME_GENERATIVE),
        _evaluation(
            robot_morphology=GateVerdict.FAIL,
            hand_flower_contact=GateVerdict.FAIL,
        ),
        FlowerAcceptanceContract.strict(),
    )

    assert decision.status == "PARTIAL"
    assert decision.next_config.family is PipelineFamily.HYBRID_3D_LAYERED
    assert decision.next_config.explicit_robot_geometry
    assert decision.next_config.robot_native_motion
    assert decision.next_config.contact_conditioning
    assert decision.next_config.depth_layering


def test_repeated_hybrid_semantic_failure_requires_paired_adapter() -> None:
    current = FlowerPipelineConfig(
        PipelineFamily.HYBRID_3D_LAYERED,
        action_segmentation=True,
        explicit_robot_geometry=True,
        robot_native_motion=True,
        contact_conditioning=True,
        depth_layering=True,
        local_generation_only=True,
    )
    decision = FlowerEvolutionAgent().propose(
        current,
        _evaluation(hand_flower_contact=GateVerdict.FAIL),
        FlowerAcceptanceContract.strict(),
        failure_counts={"hand_flower_contact": 2},
    )

    assert decision.training_required
    assert decision.next_config.family is PipelineFamily.ROBOT_CENTRIC_ADAPTED
    assert decision.next_config.paired_task_adapter


def test_temporal_only_failure_does_not_change_representation() -> None:
    current = FlowerPipelineConfig(PipelineFamily.LAYERED_2D)
    decision = FlowerEvolutionAgent().propose(
        current,
        _evaluation(temporal_consistency=GateVerdict.FAIL),
        FlowerAcceptanceContract.strict(),
    )

    assert decision.next_config.family is PipelineFamily.LAYERED_2D
    assert any("temporal neighborhoods" in action for action in decision.actions)


def test_all_hard_gates_and_all_frames_are_required_for_working() -> None:
    contract = FlowerAcceptanceContract.strict()
    complete = _evaluation()
    incomplete = FlowerCandidateEvaluation(
        "incomplete", complete.measurements, evaluated_frames=659, expected_frames=660
    )

    assert FlowerEvolutionAgent().propose(
        FlowerPipelineConfig(PipelineFamily.LAYERED_2D), complete, contract
    ).status == "WORKING"
    assert not incomplete.accepted(contract)


def test_coordinate_frames_are_explicit_and_not_interchangeable() -> None:
    with pytest.raises(ValueError, match="camera:source_pixels"):
        FlowerPipelineConfig(PipelineFamily.LAYERED_2D, source_frame="robot:base")


def test_pass_score_must_reach_threshold_even_with_pass_verdict() -> None:
    measurements = tuple(
        _measurement(name, 0.2 if name == "robot_morphology" else 1.0)
        for name in GATE_NAMES
    )
    evaluation = FlowerCandidateEvaluation("low-score", measurements, 660, 660)

    assert "robot_morphology" in evaluation.failed_gates(
        FlowerAcceptanceContract.strict()
    )
