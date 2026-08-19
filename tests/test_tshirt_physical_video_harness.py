from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from phiagent.acwm.adapters import MiniMaxH3Config, MiniMaxH3Renderer
from phiagent.acwm.schema import (
    ACWMActionCondition,
    ACWMCase,
    ActionRepresentation,
)
from phiagent.agent.acwm import ACWMScorecard, ACWMThresholds
from phiagent.evaluation.tshirt_fold_video import (
    FrameWindow,
    TrackedMaterialFrame,
    TshirtFoldTrackingContract,
    TshirtFoldTrackingThresholds,
    score_tshirt_fold_tracks,
)
from phiagent.harness.task_reasoning import (
    TSHIRT_FOLD_TASK,
    TaskEntity,
    TaskReasoningRequest,
    TshirtFoldReasoningPlugin,
)
from phiagent.harness.cloth_carrier import (
    TSHIRT_832X480_CARRIER,
    polyline_segment_lengths,
)
from phiagent.harness.articulated_camera_rig import (
    LOWER_LEFT_RIG,
    UPPER_RIGHT_RIG,
    compile_tshirt_dual_arm_trajectory,
)
from phiagent.harness.test_time_scaling import (
    HardGateTestTimeScalingRepairAgent,
    ScalingRound,
    TestTimeScalingPolicy,
    compile_task_reasoning_prompt,
    initial_scaled_proposals,
)
from scripts.run_acwm_backend import _run_minimax_h3


def _request() -> TaskReasoningRequest:
    return TaskReasoningRequest(
        task_id="two-arm-tshirt-fold",
        task_type=TSHIRT_FOLD_TASK,
        instruction="先叠画面左边袖子，再叠画面右边袖子，最后把衣服叠好放到左边。",
        coordinate_frame="camera:tshirt_fold_832x480_pixels",
        duration_seconds=5.166667,
        entities=(
            TaskEntity("lower_left_robot", "manipulator", "lower-left white robot arm"),
            TaskEntity("upper_right_robot", "manipulator", "upper-right white robot arm"),
            TaskEntity("viewer_left_sleeve", "cloth_part", "viewer-left black sleeve"),
            TaskEntity("viewer_right_sleeve", "cloth_part", "viewer-right black sleeve"),
            TaskEntity("shirt_body", "cloth_body", "gray T-shirt torso"),
        ),
        available_evidence=("single RGB first frame", "named camera pixel frame"),
        unavailable_evidence=("metric depth", "cloth mesh", "force", "joint trajectory"),
        user_constraints=(
            "viewer-left sleeve first",
            "viewer-right sleeve second",
            "sleeve length cannot change",
            "move completed fold aside",
        ),
    )


def test_tshirt_planner_expands_causal_phases_and_material_gates() -> None:
    plan = TshirtFoldReasoningPlugin().analyze(_request())

    assert [phase.phase_id for phase in plan.phases] == [
        "initial_state_hold",
        "establish_viewer_left_two_point_contact",
        "fold_viewer_left_sleeve",
        "settle_viewer_left_sleeve",
        "establish_viewer_right_two_point_contact",
        "fold_viewer_right_sleeve",
        "fold_body_bottom_to_top",
        "compress_bundle_without_stretch",
        "move_folded_bundle_viewer_left",
        "terminal_bundle_hold",
    ]
    gate_ids = {gate.gate_id for gate in plan.verification_gates}
    assert "viewer_left_sleeve_length_conserved" in gate_ids
    assert "viewer_right_sleeve_length_conserved" in gate_ids
    assert "no_teleportation_or_crossfade" in gate_ids
    assert "viewer_left_fold_precedes_viewer_right_fold" in gate_ids
    assert all(phase.motion_frame == _request().coordinate_frame for phase in plan.phases)


def test_compiled_prompt_contains_hash_timeline_and_nonnegotiable_gates() -> None:
    plan = TshirtFoldReasoningPlugin().analyze(_request())
    prompt = compile_task_reasoning_prompt(plan)

    assert plan.plan_sha256 in prompt
    assert "Do not use cuts, dissolves, crossfades" in prompt
    assert "viewer_left_sleeve_length_conserved" in prompt
    assert "fold_viewer_right_sleeve" in prompt


class _Renderer:
    def supports(self, _case):
        return SimpleNamespace(supported=True)


def test_initial_scaling_allocates_diverse_seeds_and_more_than_baseline_steps() -> None:
    policy = TestTimeScalingPolicy(
        rounds=(ScalingRound(3, 28, 10, "diverse search"),),
        maximum_candidates=3,
    )
    proposals = initial_scaled_proposals(
        (SimpleNamespace(case_id="fold"),),
        {"minimax-h3": _Renderer()},
        policy=policy,
        base_seed=100,
    )

    assert [item.seed for item in proposals] == [110, 1119, 2128]
    assert {item.num_inference_steps for item in proposals} == {28}


def test_h3_accepts_camera_control_and_per_candidate_scaled_steps(tmp_path) -> None:
    first_frame = tmp_path / "first.png"
    source_video = tmp_path / "source.mp4"
    embodiment = tmp_path / "embodiment.png"
    for path in (first_frame, source_video, embodiment):
        path.write_bytes(b"non-empty-test-fixture")
    timestamps = tuple(index / 24.0 for index in range(124))
    condition = ACWMActionCondition(
        label="fold-shirt",
        instruction="fold continuously",
        timeline="124 frames at 24 FPS",
        representation=ActionRepresentation.CAMERA_PIXEL_CONTROL_VIDEO,
        coordinate_frame="camera:tshirt_fold_832x480_pixels",
        timestamps_s=timestamps,
        channels=("phase",),
        values=tuple((float(index),) for index in range(124)),
        visual_condition=source_video,
    )
    case = ACWMCase(
        case_id="fold-shirt",
        first_frame=first_frame,
        source_video=source_video,
        action=condition,
        prompt="fold left sleeve, right sleeve, then body",
        auxiliary_inputs=(("embodiment_reference", embodiment),),
    )
    renderer = MiniMaxH3Renderer(
        MiniMaxH3Config(
            repository=tmp_path / "h3",
            model_base_path=tmp_path / "models",
        )
    )

    assert renderer.supports(case).supported is True
    source = inspect.getsource(_run_minimax_h3)
    assert 'num_inference_steps=int(item["num_inference_steps"])' in source
    assert "video[0] = exact_first_frame" in source
    assert "must equal the frozen H3 adapter setting" not in source


def test_carrier_rigid_sleeve_phases_preserve_every_material_segment() -> None:
    geometry = TSHIRT_832X480_CARRIER
    baseline = {
        "viewer_left": polyline_segment_lengths(geometry.viewer_left_material),
        "viewer_right": polyline_segment_lengths(geometry.viewer_right_material),
    }

    for frame in range(124):
        transformed = geometry.sleeve_material_at(frame)
        for side, points in transformed.items():
            assert polyline_segment_lengths(points) == pytest.approx(
                baseline[side], abs=1e-9
            )


def test_articulated_contact_carrier_is_connected_and_contact_first() -> None:
    trajectory = compile_tshirt_dual_arm_trajectory()

    assert trajectory.maximum_link_length_error_pixels < 1e-6
    assert trajectory.maximum_tip_step_pixels < 18.0
    assert trajectory.maximum_joint_step_radians < 0.20
    assert trajectory.mean_tip_error_pixels < 1e-4
    for name, rig in (
        ("lower_left", LOWER_LEFT_RIG),
        ("upper_right", UPPER_RIGHT_RIG),
    ):
        for frame in trajectory.frames[name]:
            observed = tuple(
                ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
                for a, b in zip(frame.nodes_xy, frame.nodes_xy[1:])
            )
            assert observed == pytest.approx(rig.link_lengths_pixels, abs=1e-6)
    assert trajectory.frames["lower_left"][20].contact_entity == "viewer_left_sleeve"
    assert trajectory.frames["upper_right"][60].contact_entity == "viewer_right_sleeve"
    assert trajectory.frames["lower_left"][19].contact_entity is None
    assert trajectory.frames["upper_right"][59].contact_entity is None


def test_scaling_policy_rejects_threshold_relaxation_and_decreasing_steps() -> None:
    with pytest.raises(ValueError, match="cannot relax"):
        TestTimeScalingPolicy(
            rounds=(ScalingRound(1, 24, 0, "probe"),),
            maximum_candidates=1,
            threshold_policy="mean_score_override",
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        TestTimeScalingPolicy(
            rounds=(
                ScalingRound(1, 32, 0, "first"),
                ScalingRound(1, 24, 100, "second"),
            ),
            maximum_candidates=2,
        )


def test_repair_scaling_uses_failed_hard_gate_without_changing_thresholds() -> None:
    policy = TestTimeScalingPolicy(
        rounds=(
            ScalingRound(1, 24, 0, "probe"),
            ScalingRound(1, 36, 100, "repair"),
        ),
        maximum_candidates=2,
    )
    scorecard = ACWMScorecard(
        evaluator="test",
        action_adherence=0.99,
        embodiment_consistency=0.99,
        object_interaction=0.99,
        temporal_consistency=0.99,
        background_consistency=0.99,
        human_review_passed=None,
        hard_gates_passed=False,
        diagnoses=("hard_gate:viewer_left_sleeve_length_conserved",),
    )
    history = (
        SimpleNamespace(
            round_index=0,
            proposal=SimpleNamespace(case_id="fold"),
            scorecard=scorecard,
        ),
    )
    thresholds = ACWMThresholds()
    proposals = HardGateTestTimeScalingRepairAgent(policy, base_seed=10).propose(
        cases={"fold": SimpleNamespace(case_id="fold")},
        renderers={"minimax-h3": _Renderer()},
        history=history,
        thresholds=thresholds,
    )

    assert len(proposals) == 1
    assert proposals[0].num_inference_steps == 36
    assert "never by shrinking" in proposals[0].prompt_suffix
    assert thresholds == ACWMThresholds()


def _contract() -> TshirtFoldTrackingContract:
    return TshirtFoldTrackingContract(
        coordinate_frame="camera:tshirt_fold_832x480_pixels",
        frame_count=12,
        viewer_left_sleeve_xy=((0.0, 0.0), (5.0, 0.0), (10.0, 0.0)),
        viewer_right_sleeve_xy=((30.0, 0.0), (35.0, 0.0), (40.0, 0.0)),
        body_points_xy=((0.0, 10.0), (20.0, 30.0), (40.0, 10.0)),
        background_rectangles_xywh=((0, 0, 4, 4),),
        left_fold=FrameWindow(1, 3),
        right_fold=FrameWindow(3, 5),
        body_fold=FrameWindow(5, 7),
        bundle_move=FrameWindow(7, 11),
        lower_left_gripper_xy=((-5.0, 0.0), (-2.0, 0.0)),
        upper_right_gripper_xy=((42.0, 0.0), (45.0, 0.0)),
        thresholds=TshirtFoldTrackingThresholds(
            minimum_motion_pixels=2.0,
            maximum_material_step_pixels=16.0,
            maximum_terminal_bbox_area_ratio=0.62,
            minimum_bundle_move_left_pixels=24.0,
        ),
    )


def _passing_tracks(*, shorten_left: bool = False, right_early: bool = False):
    frames = []
    for index in range(12):
        left_shift = 8.0 if index >= 1 else 0.0
        right_shift = -8.0 if index >= (1 if right_early else 3) else 0.0
        body = (
            ((8.0, 8.0), (20.0, 14.0), (32.0, 8.0))
            if index >= 5
            else ((0.0, 10.0), (20.0, 30.0), (40.0, 10.0))
        )
        bundle_shift = -12.0 * min(max(index - 7, 0), 2)
        left_scale = 0.5 if shorten_left and index >= 2 else 1.0
        frames.append(
            TrackedMaterialFrame(
                frame_index=index,
                viewer_left_sleeve_xy=tuple(
                    (left_shift + bundle_shift + value * left_scale, 0.0)
                    for value in (0.0, 5.0, 10.0)
                ),
                viewer_right_sleeve_xy=tuple(
                    (30.0 + right_shift + bundle_shift + value, 0.0)
                    for value in (0.0, 5.0, 10.0)
                ),
                body_points_xy=tuple((x + bundle_shift, y) for x, y in body),
                confidence=1.0,
                lower_left_gripper_xy=tuple(
                    (x + bundle_shift, y) for x, y in ((-5.0, 0.0), (-2.0, 0.0))
                ),
                upper_right_gripper_xy=tuple(
                    (x + bundle_shift, y) for x, y in ((42.0, 0.0), (45.0, 0.0))
                ),
                manipulator_confidence=1.0,
            )
        )
    return tuple(frames)


def test_track_gates_pass_continuous_length_preserving_ordered_fold() -> None:
    score = score_tshirt_fold_tracks(
        _passing_tracks(),
        contract=_contract(),
        first_frame_score=1.0,
        background_score=1.0,
    )

    assert score.hard_gates_passed
    assert score.motion_onsets == {
        "viewer_left_sleeve": 1,
        "viewer_right_sleeve": 3,
        "body": 5,
    }


def test_track_gates_fail_closed_on_sleeve_shrink_and_wrong_order() -> None:
    shortened = score_tshirt_fold_tracks(
        _passing_tracks(shorten_left=True),
        contract=_contract(),
        first_frame_score=1.0,
        background_score=1.0,
    )
    wrong_order = score_tshirt_fold_tracks(
        _passing_tracks(right_early=True),
        contract=_contract(),
        first_frame_score=1.0,
        background_score=1.0,
    )

    assert "viewer_left_sleeve_length_conserved" in shortened.failed_gates
    assert "viewer_left_fold_precedes_viewer_right_fold" in wrong_order.failed_gates


def test_track_gates_fail_closed_when_cloth_moves_without_gripper_contact() -> None:
    frames = tuple(
        TrackedMaterialFrame(
            frame_index=frame.frame_index,
            viewer_left_sleeve_xy=frame.viewer_left_sleeve_xy,
            viewer_right_sleeve_xy=frame.viewer_right_sleeve_xy,
            body_points_xy=frame.body_points_xy,
            confidence=frame.confidence,
            lower_left_gripper_xy=((100.0, 100.0), (105.0, 100.0)),
            upper_right_gripper_xy=((120.0, 100.0), (125.0, 100.0)),
            manipulator_confidence=1.0,
        )
        for frame in _passing_tracks()
    )

    score = score_tshirt_fold_tracks(
        frames,
        contract=_contract(),
        first_frame_score=1.0,
        background_score=1.0,
    )

    assert "contact_precedes_cloth_motion" in score.failed_gates
