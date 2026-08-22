from __future__ import annotations

import copy

import pytest

from phiagent.harness.task_reasoning import (
    TSHIRT_FOLD_TASK,
    ReasoningPluginRegistry,
    TaskEntity,
    TaskReasoningPlan,
    TaskReasoningRequest,
)
from phiagent.harness.tshirt_fold_strategy import (
    RIGHT_THEN_LEFT,
    SIMULTANEOUS,
    VIEWER_RIGHT,
    TshirtFoldStrategy,
    TshirtFoldStrategyReasoningPlugin,
    all_tshirt_fold_strategies,
    strategy_from_plan,
)


def _request(instruction: str = "生成多种物理一致的双臂叠衣服候选。") -> TaskReasoningRequest:
    return TaskReasoningRequest(
        task_id="two-arm-tshirt-fold",
        task_type=TSHIRT_FOLD_TASK,
        instruction=instruction,
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
            "sleeve length cannot change",
            "cloth motion requires contact",
            "candidate strategies must stay distinct",
        ),
    )


def test_strategy_plugin_expands_full_six_way_candidate_matrix() -> None:
    plugin = TshirtFoldStrategyReasoningPlugin()
    plans = plugin.analyze_candidates(_request())

    assert len(plans) == 6
    assert len({plan.task_id for plan in plans}) == 6
    assert len({plan.plan_sha256 for plan in plans}) == 6
    assert {strategy_from_plan(plan) for plan in plans} == set(
        all_tshirt_fold_strategies()
    )
    assert all(plan.plugin.name == plugin.descriptor.name for plan in plans)


def test_strategy_plugin_is_available_from_builtin_registry() -> None:
    plugin = ReasoningPluginRegistry().get("tshirt-fold-strategy-language-planner")

    assert isinstance(plugin, TshirtFoldStrategyReasoningPlugin)
    assert plugin.descriptor.heavyweight is False


def test_right_first_place_right_plan_binds_order_direction_and_gates() -> None:
    strategy = TshirtFoldStrategy(RIGHT_THEN_LEFT, VIEWER_RIGHT)
    plan = TshirtFoldStrategyReasoningPlugin().analyze_strategy(_request(), strategy)
    phase_ids = [phase.phase_id for phase in plan.phases]
    gate_ids = {gate.gate_id for gate in plan.verification_gates}

    assert phase_ids.index("fold_viewer_right_sleeve") < phase_ids.index(
        "fold_viewer_left_sleeve"
    )
    assert "move_folded_bundle_viewer_right" in phase_ids
    assert "viewer_right_fold_precedes_viewer_left_fold" in gate_ids
    assert "viewer_left_fold_precedes_viewer_right_fold" not in gate_ids
    assert "viewer-right" in plan.language_analysis.normalized_instruction
    assert strategy_from_plan(plan) == strategy


def test_simultaneous_plan_uses_two_cuffs_and_table_support_without_extra_hands() -> None:
    strategy = TshirtFoldStrategy(SIMULTANEOUS, VIEWER_RIGHT)
    plan = TshirtFoldStrategyReasoningPlugin().analyze_strategy(_request(), strategy)
    phases = {phase.phase_id: phase for phase in plan.phases}
    gate_ids = {gate.gate_id for gate in plan.verification_gates}

    contact = phases["establish_bilateral_cuff_contacts"]
    fold = phases["fold_both_sleeves_synchronously"]
    assert "torso and both shoulder seams remain table-supported" in contact.language_directive
    assert "do not invent extra hands" in contact.language_directive
    assert "both_sleeves_fold_synchronously" in fold.gate_ids
    assert "both_sleeves_fold_synchronously" in gate_ids
    assert "viewer_left_fold_precedes_viewer_right_fold" not in gate_ids
    assert strategy_from_plan(plan) == strategy


def test_strategy_plan_hash_fails_closed_after_direction_tampering() -> None:
    plan = TshirtFoldStrategyReasoningPlugin().analyze_strategy(
        _request(),
        TshirtFoldStrategy(RIGHT_THEN_LEFT, VIEWER_RIGHT),
    )
    payload = copy.deepcopy(plan.to_dict())
    payload["phases"][-2]["language_directive"] = payload["phases"][-2][
        "language_directive"
    ].replace("viewer-right", "viewer-left")

    with pytest.raises(ValueError, match="hash mismatch"):
        TaskReasoningPlan.from_dict(payload)


def test_plugin_infers_explicit_simultaneous_right_placement_instruction() -> None:
    plan = TshirtFoldStrategyReasoningPlugin().analyze(
        _request("两边袖子一起叠，叠好之后放到右边。")
    )

    assert strategy_from_plan(plan) == TshirtFoldStrategy(SIMULTANEOUS, VIEWER_RIGHT)


def test_strategy_plugin_rejects_duration_too_short_for_continuous_fold() -> None:
    payload = _request().to_dict()
    payload["duration_seconds"] = 3.5

    with pytest.raises(ValueError, match="at least 4 seconds"):
        TshirtFoldStrategyReasoningPlugin().analyze_candidates(
            TaskReasoningRequest.from_dict(payload)
        )
