from __future__ import annotations

import copy

import pytest

from phiagent.harness.task_reasoning import (
    OPTICAL_MODULE_TASK,
    PhysicalTaskReasoningPlugin,
    ReasoningPluginRegistry,
    TaskEntity,
    TaskReasoningPlan,
    TaskReasoningRequest,
)


def _request() -> TaskReasoningRequest:
    return TaskReasoningRequest(
        task_id="optical-module-insertion",
        task_type=OPTICAL_MODULE_TASK,
        instruction=(
            "夹爪夹住刚性尾部套环并保持模块贴桌，沿长边支点缓慢拨动翻面；反面落稳后松开"
            "重抓，再抬起、与卡槽平齐共轴并慢慢插入。"
        ),
        coordinate_frame="camera:optical_module_fit_blur_832x480",
        duration_seconds=10.0,
        entities=(
            TaskEntity("parallel_gripper", "manipulator", "black two-jaw gripper"),
            TaskEntity("optical_module", "manipulated_object", "silver optical transceiver"),
            TaskEntity("receptacle", "insertion_target", "fixed black receptacle"),
        ),
        available_evidence=("single RGB first frame", "named camera pixel frame"),
        unavailable_evidence=(
            "metric depth",
            "world-to-camera calibration",
            "force",
            "tactile",
            "electrical seating",
        ),
        user_constraints=(
            "rigid tail-collar grasp before flip",
            "table-edge support throughout the pivot",
            "opposite-face support before regrasp",
            "align level and coaxial before insertion",
            "insert slowly",
        ),
    )


def test_physical_task_plugin_expands_language_into_ordered_phases() -> None:
    plan = PhysicalTaskReasoningPlugin().analyze(_request())

    assert plan.language_analysis.source_language == "zh-CN"
    assert [phase.phase_id for phase in plan.phases] == [
        "observe_and_localize",
        "approach_tail_grasp_pose",
        "close_tail_collar_grasp",
        "establish_table_edge_pivot",
        "tail_arc_to_edge_on",
        "tail_arc_to_opposite_face",
        "settle_opposite_face_under_tail_control",
        "release_and_reposition_after_flip",
        "regrasp_flipped_metal_body",
        "lift_for_transport_clearance",
        "coarse_transport_to_standoff",
        "coaxial_preinsert_alignment",
        "slow_axial_insertion",
        "seated_hold_without_release",
    ]
    assert plan.phases[0].start_seconds == 0.0
    assert plan.phases[-1].end_seconds == 10.0
    assert all(
        left.end_seconds == right.start_seconds
        for left, right in zip(plan.phases, plan.phases[1:])
    )


def test_physical_task_plugin_separates_camera_relations_from_metric_physics() -> None:
    plan = PhysicalTaskReasoningPlugin().analyze(_request())

    assert all(phase.motion_frame.startswith("camera:") for phase in plan.phases)
    assert "does not provide calibrated angle" in plan.language_analysis.ambiguity_resolutions[0]
    assert any(
        finding.evidence_level == "UNAVAILABLE"
        and finding.finding_id == "seating_signal_unavailable"
        for finding in plan.physical_analysis
    )
    assert "not calibrated 3-D geometry" in plan.claim_boundary


def test_physical_task_plugin_requires_slow_insertion_after_alignment() -> None:
    plan = PhysicalTaskReasoningPlugin().analyze(_request())
    phases = {phase.phase_id: phase for phase in plan.phases}

    assert phases["tail_arc_to_edge_on"].speed_class == "slow"
    assert phases["tail_arc_to_opposite_face"].speed_class == "slow"
    assert phases["coarse_transport_to_standoff"].speed_class == "coarse"
    assert phases["coaxial_preinsert_alignment"].speed_class == "fine"
    assert phases["slow_axial_insertion"].speed_class == "slow"
    assert "35 percent" in phases["slow_axial_insertion"].language_directive
    assert (
        phases["coaxial_preinsert_alignment"].end_seconds
        == phases["slow_axial_insertion"].start_seconds
    )


def test_physical_task_plugin_requires_tail_driven_table_pivot() -> None:
    plan = PhysicalTaskReasoningPlugin().analyze(_request())
    phases = {phase.phase_id: phase for phase in plan.phases}
    establish = phases["establish_table_edge_pivot"].physical_contract
    first_roll = phases["tail_arc_to_edge_on"].physical_contract
    second_roll = phases["tail_arc_to_opposite_face"].physical_contract

    assert establish is not None
    assert first_roll is not None
    assert second_roll is not None
    assert (
        first_roll.rotation_start_degrees,
        first_roll.rotation_end_degrees,
    ) == (15.0, 90.0)
    assert (
        second_roll.rotation_start_degrees,
        second_roll.rotation_end_degrees,
    ) == (90.0, 180.0)
    assert (
        establish.rotation_start_degrees,
        establish.rotation_end_degrees,
    ) == (0.0, 15.0)
    assert first_roll.gripper_contact == "maintained_tail_collar_pivot"
    assert first_roll.module_support == "table_long_edge_a_tail_gripper"
    assert second_roll.module_support == "table_long_edge_b_tail_gripper"
    assert first_roll.requires_continuous_gripper_contact is True
    assert (
        phases["release_and_reposition_after_flip"].physical_contract.module_support
        == "tabletop_opposite_face"
    )


def test_physical_task_plugin_rejects_unreasonably_short_flip_insert() -> None:
    request = copy.deepcopy(_request())

    with pytest.raises(ValueError, match="at least 10 seconds"):
        PhysicalTaskReasoningPlugin().analyze(
            TaskReasoningRequest.from_dict(
                {**request.to_dict(), "duration_seconds": 8.0}
            )
        )


def test_physical_task_plan_hash_fails_closed_after_tampering() -> None:
    payload = PhysicalTaskReasoningPlugin().analyze(_request()).to_dict()
    tampered = copy.deepcopy(payload)
    tampered["phases"][4]["physical_contract"]["forbidden_module_motion"] = (
        "allow free flight",
        *tampered["phases"][4]["physical_contract"]["forbidden_module_motion"][1:],
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        TaskReasoningPlan.from_dict(tampered)


def test_reasoning_plugin_registry_exposes_builtin_plugin() -> None:
    registry = ReasoningPluginRegistry()

    plugin = registry.get("physical-task-language-planner")
    assert plugin.descriptor.deterministic is True
    assert plugin.descriptor.heavyweight is False
