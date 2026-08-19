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
            "夹爪先往下移动到光模块的抓取高度，夹住后抬起；移动到卡槽前先与卡槽平齐并共轴，"
            "再慢慢插入。"
        ),
        coordinate_frame="camera:optical_module_fit_blur_832x480",
        duration_seconds=5.0,
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
            "descend before grasp",
            "align level and coaxial before insertion",
            "insert slowly",
        ),
    )


def test_physical_task_plugin_expands_language_into_ordered_phases() -> None:
    plan = PhysicalTaskReasoningPlugin().analyze(_request())

    assert plan.language_analysis.source_language == "zh-CN"
    assert [phase.phase_id for phase in plan.phases] == [
        "observe_and_localize",
        "descend_to_grasp_height",
        "close_on_metal_body",
        "lift_for_clearance",
        "coarse_transport_to_standoff",
        "coaxial_preinsert_alignment",
        "slow_axial_insertion",
        "seated_hold_without_release",
    ]
    assert plan.phases[0].start_seconds == 0.0
    assert plan.phases[-1].end_seconds == 5.0
    assert all(
        left.end_seconds == right.start_seconds
        for left, right in zip(plan.phases, plan.phases[1:])
    )


def test_physical_task_plugin_separates_camera_relations_from_metric_physics() -> None:
    plan = PhysicalTaskReasoningPlugin().analyze(_request())

    assert all(phase.motion_frame.startswith("camera:") for phase in plan.phases)
    assert "not a calibrated world-z" in plan.language_analysis.ambiguity_resolutions[0]
    assert any(
        finding.evidence_level == "UNAVAILABLE"
        and finding.finding_id == "seating_signal_unavailable"
        for finding in plan.physical_analysis
    )
    assert "not calibrated 3-D geometry" in plan.claim_boundary


def test_physical_task_plugin_requires_slow_insertion_after_alignment() -> None:
    plan = PhysicalTaskReasoningPlugin().analyze(_request())
    phases = {phase.phase_id: phase for phase in plan.phases}

    assert phases["coarse_transport_to_standoff"].speed_class == "coarse"
    assert phases["coaxial_preinsert_alignment"].speed_class == "fine"
    assert phases["slow_axial_insertion"].speed_class == "slow"
    assert "35 percent" in phases["slow_axial_insertion"].language_directive
    assert (
        phases["coaxial_preinsert_alignment"].end_seconds
        == phases["slow_axial_insertion"].start_seconds
    )


def test_physical_task_plan_hash_fails_closed_after_tampering() -> None:
    payload = PhysicalTaskReasoningPlugin().analyze(_request()).to_dict()
    tampered = copy.deepcopy(payload)
    tampered["phases"][1]["language_directive"] = "close immediately"

    with pytest.raises(ValueError, match="hash mismatch"):
        TaskReasoningPlan.from_dict(tampered)


def test_reasoning_plugin_registry_exposes_builtin_plugin() -> None:
    registry = ReasoningPluginRegistry()

    plugin = registry.get("physical-task-language-planner")
    assert plugin.descriptor.deterministic is True
    assert plugin.descriptor.heavyweight is False
