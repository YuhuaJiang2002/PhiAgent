"""Dependency-light planning and validation plugins for generation harnesses."""

from .cloth_carrier import (
    TSHIRT_832X480_CARRIER,
    TshirtCarrierGeometry,
    phase_progress,
    polyline_segment_lengths,
    rigid_transform_points,
)
from .task_reasoning import (
    OPTICAL_MODULE_TASK,
    TSHIRT_FOLD_TASK,
    PhysicalTaskReasoningPlugin,
    ReasoningPluginRegistry,
    TaskEntity,
    TaskReasoningPlan,
    TaskReasoningRequest,
    TshirtFoldReasoningPlugin,
    validate_task_reasoning_plan,
)
from .test_time_scaling import (
    HardGateTestTimeScalingRepairAgent,
    ScalingRound,
    TestTimeScalingPolicy,
    compile_task_reasoning_prompt,
)

__all__ = (
    "TSHIRT_832X480_CARRIER",
    "TshirtCarrierGeometry",
    "phase_progress",
    "polyline_segment_lengths",
    "rigid_transform_points",
    "OPTICAL_MODULE_TASK",
    "TSHIRT_FOLD_TASK",
    "PhysicalTaskReasoningPlugin",
    "ReasoningPluginRegistry",
    "TaskEntity",
    "TaskReasoningPlan",
    "TaskReasoningRequest",
    "TshirtFoldReasoningPlugin",
    "validate_task_reasoning_plan",
    "HardGateTestTimeScalingRepairAgent",
    "ScalingRound",
    "TestTimeScalingPolicy",
    "compile_task_reasoning_prompt",
)
