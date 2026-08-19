"""Dependency-light planning and validation plugins for generation harnesses."""

from .cloth_carrier import (
    TSHIRT_832X480_CARRIER,
    TshirtCarrierGeometry,
    phase_progress,
    polyline_segment_lengths,
    rigid_transform_points,
)
from .articulated_camera_rig import (
    LOWER_LEFT_RIG,
    UPPER_RIGHT_RIG,
    DualArmCarrierTrajectory,
    PlanarArmRig,
    PlanarRigFrame,
    compile_tshirt_dual_arm_trajectory,
    solve_fabrik,
    tshirt_gripper_targets,
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
    "LOWER_LEFT_RIG",
    "UPPER_RIGHT_RIG",
    "DualArmCarrierTrajectory",
    "PlanarArmRig",
    "PlanarRigFrame",
    "compile_tshirt_dual_arm_trajectory",
    "solve_fabrik",
    "tshirt_gripper_targets",
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
