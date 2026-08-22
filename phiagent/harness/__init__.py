"""Dependency-light physical video planning and reference-conditioning interfaces."""

from .task_reasoning import (
    OPTICAL_MODULE_TASK,
    TSHIRT_FOLD_TASK,
    PhysicalTaskReasoningPlugin,
    ReasoningPluginRegistry,
    TaskEntity,
    TaskReasoningPlan,
    TaskReasoningRequest,
    TshirtFoldReasoningPlugin,
    validate_task_reasoning_human_review,
    validate_task_reasoning_plan,
)
from .tshirt_fold_strategy import (
    LEFT_THEN_RIGHT,
    RIGHT_THEN_LEFT,
    SIMULTANEOUS,
    VIEWER_LEFT,
    VIEWER_RIGHT,
    TshirtFoldStrategy,
    TshirtFoldStrategyReasoningPlugin,
    all_tshirt_fold_strategies,
    strategy_from_plan,
)
from .tshirt_positive_reference import (
    PositiveFoldReference,
    PositiveReferenceBank,
    ReferenceConditioningPlan,
    compile_reference_conditioning,
    compile_reference_conditioning_batch,
    load_positive_reference_bank,
)


__all__ = (
    "LEFT_THEN_RIGHT",
    "OPTICAL_MODULE_TASK",
    "PhysicalTaskReasoningPlugin",
    "PositiveFoldReference",
    "PositiveReferenceBank",
    "RIGHT_THEN_LEFT",
    "ReasoningPluginRegistry",
    "ReferenceConditioningPlan",
    "SIMULTANEOUS",
    "TSHIRT_FOLD_TASK",
    "TaskEntity",
    "TaskReasoningPlan",
    "TaskReasoningRequest",
    "TshirtFoldReasoningPlugin",
    "TshirtFoldStrategy",
    "TshirtFoldStrategyReasoningPlugin",
    "VIEWER_LEFT",
    "VIEWER_RIGHT",
    "all_tshirt_fold_strategies",
    "compile_reference_conditioning",
    "compile_reference_conditioning_batch",
    "load_positive_reference_bank",
    "strategy_from_plan",
    "validate_task_reasoning_human_review",
    "validate_task_reasoning_plan",
)
