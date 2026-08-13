"""Fail-closed delivery routing for H3 identity candidates.

Both a learned candidate and any fallback must pass the same task-bound
promotion assessment.  Historical booleans from another clip or control
context are deliberately insufficient for delivery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ACTION_CONTEXT_KEYS = (
    "source",
    "motion_reference",
    "robot_reference",
    "anchor_mask",
)
METRIC_CONTEXT_KEYS = (
    "baseline",
    "baseline_action_evaluation",
    "identity_mask",
    "reference_image",
    "scene_image",
)
REQUIRED_PROMOTION_GATES = (
    "identity_gain",
    "identity_floor",
    "topology_evidence",
    "topology_full_frame_coverage",
    "topology_decoded_frame_digests",
    "topology_kinematic_detail",
    "topology",
    "topology_metric_matches_evidence",
    "motion_non_regression",
    "action_non_regression",
    "scene_non_regression",
    "temporal_non_regression",
)


@dataclass(frozen=True)
class IdentityDeliveryDecision:
    route: str
    reason: str
    failed_candidate_gates: tuple[str, ...]
    failed_fallback_gates: tuple[str, ...]

    @property
    def deliverable(self) -> bool:
        return self.route in {"candidate", "task_bound_fallback"}


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _assessment_state(
    payload: Mapping[str, object], name: str
) -> tuple[bool, tuple[str, ...]]:
    assessment = _mapping(payload.get("assessment"), f"{name}.assessment")
    gates = _mapping(assessment.get("gates"), f"{name}.assessment.gates")
    if not gates or any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError(f"{name}.assessment.gates must be non-empty booleans")
    missing_gates = [gate for gate in REQUIRED_PROMOTION_GATES if gate not in gates]
    if missing_gates:
        raise ValueError(
            f"{name}.assessment.gates is missing required gates: {missing_gates}"
        )
    raw_failed = assessment.get("failed_gates")
    if not isinstance(raw_failed, list) or any(
        not isinstance(gate, str) for gate in raw_failed
    ):
        raise ValueError(f"{name}.assessment.failed_gates must be a string list")
    failed = list(raw_failed)
    for gate in REQUIRED_PROMOTION_GATES:
        if gates[gate] is False and gate not in failed:
            failed.append(gate)
    if payload.get("status") != "accepted":
        failed.append("assessment_status")
    if payload.get("honest_status") != "WORKING":
        failed.append("assessment_honest_status")
    if assessment.get("passed") is not True:
        failed.append("assessment_passed")
    unique_failed = tuple(dict.fromkeys(failed))
    return not unique_failed, unique_failed


def metric_delivery_context(payload: Mapping[str, object]) -> dict[str, str]:
    """Extract hashes that define the exact baseline and action-control task."""

    inputs = _mapping(payload.get("inputs"), "metrics.inputs")
    action_evidence = _mapping(payload.get("action_evidence"), "metrics.action_evidence")
    action_context = _mapping(
        action_evidence.get("matched_context_sha256"),
        "metrics.action_evidence.matched_context_sha256",
    )
    context: dict[str, str] = {}
    for key in ACTION_CONTEXT_KEYS:
        value = action_context.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"metrics action context is missing SHA-256 for {key}")
        context[f"action:{key}"] = value
    for key in METRIC_CONTEXT_KEYS:
        entry = _mapping(inputs.get(key), f"metrics.inputs.{key}")
        value = entry.get("sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"metrics inputs are missing SHA-256 for {key}")
        context[f"metric:{key}"] = value
    return context


def require_matched_delivery_context(
    candidate_metrics: Mapping[str, object],
    fallback_metrics: Mapping[str, object],
) -> dict[str, str]:
    candidate_context = metric_delivery_context(candidate_metrics)
    fallback_context = metric_delivery_context(fallback_metrics)
    mismatched = {
        key: (candidate_context[key], fallback_context.get(key))
        for key in candidate_context
        if candidate_context[key] != fallback_context.get(key)
    }
    if mismatched:
        raise ValueError(f"fallback metrics use a different delivery context: {mismatched}")
    return candidate_context


def decide_identity_delivery(
    candidate_assessment: Mapping[str, object],
    fallback_assessment: Mapping[str, object],
) -> IdentityDeliveryDecision:
    """Choose a task-bound candidate/fallback or block without an output."""

    candidate_passed, candidate_failed = _assessment_state(
        candidate_assessment, "candidate"
    )
    fallback_passed, fallback_failed = _assessment_state(
        fallback_assessment, "fallback"
    )
    if candidate_passed:
        return IdentityDeliveryDecision(
            route="candidate",
            reason="learned candidate passed every task-bound promotion gate",
            failed_candidate_gates=(),
            failed_fallback_gates=fallback_failed,
        )
    if fallback_passed:
        return IdentityDeliveryDecision(
            route="task_bound_fallback",
            reason="candidate rejected; fallback passed the same task-bound assessment",
            failed_candidate_gates=candidate_failed,
            failed_fallback_gates=(),
        )
    return IdentityDeliveryDecision(
        route="blocked",
        reason="candidate and fallback both fail the current task-bound assessment",
        failed_candidate_gates=candidate_failed,
        failed_fallback_gates=fallback_failed,
    )
