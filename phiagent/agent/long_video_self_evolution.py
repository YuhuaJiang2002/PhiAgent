"""Fail-closed decisions for an unattended long-video improvement loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrozenVisualContract:
    maximum_stage_wall_seconds: float
    maximum_self_flow_mean: float
    maximum_self_flow_p95: float
    maximum_self_flow_high_count: int
    maximum_source_flow_mean: float
    maximum_source_flow_p95: float
    maximum_source_flow_high_count: int
    maximum_wrong_occlusion_mean: float
    maximum_wrong_occlusion_p95: float
    maximum_owner_flip_mean: float
    maximum_owner_flip_p95: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrozenVisualContract":
        return cls(**payload)


def evaluate_visual_iteration(
    *,
    repair_manifest: dict[str, Any],
    specialty_report: dict[str, Any],
    full_report: dict[str, Any],
    contract: FrozenVisualContract,
) -> dict[str, Any]:
    """Return a conjunctive promotion decision without changing any gate."""

    specialty_decision = evaluate_specialty_contract(
        repair_manifest=repair_manifest,
        specialty_report=specialty_report,
        contract=contract,
    )
    candidates = full_report.get("candidates", [])
    if len(candidates) != 1:
        raise ValueError("full audit must contain exactly one challenger")
    full_candidate = candidates[0]
    full_gates = full_candidate["summary"]["gates"]
    adversarial_gates = full_candidate["adversarial"]["gates"]
    checks = {
        **specialty_decision["checks"],
        "all_full_video_gates": all(bool(value) for value in full_gates.values()),
        "all_adversarial_attacks_detected": all(
            bool(value) for value in adversarial_gates.values()
        ),
    }
    automatic_pass = all(checks.values())
    return {
        "automatic_pass": automatic_pass,
        "status": "AWAITING_HIGH_RESOLUTION_REVIEW" if automatic_pass else "REJECTED",
        "checks": checks,
        "failed_checks": sorted(name for name, value in checks.items() if not value),
        "thresholds_frozen": True,
        "physical_evidence": False,
        "claim_scope": "perceptually plausible synthetic long-video data",
    }


def evaluate_specialty_contract(
    *,
    repair_manifest: dict[str, Any],
    specialty_report: dict[str, Any],
    contract: FrozenVisualContract,
) -> dict[str, Any]:
    """Reject a challenger before the expensive full audit when possible."""

    specialty = specialty_report["metrics"]["challenger"]
    high_counts = specialty_report["metrics"]["high_flicker_counts"]["challenger"]
    checks = {
        "repair_internal_gates": all(repair_manifest["gates"].values()),
        "stage_cost_bound": (
            float(repair_manifest["metrics"]["wall_seconds"])
            <= contract.maximum_stage_wall_seconds
        ),
        "self_flow_mean_non_regression": (
            float(specialty["self_flow_arm_mae"]["mean"])
            <= contract.maximum_self_flow_mean
        ),
        "self_flow_p95_non_regression": (
            float(specialty["self_flow_arm_mae"]["p95"])
            <= contract.maximum_self_flow_p95
        ),
        "self_flow_high_count_non_regression": (
            int(high_counts["self_flow_arm_mae"])
            <= contract.maximum_self_flow_high_count
        ),
        "source_flow_mean_non_regression": (
            float(specialty["source_flow_residual_mae"]["mean"])
            <= contract.maximum_source_flow_mean
        ),
        "source_flow_p95_non_regression": (
            float(specialty["source_flow_residual_mae"]["p95"])
            <= contract.maximum_source_flow_p95
        ),
        "source_flow_high_count_non_regression": (
            int(high_counts["source_flow_residual_mae"])
            <= contract.maximum_source_flow_high_count
        ),
        "wrong_occlusion_mean_non_regression": (
            float(specialty["wrong_flower_occlusion_fraction"]["mean"])
            <= contract.maximum_wrong_occlusion_mean
        ),
        "wrong_occlusion_p95_non_regression": (
            float(specialty["wrong_flower_occlusion_fraction"]["p95"])
            <= contract.maximum_wrong_occlusion_p95
        ),
        "owner_flip_mean_non_regression": (
            float(specialty["flower_owner_flip_fraction"]["mean"])
            <= contract.maximum_owner_flip_mean
        ),
        "owner_flip_p95_non_regression": (
            float(specialty["flower_owner_flip_fraction"]["p95"])
            <= contract.maximum_owner_flip_p95
        ),
    }
    return {
        "automatic_pass": all(checks.values()),
        "checks": checks,
        "failed_checks": sorted(name for name, value in checks.items() if not value),
        "thresholds_frozen": True,
    }
