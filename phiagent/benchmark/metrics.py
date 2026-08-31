"""L1--L5 evaluation and non-cherry-picked suite aggregation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from phiagent.benchmark.schema import (
    DIMENSIONS,
    BenchmarkCase,
    BenchmarkSuite,
    ScalarEvidence,
    Submission,
    SubmissionRecord,
)


@dataclass(frozen=True)
class MetricRule:
    threshold: float
    direction: str
    unit: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.direction not in {"max", "min"}:
            raise ValueError("metric rules require a finite threshold and max/min direction")

    def passes(self, value: float) -> bool:
        return value <= self.threshold if self.direction == "max" else value >= self.threshold


@dataclass(frozen=True)
class BenchmarkPolicy:
    geometry_rules: dict[str, MetricRule] = field(
        default_factory=lambda: {
            "camera_ate_m": MetricRule(0.05, "max", "m"),
            "camera_rpe_translation_m": MetricRule(0.02, "max", "m"),
            "camera_rpe_rotation_deg": MetricRule(5.0, "max", "deg"),
            "depth_abs_rel": MetricRule(0.15, "max", "ratio"),
            "reprojection_rmse_px": MetricRule(8.0, "max", "px"),
            "hand_mpjpe_m": MetricRule(0.03, "max", "m"),
            "object_translation_error_m": MetricRule(0.03, "max", "m"),
            "object_rotation_error_deg": MetricRule(10.0, "max", "deg"),
        }
    )
    action_rules: dict[str, MetricRule] = field(
        default_factory=lambda: {
            "eef_position_rmse_m": MetricRule(0.01, "max", "m"),
            "eef_orientation_rmse_deg": MetricRule(10.0, "max", "deg"),
            "joint_rmse_rad": MetricRule(0.15, "max", "rad"),
            "gripper_width_mae_m": MetricRule(0.005, "max", "m"),
            "gripper_event_f1": MetricRule(0.80, "min", "ratio"),
            "contact_event_f1": MetricRule(0.80, "min", "ratio"),
            "trajectory_coverage": MetricRule(0.95, "min", "ratio"),
        }
    )
    minimum_ik_success_rate: float = 0.99
    maximum_violation_rate: float = 0.0
    bootstrap_iterations: int = 2_000
    bootstrap_seed: int = 20260831

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_ik_success_rate <= 1.0:
            raise ValueError("minimum_ik_success_rate must be in [0, 1]")
        if not 0.0 <= self.maximum_violation_rate <= 1.0:
            raise ValueError("maximum_violation_rate must be in [0, 1]")
        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations must be at least 100")


def _scalar_dimension(
    evidence: ScalarEvidence | None,
    required_names: tuple[str, ...],
    rules: dict[str, MetricRule],
    *,
    allowed_frames: set[str],
) -> dict[str, Any]:
    if not required_names:
        raise ValueError("an evaluated scalar dimension requires explicit metric names")
    unknown = set(required_names) - set(rules)
    if unknown:
        raise ValueError(f"suite requests metrics without policy rules: {sorted(unknown)}")
    if evidence is None:
        return {
            "score": 0.0,
            "pass": False,
            "missing": list(required_names),
            "checks": {},
        }
    if evidence.coordinate_frame not in allowed_frames:
        raise ValueError(
            f"evidence frame {evidence.coordinate_frame!r} does not match declared case frames"
        )
    checks: dict[str, Any] = {}
    missing: list[str] = []
    passed = 0
    for name in required_names:
        if name not in evidence.values:
            missing.append(name)
            continue
        value = evidence.values[name]
        rule = rules[name]
        accepted = rule.passes(value)
        passed += int(accepted)
        checks[name] = {
            "value": value,
            "threshold": rule.threshold,
            "direction": rule.direction,
            "unit": rule.unit,
            "pass": accepted,
        }
    return {
        "score": passed / len(required_names),
        "pass": passed == len(required_names),
        "missing": missing,
        "checks": checks,
    }


def simulation_pass(record: SubmissionRecord, policy: BenchmarkPolicy) -> bool:
    evidence = record.simulation
    if evidence is None:
        return False
    return (
        evidence.attempted
        and evidence.physical_gate_complete
        and evidence.physically_valid
        and evidence.task_success
        and evidence.ik_success_rate >= policy.minimum_ik_success_rate
        and evidence.joint_limit_violation_rate <= policy.maximum_violation_rate
        and evidence.velocity_violation_rate <= policy.maximum_violation_rate
        and evidence.collision_rate <= policy.maximum_violation_rate
        and evidence.singularity_rate <= policy.maximum_violation_rate
    )


def evaluate_record(
    case: BenchmarkCase, record: SubmissionRecord, policy: BenchmarkPolicy
) -> dict[str, Any]:
    if case.case_id != record.case_id:
        raise ValueError("case and submission record identifiers do not match")
    scores: dict[str, float | None] = {dimension: None for dimension in DIMENSIONS}
    gates: dict[str, bool | None] = {dimension: None for dimension in DIMENSIONS}
    diagnostics: dict[str, Any] = {}
    missing_dimensions: list[str] = []

    if "l1_visual" in case.required_dimensions:
        if record.visual is None:
            scores["l1_visual"] = 0.0
            gates["l1_visual"] = False
            missing_dimensions.append("l1_visual")
        else:
            scores["l1_visual"] = record.visual.h2r_core / 100.0
            gates["l1_visual"] = True
            diagnostics["l1_visual"] = record.visual.to_dict()

    if "l2_geometry" in case.required_dimensions:
        result = _scalar_dimension(
            record.geometry,
            case.required_metrics.get("l2_geometry", ()),
            policy.geometry_rules,
            allowed_frames={case.camera_frame, case.world_frame, case.robot_base_frame},
        )
        scores["l2_geometry"] = result["score"]
        gates["l2_geometry"] = result["pass"]
        diagnostics["l2_geometry"] = result
        if record.geometry is None:
            missing_dimensions.append("l2_geometry")

    if "l3_action" in case.required_dimensions:
        result = _scalar_dimension(
            record.action,
            case.required_metrics.get("l3_action", ()),
            policy.action_rules,
            allowed_frames={case.camera_frame, case.world_frame, case.robot_base_frame},
        )
        scores["l3_action"] = result["score"]
        gates["l3_action"] = result["pass"]
        diagnostics["l3_action"] = result
        if record.action is None:
            missing_dimensions.append("l3_action")

    sim_valid = simulation_pass(record, policy)
    if "l4_sim" in case.required_dimensions:
        scores["l4_sim"] = float(sim_valid)
        gates["l4_sim"] = sim_valid
        diagnostics["l4_sim"] = record.simulation.to_dict() if record.simulation else None
        if record.simulation is None:
            missing_dimensions.append("l4_sim")

    if "l5_real" in case.required_dimensions:
        valid_success = record.real.valid_success if record.real is not None else False
        scores["l5_real"] = float(valid_success)
        gates["l5_real"] = valid_success
        diagnostics["l5_real"] = record.real.to_dict() if record.real else None
        if sim_valid and record.real is None:
            missing_dimensions.append("l5_real")

    required_gates = [bool(gates[name]) for name in case.required_dimensions]
    return {
        "case_id": case.case_id,
        "task_name": case.task_name,
        "task_family": case.task_family,
        "track": case.track,
        "scores": scores,
        "gates": gates,
        "all_required_pass": all(required_gates),
        "missing_dimensions": missing_dimensions,
        "diagnostics": diagnostics,
    }


def _stratified_bootstrap_ci(
    rows: list[tuple[str, float]], *, iterations: int, seed: int
) -> tuple[float, float] | None:
    if len(rows) < 2:
        return None
    grouped: dict[str, list[float]] = {}
    for family, value in rows:
        grouped.setdefault(family, []).append(value)
    generator = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        draw = [
            values[generator.randrange(len(values))]
            for _, values in sorted(grouped.items())
            for _ in values
        ]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    lower = samples[max(0, int(0.025 * iterations) - 1)]
    upper = samples[min(iterations - 1, int(0.975 * iterations))]
    return lower, upper


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def evaluate_submission(
    suite: BenchmarkSuite,
    submission: Submission,
    policy: BenchmarkPolicy | None = None,
) -> dict[str, Any]:
    """Evaluate one complete submission without dropping rejected or missing cases."""

    policy = policy or BenchmarkPolicy()
    if submission.suite_name != suite.name:
        raise ValueError("submission suite_name does not match the benchmark suite")
    case_by_id = {case.case_id: case for case in suite.cases}
    record_by_id = {record.case_id: record for record in submission.records}
    if set(case_by_id) != set(record_by_id):
        missing = sorted(set(case_by_id) - set(record_by_id))
        extra = sorted(set(record_by_id) - set(case_by_id))
        raise ValueError(f"submission must cover the exact suite; missing={missing}, extra={extra}")

    per_case = [
        evaluate_record(case, record_by_id[case.case_id], policy)
        for case in suite.cases
    ]
    dimension_scores: dict[str, float | None] = {}
    for dimension in DIMENSIONS:
        values = [
            float(row["scores"][dimension])
            for row, case in zip(per_case, suite.cases)
            if dimension in case.required_dimensions and row["scores"][dimension] is not None
        ]
        dimension_scores[dimension] = sum(values) / len(values) if values else None

    visual_rows = [
        (case.task_family, record_by_id[case.case_id].visual.h2r_core)
        for case in suite.cases
        if "l1_visual" in case.required_dimensions and record_by_id[case.case_id].visual is not None
    ]
    h2r_ci = _stratified_bootstrap_ci(
        visual_rows,
        iterations=policy.bootstrap_iterations,
        seed=policy.bootstrap_seed,
    )

    real_cases = [case for case in suite.cases if "l5_real" in case.required_dimensions]
    sim_pass_count = sum(simulation_pass(record_by_id[case.case_id], policy) for case in real_cases)
    real_attempted = sum(
        bool(record_by_id[case.case_id].real and record_by_id[case.case_id].real.attempted)
        for case in real_cases
        if simulation_pass(record_by_id[case.case_id], policy)
    )
    valid_real_successes = sum(
        bool(
            simulation_pass(record_by_id[case.case_id], policy)
            and record_by_id[case.case_id].real
            and record_by_id[case.case_id].real.valid_success
        )
        for case in real_cases
    )
    real_failures_after_sim_pass = sum(
        bool(
            simulation_pass(record_by_id[case.case_id], policy)
            and record_by_id[case.case_id].real
            and record_by_id[case.case_id].real.attempted
            and not record_by_id[case.case_id].real.valid_success
        )
        for case in real_cases
    )
    real_total = len(real_cases)
    coverage = sim_pass_count / real_total if real_total else None
    e2e_vsr = valid_real_successes / real_total if real_total else None
    real_precision_lower_bound = valid_real_successes / sim_pass_count if sim_pass_count else None
    real_audit_completion = real_attempted / sim_pass_count if sim_pass_count else None
    false_accept_rate_observed = (
        real_failures_after_sim_pass / real_attempted if real_attempted else None
    )

    total_gpu_hours = sum(
        record.runtime.gpu_hours for record in submission.records if record.runtime is not None
    )
    valid_video_seconds = sum(
        record_by_id[case.case_id].runtime.generated_video_seconds
        for case in real_cases
        if simulation_pass(record_by_id[case.case_id], policy)
        and record_by_id[case.case_id].runtime is not None
        and record_by_id[case.case_id].real is not None
        and record_by_id[case.case_id].real.valid_success
    )
    valid_data_goodput = valid_video_seconds / total_gpu_hours if total_gpu_hours > 0 else None
    utility = [
        record.policy_utility.delta_success_rate
        for record in submission.records
        if record.policy_utility is not None and record.policy_utility.matched_training_budget
    ]

    return {
        "schema_version": "0.1.0",
        "suite": suite.name,
        "suite_version": suite.version,
        "method": submission.method,
        "case_count": len(suite.cases),
        "dimension_scores": dimension_scores,
        "h2r_core": (
            sum(value for _, value in visual_rows) / len(visual_rows) if visual_rows else None
        ),
        "h2r_core_stratified_bootstrap_95ci": list(h2r_ci) if h2r_ci else None,
        "real_audit": {
            "requested_cases": real_total,
            "sim_pass_count": sim_pass_count,
            "real_attempted_count": real_attempted,
            "valid_real_success_count": valid_real_successes,
            "coverage": coverage,
            "real_precision_lower_bound": real_precision_lower_bound,
            "real_audit_completion": real_audit_completion,
            "observed_sim_false_accept_rate": false_accept_rate_observed,
            "e2e_valid_success_rate": e2e_vsr,
            "e2e_valid_success_wilson_95ci": (
                list(_wilson(valid_real_successes, real_total)) if real_total else None
            ),
        },
        "efficiency": {
            "total_gpu_hours": total_gpu_hours,
            "valid_real_trajectory_seconds": valid_video_seconds,
            "real_valid_data_goodput_seconds_per_gpu_hour": valid_data_goodput,
        },
        "policy_utility": {
            "matched_records": len(utility),
            "mean_delta_real_success_rate": sum(utility) / len(utility) if utility else None,
        },
        "complete": not any(row["missing_dimensions"] for row in per_case),
        "all_required_pass_rate": sum(row["all_required_pass"] for row in per_case) / len(per_case),
        "per_case": per_case,
        "claim_boundary": (
            "L1 reproduces the public H2R scoring equations; L4 imports or runs an external "
            "physics backend; L5 is valid only for recorded, blind-reviewed hardware trials."
        ),
    }
