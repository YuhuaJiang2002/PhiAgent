"""L1--L5 evaluation and non-cherry-picked suite aggregation."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
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
    policy_id: str | None = None
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
    minimum_h2r_core: float = 70.0
    minimum_contact_transfer: float = 0.60
    minimum_embodiment_correctness: float = 0.60
    minimum_simulation_episodes: int = 1
    minimum_simulation_pass_rate: float = 1.0
    require_simulation_provenance: bool = False
    minimum_real_trials: int = 1
    minimum_real_sessions: int = 1
    minimum_real_success_rate: float = 0.5
    require_real_eligibility_checks: bool = False
    bootstrap_iterations: int = 2_000
    bootstrap_seed: int = 20260831

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_ik_success_rate <= 1.0:
            raise ValueError("minimum_ik_success_rate must be in [0, 1]")
        if not 0.0 <= self.maximum_violation_rate <= 1.0:
            raise ValueError("maximum_violation_rate must be in [0, 1]")
        if not 0.0 <= self.minimum_h2r_core <= 100.0:
            raise ValueError("minimum_h2r_core must be in [0, 100]")
        for name in (
            "minimum_contact_transfer",
            "minimum_embodiment_correctness",
            "minimum_simulation_pass_rate",
            "minimum_real_success_rate",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.minimum_simulation_episodes <= 0:
            raise ValueError("minimum_simulation_episodes must be positive")
        if self.minimum_real_trials <= 0 or self.minimum_real_sessions <= 0:
            raise ValueError("minimum real trials and sessions must be positive")
        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations must be at least 100")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkPolicy":
        if payload.get("schema_version") not in {None, "0.2.0"}:
            raise ValueError("benchmark policy schema_version must be 0.2.0")

        def rules(name: str, defaults: dict[str, MetricRule]) -> dict[str, MetricRule]:
            raw = payload.get(name)
            if raw is None:
                return defaults
            if not isinstance(raw, dict):
                raise ValueError(f"{name} must be an object")
            return {
                str(metric): MetricRule(
                    threshold=float(spec["threshold"]),
                    direction=str(spec["direction"]),
                    unit=str(spec["unit"]),
                )
                for metric, spec in raw.items()
            }

        defaults = cls()
        require_checks = payload.get(
            "require_real_eligibility_checks",
            defaults.require_real_eligibility_checks,
        )
        if not isinstance(require_checks, bool):
            raise ValueError("require_real_eligibility_checks must be boolean")
        require_simulation_provenance = payload.get(
            "require_simulation_provenance",
            defaults.require_simulation_provenance,
        )
        if not isinstance(require_simulation_provenance, bool):
            raise ValueError("require_simulation_provenance must be boolean")
        return cls(
            policy_id=(str(payload["policy_id"]) if payload.get("policy_id") else None),
            geometry_rules=rules("geometry_rules", defaults.geometry_rules),
            action_rules=rules("action_rules", defaults.action_rules),
            minimum_ik_success_rate=float(
                payload.get("minimum_ik_success_rate", defaults.minimum_ik_success_rate)
            ),
            maximum_violation_rate=float(
                payload.get("maximum_violation_rate", defaults.maximum_violation_rate)
            ),
            minimum_h2r_core=float(payload.get("minimum_h2r_core", defaults.minimum_h2r_core)),
            minimum_contact_transfer=float(
                payload.get("minimum_contact_transfer", defaults.minimum_contact_transfer)
            ),
            minimum_embodiment_correctness=float(
                payload.get(
                    "minimum_embodiment_correctness",
                    defaults.minimum_embodiment_correctness,
                )
            ),
            minimum_simulation_episodes=int(
                payload.get(
                    "minimum_simulation_episodes", defaults.minimum_simulation_episodes
                )
            ),
            minimum_simulation_pass_rate=float(
                payload.get(
                    "minimum_simulation_pass_rate", defaults.minimum_simulation_pass_rate
                )
            ),
            require_simulation_provenance=require_simulation_provenance,
            minimum_real_trials=int(
                payload.get("minimum_real_trials", defaults.minimum_real_trials)
            ),
            minimum_real_sessions=int(
                payload.get("minimum_real_sessions", defaults.minimum_real_sessions)
            ),
            minimum_real_success_rate=float(
                payload.get("minimum_real_success_rate", defaults.minimum_real_success_rate)
            ),
            require_real_eligibility_checks=require_checks,
            bootstrap_iterations=int(
                payload.get("bootstrap_iterations", defaults.bootstrap_iterations)
            ),
            bootstrap_seed=int(payload.get("bootstrap_seed", defaults.bootstrap_seed)),
        )

    @classmethod
    def from_json(cls, path: Path) -> "BenchmarkPolicy":
        payload = json.loads(path.expanduser().resolve().read_text())
        if not isinstance(payload, dict):
            raise ValueError("benchmark policy must be a JSON object")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        def rules(values: dict[str, MetricRule]) -> dict[str, dict[str, Any]]:
            return {
                name: {
                    "threshold": rule.threshold,
                    "direction": rule.direction,
                    "unit": rule.unit,
                }
                for name, rule in values.items()
            }

        return {
            "policy_id": self.policy_id,
            "geometry_rules": rules(self.geometry_rules),
            "action_rules": rules(self.action_rules),
            "minimum_ik_success_rate": self.minimum_ik_success_rate,
            "maximum_violation_rate": self.maximum_violation_rate,
            "minimum_h2r_core": self.minimum_h2r_core,
            "minimum_contact_transfer": self.minimum_contact_transfer,
            "minimum_embodiment_correctness": self.minimum_embodiment_correctness,
            "minimum_simulation_episodes": self.minimum_simulation_episodes,
            "minimum_simulation_pass_rate": self.minimum_simulation_pass_rate,
            "require_simulation_provenance": self.require_simulation_provenance,
            "minimum_real_trials": self.minimum_real_trials,
            "minimum_real_sessions": self.minimum_real_sessions,
            "minimum_real_success_rate": self.minimum_real_success_rate,
            "require_real_eligibility_checks": self.require_real_eligibility_checks,
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
        }


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


def _simulation_evidence_pass(evidence: Any, policy: BenchmarkPolicy) -> bool:
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


def _simulation_summary(record: SubmissionRecord, policy: BenchmarkPolicy) -> dict[str, Any]:
    episodes = record.simulation_evidence
    passing = sum(_simulation_evidence_pass(item, policy) for item in episodes)
    provenance_complete = bool(episodes) and all(
        item.episode_id is not None
        and item.initial_state_id is not None
        and item.seed is not None
        and bool(item.artifact_hashes)
        for item in episodes
    )
    complete = bool(episodes) and all(
        item.attempted and item.physical_gate_complete for item in episodes
    ) and (provenance_complete or not policy.require_simulation_provenance)
    pass_rate = passing / len(episodes) if episodes else 0.0
    protocol_complete = len(episodes) >= policy.minimum_simulation_episodes and complete
    aggregate = (
        {
            "ik_success_rate": min(item.ik_success_rate for item in episodes),
            "joint_limit_violation_rate": max(
                item.joint_limit_violation_rate for item in episodes
            ),
            "velocity_violation_rate": max(
                item.velocity_violation_rate for item in episodes
            ),
            "collision_rate": max(item.collision_rate for item in episodes),
            "singularity_rate": max(item.singularity_rate for item in episodes),
            "stage_success_rate": sum(item.stage_success_rate for item in episodes)
            / len(episodes),
            "contact_success_rate": sum(item.contact_success_rate for item in episodes)
            / len(episodes),
        }
        if episodes
        else {}
    )
    return {
        **aggregate,
        "episode_count": len(episodes),
        "minimum_episodes": policy.minimum_simulation_episodes,
        "passing_episode_count": passing,
        "pass_rate": pass_rate,
        "minimum_pass_rate": policy.minimum_simulation_pass_rate,
        "physical_evidence_complete": complete,
        "provenance_required": policy.require_simulation_provenance,
        "provenance_complete": provenance_complete,
        "protocol_complete": protocol_complete,
        "pass": protocol_complete and pass_rate >= policy.minimum_simulation_pass_rate,
        "episodes": [item.to_dict() for item in episodes],
    }


def simulation_pass(record: SubmissionRecord, policy: BenchmarkPolicy) -> bool:
    return bool(_simulation_summary(record, policy)["pass"])


def _visual_summary(record: SubmissionRecord, policy: BenchmarkPolicy) -> dict[str, Any]:
    evidence = record.visual
    if evidence is None:
        return {"pass": False, "missing": True}
    checks = {
        "h2r_core": {
            "value": evidence.h2r_core,
            "threshold": policy.minimum_h2r_core,
            "direction": "min",
            "pass": evidence.h2r_core >= policy.minimum_h2r_core,
        },
        "contact_transfer": {
            "value": evidence.contact_transfer,
            "threshold": policy.minimum_contact_transfer,
            "direction": "min",
            "pass": evidence.contact_transfer >= policy.minimum_contact_transfer,
        },
        "embodiment_correctness": {
            "value": evidence.embodiment_correctness,
            "threshold": policy.minimum_embodiment_correctness,
            "direction": "min",
            "pass": evidence.embodiment_correctness
            >= policy.minimum_embodiment_correctness,
        },
    }
    return {
        "pass": all(bool(item["pass"]) for item in checks.values()),
        "missing": False,
        "checks": checks,
        "evidence": evidence.to_dict(),
    }


def _real_summary(
    case: BenchmarkCase,
    record: SubmissionRecord,
    policy: BenchmarkPolicy,
    *,
    simulation_valid: bool,
) -> dict[str, Any]:
    trials = record.real_evidence
    expected_protocol = case.annotation.get("real_protocol_id")
    protocol_matches = [
        expected_protocol is None or trial.protocol_id == expected_protocol for trial in trials
    ]
    required_eligibility = {
        "adapter_execution_enabled",
        "calibration_bound_to_robot_serial",
        "action_and_scene_hash_frozen_before_execution",
        "site_safety_approved",
    }
    eligibility_matches = [
        required_eligibility.issubset(trial.eligibility_checks)
        and all(trial.eligibility_checks[name] for name in required_eligibility)
        if policy.require_real_eligibility_checks
        else True
        for trial in trials
    ]
    attempted = sum(trial.attempted for trial in trials)
    valid_successes = sum(
        trial.valid_success and protocol_match and eligibility_match
        for trial, protocol_match, eligibility_match in zip(
            trials, protocol_matches, eligibility_matches
        )
    )
    sessions = {trial.session_id for trial in trials if trial.attempted}
    protocol_complete = (
        len(trials) >= policy.minimum_real_trials
        and attempted >= policy.minimum_real_trials
        and len(sessions) >= policy.minimum_real_sessions
        and all(protocol_matches)
        and all(eligibility_matches)
    )
    conservative_denominator = max(policy.minimum_real_trials, len(trials))
    success_rate = valid_successes / conservative_denominator
    return {
        "expected_protocol_id": expected_protocol,
        "protocol_match": bool(protocol_matches) and all(protocol_matches),
        "eligibility_checks_required": policy.require_real_eligibility_checks,
        "eligibility_match": bool(eligibility_matches) and all(eligibility_matches),
        "trial_count": len(trials),
        "attempted_count": attempted,
        "minimum_trials": policy.minimum_real_trials,
        "session_count": len(sessions),
        "minimum_sessions": policy.minimum_real_sessions,
        "valid_success_count": valid_successes,
        "valid_success_rate": success_rate,
        "minimum_success_rate": policy.minimum_real_success_rate,
        "protocol_complete": protocol_complete,
        "simulation_gate_pass": simulation_valid,
        "pass": simulation_valid
        and protocol_complete
        and success_rate >= policy.minimum_real_success_rate,
        "trials": [
            {
                **trial.to_dict(),
                "protocol_match": protocol_match,
                "eligibility_match": eligibility_match,
            }
            for trial, protocol_match, eligibility_match in zip(
                trials, protocol_matches, eligibility_matches
            )
        ],
    }


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
            result = _visual_summary(record, policy)
            scores["l1_visual"] = record.visual.h2r_core / 100.0
            gates["l1_visual"] = bool(result["pass"])
            diagnostics["l1_visual"] = result

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

    simulation = _simulation_summary(record, policy)
    sim_valid = bool(simulation["pass"])
    if "l4_sim" in case.required_dimensions:
        scores["l4_sim"] = float(simulation["pass_rate"])
        gates["l4_sim"] = sim_valid
        diagnostics["l4_sim"] = simulation
        if not simulation["protocol_complete"]:
            missing_dimensions.append("l4_sim")

    if "l5_real" in case.required_dimensions:
        real = _real_summary(case, record, policy, simulation_valid=sim_valid)
        scores["l5_real"] = float(real["valid_success_rate"])
        gates["l5_real"] = bool(real["pass"])
        diagnostics["l5_real"] = real
        if sim_valid and not real["protocol_complete"]:
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
    if suite.policy_id is not None and policy.policy_id != suite.policy_id:
        raise ValueError(
            f"suite requires policy {suite.policy_id!r}, got {policy.policy_id!r}"
        )
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
    evaluated_by_id = {str(row["case_id"]): row for row in per_case}
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
    real_summaries = {
        case.case_id: evaluated_by_id[case.case_id]["diagnostics"]["l5_real"]
        for case in real_cases
    }
    real_attempted = sum(
        int(real_summaries[case.case_id]["attempted_count"])
        for case in real_cases
        if simulation_pass(record_by_id[case.case_id], policy)
    )
    valid_real_successes = sum(
        int(real_summaries[case.case_id]["valid_success_count"])
        for case in real_cases
        if simulation_pass(record_by_id[case.case_id], policy)
    )
    real_failures_after_sim_pass = sum(
        int(real_summaries[case.case_id]["attempted_count"])
        - int(real_summaries[case.case_id]["valid_success_count"])
        for case in real_cases
        if simulation_pass(record_by_id[case.case_id], policy)
    )
    real_total = len(real_cases)
    requested_real_trials = sum(
        max(
            policy.minimum_real_trials,
            int(real_summaries[case.case_id]["trial_count"]),
        )
        for case in real_cases
    )
    eligible_real_trials = sum(
        max(
            policy.minimum_real_trials,
            int(real_summaries[case.case_id]["trial_count"]),
        )
        for case in real_cases
        if simulation_pass(record_by_id[case.case_id], policy)
    )
    protocol_complete_cases = sum(
        bool(real_summaries[case.case_id]["protocol_complete"])
        for case in real_cases
        if simulation_pass(record_by_id[case.case_id], policy)
    )
    coverage = sim_pass_count / real_total if real_total else None
    e2e_vsr = (
        valid_real_successes / requested_real_trials if requested_real_trials else None
    )
    real_precision_lower_bound = (
        valid_real_successes / eligible_real_trials if eligible_real_trials else None
    )
    real_audit_completion = (
        min(real_attempted, eligible_real_trials) / eligible_real_trials
        if eligible_real_trials
        else None
    )
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
        and int(real_summaries[case.case_id]["valid_success_count"]) > 0
    )
    valid_data_goodput = valid_video_seconds / total_gpu_hours if total_gpu_hours > 0 else None
    utility = [
        record.policy_utility.delta_success_rate
        for record in submission.records
        if record.policy_utility is not None and record.policy_utility.matched_training_budget
    ]

    return {
        "schema_version": "0.2.0",
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
            "requested_trials": requested_real_trials,
            "sim_pass_count": sim_pass_count,
            "real_attempted_count": real_attempted,
            "valid_real_success_count": valid_real_successes,
            "protocol_complete_case_count": protocol_complete_cases,
            "coverage": coverage,
            "real_precision_lower_bound": real_precision_lower_bound,
            "real_audit_completion": real_audit_completion,
            "observed_sim_false_accept_rate": false_accept_rate_observed,
            "e2e_valid_success_rate": e2e_vsr,
            "e2e_valid_success_wilson_95ci": (
                list(_wilson(valid_real_successes, requested_real_trials))
                if requested_real_trials
                else None
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
        "protocol_complete": not any(row["missing_dimensions"] for row in per_case),
        "all_required_pass_rate": sum(row["all_required_pass"] for row in per_case) / len(per_case),
        "per_case": per_case,
        "claim_boundary": (
            "L1 reproduces the public H2R scoring equations; L4 imports or runs an external "
            "physics backend; L5 is valid only for recorded, blind-reviewed hardware trials."
        ),
    }
