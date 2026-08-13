"""Continual, fail-closed supervision for foundation contact architectures.

This module is deliberately standard-library only.  It monitors evidence
produced by heavyweight model/simulator adapters without importing them, keeps
physical promotion separate from diagnostic progress, and never relaxes a
failed gate to manufacture improvement.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


REQUIRED_STAGES = (
    "metric_camera",
    "robot_trajectory",
    "stem_centerlines",
    "contact_forces",
)

REQUIRED_ATTACKS = (
    "learned_scale_spoof_rejected",
    "partial_q_or_wrong_asset_rejected",
    "stem_identity_or_rigidity_spoof_rejected",
    "visual_force_spoof_rejected",
    "mean_score_override_rejected",
)


@dataclass(frozen=True)
class ContinualPromotionContract:
    """Immutable rules for replacing a production architecture."""

    required_groups: int = 2
    bootstrap_samples: int = 2000
    promotion_probability: float = 0.95
    maximum_cost_regression_fraction: float = 0.10
    maximum_quality_regression: float = 0.0
    seed: int = 20260812

    def validate(self) -> None:
        if self.required_groups < 2:
            raise ValueError("promotion requires at least two independent groups")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap requires at least 100 samples")
        if not 0.5 < self.promotion_probability < 1.0:
            raise ValueError("promotion probability must lie between 0.5 and 1")
        if self.maximum_cost_regression_fraction < 0:
            raise ValueError("cost regression bound must be non-negative")
        if self.maximum_quality_regression < 0:
            raise ValueError("quality regression bound must be non-negative")


@dataclass(frozen=True)
class GroupEvaluation:
    """One candidate evaluated on one independent scene/object group."""

    candidate_id: str
    group_id: str
    quality: float
    cost_units: float
    hard_gates: tuple[tuple[str, bool], ...]
    adversarial_attacks: tuple[tuple[str, bool], ...]
    evidence_path: str

    def validate(self) -> None:
        if not self.candidate_id.strip() or not self.group_id.strip():
            raise ValueError("candidate and group IDs must be non-empty")
        if not math.isfinite(self.quality) or not math.isfinite(self.cost_units):
            raise ValueError("quality and cost must be finite")
        if self.cost_units < 0:
            raise ValueError("cost must be non-negative")
        if not self.evidence_path.strip():
            raise ValueError("an immutable evidence path is required")
        gates = dict(self.hard_gates)
        attacks = dict(self.adversarial_attacks)
        if len(gates) != len(self.hard_gates) or len(attacks) != len(
            self.adversarial_attacks
        ):
            raise ValueError("gate and attack names must be unique")
        if tuple(gates) != REQUIRED_STAGES:
            raise ValueError("hard gates must use the frozen physical-stage order")
        if tuple(attacks) != REQUIRED_ATTACKS:
            raise ValueError("attacks must use the frozen adversarial order")

    @property
    def physically_eligible(self) -> bool:
        return all(value for _, value in self.hard_gates) and all(
            value for _, value in self.adversarial_attacks
        )


def canonical_digest(value: Mapping[str, Any]) -> str:
    """Hash a report without depending on filesystem or model packages."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _stage(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    stages = report.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("pipeline report requires a stages mapping")
    row = stages.get(name)
    if not isinstance(row, Mapping):
        raise ValueError(f"pipeline report is missing stage {name!r}")
    return row


def audit_pipeline_report(report: Mapping[str, Any]) -> dict[str, object]:
    """Run semantic attacks against one compiled physical evidence report."""

    camera = _stage(report, "metric_camera")
    robot = _stage(report, "robot_trajectory")
    stems = _stage(report, "stem_centerlines")
    forces = _stage(report, "contact_forces")
    stage_gates = tuple(
        (name, bool(_stage(report, name).get("passed"))) for name in REQUIRED_STAGES
    )

    camera_gates = camera.get("gates", {})
    rejected_learned_spoof = bool(
        camera.get("proposal_passed")
        and not camera.get("passed")
        and not camera.get("calibrated_scale")
        and camera_gates.get("absolute_metric_scale_calibrated") is False
    )
    accepted_bound_calibration = bool(
        camera.get("passed")
        and camera.get("calibrated_scale")
        and camera_gates.get("absolute_metric_scale_calibrated") is True
        and camera.get("evidence_class") in {"sensor_measurement", "calibrated_geometry"}
        and (
            camera.get("evidence_class") == "sensor_measurement"
            or camera.get("calibration_bridge", {}).get("bound") is True
        )
    )
    scale_spoof_rejected = bool(
        isinstance(camera_gates, Mapping)
        and (rejected_learned_spoof or accepted_bound_calibration)
    )
    robot_reasons = {str(value) for value in robot.get("reasons", ())}
    asset_registry_passed = bool(robot.get("exact_asset_registry_passed"))
    partial_q_rejected = bool(
        robot.get("passed")
        or (
            not robot.get("passed")
            and asset_registry_passed
            and "missing_full_generalized_coordinate_trajectory" in robot_reasons
        )
    )
    segment_cv = stems.get("maximum_segment_length_cv")
    stem_spoof_rejected = bool(
        stems.get("passed")
        or (
            not stems.get("passed")
            and isinstance(segment_cv, (int, float))
            and math.isfinite(float(segment_cv))
            and float(segment_cv) > 0.12
        )
    )
    force_reasons = {str(value) for value in forces.get("reasons", ())}
    force_note = str(forces.get("note", "")).lower()
    visual_force_spoof_rejected = bool(
        forces.get("passed")
        or (
            not forces.get("passed")
            and "missing_sensor_or_physics_solver_contact_forces" in force_reasons
            and "visual" in force_note
        )
    )
    all_stages_pass = all(value for _, value in stage_gates)
    status = str(report.get("status", ""))
    mean_override_rejected = (all_stages_pass and status == "WORKING") or (
        not all_stages_pass and status == "PARTIAL"
    )
    attacks = (
        ("learned_scale_spoof_rejected", scale_spoof_rejected),
        ("partial_q_or_wrong_asset_rejected", partial_q_rejected),
        ("stem_identity_or_rigidity_spoof_rejected", stem_spoof_rejected),
        ("visual_force_spoof_rejected", visual_force_spoof_rejected),
        ("mean_score_override_rejected", mean_override_rejected),
    )
    return {
        "report_sha256": canonical_digest(report),
        "stage_gates": dict(stage_gates),
        "attacks": dict(attacks),
        "passed": all(value for _, value in attacks),
    }


def extract_progress_metrics(
    report: Mapping[str, Any],
    *,
    da3_manifest: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Extract diagnostics without turning them into physical promotion gates."""

    camera = _stage(report, "metric_camera")
    robot = _stage(report, "robot_trajectory")
    stems = _stage(report, "stem_centerlines")
    forces = _stage(report, "contact_forces")
    segment_cv = float(stems.get("maximum_segment_length_cv", math.inf))
    rigidity_progress = 0.0
    if math.isfinite(segment_cv) and segment_cv > 0:
        rigidity_progress = min(1.0, 0.12 / segment_cv)
    metrics = {
        "physical_stage_pass_rate": sum(
            bool(_stage(report, name).get("passed")) for name in REQUIRED_STAGES
        )
        / len(REQUIRED_STAGES),
        "camera_proposal_pass": float(bool(camera.get("proposal_passed"))),
        "exact_asset_registry_pass": float(
            bool(robot.get("exact_asset_registry_passed"))
        ),
        "stem_rigidity_progress": rigidity_progress,
        "force_evidence_pass": float(bool(forces.get("passed"))),
    }
    if da3_manifest is not None:
        performance = da3_manifest.get("performance")
        if not isinstance(performance, Mapping):
            raise ValueError("DA3 manifest requires a performance mapping")
        metrics["geometry_sample_fps"] = float(
            performance["sampled_frames_per_inference_second"]
        )
        metrics["warm_source_realtime_factor"] = float(
            performance["source_video_seconds_per_inference_second"]
        )
        extraction = float(performance["frame_extraction_seconds"])
        loading = float(performance["model_load_seconds"])
        inference = float(performance["inference_seconds"])
        duration = float(da3_manifest["input"]["frames"]) / float(
            da3_manifest["input"]["fps"]
        )
        metrics["cold_source_realtime_factor"] = duration / (
            extraction + loading + inference
        )
    if any(not math.isfinite(value) or value < 0 for value in metrics.values()):
        raise ValueError("progress metrics must be finite and non-negative")
    return metrics


def rank_architecture_experiments(
    evolution_plan: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Rank structural experiments by dependency-aware information gain."""

    rows = evolution_plan.get("experiments")
    if not isinstance(rows, list):
        raise ValueError("evolution plan requires an experiments list")
    failed_stages = {str(value) for value in evolution_plan.get("failed_stages", ())}
    known = {str(row.get("failed_stage")) for row in rows if isinstance(row, Mapping)}
    if known != failed_stages:
        raise ValueError("experiment stages must exactly cover failed stages")
    ranked = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("experiment rows must be mappings")
        stage = str(row["failed_stage"])
        blocked_by = tuple(str(value) for value in row.get("blocked_by", ()))
        unresolved = tuple(value for value in blocked_by if value in failed_stages)
        unblocks = sum(
            stage in {str(value) for value in candidate.get("blocked_by", ())}
            for candidate in rows
            if isinstance(candidate, Mapping)
        )
        evidence_count = len(tuple(row.get("required_evidence", ())))
        attack_count = len(tuple(row.get("promotion_gates", ())))
        information_gain = 1.0 + unblocks + 0.25 * attack_count
        cost_proxy = 1.0 + evidence_count + len(unresolved)
        priority = information_gain / cost_proxy if not unresolved else 0.0
        ranked.append(
            {
                "experiment_id": str(row["experiment_id"]),
                "failed_stage": stage,
                "blocked_by": list(unresolved),
                "ready": not unresolved,
                "information_gain_proxy": information_gain,
                "cost_proxy": cost_proxy,
                "priority": priority,
                "mutation_class": str(row.get("mutation_class", "")),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            not bool(item["ready"]),
            len(item["blocked_by"]),
            -float(item["priority"]),
            str(item["experiment_id"]),
        ),
    )


def _paired_bootstrap_probability(
    incumbent: tuple[float, ...],
    challenger: tuple[float, ...],
    *,
    samples: int,
    seed: int,
) -> float:
    if len(incumbent) != len(challenger) or not incumbent:
        raise ValueError("paired bootstrap requires aligned non-empty groups")
    rng = random.Random(seed)
    deltas = tuple(new - old for old, new in zip(incumbent, challenger, strict=True))
    positive = 0
    for _ in range(samples):
        mean = sum(rng.choice(deltas) for _ in deltas) / len(deltas)
        positive += mean > 0
    return positive / samples


def decide_continual_promotion(
    incumbent: Iterable[GroupEvaluation],
    challenger: Iterable[GroupEvaluation],
    contract: ContinualPromotionContract,
) -> dict[str, object]:
    """Promote only a significant, non-regressing, all-gate challenger."""

    contract.validate()
    old_rows = tuple(incumbent)
    new_rows = tuple(challenger)
    for row in (*old_rows, *new_rows):
        row.validate()
    old_by_group = {row.group_id: row for row in old_rows}
    new_by_group = {row.group_id: row for row in new_rows}
    if len(old_by_group) != len(old_rows) or len(new_by_group) != len(new_rows):
        raise ValueError("candidate evaluations require unique groups")
    if old_by_group.keys() != new_by_group.keys():
        raise ValueError("incumbent and challenger groups must match")
    groups = tuple(sorted(old_by_group))
    if len(groups) < contract.required_groups:
        return {
            "promoted": False,
            "reason": "insufficient_independent_groups",
            "groups": list(groups),
            "required_groups": contract.required_groups,
        }
    candidate_ids = {row.candidate_id for row in new_rows}
    if len(candidate_ids) != 1:
        raise ValueError("challenger rows must describe exactly one candidate")
    all_gates = all(new_by_group[group].physically_eligible for group in groups)
    old_quality = tuple(old_by_group[group].quality for group in groups)
    new_quality = tuple(new_by_group[group].quality for group in groups)
    quality_nonregression = all(
        new >= old - contract.maximum_quality_regression
        for old, new in zip(old_quality, new_quality, strict=True)
    )
    probability = _paired_bootstrap_probability(
        old_quality,
        new_quality,
        samples=contract.bootstrap_samples,
        seed=contract.seed,
    )
    old_cost = sum(old_by_group[group].cost_units for group in groups) / len(groups)
    new_cost = sum(new_by_group[group].cost_units for group in groups) / len(groups)
    cost_gate = new_cost <= old_cost * (1.0 + contract.maximum_cost_regression_fraction)
    promoted = bool(
        all_gates
        and quality_nonregression
        and probability >= contract.promotion_probability
        and cost_gate
    )
    reasons = []
    if not all_gates:
        reasons.append("physical_or_adversarial_gate_failed")
    if not quality_nonregression:
        reasons.append("quality_regression")
    if probability < contract.promotion_probability:
        reasons.append("improvement_not_statistically_supported")
    if not cost_gate:
        reasons.append("cost_regression")
    return {
        "promoted": promoted,
        "selected_candidate": next(iter(candidate_ids)) if promoted else None,
        "groups": list(groups),
        "all_physical_and_attack_gates_pass": all_gates,
        "quality_nonregression": quality_nonregression,
        "bootstrap_improvement_probability": probability,
        "minimum_probability": contract.promotion_probability,
        "incumbent_mean_quality": sum(old_quality) / len(old_quality),
        "challenger_mean_quality": sum(new_quality) / len(new_quality),
        "incumbent_mean_cost": old_cost,
        "challenger_mean_cost": new_cost,
        "cost_gate_pass": cost_gate,
        "reasons": reasons,
    }


def supervise_once(
    *,
    pipeline_report: Mapping[str, Any],
    evolution_plan: Mapping[str, Any],
    da3_manifest: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Audit the latest candidate and choose exactly one ready experiment."""

    audit = audit_pipeline_report(pipeline_report)
    metrics = extract_progress_metrics(pipeline_report, da3_manifest=da3_manifest)
    ranked = rank_architecture_experiments(evolution_plan)
    ready = [row for row in ranked if row["ready"]]
    next_experiment = ready[0] if ready else None
    physical_gates = dict(audit["stage_gates"])
    promotable = bool(all(physical_gates.values()) and audit["passed"])
    return {
        "candidate_report_sha256": audit["report_sha256"],
        "monitor_status": "PROMOTION_ELIGIBLE" if promotable else "EVOLVE",
        "promoted": False,
        "promotion_note": (
            "Eligibility is necessary but promotion still requires two independent groups "
            "and paired statistical non-regression."
        ),
        "physical_gates": physical_gates,
        "adversarial_audit": audit,
        "diagnostic_progress_metrics": metrics,
        "ranked_experiments": ranked,
        "next_experiment": next_experiment,
        "invariant": (
            "Only a measured all-gate, all-attack, statistically supported challenger can "
            "replace the incumbent; diagnostic progress never overrides physics."
        ),
    }
