from __future__ import annotations

import copy

import pytest

from phiagent.agent.foundation_contact_supervisor import (
    REQUIRED_ATTACKS,
    REQUIRED_STAGES,
    ContinualPromotionContract,
    GroupEvaluation,
    audit_pipeline_report,
    decide_continual_promotion,
    rank_architecture_experiments,
    supervise_once,
)


def _partial_report() -> dict[str, object]:
    return {
        "status": "PARTIAL",
        "stages": {
            "metric_camera": {
                "passed": False,
                "proposal_passed": True,
                "calibrated_scale": False,
                "gates": {"absolute_metric_scale_calibrated": False},
            },
            "robot_trajectory": {
                "passed": False,
                "exact_asset_registry_passed": True,
                "reasons": ["missing_full_generalized_coordinate_trajectory"],
            },
            "stem_centerlines": {
                "passed": False,
                "maximum_segment_length_cv": 1.857,
            },
            "contact_forces": {
                "passed": False,
                "reasons": ["missing_sensor_or_physics_solver_contact_forces"],
                "note": "Visual confidence is forbidden as force evidence.",
            },
        },
    }


def _plan() -> dict[str, object]:
    return {
        "failed_stages": list(REQUIRED_STAGES),
        "experiments": [
            {
                "experiment_id": "camera-v1",
                "failed_stage": "metric_camera",
                "blocked_by": [],
                "required_evidence": ["rgbd"],
                "promotion_gates": ["scale", "spoof"],
                "mutation_class": "architecture_not_hyperparameter",
            },
            {
                "experiment_id": "robot-v1",
                "failed_stage": "robot_trajectory",
                "blocked_by": ["metric_camera"],
                "required_evidence": ["asset", "q"],
                "promotion_gates": ["reprojection"],
                "mutation_class": "architecture_not_hyperparameter",
            },
            {
                "experiment_id": "stem-v1",
                "failed_stage": "stem_centerlines",
                "blocked_by": ["metric_camera"],
                "required_evidence": ["track", "rod"],
                "promotion_gates": ["rigidity"],
                "mutation_class": "architecture_not_hyperparameter",
            },
            {
                "experiment_id": "force-v1",
                "failed_stage": "contact_forces",
                "blocked_by": [
                    "metric_camera",
                    "robot_trajectory",
                    "stem_centerlines",
                ],
                "required_evidence": ["sensor", "solver"],
                "promotion_gates": ["residual"],
                "mutation_class": "architecture_not_hyperparameter",
            },
        ],
    }


def _row(candidate: str, group: str, quality: float, *, passed: bool = True) -> GroupEvaluation:
    return GroupEvaluation(
        candidate_id=candidate,
        group_id=group,
        quality=quality,
        cost_units=1.0,
        hard_gates=tuple((name, passed) for name in REQUIRED_STAGES),
        adversarial_attacks=tuple((name, passed) for name in REQUIRED_ATTACKS),
        evidence_path=f"{candidate}/{group}.json",
    )


def test_real_partial_report_rejects_spoofs_without_promotion() -> None:
    result = audit_pipeline_report(_partial_report())
    assert result["passed"] is True
    assert all(result["attacks"].values())
    supervised = supervise_once(
        pipeline_report=_partial_report(),
        evolution_plan=_plan(),
    )
    assert supervised["monitor_status"] == "EVOLVE"
    assert supervised["promoted"] is False
    assert supervised["next_experiment"]["experiment_id"] == "camera-v1"


def test_scale_spoof_attack_detects_false_calibration_upgrade() -> None:
    attacked = copy.deepcopy(_partial_report())
    attacked["stages"]["metric_camera"]["passed"] = True
    result = audit_pipeline_report(attacked)
    assert result["attacks"]["learned_scale_spoof_rejected"] is False
    assert result["passed"] is False


def test_scale_spoof_attack_accepts_hash_bound_independent_calibration() -> None:
    calibrated = copy.deepcopy(_partial_report())
    camera = calibrated["stages"]["metric_camera"]
    camera.update(
        {
            "passed": True,
            "calibrated_scale": True,
            "evidence_class": "calibrated_geometry",
            "calibration_bridge": {"bound": True},
        }
    )
    camera["gates"]["absolute_metric_scale_calibrated"] = True
    result = audit_pipeline_report(calibrated)
    assert result["attacks"]["learned_scale_spoof_rejected"] is True


def test_ranker_never_schedules_blocked_force_before_camera() -> None:
    ranked = rank_architecture_experiments(_plan())
    assert ranked[0]["experiment_id"] == "camera-v1"
    assert ranked[0]["ready"] is True
    assert ranked[-1]["experiment_id"] == "force-v1"
    assert ranked[-1]["ready"] is False


def test_promotion_requires_independent_groups_and_all_gates() -> None:
    contract = ContinualPromotionContract(bootstrap_samples=500)
    insufficient = decide_continual_promotion(
        [_row("old", "scene-a", 0.5)],
        [_row("new", "scene-a", 0.8)],
        contract,
    )
    assert insufficient["promoted"] is False
    assert insufficient["reason"] == "insufficient_independent_groups"

    rejected = decide_continual_promotion(
        [_row("old", "scene-a", 0.5), _row("old", "scene-b", 0.5)],
        [
            _row("new", "scene-a", 0.9, passed=False),
            _row("new", "scene-b", 0.9, passed=False),
        ],
        contract,
    )
    assert rejected["promoted"] is False
    assert "physical_or_adversarial_gate_failed" in rejected["reasons"]


def test_significant_nonregressing_all_gate_candidate_can_promote() -> None:
    groups = tuple(f"scene-{index}" for index in range(6))
    result = decide_continual_promotion(
        [_row("old", group, 0.50) for group in groups],
        [_row("new", group, 0.70) for group in groups],
        ContinualPromotionContract(bootstrap_samples=500),
    )
    assert result["promoted"] is True
    assert result["selected_candidate"] == "new"


def test_unknown_or_incomplete_plan_is_rejected() -> None:
    broken = _plan()
    broken["experiments"] = broken["experiments"][:-1]
    with pytest.raises(ValueError, match="exactly cover"):
        rank_architecture_experiments(broken)
