from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from phiagent.benchmark.adapters import RoboWMBenchAdapter
from phiagent.benchmark.embodiments import EmbodimentRegistry
from phiagent.benchmark.h2r import H2RAnnotation, H2RJudgeOutput, aggregate_h2r_judges
from phiagent.benchmark.hardware import HardwareAdapterManifest
from phiagent.benchmark.integrity import verify_freeze_manifest
from phiagent.benchmark.metrics import BenchmarkPolicy, evaluate_submission, simulation_pass
from phiagent.benchmark.schema import BenchmarkSuite, ScalarEvidence, Submission
from phiagent.benchmark.trajectory import (
    ActionTrajectory,
    MultiArmActionTrajectory,
    compare_action_trajectories,
    compare_multi_arm_trajectories,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "benchmark" / "suites" / "smoke-v0.1"
PUBLIC_PREVIEW = ROOT / "benchmark" / "suites" / "public-preview-v0.1"
PUBLIC_VISUAL_PILOT = ROOT / "benchmark" / "suites" / "public-visual-pilot-v0.1"


def _payload(name: str) -> dict[str, object]:
    return json.loads((SMOKE / name).read_text())


def test_h2r_public_equations_are_reproduced() -> None:
    annotation = H2RAnnotation.from_dict(_payload("h2r-annotation.json"))
    judges = tuple(
        H2RJudgeOutput.from_dict(_payload(f"h2r-judge-{suffix}.json"))
        for suffix in ("a", "b", "c")
    )
    evidence = aggregate_h2r_judges(
        annotation,
        judges,
        video_quality_components=_payload("video-quality.json"),
    )
    assert evidence.goal_completion == pytest.approx(1.0)
    assert evidence.action_completion == pytest.approx(0.875)
    assert evidence.contact_transfer == pytest.approx(0.95)
    assert evidence.embodiment_correctness == pytest.approx(1.0)
    assert evidence.video_quality == pytest.approx(0.8)
    assert evidence.h2r_core == pytest.approx(94.625)
    assert evidence.diagnostics["goal_coverage_eq4"] == pytest.approx(1.0)


def test_h2r_hard_failures_zero_contact_and_embodiment() -> None:
    annotation = H2RAnnotation.from_dict(_payload("h2r-annotation.json"))
    judge = H2RJudgeOutput.from_dict(_payload("h2r-judge-a.json"))
    rejected = replace(judge, source_grounded=False, embodiment_hard_failure=True)
    evidence = aggregate_h2r_judges(
        annotation,
        (rejected,),
        video_quality_components=_payload("video-quality.json"),
        strict_three_judges=False,
    )
    assert evidence.contact_transfer == 0.0
    assert evidence.embodiment_correctness == 0.0


def test_smoke_suite_closes_l1_to_l5_and_counts_all_requests() -> None:
    suite = BenchmarkSuite.from_json(SMOKE / "suite.json")
    submission = Submission.from_json(SMOKE / "submission-reference.json")
    report = evaluate_submission(
        suite,
        submission,
        BenchmarkPolicy(bootstrap_iterations=100),
    )
    assert report["complete"] is True
    assert report["dimension_scores"] == {
        "l1_visual": pytest.approx(0.94625),
        "l2_geometry": 1.0,
        "l3_action": 1.0,
        "l4_sim": 1.0,
        "l5_real": 1.0,
    }
    assert report["h2r_core"] == pytest.approx(94.625)
    assert report["real_audit"]["coverage"] == 1.0
    assert report["real_audit"]["real_audit_completion"] == 1.0
    assert report["real_audit"]["e2e_valid_success_rate"] == 1.0
    assert report["efficiency"]["real_valid_data_goodput_seconds_per_gpu_hour"] == 50.0
    assert report["policy_utility"]["mean_delta_real_success_rate"] == pytest.approx(0.2)


def test_submission_cannot_drop_a_hard_case() -> None:
    suite = BenchmarkSuite.from_json(SMOKE / "suite.json")
    submission = Submission.from_json(SMOKE / "submission-reference.json")
    with pytest.raises(ValueError, match="exact suite"):
        evaluate_submission(suite, replace(submission, records=()))


def test_coordinate_frame_mismatch_is_rejected() -> None:
    suite = BenchmarkSuite.from_json(SMOKE / "suite.json")
    submission = Submission.from_json(SMOKE / "submission-reference.json")
    bad_record = replace(
        submission.records[0],
        action=ScalarEvidence(
            coordinate_frame="camera:undeclared",
            values=submission.records[0].action.values,
        ),
    )
    with pytest.raises(ValueError, match="does not match declared case frames"):
        evaluate_submission(suite, replace(submission, records=(bad_record,)))


def test_robowm_task_success_does_not_fake_full_physics() -> None:
    adapter = RoboWMBenchAdapter(Path("/unused"), "revision")
    evidence = adapter.parse_task_outcome_log("Total attempts: 1\nSuccesses: 1\n")
    record = replace(
        Submission.from_json(SMOKE / "submission-reference.json").records[0],
        simulation=evidence,
    )
    assert evidence.task_success is True
    assert evidence.physical_gate_complete is False
    assert simulation_pass(record, BenchmarkPolicy()) is False


def test_identical_action_trajectories_have_zero_error_and_perfect_events() -> None:
    payload = {
        "coordinate_frame": "robot_base:fixture",
        "timestamps_s": [0.0, 0.1, 0.2, 0.3],
        "eef_positions_m": [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]],
        "eef_quaternions_xyzw": [[0, 0, 0, 1]] * 4,
        "joint_names": ["joint1"],
        "joint_positions_rad": [[0.0], [0.1], [0.2], [0.3]],
        "gripper_width_m": [0.04, 0.0, 0.0, 0.04],
        "contact_state": [False, True, True, False],
    }
    trajectory = ActionTrajectory.from_dict(payload)
    metrics = compare_action_trajectories(trajectory, trajectory)
    assert metrics["eef_position_rmse_m"] == 0.0
    assert metrics["eef_orientation_rmse_deg"] == 0.0
    assert metrics["joint_rmse_rad"] == 0.0
    assert metrics["gripper_width_mae_m"] == 0.0
    assert metrics["gripper_event_f1"] == 1.0
    assert metrics["contact_event_f1"] == 1.0
    assert metrics["trajectory_coverage"] == 1.0


def test_recorded_rm65_adapter_matches_smoke_case_without_enabling_hardware() -> None:
    suite = BenchmarkSuite.from_json(SMOKE / "suite.json")
    manifest = HardwareAdapterManifest.from_json(
        ROOT / "benchmark" / "adapters" / "rm65-ag2f90d-recorded.json"
    )
    result = manifest.compatibility(suite.cases[0])
    assert result["compatible"] is True
    assert result["execution_enabled"] is False
    assert result["evidence_only"] is True
    assert result["checks"]["end_effector_limits"] is True
    assert manifest.end_effector_limits is not None
    assert manifest.end_effector_limits.speed_modes_m_s == (0.1, 0.25)


def test_public_preview_preserves_missing_physical_evidence() -> None:
    suite = BenchmarkSuite.from_json(PUBLIC_PREVIEW / "suite.json")
    submission = Submission.from_json(PUBLIC_PREVIEW / "submission-current.json")
    report = evaluate_submission(
        suite,
        submission,
        BenchmarkPolicy(bootstrap_iterations=100),
    )
    row = report["per_case"][0]
    assert report["complete"] is False
    assert report["all_required_pass_rate"] == 0.0
    assert row["gates"]["l4_sim"] is False
    assert row["diagnostics"]["l4_sim"]["ik_success_rate"] == 1.0
    assert report["real_audit"]["e2e_valid_success_rate"] == 0.0
    assert report["h2r_core"] is None


def test_real_success_cannot_bypass_a_failed_simulation_gate() -> None:
    suite = BenchmarkSuite.from_json(SMOKE / "suite.json")
    submission = Submission.from_json(SMOKE / "submission-reference.json")
    failed_simulation = replace(
        submission.records[0].simulation,
        physically_valid=False,
        task_success=False,
    )
    report = evaluate_submission(
        suite,
        replace(
            submission,
            records=(replace(submission.records[0], simulation=failed_simulation),),
        ),
        BenchmarkPolicy(bootstrap_iterations=100),
    )
    assert report["real_audit"]["sim_pass_count"] == 0
    assert report["real_audit"]["valid_real_success_count"] == 0
    assert report["real_audit"]["e2e_valid_success_rate"] == 0.0
    assert report["efficiency"]["valid_real_trajectory_seconds"] == 0


def test_real_success_requires_pre_registration() -> None:
    suite = BenchmarkSuite.from_json(SMOKE / "suite.json")
    submission = Submission.from_json(SMOKE / "submission-reference.json")
    unregistered = replace(submission.records[0].real, pre_registered=False)
    report = evaluate_submission(
        suite,
        replace(
            submission,
            records=(replace(submission.records[0], real=unregistered),),
        ),
        BenchmarkPolicy(bootstrap_iterations=100),
    )
    assert report["dimension_scores"]["l5_real"] == 0.0
    assert report["real_audit"]["valid_real_success_count"] == 0


def test_real_success_requires_the_case_protocol() -> None:
    suite = BenchmarkSuite.from_json(SMOKE / "suite.json")
    submission = Submission.from_json(SMOKE / "submission-reference.json")
    wrong_protocol = replace(submission.records[0].real, protocol_id="unreviewed-v0")
    report = evaluate_submission(
        suite,
        replace(
            submission,
            records=(replace(submission.records[0], real=wrong_protocol),),
        ),
        BenchmarkPolicy(bootstrap_iterations=100),
    )
    row = report["per_case"][0]
    assert row["diagnostics"]["l5_real"]["protocol_match"] is False
    assert report["real_audit"]["valid_real_success_count"] == 0


def test_multi_arm_action_uses_conservative_worst_arm_aggregation() -> None:
    arm = {
        "timestamps_s": [0.0, 0.1],
        "eef_positions_m": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        "eef_quaternions_xyzw": [[0.0, 0.0, 0.0, 1.0]] * 2,
        "gripper_width_m": [0.04, 0.0],
    }
    reference = MultiArmActionTrajectory.from_dict(
        {
            "coordinate_frame": "robot_base:dual",
            "arms": {"left": arm, "right": arm},
        }
    )
    shifted = dict(arm)
    shifted["eef_positions_m"] = [[0.02, 0.0, 0.0], [0.12, 0.0, 0.0]]
    candidate = MultiArmActionTrajectory.from_dict(
        {
            "coordinate_frame": "robot_base:dual",
            "arms": {"left": arm, "right": shifted},
        }
    )
    aggregate, per_arm = compare_multi_arm_trajectories(reference, candidate)
    assert per_arm["left"]["eef_position_rmse_m"] == 0.0
    assert per_arm["right"]["eef_position_rmse_m"] == pytest.approx(0.02)
    assert aggregate["eef_position_rmse_m"] == pytest.approx(0.02)
    assert aggregate["trajectory_coverage"] == 1.0


def test_embodiment_registry_pins_sources_without_claiming_validation() -> None:
    registry = EmbodimentRegistry.from_json(
        ROOT / "benchmark" / "embodiments" / "registry-v0.1.json"
    )
    summary = registry.summary()
    assert summary["asset_count"] == 8
    assert summary["by_kind"] == {"arm": 6, "end_effector": 2}
    assert summary["by_validation_tier"]["source_pinned"] == 6
    assert summary["by_validation_tier"]["hardware_validated"] == 0


def test_public_visual_pilot_is_hash_frozen_and_explicitly_l1_only() -> None:
    result = verify_freeze_manifest(
        PUBLIC_VISUAL_PILOT / "freeze.json",
        repository_root=ROOT,
    )
    suite = BenchmarkSuite.from_json(PUBLIC_VISUAL_PILOT / "suite.json")
    assert result["valid"] is True
    assert result["artifact_count"] == 3
    assert len(suite.cases) == 3
    assert {case.task_family for case in suite.cases} == {
        "f1_rigid_rearrangement",
        "f4_deformable_configuration",
        "f6_surface_transformation",
    }
    assert all(case.required_dimensions == ("l1_visual",) for case in suite.cases)
