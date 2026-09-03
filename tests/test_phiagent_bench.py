from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from phiagent.benchmark.adapters import HarnessEvalWAdapter, RoboWMBenchAdapter
from phiagent.benchmark.batch import BatchController, compile_submission, plan_batch_run
from phiagent.benchmark.embodiments import EmbodimentRegistry
from phiagent.benchmark.h2r import H2RAnnotation, H2RJudgeOutput, aggregate_h2r_judges
from phiagent.benchmark.hardware import HardwareAdapterManifest
from phiagent.benchmark.integrity import verify_freeze_manifest
from phiagent.benchmark.metrics import BenchmarkPolicy, evaluate_submission, simulation_pass
from phiagent.benchmark.physical_gate import simulation_evidence_from_trace
from phiagent.benchmark.real_plan import create_real_trial_plan
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


def test_robowm_command_is_headless(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(RoboWMBenchAdapter, "preflight", lambda self: {"status": "ready"})
    adapter = RoboWMBenchAdapter(tmp_path / "RoboWM-Bench", "revision")
    command = adapter.command(
        task="pick",
        trajectory_root=tmp_path / "trajectories",
        output_root=tmp_path / "output",
        episode_index=0,
        device="cuda:0",
    )
    assert "--headless" in command
    assert "--enable_cameras" in command
    assert command[command.index("--device") + 1] == "cuda:0"


def test_robowm_frozen_episode_hashes_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "pick"
    root.mkdir()
    episode = root / "episode_000000.json"
    pose = root / "pose.jsonl"
    episode.write_bytes(b"episode")
    pose.write_bytes(b"pose")
    adapter = RoboWMBenchAdapter(tmp_path / "RoboWM-Bench", "revision")
    result = adapter.verify_frozen_episode(
        trajectory_root=root,
        episode_index=0,
        episode_sha256=hashlib.sha256(b"episode").hexdigest(),
        pose_sha256=hashlib.sha256(b"pose").hexdigest(),
    )
    assert result["status"] == "verified"
    episode.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="hash mismatch"):
        adapter.verify_frozen_episode(
            trajectory_root=root,
            episode_index=0,
            episode_sha256=hashlib.sha256(b"episode").hexdigest(),
            pose_sha256=hashlib.sha256(b"pose").hexdigest(),
        )


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


def test_visual_evidence_does_not_pass_only_by_existing() -> None:
    suite = BenchmarkSuite.from_json(SMOKE / "suite.json")
    submission = Submission.from_json(SMOKE / "submission-reference.json")
    weak = replace(
        submission.records[0].visual,
        goal_completion=0.1,
        action_completion=0.1,
        contact_transfer=0.1,
        embodiment_correctness=0.1,
        video_quality=0.1,
    )
    report = evaluate_submission(
        suite,
        replace(submission, records=(replace(submission.records[0], visual=weak),)),
        BenchmarkPolicy(bootstrap_iterations=100),
    )
    assert report["per_case"][0]["gates"]["l1_visual"] is False
    assert report["per_case"][0]["diagnostics"]["l1_visual"]["checks"]["h2r_core"][
        "pass"
    ] is False


def test_repeated_simulation_and_real_protocol_are_enforced() -> None:
    suite = BenchmarkSuite.from_json(SMOKE / "suite.json")
    submission = Submission.from_json(SMOKE / "submission-reference.json")
    source = submission.records[0]
    simulations = tuple(
        replace(source.simulation, episode_id=f"episode-{index}", seed=index)
        for index in range(5)
    )
    repeated_real = tuple(
        replace(
            source.real,
            trial_id=f"trial-{index}",
            trial_index=index,
            session_id=f"session-{index % 3}",
        )
        for index in range(10)
    )
    record = replace(
        source,
        simulation=None,
        simulations=simulations,
        real=None,
        real_trials=repeated_real,
    )
    policy = BenchmarkPolicy(
        minimum_simulation_episodes=5,
        minimum_real_trials=10,
        minimum_real_sessions=3,
        bootstrap_iterations=100,
    )
    report = evaluate_submission(suite, replace(submission, records=(record,)), policy)
    assert report["protocol_complete"] is True
    assert report["dimension_scores"]["l4_sim"] == 1.0
    assert report["dimension_scores"]["l5_real"] == 1.0
    assert report["real_audit"]["requested_trials"] == 10
    assert report["real_audit"]["protocol_complete_case_count"] == 1

    insufficient = replace(record, real_trials=repeated_real[:9])
    rejected = evaluate_submission(
        suite,
        replace(submission, records=(insufficient,)),
        policy,
    )
    assert rejected["protocol_complete"] is False
    assert rejected["per_case"][0]["gates"]["l5_real"] is False


def test_formal_simulation_policy_requires_hash_bound_episode_provenance() -> None:
    suite = BenchmarkSuite.from_json(SMOKE / "suite.json")
    submission = Submission.from_json(SMOKE / "submission-reference.json")
    report = evaluate_submission(
        suite,
        submission,
        BenchmarkPolicy(
            require_simulation_provenance=True,
            bootstrap_iterations=100,
        ),
    )
    diagnostics = report["per_case"][0]["diagnostics"]["l4_sim"]
    assert diagnostics["provenance_complete"] is False
    assert diagnostics["protocol_complete"] is False
    assert report["per_case"][0]["gates"]["l4_sim"] is False


def test_real_gate_cannot_pass_when_simulation_fails() -> None:
    suite = BenchmarkSuite.from_json(SMOKE / "suite.json")
    submission = Submission.from_json(SMOKE / "submission-reference.json")
    failed_simulation = replace(submission.records[0].simulation, task_success=False)
    report = evaluate_submission(
        suite,
        replace(
            submission,
            records=(replace(submission.records[0], simulation=failed_simulation),),
        ),
        BenchmarkPolicy(bootstrap_iterations=100),
    )
    assert report["per_case"][0]["diagnostics"]["l5_real"]["valid_success_count"] == 1
    assert report["per_case"][0]["gates"]["l5_real"] is False


def test_batch_run_is_resumable_hash_bound_and_compilable(tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--job-dir', type=Path, required=True)
p.add_argument('--case-id', required=True)
p.add_argument('--stage', required=True)
a = p.parse_args()
a.job_dir.mkdir(parents=True, exist_ok=True)
if a.stage == 'generate':
    patch = {'case_id': a.case_id, 'generated_uri': str(a.job_dir / 'generated.mp4')}
    (a.job_dir / 'generated.mp4').write_bytes(b'video')
else:
    patch = {
        'case_id': a.case_id,
        'visual': {
            'goal_completion': 1.0, 'action_completion': 1.0,
            'contact_transfer': 1.0, 'embodiment_correctness': 1.0,
            'video_quality': 1.0, 'judge_count': 3, 'evidence_frames': 25,
            'protocol': 'fixture', 'diagnostics': {'fixture': True}
        }
    }
(a.job_dir / 'record-patch.json').write_text(json.dumps(patch))
""".strip()
        + "\n"
    )
    method = tmp_path / "method.json"
    method.write_text(
        json.dumps(
            {
                "schema_version": "0.2.0",
                "method": "batch-fixture",
                "working_directory": str(tmp_path),
                "candidates_per_case": 1,
                "seed": 7,
                "stages": [
                    {
                        "name": "generate",
                        "command": [
                            sys.executable,
                            str(worker),
                            "--job-dir",
                            "{job_dir}",
                            "--case-id",
                            "{case_id}",
                            "--stage",
                            "generate",
                        ],
                        "expected_outputs": ["generated.mp4", "record-patch.json"],
                    },
                    {
                        "name": "visual",
                        "depends_on": ["generate"],
                        "command": [
                            sys.executable,
                            str(worker),
                            "--job-dir",
                            "{job_dir}",
                            "--case-id",
                            "{case_id}",
                            "--stage",
                            "visual",
                        ],
                        "expected_outputs": ["record-patch.json"],
                    },
                ],
            }
        )
        + "\n"
    )
    run_dir = tmp_path / "run"
    manifest = plan_batch_run(
        suite_path=PUBLIC_VISUAL_PILOT / "suite.json",
        method_path=method,
        output_dir=run_dir,
    )
    assert manifest["job_count"] == 6
    controller = BatchController(run_dir)
    status = controller.run(max_workers=3)
    assert status["complete"] is True
    assert controller.run(max_workers=3) == status

    output = tmp_path / "submission.json"
    payload = compile_submission(run_dir=run_dir, output=output)
    assert payload["schema_version"] == "0.2.0"
    assert len(payload["records"]) == 3
    assert all(record["visual"]["goal_completion"] == 1.0 for record in payload["records"])

    first_job = manifest["jobs"][0]
    artifact = run_dir / "jobs" / first_job / "generated.mp4"
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        BatchController(run_dir).run()


def test_batch_retry_is_bounded_to_one_attempt_per_invocation(tmp_path: Path) -> None:
    method = tmp_path / "method.json"
    method.write_text(
        json.dumps(
            {
                "schema_version": "0.2.0",
                "method": "failing-fixture",
                "working_directory": str(tmp_path),
                "stages": [
                    {
                        "name": "fail",
                        "command": [sys.executable, "-c", "raise SystemExit(2)"],
                    }
                ],
            }
        )
    )
    run_dir = tmp_path / "run"
    manifest = plan_batch_run(
        suite_path=PUBLIC_VISUAL_PILOT / "suite.json",
        method_path=method,
        output_dir=run_dir,
    )
    controller = BatchController(run_dir)
    first = controller.run(max_workers=3)
    assert first["counts"]["failed"] == 3
    second = controller.run(max_workers=3, retry_failed=True)
    assert second["counts"]["failed"] == 3
    assert all(
        json.loads((run_dir / "jobs" / job / "status.json").read_text())["attempts"] == 2
        for job in manifest["jobs"]
    )


def test_harnesseval_adapter_is_visual_only_and_emits_explicit_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        HarnessEvalWAdapter,
        "preflight",
        lambda self: {
            "status": "ready",
            "executable": "/usr/bin/harnesseval",
            "claim_boundary": "visual only",
        },
    )
    command = HarnessEvalWAdapter(tmp_path, "revision").command(
        results=tmp_path / "results",
        model_id="phiagent-test",
        run_root=tmp_path / "run",
        manifest=tmp_path / "manifest.json",
        plan_root=tmp_path / "plans",
    )
    assert command[:2] == ["/usr/bin/harnesseval", "eval"]
    assert command[command.index("--model-id") + 1] == "phiagent-test"


def test_real_plan_expands_trials_and_stays_blocked_for_evidence_only_adapter(
    tmp_path: Path,
) -> None:
    suite = BenchmarkSuite.from_json(SMOKE / "suite.json")
    source_submission = Submission.from_json(SMOKE / "submission-reference.json")
    case = replace(
        suite.cases[0],
        annotation={
            **suite.cases[0].annotation,
            "real_protocol_id": "phiagent-real-robot-blind-v0.2",
        },
    )
    suite_v2 = replace(
        suite,
        cases=(case,),
        schema_version="0.2.0",
        policy_id="phiagent-real-pilot-v0.2",
    )
    source_record = source_submission.records[0]
    simulations = tuple(
        replace(
            source_record.simulation,
            episode_id=f"episode-{index}",
            initial_state_id=f"state-{index}",
            seed=index,
            artifact_hashes={"trace": f"{index + 1:064x}"},
        )
        for index in range(5)
    )
    submission_v2 = replace(
        source_submission,
        records=(
            replace(
                source_record,
                simulation=None,
                simulations=simulations,
                real=None,
            ),
        ),
        schema_version="0.2.0",
    )
    suite_path = tmp_path / "suite.json"
    submission_path = tmp_path / "submission.json"
    suite_path.write_text(json.dumps(suite_v2.to_dict()))
    submission_path.write_text(json.dumps(submission_v2.to_dict()))
    output = tmp_path / "real-plan"
    summary = create_real_trial_plan(
        suite_path=suite_path,
        submission_path=submission_path,
        policy_path=ROOT / "benchmark" / "policies" / "real-pilot-v0.2.json",
        protocol_path=ROOT
        / "benchmark"
        / "protocols"
        / "real-robot-blind-v0.2.json",
        adapter_path=ROOT / "benchmark" / "adapters" / "rm65-ag2f90d-recorded.json",
        session_ids=("session-a", "session-b", "session-c"),
        output_dir=output,
        random_seed=7,
    )
    assert summary["planned_trial_count"] == 10
    assert summary["session_count"] == 3
    assert summary["hardware_control_invoked"] is False
    assert summary["status"] == "blocked_pending_site_authorization"
    assert len(summary["schedule_hashes"]["operator_schedule_sha256"]) == 64
    assert len(summary["inputs"]["policy_sha256"]) == 64
    schedule = json.loads((output / "operator-schedule.json").read_text())
    assert len(schedule["rows"]) == 10
    assert all("method" not in row for row in schedule["rows"])


def test_physical_gate_derives_rates_from_every_sample() -> None:
    payload = {
        "schema_version": "0.2.0",
        "backend": "mujoco-3.3.7",
        "source_revision": "fixture-revision",
        "episode_id": "episode-001",
        "initial_state_id": "state-001",
        "seed": 7,
        "task_success": True,
        "stage_results": [True, True],
        "contact_results": [True, True],
        "samples": [
            {
                "timestamp_s": timestamp,
                "ik_success": True,
                "joint_limit_violation": False,
                "velocity_violation": False,
                "forbidden_collision": index == 1,
                "singularity": False,
            }
            for index, timestamp in enumerate((0.0, 0.1, 0.2, 0.3))
        ],
    }
    evidence = simulation_evidence_from_trace(payload, trace_sha256="a" * 64)
    assert evidence.physical_gate_complete is True
    assert evidence.collision_rate == 0.25
    assert evidence.physically_valid is False
    assert evidence.artifact_hashes["physical_gate_trace"] == "a" * 64


def test_physical_gate_rejects_incomplete_or_unsorted_samples() -> None:
    sample = {
        "timestamp_s": 0.0,
        "ik_success": True,
        "joint_limit_violation": False,
        "velocity_violation": False,
        "forbidden_collision": False,
        "singularity": False,
    }
    payload = {
        "schema_version": "0.2.0",
        "backend": "mujoco",
        "source_revision": "revision",
        "episode_id": "episode",
        "initial_state_id": "state",
        "seed": 0,
        "task_success": True,
        "stage_results": [True],
        "contact_results": [True],
        "samples": [sample, sample],
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        simulation_evidence_from_trace(payload, trace_sha256="a" * 64)
    payload["samples"] = [sample, {key: value for key, value in sample.items() if key != "singularity"}]
    payload["samples"][1]["timestamp_s"] = 0.1
    with pytest.raises(ValueError, match="lacks required flags"):
        simulation_evidence_from_trace(payload, trace_sha256="a" * 64)
