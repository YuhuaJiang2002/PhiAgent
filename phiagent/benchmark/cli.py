"""Command-line entry point for dependency-light PhiAgent-Bench evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phiagent.acwm.real_robot import RealRobotTrialEvidence
from phiagent.benchmark.adapters import (
    HarnessEvalWAdapter,
    RoboWMBenchAdapter,
    h2r_judge_packet,
    real_evidence_from_recorded_trial,
)
from phiagent.benchmark.batch import BatchController, compile_submission, plan_batch_run
from phiagent.benchmark.embodiments import EmbodimentRegistry
from phiagent.benchmark.h2r import H2RAnnotation, H2RJudgeOutput, aggregate_h2r_judges
from phiagent.benchmark.hardware import HardwareAdapterManifest
from phiagent.benchmark.integrity import verify_freeze_manifest
from phiagent.benchmark.metrics import BenchmarkPolicy, evaluate_submission
from phiagent.benchmark.physical_gate import simulation_evidence_from_trace_file
from phiagent.benchmark.real_plan import create_real_trial_plan
from phiagent.benchmark.schema import BenchmarkSuite, Submission
from phiagent.benchmark.trajectory import (
    ActionTrajectory,
    MultiArmActionTrajectory,
    compare_action_trajectories,
    compare_multi_arm_trajectories,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(rendered, end="")
        return
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    diff = run("diff", "--binary", "HEAD")
    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short").splitlines(),
        "working_tree_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def _package_inventory() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[str(name)] = distribution.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phiagent-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate exact suite coverage")
    validate.add_argument("--suite", type=Path, required=True)
    validate.add_argument("--submission", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="compute L1--L5 metrics")
    evaluate.add_argument("--suite", type=Path, required=True)
    evaluate.add_argument("--submission", type=Path, required=True)
    evaluate.add_argument("--policy", type=Path)
    evaluate.add_argument("--output", type=Path)

    run = subparsers.add_parser("run", help="evaluate into a new provenance-complete run")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--submission", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--policy", type=Path)
    run.add_argument("--bootstrap-iterations", type=int, default=2_000)
    run.add_argument("--bootstrap-seed", type=int, default=20260831)

    leaderboard = subparsers.add_parser("leaderboard", help="rank complete submissions")
    leaderboard.add_argument("--suite", type=Path, required=True)
    leaderboard.add_argument("--submissions", type=Path, nargs="+", required=True)
    leaderboard.add_argument("--policy", type=Path)
    leaderboard.add_argument("--output", type=Path)

    judge = subparsers.add_parser("h2r-score", help="aggregate three structured H2R judges")
    judge.add_argument("--annotation", type=Path, required=True)
    judge.add_argument("--judge", type=Path, nargs=3, required=True)
    judge.add_argument("--video-quality", type=Path, required=True)
    judge.add_argument("--output", type=Path)

    packet = subparsers.add_parser("h2r-packet", help="emit a case-specific judge packet")
    packet.add_argument("--suite", type=Path, required=True)
    packet.add_argument("--case-id", required=True)
    packet.add_argument("--output", type=Path)

    robowm = subparsers.add_parser("robowm-command", help="emit a pinned upstream replay command")
    robowm.add_argument("--checkout", type=Path, required=True)
    robowm.add_argument("--revision", required=True)
    robowm.add_argument("--task", required=True)
    robowm.add_argument("--trajectory-root", type=Path, required=True)
    robowm.add_argument("--output-root", type=Path, required=True)
    robowm.add_argument("--episode-index", type=int, required=True)
    robowm.add_argument("--device", default="cpu")
    robowm.add_argument("--episode-sha256")
    robowm.add_argument("--pose-sha256")

    robowm_preflight = subparsers.add_parser(
        "robowm-preflight", help="inspect optional Isaac Lab 5.1 prerequisites"
    )
    robowm_preflight.add_argument("--checkout", type=Path, required=True)
    robowm_preflight.add_argument("--revision", required=True)
    robowm_preflight.add_argument("--output", type=Path)

    action = subparsers.add_parser("action-metrics", help="compare synchronized L3 action files")
    action.add_argument("--reference", type=Path, required=True)
    action.add_argument("--candidate", type=Path, required=True)
    action.add_argument("--gripper-closed-threshold-m", type=float, default=0.01)
    action.add_argument("--event-tolerance-s", type=float, default=0.15)
    action.add_argument("--output", type=Path)

    physical_gate = subparsers.add_parser(
        "physical-gate", help="normalize a complete per-step simulator trace into L4 evidence"
    )
    physical_gate.add_argument("--trace", type=Path, required=True)
    physical_gate.add_argument("--output", type=Path, required=True)

    hardware = subparsers.add_parser("adapter-check", help="check an L5 hardware manifest")
    hardware.add_argument("--manifest", type=Path, required=True)
    hardware.add_argument("--suite", type=Path, required=True)
    hardware.add_argument("--output", type=Path)

    registry = subparsers.add_parser("registry-check", help="validate embodiment sources")
    registry.add_argument("--registry", type=Path, required=True)
    registry.add_argument("--output", type=Path)

    freeze = subparsers.add_parser("freeze-check", help="verify frozen source hashes")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--repository-root", type=Path, default=Path.cwd())
    freeze.add_argument("--output", type=Path)

    real_trial = subparsers.add_parser(
        "real-trial-check",
        help="validate and hash a recorded blind real-robot trial without commanding hardware",
    )
    real_trial.add_argument("--descriptor", type=Path, required=True)
    real_trial.add_argument("--adapter-manifest", type=Path, required=True)
    real_trial.add_argument("--protocol", type=Path, required=True)
    real_trial.add_argument("--session-id", required=True)
    real_trial.add_argument("--trial-index", type=int, required=True)
    real_trial.add_argument("--reviewer-id-hash", required=True)
    real_trial.add_argument("--output", type=Path, required=True)

    batch_plan = subparsers.add_parser(
        "batch-plan", help="expand a suite and method manifest into immutable jobs"
    )
    batch_plan.add_argument("--suite", type=Path, required=True)
    batch_plan.add_argument("--method", type=Path, required=True)
    batch_plan.add_argument("--output-dir", type=Path, required=True)

    batch_run = subparsers.add_parser(
        "batch-run", help="run or resume dependency-ordered benchmark jobs"
    )
    batch_run.add_argument("--run-dir", type=Path, required=True)
    batch_run.add_argument("--max-workers", type=int, default=1)
    batch_run.add_argument("--retry-failed", action="store_true")
    batch_run.add_argument(
        "--gpu-device",
        action="append",
        default=[],
        help="physical GPU index/UUID in the local scheduling pool; repeat for multiple GPUs",
    )

    batch_status = subparsers.add_parser("batch-status", help="summarize batch job states")
    batch_status.add_argument("--run-dir", type=Path, required=True)
    batch_status.add_argument("--output", type=Path)

    batch_compile = subparsers.add_parser(
        "batch-compile", help="compile selected immutable job evidence into a submission"
    )
    batch_compile.add_argument("--run-dir", type=Path, required=True)
    batch_compile.add_argument("--selection", type=Path)
    batch_compile.add_argument("--output", type=Path, required=True)

    real_plan = subparsers.add_parser(
        "real-plan", help="create a blinded repeated-trial schedule without hardware control"
    )
    real_plan.add_argument("--suite", type=Path, required=True)
    real_plan.add_argument("--submission", type=Path, required=True)
    real_plan.add_argument("--policy", type=Path, required=True)
    real_plan.add_argument("--protocol", type=Path, required=True)
    real_plan.add_argument("--adapter-manifest", type=Path, required=True)
    real_plan.add_argument("--session-id", action="append", required=True)
    real_plan.add_argument("--random-seed", type=int, default=20260831)
    real_plan.add_argument("--output-dir", type=Path, required=True)

    harness_preflight = subparsers.add_parser(
        "harnesseval-preflight", help="verify a pinned external HarnessEval-W checkout"
    )
    harness_preflight.add_argument("--checkout", type=Path, required=True)
    harness_preflight.add_argument("--revision", required=True)
    harness_preflight.add_argument("--output", type=Path)

    harness_command = subparsers.add_parser(
        "harnesseval-command", help="emit a pinned HarnessEval-W visual evaluation command"
    )
    harness_command.add_argument("--checkout", type=Path, required=True)
    harness_command.add_argument("--revision", required=True)
    harness_command.add_argument("--results", type=Path, required=True)
    harness_command.add_argument("--model-id", required=True)
    harness_command.add_argument("--run-root", type=Path, required=True)
    harness_command.add_argument("--manifest", type=Path, required=True)
    harness_command.add_argument("--plan-root", type=Path, required=True)
    return parser


def _load_pair(suite_path: Path, submission_path: Path) -> tuple[BenchmarkSuite, Submission]:
    suite = BenchmarkSuite.from_json(suite_path.expanduser().resolve())
    submission = Submission.from_json(submission_path.expanduser().resolve())
    evaluate_submission(suite, submission, BenchmarkPolicy(bootstrap_iterations=100))
    return suite, submission


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate":
        suite, submission = _load_pair(args.suite, args.submission)
        _write(
            {
                "status": "valid",
                "suite": suite.name,
                "method": submission.method,
                "case_count": len(suite.cases),
            },
            None,
        )
        return 0
    if args.command == "evaluate":
        suite = BenchmarkSuite.from_json(args.suite)
        submission = Submission.from_json(args.submission)
        policy = BenchmarkPolicy.from_json(args.policy) if args.policy else BenchmarkPolicy()
        _write(evaluate_submission(suite, submission, policy), args.output)
        return 0
    if args.command == "run":
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists():
            raise ValueError(f"benchmark run directory already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        suite_path = args.suite.expanduser().resolve()
        submission_path = args.submission.expanduser().resolve()
        suite_payload = _json(suite_path)
        submission_payload = _json(submission_path)
        suite = BenchmarkSuite.from_dict(suite_payload)
        submission = Submission.from_dict(submission_payload)
        policy = (
            BenchmarkPolicy.from_json(args.policy)
            if args.policy
            else BenchmarkPolicy(
                bootstrap_iterations=args.bootstrap_iterations,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
        report = evaluate_submission(suite, submission, policy)
        _write(suite_payload, output_dir / "suite.json")
        _write(submission_payload, output_dir / "submission.json")
        _write(report, output_dir / "report.json")
        _write(
            {
                "schema_version": "0.2.0",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "command": sys.argv,
                "inputs": {
                    "suite": str(suite_path),
                    "suite_sha256": _sha256(suite_path),
                    "submission": str(submission_path),
                    "submission_sha256": _sha256(submission_path),
                },
                "policy": {
                    **policy.to_dict(),
                    "source": str(args.policy.expanduser().resolve()) if args.policy else None,
                    "source_sha256": _sha256(args.policy.expanduser().resolve())
                    if args.policy
                    else None,
                },
                "gpu_selection": {
                    "mode": "cpu_only",
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
                "git": _git_state(Path.cwd()),
                "packages": _package_inventory(),
            },
            output_dir / "provenance.json",
        )
        _write(
            {
                "status": "complete",
                "run_dir": str(output_dir),
                "report": str(output_dir / "report.json"),
            },
            None,
        )
        return 0
    if args.command == "leaderboard":
        suite = BenchmarkSuite.from_json(args.suite)
        policy = BenchmarkPolicy.from_json(args.policy) if args.policy else BenchmarkPolicy()
        rows = [
            evaluate_submission(suite, Submission.from_json(path), policy)
            for path in args.submissions
        ]

        def primary_key(row: dict[str, Any]) -> tuple[float, float, float]:
            real = row["real_audit"]["e2e_valid_success_rate"]
            sim = row["dimension_scores"]["l4_sim"]
            visual = row["h2r_core"]
            return (
                float(real) if real is not None else -1.0,
                float(sim) if sim is not None else -1.0,
                float(visual) if visual is not None else -1.0,
            )

        def dimension_complete(row: dict[str, Any], dimension: str) -> bool:
            required_rows = [
                case for case in row["per_case"] if case["gates"][dimension] is not None
            ]
            return bool(required_rows) and all(
                dimension not in case["missing_dimensions"] for case in required_rows
            )

        def view(
            key_name: str,
            score: Any,
            eligible: Any,
        ) -> list[dict[str, Any]]:
            ordered = sorted(
                rows,
                key=lambda row: (
                    bool(eligible(row)),
                    float(score(row)) if score(row) is not None else -1.0,
                    row["method"],
                ),
                reverse=True,
            )
            current_rank = 0
            output_rows: list[dict[str, Any]] = []
            for row in ordered:
                accepted = bool(eligible(row))
                if accepted:
                    current_rank += 1
                output_rows.append(
                    {
                        "rank": current_rank if accepted else None,
                        "eligible": accepted,
                        "method": row["method"],
                        key_name: score(row),
                    }
                )
            return output_rows

        primary_rows = sorted(
            rows,
            key=lambda row: (
                bool(row["complete"] and row["protocol_complete"]),
                *primary_key(row),
            ),
            reverse=True,
        )
        primary = [
            {
                "rank": rank
                if row["complete"] and row["protocol_complete"]
                else None,
                "eligible": row["complete"] and row["protocol_complete"],
                "method": row["method"],
                "dimension_scores": row["dimension_scores"],
                "h2r_core": row["h2r_core"],
                "real_audit": row["real_audit"],
                "efficiency": row["efficiency"],
            }
            for rank, row in enumerate(primary_rows, start=1)
        ]
        _write(
            {
                "schema_version": "0.2.0",
                "suite": suite.name,
                "leaderboard": primary,
                "leaderboards": {
                    "primary_gated": primary,
                    "visual_h2r_core": view(
                        "h2r_core",
                        lambda row: row["h2r_core"],
                        lambda row: dimension_complete(row, "l1_visual"),
                    ),
                    "action_l3": view(
                        "l3_score",
                        lambda row: row["dimension_scores"]["l3_action"],
                        lambda row: dimension_complete(row, "l3_action"),
                    ),
                    "simulation_l4": view(
                        "l4_score",
                        lambda row: row["dimension_scores"]["l4_sim"],
                        lambda row: dimension_complete(row, "l4_sim"),
                    ),
                    "real_e2e": view(
                        "e2e_valid_success_rate",
                        lambda row: row["real_audit"]["e2e_valid_success_rate"],
                        lambda row: dimension_complete(row, "l5_real"),
                    ),
                    "policy_utility": view(
                        "mean_delta_real_success_rate",
                        lambda row: row["policy_utility"][
                            "mean_delta_real_success_rate"
                        ],
                        lambda row: row["policy_utility"]["matched_records"] > 0,
                    ),
                },
            },
            args.output,
        )
        return 0
    if args.command == "h2r-score":
        annotation = H2RAnnotation.from_dict(_json(args.annotation))
        judges = tuple(H2RJudgeOutput.from_dict(_json(path)) for path in args.judge)
        quality = _json(args.video_quality)
        _write(
            aggregate_h2r_judges(
                annotation,
                judges,
                video_quality_components=quality,
            ).to_dict(),
            args.output,
        )
        return 0
    if args.command == "h2r-packet":
        suite = BenchmarkSuite.from_json(args.suite)
        matches = [case for case in suite.cases if case.case_id == args.case_id]
        if len(matches) != 1:
            raise ValueError(f"unknown case_id: {args.case_id}")
        _write(h2r_judge_packet(matches[0]), args.output)
        return 0
    if args.command == "robowm-command":
        adapter = RoboWMBenchAdapter(args.checkout, args.revision)
        if bool(args.episode_sha256) != bool(args.pose_sha256):
            raise ValueError("frozen replay requires both episode and pose SHA-256 values")
        frozen = (
            adapter.verify_frozen_episode(
                trajectory_root=args.trajectory_root,
                episode_index=args.episode_index,
                episode_sha256=args.episode_sha256,
                pose_sha256=args.pose_sha256,
            )
            if args.episode_sha256
            else None
        )
        command = adapter.command(
            task=args.task,
            trajectory_root=args.trajectory_root,
            output_root=args.output_root,
            episode_index=args.episode_index,
            device=args.device,
        )
        _write(
            {"command": command, "shell_preview": shlex.join(command), "frozen_inputs": frozen},
            None,
        )
        return 0
    if args.command == "robowm-preflight":
        adapter = RoboWMBenchAdapter(args.checkout, args.revision)
        _write(adapter.runtime_preflight(), args.output)
        return 0
    if args.command == "action-metrics":
        reference_payload = _json(args.reference)
        candidate_payload = _json(args.candidate)
        if ("arms" in reference_payload) != ("arms" in candidate_payload):
            raise ValueError("reference and candidate must both be single-arm or multi-arm")
        if "arms" in reference_payload:
            reference_bundle = MultiArmActionTrajectory.from_dict(reference_payload)
            candidate_bundle = MultiArmActionTrajectory.from_dict(candidate_payload)
            values, per_arm = compare_multi_arm_trajectories(
                reference_bundle,
                candidate_bundle,
                gripper_closed_threshold_m=args.gripper_closed_threshold_m,
                event_tolerance_s=args.event_tolerance_s,
            )
            _write(
                {
                    "coordinate_frame": reference_bundle.coordinate_frame,
                    "aggregation": "worst_arm",
                    "values": values,
                    "per_arm": per_arm,
                },
                args.output,
            )
            return 0
        reference = ActionTrajectory.from_dict(reference_payload)
        candidate = ActionTrajectory.from_dict(candidate_payload)
        _write(
            {
                "coordinate_frame": reference.coordinate_frame,
                "values": compare_action_trajectories(
                    reference,
                    candidate,
                    gripper_closed_threshold_m=args.gripper_closed_threshold_m,
                    event_tolerance_s=args.event_tolerance_s,
                ),
            },
            args.output,
        )
        return 0
    if args.command == "physical-gate":
        evidence = simulation_evidence_from_trace_file(args.trace)
        _write(
            {
                "status": "normalized_physical_gate",
                "claim_boundary": (
                    "Rates are derived from every trace sample. Task and contact outcomes "
                    "remain assertions of the named simulator backend, not real-robot proof."
                ),
                "evidence": evidence.to_dict(),
            },
            args.output,
        )
        return 0
    if args.command == "adapter-check":
        manifest = HardwareAdapterManifest.from_json(args.manifest)
        suite = BenchmarkSuite.from_json(args.suite)
        results = [manifest.compatibility(case) for case in suite.cases]
        _write(
            {
                "adapter_name": manifest.adapter_name,
                "adapter_version": manifest.adapter_version,
                "all_compatible": all(result["compatible"] for result in results),
                "cases": results,
            },
            args.output,
        )
        return 0
    if args.command == "registry-check":
        registry = EmbodimentRegistry.from_json(args.registry)
        _write(registry.summary(), args.output)
        return 0
    if args.command == "freeze-check":
        _write(
            verify_freeze_manifest(
                args.manifest,
                repository_root=args.repository_root,
            ),
            args.output,
        )
        return 0
    if args.command == "real-trial-check":
        descriptor = args.descriptor.expanduser().resolve()
        trial = RealRobotTrialEvidence.from_dict(_json(descriptor), root=descriptor.parent)
        protocol = _json(args.protocol)
        protocol_id = str(protocol.get("protocol_id", ""))
        registration = _json(trial.pre_registered_case_manifest)
        if registration.get("protocol_id") != protocol_id:
            raise ValueError("pre-registration protocol_id does not match the selected protocol")
        adapter = HardwareAdapterManifest.from_json(args.adapter_manifest)
        if registration.get("adapter_name") != adapter.adapter_name:
            raise ValueError("pre-registration adapter does not match the selected adapter")
        if registration.get("trial_index") != args.trial_index:
            raise ValueError("pre-registration trial_index does not match the requested trial")
        calibration = _json(trial.calibration)
        site_safety_approved = registration.get("site_safety_approved", False)
        if not isinstance(site_safety_approved, bool):
            raise ValueError("pre-registration site_safety_approved must be boolean")
        eligibility_checks = {
            "adapter_execution_enabled": adapter.execution_enabled
            and not adapter.evidence_only,
            "calibration_bound_to_robot_serial": calibration.get("hardware_serial")
            == trial.hardware_serial,
            "action_and_scene_hash_frozen_before_execution": True,
            "site_safety_approved": site_safety_approved,
        }
        evidence = real_evidence_from_recorded_trial(
            trial,
            adapter_name=adapter.adapter_name,
            session_id=args.session_id,
            protocol_id=protocol_id,
            trial_index=args.trial_index,
            reviewer_id_hash=args.reviewer_id_hash,
            pre_registered=True,
            method_blind_code=(
                str(registration["method_blind_code"])
                if registration.get("method_blind_code")
                else None
            ),
            eligibility_checks=eligibility_checks,
        )
        _write(
            {
                "status": "validated_recorded_evidence",
                "hardware_control_invoked": False,
                "adapter_execution_enabled": adapter.execution_enabled,
                "evidence_only": adapter.evidence_only,
                "eligible_for_l5": all(eligibility_checks.values()),
                "eligibility_checks": eligibility_checks,
                "evidence": evidence.to_dict(),
            },
            args.output,
        )
        return 0
    if args.command == "batch-plan":
        _write(
            plan_batch_run(
                suite_path=args.suite,
                method_path=args.method,
                output_dir=args.output_dir,
            ),
            None,
        )
        return 0
    if args.command == "batch-run":
        _write(
            BatchController(args.run_dir).run(
                max_workers=args.max_workers,
                retry_failed=args.retry_failed,
                gpu_devices=tuple(args.gpu_device),
            ),
            None,
        )
        return 0
    if args.command == "batch-status":
        _write(BatchController(args.run_dir).status(), args.output)
        return 0
    if args.command == "batch-compile":
        payload = compile_submission(
            run_dir=args.run_dir,
            output=args.output,
            selection_path=args.selection,
        )
        _write(
            {
                "status": "compiled",
                "method": payload["method"],
                "case_count": len(payload["records"]),
                "output": str(args.output.expanduser().resolve()),
            },
            None,
        )
        return 0
    if args.command == "real-plan":
        _write(
            create_real_trial_plan(
                suite_path=args.suite,
                submission_path=args.submission,
                policy_path=args.policy,
                protocol_path=args.protocol,
                adapter_path=args.adapter_manifest,
                session_ids=tuple(args.session_id),
                output_dir=args.output_dir,
                random_seed=args.random_seed,
            ),
            None,
        )
        return 0
    if args.command == "harnesseval-preflight":
        _write(
            HarnessEvalWAdapter(args.checkout, args.revision).preflight(),
            args.output,
        )
        return 0
    if args.command == "harnesseval-command":
        command = HarnessEvalWAdapter(args.checkout, args.revision).command(
            results=args.results,
            model_id=args.model_id,
            run_root=args.run_root,
            manifest=args.manifest,
            plan_root=args.plan_root,
        )
        _write({"command": command, "shell_preview": shlex.join(command)}, None)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
