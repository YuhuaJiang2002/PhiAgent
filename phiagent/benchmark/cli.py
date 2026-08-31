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

from phiagent.benchmark.adapters import RoboWMBenchAdapter, h2r_judge_packet
from phiagent.benchmark.h2r import H2RAnnotation, H2RJudgeOutput, aggregate_h2r_judges
from phiagent.benchmark.hardware import HardwareAdapterManifest
from phiagent.benchmark.metrics import BenchmarkPolicy, evaluate_submission
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
    evaluate.add_argument("--output", type=Path)

    run = subparsers.add_parser("run", help="evaluate into a new provenance-complete run")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--submission", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--bootstrap-iterations", type=int, default=2_000)
    run.add_argument("--bootstrap-seed", type=int, default=20260831)

    leaderboard = subparsers.add_parser("leaderboard", help="rank complete submissions")
    leaderboard.add_argument("--suite", type=Path, required=True)
    leaderboard.add_argument("--submissions", type=Path, nargs="+", required=True)
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

    action = subparsers.add_parser("action-metrics", help="compare synchronized L3 action files")
    action.add_argument("--reference", type=Path, required=True)
    action.add_argument("--candidate", type=Path, required=True)
    action.add_argument("--gripper-closed-threshold-m", type=float, default=0.01)
    action.add_argument("--event-tolerance-s", type=float, default=0.15)
    action.add_argument("--output", type=Path)

    hardware = subparsers.add_parser("adapter-check", help="check an L5 hardware manifest")
    hardware.add_argument("--manifest", type=Path, required=True)
    hardware.add_argument("--suite", type=Path, required=True)
    hardware.add_argument("--output", type=Path)
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
        _write(evaluate_submission(suite, submission), args.output)
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
        policy = BenchmarkPolicy(
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
        report = evaluate_submission(suite, submission, policy)
        _write(suite_payload, output_dir / "suite.json")
        _write(submission_payload, output_dir / "submission.json")
        _write(report, output_dir / "report.json")
        _write(
            {
                "schema_version": "0.1.0",
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
                    "bootstrap_iterations": args.bootstrap_iterations,
                    "bootstrap_seed": args.bootstrap_seed,
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
        rows = [evaluate_submission(suite, Submission.from_json(path)) for path in args.submissions]

        def key(row: dict[str, Any]) -> tuple[float, float, float]:
            real = row["real_audit"]["e2e_valid_success_rate"]
            sim = row["dimension_scores"]["l4_sim"]
            visual = row["h2r_core"]
            return (
                float(real) if real is not None else -1.0,
                float(sim) if sim is not None else -1.0,
                float(visual) if visual is not None else -1.0,
            )

        rows.sort(key=lambda row: (bool(row["complete"]), *key(row)), reverse=True)
        _write(
            {
                "schema_version": "0.1.0",
                "suite": suite.name,
                "leaderboard": [
                    {
                        "rank": rank if row["complete"] else None,
                        "eligible": row["complete"],
                        "method": row["method"],
                        "dimension_scores": row["dimension_scores"],
                        "h2r_core": row["h2r_core"],
                        "real_audit": row["real_audit"],
                        "efficiency": row["efficiency"],
                    }
                    for rank, row in enumerate(rows, start=1)
                ],
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
        command = adapter.command(
            task=args.task,
            trajectory_root=args.trajectory_root,
            output_root=args.output_root,
            episode_index=args.episode_index,
            device=args.device,
        )
        _write({"command": command, "shell_preview": shlex.join(command)}, None)
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
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
