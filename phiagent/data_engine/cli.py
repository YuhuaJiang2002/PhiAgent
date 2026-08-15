"""Command-line control plane for planning and sizing video data campaigns."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from phiagent.data_engine.capacity import (
    CapacityAssumptions,
    estimate_capacity,
    load_profiles,
)
from phiagent.data_engine.controller import CampaignController
from phiagent.data_engine.planner import compile_campaign
from phiagent.data_engine.plugins import PluginRegistry
from phiagent.data_engine.provenance import capture_provenance, utc_now, write_json_atomic
from phiagent.data_engine.schema import CampaignSpec
from phiagent.data_engine.state import AuditReport, CampaignState


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phiagent-data-engine",
        description="Plan, audit, and capacity-size cross-embodiment video campaigns.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plugins = subparsers.add_parser("plugins", help="list the plugin contract surface")
    plugins.add_argument("--discover", action="store_true", help="load installed entry points")

    plan = subparsers.add_parser("plan", help="compile a campaign into immutable jobs")
    plan.add_argument("campaign", type=Path)
    plan.add_argument("--output-root", type=Path, default=Path("outputs/data-engine"))

    estimate = subparsers.add_parser("estimate", help="estimate accepted-video capacity")
    estimate.add_argument("profiles", type=Path)
    estimate.add_argument("--profile", required=True)
    estimate.add_argument("--target-hours", type=float, default=100.0)
    estimate.add_argument("--accelerators", type=int, default=32)
    estimate.add_argument("--yield", dest="first_pass_yield", type=float, default=0.8)
    estimate.add_argument("--utilization", type=float, default=0.85)
    estimate.add_argument("--overhead", type=float, default=0.15)
    estimate.add_argument("--average-clip-seconds", type=float, default=10.0)
    estimate.add_argument("--reviewers", type=int, default=4)

    dashboard = subparsers.add_parser(
        "export-dashboard", help="export measured scenarios for the static demo"
    )
    dashboard.add_argument("campaign", type=Path)
    dashboard.add_argument("profiles", type=Path)
    dashboard.add_argument("--profile", required=True)
    dashboard.add_argument("--output", type=Path, required=True)
    dashboard.add_argument("--force", action="store_true")

    status = subparsers.add_parser("status", help="show persisted campaign state")
    status.add_argument("run_dir", type=Path)

    claim = subparsers.add_parser("claim", help="atomically lease one job to a worker")
    claim.add_argument("run_dir", type=Path)
    claim.add_argument("--worker", required=True)
    claim.add_argument("--job-id")
    claim.add_argument("--retry-rejected", action="store_true")

    submit = subparsers.add_parser("submit", help="submit a hash-bound artifact manifest")
    submit.add_argument("run_dir", type=Path)
    submit.add_argument("--job-id", required=True)
    submit.add_argument("--worker", required=True)
    submit.add_argument("--artifact-manifest-uri", required=True)
    submit.add_argument("--artifact-manifest-sha256", required=True)

    requeue = subparsers.add_parser(
        "requeue", help="recover a stranded running or audit-pending job"
    )
    requeue.add_argument("run_dir", type=Path)
    requeue.add_argument("--job-id", required=True)
    requeue.add_argument("--reason", required=True)

    audit = subparsers.add_parser("audit", help="apply an independent audit report")
    audit.add_argument("run_dir", type=Path)
    audit.add_argument("report", type=Path)
    return parser


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _new_run_directory(output_root: Path, campaign_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = output_root.expanduser().resolve() / f"{timestamp}-{campaign_id}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _plan(args: argparse.Namespace) -> int:
    campaign_path = args.campaign.expanduser().resolve()
    campaign = CampaignSpec.from_json(campaign_path)
    plan = compile_campaign(campaign)
    run_dir = _new_run_directory(args.output_root, campaign.campaign_id)
    shutil.copy2(campaign_path, run_dir / "campaign.json")
    write_json_atomic(run_dir / "plan.json", plan.to_dict())
    write_json_atomic(run_dir / "state.json", CampaignState.from_plan(plan).to_dict())
    write_json_atomic(
        run_dir / "provenance.json",
        capture_provenance(_repository_root(), sys.argv, campaign.seed),
    )
    events = run_dir / "logs" / "events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=False)
    events.write_text(
        json.dumps(
            {
                "event": "campaign_planned",
                "at": utc_now(),
                "campaign_id": campaign.campaign_id,
                "jobs": len(plan.jobs),
                "plan_sha256": plan.plan_sha256,
            },
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "campaign_id": campaign.campaign_id,
                "jobs": len(plan.jobs),
                "plan_sha256": plan.plan_sha256,
                "run_dir": str(run_dir),
                "useful_video_hours": plan.useful_video_hours,
                "generated_window_hours": plan.generated_window_hours,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _assumptions(args: argparse.Namespace) -> CapacityAssumptions:
    return CapacityAssumptions(
        target_accepted_hours=args.target_hours,
        accelerator_count=args.accelerators,
        first_pass_yield=args.first_pass_yield,
        utilization=args.utilization,
        non_generation_overhead_fraction=args.overhead,
        average_clip_seconds=args.average_clip_seconds,
        reviewer_count=args.reviewers,
    )


def _estimate(args: argparse.Namespace) -> int:
    profiles = load_profiles(args.profiles.expanduser().resolve())
    if args.profile not in profiles:
        raise ValueError(f"unknown benchmark profile {args.profile!r}")
    estimate = estimate_capacity(profiles[args.profile], _assumptions(args))
    print(json.dumps(estimate.to_dict(), indent=2, sort_keys=True))
    return 0


def _export_dashboard(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite dashboard export: {output}")
    campaign = CampaignSpec.from_json(args.campaign.expanduser().resolve())
    plan = compile_campaign(campaign)
    profiles = load_profiles(args.profiles.expanduser().resolve())
    if args.profile not in profiles:
        raise ValueError(f"unknown benchmark profile {args.profile!r}")
    profile = profiles[args.profile]
    scenarios: list[dict[str, object]] = []
    for accelerator_count in (2, 8, 32, 64):
        for first_pass_yield in (0.6, 0.8, 0.9):
            estimate = estimate_capacity(
                profile,
                CapacityAssumptions(
                    target_accepted_hours=campaign.target_output_hours,
                    accelerator_count=accelerator_count,
                    first_pass_yield=first_pass_yield,
                ),
            )
            scenarios.append(estimate.to_dict())
    payload = {
        "schema_version": "1.0.0",
        "generated_at": utc_now(),
        "status": "PARTIAL",
        "claim_boundary": (
            "Measured infrastructure projection, not a completed 100-hour production run."
        ),
        "campaign": {
            "campaign_id": campaign.campaign_id,
            "target_output_hours": campaign.target_output_hours,
            "pilot_jobs": len(plan.jobs),
            "pilot_useful_video_hours": plan.useful_video_hours,
            "source_count": len(campaign.sources),
            "target_count": len(campaign.targets),
            "targets": [
                {
                    "target_id": target.target_id,
                    "replacement_scope": target.replacement_scope.value,
                }
                for target in campaign.targets
            ],
            "plan_sha256": plan.plan_sha256,
        },
        "profile": {
            "profile_id": profile.profile_id,
            "accelerator": profile.accelerator,
            "accelerators_per_worker": profile.accelerators_per_worker,
            "wall_seconds_per_output_second": profile.wall_seconds_per_output_second,
            "accelerator_seconds_per_output_second": (
                profile.accelerator_seconds_per_output_second
            ),
            "benchmark_uri": profile.benchmark_uri,
            "benchmark_sha256": profile.benchmark_sha256,
            "evidence_status": profile.evidence_status,
            "claim_boundary": profile.claim_boundary,
        },
        "scenarios": scenarios,
    }
    write_json_atomic(output, payload)
    print(json.dumps({"output": str(output), "scenarios": len(scenarios)}, indent=2))
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "plugins":
        registry = PluginRegistry()
        if args.discover:
            registry.discover()
        print(
            json.dumps(
                {"plugins": [item.to_dict() for item in registry.descriptors()]},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "plan":
        return _plan(args)
    if args.command == "estimate":
        return _estimate(args)
    if args.command == "export-dashboard":
        return _export_dashboard(args)
    if args.command == "status":
        print(json.dumps(CampaignController(args.run_dir).status(), indent=2, sort_keys=True))
        return 0
    if args.command == "claim":
        job = CampaignController(args.run_dir).claim(
            args.worker,
            job_id=args.job_id,
            retry_rejected=args.retry_rejected,
        )
        print(
            json.dumps(
                {"job": None if job is None else asdict(job)},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "submit":
        CampaignController(args.run_dir).submit(
            args.job_id,
            args.worker,
            args.artifact_manifest_uri,
            args.artifact_manifest_sha256,
        )
        print(json.dumps({"job_id": args.job_id, "status": "audit_pending"}, indent=2))
        return 0
    if args.command == "requeue":
        CampaignController(args.run_dir).requeue(args.job_id, args.reason)
        print(json.dumps({"job_id": args.job_id, "status": "rejected"}, indent=2))
        return 0
    if args.command == "audit":
        payload = json.loads(args.report.expanduser().resolve().read_text())
        status = CampaignController(args.run_dir).audit(AuditReport.from_dict(payload))
        print(json.dumps({"job_id": payload["job_id"], "status": status.value}, indent=2))
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
