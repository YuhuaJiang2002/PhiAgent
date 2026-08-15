from __future__ import annotations

import json
from pathlib import Path

import pytest

from phiagent.data_engine.capacity import (
    BenchmarkProfile,
    CapacityAssumptions,
    estimate_capacity,
)
from phiagent.data_engine.controller import CampaignController
from phiagent.data_engine.planner import CampaignPlan, compile_campaign, split_windows
from phiagent.data_engine.plugins import PluginRegistry
from phiagent.data_engine.schema import (
    PHYSICAL_REQUIRED_GATES,
    VISUAL_REQUIRED_GATES,
    CampaignSpec,
    ClaimScope,
    PipelineContract,
    ReplacementScope,
    SourceClip,
    TargetAsset,
)
from phiagent.data_engine.state import (
    AuditReport,
    CampaignState,
    EvidenceRef,
    JobStatus,
)


_HASH_A = "a" * 64
_HASH_B = "b" * 64


def _campaign(*, claim_scope: ClaimScope = ClaimScope.VISUAL_TRAINING_DATA) -> CampaignSpec:
    auditors = (
        ("local-video-auditor", "physical-auditor")
        if claim_scope is ClaimScope.PHYSICALLY_GROUNDED
        else ("local-video-auditor",)
    )
    gates = (
        PHYSICAL_REQUIRED_GATES
        if claim_scope is ClaimScope.PHYSICALLY_GROUNDED
        else VISUAL_REQUIRED_GATES
    )
    return CampaignSpec(
        campaign_id="test-campaign",
        seed=7,
        target_output_hours=100.0,
        sources=(
            SourceClip(
                source_id="source-01",
                uri="s3://bucket/source.mp4",
                sha256=_HASH_A,
                duration_seconds=12.0,
                fps=24.0,
                coordinate_frame="camera:ego",
                scene_group="scene-a",
                rights_basis="authorized synthetic fixture",
                tasks=("pick", "place"),
            ),
        ),
        targets=(
            TargetAsset(
                target_id="sharpa-right",
                replacement_scope=ReplacementScope.HAND,
                asset_uri="s3://assets/sharpa.json",
                asset_sha256=_HASH_B,
                coordinate_frame="robot_base:sharpa",
                retarget_plugin="dex-retarget",
            ),
        ),
        candidate_seeds=(42, 43),
        pipeline=PipelineContract(
            source_plugin="local-video-source",
            generator_plugin="wan-animate2",
            auditor_plugins=auditors,
            window_seconds=5.0,
            overlap_seconds=1.0,
            claim_scope=claim_scope,
            required_gates=gates,
        ),
    )


def _profile() -> BenchmarkProfile:
    return BenchmarkProfile(
        profile_id="wan-test",
        accelerator="A800",
        accelerators_per_worker=2,
        wall_seconds_per_output_second=30.0,
        accelerator_seconds_per_output_second=60.0,
        benchmark_uri="outputs/benchmark.json",
        benchmark_sha256=_HASH_A,
        evidence_status="WORKING",
        claim_boundary="measured infrastructure only",
    )


def test_source_and_target_require_explicit_coordinate_frames() -> None:
    with pytest.raises(ValueError, match="camera"):
        SourceClip(
            source_id="source-01",
            uri="source.mp4",
            sha256=_HASH_A,
            duration_seconds=1,
            fps=24,
            coordinate_frame="world",
            scene_group="scene-a",
            rights_basis="fixture",
            tasks=("pick",),
        )
    with pytest.raises(ValueError, match="robot_base"):
        TargetAsset(
            target_id="target-01",
            replacement_scope=ReplacementScope.HAND,
            asset_uri="asset.json",
            asset_sha256=_HASH_B,
            coordinate_frame="camera:ego",
            retarget_plugin="dex-retarget",
        )


def test_physical_campaign_cannot_drop_physical_hard_gates() -> None:
    with pytest.raises(ValueError, match="missing hard gates"):
        PipelineContract(
            source_plugin="local-video-source",
            generator_plugin="wan-animate2",
            auditor_plugins=("local-video-auditor",),
            window_seconds=5,
            overlap_seconds=1,
            claim_scope=ClaimScope.PHYSICALLY_GROUNDED,
            required_gates=VISUAL_REQUIRED_GATES,
        )


def test_window_split_preserves_useful_duration_and_accounts_for_overlap() -> None:
    windows = split_windows(12.0, window_seconds=5.0, overlap_seconds=1.0)

    assert [(item.source_start_seconds, item.source_end_seconds) for item in windows] == [
        (0.0, 5.0),
        (4.0, 9.0),
        (8.0, 12.0),
    ]
    assert sum(item.useful_seconds for item in windows) == pytest.approx(12.0)
    assert sum(item.source_end_seconds - item.source_start_seconds for item in windows) == 14.0


def test_campaign_plan_is_deterministic_and_expands_seed_matrix() -> None:
    first = compile_campaign(_campaign())
    second = compile_campaign(_campaign())

    assert first.plan_sha256 == second.plan_sha256
    assert len(first.jobs) == 2
    assert len({job.job_id for job in first.jobs}) == 2
    assert first.useful_video_hours == pytest.approx(24 / 3600)
    assert first.generated_window_hours == pytest.approx(28 / 3600)
    assert first.jobs[0].source_coordinate_frame == "camera:ego"
    assert first.jobs[0].target_coordinate_frame == "robot_base:sharpa"


def test_registry_rejects_generator_without_target_capability() -> None:
    payload = _campaign().to_dict()
    payload["pipeline"]["generator_plugin"] = "oscar"
    campaign = CampaignSpec.from_dict(payload)

    with pytest.raises(ValueError, match="does not support 'hand'"):
        PluginRegistry().validate_campaign(campaign)


def test_registry_rejects_retargeter_without_target_capability() -> None:
    payload = _campaign().to_dict()
    payload["targets"][0]["retarget_plugin"] = "epl-retarget"
    campaign = CampaignSpec.from_dict(payload)

    with pytest.raises(ValueError, match="does not support 'hand'"):
        PluginRegistry().validate_campaign(campaign)


def test_independent_audit_is_only_path_to_accepted_state() -> None:
    plan = compile_campaign(_campaign())
    job = plan.jobs[0]
    state = CampaignState.from_plan(plan)
    state = state.claim(job.job_id, "worker-a")
    state = state.submit(
        job.job_id, "worker-a", "s3://runs/artifact.json", _HASH_B
    )
    report = AuditReport(
        job_id=job.job_id,
        auditor="auditor-b",
        independent=True,
        gates={gate: True for gate in job.required_gates},
        evidence=(EvidenceRef("s3://runs/evidence.json", _HASH_A),),
        diagnostic_mean_score=0.91,
    )

    accepted = state.audit(job, report)

    assert accepted.jobs[0].status is JobStatus.ACCEPTED
    assert accepted.revision == 3


def test_mean_score_cannot_override_one_failed_hard_gate() -> None:
    plan = compile_campaign(_campaign())
    job = plan.jobs[0]
    state = CampaignState.from_plan(plan).claim(job.job_id, "worker-a")
    state = state.submit(
        job.job_id, "worker-a", "s3://runs/artifact.json", _HASH_B
    )
    gates = {gate: True for gate in job.required_gates}
    gates["object_preservation"] = False

    rejected = state.audit(
        job,
        AuditReport(
            job_id=job.job_id,
            auditor="auditor-b",
            independent=True,
            gates=gates,
            evidence=(EvidenceRef("s3://runs/evidence.json", _HASH_A),),
            diagnostic_mean_score=0.999,
        ),
    )

    assert rejected.jobs[0].status is JobStatus.REJECTED
    assert rejected.jobs[0].diagnoses == ("hard_gate_failed:object_preservation",)


def test_executor_self_audit_is_rejected() -> None:
    plan = compile_campaign(_campaign())
    job = plan.jobs[0]
    state = CampaignState.from_plan(plan).claim(job.job_id, "worker-a")
    state = state.submit(
        job.job_id, "worker-a", "s3://runs/artifact.json", _HASH_B
    )

    rejected = state.audit(
        job,
        AuditReport(
            job_id=job.job_id,
            auditor="worker-a",
            independent=True,
            gates={gate: True for gate in job.required_gates},
            evidence=(EvidenceRef("s3://runs/evidence.json", _HASH_A),),
        ),
    )

    assert rejected.jobs[0].status is JobStatus.REJECTED
    assert rejected.jobs[0].diagnoses == ("executor_self_audit",)


def test_audit_boolean_cannot_be_coerced_from_a_string() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        AuditReport.from_dict(
            {
                "job_id": "job-a",
                "auditor": "auditor-b",
                "independent": "false",
                "gates": {"source_lineage": True},
                "evidence": [{"uri": "evidence.json", "sha256": _HASH_A}],
            }
        )


def test_capacity_estimate_scales_workers_but_not_total_accelerator_hours() -> None:
    small = estimate_capacity(
        _profile(),
        CapacityAssumptions(target_accepted_hours=100, accelerator_count=8),
    )
    large = estimate_capacity(
        _profile(),
        CapacityAssumptions(target_accepted_hours=100, accelerator_count=32),
    )

    assert small.accelerator_hours == pytest.approx(large.accelerator_hours)
    assert small.generation_calendar_hours == pytest.approx(
        large.generation_calendar_hours * 4
    )
    assert small.raw_candidate_hours == 125.0
    assert small.accepted_clip_count == 36000
    assert small.raw_candidate_clip_count == 45000


def test_repository_pilot_manifest_compiles() -> None:
    root = Path(__file__).resolve().parents[1]
    campaign = CampaignSpec.from_json(root / "configs/data_engine/pilot-100h.json")
    plan = compile_campaign(campaign)

    assert len(plan.jobs) == 8
    assert {item.replacement_scope for item in plan.jobs} == {
        "hand",
        "full_embodiment",
    }
    assert json.loads(json.dumps(plan.to_dict()))["plan_sha256"] == plan.plan_sha256


def test_file_backed_controller_runs_recoverable_claim_submit_audit(
    tmp_path: Path,
) -> None:
    plan = compile_campaign(_campaign())
    run_dir = tmp_path / "campaign-run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "plan.json").write_text(json.dumps(plan.to_dict()))
    (run_dir / "state.json").write_text(
        json.dumps(CampaignState.from_plan(plan).to_dict())
    )
    (run_dir / "logs" / "events.jsonl").write_text("")
    controller = CampaignController(run_dir)

    job = controller.claim("worker-a")
    assert job is not None
    controller.submit(job.job_id, "worker-a", "s3://runs/artifact.json", _HASH_B)
    decision = controller.audit(
        AuditReport(
            job_id=job.job_id,
            auditor="auditor-b",
            independent=True,
            gates={gate: True for gate in job.required_gates},
            evidence=(EvidenceRef("s3://runs/evidence.json", _HASH_A),),
        )
    )

    assert decision is JobStatus.ACCEPTED
    summary = controller.status()
    assert summary["state_revision"] == 3
    assert summary["counts"]["accepted"] == 1
    persisted = json.loads((run_dir / "state.json").read_text())
    accepted = next(item for item in persisted["jobs"] if item["job_id"] == job.job_id)
    assert accepted["artifact_manifest_sha256"] == _HASH_B
    events = (run_dir / "logs" / "events.jsonl").read_text().splitlines()
    assert [json.loads(item)["event"] for item in events] == [
        "job_claimed",
        "job_submitted",
        "job_audited",
    ]


def test_campaign_plan_loader_rejects_tampering() -> None:
    payload = compile_campaign(_campaign()).to_dict()
    payload["target_output_hours"] = 101.0

    with pytest.raises(ValueError, match="plan hash mismatch"):
        CampaignPlan.from_dict(payload)


def test_controller_can_requeue_and_reclaim_a_stranded_job(tmp_path: Path) -> None:
    plan = compile_campaign(_campaign())
    run_dir = tmp_path / "campaign-run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "plan.json").write_text(json.dumps(plan.to_dict()))
    (run_dir / "state.json").write_text(
        json.dumps(CampaignState.from_plan(plan).to_dict())
    )
    (run_dir / "logs" / "events.jsonl").write_text("")
    controller = CampaignController(run_dir)

    job = controller.claim("dead-worker")
    assert job is not None
    controller.requeue(job.job_id, "worker heartbeat expired")
    reclaimed = controller.claim("recovery-worker", job_id=job.job_id)

    assert reclaimed == job
    state = CampaignState.from_dict(json.loads((run_dir / "state.json").read_text()))
    recovered = next(item for item in state.jobs if item.job_id == job.job_id)
    assert recovered.status is JobStatus.RUNNING
    assert recovered.attempts == 2
    assert recovered.worker_id == "recovery-worker"
