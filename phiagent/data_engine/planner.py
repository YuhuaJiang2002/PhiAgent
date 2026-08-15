"""Deterministic campaign expansion into resumable source/target/seed jobs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from phiagent.data_engine.plugins import PluginRegistry
from phiagent.data_engine.schema import CampaignSpec, SCHEMA_VERSION


@dataclass(frozen=True)
class WindowSpec:
    window_index: int
    source_start_seconds: float
    source_end_seconds: float
    useful_seconds: float
    overlap_from_previous_seconds: float

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WindowSpec:
        return cls(
            window_index=int(payload["window_index"]),
            source_start_seconds=float(payload["source_start_seconds"]),
            source_end_seconds=float(payload["source_end_seconds"]),
            useful_seconds=float(payload["useful_seconds"]),
            overlap_from_previous_seconds=float(
                payload["overlap_from_previous_seconds"]
            ),
        )


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    source_id: str
    scene_group: str
    target_id: str
    replacement_scope: str
    candidate_seed: int
    source_sha256: str
    target_asset_sha256: str
    source_coordinate_frame: str
    target_coordinate_frame: str
    plugins: tuple[str, ...]
    required_gates: tuple[str, ...]
    windows: tuple[WindowSpec, ...]
    useful_video_seconds: float
    generated_window_seconds: float

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JobSpec:
        return cls(
            job_id=str(payload["job_id"]),
            source_id=str(payload["source_id"]),
            scene_group=str(payload["scene_group"]),
            target_id=str(payload["target_id"]),
            replacement_scope=str(payload["replacement_scope"]),
            candidate_seed=int(payload["candidate_seed"]),
            source_sha256=str(payload["source_sha256"]),
            target_asset_sha256=str(payload["target_asset_sha256"]),
            source_coordinate_frame=str(payload["source_coordinate_frame"]),
            target_coordinate_frame=str(payload["target_coordinate_frame"]),
            plugins=tuple(str(item) for item in payload["plugins"]),
            required_gates=tuple(str(item) for item in payload["required_gates"]),
            windows=tuple(WindowSpec.from_dict(item) for item in payload["windows"]),
            useful_video_seconds=float(payload["useful_video_seconds"]),
            generated_window_seconds=float(payload["generated_window_seconds"]),
        )


@dataclass(frozen=True)
class CampaignPlan:
    schema_version: str
    campaign_id: str
    campaign_seed: int
    claim_scope: str
    jobs: tuple[JobSpec, ...]
    plugin_lock: tuple[dict[str, object], ...]
    useful_video_hours: float
    generated_window_hours: float
    target_output_hours: float
    plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CampaignPlan:
        plan = cls(
            schema_version=str(payload["schema_version"]),
            campaign_id=str(payload["campaign_id"]),
            campaign_seed=int(payload["campaign_seed"]),
            claim_scope=str(payload["claim_scope"]),
            jobs=tuple(JobSpec.from_dict(item) for item in payload["jobs"]),
            plugin_lock=tuple(dict(item) for item in payload["plugin_lock"]),
            useful_video_hours=float(payload["useful_video_hours"]),
            generated_window_hours=float(payload["generated_window_hours"]),
            target_output_hours=float(payload["target_output_hours"]),
            plan_sha256=str(payload["plan_sha256"]),
        )
        unlocked = plan.to_dict()
        expected = str(unlocked.pop("plan_sha256"))
        actual = hashlib.sha256(
            json.dumps(unlocked, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if actual != expected:
            raise ValueError(
                f"campaign plan hash mismatch: expected {expected}, computed {actual}"
            )
        return plan


def split_windows(
    duration_seconds: float,
    *,
    window_seconds: float,
    overlap_seconds: float,
) -> tuple[WindowSpec, ...]:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive")
    if not math.isfinite(overlap_seconds) or not 0 <= overlap_seconds < window_seconds:
        raise ValueError("overlap_seconds must be in [0, window_seconds)")
    windows: list[WindowSpec] = []
    start = 0.0
    index = 0
    while start < duration_seconds - 1e-9:
        end = min(duration_seconds, start + window_seconds)
        overlap = 0.0 if index == 0 else min(overlap_seconds, end - start)
        windows.append(
            WindowSpec(
                window_index=index,
                source_start_seconds=round(start, 9),
                source_end_seconds=round(end, 9),
                useful_seconds=round(end - start - overlap, 9),
                overlap_from_previous_seconds=round(overlap, 9),
            )
        )
        if end >= duration_seconds - 1e-9:
            break
        start += window_seconds - overlap_seconds
        index += 1
    return tuple(windows)


def _job_id(campaign_id: str, source_id: str, target_id: str, seed: int) -> str:
    identity = f"{campaign_id}\0{source_id}\0{target_id}\0{seed}".encode()
    return f"job-{hashlib.sha256(identity).hexdigest()[:16]}"


def compile_campaign(
    campaign: CampaignSpec,
    registry: PluginRegistry | None = None,
) -> CampaignPlan:
    registry = registry or PluginRegistry()
    plugins = registry.validate_campaign(campaign)
    jobs: list[JobSpec] = []
    for source in sorted(campaign.sources, key=lambda item: item.source_id):
        windows = split_windows(
            source.duration_seconds,
            window_seconds=campaign.pipeline.window_seconds,
            overlap_seconds=campaign.pipeline.overlap_seconds,
        )
        generated_seconds = sum(
            item.source_end_seconds - item.source_start_seconds for item in windows
        )
        for target in sorted(campaign.targets, key=lambda item: item.target_id):
            for candidate_seed in sorted(campaign.candidate_seeds):
                jobs.append(
                    JobSpec(
                        job_id=_job_id(
                            campaign.campaign_id,
                            source.source_id,
                            target.target_id,
                            candidate_seed,
                        ),
                        source_id=source.source_id,
                        scene_group=source.scene_group,
                        target_id=target.target_id,
                        replacement_scope=target.replacement_scope.value,
                        candidate_seed=candidate_seed,
                        source_sha256=source.sha256,
                        target_asset_sha256=target.asset_sha256,
                        source_coordinate_frame=source.coordinate_frame,
                        target_coordinate_frame=target.coordinate_frame,
                        plugins=(
                            campaign.pipeline.source_plugin,
                            target.retarget_plugin,
                            campaign.pipeline.generator_plugin,
                            *campaign.pipeline.auditor_plugins,
                        ),
                        required_gates=campaign.pipeline.required_gates,
                        windows=windows,
                        useful_video_seconds=source.duration_seconds,
                        generated_window_seconds=generated_seconds,
                    )
                )
    useful_seconds = sum(item.useful_video_seconds for item in jobs)
    generated_seconds = sum(item.generated_window_seconds for item in jobs)
    unlocked = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign.campaign_id,
        "campaign_seed": campaign.seed,
        "claim_scope": campaign.pipeline.claim_scope.value,
        "jobs": [asdict(item) for item in jobs],
        "plugin_lock": [item.to_dict() for item in plugins],
        "useful_video_hours": useful_seconds / 3600.0,
        "generated_window_hours": generated_seconds / 3600.0,
        "target_output_hours": campaign.target_output_hours,
    }
    digest = hashlib.sha256(
        json.dumps(unlocked, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CampaignPlan(
        schema_version=SCHEMA_VERSION,
        campaign_id=campaign.campaign_id,
        campaign_seed=campaign.seed,
        claim_scope=campaign.pipeline.claim_scope.value,
        jobs=tuple(jobs),
        plugin_lock=tuple(item.to_dict() for item in plugins),
        useful_video_hours=useful_seconds / 3600.0,
        generated_window_hours=generated_seconds / 3600.0,
        target_output_hours=campaign.target_output_hours,
        plan_sha256=digest,
    )
