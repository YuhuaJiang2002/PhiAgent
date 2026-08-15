"""Measured capacity and cost projection for accepted video hours."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


def _positive(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


@dataclass(frozen=True)
class BenchmarkProfile:
    profile_id: str
    accelerator: str
    accelerators_per_worker: int
    wall_seconds_per_output_second: float
    accelerator_seconds_per_output_second: float
    benchmark_uri: str
    benchmark_sha256: str
    evidence_status: str
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or not self.accelerator.strip():
            raise ValueError("profile_id and accelerator must be non-empty")
        if self.accelerators_per_worker <= 0:
            raise ValueError("accelerators_per_worker must be positive")
        _positive(
            self.wall_seconds_per_output_second,
            "wall_seconds_per_output_second",
        )
        _positive(
            self.accelerator_seconds_per_output_second,
            "accelerator_seconds_per_output_second",
        )
        if self.evidence_status not in {"WORKING", "PARTIAL"}:
            raise ValueError("evidence_status must be WORKING or PARTIAL")
        if len(self.benchmark_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.benchmark_sha256
        ):
            raise ValueError("benchmark_sha256 must be lowercase hexadecimal")
        if not self.benchmark_uri.strip() or not self.claim_boundary.strip():
            raise ValueError("benchmark_uri and claim_boundary must be non-empty")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BenchmarkProfile:
        return cls(
            profile_id=str(payload["profile_id"]),
            accelerator=str(payload["accelerator"]),
            accelerators_per_worker=int(payload["accelerators_per_worker"]),
            wall_seconds_per_output_second=float(
                payload["wall_seconds_per_output_second"]
            ),
            accelerator_seconds_per_output_second=float(
                payload["accelerator_seconds_per_output_second"]
            ),
            benchmark_uri=str(payload["benchmark_uri"]),
            benchmark_sha256=str(payload["benchmark_sha256"]),
            evidence_status=str(payload["evidence_status"]),
            claim_boundary=str(payload["claim_boundary"]),
        )


@dataclass(frozen=True)
class CapacityAssumptions:
    target_accepted_hours: float
    accelerator_count: int
    first_pass_yield: float = 0.8
    utilization: float = 0.85
    non_generation_overhead_fraction: float = 0.15
    average_clip_seconds: float = 10.0
    reviewer_count: int = 4
    reviewer_hours_per_accepted_video_hour: float = 1.25
    output_mbps: float = 8.0
    artifact_storage_multiplier: float = 4.0

    def __post_init__(self) -> None:
        _positive(self.target_accepted_hours, "target_accepted_hours")
        if self.accelerator_count <= 0:
            raise ValueError("accelerator_count must be positive")
        for label in ("first_pass_yield", "utilization"):
            value = float(getattr(self, label))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{label} must be in (0, 1]")
        if (
            not math.isfinite(self.non_generation_overhead_fraction)
            or self.non_generation_overhead_fraction < 0
        ):
            raise ValueError("non_generation_overhead_fraction must be non-negative")
        _positive(self.average_clip_seconds, "average_clip_seconds")
        if self.reviewer_count <= 0:
            raise ValueError("reviewer_count must be positive")
        _positive(
            self.reviewer_hours_per_accepted_video_hour,
            "reviewer_hours_per_accepted_video_hour",
        )
        _positive(self.output_mbps, "output_mbps")
        _positive(self.artifact_storage_multiplier, "artifact_storage_multiplier")


@dataclass(frozen=True)
class CapacityEstimate:
    schema_version: str
    profile_id: str
    evidence_status: str
    claim_boundary: str
    target_accepted_hours: float
    raw_candidate_hours: float
    accepted_clip_count: int
    raw_candidate_clip_count: int
    workers: int
    accelerator_count_used: int
    accelerator_hours: float
    generation_calendar_hours: float
    review_person_hours: float
    review_calendar_hours: float
    pipeline_calendar_hours: float
    pipeline_calendar_days: float
    accepted_video_hours_per_day: float
    delivery_storage_gb: float
    working_storage_gb: float
    assumptions: dict[str, object]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_capacity(
    profile: BenchmarkProfile,
    assumptions: CapacityAssumptions,
) -> CapacityEstimate:
    workers = assumptions.accelerator_count // profile.accelerators_per_worker
    if workers <= 0:
        raise ValueError(
            f"profile {profile.profile_id!r} requires at least "
            f"{profile.accelerators_per_worker} accelerators"
        )
    used = workers * profile.accelerators_per_worker
    raw_hours = assumptions.target_accepted_hours / assumptions.first_pass_yield
    overhead = 1.0 + assumptions.non_generation_overhead_fraction
    accelerator_hours = (
        assumptions.target_accepted_hours
        * profile.accelerator_seconds_per_output_second
        * overhead
        / assumptions.first_pass_yield
    )
    generation_calendar_hours = (
        assumptions.target_accepted_hours
        * profile.wall_seconds_per_output_second
        * overhead
        / assumptions.first_pass_yield
        / workers
        / assumptions.utilization
    )
    review_person_hours = (
        assumptions.target_accepted_hours
        * assumptions.reviewer_hours_per_accepted_video_hour
    )
    review_calendar_hours = review_person_hours / assumptions.reviewer_count
    pipeline_calendar_hours = max(generation_calendar_hours, review_calendar_hours)
    accepted_clips = math.ceil(
        assumptions.target_accepted_hours * 3600 / assumptions.average_clip_seconds
    )
    raw_clips = math.ceil(accepted_clips / assumptions.first_pass_yield)
    delivery_gb = (
        assumptions.target_accepted_hours
        * 3600
        * assumptions.output_mbps
        / 8
        / 1000
    )
    return CapacityEstimate(
        schema_version="1.0.0",
        profile_id=profile.profile_id,
        evidence_status=profile.evidence_status,
        claim_boundary=profile.claim_boundary,
        target_accepted_hours=assumptions.target_accepted_hours,
        raw_candidate_hours=raw_hours,
        accepted_clip_count=accepted_clips,
        raw_candidate_clip_count=raw_clips,
        workers=workers,
        accelerator_count_used=used,
        accelerator_hours=accelerator_hours,
        generation_calendar_hours=generation_calendar_hours,
        review_person_hours=review_person_hours,
        review_calendar_hours=review_calendar_hours,
        pipeline_calendar_hours=pipeline_calendar_hours,
        pipeline_calendar_days=pipeline_calendar_hours / 24,
        accepted_video_hours_per_day=(
            assumptions.target_accepted_hours / (pipeline_calendar_hours / 24)
        ),
        delivery_storage_gb=delivery_gb,
        working_storage_gb=delivery_gb * assumptions.artifact_storage_multiplier,
        assumptions=asdict(assumptions),
    )


def load_profiles(path: Path) -> dict[str, BenchmarkProfile]:
    payload = json.loads(path.read_text())
    raw_profiles = payload.get("profiles") if isinstance(payload, Mapping) else None
    if not isinstance(raw_profiles, list):
        raise ValueError("benchmark profile file must contain a profiles array")
    profiles = [BenchmarkProfile.from_dict(item) for item in raw_profiles]
    result = {item.profile_id: item for item in profiles}
    if len(result) != len(profiles):
        raise ValueError("benchmark profile_id values must be unique")
    return result
