"""A capability-gated model router suitable for LLMRouter training data.

LLMRouter selects among endpoints; it does not merge incompatible DiT weights,
latents, or KV caches.  PhiAgent therefore routes a whole request to a pinned
backend profile after hard safety/physics gates.  Learned LLMRouter policies may
replace only the final ranking once offline evaluation has produced labels.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


LLMROUTER_REPOSITORY = "https://github.com/ulab-uiuc/LLMRouter"
LLMROUTER_REVISION = "b5a54da822fea6134ca7af55700685fc8431575f"


class RouterConfigError(ValueError):
    """Raised when profiles or a route request violate the routing contract."""


@dataclass(frozen=True)
class ModelProfile:
    name: str
    capabilities: frozenset[str]
    median_seconds: float
    quality_tier: int
    physical_gate: bool = False
    available: bool = True

    def validate(self) -> None:
        if not self.name or self.median_seconds <= 0 or self.quality_tier < 0:
            raise RouterConfigError("profile name, positive latency, and nonnegative quality tier required")


@dataclass(frozen=True)
class RouteRequest:
    required_capabilities: frozenset[str]
    minimum_quality_tier: int = 0
    requires_physical_gate: bool = False
    max_seconds: float | None = None

    def validate(self) -> None:
        if self.minimum_quality_tier < 0:
            raise RouterConfigError("minimum_quality_tier must be nonnegative")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise RouterConfigError("max_seconds must be positive")


@dataclass(frozen=True)
class RouteDecision:
    selected: ModelProfile | None
    eligible: tuple[str, ...]
    rejected: dict[str, str]


@dataclass(frozen=True)
class RouteOutcome:
    """One measured candidate result used to train/evaluate a learned router.

    The record deliberately separates a generated video from an accepted video:
    latency only affects selection after quality and physical gates have passed.
    ``request_features`` is the input representation an embedding-based
    LLMRouter policy may consume later (instruction, task class, duration,
    controls, and scene tags), without requiring PhiAgent to import Torch.
    """

    request_id: str
    profile_name: str
    request_features: Mapping[str, object]
    latency_seconds: float
    generated: bool
    automated_quality_passed: bool
    action_consistency_passed: bool
    physical_gate_passed: bool
    human_review_passed: bool

    def validate(self) -> None:
        if not self.request_id or not self.profile_name:
            raise RouterConfigError("request_id and profile_name are required")
        if self.latency_seconds <= 0:
            raise RouterConfigError("latency_seconds must be positive")

    @property
    def accepted(self) -> bool:
        return (
            self.generated
            and self.automated_quality_passed
            and self.action_consistency_passed
            and self.physical_gate_passed
            and self.human_review_passed
        )


def _revision(source: Path) -> str:
    marker = source / ".phiagent-source-revision"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    if (source / ".git").is_dir():
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True, check=False
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    return ""


def validate_llmrouter_source(source: Path) -> str:
    """Validate the optional external framework without importing it at runtime."""

    source = source.expanduser().resolve()
    if not (source / "llmrouter" / "models" / "meta_router.py").is_file():
        raise RouterConfigError("LLMRouter checkout lacks llmrouter/models/meta_router.py")
    revision = _revision(source)
    if revision != LLMROUTER_REVISION:
        raise RouterConfigError(
            f"LLMRouter revision is {revision or 'unreadable'}, expected {LLMROUTER_REVISION}"
        )
    return revision


def route_request(request: RouteRequest, profiles: Iterable[ModelProfile]) -> RouteDecision:
    """Apply hard capability/safety gates, then minimize measured median latency.

    This deterministic baseline is the correct control for a future learned
    LLMRouter policy.  It never treats an unverified generator as a physics
    validator simply because it is larger or slower.
    """

    request.validate()
    rejected: dict[str, str] = {}
    eligible: list[ModelProfile] = []
    for profile in profiles:
        profile.validate()
        if not profile.available:
            rejected[profile.name] = "unavailable"
        elif not request.required_capabilities.issubset(profile.capabilities):
            rejected[profile.name] = "missing_required_capability"
        elif profile.quality_tier < request.minimum_quality_tier:
            rejected[profile.name] = "quality_tier_below_request"
        elif request.requires_physical_gate and not profile.physical_gate:
            rejected[profile.name] = "no_physical_gate"
        elif request.max_seconds is not None and profile.median_seconds > request.max_seconds:
            rejected[profile.name] = "latency_budget_exceeded"
        else:
            eligible.append(profile)
    selected = min(eligible, key=lambda item: (item.median_seconds, -item.quality_tier, item.name), default=None)
    return RouteDecision(selected, tuple(item.name for item in eligible), rejected)


def build_llmrouter_training_rows(outcomes: Sequence[RouteOutcome]) -> tuple[dict[str, object], ...]:
    """Export accepted model-choice labels for an LLMRouter custom dataset.

    For every request this emits the lowest-latency profile that passed all
    gates.  Requests without an accepted candidate are preserved with a null
    label so they can be analysed, but must not teach a learned router to route
    unsafe tasks to a model merely because it returned a video.
    """

    grouped: dict[str, list[RouteOutcome]] = {}
    for outcome in outcomes:
        outcome.validate()
        grouped.setdefault(outcome.request_id, []).append(outcome)
    rows: list[dict[str, object]] = []
    for request_id in sorted(grouped):
        candidates = grouped[request_id]
        feature_sets = {json.dumps(dict(item.request_features), sort_keys=True) for item in candidates}
        if len(feature_sets) != 1:
            raise RouterConfigError(f"request {request_id!r} has inconsistent request_features")
        accepted = [item for item in candidates if item.accepted]
        oracle = min(accepted, key=lambda item: (item.latency_seconds, item.profile_name), default=None)
        rows.append(
            {
                "request_id": request_id,
                "request_features": dict(candidates[0].request_features),
                "candidate_profiles": [item.profile_name for item in sorted(candidates, key=lambda item: item.profile_name)],
                "accepted_profiles": [item.profile_name for item in sorted(accepted, key=lambda item: item.profile_name)],
                "oracle_profile": None if oracle is None else oracle.profile_name,
                "oracle_latency_seconds": None if oracle is None else oracle.latency_seconds,
            }
        )
    return tuple(rows)


def build_llmrouter_standard_data(
    outcomes: Sequence[RouteOutcome],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...], tuple[str, ...]]:
    """Export LLMRouter-standard query and routing-label records safely.

    Requests with no fully accepted candidate remain in the query split but are
    deliberately omitted from labels: assigning them a cheap fallback model
    would convert a failed physical evaluation into router supervision.
    """

    queries: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    unlabeled: list[str] = []
    for row in build_llmrouter_training_rows(outcomes):
        features = row["request_features"]
        if not isinstance(features, Mapping):
            raise RouterConfigError(f"request {row['request_id']!r} has invalid request_features")
        instruction = features.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise RouterConfigError(
                f"request {row['request_id']!r} needs a nonempty request_features.instruction"
            )
        request_id = str(row["request_id"])
        queries.append(
            {
                "query_id": request_id,
                "query": instruction,
                "task": str(features.get("task", "phiagent_video")),
            }
        )
        oracle = row["oracle_profile"]
        if oracle is None:
            unlabeled.append(request_id)
        else:
            labels.append({"query_id": request_id, "best_model": str(oracle)})
    return tuple(queries), tuple(labels), tuple(unlabeled)
