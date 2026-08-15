"""Typed contracts for reproducible, cross-embodiment video campaigns.

The data-engine package intentionally depends only on the Python standard
library.  Model repositories, CUDA runtimes, simulators, and checkpoints are
loaded by plugins, never while importing :mod:`phiagent`.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,95}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReplacementScope(str, Enum):
    HAND = "hand"
    FULL_EMBODIMENT = "full_embodiment"


class ClaimScope(str, Enum):
    VISUAL_TRAINING_DATA = "visual_training_data"
    PHYSICALLY_GROUNDED = "physically_grounded"


VISUAL_REQUIRED_GATES = (
    "source_lineage",
    "exact_asset_identity",
    "human_removal",
    "motion_preservation",
    "object_preservation",
    "temporal_continuity",
    "background_preservation",
)

PHYSICAL_REQUIRED_GATES = VISUAL_REQUIRED_GATES + (
    "metric_camera",
    "complete_robot_trajectory",
    "persistent_object_geometry",
    "contact_force",
)


def _identifier(value: object, label: str) -> str:
    result = str(value).strip()
    if not _ID_PATTERN.fullmatch(result):
        raise ValueError(
            f"{label} must be 2-96 lowercase letters, digits, '.', '_' or '-'"
        )
    return result


def _nonempty(value: object, label: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label} must be non-empty")
    return result


def _positive(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _sha256(value: object, label: str) -> str:
    result = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA-256 digest")
    return result


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    result = tuple(_nonempty(item, label) for item in value)
    if not result:
        raise ValueError(f"{label} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} values must be unique")
    return result


@dataclass(frozen=True)
class SourceClip:
    source_id: str
    uri: str
    sha256: str
    duration_seconds: float
    fps: float
    coordinate_frame: str
    scene_group: str
    rights_basis: str
    tasks: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(self, "uri", _nonempty(self.uri, "source uri"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "source sha256"))
        object.__setattr__(
            self,
            "duration_seconds",
            _positive(self.duration_seconds, "source duration_seconds"),
        )
        object.__setattr__(self, "fps", _positive(self.fps, "source fps"))
        coordinate_frame = _nonempty(self.coordinate_frame, "source coordinate_frame")
        if not coordinate_frame.startswith("camera:"):
            raise ValueError("source coordinate_frame must name an explicit camera:* frame")
        object.__setattr__(self, "coordinate_frame", coordinate_frame)
        object.__setattr__(self, "scene_group", _identifier(self.scene_group, "scene_group"))
        object.__setattr__(
            self, "rights_basis", _nonempty(self.rights_basis, "source rights_basis")
        )
        object.__setattr__(self, "tasks", _string_sequence(self.tasks, "source tasks"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceClip:
        return cls(
            source_id=str(payload["source_id"]),
            uri=str(payload["uri"]),
            sha256=str(payload["sha256"]),
            duration_seconds=float(payload["duration_seconds"]),
            fps=float(payload["fps"]),
            coordinate_frame=str(payload["coordinate_frame"]),
            scene_group=str(payload["scene_group"]),
            rights_basis=str(payload["rights_basis"]),
            tasks=tuple(str(item) for item in payload["tasks"]),
        )


@dataclass(frozen=True)
class TargetAsset:
    target_id: str
    replacement_scope: ReplacementScope
    asset_uri: str
    asset_sha256: str
    coordinate_frame: str
    retarget_plugin: str
    reference_uri: str | None = None
    reference_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _identifier(self.target_id, "target_id"))
        object.__setattr__(self, "asset_uri", _nonempty(self.asset_uri, "asset_uri"))
        object.__setattr__(
            self, "asset_sha256", _sha256(self.asset_sha256, "asset_sha256")
        )
        frame = _nonempty(self.coordinate_frame, "target coordinate_frame")
        if not frame.startswith("robot_base:"):
            raise ValueError("target coordinate_frame must name an explicit robot_base:* frame")
        object.__setattr__(self, "coordinate_frame", frame)
        object.__setattr__(
            self, "retarget_plugin", _identifier(self.retarget_plugin, "retarget_plugin")
        )
        if (self.reference_uri is None) != (self.reference_sha256 is None):
            raise ValueError("reference_uri and reference_sha256 must be provided together")
        if self.reference_uri is not None:
            object.__setattr__(
                self, "reference_uri", _nonempty(self.reference_uri, "reference_uri")
            )
            object.__setattr__(
                self,
                "reference_sha256",
                _sha256(self.reference_sha256, "reference_sha256"),
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TargetAsset:
        return cls(
            target_id=str(payload["target_id"]),
            replacement_scope=ReplacementScope(str(payload["replacement_scope"])),
            asset_uri=str(payload["asset_uri"]),
            asset_sha256=str(payload["asset_sha256"]),
            coordinate_frame=str(payload["coordinate_frame"]),
            retarget_plugin=str(payload["retarget_plugin"]),
            reference_uri=(
                str(payload["reference_uri"])
                if payload.get("reference_uri") is not None
                else None
            ),
            reference_sha256=(
                str(payload["reference_sha256"])
                if payload.get("reference_sha256") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PipelineContract:
    source_plugin: str
    generator_plugin: str
    auditor_plugins: tuple[str, ...]
    window_seconds: float
    overlap_seconds: float
    claim_scope: ClaimScope
    required_gates: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_plugin", _identifier(self.source_plugin, "source_plugin")
        )
        object.__setattr__(
            self, "generator_plugin", _identifier(self.generator_plugin, "generator_plugin")
        )
        object.__setattr__(
            self,
            "auditor_plugins",
            tuple(_identifier(item, "auditor_plugin") for item in self.auditor_plugins),
        )
        if not self.auditor_plugins or len(self.auditor_plugins) != len(
            set(self.auditor_plugins)
        ):
            raise ValueError("auditor_plugins must be non-empty and unique")
        window = _positive(self.window_seconds, "window_seconds")
        overlap = float(self.overlap_seconds)
        if not math.isfinite(overlap) or overlap < 0 or overlap >= window:
            raise ValueError("overlap_seconds must be finite, non-negative, and below window")
        object.__setattr__(self, "window_seconds", window)
        object.__setattr__(self, "overlap_seconds", overlap)
        gates = tuple(_identifier(item, "required gate") for item in self.required_gates)
        if not gates or len(gates) != len(set(gates)):
            raise ValueError("required_gates must be non-empty and unique")
        minimum = (
            PHYSICAL_REQUIRED_GATES
            if self.claim_scope is ClaimScope.PHYSICALLY_GROUNDED
            else VISUAL_REQUIRED_GATES
        )
        missing = set(minimum) - set(gates)
        if missing:
            raise ValueError(
                f"{self.claim_scope.value} contract is missing hard gates: {sorted(missing)}"
            )
        object.__setattr__(self, "required_gates", gates)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PipelineContract:
        claim_scope = ClaimScope(str(payload["claim_scope"]))
        default_gates = (
            PHYSICAL_REQUIRED_GATES
            if claim_scope is ClaimScope.PHYSICALLY_GROUNDED
            else VISUAL_REQUIRED_GATES
        )
        return cls(
            source_plugin=str(payload["source_plugin"]),
            generator_plugin=str(payload["generator_plugin"]),
            auditor_plugins=tuple(str(item) for item in payload["auditor_plugins"]),
            window_seconds=float(payload["window_seconds"]),
            overlap_seconds=float(payload.get("overlap_seconds", 0.0)),
            claim_scope=claim_scope,
            required_gates=tuple(
                str(item) for item in payload.get("required_gates", default_gates)
            ),
        )


@dataclass(frozen=True)
class CampaignSpec:
    campaign_id: str
    seed: int
    target_output_hours: float
    sources: tuple[SourceClip, ...]
    targets: tuple[TargetAsset, ...]
    candidate_seeds: tuple[int, ...]
    pipeline: PipelineContract
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported data-engine schema {self.schema_version!r}")
        object.__setattr__(
            self, "campaign_id", _identifier(self.campaign_id, "campaign_id")
        )
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("campaign seed must be a non-negative integer")
        object.__setattr__(
            self,
            "target_output_hours",
            _positive(self.target_output_hours, "target_output_hours"),
        )
        if not self.sources or not self.targets:
            raise ValueError("campaign requires at least one source and one target")
        source_ids = [item.source_id for item in self.sources]
        target_ids = [item.target_id for item in self.targets]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("target_id values must be unique")
        if not self.candidate_seeds or any(
            not isinstance(item, int) or item < 0 for item in self.candidate_seeds
        ):
            raise ValueError("candidate_seeds must contain non-negative integers")
        if len(self.candidate_seeds) != len(set(self.candidate_seeds)):
            raise ValueError("candidate_seeds must be unique")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CampaignSpec:
        raw_sources = payload.get("sources")
        raw_targets = payload.get("targets")
        if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
            raise ValueError("sources must be an array")
        if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)):
            raise ValueError("targets must be an array")
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            campaign_id=str(payload["campaign_id"]),
            seed=int(payload["seed"]),
            target_output_hours=float(payload["target_output_hours"]),
            sources=tuple(SourceClip.from_dict(item) for item in raw_sources),
            targets=tuple(TargetAsset.from_dict(item) for item in raw_targets),
            candidate_seeds=tuple(int(item) for item in payload["candidate_seeds"]),
            pipeline=PipelineContract.from_dict(payload["pipeline"]),
        )

    @classmethod
    def from_json(cls, path: Path) -> CampaignSpec:
        payload = json.loads(path.read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("campaign manifest must contain one JSON object")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pipeline"]["claim_scope"] = self.pipeline.claim_scope.value
        for target in payload["targets"]:
            target["replacement_scope"] = target["replacement_scope"].value
        return payload
