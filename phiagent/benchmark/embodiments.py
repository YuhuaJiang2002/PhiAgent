"""Pinned embodiment-asset registry without vendoring third-party models."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ASSET_KINDS = {"arm", "end_effector"}
VALIDATION_TIERS = {
    "metadata_only",
    "source_pinned",
    "kinematic_validated",
    "simulation_validated",
    "hardware_validated",
}


@dataclass(frozen=True)
class EmbodimentAsset:
    asset_id: str
    kind: str
    manufacturer: str
    model: str
    dof: int
    source_type: str
    source_url: str
    source_revision: str
    source_path: str
    license: str
    validation_tier: str
    physical_parameters: dict[str, float]
    caveats: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmbodimentAsset":
        kind = str(payload["kind"])
        tier = str(payload["validation_tier"])
        source_type = str(payload["source_type"])
        dof = int(payload["dof"])
        if kind not in ASSET_KINDS or tier not in VALIDATION_TIERS:
            raise ValueError("unsupported embodiment asset kind or validation tier")
        if source_type not in {"git", "manufacturer_web"}:
            raise ValueError("unsupported embodiment source type")
        if dof < 0 or (kind == "arm" and dof <= 0):
            raise ValueError("arm DOF must be positive and end-effector DOF non-negative")
        revision = str(payload["source_revision"])
        if source_type == "git" and not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Git embodiment sources require a full 40-character revision")
        parameters = {str(key): float(value) for key, value in payload.get("physical_parameters", {}).items()}
        if any(not math.isfinite(value) or value < 0 for value in parameters.values()):
            raise ValueError("embodiment physical parameters cannot be negative")
        caveats = tuple(str(value).strip() for value in payload.get("caveats", ()))
        if any(not value for value in caveats):
            raise ValueError("embodiment caveats cannot be empty")
        return cls(
            asset_id=str(payload["asset_id"]),
            kind=kind,
            manufacturer=str(payload["manufacturer"]),
            model=str(payload["model"]),
            dof=dof,
            source_type=source_type,
            source_url=str(payload["source_url"]),
            source_revision=revision,
            source_path=str(payload["source_path"]),
            license=str(payload["license"]),
            validation_tier=tier,
            physical_parameters=parameters,
            caveats=caveats,
        )


@dataclass(frozen=True)
class EmbodimentRegistry:
    version: str
    assets: tuple[EmbodimentAsset, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmbodimentRegistry":
        if payload.get("schema_version") != "0.1.0":
            raise ValueError("unsupported embodiment registry schema")
        assets = tuple(EmbodimentAsset.from_dict(item) for item in payload["assets"])
        identifiers = [asset.asset_id for asset in assets]
        if not assets or len(set(identifiers)) != len(identifiers):
            raise ValueError("embodiment registry asset IDs must be unique")
        return cls(version=str(payload["version"]), assets=assets)

    @classmethod
    def from_json(cls, path: Path) -> "EmbodimentRegistry":
        payload = json.loads(path.expanduser().resolve().read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("embodiment registry must be an object")
        return cls.from_dict(payload)

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "asset_count": len(self.assets),
            "by_kind": {
                kind: sum(asset.kind == kind for asset in self.assets)
                for kind in sorted(ASSET_KINDS)
            },
            "by_validation_tier": {
                tier: sum(asset.validation_tier == tier for asset in self.assets)
                for tier in sorted(VALIDATION_TIERS)
            },
            "assets": [
                {
                    "asset_id": asset.asset_id,
                    "kind": asset.kind,
                    "validation_tier": asset.validation_tier,
                    "source_revision": asset.source_revision,
                }
                for asset in self.assets
            ],
        }
