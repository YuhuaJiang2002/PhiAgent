"""Leakage-safe manifests for lightweight Sharpa adaptation experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


METHOD_LABEL = "sharpa_lightweight_adaptation_not_official_phizero"
SCHEMA_VERSION = "0.1.0"


class AdaptationArm(str, Enum):
    ZERO_SHOT = "zero_shot"
    APPEARANCE_LORA = "appearance_lora"
    ANIMATE_LORA = "animate_lora"
    VACE_LORA = "vace_lora"


class AdaptationSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    CONFIRMATION = "confirmation"


class AdaptationAssetKind(str, Enum):
    IDENTITY_IMAGE = "identity_image"
    MANIPULATION_VIDEO = "manipulation_video"
    TARGET_VIDEO = "target_video"
    POSE_CONTROL_VIDEO = "pose_control_video"
    FACE_CONTROL_VIDEO = "face_control_video"
    SOURCE_VIDEO = "source_video"
    REFERENCE_VIDEO = "reference_video"
    VACE_CONTROL_VIDEO = "vace_control_video"
    VACE_REFERENCE_IMAGE = "vace_reference_image"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AdaptationAsset:
    asset_id: str
    path: str
    split: AdaptationSplit
    kind: AdaptationAssetKind
    source_uri: str
    rights_basis: str
    sha256: str
    size_bytes: int
    training_authorized: bool = False

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("asset_id must be non-empty")
        if not self.source_uri.strip() or not self.rights_basis.strip():
            raise ValueError("source_uri and rights_basis must be non-empty")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("sha256 must be a lowercase hexadecimal SHA-256 digest")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.split is AdaptationSplit.TRAIN and not self.training_authorized:
            raise ValueError("training assets require explicit training_authorized=true")
        if self.kind is AdaptationAssetKind.REFERENCE_VIDEO and self.split is AdaptationSplit.TRAIN:
            raise ValueError("reference videos are evaluation-only and cannot be training assets")

    @classmethod
    def from_spec(cls, payload: Mapping[str, Any], base_dir: Path) -> AdaptationAsset:
        configured_path = Path(str(payload["path"])).expanduser()
        path = configured_path if configured_path.is_absolute() else base_dir / configured_path
        path = path.resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"adaptation asset does not exist or is empty: {path}")
        return cls(
            asset_id=str(payload["asset_id"]),
            path=str(path),
            split=AdaptationSplit(str(payload["split"])),
            kind=AdaptationAssetKind(str(payload["kind"])),
            source_uri=str(payload["source_uri"]),
            rights_basis=str(payload["rights_basis"]),
            sha256=file_sha256(path),
            size_bytes=path.stat().st_size,
            training_authorized=bool(payload.get("training_authorized", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["split"] = self.split.value
        payload["kind"] = self.kind.value
        return payload


@dataclass(frozen=True)
class AnimateTrainingExample:
    example_id: str
    target_video_asset_id: str
    pose_video_asset_id: str
    face_video_asset_id: str
    prompt: str

    def __post_init__(self) -> None:
        values = (
            self.example_id,
            self.target_video_asset_id,
            self.pose_video_asset_id,
            self.face_video_asset_id,
            self.prompt,
        )
        if any(not value.strip() for value in values):
            raise ValueError("Animate training example fields must be non-empty")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AnimateTrainingExample:
        return cls(
            example_id=str(payload["example_id"]),
            target_video_asset_id=str(payload["target_video_asset_id"]),
            pose_video_asset_id=str(payload["pose_video_asset_id"]),
            face_video_asset_id=str(payload["face_video_asset_id"]),
            prompt=str(payload["prompt"]),
        )


@dataclass(frozen=True)
class VaceTrainingExample:
    example_id: str
    target_video_asset_id: str
    control_video_asset_id: str
    reference_image_asset_id: str
    prompt: str

    def __post_init__(self) -> None:
        values = (
            self.example_id,
            self.target_video_asset_id,
            self.control_video_asset_id,
            self.reference_image_asset_id,
            self.prompt,
        )
        if any(not value.strip() for value in values):
            raise ValueError("VACE training example fields must be non-empty")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VaceTrainingExample:
        return cls(
            example_id=str(payload["example_id"]),
            target_video_asset_id=str(payload["target_video_asset_id"]),
            control_video_asset_id=str(payload["control_video_asset_id"]),
            reference_image_asset_id=str(payload["reference_image_asset_id"]),
            prompt=str(payload["prompt"]),
        )


@dataclass(frozen=True)
class AdaptationManifest:
    experiment_id: str
    arm: AdaptationArm
    assets: tuple[AdaptationAsset, ...]
    animate_examples: tuple[AnimateTrainingExample, ...] = ()
    vace_examples: tuple[VaceTrainingExample, ...] = ()
    evidence_scope: str = "development_only"
    schema_version: str = SCHEMA_VERSION
    method: str = METHOD_LABEL

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported adaptation schema {self.schema_version!r}")
        if self.method != METHOD_LABEL:
            raise ValueError(f"method must be {METHOD_LABEL!r}")
        if self.evidence_scope not in {"development_only", "claim_eligible"}:
            raise ValueError("evidence_scope must be development_only or claim_eligible")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        asset_ids = [asset.asset_id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("adaptation asset_id values must be unique")
        assets_by_hash: dict[str, list[AdaptationAsset]] = {}
        for asset in self.assets:
            assets_by_hash.setdefault(asset.sha256, []).append(asset)
        invalid_duplicates = {
            digest: items
            for digest, items in assets_by_hash.items()
            if len(items) > 1
            and any(
                item.kind is not AdaptationAssetKind.VACE_CONTROL_VIDEO for item in items
            )
        }
        if invalid_duplicates:
            raise ValueError(
                "duplicate asset content is allowed only for matched VACE control ablations"
            )

        training_kinds = {
            asset.kind for asset in self.assets if asset.split is AdaptationSplit.TRAIN
        }
        if self.arm is AdaptationArm.ZERO_SHOT:
            if training_kinds or self.animate_examples or self.vace_examples:
                raise ValueError("zero_shot arm cannot contain training data")
        if self.arm is AdaptationArm.APPEARANCE_LORA:
            if training_kinds != {AdaptationAssetKind.IDENTITY_IMAGE}:
                raise ValueError("appearance_lora requires only identity_image training assets")
            if self.animate_examples or self.vace_examples:
                raise ValueError("appearance_lora cannot contain video training examples")
        if self.arm is AdaptationArm.ANIMATE_LORA:
            required = {
                AdaptationAssetKind.TARGET_VIDEO,
                AdaptationAssetKind.POSE_CONTROL_VIDEO,
                AdaptationAssetKind.FACE_CONTROL_VIDEO,
            }
            if training_kinds != required:
                raise ValueError(
                    "animate_lora requires only target_video, pose_control_video, and "
                    "face_control_video training assets"
                )
            self._validate_animate_examples()
            if self.vace_examples:
                raise ValueError("animate_lora cannot contain VACE training examples")
        if self.arm is AdaptationArm.VACE_LORA:
            required = {
                AdaptationAssetKind.TARGET_VIDEO,
                AdaptationAssetKind.VACE_CONTROL_VIDEO,
                AdaptationAssetKind.VACE_REFERENCE_IMAGE,
            }
            if training_kinds != required:
                raise ValueError(
                    "vace_lora requires only target_video, vace_control_video, and "
                    "vace_reference_image training assets"
                )
            self._validate_vace_examples()
            if self.animate_examples:
                raise ValueError("vace_lora cannot contain Animate training examples")

    def _validate_animate_examples(self) -> None:
        if not self.animate_examples:
            raise ValueError("animate_lora requires at least one Animate training example")
        example_ids = [example.example_id for example in self.animate_examples]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("Animate training example_id values must be unique")
        assets = {asset.asset_id: asset for asset in self.assets}
        expected_kinds = (
            ("target_video_asset_id", AdaptationAssetKind.TARGET_VIDEO),
            ("pose_video_asset_id", AdaptationAssetKind.POSE_CONTROL_VIDEO),
            ("face_video_asset_id", AdaptationAssetKind.FACE_CONTROL_VIDEO),
        )
        referenced: set[str] = set()
        for example in self.animate_examples:
            for field, expected_kind in expected_kinds:
                asset_id = str(getattr(example, field))
                try:
                    asset = assets[asset_id]
                except KeyError as exc:
                    raise ValueError(
                        f"Animate example {example.example_id!r} references unknown "
                        f"asset {asset_id!r}"
                    ) from exc
                if asset.split is not AdaptationSplit.TRAIN or asset.kind is not expected_kind:
                    raise ValueError(
                        f"Animate example asset {asset_id!r} must be a train "
                        f"{expected_kind.value}"
                    )
                referenced.add(asset_id)
        unreferenced = {
            asset.asset_id
            for asset in self.assets
            if asset.split is AdaptationSplit.TRAIN and asset.asset_id not in referenced
        }
        if unreferenced:
            raise ValueError(
                f"Animate training assets are not assigned to examples: {sorted(unreferenced)}"
            )

    def _validate_vace_examples(self) -> None:
        if not self.vace_examples:
            raise ValueError("vace_lora requires at least one VACE training example")
        example_ids = [example.example_id for example in self.vace_examples]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("VACE training example_id values must be unique")
        assets = {asset.asset_id: asset for asset in self.assets}
        expected_kinds = (
            ("target_video_asset_id", AdaptationAssetKind.TARGET_VIDEO),
            ("control_video_asset_id", AdaptationAssetKind.VACE_CONTROL_VIDEO),
            ("reference_image_asset_id", AdaptationAssetKind.VACE_REFERENCE_IMAGE),
        )
        referenced: set[str] = set()
        for example in self.vace_examples:
            for field, expected_kind in expected_kinds:
                asset_id = str(getattr(example, field))
                try:
                    asset = assets[asset_id]
                except KeyError as exc:
                    raise ValueError(
                        f"VACE example {example.example_id!r} references unknown "
                        f"asset {asset_id!r}"
                    ) from exc
                if asset.split is not AdaptationSplit.TRAIN or asset.kind is not expected_kind:
                    raise ValueError(
                        f"VACE example asset {asset_id!r} must be a train "
                        f"{expected_kind.value}"
                    )
                referenced.add(asset_id)
        unreferenced = {
            asset.asset_id
            for asset in self.assets
            if asset.split is AdaptationSplit.TRAIN and asset.asset_id not in referenced
        }
        if unreferenced:
            raise ValueError(
                f"VACE training assets are not assigned to examples: {sorted(unreferenced)}"
            )

    @classmethod
    def from_spec(cls, payload: Mapping[str, Any], base_dir: Path) -> AdaptationManifest:
        raw_assets = payload.get("assets")
        if not isinstance(raw_assets, Sequence) or isinstance(raw_assets, (str, bytes)):
            raise ValueError("assets must be a sequence")
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            method=str(payload.get("method", METHOD_LABEL)),
            experiment_id=str(payload["experiment_id"]),
            arm=AdaptationArm(str(payload["arm"])),
            assets=tuple(AdaptationAsset.from_spec(item, base_dir) for item in raw_assets),
            animate_examples=tuple(
                AnimateTrainingExample.from_dict(item)
                for item in payload.get("animate_examples", ())
            ),
            vace_examples=tuple(
                VaceTrainingExample.from_dict(item)
                for item in payload.get("vace_examples", ())
            ),
            evidence_scope=str(payload.get("evidence_scope", "development_only")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "experiment_id": self.experiment_id,
            "arm": self.arm.value,
            "assets": [asset.to_dict() for asset in self.assets],
            "animate_examples": [asdict(example) for example in self.animate_examples],
            "vace_examples": [asdict(example) for example in self.vace_examples],
            "evidence_scope": self.evidence_scope,
        }

    def write_json(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite adaptation manifest: {path}")
        path.parent.mkdir(parents=True, exist_ok=False)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def load_adaptation_spec(path: Path) -> AdaptationManifest:
    resolved = path.expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid adaptation spec JSON: {resolved}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("adaptation spec must contain one JSON object")
    return AdaptationManifest.from_spec(payload, resolved.parent)
