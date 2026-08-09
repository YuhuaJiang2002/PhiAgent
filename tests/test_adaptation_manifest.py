from __future__ import annotations

import json
from pathlib import Path

import pytest

from phiagent.data.adaptation import (
    AdaptationArm,
    AdaptationAsset,
    AdaptationAssetKind,
    AdaptationManifest,
    AdaptationSplit,
    AnimateTrainingExample,
    VaceTrainingExample,
    file_sha256,
    load_adaptation_spec,
)


def _asset(
    path: Path,
    asset_id: str,
    split: AdaptationSplit,
    kind: AdaptationAssetKind,
) -> AdaptationAsset:
    return AdaptationAsset(
        asset_id=asset_id,
        path=str(path),
        split=split,
        kind=kind,
        source_uri=f"local://{asset_id}",
        rights_basis="project-owned test fixture",
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        training_authorized=True,
    )


def test_animate_manifest_requires_grouped_conditioning_assets(tmp_path: Path) -> None:
    target = tmp_path / "target.mp4"
    pose = tmp_path / "pose.mp4"
    face = tmp_path / "face.mp4"
    target.write_bytes(b"target")
    pose.write_bytes(b"pose")
    face.write_bytes(b"face")

    manifest = AdaptationManifest(
        experiment_id="animate-v1",
        arm=AdaptationArm.ANIMATE_LORA,
        assets=(
            _asset(
                target,
                "target",
                AdaptationSplit.TRAIN,
                AdaptationAssetKind.TARGET_VIDEO,
            ),
            _asset(
                pose,
                "pose",
                AdaptationSplit.TRAIN,
                AdaptationAssetKind.POSE_CONTROL_VIDEO,
            ),
            _asset(
                face,
                "face",
                AdaptationSplit.TRAIN,
                AdaptationAssetKind.FACE_CONTROL_VIDEO,
            ),
        ),
        animate_examples=(
            AnimateTrainingExample(
                "example-1",
                target_video_asset_id="target",
                pose_video_asset_id="pose",
                face_video_asset_id="face",
                prompt="A Sharpa hand manipulates an object.",
            ),
        ),
    )

    assert manifest.to_dict()["animate_examples"][0]["target_video_asset_id"] == "target"


def test_reference_video_cannot_leak_into_training(tmp_path: Path) -> None:
    reference = tmp_path / "official-reference.mp4"
    reference.write_bytes(b"reference")

    with pytest.raises(ValueError, match="evaluation-only"):
        _asset(
            reference,
            "reference",
            AdaptationSplit.TRAIN,
            AdaptationAssetKind.REFERENCE_VIDEO,
        )


def test_vace_manifest_requires_control_and_reference_assets(tmp_path: Path) -> None:
    target = tmp_path / "target.mp4"
    control = tmp_path / "control.mp4"
    reference = tmp_path / "reference.png"
    for path in (target, control, reference):
        path.write_bytes(path.name.encode())

    manifest = AdaptationManifest(
        experiment_id="vace-v1",
        arm=AdaptationArm.VACE_LORA,
        assets=(
            _asset(target, "target", AdaptationSplit.TRAIN, AdaptationAssetKind.TARGET_VIDEO),
            _asset(
                control,
                "control",
                AdaptationSplit.TRAIN,
                AdaptationAssetKind.VACE_CONTROL_VIDEO,
            ),
            _asset(
                reference,
                "reference",
                AdaptationSplit.TRAIN,
                AdaptationAssetKind.VACE_REFERENCE_IMAGE,
            ),
        ),
        vace_examples=(
            VaceTrainingExample(
                "example-1",
                target_video_asset_id="target",
                control_video_asset_id="control",
                reference_image_asset_id="reference",
                prompt="A Sharpa dexterous hand manipulates an object.",
            ),
        ),
    )

    assert manifest.to_dict()["vace_examples"][0]["control_video_asset_id"] == "control"


def test_appearance_arm_rejects_manipulation_video_training(tmp_path: Path) -> None:
    video = tmp_path / "motion.mp4"
    video.write_bytes(b"motion")

    with pytest.raises(ValueError, match="only identity_image"):
        AdaptationManifest(
            experiment_id="appearance-v1",
            arm=AdaptationArm.APPEARANCE_LORA,
            assets=(
                _asset(
                    video,
                    "motion",
                    AdaptationSplit.TRAIN,
                    AdaptationAssetKind.MANIPULATION_VIDEO,
                ),
            ),
        )


def test_animate_arm_rejects_unassigned_training_asset(tmp_path: Path) -> None:
    assets = []
    for asset_id, kind in (
        ("target", AdaptationAssetKind.TARGET_VIDEO),
        ("pose", AdaptationAssetKind.POSE_CONTROL_VIDEO),
        ("face", AdaptationAssetKind.FACE_CONTROL_VIDEO),
        ("unused", AdaptationAssetKind.TARGET_VIDEO),
    ):
        path = tmp_path / f"{asset_id}.mp4"
        path.write_bytes(asset_id.encode())
        assets.append(_asset(path, asset_id, AdaptationSplit.TRAIN, kind))

    with pytest.raises(ValueError, match="not assigned"):
        AdaptationManifest(
            experiment_id="animate-v1",
            arm=AdaptationArm.ANIMATE_LORA,
            assets=tuple(assets),
            animate_examples=(
                AnimateTrainingExample("example-1", "target", "pose", "face", "Sharpa hand"),
            ),
        )


def test_load_spec_hashes_relative_assets_and_refuses_overwrite(tmp_path: Path) -> None:
    image = tmp_path / "sharpa.png"
    image.write_bytes(b"identity")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "experiment_id": "appearance-v1",
                "arm": "appearance_lora",
                "assets": [
                    {
                        "asset_id": "identity",
                        "path": "sharpa.png",
                        "split": "train",
                        "kind": "identity_image",
                        "source_uri": "local://sharpa",
                        "rights_basis": "project-owned image",
                        "training_authorized": True,
                    }
                ],
            }
        )
    )

    manifest = load_adaptation_spec(spec)
    output = tmp_path / "run" / "manifest.json"
    manifest.write_json(output)

    assert json.loads(output.read_text())["assets"][0]["sha256"] == file_sha256(image)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        manifest.write_json(output)
