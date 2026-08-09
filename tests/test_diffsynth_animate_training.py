from __future__ import annotations

import csv
from pathlib import Path

import pytest

from phiagent.data.adaptation import (
    AdaptationArm,
    AdaptationAsset,
    AdaptationAssetKind,
    AdaptationManifest,
    AdaptationSplit,
    AnimateTrainingExample,
    file_sha256,
)
from phiagent.rendering.wan_animate import GPUInfo, WAN22_MODEL_REVISION
from phiagent.training.diffsynth_animate import (
    build_diffsynth_training_command,
    select_training_gpus,
    write_diffsynth_metadata,
)


def _asset(path: Path, asset_id: str, kind: AdaptationAssetKind) -> AdaptationAsset:
    path.write_bytes(asset_id.encode())
    return AdaptationAsset(
        asset_id,
        str(path),
        AdaptationSplit.TRAIN,
        kind,
        f"local://{asset_id}",
        "project-owned fixture",
        file_sha256(path),
        path.stat().st_size,
        True,
    )


def _manifest(tmp_path: Path) -> AdaptationManifest:
    return AdaptationManifest(
        "animate-v1",
        AdaptationArm.ANIMATE_LORA,
        (
            _asset(tmp_path / "target.mp4", "target", AdaptationAssetKind.TARGET_VIDEO),
            _asset(tmp_path / "pose.mp4", "pose", AdaptationAssetKind.POSE_CONTROL_VIDEO),
            _asset(tmp_path / "face.mp4", "face", AdaptationAssetKind.FACE_CONTROL_VIDEO),
        ),
        (AnimateTrainingExample("example", "target", "pose", "face", "Sharpa motion"),),
    )


def test_select_training_gpus_requires_eight_free_physical_devices() -> None:
    gpus = tuple(GPUInfo(index, "A800", 81920, index, 81920 - index) for index in range(8))

    selected = select_training_gpus(gpus, (), minimum_free_mib=80000)

    assert {gpu.physical_index for gpu in selected} == set(range(8))
    with pytest.raises(ValueError, match="exactly 8"):
        select_training_gpus(gpus, (0, 1), minimum_free_mib=80000)


def test_write_metadata_uses_exact_animate_columns(tmp_path: Path) -> None:
    output = tmp_path / "dataset" / "metadata.csv"

    write_diffsynth_metadata(_manifest(tmp_path), output)

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "video": str(tmp_path / "target.mp4"),
            "prompt": "Sharpa motion",
            "animate_pose_video": str(tmp_path / "pose.mp4"),
            "animate_face_video": str(tmp_path / "face.mp4"),
        }
    ]


def test_training_command_uses_pinned_local_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / ".phiagent-model-revision").write_text(WAN22_MODEL_REVISION + "\n")
    for name in (
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ):
        (checkpoint / name).write_bytes(b"model")

    command = build_diffsynth_training_command(
        tmp_path / "bin" / "accelerate",
        tmp_path / "DiffSynth-Studio",
        tmp_path / "dataset" / "metadata.csv",
        checkpoint,
        tmp_path / "output",
        rank=32,
        learning_rate=1e-4,
        epochs=5,
        dataset_repeat=100,
    )

    assert "--model_paths" in command
    assert "--model_id_with_origin_paths" not in command
    assert command[command.index("--lora_rank") + 1] == "32"
