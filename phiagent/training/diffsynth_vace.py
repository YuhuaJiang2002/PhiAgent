"""Pinned DiffSynth Wan VACE LoRA training helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from phiagent.data.adaptation import AdaptationArm, AdaptationManifest, AdaptationSplit
from phiagent.training.diffsynth_animate import verify_diffsynth_checkout

VACE_MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B"
VACE_MODEL_REVISION = "574e6a744642ce3bee319afc31496b88bde8aac4"
VACE_MODELSCOPE_REVISION = "6714b95cf37b2a609fe9087387d6313cd311f500"


def verify_vace_checkpoint(checkpoint_dir: Path) -> tuple[Path, ...]:
    resolved = checkpoint_dir.expanduser().resolve()
    marker = resolved / ".phiagent-model-revision"
    allowed_revisions = {
        VACE_MODEL_REVISION,
        f"modelscope:{VACE_MODELSCOPE_REVISION}",
    }
    actual_revision = marker.read_text().strip() if marker.is_file() else ""
    if actual_revision not in allowed_revisions:
        raise ValueError(
            f"VACE checkpoint marker is {actual_revision!r}; "
            f"expected one of {sorted(allowed_revisions)}"
        )
    required = (
        resolved / "diffusion_pytorch_model.safetensors",
        resolved / "models_t5_umt5-xxl-enc-bf16.pth",
        resolved / "Wan2.1_VAE.pth",
        resolved / "google" / "umt5-xxl" / "tokenizer.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"VACE checkpoint is missing required files: {missing}")
    return required


def write_vace_metadata(manifest: AdaptationManifest, path: Path) -> None:
    if manifest.arm is not AdaptationArm.VACE_LORA:
        raise ValueError("DiffSynth VACE training requires a vace_lora manifest")
    assets = {asset.asset_id: asset for asset in manifest.assets}
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("video", "prompt", "vace_video", "vace_reference_image"),
        )
        writer.writeheader()
        for example in manifest.vace_examples:
            selected = (
                assets[example.target_video_asset_id],
                assets[example.control_video_asset_id],
                assets[example.reference_image_asset_id],
            )
            if any(asset.split is not AdaptationSplit.TRAIN for asset in selected):
                raise ValueError("VACE metadata can contain only training assets")
            writer.writerow(
                {
                    "video": selected[0].path,
                    "prompt": example.prompt,
                    "vace_video": selected[1].path,
                    "vace_reference_image": selected[2].path,
                }
            )


def build_vace_training_command(
    accelerate: Path,
    repo: Path,
    metadata_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
    *,
    rank: int,
    learning_rate: float,
    epochs: int,
    dataset_repeat: int,
    height: int,
    width: int,
    num_frames: int,
) -> list[str]:
    if min(rank, learning_rate, epochs, dataset_repeat, height, width, num_frames) <= 0:
        raise ValueError("VACE training numeric settings must be positive")
    if height % 16 or width % 16:
        raise ValueError("VACE training dimensions must be divisible by 16")
    if (num_frames - 1) % 4:
        raise ValueError("VACE num_frames must satisfy 4n+1")
    model_files = verify_vace_checkpoint(checkpoint_dir)
    verify_diffsynth_checkout(repo)
    return [
        str(accelerate.expanduser().resolve()),
        "launch",
        "--num_processes",
        "1",
        str(repo.expanduser().resolve() / "examples/wanvideo/model_training/train.py"),
        "--dataset_base_path",
        str(metadata_path.parent),
        "--dataset_metadata_path",
        str(metadata_path),
        "--data_file_keys",
        "video,vace_video,vace_reference_image",
        "--height",
        str(height),
        "--width",
        str(width),
        "--num_frames",
        str(num_frames),
        "--dataset_repeat",
        str(dataset_repeat),
        "--model_paths",
        json.dumps([str(path) for path in model_files[:3]]),
        "--tokenizer_path",
        str(checkpoint_dir.expanduser().resolve() / "google" / "umt5-xxl"),
        "--learning_rate",
        str(learning_rate),
        "--num_epochs",
        str(epochs),
        "--remove_prefix_in_ckpt",
        "pipe.vace.",
        "--output_path",
        str(output_path),
        "--lora_base_model",
        "vace",
        "--lora_target_modules",
        "q,k,v,o,ffn.0,ffn.2",
        "--lora_rank",
        str(rank),
        "--extra_inputs",
        "vace_video,vace_reference_image",
        "--use_gradient_checkpointing_offload",
        "--initialize_model_on_cpu",
    ]
