"""Strict intake helpers for the pinned DiffSynth Wan-Animate LoRA trainer."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from phiagent.data.adaptation import (
    AdaptationArm,
    AdaptationManifest,
    AdaptationSplit,
)
from phiagent.rendering.wan_animate import GPUInfo, WAN22_MODEL_REVISION

DIFFSYNTH_COMMIT = "b1c02ce76aabc989f6bf534756b2da84532249e5"
DIFFSYNTH_REPOSITORY = "https://github.com/modelscope/DiffSynth-Studio.git"
ANIMATE_TRAIN_SCRIPT = Path("examples/wanvideo/model_training/train.py")
ACCELERATE_CONFIG = Path(
    "examples/wanvideo/model_training/full/accelerate_config_14B.yaml"
)
REQUIRED_GPU_COUNT = 8
DEFAULT_MINIMUM_FREE_GPU_MIB = 75 * 1024


def select_training_gpus(
    gpus: Sequence[GPUInfo],
    requested_indices: Sequence[int],
    minimum_free_mib: int = DEFAULT_MINIMUM_FREE_GPU_MIB,
    required_count: int = REQUIRED_GPU_COUNT,
) -> tuple[GPUInfo, ...]:
    if minimum_free_mib <= 0 or required_count <= 0:
        raise ValueError("minimum_free_mib and required_count must be positive")
    by_index = {gpu.physical_index: gpu for gpu in gpus}
    if requested_indices:
        if len(requested_indices) != required_count:
            raise ValueError(f"exactly {required_count} physical GPU indices are required")
        if len(set(requested_indices)) != len(requested_indices):
            raise ValueError("requested physical GPU indices must be unique")
        missing = [index for index in requested_indices if index not in by_index]
        if missing:
            raise ValueError(f"requested physical GPUs were not reported: {missing}")
        selected = tuple(by_index[index] for index in requested_indices)
    else:
        eligible = sorted(
            (gpu for gpu in gpus if gpu.free_mib >= minimum_free_mib),
            key=lambda gpu: gpu.free_mib,
            reverse=True,
        )
        if len(eligible) < required_count:
            raise ValueError(
                f"DiffSynth Animate training requires {required_count} GPUs with at least "
                f"{minimum_free_mib} MiB free; found {len(eligible)}"
            )
        selected = tuple(eligible[:required_count])
    busy = [gpu for gpu in selected if gpu.free_mib < minimum_free_mib]
    if busy:
        summary = ", ".join(
            f"GPU {gpu.physical_index}: {gpu.free_mib} MiB free" for gpu in busy
        )
        raise ValueError(
            f"selected GPUs do not meet the {minimum_free_mib} MiB requirement ({summary})"
        )
    return selected


def verify_diffsynth_checkout(repo: Path) -> str:
    resolved = repo.expanduser().resolve()
    required = (
        resolved / "LICENSE",
        resolved / ANIMATE_TRAIN_SCRIPT,
        resolved / ACCELERATE_CONFIG,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"DiffSynth checkout is missing required files: {missing}")
    license_text = required[0].read_text(errors="replace")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("DiffSynth checkout does not contain the reviewed Apache-2.0 license")
    accelerate_text = required[2].read_text(errors="replace")
    if "num_processes: 8" not in accelerate_text or "distributed_type: DEEPSPEED" not in accelerate_text:
        raise ValueError("DiffSynth Accelerate config is not the reviewed 8-GPU DeepSpeed preset")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=resolved,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or commit != DIFFSYNTH_COMMIT:
        raise ValueError(
            f"DiffSynth checkout is {commit or 'unreadable'}, expected {DIFFSYNTH_COMMIT}"
        )
    return commit


def checkpoint_model_paths(checkpoint_dir: Path) -> str:
    resolved = checkpoint_dir.expanduser().resolve()
    marker = resolved / ".phiagent-model-revision"
    if not marker.is_file() or marker.read_text().strip() != WAN22_MODEL_REVISION:
        raise ValueError(
            f"Wan-Animate checkpoint must be pinned to revision {WAN22_MODEL_REVISION}"
        )
    patterns = (
        "diffusion_pytorch_model*.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    )
    for pattern in patterns:
        if not any(resolved.glob(pattern)):
            raise ValueError(f"Wan-Animate checkpoint is missing {pattern}")
    return ",".join(str(resolved / pattern) for pattern in patterns)


def write_diffsynth_metadata(manifest: AdaptationManifest, path: Path) -> None:
    if manifest.arm is not AdaptationArm.ANIMATE_LORA:
        raise ValueError("DiffSynth Animate training requires an animate_lora manifest")
    assets = {asset.asset_id: asset for asset in manifest.assets}
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("video", "prompt", "animate_pose_video", "animate_face_video"),
        )
        writer.writeheader()
        for example in manifest.animate_examples:
            selected = (
                assets[example.target_video_asset_id],
                assets[example.pose_video_asset_id],
                assets[example.face_video_asset_id],
            )
            if any(asset.split is not AdaptationSplit.TRAIN for asset in selected):
                raise ValueError("DiffSynth metadata can contain only training assets")
            writer.writerow(
                {
                    "video": selected[0].path,
                    "prompt": example.prompt,
                    "animate_pose_video": selected[1].path,
                    "animate_face_video": selected[2].path,
                }
            )


def build_diffsynth_training_command(
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
) -> list[str]:
    if rank <= 0 or learning_rate <= 0 or epochs <= 0 or dataset_repeat <= 0:
        raise ValueError("rank, learning_rate, epochs, and dataset_repeat must be positive")
    resolved_repo = repo.expanduser().resolve()
    return [
        str(accelerate.expanduser().resolve()),
        "launch",
        "--config_file",
        str(resolved_repo / ACCELERATE_CONFIG),
        str(resolved_repo / ANIMATE_TRAIN_SCRIPT),
        "--dataset_base_path",
        str(metadata_path.parent),
        "--dataset_metadata_path",
        str(metadata_path),
        "--data_file_keys",
        "video,animate_pose_video,animate_face_video",
        "--height",
        "480",
        "--width",
        "832",
        "--num_frames",
        "81",
        "--dataset_repeat",
        str(dataset_repeat),
        "--model_paths",
        checkpoint_model_paths(checkpoint_dir),
        "--learning_rate",
        str(learning_rate),
        "--num_epochs",
        str(epochs),
        "--remove_prefix_in_ckpt",
        "pipe.dit.",
        "--output_path",
        str(output_path),
        "--lora_base_model",
        "dit",
        "--lora_target_modules",
        "q,k,v,o,ffn.0,ffn.2",
        "--lora_rank",
        str(rank),
        "--extra_inputs",
        "input_image,animate_pose_video,animate_face_video",
        "--use_gradient_checkpointing_offload",
    ]


def gpu_record(selected: Sequence[GPUInfo]) -> list[dict[str, int | str]]:
    return [asdict(gpu) for gpu in selected]


def load_frozen_manifest(path: Path) -> AdaptationManifest:
    payload = json.loads(path.expanduser().resolve().read_text())
    if not isinstance(payload, dict):
        raise ValueError("frozen adaptation manifest must contain one JSON object")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise ValueError("frozen adaptation manifest assets must be a list")
    for asset in assets:
        asset_path = Path(str(asset["path"]))
        if not asset_path.is_file():
            raise ValueError(f"frozen adaptation asset is missing: {asset_path}")
        from phiagent.data.adaptation import file_sha256

        if file_sha256(asset_path) != asset["sha256"]:
            raise ValueError(f"frozen adaptation asset hash changed: {asset_path}")
    return AdaptationManifest.from_spec(payload, path.parent)
