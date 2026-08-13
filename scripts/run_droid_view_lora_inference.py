#!/usr/bin/env python3
"""Generate one held-out exterior-view video with the PhiAgent DROID View LoRA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402
from phiagent.training.diffsynth_vace import verify_vace_checkpoint  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True, choices=(21, 60, 77))
    parser.add_argument("--view", choices=("target_a", "target_b"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/droid-view-lora-inference"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=30 * 1024)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def select_heldout_example(
    contract: dict[str, Any],
    episode: int,
    view: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = contract.get("holdout_records", [])
    matches = [row for row in records if int(row.get("episode_index", -1)) == episode]
    if len(matches) != 1:
        raise ValueError(f"expected one held-out record for episode {episode}")
    record = matches[0]
    if record.get("training_use") is not False:
        raise ValueError("selected evaluation episode is not held out from training")
    target = record.get("targets", {}).get(view)
    if not isinstance(target, dict):
        raise ValueError(f"held-out episode {episode} has no {view}")
    return record, target


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("torch", "diffsynth", "peft", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main() -> int:
    args = _parser().parse_args()
    if min(
        args.minimum_free_gpu_mib,
        args.steps,
        args.height,
        args.width,
        args.num_frames,
        args.fps,
    ) <= 0:
        raise ValueError("inference settings must be positive")
    if (args.num_frames - 1) % 4:
        raise ValueError("num-frames must satisfy 4n+1")

    contract_path = args.dataset_contract.expanduser().resolve()
    training_metadata_path = args.training_metadata.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    lora = args.lora.expanduser().resolve()
    for label, path in (
        ("dataset contract", contract_path),
        ("training metadata", training_metadata_path),
        ("trained LoRA", lora),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    model_files = verify_vace_checkpoint(checkpoint_dir)
    contract = json.loads(contract_path.read_text())
    training_metadata = json.loads(training_metadata_path.read_text())
    if training_metadata.get("status") != "completed":
        raise ValueError("training metadata must record a completed run")
    trained_hashes = {
        item["sha256"] for item in training_metadata.get("trained_checkpoints", [])
    }
    if _sha256(lora) not in trained_hashes:
        raise ValueError("LoRA hash is not present in the completed training metadata")
    if training_metadata.get("dataset_contract_sha256") != _sha256(contract_path):
        raise ValueError("training metadata and inference dataset contract differ")
    record, target = select_heldout_example(contract, args.episode, args.view)
    dataset_root = contract_path.parent
    control = (dataset_root / record["condition"]["path"]).resolve()
    reference = (dataset_root / target["anchor"]).resolve()
    withheld_target = (dataset_root / target["target"]).resolve()
    for label, path in (
        ("real wrist condition", control),
        ("real target-view anchor condition", reference),
        ("withheld real target", withheld_target),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    if _sha256(control) != record["condition"]["sha256"]:
        raise ValueError("real wrist condition hash differs from the frozen contract")
    if _sha256(reference) != target["anchor_sha256"]:
        raise ValueError("real target-view anchor hash differs from the frozen contract")
    if _sha256(withheld_target) != target["target_sha256"]:
        raise ValueError("withheld real target hash differs from the frozen contract")

    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"

    import torch
    from PIL import Image
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from diffsynth.utils.data import VideoData, save_video

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = (
        args.output_root.expanduser().resolve()
        / f"ep{args.episode:03d}-{args.view}-{stamp}-{uuid4().hex[:8]}"
    )
    experiment.mkdir(parents=True)
    video_path = experiment / "our-generated-video.mp4"
    metadata_path = experiment / "metadata.json"
    output_record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "method": "phiagent_droid_view_lora_heldout_generation",
        "model_label": "PhiAgent DROID View LoRA on pinned Wan2.1-VACE-1.3B base",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "episode_index": args.episode,
        "view": args.view,
        "prompt": target["prompt"],
        "coordinate_frames": {
            "condition": record["condition"]["coordinate_frame"],
            "generated": target["coordinate_frame"],
            "real_target": target["coordinate_frame"],
        },
        "conditions": {
            "real_wrist_video": {"path": str(control), "sha256": _sha256(control)},
            "real_target_view_anchor": {
                "path": str(reference),
                "sha256": _sha256(reference),
            },
            "task_text": target["prompt"],
        },
        "target_leakage_guard": {
            "real_target_path_recorded_for_post_generation_evaluation_only": str(
                withheld_target
            ),
            "real_target_sha256": _sha256(withheld_target),
            "real_target_passed_to_model": False,
            "input_video_passed_to_model": False,
            "edit_mask_passed_to_model": False,
        },
        "base_checkpoint_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in model_files
        ],
        "lora": {"path": str(lora), "sha256": _sha256(lora)},
        "training_metadata": {
            "path": str(training_metadata_path),
            "sha256": _sha256(training_metadata_path),
        },
        "dataset_contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
        "selected_gpu": asdict(selected),
        "gpu_inventory": [asdict(gpu) for gpu in gpus],
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "seed": args.seed,
        "config": {
            "steps": args.steps,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _package_versions(),
    }
    _write_json(metadata_path, output_record)
    try:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[ModelConfig(path=str(path)) for path in model_files[:3]],
            tokenizer_config=ModelConfig(path=str(checkpoint_dir / "google" / "umt5-xxl")),
        )
        pipe.load_lora(pipe.vace, str(lora), alpha=1)
        control_data = VideoData(str(control), height=args.height, width=args.width)
        control_frames = [control_data[index] for index in range(args.num_frames)]
        generated_frames = pipe(
            prompt=target["prompt"],
            negative_prompt=(
                "low quality, blurry, malformed robot, deformed gripper, duplicate object, "
                "wrong camera view, first-person view, flicker"
            ),
            vace_video=control_frames,
            vace_video_mask=None,
            vace_reference_image=Image.open(reference).convert("RGB").resize(
                (args.width, args.height)
            ),
            input_video=None,
            denoising_strength=1.0,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            seed=args.seed,
            tiled=True,
        )
        save_video(generated_frames, str(video_path), fps=args.fps, quality=5)
        output_record.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output": str(video_path),
                "output_sha256": _sha256(video_path),
            }
        )
    except Exception as exc:
        output_record.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
            }
        )
        raise
    finally:
        _write_json(metadata_path, output_record)
    print(
        json.dumps(
            {
                "experiment": str(experiment),
                "status": output_record["status"],
                "video": str(video_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
