#!/usr/bin/env python3
"""Run pinned VACE-1.3B inference with an optional trained regional LoRA."""

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
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402
from phiagent.training.diffsynth_vace import verify_vace_checkpoint  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--control-video", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sharpa-vace-inference"))
    parser.add_argument("--prompt", default="A Sharpa dexterous robot hand manipulates an object.")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=30 * 1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parser().parse_args()
    if min(
        args.minimum_free_gpu_mib,
        args.height,
        args.width,
        args.num_frames,
        args.fps,
        args.steps,
    ) <= 0:
        raise ValueError("inference dimensions, frame count, FPS, steps, and memory must be positive")
    if (args.num_frames - 1) % 4:
        raise ValueError("num_frames must satisfy 4n+1")
    checkpoint = args.checkpoint_dir.expanduser().resolve()
    model_files = verify_vace_checkpoint(checkpoint)
    control = args.control_video.expanduser().resolve()
    reference = args.reference_image.expanduser().resolve()
    lora = args.lora.expanduser().resolve() if args.lora else None
    for label, path in (("control video", control), ("reference image", reference)):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if lora is not None and not lora.is_file():
        raise ValueError(f"LoRA does not exist: {lora}")

    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    import torch
    from PIL import Image
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from diffsynth.utils.data import VideoData, save_video

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = output_root / f"{stamp}-{uuid4().hex[:8]}"
    experiment.mkdir()
    video_path = experiment / "output.mp4"
    metadata_path = experiment / "metadata.json"
    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "wan21_vace_regional_lora_not_official_phizero",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "config": {
            **vars(args),
            "checkpoint_dir": str(checkpoint),
            "control_video": str(control),
            "reference_image": str(reference),
            "lora": str(lora) if lora else None,
            "output_root": str(output_root),
        },
        "selected_gpu": asdict(selected),
        "gpu_inventory": [asdict(gpu) for gpu in gpus],
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "diffsynth", "peft", "transformers")
        },
        "inputs": {
            "control": str(control),
            "control_sha256": _sha256(control),
            "reference": str(reference),
            "reference_sha256": _sha256(reference),
            "lora": str(lora) if lora else None,
            "lora_sha256": _sha256(lora) if lora else None,
        },
    }
    metadata_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    try:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[ModelConfig(path=str(path)) for path in model_files[:3]],
            tokenizer_config=ModelConfig(path=str(checkpoint / "google" / "umt5-xxl")),
        )
        if lora is not None:
            pipe.load_lora(pipe.vace, str(lora), alpha=1)
        control_frames = VideoData(str(control), height=args.height, width=args.width)
        control_frames = [control_frames[index] for index in range(args.num_frames)]
        output_frames = pipe(
            prompt=args.prompt,
            negative_prompt=(
                "low quality, blurry, malformed robot hand, extra fingers, "
                "deformed object, flicker, duplicate object"
            ),
            vace_video=control_frames,
            vace_reference_image=Image.open(reference).convert("RGB").resize(
                (args.width, args.height)
            ),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            seed=args.seed,
            tiled=True,
        )
        save_video(output_frames, str(video_path), fps=args.fps, quality=5)
        record.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output": str(video_path),
                "output_sha256": _sha256(video_path),
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
            }
        )
        raise
    finally:
        metadata_path.write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"
        )
    print(f"EXPERIMENT={experiment}")
    print(f"VIDEO={video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
