#!/usr/bin/env python3
"""Run pinned VACE-1.3B inference with an optional trained regional LoRA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
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
    parser.add_argument("--input-video", type=Path)
    parser.add_argument("--edit-mask", type=Path)
    parser.add_argument("--denoising-strength", type=float, default=1.0)
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
    parser.add_argument("--source-git-head")
    parser.add_argument("--source-git-status-sha256")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_frame_count(path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate VACE control frames")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return int(completed.stdout.strip())
    except ValueError as exc:
        raise ValueError(f"ffprobe did not report a frame count for {path}") from exc


def _git_state(root: Path) -> dict[str, object]:
    result = {}
    for name, command in (
        ("head", ("git", "rev-parse", "HEAD")),
        ("status", ("git", "status", "--short")),
    ):
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        result[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return result


def main() -> int:
    args = _parser().parse_args()
    if bool(args.source_git_head) != bool(args.source_git_status_sha256):
        raise ValueError(
            "source Git head and status SHA-256 must be supplied together"
        )
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
    if not 0 < args.denoising_strength <= 1:
        raise ValueError("denoising-strength must be in (0, 1]")
    checkpoint = args.checkpoint_dir.expanduser().resolve()
    model_files = verify_vace_checkpoint(checkpoint)
    control = args.control_video.expanduser().resolve()
    reference = args.reference_image.expanduser().resolve()
    lora = args.lora.expanduser().resolve() if args.lora else None
    input_video = args.input_video.expanduser().resolve() if args.input_video else None
    edit_mask = args.edit_mask.expanduser().resolve() if args.edit_mask else None
    for label, path in (("control video", control), ("reference image", reference)):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if lora is not None and not lora.is_file():
        raise ValueError(f"LoRA does not exist: {lora}")
    if (input_video is None) != (edit_mask is None):
        raise ValueError("input-video and edit-mask must be provided together")
    if input_video is not None and (
        not input_video.is_file() or not edit_mask.is_file()
    ):
        raise ValueError("input video and edit mask must exist")
    available_frames = _video_frame_count(control)
    if available_frames < args.num_frames:
        raise ValueError(
            f"control video has {available_frames} frames, "
            f"but --num-frames requires {args.num_frames}"
        )

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
        "git": {
            "execution_workspace": _git_state(
                Path(__file__).resolve().parents[1]
            ),
            "source_git_head": args.source_git_head,
            "source_git_status_sha256": args.source_git_status_sha256,
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "diffsynth", "peft", "transformers")
        },
        "inputs": {
            "checkpoint_files": [
                {"path": str(path), "sha256": _sha256(path)} for path in model_files
            ],
            "entrypoint": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "control": str(control),
            "control_sha256": _sha256(control),
            "control_available_frames": available_frames,
            "reference": str(reference),
            "reference_sha256": _sha256(reference),
            "lora": str(lora) if lora else None,
            "lora_sha256": _sha256(lora) if lora else None,
            "input_video": str(input_video) if input_video else None,
            "input_video_sha256": _sha256(input_video) if input_video else None,
            "edit_mask": str(edit_mask) if edit_mask else None,
            "edit_mask_sha256": _sha256(edit_mask) if edit_mask else None,
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
        input_frames = None
        mask_frames = None
        if input_video is not None:
            input_data = VideoData(str(input_video), height=args.height, width=args.width)
            mask_data = VideoData(str(edit_mask), height=args.height, width=args.width)
            input_frames = [input_data[index] for index in range(args.num_frames)]
            mask_frames = [mask_data[index] for index in range(args.num_frames)]
        output_frames = pipe(
            prompt=args.prompt,
            negative_prompt=(
                "low quality, blurry, malformed robot hand, extra fingers, "
                "deformed object, flicker, duplicate object"
            ),
            vace_video=control_frames,
            vace_video_mask=mask_frames,
            vace_reference_image=Image.open(reference).convert("RGB").resize(
                (args.width, args.height)
            ),
            input_video=input_frames,
            denoising_strength=args.denoising_strength,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            seed=args.seed,
            tiled=True,
        )
        outside_mask_unchanged_fraction = None
        if input_frames is not None:
            import numpy as np

            composited = []
            unchanged = total_outside = 0
            for generated, source, mask in zip(output_frames, input_frames, mask_frames):
                binary_mask = mask.convert("L").point(
                    lambda value: 255 if value >= 128 else 0
                )
                merged = Image.composite(generated, source, binary_mask)
                generated_array = np.asarray(merged)
                source_array = np.asarray(source)
                outside = np.asarray(binary_mask) == 0
                unchanged += int(
                    np.count_nonzero(
                        np.all(generated_array[outside] == source_array[outside], axis=1)
                    )
                )
                total_outside += int(np.count_nonzero(outside))
                composited.append(merged)
            output_frames = composited
            outside_mask_unchanged_fraction = unchanged / total_outside
        save_video(output_frames, str(video_path), fps=args.fps, quality=5)
        record.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output": str(video_path),
                "output_sha256": _sha256(video_path),
                "outside_mask_unchanged_fraction_before_encode": (
                    outside_mask_unchanged_fraction
                ),
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
