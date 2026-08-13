#!/usr/bin/env python3
"""Generate a matched H3 baseline/native-identity-LoRA ablation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.minimax_h3 import (  # noqa: E402
    DIFFSYNTH_H3_COMMIT,
    verify_diffsynth_h3_source,
)
from phiagent.rendering.wan_animate import (  # noqa: E402
    acquire_gpu_lease,
    query_gpus,
    select_gpu,
)


NF4_FILES = (
    "minimax-h3-text-encoder-nf4.safetensors",
    "minimax-h3-ref2va-nf4.safetensors",
    "video_vae_nf4.safetensors",
    "audio_vae_nf4.safetensors",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _packages() -> dict[str, str | None]:
    result = {}
    for name in (
        "torch",
        "transformers",
        "modelscope",
        "bitsandbytes",
        "safetensors",
        "peft",
        "av",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-reference", type=Path, required=True)
    parser.add_argument("--scene-reference", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--diffsynth-repo", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=54 * 1024)
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/h3-identity-ablation"))
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=39)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--vram-reserve-gib", type=float, default=8.0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    paths = {
        "robot_reference": args.robot_reference.expanduser().resolve(),
        "scene_reference": args.scene_reference.expanduser().resolve(),
        "prompt_file": args.prompt_file.expanduser().resolve(),
        "lora_checkpoint": args.lora_checkpoint.expanduser().resolve(),
    }
    diffsynth = args.diffsynth_repo.expanduser().resolve()
    model_base = args.model_base_path.expanduser().resolve()
    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32:
        raise ValueError("width/height must be positive multiples of 32")
    if args.num_frames < 5 or (args.num_frames - 5) % 17:
        raise ValueError("num_frames must satisfy 17n+5")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = (
        args.experiment_dir.expanduser().resolve()
        if args.experiment_dir
        else args.experiment_root.expanduser().resolve() / f"{stamp}-{uuid4().hex[:8]}"
    )
    experiment.mkdir(parents=True, exist_ok=False)
    manifest_path = experiment / "manifest.json"
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "matched_minimax_h3_nf4_baseline_vs_native_identity_lora",
        "status": "preflight_started",
        "honest_status": "NOT STARTED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "seed": args.seed,
        "config": {
            "width": args.width,
            "height": args.height,
            "num_frames": args.num_frames,
            "steps": args.steps,
            "fps": 24,
        },
    }
    _write_json(manifest_path, manifest)
    lease = None
    try:
        for name, path in paths.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"{name} is missing or empty: {path}")
        revision = verify_diffsynth_h3_source(diffsynth)
        if revision != DIFFSYNTH_H3_COMMIT:
            raise ValueError(f"unexpected DiffSynth revision {revision}")
        checkpoint_root = model_base / "DiffSynth-Studio/MiniMax-H3-NF4"
        model_paths = [checkpoint_root / name for name in NF4_FILES]
        processor = model_base / "MiniMax/MiniMax-H3/Ref2VA/processor"
        for path in (*model_paths, processor):
            if not path.exists():
                raise ValueError(f"model input is missing: {path}")
        gpus, inventory_raw, processes_raw = query_gpus()
        selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(diffsynth), str(project_root), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(model_base)
        sys.path.insert(0, str(diffsynth))
        manifest.update(
            {
                "status": "preflight_passed",
                "source_revision": revision,
                "selected_gpu": asdict(selected),
                "gpu_inventory": [asdict(gpu) for gpu in gpus],
                "gpu_inventory_raw": inventory_raw,
                "gpu_processes_raw": processes_raw,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "packages": _packages(),
                "inputs": {
                    name: {"path": str(path), "sha256": _sha256(path)}
                    for name, path in paths.items()
                },
                "models": [
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                    for path in model_paths
                ],
            }
        )
        _write_json(manifest_path, manifest)
        if args.preflight_only:
            print(json.dumps({"experiment": str(experiment), "status": "preflight_passed"}))
            return 0
        _, lease = acquire_gpu_lease(selected.physical_index)
        manifest["status"] = "running"
        manifest["honest_status"] = "PARTIAL"
        _write_json(manifest_path, manifest)
        import torch
        from PIL import Image
        from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
        from diffsynth.utils.data.audio_video import write_video_audio

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("physical GPU selection did not map to one logical CUDA device")
        free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
        vram_limit = free_bytes / 1024**3 - args.vram_reserve_gib
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        pipe = MiniMaxH3Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[ModelConfig(path=str(path), **vram_config) for path in model_paths],
            processor_config=ModelConfig(
                model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/processor/"
            ),
            vram_limit=vram_limit,
        )
        robot = Image.open(paths["robot_reference"]).convert("RGB")
        scene = Image.open(paths["scene_reference"]).convert("RGB")
        prompt = paths["prompt_file"].read_text().strip()
        references = [
            {"type": "image", "image": robot},
            {"type": "image", "image": scene},
        ]

        def generate(output: Path) -> None:
            video, audio = pipe(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                seed=args.seed,
                references=references,
                ref_image_short_edge=768,
            )
            write_video_audio(
                video=video,
                audio=audio,
                output_path=str(output),
                fps=24,
                audio_sample_rate=32000,
            )

        baseline = experiment / "baseline.mp4"
        candidate = experiment / "identity-lora.mp4"
        generate(baseline)
        pipe.load_lora(pipe.dit, str(paths["lora_checkpoint"]))
        generate(candidate)
        comparison = experiment / "baseline-vs-identity-lora.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(baseline),
                "-i",
                str(candidate),
                "-filter_complex",
                "[0:v][1:v]hstack=inputs=2[v]",
                "-map",
                "[v]",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "16",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(comparison),
            ],
            check=True,
        )
        for output in (baseline, candidate, comparison):
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
                check=True,
            )
        manifest.update(
            {
                "status": "completed",
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "runtime": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                    "free_gib_at_load": free_bytes / 1024**3,
                    "total_gib": total_bytes / 1024**3,
                    "vram_limit_gib": vram_limit,
                },
                "outputs": {
                    output.name: {
                        "path": str(output),
                        "bytes": output.stat().st_size,
                        "sha256": _sha256(output),
                    }
                    for output in (baseline, candidate, comparison)
                },
                "acceptance": {
                    "matched_outputs_decode": True,
                    "held_out_metrics_evaluated": False,
                    "promotion_contract_passed": False,
                },
                "limitations": [
                    "This matched ablation is not an improvement claim until frozen identity and non-regression metrics run.",
                    "The r0 LoRA was trained on six low-resolution deterministic robot clips.",
                ],
            }
        )
        _write_json(manifest_path, manifest)
        print(json.dumps({"experiment": str(experiment), "comparison": str(comparison)}))
        return 0
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(manifest_path, manifest)
        raise
    finally:
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
