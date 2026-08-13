#!/usr/bin/env python3
"""Run a reproducible MiniMax-H3 NF4 Ref2VA flower-replacement validation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import signal
import shlex
import socket
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.minimax_h3 import (  # noqa: E402
    DIFFSYNTH_H3_COMMIT,
    MINIMAX_H3_MODELSCOPE_ID,
    MINIMAX_H3_NF4_MODEL_ID,
    MiniMaxH3ValidationConfig,
    file_sha256,
    verify_diffsynth_h3_source,
)
from phiagent.rendering.wan_animate import query_gpus  # noqa: E402


class _Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _raise_on_termination(signum: int, _frame: object) -> None:
    """Turn transport/process termination into a persisted experiment failure."""

    signal_name = signal.Signals(signum).name
    raise RuntimeError(f"received {signal_name}; experiment did not complete")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _packages() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in (
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "modelscope",
        "bitsandbytes",
        "safetensors",
        "accelerate",
        "peft",
        "huggingface-hub",
        "av",
        "diffsynth",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _git_state(root: Path) -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "available": status.returncode == 0,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
        "error": status.stderr.strip() if status.returncode != 0 else None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--robot-reference", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--diffsynth-repo", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/minimax-h3-flower-validation"))
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=54 * 1024)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--reference-short-edge", type=int, default=768)
    parser.add_argument("--reference-video-short-edge", type=int, default=480)
    parser.add_argument("--vram-reserve-gib", type=float, default=8.0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.experiment_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment = args.experiment_root.expanduser().resolve() / f"{stamp}-{uuid4().hex[:8]}"
    else:
        experiment = args.experiment_dir.expanduser().resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    metadata_path = experiment / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"experiment metadata already exists: {metadata_path}")

    config = MiniMaxH3ValidationConfig(
        source_video=args.source_video.expanduser().resolve(),
        robot_reference=args.robot_reference.expanduser().resolve(),
        prompt_file=args.prompt_file.expanduser().resolve(),
        diffsynth_repo=args.diffsynth_repo.expanduser().resolve(),
        model_base_path=args.model_base_path.expanduser().resolve(),
        width=args.width,
        height=args.height,
        fps=args.fps,
        num_frames=args.num_frames,
        steps=args.steps,
        seed=args.seed,
        minimum_free_gpu_mib=args.minimum_free_gpu_mib,
        requested_gpu=args.gpu,
    )
    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "minimax_h3_nf4_ref2va_proxy_not_official_bf16_not_phizero",
        "status": "preflight_started",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": _git_state(project_root),
        "config": {
            **asdict(config),
            "source_video": str(config.source_video),
            "robot_reference": str(config.robot_reference),
            "prompt_file": str(config.prompt_file),
            "diffsynth_repo": str(config.diffsynth_repo),
            "model_base_path": str(config.model_base_path),
            "reference_short_edge": args.reference_short_edge,
            "reference_video_short_edge": args.reference_video_short_edge,
            "vram_reserve_gib": args.vram_reserve_gib,
        },
        "model": {
            "base": "MiniMax-H3",
            "weights": MINIMAX_H3_NF4_MODEL_ID,
            "processor": MINIMAX_H3_MODELSCOPE_ID,
            "quantization": "third-party prequantized bitsandbytes NF4",
            "revision_policy": "downloaded files are hashed after inference",
        },
        "source_revision": DIFFSYNTH_H3_COMMIT,
        "inputs": {},
        "limitations": [
            "This is a third-party NF4 quantization smoke/quality validation, not the official BF16 checkpoint result.",
            "This is a visual editing proxy, not official PhiZero inference or real-robot execution.",
            "The source clip is silent; H3 audio is not an acceptance target.",
        ],
    }
    _write_json(metadata_path, record)
    for signal_name in ("SIGHUP", "SIGTERM"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), _raise_on_termination)
    log_path = experiment / "inference.log"
    try:
        config.validate()
        source_commit = verify_diffsynth_h3_source(config.diffsynth_repo)
        gpus, inventory_raw, processes_raw = query_gpus()
        selected = config.select_gpu(gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
        os.environ["PYTHONHASHSEED"] = str(config.seed)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(config.diffsynth_repo), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        # This process was already started before PYTHONPATH was finalized, so
        # also make the reviewed checkout win over any editable DiffSynth install
        # inherited from the copied GPU environment.
        sys.path.insert(0, str(config.diffsynth_repo))
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(config.model_base_path)
        config.model_base_path.mkdir(parents=True, exist_ok=True)
        record.update(
            {
                "status": "preflight_passed" if args.preflight_only else "running",
                "source_revision": source_commit,
                "selected_gpu": asdict(selected),
                "gpu_inventory": [asdict(gpu) for gpu in gpus],
                "gpu_inventory_raw": inventory_raw,
                "gpu_processes_raw": processes_raw,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "packages": _packages(),
                "inputs": {
                    "source_video": str(config.source_video),
                    "source_sha256": file_sha256(config.source_video),
                    "robot_reference": str(config.robot_reference),
                    "robot_reference_sha256": file_sha256(config.robot_reference),
                    "prompt_file": str(config.prompt_file),
                    "prompt_sha256": file_sha256(config.prompt_file),
                },
            }
        )
        _write_json(metadata_path, record)
        if args.preflight_only:
            print(json.dumps({"experiment": str(experiment), "status": record["status"]}))
            return 0

        with log_path.open("w", encoding="utf-8") as log:
            tee_out = _Tee(sys.stdout, log)
            tee_err = _Tee(sys.stderr, log)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                import torch
                from PIL import Image
                from diffsynth.pipelines.minimax_h3_audio_video import (
                    MiniMaxH3Pipeline,
                    ModelConfig,
                )
                from diffsynth.utils.data.audio_video import read_video_audio, write_video_audio

                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise RuntimeError("selected physical GPU did not map to exactly one CUDA device")
                free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
                free_gib = free_bytes / 1024**3
                vram_limit_gib = free_gib - args.vram_reserve_gib
                if vram_limit_gib <= 8:
                    raise RuntimeError(
                        f"only {free_gib:.2f} GiB is free after CUDA selection; reserve leaves too little VRAM"
                    )
                record["runtime"] = {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "logical_devices": torch.cuda.device_count(),
                    "logical_gpu_name": torch.cuda.get_device_name(0),
                    "free_gib_at_load": free_gib,
                    "total_gib": total_bytes / 1024**3,
                    "vram_limit_gib": vram_limit_gib,
                }
                _write_json(metadata_path, record)
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
                    model_configs=[
                        ModelConfig(
                            model_id=MINIMAX_H3_NF4_MODEL_ID,
                            origin_file_pattern="minimax-h3-ref2va-nf4.safetensors",
                            **vram_config,
                        ),
                        ModelConfig(
                            model_id=MINIMAX_H3_NF4_MODEL_ID,
                            origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors",
                            **vram_config,
                        ),
                        ModelConfig(
                            model_id=MINIMAX_H3_NF4_MODEL_ID,
                            origin_file_pattern="video_vae_nf4.safetensors",
                            **vram_config,
                        ),
                        ModelConfig(
                            model_id=MINIMAX_H3_NF4_MODEL_ID,
                            origin_file_pattern="audio_vae_nf4.safetensors",
                            **vram_config,
                        ),
                    ],
                    processor_config=ModelConfig(
                        model_id=MINIMAX_H3_MODELSCOPE_ID,
                        origin_file_pattern="Ref2VA/processor/",
                    ),
                    vram_limit=vram_limit_gib,
                )
                frames, _, _ = read_video_audio(
                    str(config.source_video),
                    height=config.height,
                    width=config.width,
                    num_frames=config.num_frames,
                    fps=config.fps,
                    audio_sample_rate=pipe.audio_vae.sample_rate,
                )
                if len(frames) != config.num_frames:
                    raise RuntimeError(
                        f"reference decoder returned {len(frames)} frames, expected {config.num_frames}"
                    )
                robot = Image.open(config.robot_reference).convert("RGB")
                prompt = config.prompt_file.read_text().strip()
                video, audio = pipe(
                    prompt=prompt,
                    height=config.height,
                    width=config.width,
                    num_frames=config.num_frames,
                    num_inference_steps=config.steps,
                    seed=config.seed,
                    references=[
                        {"type": "image", "image": robot},
                        {"type": "video", "video": frames},
                    ],
                    ref_image_short_edge=args.reference_short_edge,
                    ref_video_short_edge=args.reference_video_short_edge,
                    ref_video_max_pixels=config.height * config.width,
                )
                result = experiment / "raw-h3-nf4.mp4"
                write_video_audio(
                    video=video,
                    audio=audio,
                    output_path=str(result),
                    fps=config.fps,
                    audio_sample_rate=32000,
                )

        checkpoint_root = config.model_base_path / MINIMAX_H3_NF4_MODEL_ID
        checkpoint_files = sorted(path for path in checkpoint_root.rglob("*") if path.is_file())
        processor_root = config.model_base_path / MINIMAX_H3_MODELSCOPE_ID / "Ref2VA" / "processor"
        processor_files = sorted(path for path in processor_root.rglob("*") if path.is_file())
        result = experiment / "raw-h3-nf4.mp4"
        if not result.is_file() or result.stat().st_size == 0:
            raise RuntimeError("H3 did not produce a non-empty result")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(result), "-f", "null", "-"],
            check=True,
        )
        record.update(
            {
                "status": "completed",
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "result": str(result),
                "result_sha256": file_sha256(result),
                "checkpoint_files": [
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                    for path in checkpoint_files
                ],
                "processor_files": [
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                    for path in processor_files
                ],
                "acceptance": {
                    "inference_completed": True,
                    "output_decoded": True,
                    "visual_quality_evaluated": False,
                    "background_lock_evaluated": False,
                },
            }
        )
        _write_json(metadata_path, record)
        print(json.dumps({"experiment": str(experiment), "result": str(result)}))
        return 0
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "honest_status": "BLOCKED",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(metadata_path, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
