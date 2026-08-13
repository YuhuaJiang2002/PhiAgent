#!/usr/bin/env python3
"""Run a protected low-noise MiniMax-H3 flow inversion/refinement window.

The accepted input is encoded as H3 latents, integrated from data toward a
bounded noise level with a second-order predictor/corrector, and then integrated
back under a refinement prompt.  Only a tracked robot envelope is allowed to
change; flowers and all pixels outside a feathered safety envelope are restored
from the accepted input before delivery.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.h3_flow_retake import (  # noqa: E402
    h3_latent_frame_count,
    h3_model_frame_count,
    project_binary_masks_to_h3_latents,
)
from phiagent.rendering.minimax_h3 import (  # noqa: E402
    DIFFSYNTH_H3_COMMIT,
    MINIMAX_H3_MODELSCOPE_ID,
    MINIMAX_H3_NF4_MODEL_ID,
    build_flower_window_epl_constraint,
    file_sha256,
    verify_diffsynth_h3_source,
)
from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402


REFINEMENT_DIRECTIVE = """
flow_refinement_pass:
The retake video is the authoritative robot geometry, identity, pose, contact,
flower ordering, camera and timing. Refine only the robot surface inside the
provided protected retake region. Increase temporal coherence of the same
silver-and-graphite shell panels, dark face panel, shoulder joints, elbows,
wrists and five-finger hands. Keep every joint count, silhouette, pose, hand
contact and flower occlusion unchanged frame by frame. Remove transient human
skin or cloth textures. Do not redesign, relight, enlarge, shrink or reposition
the robot. Do not alter flowers, stems, leaves, vase, table, background, shadows,
reflections, exposure, depth of field or camera. The output remains one
continuous natural motion with no flicker, pulsing, morphing or texture crawl.
""".strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-video", type=Path, required=True)
    parser.add_argument("--motion-video", type=Path, required=True)
    parser.add_argument("--robot-reference", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--robot-mask", type=Path, action="append", required=True)
    parser.add_argument("--flower-mask", type=Path, required=True)
    parser.add_argument("--diffsynth-repo", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--source-frames", type=int, default=56)
    parser.add_argument("--target-sigma", type=float, default=0.12)
    parser.add_argument("--inversion-steps", type=int, default=8)
    parser.add_argument("--forward-steps", type=int, default=20)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=54 * 1024)
    parser.add_argument("--vram-reserve-gib", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--mask-dilation", type=int, default=18)
    parser.add_argument("--mask-close", type=int, default=9)
    parser.add_argument("--mask-temporal-radius", type=int, default=2)
    parser.add_argument("--latent-temporal-radius", type=int, default=1)
    parser.add_argument("--flower-protect-radius", type=int, default=5)
    parser.add_argument("--composite-feather-sigma", type=float, default=3.0)
    parser.add_argument("--model-mix", type=float, default=1.0)
    parser.add_argument("--base-detail-gain", type=float, default=0.0)
    parser.add_argument("--noise-refresh", type=float, default=0.0)
    parser.add_argument("--showcase-output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _package_versions() -> dict[str, str | None]:
    names = (
        "torch",
        "torchvision",
        "transformers",
        "diffsynth",
        "bitsandbytes",
        "opencv-python",
        "numpy",
        "av",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_state() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def _video_info(path: Path) -> dict[str, int | float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/", maxsplit=1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(numerator) / int(denominator),
        "frames": int(stream["nb_frames"]),
        "duration": float(payload["format"]["duration"]),
    }


def _extract_segment(
    source: Path,
    output: Path,
    *,
    start_frame: int,
    source_frames: int,
    model_frames: int,
    width: int,
    height: int,
) -> list[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    padding = model_frames - source_frames
    filters = [
        f"trim=start_frame={start_frame}:end_frame={start_frame + source_frames}",
        "setpts=PTS-STARTPTS",
        f"scale={width}:{height}:force_original_aspect_ratio=increase",
        f"crop={width}:{height}",
    ]
    if padding:
        filters.append(f"tpad=stop_mode=clone:stop_duration={padding / 24.0:.9f}")
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        ",".join(filters),
        "-frames:v",
        str(model_frames),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "10",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(command, check=True)
    return command


def _load_packed_masks(np: Any, path: Path) -> Any:
    data = np.load(path)
    required = {"packed", "height", "width", "bitorder"}
    if not required.issubset(data.files):
        raise ValueError(f"packed mask file has the wrong schema: {path}")
    height, width = int(data["height"]), int(data["width"])
    bitorder = str(data["bitorder"])
    unpacked = np.unpackbits(data["packed"], axis=1, bitorder=bitorder)
    return unpacked[:, : height * width].reshape(-1, height, width).astype(np.uint8) * 255


def _build_edit_masks(
    cv2: Any,
    np: Any,
    robot_masks: list[Any],
    flower_masks: Any,
    *,
    start_frame: int,
    source_frames: int,
    model_frames: int,
    dilation: int,
    close: int,
    temporal_radius: int,
    flower_protect_radius: int,
) -> list[Any]:
    union = np.maximum.reduce(robot_masks)
    if start_frame + source_frames > len(union):
        raise ValueError("requested frame range exceeds robot masks")
    if start_frame + source_frames > len(flower_masks):
        raise ValueError("requested frame range exceeds flower masks")
    selected = [union[index].copy() for index in range(start_frame, start_frame + source_frames)]
    flowers = [
        flower_masks[index].copy()
        for index in range(start_frame, start_frame + source_frames)
    ]
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    dilate_size = dilation * 2 + 1
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)
    )
    protect_size = flower_protect_radius * 2 + 1
    protect_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (protect_size, protect_size)
    )
    processed = []
    for mask, flower in zip(selected, flowers):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.dilate(mask, dilate_kernel)
        protected = cv2.dilate(flower, protect_kernel)
        mask[protected > 0] = 0
        processed.append(mask)
    if temporal_radius:
        processed = [
            np.maximum.reduce(
                processed[
                    max(0, index - temporal_radius) : min(
                        len(processed), index + temporal_radius + 1
                    )
                ]
            )
            for index in range(len(processed))
        ]
    while len(processed) < model_frames:
        processed.append(processed[-1].copy())
    return processed


def _save_mask_video(cv2: Any, masks: list[Any], output: Path, fps: int) -> None:
    height, width = masks[0].shape
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height), True
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open mask writer: {output}")
    for mask in masks:
        writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    writer.release()


def _temporal_metrics(np: Any, frames: list[Any], masks: list[Any]) -> dict[str, float]:
    if len(frames) < 3:
        return {"mean_abs_delta": 0.0, "mean_abs_jerk": 0.0}
    gray = [np.asarray(frame.convert("L"), dtype=np.float32) for frame in frames]
    deltas = []
    for index in range(1, len(gray)):
        roi = (masks[index] > 0) | (masks[index - 1] > 0)
        if roi.any():
            deltas.append(float(np.abs(gray[index] - gray[index - 1])[roi].mean()))
    jerks = []
    for index in range(2, len(gray)):
        roi = (masks[index] > 0) | (masks[index - 1] > 0) | (masks[index - 2] > 0)
        if roi.any():
            jerk = gray[index] - 2.0 * gray[index - 1] + gray[index - 2]
            jerks.append(float(np.abs(jerk)[roi].mean()))
    return {
        "mean_abs_delta": float(np.mean(deltas)) if deltas else 0.0,
        "mean_abs_jerk": float(np.mean(jerks)) if jerks else 0.0,
    }


def _freeze_sources(experiment: Path) -> list[dict[str, object]]:
    destination = experiment / "provenance" / "execution-sources"
    destination.mkdir(parents=True, exist_ok=True)
    sources = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "phiagent" / "rendering" / "h3_flow_retake.py",
        PROJECT_ROOT / "phiagent" / "rendering" / "minimax_h3.py",
        PROJECT_ROOT / "external" / "DiffSynth-Studio" / "diffsynth" / "pipelines" / "minimax_h3_audio_video.py",
        PROJECT_ROOT / "external" / "DiffSynth-Studio" / "diffsynth" / "diffusion" / "flow_match.py",
    )
    records = []
    for source in sources:
        target = destination / source.name
        shutil.copy2(source, target)
        records.append(
            {"source": str(source), "copy": str(target), "sha256": file_sha256(target)}
        )
    return records


def main() -> int:
    args = _parser().parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    metadata_path = experiment / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"experiment already exists: {experiment}")
    experiment.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, *sys.argv]
    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "minimax_h3_nf4_protected_flow_ode_inversion_heun_refinement",
        "status": "preflight_started",
        "honest_status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "command_shell": shlex.join(command),
    }
    _write_json(metadata_path, record)
    try:
        paths = {
            "base_video": args.base_video.expanduser().resolve(),
            "motion_video": args.motion_video.expanduser().resolve(),
            "robot_reference": args.robot_reference.expanduser().resolve(),
            "prompt_file": args.prompt_file.expanduser().resolve(),
            "flower_mask": args.flower_mask.expanduser().resolve(),
            "diffsynth_repo": args.diffsynth_repo.expanduser().resolve(),
            "model_base_path": args.model_base_path.expanduser().resolve(),
        }
        robot_mask_paths = [path.expanduser().resolve() for path in args.robot_mask]
        for label, path in paths.items():
            if label == "model_base_path":
                if not path.is_dir():
                    raise FileNotFoundError(path)
            elif not path.is_file() and label != "diffsynth_repo":
                raise FileNotFoundError(path)
        if not paths["diffsynth_repo"].is_dir():
            raise FileNotFoundError(paths["diffsynth_repo"])
        for path in robot_mask_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        if not 0.0 < args.target_sigma < 1.0:
            raise ValueError("target-sigma must be in (0, 1)")
        if args.inversion_steps < 1 or args.forward_steps < 1:
            raise ValueError("solver step counts must be positive")
        if not 0.0 <= args.noise_refresh <= 0.25:
            raise ValueError("noise-refresh must be in [0, 0.25]")
        if not 0.0 < args.model_mix <= 1.0:
            raise ValueError("model-mix must be in (0, 1]")
        if not 0.0 <= args.base_detail_gain <= 0.5:
            raise ValueError("base-detail-gain must be in [0, 0.5]")
        base_info = _video_info(paths["base_video"])
        motion_info = _video_info(paths["motion_video"])
        if int(base_info["frames"]) != int(motion_info["frames"]):
            raise ValueError("base and motion videos must have the same frame count")
        if args.start_frame < 0 or args.source_frames <= 0:
            raise ValueError("start-frame and source-frames are invalid")
        if args.start_frame + args.source_frames > int(base_info["frames"]):
            raise ValueError("requested segment exceeds the video")
        model_frames = h3_model_frame_count(args.source_frames)
        latent_frames = h3_latent_frame_count(model_frames)
        source_revision = verify_diffsynth_h3_source(paths["diffsynth_repo"])
        gpus, inventory_raw, processes_raw = query_gpus()
        selected_gpu = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
        input_dir = experiment / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        base_segment = input_dir / "accepted-base.mp4"
        motion_segment = input_dir / "motion-reference.mp4"
        base_extract = _extract_segment(
            paths["base_video"],
            base_segment,
            start_frame=args.start_frame,
            source_frames=args.source_frames,
            model_frames=model_frames,
            width=args.width,
            height=args.height,
        )
        motion_extract = _extract_segment(
            paths["motion_video"],
            motion_segment,
            start_frame=args.start_frame,
            source_frames=args.source_frames,
            model_frames=model_frames,
            width=args.width,
            height=args.height,
        )
        base_prompt = paths["prompt_file"].read_text().strip()
        if not base_prompt:
            raise ValueError("prompt file is empty")
        source_prompt = base_prompt + build_flower_window_epl_constraint(
            args.start_frame,
            args.source_frames,
            full_frame_count=int(base_info["frames"]),
        )
        target_prompt = source_prompt + "\n\n" + REFINEMENT_DIRECTIVE
        source_prompt_path = input_dir / "source-prompt.txt"
        target_prompt_path = input_dir / "target-prompt.txt"
        source_prompt_path.write_text(source_prompt + "\n")
        target_prompt_path.write_text(target_prompt + "\n")

        import cv2
        import numpy as np

        robot_masks = [_load_packed_masks(np, path) for path in robot_mask_paths]
        flower_masks = _load_packed_masks(np, paths["flower_mask"])
        edit_masks = _build_edit_masks(
            cv2,
            np,
            robot_masks,
            flower_masks,
            start_frame=args.start_frame,
            source_frames=args.source_frames,
            model_frames=model_frames,
            dilation=args.mask_dilation,
            close=args.mask_close,
            temporal_radius=args.mask_temporal_radius,
            flower_protect_radius=args.flower_protect_radius,
        )
        mask_video = input_dir / "protected-edit-mask.mp4"
        _save_mask_video(cv2, edit_masks, mask_video, args.fps)
        mask_fraction = float(np.mean([np.mean(mask > 0) for mask in edit_masks]))
        if mask_fraction <= 0.005 or mask_fraction >= 0.35:
            raise ValueError(f"unsafe mean edit-mask fraction: {mask_fraction:.6f}")

        record.update(
            {
                "status": "preflight_complete" if args.preflight_only else "running",
                "source_revision": source_revision,
                "expected_source_revision": DIFFSYNTH_H3_COMMIT,
                "model": {
                    "base": "MiniMax-H3",
                    "weights": MINIMAX_H3_NF4_MODEL_ID,
                    "processor": MINIMAX_H3_MODELSCOPE_ID,
                    "quantization": "third-party prequantized bitsandbytes NF4",
                },
                "config": {
                    "start_frame": args.start_frame,
                    "source_frames": args.source_frames,
                    "model_frames": model_frames,
                    "latent_frames": latent_frames,
                    "target_sigma": args.target_sigma,
                    "inversion_steps": args.inversion_steps,
                    "forward_steps": args.forward_steps,
                    "solver": "Heun predictor-corrector in actual flow sigma",
                    "flow_shift": args.flow_shift,
                    "cfg_scale": args.cfg_scale,
                    "seed": args.seed,
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "mask_dilation": args.mask_dilation,
                    "mask_close": args.mask_close,
                    "mask_temporal_radius": args.mask_temporal_radius,
                    "latent_temporal_radius": args.latent_temporal_radius,
                    "flower_protect_radius": args.flower_protect_radius,
                    "composite_feather_sigma": args.composite_feather_sigma,
                    "model_mix": args.model_mix,
                    "base_detail_gain": args.base_detail_gain,
                    "noise_refresh": args.noise_refresh,
                    "coordinate_frame": "camera:H3_832x480_pixels and absolute source frame",
                },
                "inputs": {
                    **{
                        label: {"path": str(path), "sha256": file_sha256(path)}
                        for label, path in paths.items()
                        if path.is_file()
                    },
                    "robot_masks": [
                        {"path": str(path), "sha256": file_sha256(path)}
                        for path in robot_mask_paths
                    ],
                    "base_info": base_info,
                    "motion_info": motion_info,
                    "base_segment": str(base_segment),
                    "base_segment_sha256": file_sha256(base_segment),
                    "motion_segment": str(motion_segment),
                    "motion_segment_sha256": file_sha256(motion_segment),
                    "base_extract_command": base_extract,
                    "motion_extract_command": motion_extract,
                    "source_prompt": str(source_prompt_path),
                    "target_prompt": str(target_prompt_path),
                    "mask_video": str(mask_video),
                    "mean_edit_mask_fraction": mask_fraction,
                },
                "selected_gpu": asdict(selected_gpu),
                "gpu_inventory": [asdict(gpu) for gpu in gpus],
                "gpu_inventory_raw": inventory_raw,
                "gpu_processes_raw": processes_raw,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "packages": _package_versions(),
                "git": _git_state(),
                "execution_sources": _freeze_sources(experiment),
                "limitations": [
                    "This run uses third-party NF4 H3 weights because official BF16 weights are not installed on the execution host.",
                    "The accepted video remains authoritative outside the tracked robot envelope.",
                    "Human review is required before promotion to a complete 27.5-second delivery.",
                ],
            }
        )
        _write_json(metadata_path, record)
        if args.preflight_only:
            print(json.dumps({"status": record["status"], "experiment": str(experiment)}))
            return 0

        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu.physical_index)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(paths["diffsynth_repo"]), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(paths["model_base_path"])
        sys.path.insert(0, str(paths["diffsynth_repo"]))

        import torch
        from PIL import Image
        from tqdm import tqdm
        from diffsynth.pipelines.minimax_h3_audio_video import (
            MiniMaxH3Pipeline,
            ModelConfig,
        )
        from diffsynth.utils.data import save_video
        from diffsynth.utils.data.audio_video import read_video_audio

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("selected physical GPU did not map to exactly one CUDA device")
        free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
        free_gib = free_bytes / 1024**3
        vram_limit_gib = free_gib - args.vram_reserve_gib
        if vram_limit_gib <= 8:
            raise RuntimeError("GPU reserve leaves too little H3 VRAM")
        record["runtime"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "logical_gpu": torch.cuda.get_device_name(0),
            "free_gib_at_load": free_gib,
            "total_gib": total_bytes / 1024**3,
            "vram_limit_gib": vram_limit_gib,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
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
                    origin_file_pattern=pattern,
                    **vram_config,
                )
                for pattern in (
                    "minimax-h3-ref2va-nf4.safetensors",
                    "minimax-h3-text-encoder-nf4.safetensors",
                    "video_vae_nf4.safetensors",
                    "audio_vae_nf4.safetensors",
                )
            ],
            processor_config=ModelConfig(
                model_id=MINIMAX_H3_MODELSCOPE_ID,
                origin_file_pattern="Ref2VA/processor/",
            ),
            vram_limit=vram_limit_gib,
        )
        robot = Image.open(paths["robot_reference"]).convert("RGB")
        base_frames, _, _ = read_video_audio(
            str(base_segment),
            height=args.height,
            width=args.width,
            num_frames=model_frames,
            fps=args.fps,
            audio_sample_rate=pipe.audio_vae.sample_rate,
        )
        motion_frames, _, _ = read_video_audio(
            str(motion_segment),
            height=args.height,
            width=args.width,
            num_frames=model_frames,
            fps=args.fps,
            audio_sample_rate=pipe.audio_vae.sample_rate,
        )
        if len(base_frames) != model_frames or len(motion_frames) != model_frames:
            raise RuntimeError("segment decoder returned the wrong frame count")

        latent_mask_np = project_binary_masks_to_h3_latents(
            edit_masks,
            latent_frames=latent_frames,
            latent_height=args.height // 16,
            latent_width=args.width // 16,
            cv2=cv2,
            np=np,
            temporal_radius=args.latent_temporal_radius,
            patch_size=2,
        )

        def prepare_condition(prompt: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            pipe.scheduler.set_timesteps(args.forward_steps, shift=args.flow_shift)
            pipe.scheduler_audio.set_timesteps(args.forward_steps, shift=3.0)
            silence = torch.zeros(
                (2, round(model_frames / args.fps * pipe.audio_vae.sample_rate)),
                dtype=torch.float32,
            )
            inputs_posi: dict[str, Any] = {"prompt": prompt}
            inputs_nega: dict[str, Any] = {"negative_prompt": " "}
            inputs_shared: dict[str, Any] = {
                "cfg_scale": args.cfg_scale,
                "height": args.height,
                "width": args.width,
                "num_frames": model_frames,
                "seed": args.seed,
                "rand_device": "cpu",
                "tiled": True,
                "tile_size": 256,
                "tile_overlap": 64,
                "use_gradient_checkpointing": False,
                "use_gradient_checkpointing_offload": False,
                "keyframes": None,
                "keyframe_indices": None,
                "references": [
                    {"type": "image", "image": robot},
                    {"type": "video", "video": motion_frames},
                ],
                "ref_image_short_edge": 768,
                "ref_video_short_edge": 480,
                "ref_video_max_pixels": args.height * args.width,
                "retake_video": base_frames,
                "frame_regions_to_retake": [(0, model_frames)],
                "retake_audio": (silence, pipe.audio_vae.sample_rate),
                "seconds_regions_to_retake": [],
                "imgvid_cond_noise_aug": pipe.imgvid_cond_noise_aug,
                "audio_cond_noise_aug": pipe.audio_cond_noise_aug,
            }
            for unit in pipe.units:
                inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                    unit, pipe, inputs_shared, inputs_posi, inputs_nega
                )
            latent_mask = torch.from_numpy(latent_mask_np).to(
                device=pipe.device, dtype=pipe.torch_dtype
            )[None, None]
            expected_shape = inputs_shared["input_latents_video"].shape
            if latent_mask.shape[2:] != expected_shape[2:]:
                raise RuntimeError(
                    f"latent mask {tuple(latent_mask.shape)} does not match H3 latent {tuple(expected_shape)}"
                )
            inputs_shared["denoise_mask_video"] = latent_mask.expand(
                expected_shape[0], 1, *expected_shape[2:]
            )
            inputs_shared["video_latents"] = inputs_shared["input_latents_video"].clone()
            inputs_shared["audio_latents"] = inputs_shared["input_latents_audio"].clone()
            return inputs_shared, inputs_posi, inputs_nega

        def predict_velocity(
            shared: dict[str, Any],
            posi: dict[str, Any],
            nega: dict[str, Any],
            latents: Any,
            sigma: float,
        ) -> Any:
            shared["video_latents"] = latents
            timestep_video = torch.tensor(
                [sigma * 1000.0], dtype=torch.float32, device=pipe.device
            )
            timestep_audio = torch.zeros(1, dtype=torch.float32, device=pipe.device)
            velocity, _ = pipe.cfg_guided_model_fn(
                pipe.model_fn,
                args.cfg_scale,
                shared,
                posi,
                nega,
                dit=pipe.dit,
                timestep_video=timestep_video,
                timestep_audio=timestep_audio,
            )
            return velocity

        def heun_integrate(
            shared: dict[str, Any],
            posi: dict[str, Any],
            nega: dict[str, Any],
            latents: Any,
            sigmas: list[float],
            label: str,
        ) -> Any:
            pipe.load_models_to_device(pipe.in_iteration_models)
            mask = shared["denoise_mask_video"]
            stable = shared["input_latents_video"]
            for index, (sigma, sigma_next) in enumerate(
                tqdm(list(zip(sigmas[:-1], sigmas[1:])), desc=label)
            ):
                delta = sigma_next - sigma
                velocity = predict_velocity(shared, posi, nega, latents, sigma)
                predicted = latents + velocity * delta
                predicted = predicted * mask + stable * (1.0 - mask)
                velocity_next = predict_velocity(
                    shared, posi, nega, predicted, sigma_next
                )
                latents = latents + (velocity + velocity_next) * (0.5 * delta)
                latents = latents * mask + stable * (1.0 - mask)
                record.setdefault("solver_events", []).append(
                    {
                        "phase": label,
                        "step": index,
                        "sigma": sigma,
                        "sigma_next": sigma_next,
                    }
                )
                _write_json(metadata_path, record)
            return latents

        pipe.load_models_to_device(pipe.in_iteration_models)
        with torch.no_grad():
            source_shared, source_posi, source_nega = prepare_condition(source_prompt)
            data_latents = source_shared["input_latents_video"].clone()
            inversion_sigmas = [
                args.target_sigma * index / args.inversion_steps
                for index in range(args.inversion_steps + 1)
            ]
            inverted = heun_integrate(
                source_shared,
                source_posi,
                source_nega,
                data_latents,
                inversion_sigmas,
                "flow-inversion",
            )
            if args.noise_refresh:
                refresh = pipe.generate_noise(
                    inverted.shape,
                    seed=args.seed + 1,
                    rand_device="cpu",
                    rand_torch_dtype=torch.float32,
                    device=pipe.device,
                    torch_dtype=pipe.torch_dtype,
                )
                mask = source_shared["denoise_mask_video"]
                inverted = inverted + args.noise_refresh * args.target_sigma * refresh * mask
            inverted_cpu = inverted.to("cpu")
            inversion_digest = hashlib.sha256(
                inverted_cpu.float().numpy().tobytes()
            ).hexdigest()
            del source_shared, source_posi, source_nega, data_latents, inverted
            gc.collect()
            torch.cuda.empty_cache()

            target_shared, target_posi, target_nega = prepare_condition(target_prompt)
            inverted = inverted_cpu.to(device=pipe.device, dtype=pipe.torch_dtype)
            forward_sigmas = [
                args.target_sigma * (1.0 - index / args.forward_steps)
                for index in range(args.forward_steps + 1)
            ]
            refined_latents = heun_integrate(
                target_shared,
                target_posi,
                target_nega,
                inverted,
                forward_sigmas,
                "flow-refinement",
            )
            pipe.load_models_to_device(["video_vae"])
            decoded = pipe.video_vae.decode_video(
                refined_latents,
                dtype=pipe.torch_dtype,
                tiled=True,
                tile_size=256,
                tile_overlap=64,
            )
            generated_frames = pipe.vae_output_to_video(decoded, min_value=0, max_value=1)

        raw_output = experiment / "raw-h3-flow-refinement.mp4"
        save_video(
            generated_frames[: args.source_frames],
            str(raw_output),
            fps=args.fps,
            quality=10,
            ffmpeg_params=["-crf", "10", "-preset", "medium", "-pix_fmt", "yuv420p"],
        )
        protected_frames = []
        for base_frame, generated_frame, mask in zip(
            base_frames[: args.source_frames],
            generated_frames[: args.source_frames],
            edit_masks[: args.source_frames],
        ):
            base_array = np.asarray(base_frame.convert("RGB"))
            generated_array = np.asarray(generated_frame.convert("RGB"))
            alpha = mask.astype(np.float32) / 255.0
            if args.composite_feather_sigma > 0:
                alpha = cv2.GaussianBlur(
                    alpha, (0, 0), args.composite_feather_sigma
                ).clip(0.0, 1.0)
                support_radius = max(1, round(args.composite_feather_sigma * 4))
                support_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (support_radius * 2 + 1, support_radius * 2 + 1),
                )
                support = cv2.dilate(mask, support_kernel) > 0
                alpha[~support] = 0.0
            detail_alpha = alpha.copy()
            alpha *= args.model_mix
            composite = np.rint(
                generated_array.astype(np.float32) * alpha[..., None]
                + base_array.astype(np.float32) * (1.0 - alpha[..., None])
            )
            if args.base_detail_gain:
                base_blur = cv2.GaussianBlur(base_array, (0, 0), 1.0).astype(np.float32)
                base_detail = base_array.astype(np.float32) - base_blur
                composite += (
                    args.base_detail_gain * detail_alpha[..., None] * base_detail
                )
            composite = composite.clip(0, 255).astype(np.uint8)
            protected_frames.append(Image.fromarray(composite, mode="RGB"))

        protected_output = experiment / "protected-h3-flow-refinement.mp4"
        save_video(
            protected_frames,
            str(protected_output),
            fps=args.fps,
            quality=10,
            ffmpeg_params=["-crf", "10", "-preset", "medium", "-pix_fmt", "yuv420p"],
        )
        base_metrics = _temporal_metrics(
            np, base_frames[: args.source_frames], edit_masks[: args.source_frames]
        )
        candidate_metrics = _temporal_metrics(
            np, protected_frames, edit_masks[: args.source_frames]
        )
        safety_radius = max(1, round(args.composite_feather_sigma * 4))
        safety_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (safety_radius * 2 + 1, safety_radius * 2 + 1)
        )
        outside_exact = True
        for base_frame, protected_frame, mask in zip(
            base_frames[: args.source_frames], protected_frames, edit_masks
        ):
            safety = cv2.dilate(mask, safety_kernel) > 0
            base_array = np.asarray(base_frame.convert("RGB"))
            protected_array = np.asarray(protected_frame.convert("RGB"))
            if not np.array_equal(base_array[~safety], protected_array[~safety]):
                outside_exact = False
                break
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(protected_output), "-f", "null", "-"],
            check=True,
        )
        result_info = _video_info(protected_output)
        acceptance = {
            "full_decode": int(result_info["frames"]) == args.source_frames,
            "outside_safety_exact_preencode": outside_exact,
            "edit_mask_fraction_bounded": mask_fraction < 0.35,
            "roi_jerk_ratio": (
                candidate_metrics["mean_abs_jerk"]
                / max(base_metrics["mean_abs_jerk"], 1e-6)
            ),
            "automatic_temporal_gate": candidate_metrics["mean_abs_jerk"]
            <= max(base_metrics["mean_abs_jerk"] * 1.25, base_metrics["mean_abs_jerk"] + 1.0),
            "human_review": False,
        }
        record.update(
            {
                "status": "completed",
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "inverted_latent_sha256_float32": inversion_digest,
                "outputs": {
                    "raw": str(raw_output),
                    "raw_sha256": file_sha256(raw_output),
                    "protected": str(protected_output),
                    "protected_sha256": file_sha256(protected_output),
                    "protected_info": result_info,
                },
                "metrics": {"base": base_metrics, "candidate": candidate_metrics},
                "acceptance": acceptance,
            }
        )
        _write_json(metadata_path, record)
        if args.showcase_output is not None:
            showcase = args.showcase_output.expanduser().resolve()
            showcase.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(protected_output, showcase)
            record["showcase_output"] = str(showcase)
            record["showcase_sha256"] = file_sha256(showcase)
            _write_json(metadata_path, record)
        print(
            json.dumps(
                {
                    "status": record["status"],
                    "experiment": str(experiment),
                    "output": str(protected_output),
                    "acceptance": acceptance,
                }
            )
        )
        return 0
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "honest_status": "PARTIAL",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(metadata_path, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
