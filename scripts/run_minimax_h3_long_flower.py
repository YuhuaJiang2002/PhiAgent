#!/usr/bin/env python3
"""Generate one reproducible set of overlapping MiniMax-H3 flower windows.

The checkpoint is loaded once and every window uses the same physical GPU,
robot reference, base prompt, seed, and numerical settings.  Each prompt adds
absolute EPL phase/contact constraints; post-processing and stitching remain a
separate immutable CPU experiment.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.minimax_h3 import (  # noqa: E402
    DIFFSYNTH_H3_COMMIT,
    MINIMAX_H3_MODELSCOPE_ID,
    MINIMAX_H3_NF4_MODEL_ID,
    MiniMaxH3ValidationConfig,
    build_flower_window_epl_constraint,
    file_sha256,
    plan_h3_long_windows,
    verify_diffsynth_h3_source,
)
from phiagent.rendering.wan_animate import query_gpus  # noqa: E402
from scripts.run_minimax_h3_flower_validation import (  # noqa: E402
    _Tee,
    _git_state,
    _packages,
    _raise_on_termination,
    _write_json,
)


def _video_info(ffprobe: Path, video: Path) -> dict[str, int | float]:
    completed = subprocess.run(
        [
            str(ffprobe),
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
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/", maxsplit=1)
    frames = stream.get("nb_frames")
    if frames in (None, "N/A"):
        raise RuntimeError(f"ffprobe did not report an exact frame count for {video}")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(numerator) / int(denominator),
        "frames": int(frames),
        "duration": float(payload["format"]["duration"]),
    }


def _extract_window(
    ffmpeg: Path,
    source: Path,
    output: Path,
    start_frame: int,
    frame_count: int,
    width: int,
    height: int,
) -> list[str]:
    command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        (
            f"trim=start_frame={start_frame}:end_frame={start_frame + frame_count},"
            f"setpts=PTS-STARTPTS,scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        ),
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    return command


def _extract_continuation_frame(
    ffmpeg: Path,
    source: Path,
    output: Path,
    frame_index: int,
) -> list[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-frames:v",
        "1",
        str(output),
    ]
    subprocess.run(command, check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"failed to extract continuation frame {frame_index}")
    return command


def _freeze_execution_sources(experiment: Path) -> list[dict[str, str]]:
    destination = experiment / "provenance" / "execution-sources"
    destination.mkdir(parents=True, exist_ok=True)
    sources = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "run_minimax_h3_flower_validation.py",
        PROJECT_ROOT / "phiagent" / "rendering" / "minimax_h3.py",
        PROJECT_ROOT / "requirements" / "minimax-h3.txt",
    )
    records = []
    for source in sources:
        target = destination / source.name
        shutil.copy2(source, target)
        records.append(
            {
                "source": str(source),
                "frozen_copy": str(target),
                "sha256": file_sha256(target),
            }
        )
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--robot-reference", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--diffsynth-repo", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/usr/bin/ffprobe"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=54 * 1024)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--window-frames", type=int, default=124)
    parser.add_argument("--overlap-frames", type=int, default=28)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--reference-short-edge", type=int, default=768)
    parser.add_argument("--reference-video-short-edge", type=int, default=480)
    parser.add_argument("--vram-reserve-gib", type=float, default=8.0)
    parser.add_argument("--continuation-conditioning", action="store_true")
    parser.add_argument("--reuse-first-window", type=Path)
    parser.add_argument("--reuse-prefix-window", type=Path, action="append", default=[])
    parser.add_argument("--window-limit", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    metadata_path = experiment / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"experiment already initialized: {metadata_path}")
    experiment.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, *sys.argv]
    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "minimax_h3_nf4_ref2va_epl_overlapping_full_video",
        "status": "preflight_started",
        "honest_status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "command_shell": shlex.join(command),
        "seed": args.seed,
        "source_revision": DIFFSYNTH_H3_COMMIT,
    }
    _write_json(metadata_path, record)
    for signal_name in ("SIGHUP", "SIGTERM"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), _raise_on_termination)
    try:
        paths = {
            "source_video": args.source_video.expanduser().resolve(),
            "robot_reference": args.robot_reference.expanduser().resolve(),
            "prompt_file": args.prompt_file.expanduser().resolve(),
            "diffsynth_repo": args.diffsynth_repo.expanduser().resolve(),
            "model_base_path": args.model_base_path.expanduser().resolve(),
            "ffmpeg": args.ffmpeg.expanduser().resolve(),
            "ffprobe": args.ffprobe.expanduser().resolve(),
        }
        validation = MiniMaxH3ValidationConfig(
            source_video=paths["source_video"],
            robot_reference=paths["robot_reference"],
            prompt_file=paths["prompt_file"],
            diffsynth_repo=paths["diffsynth_repo"],
            model_base_path=paths["model_base_path"],
            width=args.width,
            height=args.height,
            fps=args.fps,
            num_frames=args.window_frames,
            steps=args.steps,
            seed=args.seed,
            minimum_free_gpu_mib=args.minimum_free_gpu_mib,
            requested_gpu=args.gpu,
        )
        validation.validate()
        reuse_first = (
            args.reuse_first_window.expanduser().resolve()
            if args.reuse_first_window is not None
            else None
        )
        reuse_prefix = ([reuse_first] if reuse_first is not None else []) + [
            path.expanduser().resolve() for path in args.reuse_prefix_window
        ]
        if args.continuation_conditioning and not reuse_prefix:
            raise ValueError("continuation conditioning requires a reused prefix window")
        for reused in reuse_prefix:
            if not reused.is_file() or reused.stat().st_size == 0:
                raise ValueError(f"reused prefix window is missing or empty: {reused}")
        for label in ("ffmpeg", "ffprobe"):
            if not paths[label].is_file():
                raise ValueError(f"{label} does not exist: {paths[label]}")
        source_info = _video_info(paths["ffprobe"], paths["source_video"])
        if abs(float(source_info["fps"]) - args.fps) > 1e-6:
            raise ValueError(f"source FPS is {source_info['fps']}, expected {args.fps}")
        full_windows = plan_h3_long_windows(
            int(source_info["frames"]),
            window_frames=args.window_frames,
            overlap_frames=args.overlap_frames,
        )
        if args.window_limit is not None and not 1 <= args.window_limit <= len(full_windows):
            raise ValueError("window-limit must be within the planned window count")
        windows = (
            full_windows[: args.window_limit]
            if args.window_limit is not None
            else full_windows
        )
        if len(reuse_prefix) > len(windows):
            raise ValueError("reused prefix is longer than the planned window set")
        source_commit = verify_diffsynth_h3_source(paths["diffsynth_repo"])
        gpus, inventory_raw, processes_raw = query_gpus()
        selected = validation.select_gpu(gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(paths["diffsynth_repo"]), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(paths["model_base_path"])
        sys.path.insert(0, str(paths["diffsynth_repo"]))

        inputs = experiment / "input"
        clips = inputs / "windows"
        prompts = inputs / "prompts"
        clips.mkdir(parents=True, exist_ok=True)
        prompts.mkdir(parents=True, exist_ok=True)
        base_prompt = paths["prompt_file"].read_text().strip()
        if not base_prompt:
            raise ValueError("prompt file is empty")
        window_records: list[dict[str, object]] = []
        for window in windows:
            clip = clips / f"window-{window.index:02d}-{window.start_frame:04d}.mp4"
            extract_command = _extract_window(
                paths["ffmpeg"],
                paths["source_video"],
                clip,
                window.start_frame,
                window.frame_count,
                args.width,
                args.height,
            )
            clip_info = _video_info(paths["ffprobe"], clip)
            if int(clip_info["frames"]) != window.frame_count:
                raise RuntimeError(f"window {window.index} decoded {clip_info['frames']} frames")
            prompt = base_prompt + build_flower_window_epl_constraint(
                window.start_frame,
                window.frame_count,
                full_frame_count=int(source_info["frames"]),
            )
            prompt_path = prompts / f"window-{window.index:02d}-{window.start_frame:04d}.txt"
            prompt_path.write_text(prompt + "\n")
            window_records.append(
                {
                    **asdict(window),
                    "status": "prepared",
                    "input": str(clip),
                    "input_sha256": file_sha256(clip),
                    "input_info": clip_info,
                    "extraction_command": extract_command,
                    "prompt": str(prompt_path),
                    "prompt_sha256": file_sha256(prompt_path),
                }
            )
        record.update(
            {
                "status": "preflight_complete" if args.preflight_only else "running",
                "source_revision": source_commit,
                "model": {
                    "base": "MiniMax-H3",
                    "weights": MINIMAX_H3_NF4_MODEL_ID,
                    "processor": MINIMAX_H3_MODELSCOPE_ID,
                    "quantization": "third-party prequantized bitsandbytes NF4",
                },
                "config": {
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "window_frames": args.window_frames,
                    "overlap_frames": args.overlap_frames,
                    "steps": args.steps,
                    "seed": args.seed,
                    "reference_short_edge": args.reference_short_edge,
                    "reference_video_short_edge": args.reference_video_short_edge,
                    "vram_reserve_gib": args.vram_reserve_gib,
                    "pytorch_cuda_alloc_conf": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
                    "contact_interval_frames": [236, 316],
                    "continuation_conditioning": args.continuation_conditioning,
                    "window_limit": args.window_limit,
                    "planned_full_window_count": len(full_windows),
                    "full_timeline_covered": len(windows) == len(full_windows),
                    "coordinate_frame": "camera:full_source_pixels and absolute_frame_index",
                },
                "inputs": {
                    "source_video": str(paths["source_video"]),
                    "source_sha256": file_sha256(paths["source_video"]),
                    "source_info": source_info,
                    "robot_reference": str(paths["robot_reference"]),
                    "robot_reference_sha256": file_sha256(paths["robot_reference"]),
                    "prompt_file": str(paths["prompt_file"]),
                    "prompt_sha256": file_sha256(paths["prompt_file"]),
                    "reuse_first_window": str(reuse_first) if reuse_first else None,
                    "reuse_first_window_sha256": (
                        file_sha256(reuse_first) if reuse_first else None
                    ),
                    "reuse_prefix_windows": [str(path) for path in reuse_prefix],
                    "reuse_prefix_window_sha256": [
                        file_sha256(path) for path in reuse_prefix
                    ],
                },
                "windows": window_records,
                "execution_sources": _freeze_execution_sources(experiment),
                "selected_gpu": asdict(selected),
                "gpu_inventory": [asdict(gpu) for gpu in gpus],
                "gpu_inventory_raw": inventory_raw,
                "gpu_processes_raw": processes_raw,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "hostname": os.uname().nodename,
                "python": sys.version,
                "packages": _packages(),
                "git": _git_state(PROJECT_ROOT),
                "limitations": [
                    "The visual model is third-party NF4 MiniMax-H3, not official BF16 H3 or PhiZero.",
                    "EPL phase/contact text conditions visual generation; physical execution is independently evidenced only by the recorded MuJoCo insertion experiment.",
                    "Independent overlapping windows do not share diffusion state; stitching must enforce identity and temporal gates.",
                ],
            }
        )
        _write_json(metadata_path, record)
        if args.preflight_only:
            print(json.dumps({"experiment": str(experiment), "status": record["status"]}))
            return 0

        log_path = experiment / "inference.log"
        with log_path.open("w", encoding="utf-8") as log:
            with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
                import torch
                from PIL import Image
                from diffsynth.pipelines.minimax_h3_audio_video import (
                    MiniMaxH3Pipeline,
                    ModelConfig,
                )
                from diffsynth.utils.data.audio_video import read_video_audio, write_video_audio

                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise RuntimeError("selected physical GPU did not map to one CUDA device")
                free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
                free_gib = free_bytes / 1024**3
                vram_limit_gib = free_gib - args.vram_reserve_gib
                if vram_limit_gib <= 8:
                    raise RuntimeError("GPU reserve leaves too little H3 VRAM")
                record["runtime"] = {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
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
                previous_result: Path | None = None
                previous_start: int | None = None
                for item in window_records:
                    window_dir = (
                        experiment
                        / "windows"
                        / f"window-{int(item['index']):02d}-{int(item['start_frame']):04d}"
                    )
                    window_dir.mkdir(parents=True, exist_ok=True)
                    result = window_dir / "raw-h3-nf4.mp4"
                    item_index = int(item["index"])
                    if item_index < len(reuse_prefix):
                        reused_path = reuse_prefix[item_index]
                        shutil.copy2(reused_path, result)
                        result_info = _video_info(paths["ffprobe"], result)
                        if int(result_info["frames"]) != args.window_frames:
                            raise RuntimeError("reused first window has the wrong frame count")
                        item.update(
                            {
                                "status": "completed",
                                "started_at": datetime.now(timezone.utc).isoformat(),
                                "completed_at": datetime.now(timezone.utc).isoformat(),
                                "result": str(result),
                                "result_sha256": file_sha256(result),
                                "result_info": result_info,
                                "reused": True,
                                "reused_from": str(reused_path),
                            }
                        )
                        previous_result = result
                        previous_start = int(item["start_frame"])
                        _write_json(metadata_path, record)
                        print(
                            json.dumps({"event": "window_reused", "index": item_index}),
                            flush=True,
                        )
                        continue
                    item["status"] = "running"
                    item["started_at"] = datetime.now(timezone.utc).isoformat()
                    _write_json(metadata_path, record)
                    print(
                        json.dumps(
                            {
                                "event": "window_started",
                                "index": item["index"],
                                "start_frame": item["start_frame"],
                            }
                        ),
                        flush=True,
                    )
                    frames, _, _ = read_video_audio(
                        str(item["input"]),
                        height=args.height,
                        width=args.width,
                        num_frames=args.window_frames,
                        fps=args.fps,
                        audio_sample_rate=pipe.audio_vae.sample_rate,
                    )
                    if len(frames) != args.window_frames:
                        raise RuntimeError(f"window {item['index']} decoder returned {len(frames)}")
                    prompt = Path(str(item["prompt"])).read_text().strip()
                    references = [{"type": "image", "image": robot}]
                    if args.continuation_conditioning:
                        if previous_result is None or previous_start is None:
                            raise RuntimeError("previous window is unavailable for continuation")
                        continuation_index = int(item["start_frame"]) - previous_start
                        continuation_path = (
                            experiment
                            / "input"
                            / "continuation"
                            / f"window-{int(item['index']):02d}-{int(item['start_frame']):04d}.png"
                        )
                        continuation_command = _extract_continuation_frame(
                            paths["ffmpeg"],
                            previous_result,
                            continuation_path,
                            continuation_index,
                        )
                        continuation = Image.open(continuation_path).convert("RGB")
                        references.append({"type": "image", "image": continuation})
                        prompt += (
                            "\ncontinuation_reference:\n"
                            "<Picture 2> is the exact robot appearance and pose from the preceding "
                            "overlapping window at this window's first absolute source frame. Match "
                            "<Subject 1> to <Picture 2> in the first frame without changing the "
                            "workspace, then continue the motion from <Video 1> without a pose reset."
                        )
                        conditioned_prompt = Path(str(item["prompt"])).with_name(
                            Path(str(item["prompt"])).stem + "-conditioned.txt"
                        )
                        conditioned_prompt.write_text(prompt + "\n")
                        item.update(
                            {
                                "continuation_reference": str(continuation_path),
                                "continuation_reference_sha256": file_sha256(continuation_path),
                                "continuation_source_frame": continuation_index,
                                "continuation_extraction_command": continuation_command,
                                "conditioned_prompt": str(conditioned_prompt),
                                "conditioned_prompt_sha256": file_sha256(conditioned_prompt),
                            }
                        )
                    references.append({"type": "video", "video": frames})
                    video, audio = pipe(
                        prompt=prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.window_frames,
                        num_inference_steps=args.steps,
                        seed=args.seed,
                        references=references,
                        ref_image_short_edge=args.reference_short_edge,
                        ref_video_short_edge=args.reference_video_short_edge,
                        ref_video_max_pixels=args.height * args.width,
                    )
                    write_video_audio(
                        video=video,
                        audio=audio,
                        output_path=str(result),
                        fps=args.fps,
                        audio_sample_rate=32000,
                    )
                    subprocess.run(
                        [str(paths["ffmpeg"]), "-v", "error", "-i", str(result), "-f", "null", "-"],
                        check=True,
                    )
                    result_info = _video_info(paths["ffprobe"], result)
                    if int(result_info["frames"]) != args.window_frames:
                        raise RuntimeError(f"window {item['index']} produced {result_info['frames']} frames")
                    item.update(
                        {
                            "status": "completed",
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                            "result": str(result),
                            "result_sha256": file_sha256(result),
                            "result_info": result_info,
                        }
                    )
                    _write_json(metadata_path, record)
                    print(json.dumps({"event": "window_completed", "index": item["index"]}), flush=True)
                    previous_result = result
                    previous_start = int(item["start_frame"])
                    del frames, video, audio, references
                    if "continuation" in locals():
                        continuation.close()
                        del continuation
                    gc.collect()
                    torch.cuda.empty_cache()

        checkpoint_root = paths["model_base_path"] / MINIMAX_H3_NF4_MODEL_ID
        processor_root = paths["model_base_path"] / MINIMAX_H3_MODELSCOPE_ID / "Ref2VA" / "processor"
        record.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "completed_windows": len(window_records),
                "checkpoint_files": [
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
                    for path in sorted(checkpoint_root.rglob("*"))
                    if path.is_file()
                ],
                "processor_files": [
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
                    for path in sorted(processor_root.rglob("*"))
                    if path.is_file()
                ],
                "acceptance": {
                    "all_windows_completed": True,
                    "all_windows_decoded": True,
                    "full_timeline_windows_completed": len(windows) == len(full_windows),
                    "full_stitch_evaluated": False,
                },
            }
        )
        _write_json(metadata_path, record)
        print(json.dumps({"experiment": str(experiment), "status": "completed"}))
        return 0
    except Exception as error:
        record.update(
            {
                "status": "failed",
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(metadata_path, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
