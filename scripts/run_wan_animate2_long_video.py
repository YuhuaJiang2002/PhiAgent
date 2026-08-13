#!/usr/bin/env python3
"""Generate overlapping Wan-Animate-2 windows for a long driving video.

This entry point performs model/checkpoint/GPU validation once, keeps one model
resident per disjoint physical-GPU pair, and assigns each pair a contiguous
temporal chain.  A chain may use the previous result's aligned source-camera
frame as its next identity anchor.  Stitching is a separate CPU step so raw
generation evidence remains immutable.
"""

from __future__ import annotations

import argparse
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
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.wan_animate import query_gpus  # noqa: E402
from phiagent.rendering.wan_animate2 import (  # noqa: E402
    WAN_ANIMATE2_MODEL_REVISION,
    file_sha256,
    select_wan_animate2_gpus,
    verify_wan_animate2_checkpoint,
    verify_wan_animate2_source,
    write_runtime_config,
)


@dataclass(frozen=True)
class Window:
    index: int
    start_frame: int
    input_frames: int
    expected_output_frames: int
    source_frames: int
    padded_frames: int


def plan_windows(
    frame_count: int,
    *,
    clip_len: int = 81,
    overlap: int = 16,
) -> tuple[Window, ...]:
    """Cover a timeline with overlapping ``clip_len - 1`` output windows."""

    if frame_count < clip_len:
        raise ValueError("frame_count must be at least clip_len")
    if clip_len < 5 or (clip_len - 1) % 4:
        raise ValueError("clip_len must satisfy clip_len = 4n + 1")
    output_frames = clip_len - 1
    if not 1 <= overlap < output_frames:
        raise ValueError("overlap must be in [1, clip_len - 2]")
    stride = output_frames - overlap
    final_start = frame_count - output_frames
    starts = [0]
    while starts[-1] < final_start:
        candidate = starts[-1] + stride
        if candidate >= final_start or candidate + stride > final_start:
            candidate = final_start
        if candidate == starts[-1]:
            break
        starts.append(candidate)
    windows = []
    for index, start in enumerate(starts):
        source_frames = min(clip_len, frame_count - start)
        windows.append(
            Window(
                index=index,
                start_frame=start,
                input_frames=clip_len,
                expected_output_frames=output_frames,
                source_frames=source_frames,
                padded_frames=clip_len - source_frames,
            )
        )
    return tuple(windows)


def plan_selected_windows(
    frame_count: int,
    starts: list[int] | tuple[int, ...],
    *,
    clip_len: int = 81,
) -> tuple[Window, ...]:
    """Build explicitly offset windows for long-video bridge generation."""

    if clip_len < 5 or (clip_len - 1) % 4:
        raise ValueError("clip_len must satisfy clip_len = 4n + 1")
    if not starts:
        raise ValueError("at least one selected window start is required")
    output_frames = clip_len - 1
    normalized = sorted(set(starts))
    if len(normalized) != len(starts):
        raise ValueError("selected window starts must be unique")
    if normalized[0] < 0 or normalized[-1] > frame_count - output_frames:
        raise ValueError("selected window start is outside the source timeline")
    return tuple(
        Window(
            index=index,
            start_frame=start,
            input_frames=clip_len,
            expected_output_frames=output_frames,
            source_frames=min(clip_len, frame_count - start),
            padded_frames=max(0, clip_len - (frame_count - start)),
        )
        for index, start in enumerate(normalized)
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(numerator) / int(denominator),
        "frames": int(stream["nb_frames"]),
        "duration": float(payload["format"]["duration"]),
    }


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
    }


def _packages() -> dict[str, str | None]:
    result = {}
    for package in ("torch", "torchvision", "transformers", "flash-attn"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _extract_window(
    *,
    ffmpeg: Path,
    source: Path,
    output: Path,
    window: Window,
    fps: float,
) -> list[str]:
    filters = [
        f"trim=start_frame={window.start_frame}:"
        f"end_frame={window.start_frame + window.source_frames}",
        "setpts=PTS-STARTPTS",
    ]
    if window.padded_frames:
        filters.append(
            "tpad=stop_mode=clone:"
            f"stop_duration={window.padded_frames / fps:.12f}"
        )
    command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        ",".join(filters),
        "-frames:v",
        str(window.input_frames),
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


def _parse_gpu_pairs(
    values: list[str],
) -> tuple[tuple[int, int], ...]:
    pairs = []
    for value in values:
        fields = value.split(",")
        if len(fields) != 2:
            raise ValueError("--gpu-pair must be PHYSICAL_GPU_A,PHYSICAL_GPU_B")
        try:
            pair = (int(fields[0]), int(fields[1]))
        except ValueError as error:
            raise ValueError("--gpu-pair values must be physical GPU integers") from error
        if pair[0] == pair[1]:
            raise ValueError("a GPU pair must contain two distinct physical GPUs")
        pairs.append(pair)
    flattened = [index for pair in pairs for index in pair]
    if len(flattened) != len(set(flattened)):
        raise ValueError("physical GPUs cannot be shared by concurrent GPU pairs")
    return tuple(pairs)


def _parse_reused_windows(values: list[str]) -> dict[int, Path]:
    reused: dict[int, Path] = {}
    for value in values:
        start_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError("--reuse-window-result must be START_FRAME=/path/result.mp4")
        try:
            start = int(start_text)
        except ValueError as error:
            raise ValueError("reused window start frame must be an integer") from error
        if start in reused:
            raise ValueError(f"duplicate reused window start frame: {start}")
        reused[start] = Path(path_text).expanduser().resolve()
    return reused


def partition_contiguous(
    items: list[dict[str, Any]], worker_count: int
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Split ordered windows into balanced contiguous temporal chains."""

    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if not items:
        raise ValueError("items must not be empty")
    worker_count = min(worker_count, len(items))
    base, remainder = divmod(len(items), worker_count)
    groups = []
    offset = 0
    for worker in range(worker_count):
        size = base + (1 if worker < remainder else 0)
        groups.append(tuple(items[offset : offset + size]))
        offset += size
    return tuple(groups)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"persistent batch did not write status events: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _iso_elapsed_seconds(start: str, end: str) -> float:
    return (
        datetime.fromisoformat(end) - datetime.fromisoformat(start)
    ).total_seconds()


def _throughput_metrics(
    *,
    source_frames: int,
    fps: float,
    generated_frames: int,
    generation_wall_seconds: float,
    end_to_end_wall_seconds: float,
    batch_gpu_seconds: float,
) -> dict[str, float | int]:
    if min(generation_wall_seconds, end_to_end_wall_seconds, fps) <= 0:
        raise ValueError("throughput durations and FPS must be positive")
    video_seconds = source_frames / fps
    return {
        "source_frames": source_frames,
        "generated_window_frames": generated_frames,
        "useful_video_seconds": video_seconds,
        "generation_wall_seconds": generation_wall_seconds,
        "end_to_end_wall_seconds": end_to_end_wall_seconds,
        "effective_generation_fps": source_frames / generation_wall_seconds,
        "effective_end_to_end_fps": source_frames / end_to_end_wall_seconds,
        "generated_window_fps": generated_frames / generation_wall_seconds,
        "generation_realtime_factor": generation_wall_seconds / video_seconds,
        "end_to_end_realtime_factor": end_to_end_wall_seconds / video_seconds,
        "a800_gpu_hours": batch_gpu_seconds / 3600.0,
        "a800_gpu_seconds_per_video_second": batch_gpu_seconds / video_seconds,
    }


def _covered_source_frames(windows: list[dict[str, Any]]) -> int:
    intervals = sorted(
        (
            int(item["start_frame"]),
            int(item["start_frame"]) + int(item["expected_output_frames"]),
        )
        for item in windows
    )
    if not intervals:
        return 0
    covered = 0
    start, end = intervals[0]
    for following_start, following_end in intervals[1:]:
        if following_start > end:
            covered += end - start
            start, end = following_start, following_end
        else:
            end = max(end, following_end)
    return covered + end - start


def _recovered_persistent_generation_metrics(
    metadata: dict[str, Any],
    *,
    source_frames: int,
    fps: float,
    generated_frames: int,
) -> dict[str, Any]:
    """Recover generation-only timing from completed persistent subprocesses."""

    batches = metadata.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("reuse source metadata has no persistent batches")
    if any(
        batch.get("status") != "completed" or int(batch.get("returncode", -1)) != 0
        for batch in batches
    ):
        raise ValueError("reuse source metadata contains an incomplete batch")
    generation_started_at = min(str(batch["started_at"]) for batch in batches)
    generation_completed_at = max(str(batch["completed_at"]) for batch in batches)
    generation_wall_seconds = _iso_elapsed_seconds(
        generation_started_at, generation_completed_at
    )
    batch_gpu_seconds = sum(
        _iso_elapsed_seconds(str(batch["started_at"]), str(batch["completed_at"]))
        * len(batch["physical_gpu_pair"])
        for batch in batches
    )
    useful_video_seconds = source_frames / fps
    return {
        "timing_scope": "persistent_batch_start_to_all_batches_complete",
        "generation_started_at": generation_started_at,
        "generation_completed_at": generation_completed_at,
        "source_frames": source_frames,
        "generated_window_frames": generated_frames,
        "useful_video_seconds": useful_video_seconds,
        "generation_wall_seconds": generation_wall_seconds,
        "effective_generation_fps": source_frames / generation_wall_seconds,
        "generated_window_fps": generated_frames / generation_wall_seconds,
        "generation_realtime_factor": generation_wall_seconds
        / useful_video_seconds,
        "a800_gpu_hours": batch_gpu_seconds / 3600.0,
        "a800_gpu_seconds_per_video_second": batch_gpu_seconds
        / useful_video_seconds,
    }


def _decode_frames(path: Path) -> list[Any]:
    import cv2  # Heavy dependency remains isolated to post-generation QA.

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode generated window for continuity QA: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"decoded no generated frames for continuity QA: {path}")
    return frames


def _long_horizon_continuity(
    windows: list[dict[str, Any]], *, fps: float
) -> dict[str, Any]:
    import numpy as np

    ordered = sorted(windows, key=lambda item: int(item["start_frame"]))
    overlaps = []
    for previous, following in zip(ordered, ordered[1:]):
        previous_start = int(previous["start_frame"])
        following_start = int(following["start_frame"])
        previous_frames = _decode_frames(Path(previous["result"]))
        following_frames = _decode_frames(Path(following["result"]))
        overlap_start = following_start
        overlap_end = min(
            previous_start + len(previous_frames),
            following_start + len(following_frames),
        )
        if overlap_end - overlap_start < 2:
            raise RuntimeError("adjacent generated windows lack a measurable overlap")
        same_time = []
        seams = []
        for source_camera_frame in range(overlap_start, overlap_end):
            first = previous_frames[source_camera_frame - previous_start].astype(
                np.float32
            )
            second = following_frames[source_camera_frame - following_start].astype(
                np.float32
            )
            same_time.append(float(np.mean(np.abs(first - second))))
            if source_camera_frame > overlap_start:
                before = previous_frames[
                    source_camera_frame - 1 - previous_start
                ].astype(np.float32)
                seams.append(
                    (
                        float(np.mean(np.abs(before - second))),
                        source_camera_frame,
                    )
                )
        best_seam_mad, best_seam_frame = min(seams)
        overlaps.append(
            {
                "previous_window_index": int(previous["index"]),
                "following_window_index": int(following["index"]),
                "coordinate_frame": "source_camera_frame",
                "overlap_start_frame": overlap_start,
                "overlap_end_frame_exclusive": overlap_end,
                "overlap_frames": overlap_end - overlap_start,
                "same_time_mad_mean": float(np.mean(same_time)),
                "same_time_mad_maximum": max(same_time),
                "best_seam_frame": best_seam_frame,
                "best_seam_mad": best_seam_mad,
                "cross_batch_boundary": int(previous["batch_index"])
                != int(following["batch_index"]),
                "following_reference_kind": following["reference"]["kind"],
            }
        )
    covered_frames = _covered_source_frames(ordered)
    if not overlaps:
        return {
            "covered_source_frames": covered_frames,
            "covered_video_seconds": covered_frames / fps,
            "is_20s_or_longer": covered_frames / fps >= 20.0,
            "temporal_chain_count": len(
                {int(item["batch_index"]) for item in ordered}
            ),
            "rolling_reference_links": 0,
            "overlaps": [],
            "mean_same_time_mad": None,
            "maximum_same_time_mad": None,
            "mean_best_seam_mad": None,
            "maximum_best_seam_mad": None,
        }
    return {
        "covered_source_frames": covered_frames,
        "covered_video_seconds": covered_frames / fps,
        "is_20s_or_longer": covered_frames / fps >= 20.0,
        "temporal_chain_count": len({int(item["batch_index"]) for item in ordered}),
        "rolling_reference_links": sum(
            item["reference"]["kind"] == "previous_window_camera_frame"
            for item in ordered
        ),
        "overlaps": overlaps,
        "mean_same_time_mad": float(
            np.mean([item["same_time_mad_mean"] for item in overlaps])
        ),
        "maximum_same_time_mad": max(
            item["same_time_mad_maximum"] for item in overlaps
        ),
        "mean_best_seam_mad": float(
            np.mean([item["best_seam_mad"] for item in overlaps])
        ),
        "maximum_best_seam_mad": max(item["best_seam_mad"] for item in overlaps),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument(
        "--reference-coordinate-frame",
        choices=("identity_reference", "source_camera_frame"),
        default="identity_reference",
    )
    parser.add_argument("--reference-source-camera-frame", type=int)
    parser.add_argument("--quality-anchor", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/usr/bin/ffprobe"))
    parser.add_argument("--gpu", type=int, action="append", default=[])
    parser.add_argument(
        "--gpu-pair",
        action="append",
        default=[],
        metavar="GPU_A,GPU_B",
        help="run one persistent model worker on each disjoint physical GPU pair",
    )
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=70_000)
    parser.add_argument(
        "--compile-cache-root",
        type=Path,
        help="optional reusable Triton/Inductor cache root for the pinned runtime",
    )
    parser.add_argument("--triton-cache-dir", type=Path)
    parser.add_argument("--torchinductor-cache-dir", type=Path)
    parser.add_argument(
        "--master-port-base",
        type=int,
        default=29_500,
        help="first per-batch torch.distributed TCP port",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--clip-len", type=int, default=81)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--anchor-start-frame", type=int, default=236)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--execution-mode",
        choices=("persistent", "legacy"),
        default="persistent",
        help="persistent amortizes one model load over every window assigned to a pair",
    )
    parser.add_argument(
        "--temporal-anchor-mode",
        choices=("fixed", "rolling"),
        default="rolling",
        help=(
            "rolling (default) conditions each chain window on the matching "
            "previous generated frame; fixed is the identity-reset ablation"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--window-start-frame", type=int, action="append", default=[])
    parser.add_argument(
        "--reuse-window-result",
        action="append",
        default=[],
        metavar="START_FRAME=/path/result.mp4",
    )
    parser.add_argument(
        "--reuse-source-metadata",
        type=Path,
        help=(
            "failed controller metadata whose completed batch events prove the "
            "reference lineage and timing of every reused result"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    metadata_path = experiment / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"experiment already initialized: {metadata_path}")
    experiment.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    launcher = [sys.executable, *sys.argv]
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "method": "wan_animate2_distilled_overlapping_long_video",
        "status": "preflight_running",
        "honest_status": "PARTIAL",
        "created_at": created_at,
        "launcher_command": launcher,
        "launcher_command_shell": shlex.join(launcher),
    }
    _write_json(metadata_path, record)
    active_children: list[subprocess.Popen[Any]] = []
    try:
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("Wan-Animate-2 requires the pinned Python 3.11 environment")
        if args.width % 16 or args.height % 16:
            raise ValueError("width and height must be multiples of 16")
        if args.reference_coordinate_frame == "source_camera_frame":
            if args.reference_source_camera_frame is None:
                raise ValueError(
                    "source-camera reference requires --reference-source-camera-frame"
                )
        elif args.reference_source_camera_frame is not None:
            raise ValueError(
                "--reference-source-camera-frame requires a source_camera_frame reference"
            )
        source = args.source_video.expanduser().resolve()
        reference = args.reference_image.expanduser().resolve()
        quality_anchor = args.quality_anchor.expanduser().resolve()
        prompt_file = args.prompt_file.expanduser().resolve()
        repo = args.repo.expanduser().resolve()
        checkpoint = args.checkpoint_dir.expanduser().resolve()
        python = Path(os.path.abspath(args.python.expanduser()))
        ffmpeg = args.ffmpeg.expanduser().resolve()
        ffprobe = args.ffprobe.expanduser().resolve()
        for label, path in (
            ("source video", source),
            ("reference image", reference),
            ("quality anchor", quality_anchor),
            ("prompt file", prompt_file),
            ("Python", python),
            ("FFmpeg", ffmpeg),
            ("FFprobe", ffprobe),
        ):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"{label} does not exist or is empty: {path}")
        source_info = _video_info(ffprobe, source)
        anchor_info = _video_info(ffprobe, quality_anchor)
        if abs(float(source_info["fps"]) - args.fps) > 1e-6:
            raise ValueError(f"source FPS mismatch: {source_info['fps']} vs {args.fps}")
        if int(anchor_info["frames"]) != args.clip_len - 1:
            raise ValueError("quality anchor must contain exactly clip_len - 1 frames")
        if not 0 <= args.anchor_start_frame <= int(source_info["frames"]) - int(
            anchor_info["frames"]
        ):
            raise ValueError("quality anchor interval is outside the source timeline")
        if args.window_start_frame:
            windows = plan_selected_windows(
                int(source_info["frames"]),
                args.window_start_frame,
                clip_len=args.clip_len,
            )
        else:
            windows = plan_windows(
                int(source_info["frames"]),
                clip_len=args.clip_len,
                overlap=args.overlap,
            )
        reused_windows = _parse_reused_windows(args.reuse_window_result)
        reuse_source_metadata_path = (
            args.reuse_source_metadata.expanduser().resolve()
            if args.reuse_source_metadata is not None
            else None
        )
        if reuse_source_metadata_path is not None and not reused_windows:
            raise ValueError("--reuse-source-metadata requires reused window results")
        if reuse_source_metadata_path is not None and (
            not reuse_source_metadata_path.is_file()
            or reuse_source_metadata_path.stat().st_size == 0
        ):
            raise ValueError(
                "reuse source metadata does not exist or is empty: "
                f"{reuse_source_metadata_path}"
            )
        planned_starts = {window.start_frame for window in windows}
        unexpected_reuse = sorted(set(reused_windows) - planned_starts)
        if unexpected_reuse:
            raise ValueError(
                f"reused window starts are not in the plan: {unexpected_reuse}"
            )
        source_commit = verify_wan_animate2_source(repo)
        checkpoint_hashes = verify_wan_animate2_checkpoint(checkpoint, distilled=True)
        gpus, inventory, processes = query_gpus()
        requested_pairs = _parse_gpu_pairs(args.gpu_pair)
        if requested_pairs and args.gpu:
            raise ValueError("use either --gpu or --gpu-pair, not both")
        if requested_pairs and args.execution_mode != "persistent":
            raise ValueError("multiple GPU pairs require --execution-mode persistent")
        if not 1024 <= args.master_port_base <= 65535 - max(1, len(requested_pairs)):
            raise ValueError("--master-port-base leaves the valid TCP port range")
        if requested_pairs:
            selected_pairs = tuple(
                select_wan_animate2_gpus(
                    gpus,
                    pair,
                    minimum_free_mib=args.minimum_free_gpu_mib,
                )
                for pair in requested_pairs
            )
        else:
            selected_pairs = (
                select_wan_animate2_gpus(
                    gpus,
                    args.gpu,
                    minimum_free_mib=args.minimum_free_gpu_mib,
                ),
            )
        selected = selected_pairs[0]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu.physical_index) for gpu in selected
        )
        environment["PYTHONHASHSEED"] = str(args.seed)
        environment["PYTHONPATH"] = str(repo)
        runtimes = []
        for pair in selected_pairs:
            pair_environment = environment.copy()
            pair_environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(gpu.physical_index) for gpu in pair
            )
            probe = subprocess.run(
                [
                    str(python),
                    "-c",
                    (
                        "import json, torch; print(json.dumps({"
                        "'torch':torch.__version__,'cuda':torch.version.cuda,"
                        "'available':torch.cuda.is_available(),"
                        "'devices':torch.cuda.device_count(),"
                        "'names':[torch.cuda.get_device_name(i) "
                        "for i in range(torch.cuda.device_count())]}))"
                    ),
                ],
                cwd=repo,
                env=pair_environment,
                check=True,
                capture_output=True,
                text=True,
            )
            runtime = json.loads(probe.stdout.splitlines()[-1])
            if not runtime["available"] or runtime["devices"] != 2:
                raise RuntimeError("selected GPUs did not map to exactly two CUDA devices")
            runtimes.append(runtime)

        prompt = prompt_file.read_text().strip()
        if not prompt:
            raise ValueError("prompt file is empty")
        compile_cache_contract = {
            "source_commit": source_commit,
            "model_revision": WAN_ANIMATE2_MODEL_REVISION,
            "python": str(runtimes[0].get("python", sys.version.split()[0])),
            "torch": str(runtimes[0]["torch"]),
            "cuda": str(runtimes[0]["cuda"]),
            "gpu_names": list(runtimes[0].get("names", [])),
            "width": args.width,
            "height": args.height,
            "clip_len": args.clip_len,
        }
        compile_cache_fingerprint = hashlib.sha256(
            json.dumps(
                compile_cache_contract, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()[:20]
        compile_cache_root = (
            args.compile_cache_root.expanduser().resolve()
            if args.compile_cache_root is not None
            else Path(__file__).resolve().parents[1]
            / ".cache"
            / "wan-animate2-compiled"
            / compile_cache_fingerprint
        )
        if (args.triton_cache_dir is None) != (
            args.torchinductor_cache_dir is None
        ):
            raise ValueError(
                "--triton-cache-dir and --torchinductor-cache-dir must be set together"
            )
        if args.triton_cache_dir is not None:
            triton_cache_dir = args.triton_cache_dir.expanduser().resolve()
            torchinductor_cache_dir = (
                args.torchinductor_cache_dir.expanduser().resolve()
            )
        else:
            triton_cache_dir = compile_cache_root / "triton"
            torchinductor_cache_dir = compile_cache_root / "torchinductor"
        config = experiment / "config" / "wan_animate_2_distillation.yaml"
        write_runtime_config(repo, checkpoint, config, distilled=True)
        input_windows = experiment / "input" / "windows"
        input_windows.mkdir(parents=True, exist_ok=True)
        window_records = []
        for window in windows:
            clip = input_windows / f"window-{window.index:02d}-{window.start_frame:04d}.mp4"
            extraction_command = _extract_window(
                ffmpeg=ffmpeg,
                source=source,
                output=clip,
                window=window,
                fps=float(source_info["fps"]),
            )
            clip_info = _video_info(ffprobe, clip)
            if int(clip_info["frames"]) != args.clip_len:
                raise RuntimeError(f"window {window.index} has {clip_info['frames']} frames")
            window_records.append(
                {
                    **asdict(window),
                    "input": str(clip),
                    "input_sha256": file_sha256(clip),
                    "input_info": clip_info,
                    "extraction_command": extraction_command,
                    "status": "prepared",
                }
            )

        record.update(
            {
                "status": "preflight_complete" if args.preflight_only else "running",
                "source": {
                    "path": str(source),
                    "sha256": file_sha256(source),
                    "info": source_info,
                },
                "reference": {
                    "path": str(reference),
                    "sha256": file_sha256(reference),
                    "coordinate_frame": args.reference_coordinate_frame,
                    "source_camera_frame": args.reference_source_camera_frame,
                },
                "quality_anchor": {
                    "path": str(quality_anchor),
                    "sha256": file_sha256(quality_anchor),
                    "start_frame": args.anchor_start_frame,
                    "info": anchor_info,
                },
                "prompt": prompt,
                "prompt_file": str(prompt_file),
                "prompt_sha256": file_sha256(prompt_file),
                "config": {
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "clip_len": args.clip_len,
                    "overlap": args.overlap,
                    "steps": args.steps,
                    "guidance_scale": args.guidance_scale,
                    "seed": args.seed,
                    "distilled": True,
                    "execution_mode": args.execution_mode,
                    "temporal_anchor_mode": args.temporal_anchor_mode,
                    "compile_cache": {
                        "root": str(compile_cache_root),
                        "triton_dir": str(triton_cache_dir),
                        "torchinductor_dir": str(torchinductor_cache_dir),
                        "fingerprint": compile_cache_fingerprint,
                        "contract": compile_cache_contract,
                    },
                    "selected_window_starts": (
                        [window.start_frame for window in windows]
                        if args.window_start_frame
                        else None
                    ),
                },
                "windows": window_records,
                "source_commit": source_commit,
                "model_revision": WAN_ANIMATE2_MODEL_REVISION,
                "checkpoint_revision_marker": (
                    checkpoint / ".phiagent-model-revision"
                ).read_text().strip(),
                "checkpoint_hashes": checkpoint_hashes,
                "selected_gpus": [
                    asdict(gpu) for pair in selected_pairs for gpu in pair
                ],
                "selected_gpu_pairs": [
                    [asdict(gpu) for gpu in pair] for pair in selected_pairs
                ],
                "gpu_inventory": [asdict(gpu) for gpu in gpus],
                "gpu_inventory_raw": inventory,
                "gpu_processes_raw": processes,
                "cuda_visible_devices": [
                    ",".join(str(gpu.physical_index) for gpu in pair)
                    for pair in selected_pairs
                ],
                "runtime": runtimes,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": sys.version,
                "packages": _packages(),
                "git": _git_state(Path(__file__).resolve().parents[1]),
            }
        )
        reuse_provenance: dict[int, dict[str, Any]] = {}
        recovered_generation_throughput: dict[str, Any] | None = None
        if reuse_source_metadata_path is not None:
            reuse_source_metadata = json.loads(
                reuse_source_metadata_path.read_text()
            )
            for key_path in (
                ("source", "sha256"),
                ("reference", "sha256"),
                ("quality_anchor", "sha256"),
                ("prompt_sha256",),
                ("source_commit",),
                ("model_revision",),
                ("checkpoint_hashes",),
                ("config", "clip_len"),
                ("config", "steps"),
                ("config", "guidance_scale"),
                ("config", "seed"),
                ("config", "temporal_anchor_mode"),
            ):
                current_value: Any = record
                source_value: Any = reuse_source_metadata
                for key in key_path:
                    current_value = current_value[key]
                    source_value = source_value[key]
                if current_value != source_value:
                    joined = ".".join(key_path)
                    raise ValueError(f"reuse source metadata mismatch: {joined}")
            source_windows_by_index = {
                int(item["index"]): item
                for item in reuse_source_metadata.get("windows", [])
            }
            source_batches_by_index = {
                int(item["index"]): item
                for item in reuse_source_metadata.get("batches", [])
            }
            started_events: dict[int, dict[str, Any]] = {}
            completed_events: dict[int, dict[str, Any]] = {}
            batch_by_window: dict[int, int] = {}
            for batch_index, batch in source_batches_by_index.items():
                for window_index in batch["window_indices"]:
                    batch_by_window[int(window_index)] = batch_index
                for event in _read_jsonl(Path(batch["status_file"])):
                    if event["event"] == "window_started":
                        started_events[int(event["index"])] = event
                    elif event["event"] == "window_completed":
                        completed_events[int(event["index"])] = event
            for window in windows:
                index = int(window.index)
                start = int(window.start_frame)
                if (
                    index not in source_windows_by_index
                    or index not in batch_by_window
                    or index not in started_events
                    or index not in completed_events
                ):
                    raise ValueError(
                        f"reuse source metadata lacks complete provenance for window {index}"
                    )
                source_window = source_windows_by_index[index]
                if int(source_window["start_frame"]) != start:
                    raise ValueError(
                        f"reuse source metadata start mismatch for window {index}"
                    )
                completed_event = completed_events[index]
                reused_path = reused_windows[start]
                if not reused_path.is_file() or reused_path.stat().st_size == 0:
                    raise ValueError(
                        f"reused window does not exist or is empty: {reused_path}"
                    )
                reused_sha256 = file_sha256(reused_path)
                if reused_sha256 != completed_event["result_sha256"]:
                    raise ValueError(
                        f"reused window {index} does not match its completed event"
                    )
                started_event = started_events[index]
                reference_kind = str(started_event["reference_kind"])
                reuse_provenance[start] = {
                    "batch_index": batch_by_window[index],
                    "reference": {
                        "path": started_event["reference"],
                        "kind": reference_kind,
                        "sha256": started_event["reference_sha256"],
                        "coordinate_frame": (
                            args.reference_coordinate_frame
                            if reference_kind == "canonical"
                            else "source_camera_frame"
                        ),
                        "source_camera_frame": (
                            args.reference_source_camera_frame
                            if reference_kind == "canonical"
                            else start
                        ),
                        "extraction_command": started_event.get(
                            "reference_extraction_command"
                        ),
                    },
                    "started_at": started_event["at"],
                    "completed_at": completed_event["at"],
                    "elapsed_seconds": completed_event["elapsed_seconds"],
                }
            recovered_generation_throughput = (
                _recovered_persistent_generation_metrics(
                    reuse_source_metadata,
                    source_frames=_covered_source_frames(window_records),
                    fps=float(source_info["fps"]),
                    generated_frames=sum(
                        int(item["expected_output_frames"])
                        for item in window_records
                    ),
                )
            )
            record["reuse_source_metadata"] = {
                "path": str(reuse_source_metadata_path),
                "sha256": file_sha256(reuse_source_metadata_path),
                "controller_status": reuse_source_metadata.get("status"),
                "controller_error": reuse_source_metadata.get("error"),
            }
        _write_json(metadata_path, record)
        if args.preflight_only:
            print(json.dumps({"experiment": str(experiment), "status": record["status"]}))
            return 0

        for window_record in window_records:
            start = int(window_record["start_frame"])
            if start not in reused_windows:
                continue
            reused = reused_windows[start]
            if not reused.is_file() or reused.stat().st_size == 0:
                raise ValueError(f"reused window does not exist or is empty: {reused}")
            reused_info = _video_info(ffprobe, reused)
            if int(reused_info["frames"]) != int(
                window_record["expected_output_frames"]
            ):
                raise ValueError(
                    f"reused window at {start} has {reused_info['frames']} frames"
                )
            output_dir = (
                experiment
                / "windows"
                / f"window-{int(window_record['index']):02d}-{start:04d}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            result = output_dir / "result.mp4"
            shutil.copy2(reused, result)
            window_record.update(
                {
                    "status": "completed",
                    "result": str(result),
                    "result_sha256": file_sha256(result),
                    "result_info": reused_info,
                    "reuse": {
                        "source": str(reused),
                        "source_sha256": file_sha256(reused),
                        "verified_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "batch_index": reuse_provenance.get(start, {}).get(
                        "batch_index", -1
                    ),
                    "reference": reuse_provenance.get(start, {}).get(
                        "reference",
                        {
                            "path": str(reference),
                            "kind": args.reference_coordinate_frame,
                            "sha256": file_sha256(reference),
                            "coordinate_frame": args.reference_coordinate_frame,
                            "source_camera_frame": args.reference_source_camera_frame,
                        },
                    ),
                }
            )
            for key in ("started_at", "completed_at", "elapsed_seconds"):
                if key in reuse_provenance.get(start, {}):
                    window_record[key] = reuse_provenance[start][key]
        pending_window_records = [
            item for item in window_records if item["status"] != "completed"
        ]
        if reused_windows and not pending_window_records:
            completed_at = datetime.now(timezone.utc).isoformat()
            record.update(
                {
                    "method": "wan_animate2_verified_window_reuse",
                    "status": "completed",
                    "completed_at": completed_at,
                    "completed_windows": len(window_records),
                    "generation_skipped_due_to_verified_reuse": True,
                    "throughput": None,
                    "recovered_generation_throughput": (
                        recovered_generation_throughput
                    ),
                    "long_horizon_continuity": _long_horizon_continuity(
                        window_records, fps=float(source_info["fps"])
                    ),
                }
            )
            _write_json(metadata_path, record)
            print(
                json.dumps(
                    {
                        "experiment": str(experiment),
                        "status": "completed",
                        "reused_windows": len(window_records),
                    }
                )
            )
            return 0
        if reused_windows:
            raise ValueError(
                "mixed reused and newly generated windows are not supported yet"
            )

        if args.execution_mode == "persistent":
            batch_script = (
                Path(__file__).resolve().parent
                / "run_wan_animate2_persistent_batch.py"
            )
            if not batch_script.is_file():
                raise ValueError(f"persistent batch entry point is missing: {batch_script}")
            groups = partition_contiguous(window_records, len(selected_pairs))
            active_pairs = selected_pairs[: len(groups)]
            # Validate every pair from one fresh snapshot before starting any
            # worker, so a capacity failure cannot leave a partial batch live.
            launch_gpus, launch_inventory, launch_processes = query_gpus()
            launch_pairs = tuple(
                select_wan_animate2_gpus(
                    launch_gpus,
                    [gpu.physical_index for gpu in pair],
                    minimum_free_mib=args.minimum_free_gpu_mib,
                )
                for pair in active_pairs
            )
            batches = []
            processes_by_batch: list[tuple[dict[str, Any], subprocess.Popen[Any]]] = []
            generation_started_at = datetime.now(timezone.utc).isoformat()
            generation_clock = time.perf_counter()
            for batch_index, (pair, group, current_pair) in enumerate(
                zip(active_pairs, groups, launch_pairs)
            ):
                pair_indices = [gpu.physical_index for gpu in pair]
                batch_dir = experiment / "batches" / f"batch-{batch_index:02d}"
                batch_dir.mkdir(parents=True, exist_ok=False)
                job_file = batch_dir / "jobs.json"
                status_file = batch_dir / "events.jsonl"
                jobs = []
                previous: dict[str, Any] | None = None
                for item in group:
                    index = int(item["index"])
                    start = int(item["start_frame"])
                    output_dir = (
                        experiment
                        / "windows"
                        / f"window-{index:02d}-{start:04d}"
                        / "upstream"
                    )
                    job: dict[str, Any] = {
                        "index": index,
                        "source_camera_start_frame": start,
                        "input": item["input"],
                        "output_dir": str(output_dir),
                        "canonical_reference": str(reference),
                        "width": args.width,
                        "height": args.height,
                        "fps": args.fps,
                        "clip_len": args.clip_len,
                        "guidance_scale": args.guidance_scale,
                        "steps": args.steps,
                        "seed": args.seed,
                        "prompt": prompt,
                    }
                    if previous is not None:
                        local_frame = start - int(previous["start_frame"])
                        if not 0 <= local_frame < int(previous["expected_output_frames"]):
                            raise RuntimeError(
                                "rolling temporal chain requires overlapping adjacent windows"
                            )
                        job.update(
                            {
                                "previous_window_index": int(previous["index"]),
                                "previous_window_local_frame": local_frame,
                                "continuation_source_camera_frame": start,
                            }
                        )
                    jobs.append(job)
                    previous = item
                    item.update(
                        {
                            "status": "queued",
                            "batch_index": batch_index,
                            "physical_gpu_pair": pair_indices,
                        }
                    )
                _write_json(
                    job_file,
                    {
                        "schema_version": "1.0.0",
                        "temporal_anchor_mode": args.temporal_anchor_mode,
                        "jobs": jobs,
                    },
                )
                command = [
                    str(python),
                    str(batch_script),
                    "--repo",
                    str(repo),
                    "--config",
                    str(config),
                    "--job-file",
                    str(job_file),
                    "--status-file",
                    str(status_file),
                    "--ffmpeg",
                    str(ffmpeg),
                ]
                for physical_index in pair_indices:
                    command.extend(("--physical-gpu", str(physical_index)))
                batch_environment = environment.copy()
                batch_environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(index) for index in pair_indices
                )
                batch_environment["MASTER_ADDR"] = "127.0.0.1"
                batch_environment["MASTER_PORT"] = str(
                    args.master_port_base + batch_index
                )
                batch_environment["RANK"] = "0"
                batch_environment["WORLD_SIZE"] = "1"
                port_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    port_probe.bind(
                        (
                            batch_environment["MASTER_ADDR"],
                            int(batch_environment["MASTER_PORT"]),
                        )
                    )
                except OSError as error:
                    raise RuntimeError(
                        "torch.distributed master port is unavailable: "
                        f"{batch_environment['MASTER_PORT']}"
                    ) from error
                finally:
                    port_probe.close()
                # Do not inherit an ABI-stale Triton/Inductor cache from another
                # Python or torch build. Each experiment owns its compiled code.
                batch_environment["TRITON_CACHE_DIR"] = str(
                    triton_cache_dir
                )
                batch_environment["TORCHINDUCTOR_CACHE_DIR"] = str(
                    torchinductor_cache_dir
                )
                log_path = batch_dir / "inference.log"
                log_handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    cwd=repo / "infer",
                    env=batch_environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                active_children.append(process)
                batch_record: dict[str, Any] = {
                    "index": batch_index,
                    "status": "running",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "physical_gpu_pair": [asdict(gpu) for gpu in current_pair],
                    "gpu_inventory_raw": launch_inventory,
                    "gpu_processes_raw": launch_processes,
                    "cuda_visible_devices": batch_environment["CUDA_VISIBLE_DEVICES"],
                    "master_addr": batch_environment["MASTER_ADDR"],
                    "master_port": int(batch_environment["MASTER_PORT"]),
                    "rank": int(batch_environment["RANK"]),
                    "world_size": int(batch_environment["WORLD_SIZE"]),
                    "triton_cache_dir": batch_environment["TRITON_CACHE_DIR"],
                    "torchinductor_cache_dir": batch_environment[
                        "TORCHINDUCTOR_CACHE_DIR"
                    ],
                    "window_indices": [int(item["index"]) for item in group],
                    "job_file": str(job_file),
                    "status_file": str(status_file),
                    "log": str(log_path),
                    "command": command,
                    "command_shell": shlex.join(command),
                    "pid": process.pid,
                    "_log_handle": log_handle,
                }
                batches.append(batch_record)
                processes_by_batch.append((batch_record, process))
            record["batches"] = [
                {key: value for key, value in item.items() if key != "_log_handle"}
                for item in batches
            ]
            record["generation_started_at"] = generation_started_at
            _write_json(metadata_path, record)
            print(
                json.dumps(
                    {
                        "event": "persistent_batches_started",
                        "batches": len(batches),
                        "windows": len(window_records),
                        "gpu_pairs": [
                            [gpu.physical_index for gpu in pair]
                            for pair in active_pairs
                        ],
                    }
                ),
                flush=True,
            )
            failures = []
            for batch_record, process in processes_by_batch:
                returncode = process.wait()
                active_children.remove(process)
                batch_record["_log_handle"].close()
                batch_record["returncode"] = returncode
                batch_record["completed_at"] = datetime.now(timezone.utc).isoformat()
                batch_record["status"] = "completed" if returncode == 0 else "failed"
                if returncode:
                    failures.append(batch_record)
            generation_wall_seconds = time.perf_counter() - generation_clock
            record["batches"] = [
                {key: value for key, value in item.items() if key != "_log_handle"}
                for item in batches
            ]
            if failures:
                record["status"] = "failed"
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
                _write_json(metadata_path, record)
                failed_indices = [item["index"] for item in failures]
                raise RuntimeError(f"persistent batches failed: {failed_indices}")

            events_by_window: dict[int, dict[str, dict[str, Any]]] = {}
            batch_gpu_seconds = 0.0
            for batch in batches:
                events = _read_jsonl(Path(batch["status_file"]))
                starts = {
                    int(event["index"]): event
                    for event in events
                    if event["event"] == "window_started"
                }
                completes = {
                    int(event["index"]): event
                    for event in events
                    if event["event"] == "window_completed"
                }
                events_by_window.update(
                    {
                        index: {"started": starts[index], "completed": completed}
                        for index, completed in completes.items()
                        if index in starts
                    }
                )
                batch_elapsed = _iso_elapsed_seconds(
                    str(batch["started_at"]), str(batch["completed_at"])
                )
                batch["elapsed_seconds"] = batch_elapsed
                batch["events"] = events
                batch_gpu_seconds += batch_elapsed * 2

            for window_record in window_records:
                index = int(window_record["index"])
                start = int(window_record["start_frame"])
                if index not in events_by_window:
                    raise RuntimeError(f"persistent batch omitted window {index}")
                events = events_by_window[index]
                completed_event = events["completed"]
                candidate = Path(completed_event["result"])
                output_dir = experiment / "windows" / f"window-{index:02d}-{start:04d}"
                result = output_dir / "result.mp4"
                shutil.copy2(candidate, result)
                result_info = _video_info(ffprobe, result)
                if int(result_info["frames"]) != int(
                    window_record["expected_output_frames"]
                ):
                    raise RuntimeError(
                        f"window {index} produced {result_info['frames']} frames; "
                        f"expected {window_record['expected_output_frames']}"
                    )
                window_record.update(
                    {
                        "status": "completed",
                        "started_at": events["started"]["at"],
                        "completed_at": completed_event["at"],
                        "elapsed_seconds": completed_event["elapsed_seconds"],
                        "reference": {
                            "path": events["started"]["reference"],
                            "kind": events["started"]["reference_kind"],
                            "sha256": events["started"]["reference_sha256"],
                            "coordinate_frame": "camera_pixels",
                            "source_camera_frame": start,
                            "extraction_command": events["started"].get(
                                "reference_extraction_command"
                            ),
                        },
                        "result": str(result),
                        "result_sha256": file_sha256(result),
                        "result_info": result_info,
                    }
                )
                _write_json(metadata_path, record)
                print(
                    json.dumps(
                        {
                            "event": "window_completed",
                            "index": index,
                            "start_frame": start,
                            "elapsed_seconds": completed_event["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )

            completed_at = datetime.now(timezone.utc).isoformat()
            throughput = _throughput_metrics(
                source_frames=_covered_source_frames(window_records),
                fps=float(source_info["fps"]),
                generated_frames=sum(
                    int(item["expected_output_frames"]) for item in window_records
                ),
                generation_wall_seconds=generation_wall_seconds,
                end_to_end_wall_seconds=_iso_elapsed_seconds(created_at, completed_at),
                batch_gpu_seconds=batch_gpu_seconds,
            )
            long_horizon = _long_horizon_continuity(
                window_records, fps=float(source_info["fps"])
            )
            record.update(
                {
                    "status": "completed",
                    "completed_at": completed_at,
                    "completed_windows": len(window_records),
                    "throughput": throughput,
                    "long_horizon_continuity": long_horizon,
                    "batches": [
                        {
                            key: value
                            for key, value in item.items()
                            if key != "_log_handle"
                        }
                        for item in batches
                    ],
                }
            )
            _write_json(metadata_path, record)
            print(
                json.dumps(
                    {
                        "experiment": str(experiment),
                        "status": "completed",
                        "throughput": throughput,
                    },
                    sort_keys=True,
                )
            )
            return 0

        for window_record in window_records:
            index = int(window_record["index"])
            start = int(window_record["start_frame"])
            current_gpus, current_inventory, current_processes = query_gpus()
            current_selected = select_wan_animate2_gpus(
                current_gpus,
                args.gpu,
                minimum_free_mib=args.minimum_free_gpu_mib,
            )
            current_indices = [gpu.physical_index for gpu in current_selected]
            expected_indices = [gpu.physical_index for gpu in selected]
            if current_indices != expected_indices:
                raise RuntimeError(
                    "physical GPU selection changed between preflight and window launch"
                )
            window_record["gpu_pre_window"] = {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "selected": [asdict(gpu) for gpu in current_selected],
                "inventory": [asdict(gpu) for gpu in current_gpus],
                "inventory_raw": current_inventory,
                "processes_raw": current_processes,
            }
            output_dir = experiment / "windows" / f"window-{index:02d}-{start:04d}"
            upstream = output_dir / "upstream"
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(python),
                str(repo / "infer" / "wan_animate_2_demo.py"),
                "--prompt",
                prompt,
                "--refer-img-file",
                str(reference),
                "--refer-video-file",
                str(window_record["input"]),
                "--config",
                str(config),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--fps",
                str(args.fps),
                "--clip_len",
                str(args.clip_len),
                "--sample_guide_scale",
                str(args.guidance_scale),
                "--step",
                str(args.steps),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(upstream),
            ]
            window_record.update(
                {
                    "status": "running",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "command": command,
                    "command_shell": shlex.join(command),
                }
            )
            _write_json(metadata_path, record)
            print(
                json.dumps(
                    {
                        "event": "window_started",
                        "index": index,
                        "start_frame": start,
                        "total": len(window_records),
                    }
                ),
                flush=True,
            )
            log_path = output_dir / "inference.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=repo / "infer",
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            candidates = list(upstream.glob("session_*/results.mp4"))
            window_record["completed_at"] = datetime.now(timezone.utc).isoformat()
            window_record["returncode"] = completed.returncode
            if completed.returncode != 0 or len(candidates) != 1:
                window_record["status"] = "failed"
                record["status"] = "failed"
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
                _write_json(metadata_path, record)
                raise RuntimeError(f"window {index} failed; see {log_path}")
            result = output_dir / "result.mp4"
            shutil.copy2(candidates[0], result)
            result_info = _video_info(ffprobe, result)
            if int(result_info["frames"]) != window_record["expected_output_frames"]:
                window_record["status"] = "failed"
                record["status"] = "failed"
                _write_json(metadata_path, record)
                raise RuntimeError(
                    f"window {index} produced {result_info['frames']} frames; "
                    f"expected {window_record['expected_output_frames']}"
                )
            window_record.update(
                {
                    "status": "completed",
                    "result": str(result),
                    "result_sha256": file_sha256(result),
                    "result_info": result_info,
                }
            )
            _write_json(metadata_path, record)
            print(
                json.dumps(
                    {
                        "event": "window_completed",
                        "index": index,
                        "start_frame": start,
                        "result": str(result),
                    }
                ),
                flush=True,
            )

        record.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "completed_windows": len(window_records),
            }
        )
        _write_json(metadata_path, record)
        print(json.dumps({"experiment": str(experiment), "status": "completed"}))
        return 0
    except Exception as error:
        for process in active_children:
            if process.poll() is None:
                process.terminate()
        for process in active_children:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        record.update(
            {
                "status": "failed",
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write_json(metadata_path, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
