#!/usr/bin/env python3
"""Prepare a full source video for one uninterrupted JoyAI causal session."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.joyai_video_edit import (  # noqa: E402
    causal_padded_frame_count,
    causal_tail_padding_frames,
    sha256_file,
    write_json,
)
from scripts.prepare_joyai_flower_windows import probe_video  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-frames", type=int, default=660)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--source-width", type=int, default=1280)
    parser.add_argument("--source-height", type=int, default=720)
    parser.add_argument("--model-width", type=int, default=1248)
    parser.add_argument("--model-height", type=int, default=720)
    parser.add_argument("--crop-left", type=int, default=16)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path(shutil.which("ffprobe") or "ffprobe"))
    return parser


def build_prepare_command(
    *,
    ffmpeg: Path,
    source: Path,
    output: Path,
    source_frames: int,
    model_width: int,
    model_height: int,
    crop_left: int,
    crop_top: int,
    chunk_frames: int = 8,
) -> list[str]:
    padding = causal_tail_padding_frames(source_frames, chunk_frames=chunk_frames)
    filters = [f"crop={model_width}:{model_height}:{crop_left}:{crop_top}"]
    if padding:
        filters.append(f"tpad=stop_mode=clone:stop={padding}")
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        ",".join(filters),
        "-frames:v",
        str(causal_padded_frame_count(source_frames, chunk_frames=chunk_frames)),
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "bgr0",
        str(output),
    ]


def _git_state() -> dict[str, Any]:
    state: dict[str, Any] = {}
    for label, command in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        )
        state[label] = {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    return state


def main() -> int:
    args = _parser().parse_args()
    source = args.source_video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    ffprobe = args.ffprobe.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"experiment already exists: {output_dir}")
    for path in (source, ffmpeg, ffprobe):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.crop_left < 0 or args.crop_top < 0:
        raise ValueError("crop offsets must be non-negative")
    if args.crop_left + args.model_width > args.source_width:
        raise ValueError("model crop exceeds source width")
    if args.crop_top + args.model_height > args.source_height:
        raise ValueError("model crop exceeds source height")

    source_probe = probe_video(ffprobe, source)
    observed = (
        source_probe["width"],
        source_probe["height"],
        source_probe["frames"],
    )
    expected = (args.source_width, args.source_height, args.expected_source_frames)
    if observed != expected:
        raise ValueError(f"source geometry/timeline {observed} does not match {expected}")

    padded_frames = causal_padded_frame_count(
        args.expected_source_frames, chunk_frames=args.chunk_frames
    )
    tail_padding = padded_frames - args.expected_source_frames
    output_dir.mkdir(parents=True)
    logs = output_dir / "logs"
    logs.mkdir()
    prepared = output_dir / "joyai-full-stream-input-ffv1.mkv"
    command = build_prepare_command(
        ffmpeg=ffmpeg,
        source=source,
        output=prepared,
        source_frames=args.expected_source_frames,
        model_width=args.model_width,
        model_height=args.model_height,
        crop_left=args.crop_left,
        crop_top=args.crop_top,
        chunk_frames=args.chunk_frames,
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    (logs / "prepare.log").write_text(
        "$ " + shlex.join(command) + "\n" + result.stdout + result.stderr,
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError("full-stream preparation failed; inspect logs/prepare.log")
    prepared_probe = probe_video(ffprobe, prepared)
    prepared_geometry = (
        prepared_probe["width"],
        prepared_probe["height"],
        prepared_probe["frames"],
    )
    expected_prepared = (args.model_width, args.model_height, padded_frames)
    if prepared_geometry != expected_prepared:
        raise RuntimeError(
            f"prepared geometry/timeline {prepared_geometry} does not match {expected_prepared}"
        )

    manifest = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "stage": "joyai_full_stream_preparation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "git": _git_state(),
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "probe": source_probe,
            "coordinate_frame": f"camera:source_native_{args.source_width}x{args.source_height}",
        },
        "output": {
            "path": str(prepared),
            "sha256": sha256_file(prepared),
            "probe": prepared_probe,
            "coordinate_frame": f"camera:joyai_center_crop_{args.model_width}x{args.model_height}",
        },
        "causal_contract": {
            "source_frames": args.expected_source_frames,
            "chunk_frames": args.chunk_frames,
            "padded_frames": padded_frames,
            "cloned_tail_frames": tail_padding,
            "source_duration_seconds": args.expected_source_frames / args.fps,
            "model_duration_seconds": padded_frames / args.fps,
            "trim_generated_tail_after_inference": tail_padding,
        },
        "coordinate_transform": {
            "kind": "integer_center_crop_no_rescale",
            "x_joyai": f"x_source - {args.crop_left}",
            "y_joyai": f"y_source - {args.crop_top}",
            "crop_left_px": args.crop_left,
            "crop_top_px": args.crop_top,
        },
        "ffmpeg_command": command,
        "model_authority": "input_preparation_only",
        "physical_evidence": False,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
