#!/usr/bin/env python3
"""Trim JoyAI tail padding and restore only the uncropped source border."""

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

from phiagent.rendering.joyai_video_edit import sha256_file, write_json  # noqa: E402
from scripts.prepare_joyai_flower_windows import probe_video  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--joyai-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--source-width", type=int, default=1280)
    parser.add_argument("--source-height", type=int, default=720)
    parser.add_argument("--model-width", type=int, default=1248)
    parser.add_argument("--model-height", type=int, default=720)
    parser.add_argument("--crop-left", type=int, default=16)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path(shutil.which("ffprobe") or "ffprobe"))
    return parser


def build_finalize_filter(*, expected_frames: int, crop_left: int, crop_top: int) -> str:
    return (
        f"[0:v]trim=start_frame=0:end_frame={expected_frames},setpts=PTS-STARTPTS[base];"
        f"[1:v]trim=start_frame=0:end_frame={expected_frames},setpts=PTS-STARTPTS[edit];"
        f"[base][edit]overlay=x={crop_left}:y={crop_top}:shortest=0:"
        "eof_action=repeat[out]"
    )


def build_finalize_command(
    *,
    ffmpeg: Path,
    source: Path,
    joyai: Path,
    output: Path,
    expected_frames: int,
    fps: int,
    crop_left: int,
    crop_top: int,
    lossless: bool,
) -> list[str]:
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-i", str(joyai), "-filter_complex",
        build_finalize_filter(
            expected_frames=expected_frames, crop_left=crop_left, crop_top=crop_top
        ),
        "-map", "[out]", "-frames:v", str(expected_frames), "-r", str(fps), "-an",
    ]
    if lossless:
        command.extend(["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0"])
    else:
        command.extend(
            ["-c:v", "libx264", "-crf", "8", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        )
    command.append(str(output))
    return command


def _git_state() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, command in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        )
        result[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    return result


def main() -> int:
    args = _parser().parse_args()
    source = args.source_video.expanduser().resolve()
    joyai = args.joyai_video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    ffprobe = args.ffprobe.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"experiment already exists: {output_dir}")
    for path in (source, joyai, ffmpeg, ffprobe):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.crop_left + args.model_width > args.source_width:
        raise ValueError("JoyAI overlay exceeds source width")
    if args.crop_top + args.model_height > args.source_height:
        raise ValueError("JoyAI overlay exceeds source height")

    source_probe = probe_video(ffprobe, source)
    joyai_probe = probe_video(ffprobe, joyai)
    if (source_probe["width"], source_probe["height"], source_probe["frames"]) != (
        args.source_width, args.source_height, args.expected_frames
    ):
        raise ValueError("source video does not match the declared native timeline")
    if (joyai_probe["width"], joyai_probe["height"]) != (
        args.model_width, args.model_height
    ) or joyai_probe["frames"] < args.expected_frames:
        raise ValueError("JoyAI video does not cover the declared full timeline")

    output_dir.mkdir(parents=True)
    logs = output_dir / "logs"
    logs.mkdir()
    outputs = {
        "lossless": output_dir / "joyai-full-27p5s-lossless.mkv",
        "review": output_dir / "joyai-full-27p5s-720p.mp4",
    }
    commands = {
        name: build_finalize_command(
            ffmpeg=ffmpeg,
            source=source,
            joyai=joyai,
            output=path,
            expected_frames=args.expected_frames,
            fps=args.fps,
            crop_left=args.crop_left,
            crop_top=args.crop_top,
            lossless=name == "lossless",
        )
        for name, path in outputs.items()
    }
    for name, command in commands.items():
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        (logs / f"{name}.log").write_text(
            "$ " + shlex.join(command) + "\n" + completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode:
            raise RuntimeError(f"{name} finalization failed; inspect logs/{name}.log")

    records = {}
    for name, path in outputs.items():
        probe = probe_video(ffprobe, path)
        if (probe["width"], probe["height"], probe["frames"]) != (
            args.source_width, args.source_height, args.expected_frames
        ):
            raise RuntimeError(f"final {name} geometry/timeline is invalid: {probe}")
        records[name] = {"path": str(path), "sha256": sha256_file(path), "probe": probe}
    manifest = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "stage": "joyai_full_stream_finalized_pending_quality_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "git": _git_state(),
        "inputs": {
            "source": {"path": str(source), "sha256": sha256_file(source), "probe": source_probe},
            "joyai": {"path": str(joyai), "sha256": sha256_file(joyai), "probe": joyai_probe},
        },
        "outputs": records,
        "coordinate_transform": {
            "kind": "inverse_integer_center_crop_no_rescale",
            "x_source": f"x_joyai + {args.crop_left}",
            "y_source": f"y_joyai + {args.crop_top}",
            "generated_area": [args.crop_left, args.crop_top, args.model_width, args.model_height],
            "source_only_area": "uncropped border only",
            "flower_motion_locked": False,
        },
        "timeline": {
            "frames": args.expected_frames,
            "fps": args.fps,
            "seconds": args.expected_frames / args.fps,
            "generated_tail_padding_trimmed": joyai_probe["frames"] - args.expected_frames,
            "interpolation": "none",
        },
        "ffmpeg_commands": commands,
        "model_authority": "full_timeline_visual_proposal",
        "physical_evidence": False,
        "promotion_status": "NOT_EVALUATED",
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
