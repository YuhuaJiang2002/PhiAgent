#!/usr/bin/env python3
"""Retimestamp an immutable Wan 30-FPS raw window to the 89-frame 24-FPS source timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="raw-retimed-89f-24fps.mp4")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--target-frames", type=int, default=89)
    parser.add_argument("--crf", type=int, default=0)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,r_frame_rate,width,height,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"expected one video stream: {path}")
    return streams[0]


def _git_state(project_root: Path) -> dict[str, str | None]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args], cwd=project_root, capture_output=True, text=True
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "head": run("rev-parse", "HEAD"),
        "status_porcelain": run("status", "--porcelain"),
    }


def build_filter(source_fps: float, target_fps: float) -> str:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source and target FPS must be positive")
    timing_scale = source_fps / target_fps
    terminal_duration = 1.0 / target_fps
    return (
        f"setpts={timing_scale:.12g}*PTS,"
        f"tpad=stop_mode=clone:stop_duration={terminal_duration:.12g},"
        f"fps={target_fps:.12g}"
    )


def main() -> int:
    args = _parser().parse_args()
    source = args.input_video.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith(".mp4"):
        raise ValueError("--output-name must be a plain .mp4 filename")
    if args.target_frames < 2 or not 0 <= args.crf <= 51:
        raise ValueError("target frames or CRF is invalid")

    source_probe = _probe(source)
    source_rate = source_probe["r_frame_rate"]
    expected_rate = f"{int(args.source_fps)}/1"
    if source_rate != expected_rate:
        raise ValueError(f"expected source FPS {expected_rate}, received {source_rate}")
    output.mkdir(parents=True)
    output_video = output / args.output_name
    filter_graph = build_filter(args.source_fps, args.target_fps)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        filter_graph,
        "-frames:v",
        str(args.target_frames),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    log = output / "ffmpeg.log"
    with log.open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    result_probe = _probe(output_video)
    expected_target_rate = f"{int(args.target_fps)}/1"
    if (
        int(result_probe["nb_read_frames"]) != args.target_frames
        or result_probe["r_frame_rate"] != expected_target_rate
    ):
        raise RuntimeError(f"retimed output failed its timeline contract: {result_probe}")
    project_root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING",
        "method": "wan_raw_pts_retime_and_terminal_frame_clone",
        "command": [sys.executable, *sys.argv],
        "ffmpeg_command": command,
        "filter_graph": filter_graph,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": _git_state(project_root),
        "inputs": {
            "video": {
                "path": str(source),
                "sha256": _sha256(source),
                "probe": source_probe,
            }
        },
        "outputs": {
            "video": {
                "path": str(output_video),
                "sha256": _sha256(output_video),
                "probe": result_probe,
            },
            "log": {"path": str(log), "sha256": _sha256(log)},
        },
        "limitations": [
            "The missing terminal frame is an exact clone of the final generated frame.",
            "This operation changes timestamps and encoding only; it does not repair geometry, identity, contact, or temporal artifacts.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output_dir": str(output), "status": "WORKING"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
