#!/usr/bin/env python3
"""Bind an exact source frame to a generated 192-frame proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_exact_first_frame(
    *,
    raw_video: Path,
    exact_frame: Path,
    output_video: Path,
    expected_frames: int,
    fps: int,
) -> dict[str, Any]:
    """Prepend one exact input frame without deleting the raw model proposal."""

    raw_video = raw_video.expanduser().resolve()
    exact_frame = exact_frame.expanduser().resolve()
    output_video = output_video.expanduser().resolve()
    if not raw_video.is_file() or not exact_frame.is_file():
        raise FileNotFoundError("raw video and exact frame must both exist")
    if output_video.exists():
        raise FileExistsError(f"refusing to overwrite boundary output: {output_video}")
    if expected_frames < 2 or fps < 1:
        raise ValueError("boundary binding requires at least two frames and positive FPS")
    generated_frames = expected_frames - 1
    duration = expected_frames / fps
    filter_graph = (
        "[0:v]trim=start_frame=0:end_frame=1,setpts=PTS-STARTPTS[first];"
        f"[1:v]trim=start_frame=0:end_frame={generated_frames},"
        "setpts=PTS-STARTPTS[rest];"
        "[first][rest]concat=n=2:v=1:a=0,format=yuv420p[v]"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(exact_frame),
        "-i",
        str(raw_video),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-map",
        "1:a?",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-t",
        f"{duration:.9f}",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=1800,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "exact-first-frame binding failed")
    return {
        "method": "prepend_exact_source_frame_then_first_191_generated_frames",
        "command": command,
        "raw_video": str(raw_video),
        "raw_video_sha256": _sha256(raw_video),
        "bound_video": str(output_video),
        "bound_video_sha256": _sha256(output_video),
        "source_frame_sha256": _sha256(exact_frame),
        "source_frame_count": 1,
        "generated_frame_count": generated_frames,
        "thresholds_unchanged": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-video", type=Path, required=True)
    parser.add_argument("--exact-frame", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=192)
    parser.add_argument("--fps", type=int, default=24)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = bind_exact_first_frame(
        raw_video=args.raw_video,
        exact_frame=args.exact_frame,
        output_video=args.output_video,
        expected_frames=args.expected_frames,
        fps=args.fps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
