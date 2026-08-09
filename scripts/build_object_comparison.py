#!/usr/bin/env python3
"""Build source | PhiZero | improved video and start/middle/end frame artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _duration(ffprobe: str, video: Path) -> float:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    duration = float(json.loads(completed.stdout)["format"]["duration"])
    if duration <= 0:
        raise ValueError(f"video has invalid duration: {video}")
    return duration


def _filter() -> str:
    labels = ("source", "PhiZero", "improved")
    streams = []
    for index, label in enumerate(labels):
        streams.append(
            f"[{index}:v]scale=448:256:flags=lanczos,"
            "drawbox=x=0:y=0:w=iw:h=34:color=black@0.65:t=fill,"
            f"drawtext=text='{label}':x=(w-text_w)/2:y=8:fontsize=20:fontcolor=white"
            f"[v{index}]"
        )
    return ";".join(streams) + ";[v0][v1][v2]hstack=inputs=3[out]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--improved", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--postprocess-description", required=True)
    args = parser.parse_args()
    videos = (args.source.resolve(), args.reference.resolve(), args.improved.resolve())
    for video in videos:
        if not video.is_file():
            raise ValueError(f"comparison input does not exist: {video}")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for comparison artifacts")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    filter_graph = _filter()
    comparison = output_dir / "source-phizero-improved.mp4"
    _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            *[item for video in videos for item in ("-i", str(video))],
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-r",
            str(args.fps),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(comparison),
        ]
    )
    duration = min(_duration(ffprobe, video) for video in videos)
    timestamps = {
        "start": 0.0,
        "middle": duration / 2,
        "end": max(0.0, duration - max(0.15, 1 / args.fps)),
    }
    for name, timestamp in timestamps.items():
        _run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                *[
                    item
                    for video in videos
                    for item in ("-ss", f"{timestamp:.6f}", "-i", str(video))
                ],
                "-filter_complex",
                filter_graph,
                "-map",
                "[out]",
                "-frames:v",
                "1",
                str(output_dir / f"{name}.png"),
            ]
        )
        keyframe = output_dir / f"{name}.png"
        if not keyframe.is_file() or keyframe.stat().st_size == 0:
            raise RuntimeError(f"comparison keyframe was not created: {keyframe}")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(videos[0]),
                "phizero_reference": str(videos[1]),
                "improved": str(videos[2]),
                "comparison": str(comparison),
                "keyframes": {
                    name: str(output_dir / f"{name}.png") for name in timestamps
                },
                "timestamps_seconds": timestamps,
                "postprocess_description": args.postprocess_description,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
