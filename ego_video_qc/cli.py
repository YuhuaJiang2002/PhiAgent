"""Command line interface for local ego-video preflight checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .metrics import QCThresholds, VideoMetadata, build_checks, evaluate_samples, parse_fraction, report_dict


def _tool_path(explicit: str | None, name: str) -> str:
    path = explicit or shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} is required; install FFmpeg or pass --{name}")
    return path


def probe(path: Path, ffprobe: str) -> VideoMetadata:
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError("input has no video stream")
    stream = streams[0]
    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(stream.get("r_frame_rate"))
    duration = parse_fraction(stream.get("duration"))
    if fps is None or duration is None or fps <= 0 or duration <= 0:
        raise ValueError("ffprobe did not provide a positive FPS and duration")
    raw_frames = stream.get("nb_frames")
    frame_count = int(raw_frames) if raw_frames not in (None, "N/A") else None
    return VideoMetadata(
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        fps=fps,
        duration_seconds=duration,
        frame_count=frame_count,
        codec=stream.get("codec_name"),
    )


def sample(path: Path, ffmpeg: str, sample_fps: float, max_seconds: float) -> tuple[bytes, ...]:
    if sample_fps <= 0 or max_seconds <= 0:
        raise ValueError("sample FPS and maximum seconds must be positive")
    command = [
        ffmpeg, "-v", "error", "-i", str(path), "-t", str(max_seconds),
        "-vf", f"fps={sample_fps},scale=64:64:flags=area,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    frame_size = 64 * 64
    if not result.stdout or len(result.stdout) % frame_size:
        raise ValueError("ffmpeg returned no complete grayscale frames")
    return tuple(result.stdout[offset : offset + frame_size] for offset in range(0, len(result.stdout), frame_size))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run conservative ego-video quality preflight gates")
    parser.add_argument("video", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ffprobe")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--max-sample-seconds", type=float, default=30.0)
    parser.add_argument("--min-width", type=int, default=320)
    parser.add_argument("--min-height", type=int, default=240)
    parser.add_argument("--min-fps", type=float, default=10.0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=600.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        video = args.video.expanduser().resolve()
        if not video.is_file():
            raise ValueError(f"video does not exist: {video}")
        ffprobe = _tool_path(args.ffprobe, "ffprobe")
        ffmpeg = _tool_path(args.ffmpeg, "ffmpeg")
        metadata = probe(video, ffprobe)
        frames = sample(video, ffmpeg, args.sample_fps, min(args.max_sample_seconds, metadata.duration_seconds))
        samples = evaluate_samples(frames)
        thresholds = QCThresholds(
            min_width=args.min_width, min_height=args.min_height, min_fps=args.min_fps,
            min_duration_seconds=args.min_duration, max_duration_seconds=args.max_duration,
        )
        payload = report_dict(metadata, samples, build_checks(metadata, samples, thresholds))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"ego-video-qc: error: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.expanduser().resolve().write_text(rendered + "\n", encoding="utf-8")
    if args.as_json:
        print(rendered)
    else:
        print(f"ego-video-qc: {'PASS' if payload['passed'] else 'FAIL'}")
        for check in payload["checks"]:
            print(f"  {'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['observed']}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
