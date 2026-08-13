#!/usr/bin/env python3
"""Build a provenance-carrying real/human versus robot comparison video."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _probe(ffprobe: Path, path: Path) -> dict[str, float | int]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
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
    numerator, denominator = (
        int(value) for value in str(stream["avg_frame_rate"]).split("/", 1)
    )
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "fps": numerator / denominator,
        "duration": float(payload["format"]["duration"]),
    }


def comparison_filter(*, panel_width: int, panel_height: int) -> str:
    """Return the deterministic side-by-side layout filter."""

    return (
        f"[0:v]scale={panel_width}:{panel_height}:flags=lanczos,setsar=1[left];"
        f"[1:v]scale={panel_width}:{panel_height}:flags=lanczos,setsar=1[right];"
        "[left][right]hstack=inputs=2[body];[2:v][body]vstack=inputs=2[out]"
    )


def _render_header(path: Path, *, width: int, height: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), (8, 14, 22))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    labels = (("REAL HUMAN REFERENCE", width // 4), ("PHIAGENT ROBOT RESULT", 3 * width // 4))
    for label, center in labels:
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text((center - text_width // 2, 9), label, fill=(235, 242, 248), font=font)
    draw.line((width // 2, 0, width // 2, height), fill=(57, 74, 89), width=1)
    image.save(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--robot-video", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("ffprobe"))
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--header-height", type=int, default=48)
    parser.add_argument("--poster-frame", type=int, default=572)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    source = args.source_video.expanduser().resolve()
    robot = args.robot_video.expanduser().resolve()
    audit_path = args.audit_report.expanduser().resolve()
    for path in (source, robot, audit_path, args.ffmpeg, args.ffprobe):
        if not path.is_file():
            raise ValueError(f"required input is missing: {path}")
    source_info = _probe(args.ffprobe, source)
    robot_info = _probe(args.ffprobe, robot)
    for name, info in (("source", source_info), ("robot", robot_info)):
        if info["frames"] != args.expected_frames or abs(info["fps"] - args.fps) > 1e-6:
            raise ValueError(f"{name} video does not match the declared timeline")

    audit = json.loads(audit_path.read_text())
    candidates = audit.get("candidates", [])
    if len(candidates) != 1:
        raise ValueError("audit report must contain exactly one comparison candidate")
    persistent = candidates[0]["summary"].get("persistent_grasp")
    if not persistent:
        raise ValueError("audit report has no persistent-grasp result")

    header = output_dir / "comparison-header.png"
    video = output_dir / "real-vs-robot-persistent-grasp-27p5s.mp4"
    poster = output_dir / "real-vs-robot-persistent-grasp-poster.jpg"
    _render_header(
        header,
        width=args.panel_width * 2,
        height=args.header_height,
    )
    filter_value = comparison_filter(
        panel_width=args.panel_width,
        panel_height=args.panel_height,
    )
    encode_command = [
        str(args.ffmpeg), "-y", "-v", "error",
        "-i", str(source), "-i", str(robot),
        "-loop", "1", "-framerate", f"{args.fps:.8f}", "-i", str(header),
        "-filter_complex", filter_value,
        "-map", "[out]", "-frames:v", str(args.expected_frames),
        "-r", f"{args.fps:.8f}", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ]
    subprocess.run(encode_command, check=True)
    poster_command = [
        str(args.ffmpeg), "-y", "-v", "error", "-ss",
        f"{args.poster_frame / args.fps:.8f}", "-i", str(video),
        "-frames:v", "1", "-q:v", "2", str(poster),
    ]
    subprocess.run(poster_command, check=True)
    video_info = _probe(args.ffprobe, video)
    if video_info["frames"] != args.expected_frames:
        raise RuntimeError("comparison output is not the complete timeline")

    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "honest_status": (
            "PARTIAL: source-visible object trajectory and the 2-D occlusion-aware "
            "persistent-grasp contract pass; no metric depth or force-closure claim."
        ),
        "inputs": {
            "source": {"path": _display_path(source), "sha256": _sha256(source), "video": source_info},
            "robot": {"path": _display_path(robot), "sha256": _sha256(robot), "video": robot_info},
            "audit": {"path": _display_path(audit_path), "sha256": _sha256(audit_path)},
        },
        "persistent_grasp": persistent,
        "commands": {"encode": encode_command, "poster": poster_command},
        "outputs": {
            "video": {"path": _display_path(video), "sha256": _sha256(video), "video": video_info},
            "poster": {"path": _display_path(poster), "sha256": _sha256(poster)},
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "limitations": [
            "The grasp gate is a camera-frame visual invariant, not 3-D contact evidence.",
            "The robot remains a generated visual replacement rather than a verified real-robot execution.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"video": str(video), "poster": str(poster), "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
