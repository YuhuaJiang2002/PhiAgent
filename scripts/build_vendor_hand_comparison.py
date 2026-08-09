#!/usr/bin/env python3
"""Build a matched 2x2 apple-grasp comparison across robot-hand vendors."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


LABELS = ("Human source", "Sharpa reference", "Wonik Allegro", "Shadow Robot")
PHASES = (
    ("APPROACH", 0.0, 0.75),
    ("GRASP", 0.75, 1.75),
    ("LIFT/HOLD", 1.75, None),
)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe(ffprobe: str, video: Path) -> dict[str, object]:
    completed = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames:format=duration",
            "-of",
            "json",
            str(video),
        ]
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    duration = float(payload["format"]["duration"])
    if duration <= 0:
        raise ValueError(f"video has invalid duration: {video}")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": str(stream["r_frame_rate"]),
        "frames": int(stream["nb_read_frames"]),
        "duration_s": duration,
    }


def _drawtext(text: str, y: int, enable: str | None = None) -> str:
    escaped = text.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")
    result = (
        f"drawtext=text='{escaped}':x=(w-text_w)/2:y={y}:"
        "fontsize=24:fontcolor=white"
    )
    if enable is not None:
        result += f":enable='{enable}'"
    return result


def _filter_graph(duration_s: float, fps: int, vendor_label_suffix: str) -> str:
    cells = []
    labels = (*LABELS[:2], *(f"{label}{vendor_label_suffix}" for label in LABELS[2:]))
    for index, label in enumerate(labels):
        cells.append(
            f"[{index}:v]fps={fps},"
            "scale=640:360:force_original_aspect_ratio=decrease:flags=lanczos,"
            "pad=640:360:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
            "drawbox=x=0:y=0:w=iw:h=42:color=black@0.68:t=fill,"
            f"{_drawtext(label, 9)}[v{index}]"
        )
    phases = []
    for label, start, declared_end in PHASES:
        end = duration_s if declared_end is None else min(declared_end, duration_s)
        if end > start:
            phases.append(_drawtext(f"EPL: {label}", 678, f"between(t,{start},{end})"))
    return ";".join(
        [
            *cells,
            "[v0][v1]hstack=inputs=2[top]",
            "[v2][v3]hstack=inputs=2[bottom]",
            "[top][bottom]vstack=inputs=2[grid]",
            "[grid]drawbox=x=0:y=664:w=iw:h=56:color=black@0.72:t=fill,"
            + ",".join(phases)
            + "[out]",
        ]
    )


def _git_state(root: Path) -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "head": head.stdout.strip() if head.returncode == 0 else "UNBORN",
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sharpa", type=Path, required=True)
    parser.add_argument("--allegro", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--vendor-label-suffix", default="")
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    videos = tuple(
        path.expanduser().resolve()
        for path in (args.source, args.sharpa, args.allegro, args.shadow)
    )
    for video in videos:
        if not video.is_file():
            raise ValueError(f"comparison input does not exist: {video}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    input_probes = [_probe(ffprobe, video) for video in videos]
    duration_s = min(float(probe["duration_s"]) for probe in input_probes)
    output = output_dir / "vendor-hand-apple-comparison.mp4"
    _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            *[item for video in videos for item in ("-i", str(video))],
            "-filter_complex",
            _filter_graph(duration_s, args.fps, args.vendor_label_suffix),
            "-map",
            "[out]",
            "-t",
            f"{duration_s:.6f}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )
    keyframe_times = {
        "approach": min(0.25, duration_s / 4),
        "grasp": min(1.25, duration_s / 2),
        "lift_hold": max(0.0, duration_s - 0.25),
    }
    for name, timestamp_s in keyframe_times.items():
        _run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-ss",
                f"{timestamp_s:.6f}",
                "-i",
                str(output),
                "-frames:v",
                "1",
                str(output_dir / f"{name}.jpg"),
            ]
        )

    root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": "1.0.0",
        "method": "matched_vendor_hand_visual_transfer_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "git": _git_state(root),
        "inputs": {
            label: {
                "path": str(path),
                "sha256": _sha256(path),
                "video": probe,
            }
            for label, path, probe in zip(LABELS, videos, input_probes)
        },
        "epl_phases": [
            {
                "phase": label.lower().replace("/", "_"),
                "start_s": start,
                "end_s": duration_s if end is None else min(end, duration_s),
            }
            for label, start, end in PHASES
            if start < duration_s
        ],
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
            "video": _probe(ffprobe, output),
        },
        "keyframes": {
            name: {
                "path": str(output_dir / f"{name}.jpg"),
                "sha256": _sha256(output_dir / f"{name}.jpg"),
                "timestamp_s": timestamp_s,
            }
            for name, timestamp_s in keyframe_times.items()
        },
        "limitations": [
            "Sharpa is the pinned official PhiZero reference; Allegro and Shadow are Wan proxy outputs.",
            "The displayed EPL phases are coarse manual visual annotations, not released PhiZero tokens.",
            "This visual comparison is not physical robot execution or evidence of contact correctness.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest["output"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
