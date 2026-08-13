#!/usr/bin/env python3
"""Prepare one exact, licensed EPIC-KITCHENS egocentric task interval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":", maxsplit=2)
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def intersecting_annotations(
    rows: Sequence[Mapping[str, str]],
    *,
    video_id: str,
    start_s: float,
    end_s: float,
) -> list[dict[str, str]]:
    """Return stable annotation evidence intersecting a half-open time range."""

    selected = []
    for row in rows:
        if row.get("video_id") != video_id:
            continue
        row_start = _timestamp_seconds(str(row["start_timestamp"]))
        row_end = _timestamp_seconds(str(row["stop_timestamp"]))
        if row_start < end_s and row_end > start_s:
            selected.append(dict(row))
    return sorted(selected, key=lambda item: _timestamp_seconds(item["start_timestamp"]))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_revision(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _video_info(ffprobe: Path, video: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames",
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
    return {"stream": payload["streams"][0], "duration": payload["format"]["duration"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--annotations-repo", type=Path, required=True)
    parser.add_argument("--download-scripts-repo", type=Path, required=True)
    parser.add_argument("--download-report", type=Path)
    parser.add_argument("--video-id", default="P03_28")
    parser.add_argument("--start-seconds", type=float, default=24.83)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/opt/homebrew/bin/ffprobe"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"EPIC-KITCHENS preparation already exists: {manifest_path}")
    paths = {
        "source_video": args.source_video.expanduser().resolve(),
        "annotations": args.annotations.expanduser().resolve(),
        "license": args.license.expanduser().resolve(),
        "annotations_repo": args.annotations_repo.expanduser().resolve(),
        "download_scripts_repo": args.download_scripts_repo.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
        "ffprobe": args.ffprobe.expanduser().resolve(),
    }
    for label in ("source_video", "annotations", "license", "ffmpeg", "ffprobe"):
        path = paths[label]
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    for label in ("annotations_repo", "download_scripts_repo"):
        if not paths[label].is_dir():
            raise ValueError(f"{label} is not a directory: {paths[label]}")
    if args.start_seconds < 0 or args.duration_seconds <= 0:
        raise ValueError("source interval must have positive duration and nonnegative start")
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("output geometry must be positive")
    expected_frames = round(args.duration_seconds * args.fps)
    if abs(expected_frames / args.fps - args.duration_seconds) > 1e-9:
        raise ValueError("duration must map to an integer frame count")

    with paths["annotations"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    annotation_evidence = intersecting_annotations(
        rows,
        video_id=args.video_id,
        start_s=args.start_seconds,
        end_s=args.start_seconds + args.duration_seconds,
    )
    if not annotation_evidence:
        raise ValueError("selected interval has no official action annotations")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "input"
    provenance_dir = output_dir / "provenance"
    input_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    output = input_dir / "ego-source-240.mp4"
    command = [
        str(paths["ffmpeg"]),
        "-y",
        "-v",
        "error",
        "-i",
        str(paths["source_video"]),
        "-vf",
        (
            f"trim=start={args.start_seconds:.9f}:"
            f"end={args.start_seconds + args.duration_seconds:.9f},"
            f"setpts=PTS-STARTPTS,fps={args.fps},"
            f"scale={args.width}:{args.height}:force_original_aspect_ratio=increase,"
            f"crop={args.width}:{args.height}"
        ),
        "-frames:v",
        str(expected_frames),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "12",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    info = _video_info(paths["ffprobe"], output)
    stream = info["stream"]
    numerator, denominator = str(stream["r_frame_rate"]).split("/", maxsplit=1)
    if (
        int(stream["nb_frames"]) != expected_frames
        or int(numerator) / int(denominator) != args.fps
        or int(stream["width"]) != args.width
        or int(stream["height"]) != args.height
    ):
        raise RuntimeError(f"prepared Ego clip failed exact geometry: {info}")

    anchors = []
    for frame in (0, expected_frames // 2, expected_frames - 1):
        anchor = input_dir / f"frame-{frame:03d}.png"
        anchor_command = [
            str(paths["ffmpeg"]),
            "-y",
            "-v",
            "error",
            "-i",
            str(output),
            "-vf",
            f"select=eq(n\\,{frame})",
            "-frames:v",
            "1",
            str(anchor),
        ]
        subprocess.run(anchor_command, check=True)
        anchors.append(
            {"frame": frame, "path": str(anchor), "sha256": _sha256(anchor)}
        )

    copied_license = provenance_dir / "EPIC-KITCHENS-license.txt"
    shutil.copy2(paths["license"], copied_license)
    copied_report = None
    if args.download_report is not None:
        report = args.download_report.expanduser().resolve()
        if not report.is_file() or report.stat().st_size == 0:
            raise ValueError(f"download report is missing or empty: {report}")
        copied_report = provenance_dir / "download-report.csv"
        shutil.copy2(report, copied_report)

    manifest = {
        "schema_version": "1.0.0",
        "method": "official_epic_kitchens_video_to_exact_ego_task_interval",
        "status": "completed",
        "honest_status": "WORKING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": {"used": False, "reason": "deterministic licensed data preparation"},
        "dataset": {
            "name": "EPIC-KITCHENS-100",
            "video_id": args.video_id,
            "participant_id": args.video_id.split("_", maxsplit=1)[0],
            "source_url": (
                "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/"
                f"videos/train/{args.video_id.split('_', maxsplit=1)[0]}/"
                f"{args.video_id}.MP4"
            ),
            "license": "CC BY-NC 4.0",
            "license_path": str(copied_license),
            "license_sha256": _sha256(copied_license),
            "annotations_revision": _git_revision(paths["annotations_repo"]),
            "download_scripts_revision": _git_revision(paths["download_scripts_repo"]),
        },
        "source": {
            "path": str(paths["source_video"]),
            "sha256": _sha256(paths["source_video"]),
            "download_report": str(copied_report) if copied_report else None,
            "download_report_sha256": _sha256(copied_report) if copied_report else None,
        },
        "interval": {
            "start_s": args.start_seconds,
            "end_s_exclusive": args.start_seconds + args.duration_seconds,
            "duration_s": args.duration_seconds,
            "annotations": annotation_evidence,
        },
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
            "info": info,
            "anchors": anchors,
        },
        "acceptance": {
            "official_public_dataset_source": True,
            "license_recorded": True,
            "annotation_interval_nonempty": True,
            "exact_frame_count": int(stream["nb_frames"]) == expected_frames,
            "exact_fps": int(numerator) / int(denominator) == args.fps,
            "exact_duration_seconds": expected_frames / args.fps,
        },
        "limitations": [
            "EPIC-KITCHENS is licensed CC BY-NC 4.0; this artifact is research-only and requires attribution.",
            "The source depicts a human camera wearer; robot output is a counterfactual generated visualization, not recorded robot execution.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"output": str(output_dir), "acceptance": manifest["acceptance"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
