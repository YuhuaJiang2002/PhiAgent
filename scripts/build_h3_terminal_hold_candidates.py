#!/usr/bin/env python3
"""Build seam-free ten-second H3 candidates by holding an accepted terminal state."""

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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_info(ffprobe: Path, path: Path) -> dict[str, float | int]:
    command = [
        str(ffprobe), "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path),
    ]
    stream = json.loads(
        subprocess.run(command, check=True, capture_output=True, text=True).stdout
    )["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/", maxsplit=1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(numerator) / int(denominator),
        "frames": int(stream["nb_read_frames"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-manifest", type=Path, required=True)
    parser.add_argument("--action", action="append", default=[])
    parser.add_argument("--support-mask", action="append", default=[])
    parser.add_argument("--cutoff-frame", type=int, default=123)
    parser.add_argument("--total-frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/usr/bin/ffprobe"))
    args = parser.parse_args()

    manifest_path = args.driver_manifest.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    ffprobe = args.ffprobe.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"terminal-hold experiment already exists: {output}")
    if not 1 <= args.cutoff_frame < args.total_frames:
        raise ValueError("cutoff frame must be inside the final video")
    requested = tuple(args.action)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--action requires one or more unique labels")
    payload = json.loads(manifest_path.read_text())
    drivers = {str(item["label"]): item for item in payload["actions"]}
    if not set(requested).issubset(drivers):
        raise ValueError("requested action is absent from the driver manifest")
    support_masks: dict[str, Path] = {}
    for value in args.support_mask:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or label in support_masks:
            raise ValueError("--support-mask requires unique LABEL=PATH pairs")
        support_masks[label] = Path(raw_path).expanduser().resolve()
    if set(support_masks) != set(requested):
        raise ValueError("every requested action requires one support mask")

    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for label in requested:
        driver = Path(str(drivers[label]["output"])).expanduser().resolve()
        info = _video_info(ffprobe, driver)
        if int(info["frames"]) != args.total_frames or abs(float(info["fps"]) - args.fps) > 1e-6:
            raise ValueError(f"raw driver has unexpected geometry: {label} {info}")
        variant = output / "variants" / label
        variant.mkdir(parents=True)
        candidate = variant / f"{label}-h3-terminal-hold-10s.mp4"
        hold_seconds = (args.total_frames - args.cutoff_frame) / args.fps
        command = [
            str(ffmpeg), "-y", "-v", "error", "-i", str(driver), "-vf",
            (
                f"trim=end_frame={args.cutoff_frame},setpts=N/({args.fps}*TB),"
                f"tpad=stop_mode=clone:stop_duration={hold_seconds:.9f}"
            ),
            "-frames:v", str(args.total_frames), "-r", str(args.fps), "-an",
            "-c:v", "libx264", "-crf", "12", "-preset", "slow",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(candidate),
        ]
        subprocess.run(command, check=True)
        candidate_info = _video_info(ffprobe, candidate)
        if int(candidate_info["frames"]) != args.total_frames:
            raise RuntimeError(f"terminal-hold output is incomplete: {label}")
        metadata = {
            "schema_version": "1.0.0",
            "status": "succeeded",
            "method": "h3_nf4_terminal_state_hold_no_interpolation_no_blur",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "ffmpeg_command": command,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "seed": 20260811,
            "inputs": {
                "action_driver": {
                    "path": str(driver),
                    "sha256": _sha256(driver),
                }
            },
            "cutoff_frame_exclusive": args.cutoff_frame,
            "cloned_terminal_frames": args.total_frames - args.cutoff_frame,
            "postprocessing": {
                "alpha_repair": False,
                "blur": False,
                "cross_dissolve": False,
                "source_person_restore": False,
                "temporal_filter": False,
            },
            "final_output": str(candidate),
            "final_output_sha256": _sha256(candidate),
            "final_output_info": candidate_info,
            "claim_boundary": "H3 camera-frame action visualization; not physical execution.",
        }
        support_source = support_masks[label]
        support_output = variant / f"{label}-terminal-hold-support.mp4"
        support_command = [
            str(ffmpeg), "-y", "-v", "error", "-i", str(support_source), "-vf",
            (
                f"trim=end_frame={args.cutoff_frame},setpts=N/({args.fps}*TB),"
                f"tpad=stop_mode=clone:stop_duration={hold_seconds:.9f}"
            ),
            "-frames:v", str(args.total_frames), "-r", str(args.fps), "-an",
            "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
            str(support_output),
        ]
        subprocess.run(support_command, check=True)
        metadata["inputs"]["support_mask"] = {
            "path": str(support_source),
            "sha256": _sha256(support_source),
        }
        metadata["support_mask"] = {
            "path": str(support_output),
            "sha256": _sha256(support_output),
            "info": _video_info(ffprobe, support_output),
            "ffmpeg_command": support_command,
        }
        metadata_path = variant / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        records.append({"label": label, **metadata})

    result = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "driver_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "actions": records,
        "limitation": "The accepted terminal frame is held; no post-cutoff motion is claimed.",
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
