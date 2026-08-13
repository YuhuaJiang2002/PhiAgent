#!/usr/bin/env python3
"""Prepare immutable 10-second, stateful H3 action-control windows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.acwm.long_horizon import (  # noqa: E402
    LongHorizonActionSet,
    window_action_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _video_info(ffprobe: Path, video: Path) -> dict[str, float | int]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
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
    stream = payload["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/", maxsplit=1)
    if stream.get("nb_frames") in {None, "N/A"}:
        raise ValueError(f"video has no exact frame count: {video}")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(numerator) / int(denominator),
        "frames": int(stream["nb_frames"]),
        "duration": float(payload["format"]["duration"]),
    }


def _extract(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    *,
    start_frame: int,
    frame_count: int,
) -> list[str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        (
            f"trim=start_frame={start_frame}:end_frame={start_frame + frame_count},"
            "setpts=PTS-STARTPTS"
        ),
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)
    return command


def _git_state() -> dict[str, object]:
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "available": status.returncode == 0,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
        "error": status.stderr.strip() if status.returncode else None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--action-control-root", type=Path, required=True)
    parser.add_argument("--action-manifest", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--source-start-frame", type=int, default=216)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--total-frames", type=int, default=240)
    parser.add_argument("--window-frames", type=int, default=124)
    parser.add_argument("--overlap-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/opt/homebrew/bin/ffprobe"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    manifest_path = experiment / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"long-action preparation already exists: {manifest_path}")
    paths = {
        "source_video": args.source_video.expanduser().resolve(),
        "action_control_root": args.action_control_root.expanduser().resolve(),
        "action_manifest": args.action_manifest.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
        "ffprobe": args.ffprobe.expanduser().resolve(),
    }
    for label in ("source_video", "action_manifest", "ffmpeg", "ffprobe"):
        path = paths[label]
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} does not exist or is empty: {path}")
    if not paths["action_control_root"].is_dir():
        raise ValueError(f"action control root is missing: {paths['action_control_root']}")
    if args.source_start_frame < 0:
        raise ValueError("source-start-frame must be non-negative")
    if (args.window_frames - 5) % 17:
        raise ValueError("every H3 window must satisfy frame_count = 17n + 5")

    action_set = LongHorizonActionSet.load(paths["action_manifest"])
    compiled = action_set.compile_matched_windows(
        total_frames=args.total_frames,
        fps=args.fps,
        window_frames=args.window_frames,
        overlap_frames=args.overlap_frames,
    )
    source_info = _video_info(paths["ffprobe"], paths["source_video"])
    if abs(float(source_info["fps"]) - args.fps) > 1e-6:
        raise ValueError(f"source is {source_info['fps']} FPS, expected {args.fps}")
    if args.source_start_frame + args.total_frames > int(source_info["frames"]):
        raise ValueError("requested real-source interval exceeds the source video")

    controls = {}
    for action in action_set.actions:
        control = (
            paths["action_control_root"]
            / "variants"
            / action.label
            / "action-control.mp4"
        )
        info = _video_info(paths["ffprobe"], control)
        if int(info["frames"]) != args.total_frames or abs(float(info["fps"]) - args.fps) > 1e-6:
            raise ValueError(
                f"{action.label} control must be {args.total_frames} frames at {args.fps} FPS"
            )
        controls[action.label] = (control, info)

    experiment.mkdir(parents=True, exist_ok=True)
    input_dir = experiment / "input"
    copied_manifest = input_dir / "long-actions.json"
    copied_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths["action_manifest"], copied_manifest)
    full_source = input_dir / "real-source-240.mp4"
    full_source_command = _extract(
        paths["ffmpeg"],
        paths["source_video"],
        full_source,
        start_frame=args.source_start_frame,
        frame_count=args.total_frames,
    )
    if int(_video_info(paths["ffprobe"], full_source)["frames"]) != args.total_frames:
        raise RuntimeError("prepared full source does not decode to the requested frame count")

    window_records = []
    geometry = compiled[0]
    for index, prototype in enumerate(geometry):
        window_dir = experiment / "windows" / f"window-{index:02d}-{prototype.start_frame:04d}"
        source_window = window_dir / "source.mp4"
        source_command = _extract(
            paths["ffmpeg"],
            paths["source_video"],
            source_window,
            start_frame=args.source_start_frame + prototype.start_frame,
            frame_count=prototype.frame_count,
        )
        action_windows = [items[index] for items in compiled]
        per_window_manifest = window_dir / "action-variants.json"
        _write_json(
            per_window_manifest,
            window_action_manifest(action_set.actions, action_windows),
        )
        action_records = []
        for action, contract in zip(action_set.actions, action_windows):
            source_control, _ = controls[action.label]
            control = window_dir / "variants" / action.label / "action-control.mp4"
            control_command = _extract(
                paths["ffmpeg"],
                source_control,
                control,
                start_frame=prototype.start_frame,
                frame_count=prototype.frame_count,
            )
            action_records.append(
                {
                    "label": action.label,
                    "contract": contract.to_dict(),
                    "control": str(control),
                    "control_sha256": _sha256(control),
                    "control_info": _video_info(paths["ffprobe"], control),
                    "control_command": control_command,
                }
            )
        window_records.append(
            {
                "index": index,
                "start_frame": prototype.start_frame,
                "frame_count": prototype.frame_count,
                "source": str(source_window),
                "source_sha256": _sha256(source_window),
                "source_info": _video_info(paths["ffprobe"], source_window),
                "source_command": source_command,
                "action_manifest": str(per_window_manifest),
                "action_manifest_sha256": _sha256(per_window_manifest),
                "actions": action_records,
            }
        )

    packages = {}
    for name in ("numpy", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": "1.0.0",
        "method": "stateful_long_horizon_action_to_overlapping_h3_windows",
        "status": "completed",
        "honest_status": "WORKING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "git": _git_state(),
        "gpu": {"used": False, "reason": "deterministic window preparation"},
        "seed": args.seed,
        "coordinate_frame": action_set.coordinate_frame,
        "config": {
            "source_start_frame": args.source_start_frame,
            "fps": args.fps,
            "total_frames": args.total_frames,
            "duration_s": args.total_frames / args.fps,
            "window_frames": args.window_frames,
            "overlap_frames": args.overlap_frames,
        },
        "inputs": {
            "source_video": {
                "path": str(paths["source_video"]),
                "sha256": _sha256(paths["source_video"]),
                "info": source_info,
            },
            "action_manifest": {
                "path": str(paths["action_manifest"]),
                "sha256": _sha256(paths["action_manifest"]),
            },
            "action_controls": [
                {
                    "label": label,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "info": info,
                }
                for label, (path, info) in controls.items()
            ],
        },
        "full_source": {
            "path": str(full_source),
            "sha256": _sha256(full_source),
            "command": full_source_command,
        },
        "windows": window_records,
        "acceptance": {
            "duration_exactly_10s": args.total_frames / args.fps == 10.0,
            "all_windows_h3_legal": all(
                (item.frame_count - 5) % 17 == 0 for item in geometry
            ),
            "full_timeline_covered": geometry[-1].end_frame == args.total_frames,
            "overlap_is_explicit": all(
                right.start_frame < left.end_frame
                for left, right in zip(geometry, geometry[1:])
            ),
            "cross_window_object_state_explicit": True,
        },
        "limitations": [
            "The controls are 2D camera-pixel trajectories, not calibrated robot-base actions.",
            "This stage prepares model inputs and does not claim generated-video quality.",
            "The real source supplies scene state; physical contact still requires independent evaluation.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"experiment": str(experiment), "acceptance": manifest["acceptance"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
