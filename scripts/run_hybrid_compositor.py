#!/usr/bin/env python3
"""Compose a deterministic robot render over source video without full-frame generation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.evaluation.object_instance import (  # noqa: E402
    NormalizedROI,
    ObjectTrackerConfig,
    RGBFrames,
    decode_video,
    encode_video,
    track_colored_object,
)
from phiagent.evaluation.video_proxy import resolve_ffmpeg  # noqa: E402
from phiagent.rendering.hybrid_compositor import (  # noqa: E402
    ScreenSpaceOverlayConfig,
    composite_robot_layer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--robot-layer-video", type=Path, required=True)
    parser.add_argument(
        "--object-roi",
        nargs=4,
        type=float,
        required=True,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
    )
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/hybrid-compositor"))
    parser.add_argument("--source-width", type=int, default=896)
    parser.add_argument("--source-height", type=int, default=512)
    parser.add_argument("--robot-width", type=int, default=640)
    parser.add_argument("--robot-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-num", type=int, default=90)
    parser.add_argument("--target-width-fraction", type=float, default=0.30)
    parser.add_argument("--anchor-offset-x", type=float, default=-0.03)
    parser.add_argument("--anchor-offset-y", type=float, default=-0.13)
    parser.add_argument("--black-level", type=int, default=20)
    parser.add_argument("--robot-quarter-turns-clockwise", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--object-color-mode", choices=("chromatic", "cyan"), default="cyan")
    parser.add_argument("--object-chroma-tolerance", type=int, default=16)
    parser.add_argument("--object-brightness-tolerance", type=int, default=64)
    parser.add_argument("--object-search-margin", type=float, default=0.08)
    parser.add_argument("--maximum-object-area-fraction", type=float, default=0.08)
    parser.add_argument("--ffmpeg", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _capture(command: list[str], cwd: Path | None = None) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    args = _parser().parse_args()
    if min(
        args.source_width,
        args.source_height,
        args.robot_width,
        args.robot_height,
        args.fps,
        args.frame_num,
    ) <= 0:
        raise ValueError("video dimensions, FPS, and frame count must be positive")
    if not 0 < args.maximum_object_area_fraction < 1:
        raise ValueError("maximum-object-area-fraction must be in (0, 1)")
    source_video = args.source_video.expanduser().resolve()
    robot_video = args.robot_layer_video.expanduser().resolve()
    for label, path in (("source video", source_video), ("robot layer video", robot_video)):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    ffmpeg = resolve_ffmpeg(args.ffmpeg)

    root = args.experiment_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = root / f"{stamp}-{uuid4().hex[:8]}"
    experiment.mkdir()
    output = experiment / "hybrid.mp4"
    object_mask = experiment / "source_object_mask.mp4"
    manifest_path = experiment / "manifest.json"
    log_path = experiment / "run.log"
    started = time.monotonic()
    log_path.write_text(f"{datetime.now(timezone.utc).isoformat()} experiment created\n")

    tracker_config = ObjectTrackerConfig(
        initial_roi=NormalizedROI(*args.object_roi),
        initial_color_mode=args.object_color_mode,
        chroma_tolerance=args.object_chroma_tolerance,
        brightness_tolerance=args.object_brightness_tolerance,
        search_margin=args.object_search_margin,
    )
    overlay_config = ScreenSpaceOverlayConfig(
        target_width_fraction=args.target_width_fraction,
        anchor_offset_x_fraction=args.anchor_offset_x,
        anchor_offset_y_fraction=args.anchor_offset_y,
        black_level=args.black_level,
        quarter_turns_clockwise=args.robot_quarter_turns_clockwise,
    )
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "running",
        "method": "deterministic_source_preserving_robot_overlay",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "seed": None,
        "git": {
            "head": _capture(
                ["git", "rev-parse", "--verify", "HEAD"],
                Path(__file__).resolve().parents[1],
            ),
            "status": _capture(
                ["git", "--no-pager", "status", "--short"],
                Path(__file__).resolve().parents[1],
            ),
        },
        "packages": sorted(
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        ),
        "ffmpeg": _capture([str(ffmpeg), "-version"]),
        "inputs": {
            "source_video": str(source_video),
            "source_sha256": _sha256(source_video),
            "robot_layer_video": str(robot_video),
            "robot_layer_sha256": _sha256(robot_video),
        },
        "tracker_config": asdict(tracker_config),
        "overlay_config": asdict(overlay_config),
        "image_pixel_frames": {
            "source": [args.source_width, args.source_height],
            "robot_layer": [args.robot_width, args.robot_height],
        },
        "limitations": [
            "Placement is screen-space object-relative, not a camera-calibrated 3D wrist pose.",
            "The compositor does not remove exposed source-hand pixels outside the robot silhouette.",
            "No generative visual enhancement is applied.",
        ],
    }
    _write_json(manifest_path, manifest)

    try:
        with log_path.open("a") as log:
            log.write(f"{datetime.now(timezone.utc).isoformat()} decoding inputs\n")
        source = RGBFrames(
            decode_video(
                source_video,
                ffmpeg,
                width=args.source_width,
                height=args.source_height,
                fps=args.fps,
                frame_num=args.frame_num,
                pixel_format="rgb24",
            ),
            args.source_width,
            args.source_height,
        )
        robot = RGBFrames(
            decode_video(
                robot_video,
                ffmpeg,
                width=args.robot_width,
                height=args.robot_height,
                fps=args.fps,
                frame_num=args.frame_num,
                pixel_format="rgb24",
            ),
            args.robot_width,
            args.robot_height,
        )
        track = track_colored_object(source, tracker_config)
        maximum_area_fraction = max(track.areas) / (source.width * source.height)
        if maximum_area_fraction > args.maximum_object_area_fraction:
            raise ValueError(
                "tracked object covers "
                f"{maximum_area_fraction:.3%} of a frame, above the configured "
                f"{args.maximum_object_area_fraction:.3%} maximum"
            )
        with log_path.open("a") as log:
            log.write(f"{datetime.now(timezone.utc).isoformat()} compositing robot layer\n")
        composited, metrics = composite_robot_layer(source, robot, track, overlay_config)
        encode_video(
            composited.frames,
            output,
            ffmpeg,
            width=args.source_width,
            height=args.source_height,
            fps=args.fps,
            pixel_format="rgb24",
        )
        encode_video(
            tuple(bytes(255 if value else 0 for value in mask) for mask in track.masks),
            object_mask,
            ffmpeg,
            width=args.source_width,
            height=args.source_height,
            fps=args.fps,
            pixel_format="gray",
        )
        with log_path.open("a") as log:
            log.write(f"{datetime.now(timezone.utc).isoformat()} outputs encoded\n")
        manifest.update(
            {
                "status": "succeeded",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "metrics": metrics.to_dict(),
                "outputs": {
                    "video": str(output),
                    "video_sha256": _sha256(output),
                    "object_mask": str(object_mask),
                    "object_mask_sha256": _sha256(object_mask),
                    "log": str(log_path),
                    "log_sha256": _sha256(log_path),
                },
            }
        )
        _write_json(manifest_path, manifest)
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "error": repr(exc),
            }
        )
        _write_json(manifest_path, manifest)
        raise

    print(f"EXPERIMENT={experiment}")
    print(f"VIDEO={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
