#!/usr/bin/env python3
"""Build a fail-closed Wan driver mask while retaining task-specific pose controls."""

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
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(cv2: Any, path: Path) -> tuple[list[Any], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    return frames, fps


def _write_video(ffmpeg: Path, output: Path, frames: list[Any], fps: float) -> None:
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(
        [
            str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.8f}",
            "-i", "-", "-an", "-c:v", "libx264", "-crf", "0",
            "-pix_fmt", "yuv420p", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed for {output}")


def _pose_capsule_mask(
    cv2: Any,
    np: Any,
    pose_frame: Any,
    driver_frame: Any,
    *,
    pose_dilation_pixels: int,
    blue_dilation_pixels: int,
    minimum_component_area: int,
) -> Any:
    """Create smooth subject support from robot kinematics and the blue object."""
    pose_seed = np.max(pose_frame, axis=2) >= 12
    pose_size = pose_dilation_pixels * 2 + 1
    pose_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pose_size, pose_size))
    pose_support = cv2.dilate(pose_seed.astype(np.uint8) * 255, pose_kernel)

    hsv = cv2.cvtColor(driver_frame, cv2.COLOR_BGR2HSV)
    blue = (
        (hsv[:, :, 0] >= 88)
        & (hsv[:, :, 0] <= 135)
        & (hsv[:, :, 1] >= 70)
        & (hsv[:, :, 2] >= 40)
    ).astype(np.uint8) * 255
    blue = cv2.morphologyEx(
        blue,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    blue_size = blue_dilation_pixels * 2 + 1
    blue = cv2.dilate(
        blue,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blue_size, blue_size)),
    )
    merged = cv2.bitwise_or(pose_support, blue)
    closed = cv2.morphologyEx(
        merged,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    filtered = np.zeros_like(closed)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_component_area:
            filtered[labels == component] = 255
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--driver-video", type=Path, required=True)
    parser.add_argument("--diagnostic-preprocess", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--difference-threshold", type=int, default=48)
    parser.add_argument("--dilation-pixels", type=int, default=9)
    parser.add_argument(
        "--mask-strategy",
        choices=("difference_union", "pose_capsule"),
        default="difference_union",
    )
    parser.add_argument("--pose-dilation-pixels", type=int, default=42)
    parser.add_argument("--blue-dilation-pixels", type=int, default=14)
    parser.add_argument("--minimum-component-area", type=int, default=256)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    args = parser.parse_args()

    import cv2
    import numpy as np

    source_path = args.source_video.expanduser().resolve()
    driver_path = args.driver_video.expanduser().resolve()
    diagnostic = args.diagnostic_preprocess.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"override directory already exists: {output}")
    required = [
        source_path, driver_path, ffmpeg,
        *(diagnostic / name for name in ("src_pose.mp4", "src_face.mp4", "src_mask.mp4", "src_ref.png")),
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required override input is missing: {path}")
    if (
        not 1 <= args.difference_threshold <= 255
        or args.dilation_pixels < 0
        or args.pose_dilation_pixels < 1
        or args.blue_dilation_pixels < 1
        or args.minimum_component_area < 1
    ):
        raise ValueError("mask thresholds are outside their valid range")

    source, source_fps = _decode(cv2, source_path)
    driver, driver_fps = _decode(cv2, driver_path)
    raw_masks, mask_fps = _decode(cv2, diagnostic / "src_mask.mp4")
    pose_frames, pose_fps = _decode(cv2, diagnostic / "src_pose.mp4")
    frame_count = min(len(source), len(driver), len(raw_masks), len(pose_frames))
    if frame_count < 1 or any(abs(fps - 24.0) > 1e-6 for fps in (source_fps, driver_fps, mask_fps, pose_fps)):
        raise ValueError("override inputs must be nonempty 24 FPS videos")
    height, width = pose_frames[0].shape[:2]
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilate_size = args.dilation_pixels * 2 + 1
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    masks = []
    support_fractions = []
    raw_fractions = []
    for index in range(frame_count):
        source_frame = cv2.resize(source[index], (width, height), interpolation=cv2.INTER_LANCZOS4)
        driver_frame = cv2.resize(driver[index], (width, height), interpolation=cv2.INTER_LANCZOS4)
        raw = cv2.cvtColor(
            cv2.resize(raw_masks[index], (width, height), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_BGR2GRAY,
        )
        raw = (raw >= 127).astype(np.uint8) * 255
        if args.mask_strategy == "difference_union":
            difference = np.max(
                np.abs(
                    driver_frame.astype(np.int16) - source_frame.astype(np.int16)
                ),
                axis=2,
            )
            fallback = (difference >= args.difference_threshold).astype(np.uint8) * 255
            fallback = cv2.morphologyEx(fallback, cv2.MORPH_OPEN, open_kernel)
            fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, close_kernel)
            fallback = cv2.dilate(fallback, dilate_kernel)
            merged = cv2.bitwise_or(raw, fallback)
        else:
            merged = _pose_capsule_mask(
                cv2,
                np,
                pose_frames[index],
                driver_frame,
                pose_dilation_pixels=args.pose_dilation_pixels,
                blue_dilation_pixels=args.blue_dilation_pixels,
                minimum_component_area=args.minimum_component_area,
            )
        if not np.any(merged):
            raise RuntimeError(f"override mask is empty at frame {index}")
        masks.append(cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR))
        support_fractions.append(float(np.mean(merged > 0)))
        raw_fractions.append(float(np.mean(raw > 0)))

    output.mkdir(parents=True)
    for name in ("src_pose.mp4", "src_face.mp4", "src_ref.png"):
        shutil.copy2(diagnostic / name, output / name)
    _write_video(ffmpeg, output / "src_mask.mp4", masks, 24.0)
    manifest = {
        "schema_version": "1.0.0",
        "status": "completed",
        "method": f"wan_pose_controls_with_{args.mask_strategy}_fail_closed_mask",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "opencv-python")
        },
        "gpu": {"used": False, "reason": "deterministic mask compilation"},
        "coordinate_frame": "camera:Wan_preprocess_pixels",
        "inputs": {
            "source_video": {"path": str(source_path), "sha256": _sha256(source_path)},
            "driver_video": {"path": str(driver_path), "sha256": _sha256(driver_path)},
            "diagnostic_preprocess": str(diagnostic),
        },
        "config": {
            "difference_threshold": args.difference_threshold,
            "dilation_pixels": args.dilation_pixels,
            "mask_strategy": args.mask_strategy,
            "pose_dilation_pixels": args.pose_dilation_pixels,
            "blue_dilation_pixels": args.blue_dilation_pixels,
            "minimum_component_area": args.minimum_component_area,
            "width": width,
            "height": height,
            "fps": 24,
            "frames": frame_count,
        },
        "metrics": {
            "raw_sam_mean_fraction": float(np.mean(raw_fractions)),
            "compiled_mean_fraction": float(np.mean(support_fractions)),
            "compiled_minimum_fraction": float(np.min(support_fractions)),
            "empty_frames": 0,
        },
        "outputs": {
            name: {"sha256": _sha256(output / name)}
            for name in ("src_pose.mp4", "src_face.mp4", "src_mask.mp4", "src_ref.png")
        },
        "limitations": [
            "The fallback support is appearance-difference based and deliberately overinclusive.",
            "This bundle is an intermediate Wan control, not a generated video.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "metrics": manifest["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
