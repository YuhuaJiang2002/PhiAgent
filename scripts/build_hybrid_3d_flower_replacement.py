#!/usr/bin/env python3
"""Composite a source-driven articulated 3D robot into the real flower scene.

The input robot video is an immutable MuJoCo render whose motion was driven by
the same source frames.  This CPU-only stage recovers its alpha matte, places it
in ``camera:source_pixels``, and restores tracked source flowers as an explicit
foreground layer.  It does not claim contact physics or final visual acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_articulated_flower_robot_demo import _synthetic_scene  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_packed(np: Any, path: Path) -> Any:
    payload = np.load(path)
    height, width = int(payload["height"]), int(payload["width"])
    packed = payload["packed"]
    unpacked = np.unpackbits(packed, axis=1, bitorder=str(payload["bitorder"]))
    return unpacked[:, : height * width].reshape(len(packed), height, width).astype(bool)


def _git_state() -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
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
    }


def _largest_components(cv2: Any, np: Any, mask: Any, minimum_area: int) -> Any:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    result = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            result[labels == label] = 255
    return result


def _recover_robot_mask(
    cv2: Any,
    np: Any,
    frame: Any,
    synthetic_scene: Any,
    threshold: int,
) -> Any:
    difference = np.max(
        np.abs(frame.astype(np.int16) - synthetic_scene.astype(np.int16)), axis=2
    )
    mask = (difference >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return _largest_components(cv2, np, mask, minimum_area=36)


def _writer(ffmpeg: str, output: Path, width: int, height: int, fps: float) -> Any:
    return subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.12g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--clean-plate", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--articulated-robot-video", type=Path, required=True)
    parser.add_argument("--articulated-robot-mask", type=Path)
    parser.add_argument("--articulated-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.10)
    parser.add_argument("--translate-x", type=float, default=242.0)
    parser.add_argument("--translate-y", type=float, default=8.0)
    parser.add_argument("--difference-threshold", type=int, default=18)
    parser.add_argument("--foreground-occlusion-start-y", type=int, default=330)
    parser.add_argument("--foreground-occlusion-feather", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    inputs = {
        "source_video": args.source_video.expanduser().resolve(),
        "clean_plate": args.clean_plate.expanduser().resolve(),
        "flower_masks": args.flower_masks.expanduser().resolve(),
        "articulated_robot_video": args.articulated_robot_video.expanduser().resolve(),
        "articulated_manifest": args.articulated_manifest.expanduser().resolve(),
    }
    if args.articulated_robot_mask is not None:
        inputs["articulated_robot_mask"] = args.articulated_robot_mask.expanduser().resolve()
    for name, path in inputs.items():
        if not path.is_file():
            raise ValueError(f"{name} does not exist: {path}")
    if not 0.5 <= args.scale <= 2.0:
        raise ValueError("scale must be in [0.5, 2.0]")
    if not 1 <= args.difference_threshold <= 255:
        raise ValueError("difference threshold must be in [1, 255]")
    if not 0 <= args.foreground_occlusion_start_y < 480:
        raise ValueError("foreground occlusion start must be in [0, 479]")
    if not 1 <= args.foreground_occlusion_feather <= 100:
        raise ValueError("foreground occlusion feather must be in [1, 100]")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    import cv2
    import numpy as np

    source_capture = cv2.VideoCapture(str(inputs["source_video"]))
    robot_capture = cv2.VideoCapture(str(inputs["articulated_robot_video"]))
    robot_mask_capture = (
        cv2.VideoCapture(str(inputs["articulated_robot_mask"]))
        if "articulated_robot_mask" in inputs
        else None
    )
    if not source_capture.isOpened() or not robot_capture.isOpened():
        raise RuntimeError("could not open source or articulated robot video")
    width = int(source_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(source_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(source_capture.get(cv2.CAP_PROP_FPS))
    robot_frames = int(robot_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    mask_frames = (
        int(robot_mask_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if robot_mask_capture is not None
        else frames
    )
    if (
        (width, height, frames) != (832, 480, 660)
        or robot_frames != frames
        or mask_frames != frames
    ):
        raise RuntimeError(
            f"expected aligned 832x480x660 inputs; source={(width, height, frames)}, "
            f"robot_frames={robot_frames}, mask_frames={mask_frames}"
        )
    clean_plate = cv2.imread(str(inputs["clean_plate"]), cv2.IMREAD_COLOR)
    if clean_plate is None:
        raise RuntimeError("could not decode clean plate")
    if clean_plate.shape[:2] != (height, width):
        clean_plate = cv2.resize(clean_plate, (width, height), interpolation=cv2.INTER_LANCZOS4)
    flower_masks = _load_packed(np, inputs["flower_masks"])
    if flower_masks.shape != (frames, height, width):
        raise RuntimeError(f"unexpected flower mask shape: {flower_masks.shape}")

    ffmpeg = subprocess.run(
        ["which", "ffmpeg"], check=True, capture_output=True, text=True
    ).stdout.strip()
    output_video = output_dir / "hybrid-3d-flower-replacement.mp4"
    writer = _writer(ffmpeg, output_video, width, height, fps)
    synthetic_scene = _synthetic_scene(cv2, np, 640, 480)
    transform = np.asarray(
        [[args.scale, 0.0, args.translate_x], [0.0, args.scale, args.translate_y]],
        dtype=np.float32,
    )
    mask_area = []
    flower_area = []
    frame_count = 0
    try:
        for index in range(frames):
            source_ok, source = source_capture.read()
            robot_ok, robot = robot_capture.read()
            if not source_ok or not robot_ok:
                raise RuntimeError(f"decode stopped at frame {index}")
            if robot_mask_capture is None:
                robot_mask = _recover_robot_mask(
                    cv2, np, robot, synthetic_scene, args.difference_threshold
                )
            else:
                mask_ok, mask_frame = robot_mask_capture.read()
                if not mask_ok:
                    raise RuntimeError(f"robot mask decode stopped at frame {index}")
                robot_mask = (
                    cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY) >= 127
                ).astype(np.uint8) * 255
            placed_robot = cv2.warpAffine(
                robot,
                transform,
                (width, height),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
            )
            placed_mask = cv2.warpAffine(
                robot_mask,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            robot_alpha = cv2.GaussianBlur(placed_mask, (3, 3), 0).astype(np.float32) / 255.0
            composed = np.rint(
                placed_robot.astype(np.float32) * robot_alpha[..., None]
                + clean_plate.astype(np.float32) * (1.0 - robot_alpha[..., None])
            ).astype(np.uint8)

            # The source actor stands behind the workbench.  Restore the clean
            # foreground plate below the named camera-pixel horizon so the 3D
            # robot obeys the same scene depth instead of floating in front.
            foreground_alpha = np.clip(
                (
                    np.arange(height, dtype=np.float32)
                    - float(args.foreground_occlusion_start_y)
                )
                / float(args.foreground_occlusion_feather),
                0.0,
                1.0,
            )[:, None]
            composed = np.rint(
                clean_plate.astype(np.float32) * foreground_alpha[..., None]
                + composed.astype(np.float32) * (1.0 - foreground_alpha[..., None])
            ).astype(np.uint8)

            flower_mask = flower_masks[index].astype(np.uint8) * 255
            flower_alpha = cv2.GaussianBlur(flower_mask, (3, 3), 0).astype(np.float32) / 255.0
            composed = np.rint(
                source.astype(np.float32) * flower_alpha[..., None]
                + composed.astype(np.float32) * (1.0 - flower_alpha[..., None])
            ).astype(np.uint8)
            if writer.stdin is None:
                raise RuntimeError("ffmpeg stdin closed unexpectedly")
            writer.stdin.write(composed.tobytes())
            mask_area.append(float(np.mean(placed_mask > 0)))
            flower_area.append(float(np.mean(flower_mask > 0)))
            frame_count += 1
    finally:
        source_capture.release()
        robot_capture.release()
        if robot_mask_capture is not None:
            robot_mask_capture.release()
        if writer.stdin is not None:
            writer.stdin.close()
        return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with code {return_code}")
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(output_video), "-f", "null", "-"], check=True
    )

    packages = {}
    for package in ("numpy", "opencv-python"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    upstream = json.loads(inputs["articulated_manifest"].read_text())
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "honest_status": "PARTIAL",
        "method": "source_driven_mujoco_robot_plus_tracked_flower_depth_layers",
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "seed": args.seed,
        "git": _git_state(),
        "coordinate_frames": {
            "composition": "camera:source_pixels",
            "upstream_kinematics": "robot:base",
            "flower_layer": "object:flower projected into camera:source_pixels",
        },
        "gpu": {
            "used": False,
            "reason": "CPU composition from an immutable upstream 3D render",
            "upstream_gpu": upstream.get("gpu"),
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "parameters": {
            "scale": args.scale,
            "translate_x": args.translate_x,
            "translate_y": args.translate_y,
            "difference_threshold": args.difference_threshold,
            "explicit_robot_alpha": robot_mask_capture is not None,
            "foreground_occlusion_start_y": args.foreground_occlusion_start_y,
            "foreground_occlusion_feather": args.foreground_occlusion_feather,
            "layer_order": [
                "clean_plate",
                "articulated_3d_robot",
                "clean_plate_foreground_occluder",
                "tracked_source_flowers",
            ],
        },
        "measurements": {
            "frames": frame_count,
            "fps": fps,
            "robot_mask_fraction_min": min(mask_area),
            "robot_mask_fraction_max": max(mask_area),
            "flower_mask_fraction_min": min(flower_area),
            "flower_mask_fraction_max": max(flower_area),
        },
        "output": {"path": str(output_video), "sha256": _sha256(output_video)},
        "entrypoint": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "acceptance": {
            "full_decode": True,
            "explicit_3d_robot": True,
            "tracked_flower_layer": True,
            "contact_verified": False,
            "occlusion_verified": False,
            "full_video_human_preference": False,
        },
        "limitations": [
            "The affine camera placement is calibrated in source pixels but is not a recovered metric camera.",
            "All tracked flowers are currently composited in front of the robot; per-contact depth order remains unverified.",
            "The upstream IK follows source wrists and finger flexion but does not enforce flower contact physics.",
            "No WORKING claim is allowed before all-frame semantic evaluation and blind full-video review.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
