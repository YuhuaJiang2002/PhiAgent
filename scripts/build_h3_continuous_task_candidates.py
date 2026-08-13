#!/usr/bin/env python3
"""Build ten-second H3 task videos from state-valid continuation tails.

The continuation crop is restricted to a human-reviewed frame interval in which
the commanded task state is already valid.  The retained tail is resampled by
nearest-frame selection only, so this step cannot introduce hand ghosts or blur.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
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


def _parse_action(value: str) -> tuple[str, int, int]:
    label, separator, interval = value.partition("=")
    if not separator or not label:
        raise argparse.ArgumentTypeError("action must be LABEL=START,END")
    try:
        start_text, end_text = interval.split(",", maxsplit=1)
        start, end = int(start_text), int(end_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("action must be LABEL=START,END") from error
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("action interval must satisfy 0 <= START <= END")
    return label, start, end


def _nearest_indices(start: int, end: int, count: int, np: Any) -> Any:
    if count <= 0 or start < 0 or end < start:
        raise ValueError("invalid nearest-frame resampling request")
    return np.rint(np.linspace(start, end, count)).astype(np.int64)


def _select_seam_frame(previous: Any, following: list[Any], start: int, end: int, np: Any) -> tuple[int, list[dict[str, float | int]]]:
    if end >= len(following):
        raise ValueError("state-valid interval exceeds following video")
    height, width = previous.shape[:2]
    # Exclude the upper wall where small head-camera rotations dominate the
    # metric; the lower 82% contains the robot, bottle and interaction surface.
    y0 = int(round(0.18 * height))
    candidates = []
    for index in range(start, end + 1):
        difference = np.abs(
            previous[y0:].astype(np.float32) - following[index][y0:].astype(np.float32)
        )
        candidates.append(
            {
                "frame": index,
                "interaction_region_mad": float(np.mean(difference)),
            }
        )
    selected = min(candidates, key=lambda item: float(item["interaction_region_mad"]))
    return int(selected["frame"]), candidates


def _estimate_camera_alignment(previous: Any, following: Any, cv2: Any, np: Any) -> tuple[Any, float]:
    """Estimate a bounded camera-frame Euclidean warp, excluding lower robot pixels."""

    height, width = previous.shape[:2]
    template = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    candidate = cv2.cvtColor(following, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[: int(round(0.58 * height))] = 255
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        80,
        1e-6,
    )
    correlation, warp = cv2.findTransformECC(
        template,
        candidate,
        warp,
        cv2.MOTION_EUCLIDEAN,
        criteria,
        mask,
        5,
    )
    angle = math.degrees(math.atan2(float(warp[0, 1]), float(warp[0, 0])))
    translation_x = float(warp[0, 2])
    translation_y = float(warp[1, 2])
    if (
        abs(angle) > 12.0
        or abs(translation_x) > 0.18 * width
        or abs(translation_y) > 0.18 * height
    ):
        raise RuntimeError(
            "camera alignment exceeded bounded Euclidean transform: "
            f"angle={angle:.3f}, tx={translation_x:.3f}, ty={translation_y:.3f}"
        )
    return warp, float(correlation)


def _decayed_affine(warp: Any, offset: int, count: int, np: Any) -> Any:
    if count <= 0 or not 0 <= offset < count:
        raise ValueError("camera-alignment offset must be inside the transition")
    alpha = 1.0 if count == 1 else 1.0 - offset / (count - 1)
    identity = np.eye(2, 3, dtype=np.float32)
    return identity + (warp.astype(np.float32) - identity) * alpha


def _decode(path: Path, cv2: Any) -> tuple[list[Any], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"decoded no frames: {path}")
    return frames, fps


def _write_video(path: Path, frames: list[Any], fps: int, ffmpeg: Path) -> list[str]:
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed for {path}")
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--following-root", type=Path, required=True)
    parser.add_argument("--action", action="append", type=_parse_action, default=[])
    parser.add_argument("--previous-frames", type=int, default=123)
    parser.add_argument("--total-frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--camera-align-frames",
        type=int,
        default=0,
        help=(
            "Motion-align this many initial continuation frames with a bounded, "
            "decaying Euclidean camera transform; no alpha blending is used."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument(
        "--human-review",
        choices=("pending", "passed", "failed"),
        default="pending",
    )
    args = parser.parse_args()

    previous_root = args.previous_root.expanduser().resolve()
    following_root = args.following_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"continuous candidate experiment already exists: {manifest_path}")
    actions = tuple(args.action)
    labels = [item[0] for item in actions]
    if not actions or len(labels) != len(set(labels)):
        raise ValueError("--action requires one or more unique labels")
    if not 1 <= args.previous_frames < args.total_frames:
        raise ValueError("previous-frames must be inside the final video")
    if not 0 <= args.camera_align_frames <= args.total_frames - args.previous_frames:
        raise ValueError("camera-align-frames exceeds the continuation length")
    if not ffmpeg.is_file():
        raise ValueError(f"ffmpeg is missing: {ffmpeg}")

    import cv2
    import numpy as np

    records = []
    for label, state_start, state_end in actions:
        previous_path = previous_root / label / "variants" / label / "raw-h3-nf4.mp4"
        if not previous_path.is_file():
            previous_path = previous_root / "variants" / label / "raw-h3-nf4.mp4"
        following_path = following_root / "variants" / label / "raw-h3-nf4.mp4"
        for path in (previous_path, following_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"required H3 window is missing: {path}")
        previous, previous_fps = _decode(previous_path, cv2)
        following, following_fps = _decode(following_path, cv2)
        if previous[0].shape != following[0].shape:
            raise ValueError(f"window geometry mismatch for {label}")
        if abs(previous_fps - args.fps) > 1e-6 or abs(following_fps - args.fps) > 1e-6:
            raise ValueError(f"window FPS mismatch for {label}")
        if args.previous_frames > len(previous):
            raise ValueError(f"previous window is too short for {label}")

        selected, seam_candidates = _select_seam_frame(
            previous[args.previous_frames - 1], following, state_start, state_end, np
        )
        remaining = args.total_frames - args.previous_frames
        indices = _nearest_indices(selected, len(following) - 1, remaining, np)
        following_frames = [following[int(index)] for index in indices]
        camera_alignment = None
        if args.camera_align_frames:
            warp, correlation = _estimate_camera_alignment(
                previous[args.previous_frames - 1], following_frames[0], cv2, np
            )
            for offset in range(args.camera_align_frames):
                affine = _decayed_affine(warp, offset, args.camera_align_frames, np)
                following_frames[offset] = cv2.warpAffine(
                    following_frames[offset],
                    affine,
                    (following_frames[offset].shape[1], following_frames[offset].shape[0]),
                    flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            camera_alignment = {
                "frames": args.camera_align_frames,
                "ecc_correlation": correlation,
                "template_to_input_euclidean_warp": warp.tolist(),
                "decay": "linear_to_identity_in_camera:H3_output_pixels",
                "alpha_blending": False,
            }
        frames = previous[: args.previous_frames] + following_frames
        if len(frames) != args.total_frames:
            raise RuntimeError(f"candidate frame count mismatch for {label}")
        output = output_dir / "variants" / label / f"{label}-h3-continuous-10s.mp4"
        ffmpeg_command = _write_video(output, frames, args.fps, ffmpeg)

        seam_mad = float(
            np.mean(
                np.abs(
                    frames[args.previous_frames - 1].astype(np.float32)
                    - frames[args.previous_frames].astype(np.float32)
                )
            )
        )
        sharpness = [
            float(cv2.Laplacian(frame, cv2.CV_64F).var())
            for frame in frames
        ]
        storyboard_indices = (0, 60, args.previous_frames - 1, args.previous_frames, 160, 200, 239)
        storyboard = np.hstack([frames[index] for index in storyboard_indices])
        storyboard_path = output.parent / "storyboard.jpg"
        cv2.imwrite(str(storyboard_path), storyboard)
        record = {
                "label": label,
                "inputs": {
                    "previous": {"path": str(previous_path), "sha256": _sha256(previous_path)},
                    "following": {"path": str(following_path), "sha256": _sha256(following_path)},
                },
                "state_valid_following_interval_inclusive": [state_start, state_end],
                "selected_following_frame": selected,
                "seam_candidates": seam_candidates,
                "resampled_following_indices": indices.tolist(),
                "duplicated_following_frames": int(len(indices) - len(set(indices.tolist()))),
                "camera_alignment": camera_alignment,
                "seam_full_frame_mad": seam_mad,
                "median_laplacian_sharpness": float(np.median(sharpness)),
                "minimum_laplacian_sharpness": float(np.min(sharpness)),
                "output": str(output),
                "output_sha256": _sha256(output),
                "storyboard": str(storyboard_path),
                "storyboard_sha256": _sha256(storyboard_path),
                "ffmpeg_command": ffmpeg_command,
            }
        generation_metadata = {
            "schema_version": "1.0.0",
            "status": "succeeded" if args.human_review == "passed" else "review_required",
            "honest_status": "WORKING" if args.human_review == "passed" else "PARTIAL",
            "method": "task_state_valid_continuation_crop_with_nearest_frame_retiming",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "seed": 20260811,
            "coordinate_frame": "camera:H3_output_pixels",
            "human_review": args.human_review,
            "inputs": record["inputs"],
            "state_valid_following_interval_inclusive": [state_start, state_end],
            "selected_following_frame": selected,
            "postprocessing": {
                "frame_interpolation": False,
                "cross_dissolve": False,
                "blur": False,
                "alpha_repair": False,
                "source_person_restore": False,
                "retiming": "nearest decoded frame only",
                "camera_alignment": record["camera_alignment"],
            },
            "final_output": str(output),
            "final_output_sha256": record["output_sha256"],
            "storyboard": str(storyboard_path),
            "storyboard_sha256": record["storyboard_sha256"],
            "claim_boundary": "H3 camera-frame action visualization; not physical execution.",
        }
        metadata_path = output.parent / "metadata.json"
        metadata_path.write_text(
            json.dumps(generation_metadata, indent=2, sort_keys=True) + "\n"
        )
        record["generation_metadata"] = str(metadata_path)
        record["generation_metadata_sha256"] = _sha256(metadata_path)
        records.append(record)

    packages = {}
    for name in ("numpy", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": "1.0.0",
        "status": "accepted" if args.human_review == "passed" else "review_required",
        "honest_status": "WORKING" if args.human_review == "passed" else "PARTIAL",
        "method": "task_state_valid_continuation_crop_with_nearest_frame_retiming",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "gpu": {"used": False, "reason": "deterministic CPU stitching"},
        "seed": 20260811,
        "coordinate_frame": "camera:H3_output_pixels",
        "previous_frames": args.previous_frames,
        "total_frames": args.total_frames,
        "fps": args.fps,
        "postprocessing": {
            "frame_interpolation": False,
            "cross_dissolve": False,
            "blur": False,
            "source_person_restore": False,
            "camera_alignment": (
                "bounded decaying Euclidean resampling"
                if args.camera_align_frames
                else False
            ),
            "retiming": "nearest decoded frame only",
        },
        "actions": records,
        "acceptance": {
            "all_outputs_exactly_240_frames": True,
            "state_valid_intervals_require_human_review": True,
            "full_video_human_review": args.human_review,
            "accepted": args.human_review == "passed",
        },
        "limitations": [
            "Nearest-frame retiming can repeat frames and is not motion interpolation.",
            "A state-valid crop cannot prove physical task success or force correctness.",
            "The seam and complete timeline still require visual review before publication.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_dir), "actions": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
