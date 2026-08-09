#!/usr/bin/env python3
"""Create an explicit screen-space vendor-hand overlay on a real source video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from prepare_vendor_hand_target import (
    _parse_vector,
    _render_hand,
    _transform_foreground,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _apple_mask(cv2: object, np: object, frame: object) -> object:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (
        ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 170))
        & (hsv[:, :, 1] >= 145)
        & (hsv[:, :, 2] >= 30)
    ).astype(np.uint8)
    allowed = np.zeros(mask.shape, dtype=np.uint8)
    height, width = mask.shape
    allowed[round(height * 0.38) :, round(width * 0.2) : round(width * 0.8)] = 1
    mask = mask * allowed * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        raise RuntimeError("could not locate the apple")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def _skin_mask(cv2: object, np: object, frame: object, apple_center: tuple[int, int]) -> object:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin = (
        (hsv[:, :, 0] <= 25)
        & (hsv[:, :, 1] >= 20)
        & (hsv[:, :, 1] <= 185)
        & (hsv[:, :, 2] >= 45)
    )
    apple_x, apple_y = apple_center
    allowed = np.zeros(skin.shape, dtype=np.uint8)
    allowed[: min(frame.shape[0], apple_y + 110), : min(frame.shape[1], apple_x + 190)] = 1
    mask = (skin & allowed.astype(bool)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return cv2.dilate(mask, np.ones((21, 21), np.uint8))


def _draw_forearm(
    cv2: object,
    np: object,
    frame: object,
    apple_center: tuple[int, int],
    endpoint_offset: tuple[int, int],
    half_width: int,
) -> None:
    apple_x, apple_y = apple_center
    end_x = apple_x + endpoint_offset[0]
    end_y = apple_y + endpoint_offset[1]
    start_y = end_y - 18
    polygon = np.asarray(
        (
            (0, start_y - half_width),
            (end_x, end_y - half_width),
            (end_x + 20, end_y + half_width),
            (0, start_y + half_width + 12),
        ),
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, polygon, (72, 74, 78), lineType=cv2.LINE_AA)
    cv2.polylines(frame, [polygon], True, (30, 31, 34), 4, lineType=cv2.LINE_AA)
    cv2.line(
        frame,
        (0, start_y - half_width + 12),
        (end_x, end_y - half_width + 12),
        (132, 134, 138),
        3,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(frame, (end_x, end_y), half_width, (58, 60, 64), -1, cv2.LINE_AA)
    cv2.circle(frame, (end_x, end_y), half_width, (24, 25, 28), 4, cv2.LINE_AA)


def _overlay(
    cv2: object,
    np: object,
    background: object,
    foreground: object,
    mask: object,
    x: int,
    y: int,
) -> None:
    height, width = foreground.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(background.shape[1], x + width), min(background.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("vendor hand lies outside the video frame")
    source = foreground[y0 - y : y1 - y, x0 - x : x1 - x]
    alpha = mask[y0 - y : y1 - y, x0 - x : x1 - x]
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0).astype(np.float32)[:, :, None] / 255
    target = background[y0:y1, x0:x1]
    background[y0:y1, x0:x1] = (source * alpha + target * (1 - alpha)).astype(
        np.uint8
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-license", required=True)
    parser.add_argument("--camera", default="0,0,0.05,0.55,135,-30")
    parser.add_argument("--qpos", default="")
    parser.add_argument("--hand-width", type=int, required=True)
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument(
        "--apple-offset",
        required=True,
        help="comma-separated hand top-left offset from the tracked apple center",
    )
    parser.add_argument("--procedural-forearm", action="store_true")
    parser.add_argument("--forearm-end-offset", default="-150,-90")
    parser.add_argument("--forearm-half-width", type=int, default=52)
    args = parser.parse_args()

    source = args.source_video.expanduser().resolve()
    model = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for label, path in (("source video", source), ("model", model)):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")

    os.environ.setdefault("MUJOCO_GL", "egl")
    import cv2
    import mujoco
    import numpy as np

    camera = _parse_vector(args.camera, 6, "camera")
    qpos = tuple(float(item) for item in args.qpos.split(",")) if args.qpos else ()
    offset = tuple(int(item) for item in _parse_vector(args.apple_offset, 2, "apple-offset"))
    forearm_end_offset = tuple(
        int(item)
        for item in _parse_vector(args.forearm_end_offset, 2, "forearm-end-offset")
    )
    if args.forearm_half_width <= 0:
        raise ValueError("forearm-half-width must be positive")
    rendered, rendered_mask = _render_hand(
        mujoco, np, model, 640, 480, camera, qpos
    )
    hand, hand_mask = _transform_foreground(
        cv2, np, rendered, rendered_mask, args.hand_width, args.rotation_deg
    )

    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temporary = output.with_suffix(".temporary.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("failed to initialize temporary video writer")
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        original = frame.copy()
        apple = _apple_mask(cv2, np, original)
        moments = cv2.moments(apple)
        apple_center = (
            round(moments["m10"] / moments["m00"]),
            round(moments["m01"] / moments["m00"]),
        )
        skin = _skin_mask(cv2, np, original, apple_center)
        frame = cv2.inpaint(frame, skin, 7, cv2.INPAINT_TELEA)
        if args.procedural_forearm:
            _draw_forearm(
                cv2,
                np,
                frame,
                apple_center,
                forearm_end_offset,
                args.forearm_half_width,
            )
        _overlay(
            cv2,
            np,
            frame,
            hand,
            hand_mask,
            apple_center[0] + offset[0],
            apple_center[1] + offset[1],
        )
        apple_alpha = cv2.GaussianBlur(apple, (7, 7), 0).astype(np.float32)[:, :, None] / 255
        frame = (original * apple_alpha + frame * (1 - apple_alpha)).astype(np.uint8)
        writer.write(frame)
        frame_count += 1
    capture.release()
    writer.release()
    if frame_count == 0:
        raise RuntimeError("source video has no decodable frames")
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(temporary),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )
    temporary.unlink()

    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "deterministic_screen_space_vendor_hand_overlay",
        "vendor": args.vendor,
        "hostname": platform.node(),
        "command": [sys.executable, *sys.argv],
        "inputs": {
            "source_video": str(source),
            "source_sha256": _sha256(source),
            "model": str(model),
            "model_sha256": _sha256(model),
            "model_revision": args.model_revision,
            "model_license": args.model_license,
        },
        "configuration": {
            "camera": camera,
            "qpos": qpos,
            "hand_width": args.hand_width,
            "rotation_deg": args.rotation_deg,
            "apple_offset": offset,
            "procedural_forearm": args.procedural_forearm,
            "forearm_end_offset": forearm_end_offset,
            "forearm_half_width": args.forearm_half_width,
        },
        "output": str(output),
        "output_sha256": _sha256(output),
        "frames": frame_count,
        "fps": fps,
        "limitations": [
            "This is an explicitly labelled screen-space visualization, not model inference.",
            "The hand pose is rigid and follows the tracked apple; it is not a physics-valid grasp.",
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
