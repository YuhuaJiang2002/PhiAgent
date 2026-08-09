#!/usr/bin/env python3
"""Composite a licensed MuJoCo hand into a static real-camera source scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_vector(value: str, expected: int, label: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(","))
    if len(result) != expected:
        raise ValueError(f"{label} must have {expected} comma-separated values")
    return result


def _median_background(cv2: object, np: object, video: Path) -> tuple[object, object]:
    capture = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"source video has no decodable frames: {video}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8), frames[0]


def _render_hand(
    mujoco: object,
    np: object,
    model_path: Path,
    width: int,
    height: int,
    camera: tuple[float, float, float, float, float, float],
    qpos: tuple[float, ...],
) -> tuple[object, object]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if qpos:
        if len(qpos) > model.nq:
            raise ValueError(f"grasp qpos has {len(qpos)} values but model nq is {model.nq}")
        data.qpos[: len(qpos)] = qpos
    mujoco.mj_forward(model, data)

    look_x, look_y, look_z, distance, azimuth, elevation = camera
    view = mujoco.MjvCamera()
    view.type = mujoco.mjtCamera.mjCAMERA_FREE
    view.lookat[:] = (look_x, look_y, look_z)
    view.distance = distance
    view.azimuth = azimuth
    view.elevation = elevation
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=view)
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=view)
    segmentation = renderer.render().copy()
    renderer.close()

    # Scene models contain floor geom 0 and one final demonstration-object geom.
    mask = (
        (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        & (segmentation[:, :, 0] > 0)
        & (segmentation[:, :, 0] < model.ngeom - 1)
    )
    if int(mask.sum()) < 100:
        raise RuntimeError("segmented hand render is unexpectedly empty")
    return rgb[:, :, ::-1], mask.astype(np.uint8) * 255


def _transform_foreground(
    cv2: object,
    np: object,
    image: object,
    mask: object,
    output_width: int,
    angle_deg: float,
) -> tuple[object, object]:
    ys, xs = np.nonzero(mask)
    image = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    mask = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    scale = output_width / image.shape[1]
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    center = (image.shape[1] / 2, image.shape[0] / 2)
    rotation = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cosine, sine = abs(rotation[0, 0]), abs(rotation[0, 1])
    rotated_width = int(image.shape[0] * sine + image.shape[1] * cosine)
    rotated_height = int(image.shape[0] * cosine + image.shape[1] * sine)
    rotation[0, 2] += rotated_width / 2 - center[0]
    rotation[1, 2] += rotated_height / 2 - center[1]
    size = (rotated_width, rotated_height)
    return (
        cv2.warpAffine(image, rotation, size, flags=cv2.INTER_LANCZOS4),
        cv2.warpAffine(mask, rotation, size, flags=cv2.INTER_NEAREST),
    )


def _overlay(cv2: object, np: object, background: object, image: object, mask: object, x: int, y: int) -> None:
    height, width = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(background.shape[1], x + width), min(background.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("hand placement lies outside the output image")
    foreground = image[y0 - y : y1 - y, x0 - x : x1 - x]
    alpha = mask[y0 - y : y1 - y, x0 - x : x1 - x]
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0).astype(np.float32)[:, :, None] / 255.0
    region = background[y0:y1, x0:x1]
    background[y0:y1, x0:x1] = (foreground * alpha + region * (1 - alpha)).astype(
        np.uint8
    )


def _remove_source_hand(cv2: object, np: object, background: object, first_frame: object) -> object:
    hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
    skin = (
        (hsv[:, :, 0] <= 25)
        & (hsv[:, :, 1] >= 20)
        & (hsv[:, :, 1] <= 180)
        & (hsv[:, :, 2] >= 45)
    )
    allowed = np.zeros(skin.shape, dtype=np.uint8)
    allowed[: round(skin.shape[0] * 0.82), : round(skin.shape[1] * 0.72)] = 1
    mask = (skin & allowed.astype(bool)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    mask = cv2.dilate(mask, np.ones((23, 23), np.uint8))
    return cv2.inpaint(background, mask, 9, cv2.INPAINT_TELEA)


def _apple_mask(cv2: object, np: object, frame: object, roi: tuple[int, int, int, int]) -> object:
    x0, y0, x1, y1 = roi
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = (
        ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 170))
        & (hsv[:, :, 1] >= 150)
        & (hsv[:, :, 2] >= 35)
    )
    bounded = np.zeros(red.shape, dtype=np.uint8)
    bounded[y0:y1, x0:x1] = red[y0:y1, x0:x1].astype(np.uint8) * 255
    bounded = cv2.morphologyEx(
        bounded, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8)
    )
    bounded = cv2.morphologyEx(
        bounded, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )
    if int((bounded > 0).sum()) < 500:
        raise RuntimeError("apple segmentation is unexpectedly empty")
    return bounded


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
    parser.add_argument("--hand-width", type=int, default=520)
    parser.add_argument("--placement", default="20,90")
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--apple-roi", default="0.36,0.55,0.22,0.36")
    args = parser.parse_args()
    if args.hand_width <= 0:
        raise ValueError("hand-width must be positive")
    source = args.source_video.expanduser().resolve()
    model = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for label, path in (("source video", source), ("model", model)):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")
    import cv2
    import mujoco
    import numpy as np

    background, first_frame = _median_background(cv2, np, source)
    background = _remove_source_hand(cv2, np, background, first_frame)
    camera = _parse_vector(args.camera, 6, "camera")
    qpos = tuple(float(item) for item in args.qpos.split(",")) if args.qpos else ()
    placement = tuple(int(item) for item in _parse_vector(args.placement, 2, "placement"))
    apple_roi = _parse_vector(args.apple_roi, 4, "apple-roi")
    rendered, rendered_mask = _render_hand(
        mujoco, np, model, 640, 480, camera, qpos
    )
    hand, hand_mask = _transform_foreground(
        cv2, np, rendered, rendered_mask, args.hand_width, args.rotation_deg
    )
    _overlay(cv2, np, background, hand, hand_mask, *placement)

    x, y, width, height = apple_roi
    frame_height, frame_width = background.shape[:2]
    x0, y0 = round(x * frame_width), round(y * frame_height)
    x1, y1 = round((x + width) * frame_width), round((y + height) * frame_height)
    if not 0 <= x0 < x1 <= frame_width or not 0 <= y0 < y1 <= frame_height:
        raise ValueError("apple-roi is outside the source frame")
    apple_mask = _apple_mask(cv2, np, first_frame, (x0, y0, x1, y1))
    apple_mask = cv2.GaussianBlur(apple_mask, (15, 15), 0).astype(np.float32)[:, :, None] / 255
    background[:] = (
        first_frame * apple_mask + background * (1 - apple_mask)
    ).astype(np.uint8)
    if not cv2.imwrite(str(output), background):
        raise RuntimeError(f"failed to write target image: {output}")

    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "segmented_mujoco_hand_on_median_real_scene",
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
            "placement": placement,
            "rotation_deg": args.rotation_deg,
            "apple_roi": apple_roi,
        },
        "output": str(output),
        "output_sha256": _sha256(output),
        "limitations": [
            "The target is a deterministic visual composite, not a captured robot scene.",
            "Median background reconstruction and screen-space placement are not camera-calibrated.",
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
