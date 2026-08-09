#!/usr/bin/env python3
"""Extend a clean robot video with masked bowl/hand and spoon instances."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sprite(
    image_path: Path, mask_path: Path, *, mirror: bool
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or image.shape[:2] != mask.shape:
        raise ValueError(f"invalid sprite image or mask: {image_path}, {mask_path}")
    points = cv2.findNonZero((mask >= 128).astype(np.uint8))
    if points is None:
        raise ValueError(f"sprite mask is empty: {mask_path}")
    x, y, width, height = cv2.boundingRect(points)
    image = image[y : y + height, x : x + width]
    mask = mask[y : y + height, x : x + width]
    if mirror:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    alpha = cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) / 255.0
    return image, alpha, (x, y, width, height)


def _transform(
    image: np.ndarray,
    alpha: np.ndarray,
    *,
    scale: float,
    angle_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    if abs(angle_deg) < 1e-6:
        return image, alpha
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    rotated_width = max(1, round(height * sine + width * cosine))
    rotated_height = max(1, round(height * cosine + width * sine))
    matrix[0, 2] += rotated_width / 2 - center[0]
    matrix[1, 2] += rotated_height / 2 - center[1]
    return (
        cv2.warpAffine(
            image,
            matrix,
            (rotated_width, rotated_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ),
        cv2.warpAffine(
            alpha,
            matrix,
            (rotated_width, rotated_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ),
    )


def _overlay(
    frame: np.ndarray,
    image: np.ndarray,
    alpha: np.ndarray,
    x: int,
    y: int,
    *,
    opacity: float = 1.0,
) -> None:
    x0, y0 = max(0, x), max(0, y)
    x1 = min(frame.shape[1], x + image.shape[1])
    y1 = min(frame.shape[0], y + image.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    sx0, sy0 = x0 - x, y0 - y
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    selected_alpha = np.clip(
        alpha[sy0:sy1, sx0:sx1, None] * opacity, 0.0, 1.0
    )
    frame[y0:y1, x0:x1] = np.rint(
        selected_alpha * image[sy0:sy1, sx0:sx1]
        + (1 - selected_alpha) * frame[y0:y1, x0:x1]
    ).astype(np.uint8)


def _ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3 - 2 * value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-video", type=Path, required=True)
    parser.add_argument("--bowl-image", type=Path, required=True)
    parser.add_argument("--bowl-mask", type=Path, required=True)
    parser.add_argument("--carrier-hand-image", type=Path)
    parser.add_argument("--carrier-hand-mask", type=Path)
    parser.add_argument("--spoon-image", type=Path, required=True)
    parser.add_argument("--spoon-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--extension-seconds", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--clean-table-time", type=float, default=1.2)
    parser.add_argument(
        "--bowl-entry-side",
        choices=("left", "right"),
        default="left",
    )
    args = parser.parse_args()
    for path in (
        args.base_video,
        args.bowl_image,
        args.bowl_mask,
        args.spoon_image,
        args.spoon_mask,
    ):
        if not path.is_file():
            raise ValueError(f"input does not exist: {path}")
    if (args.carrier_hand_image is None) != (args.carrier_hand_mask is None):
        raise ValueError("carrier hand image and mask must be supplied together")
    for path in (args.carrier_hand_image, args.carrier_hand_mask):
        if path is not None and not path.is_file():
            raise ValueError(f"carrier hand input does not exist: {path}")
    for path in (args.output, args.frames_dir, args.metadata):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")
    if args.extension_seconds <= 0 or args.fps <= 0 or args.clean_table_time < 0:
        raise ValueError("duration, FPS, and clean-table time are invalid")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError("ffmpeg or imageio-ffmpeg is required") from exc

    capture = cv2.VideoCapture(str(args.base_video))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"base video has no frames: {args.base_video}")
    height, width = frames[-1].shape[:2]
    same_robot_hand = args.carrier_hand_image is not None
    entry_from_right = args.bowl_entry_side == "right"
    bowl, bowl_alpha, _ = _sprite(
        args.bowl_image,
        args.bowl_mask,
        mirror=not (same_robot_hand or entry_from_right),
    )
    bowl, bowl_alpha = _transform(bowl, bowl_alpha, scale=0.58)
    carrier = carrier_alpha = None
    if same_robot_hand:
        assert args.carrier_hand_image is not None
        assert args.carrier_hand_mask is not None
        carrier, carrier_alpha, _ = _sprite(
            args.carrier_hand_image,
            args.carrier_hand_mask,
            mirror=True,
        )
        carrier, carrier_alpha = _transform(carrier, carrier_alpha, scale=0.78)
    spoon, spoon_alpha, spoon_box = _sprite(
        args.spoon_image, args.spoon_mask, mirror=False
    )

    extension_count = round(args.extension_seconds * args.fps)
    base = frames[-1].copy()
    source_spoon_mask = cv2.imread(str(args.spoon_mask), cv2.IMREAD_GRAYSCALE)
    if source_spoon_mask is None:
        raise ValueError(f"could not read spoon mask: {args.spoon_mask}")
    clean_index = min(
        len(frames) - 1,
        round(args.clean_table_time * args.fps),
    )
    clean_table_frame = frames[clean_index]
    replacement_mask = cv2.dilate(
        (source_spoon_mask >= 128).astype(np.uint8) * 255,
        np.ones((15, 15), dtype=np.uint8),
        iterations=1,
    )
    replacement_alpha = (
        cv2.GaussianBlur(replacement_mask, (0, 0), 7.0).astype(np.float32)
        / 255.0
    )[:, :, None]
    base = np.rint(
        replacement_alpha * clean_table_frame
        + (1 - replacement_alpha) * base
    ).astype(np.uint8)
    extension: list[np.ndarray] = []
    final_bowl_x = 260 if same_robot_hand else (330 if entry_from_right else 205)
    final_bowl_y = 310 if same_robot_hand else 300
    for index in range(extension_count):
        frame = base.copy()
        arrival = _ease(index / max(1, round(1.45 * args.fps)))
        bowl_start_x = (
            width if same_robot_hand or entry_from_right else -bowl.shape[1]
        )
        bowl_x = round(bowl_start_x + arrival * (final_bowl_x - bowl_start_x))
        bowl_y = round(330 + arrival * (final_bowl_y - 330))
        carrier_x = carrier_y = 0
        if same_robot_hand:
            assert carrier is not None
            carrier_start_x = width + 80
            carrier_x = round(carrier_start_x + arrival * (465 - carrier_start_x))
            carrier_y = round(280 + arrival * (225 - 280))

        insert_start = round(1.55 * args.fps)
        insert_end = round(2.65 * args.fps)
        insertion = _ease((index - insert_start) / max(1, insert_end - insert_start))
        transformed_spoon, transformed_alpha = _transform(
            spoon,
            spoon_alpha,
            scale=1.0 - 0.28 * insertion,
            angle_deg=-8 * insertion,
        )
        spoon_target_x = final_bowl_x + 20
        spoon_target_y = final_bowl_y + 20
        spoon_x = round(
            spoon_box[0] + insertion * (spoon_target_x - spoon_box[0])
        )
        spoon_y = round(
            spoon_box[1] + insertion * (spoon_target_y - spoon_box[1])
        )
        _overlay(
            frame,
            transformed_spoon,
            transformed_alpha,
            spoon_x,
            spoon_y,
            opacity=1.0 - 0.25 * insertion,
        )

        _overlay(frame, bowl, bowl_alpha, bowl_x, bowl_y)
        if same_robot_hand:
            assert carrier is not None and carrier_alpha is not None
            _overlay(frame, carrier, carrier_alpha, carrier_x, carrier_y)
        extension.append(frame)

    all_frames = frames + extension
    args.frames_dir.mkdir(parents=True)
    for index, frame in enumerate(all_frames):
        cv2.imwrite(str(args.frames_dir / f"{index:06d}.png"), frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-framerate",
        str(args.fps),
        "-i",
        str(args.frames_dir / "%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    subprocess.run(command, check=True)
    args.metadata.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "base_video": str(args.base_video.resolve()),
                "base_video_sha256": _sha256(args.base_video),
                "bowl_image": str(args.bowl_image.resolve()),
                "bowl_mask": str(args.bowl_mask.resolve()),
                "carrier_hand_image": (
                    str(args.carrier_hand_image.resolve())
                    if args.carrier_hand_image is not None
                    else None
                ),
                "carrier_hand_mask": (
                    str(args.carrier_hand_mask.resolve())
                    if args.carrier_hand_mask is not None
                    else None
                ),
                "spoon_image": str(args.spoon_image.resolve()),
                "spoon_mask": str(args.spoon_mask.resolve()),
                "output": str(args.output.resolve()),
                "output_sha256": _sha256(args.output),
                "base_frames": len(frames),
                "extension_frames": extension_count,
                "total_frames": len(all_frames),
                "fps": args.fps,
                "duration_s": len(all_frames) / args.fps,
                "action_timing": {
                    "bowl_arrival_s": [0.0, 1.45],
                    "spoon_insertion_s": [1.55, 2.65],
                },
                "clean_table_time_s": args.clean_table_time,
                "bowl_entry_side": args.bowl_entry_side,
                "method": (
                    "confidence-routed clean base with SAM2 bowl, same-robot "
                    "mirrored left-hand, and spoon instance overlays"
                ),
                "limitations": [
                    "The extension is deterministic 2D compositing, not a generated or physical robot execution.",
                    "The bowl source uses a different camera and is scaled into the case-1 scene.",
                ],
                "encode_command": command,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"OUTPUT={args.output.resolve()}")
    print(f"METADATA={args.metadata.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
