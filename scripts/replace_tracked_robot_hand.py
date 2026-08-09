#!/usr/bin/env python3
"""Replace a tracked generated hand with a native robot-hand template."""

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


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames or fps <= 0:
        raise ValueError(f"invalid video: {path}")
    return frames, fps


def _crop_template(
    image_path: Path,
    mask_path: Path,
    *,
    x_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or image.shape[:2] != mask.shape:
        raise ValueError("template image and mask must be valid and aligned")
    points = cv2.findNonZero((mask >= 128).astype(np.uint8))
    if points is None:
        raise ValueError("template mask is empty")
    x, y, width, height = cv2.boundingRect(points)
    selected_width = max(1, round(width * x_fraction))
    return (
        image[y : y + height, x : x + selected_width],
        mask[y : y + height, x : x + selected_width],
    )


def _smooth_boxes(
    boxes: list[tuple[int, int, int, int] | None],
    radius: int = 2,
) -> list[tuple[int, int, int, int] | None]:
    smoothed: list[tuple[int, int, int, int] | None] = []
    for index, box in enumerate(boxes):
        if box is None:
            smoothed.append(None)
            continue
        nearby = [
            candidate
            for candidate in boxes[max(0, index - radius) : index + radius + 1]
            if candidate is not None
        ]
        smoothed.append(
            tuple(
                int(round(float(np.median([candidate[field] for candidate in nearby]))))
                for field in range(4)
            )
        )
    return smoothed


def _overlay(
    frame: np.ndarray,
    image: np.ndarray,
    alpha: np.ndarray,
    x: int,
    y: int,
) -> None:
    x0, y0 = max(0, x), max(0, y)
    x1 = min(frame.shape[1], x + image.shape[1])
    y1 = min(frame.shape[0], y + image.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    sx0, sy0 = x0 - x, y0 - y
    sx1, sy1 = sx0 + x1 - x0, sy0 + y1 - y0
    selected = alpha[sy0:sy1, sx0:sx1, None]
    frame[y0:y1, x0:x1] = np.rint(
        selected * image[sy0:sy1, sx0:sx1]
        + (1 - selected) * frame[y0:y1, x0:x1]
    ).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hand-mask-video", type=Path, required=True)
    parser.add_argument("--template-image", type=Path, required=True)
    parser.add_argument("--template-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-mask-video", type=Path)
    parser.add_argument("--scale", type=float, default=1.12)
    parser.add_argument("--template-x-fraction", type=float, default=1.0)
    args = parser.parse_args()
    for path in (
        args.video,
        args.hand_mask_video,
        args.template_image,
        args.template_mask,
    ):
        if not path.is_file():
            raise ValueError(f"input does not exist: {path}")
    for path in (args.output, args.metadata, args.frames_dir, args.output_mask_video):
        if path is None:
            continue
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")
    if args.scale <= 0 or not 0 < args.template_x_fraction <= 1:
        raise ValueError("template scale and x fraction are invalid")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError("ffmpeg or imageio-ffmpeg is required") from exc

    frames, fps = _read_video(args.video)
    mask_frames, mask_fps = _read_video(args.hand_mask_video)
    if abs(fps - mask_fps) > 0.1 or len(mask_frames) < len(frames):
        raise ValueError("hand mask video must cover the source video at the same FPS")
    template, template_mask = _crop_template(
        args.template_image,
        args.template_mask,
        x_fraction=args.template_x_fraction,
    )
    boxes: list[tuple[int, int, int, int] | None] = []
    binary_masks: list[np.ndarray] = []
    frame_height, frame_width = frames[0].shape[:2]
    for mask_frame in mask_frames[: len(frames)]:
        if mask_frame.shape[:2] != (frame_height, frame_width):
            mask_frame = cv2.resize(
                mask_frame,
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            )
        gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        binary = (gray >= 128).astype(np.uint8)
        binary_masks.append(binary)
        points = cv2.findNonZero(binary)
        boxes.append(cv2.boundingRect(points) if points is not None else None)
    boxes = _smooth_boxes(boxes)

    output_frames: list[np.ndarray] = []
    output_masks: list[np.ndarray] = []
    for frame, binary, box in zip(frames, binary_masks, boxes):
        output = frame.copy()
        rendered_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if box is None:
            output_frames.append(output)
            output_masks.append(rendered_mask)
            continue
        x, y, width, height = box
        removal = cv2.dilate(
            binary * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=1,
        )
        output = cv2.inpaint(output, removal, 5, cv2.INPAINT_TELEA)
        target_width = max(1, round(width * args.scale))
        target_height = max(1, round(height * args.scale))
        resized = cv2.resize(
            template, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        alpha = cv2.resize(
            template_mask,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.4).astype(np.float32) / 255.0
        place_x = round(x + width - target_width)
        place_y = round(y + (height - target_height) / 2)
        _overlay(output, resized, alpha, place_x, place_y)
        x0, y0 = max(0, place_x), max(0, place_y)
        x1 = min(frame.shape[1], place_x + target_width)
        y1 = min(frame.shape[0], place_y + target_height)
        if x0 < x1 and y0 < y1:
            sx0, sy0 = x0 - place_x, y0 - place_y
            sx1, sy1 = sx0 + x1 - x0, sy0 + y1 - y0
            rendered_mask[y0:y1, x0:x1] = (
                alpha[sy0:sy1, sx0:sx1] >= 0.25
            ).astype(np.uint8) * 255
        output_frames.append(output)
        output_masks.append(rendered_mask)

    args.frames_dir.mkdir(parents=True)
    for index, frame in enumerate(output_frames):
        cv2.imwrite(str(args.frames_dir / f"{index:06d}.png"), frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-framerate",
        f"{fps:.8g}",
        "-i",
        str(args.frames_dir / "%06d.png"),
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        str(args.output),
    ]
    subprocess.run(command, check=True)
    mask_command = None
    if args.output_mask_video is not None:
        mask_frames_dir = args.frames_dir / "masks"
        mask_frames_dir.mkdir()
        for index, mask in enumerate(output_masks):
            cv2.imwrite(str(mask_frames_dir / f"{index:06d}.png"), mask)
        args.output_mask_video.parent.mkdir(parents=True, exist_ok=True)
        mask_command = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-framerate",
            f"{fps:.8g}",
            "-i",
            str(mask_frames_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(args.output_mask_video),
        ]
        subprocess.run(mask_command, check=True)
    args.metadata.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "method": "SAM2 tracked hand replacement with official Sunday template",
                "video": str(args.video.resolve()),
                "video_sha256": _sha256(args.video),
                "hand_mask_video": str(args.hand_mask_video.resolve()),
                "hand_mask_sha256": _sha256(args.hand_mask_video),
                "template_image": str(args.template_image.resolve()),
                "template_mask": str(args.template_mask.resolve()),
                "output": str(args.output.resolve()),
                "output_sha256": _sha256(args.output),
                "frame_count": len(output_frames),
                "fps": fps,
                "scale": args.scale,
                "template_x_fraction": args.template_x_fraction,
                "output_mask_video": (
                    str(args.output_mask_video.resolve())
                    if args.output_mask_video is not None
                    else None
                ),
                "limitations": [
                    "The native hand is a 2D official-image template, not a rerendered 3D hand.",
                    "Inpainting and template warping are image-space operations.",
                ],
                "encode_command": command,
                "mask_encode_command": mask_command,
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
