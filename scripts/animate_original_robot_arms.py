#!/usr/bin/env python3
"""Animate robot-arm pixels extracted from one reference frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


def _ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value**3 * (value * (value * 6 - 15) + 10)


def _similarity_transform(
    layer: object,
    mask: object,
    anchor: tuple[float, float],
    source_tip: tuple[float, float],
    target_tip: tuple[float, float],
) -> tuple[object, object]:
    import numpy as np
    from PIL import Image

    source_vector = np.asarray(source_tip) - np.asarray(anchor)
    target_vector = np.asarray(target_tip) - np.asarray(anchor)
    source_length = float(np.linalg.norm(source_vector))
    target_length = float(np.linalg.norm(target_vector))
    if source_length <= 1e-6 or target_length <= 1e-6:
        raise ValueError("arm transform requires non-degenerate source and target vectors")
    scale = target_length / source_length
    source_angle = math.atan2(source_vector[1], source_vector[0])
    target_angle = math.atan2(target_vector[1], target_vector[0])
    angle = target_angle - source_angle
    cosine, sine = math.cos(angle), math.sin(angle)
    forward = scale * np.array(((cosine, -sine), (sine, cosine)))
    inverse = np.linalg.inv(forward)
    anchor_array = np.asarray(anchor)
    translation = anchor_array - inverse @ anchor_array
    coefficients = (
        float(inverse[0, 0]),
        float(inverse[0, 1]),
        float(translation[0]),
        float(inverse[1, 0]),
        float(inverse[1, 1]),
        float(translation[1]),
    )
    transformed_layer = layer.transform(
        layer.size,
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
    )
    transformed_mask = mask.transform(
        mask.size,
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BILINEAR,
    )
    return transformed_layer, transformed_mask


def _encode_video(
    frames: list[object],
    output: Path,
    ffmpeg: Path,
    fps: int,
) -> None:
    process = subprocess.Popen(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{frames[0].width}x{frames[0].height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed to encode {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if min(args.frames, args.fps) <= 0 or (args.frames - 1) % 4:
        raise ValueError("frames must satisfy 4n+1 and FPS must be positive")
    reference_path = args.reference.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if not reference_path.is_file() or not ffmpeg.is_file():
        raise ValueError("reference and ffmpeg must exist")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    from scipy.ndimage import distance_transform_edt

    reference = Image.open(reference_path).convert("RGB")
    if reference.size != (448, 256):
        raise ValueError("original-arm animator expects a 448x256 image-pixel frame")

    left_mask = Image.new("L", reference.size, 0)
    ImageDraw.Draw(left_mask).polygon(
        [
            (264, 78),
            (299, 80),
            (311, 115),
            (326, 146),
            (350, 202),
            (316, 207),
            (296, 165),
            (283, 127),
            (261, 96),
        ],
        fill=255,
    )
    right_mask = Image.new("L", reference.size, 0)
    ImageDraw.Draw(right_mask).polygon(
        [
            (140, 78),
            (174, 76),
            (171, 113),
            (145, 145),
            (177, 181),
            (204, 230),
            (158, 232),
            (137, 190),
            (109, 157),
            (119, 112),
        ],
        fill=255,
    )
    pixels = np.asarray(reference)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    spoon_color = (
        (green.astype(int) - red.astype(int) > 8)
        & (blue.astype(int) - red.astype(int) > 6)
        & (np.indices(red.shape)[0] > 185)
        & (np.indices(red.shape)[1] > 150)
    )
    spoon_mask = Image.fromarray((spoon_color * 255).astype(np.uint8)).filter(
        ImageFilter.MaxFilter(9)
    )
    right_mask = Image.fromarray(
        np.maximum(np.asarray(right_mask), np.asarray(spoon_mask)).astype(np.uint8)
    )
    left_mask = left_mask.filter(ImageFilter.GaussianBlur(1.5))
    right_mask = right_mask.filter(ImageFilter.GaussianBlur(1.5))

    combined = np.maximum(np.asarray(left_mask), np.asarray(right_mask)) >= 96
    source_array = np.asarray(reference)
    _, nearest = distance_transform_edt(combined, return_indices=True)
    background = source_array.copy()
    background[combined] = source_array[nearest[0][combined], nearest[1][combined]]
    background_image = Image.fromarray(background)

    left_layer = reference.copy()
    right_layer = reference.copy()
    left_anchor = (281.0, 88.0)
    left_tip = (333.0, 194.0)
    right_anchor = (155.0, 88.0)
    right_tip = (272.0, 218.0)
    bowl_final = (292.0, 218.0)
    bowl_slide_end = round((args.frames - 1) * 0.52)
    frames: list[object] = []
    timeline: list[dict[str, object]] = []

    for frame_index in range(args.frames):
        slide_phase = _ease(frame_index / bowl_slide_end)
        bowl_x = 340 + (bowl_final[0] - 340) * slide_phase
        bowl_y = bowl_final[1]
        left_target = (bowl_x + 49, bowl_y - 7)
        left_transformed, left_alpha = _similarity_transform(
            left_layer,
            left_mask,
            left_anchor,
            left_tip,
            left_target,
        )
        if frame_index <= bowl_slide_end:
            spoon_phase = 0.0
        else:
            spoon_phase = _ease(
                (frame_index - bowl_slide_end)
                / (args.frames - 1 - bowl_slide_end)
            )
        spoon_target = (
            right_tip[0] + (bowl_x - right_tip[0]) * spoon_phase,
            right_tip[1] + (bowl_y - right_tip[1]) * spoon_phase,
        )
        right_transformed, right_alpha = _similarity_transform(
            right_layer,
            right_mask,
            right_anchor,
            right_tip,
            spoon_target,
        )

        frame = background_image.copy()
        bowl_back = Image.new("RGBA", reference.size, (0, 0, 0, 0))
        bowl_draw = ImageDraw.Draw(bowl_back)
        bowl_box = (
            round(bowl_x - 56),
            round(bowl_y - 22),
            round(bowl_x + 56),
            241,
        )
        bowl_draw.ellipse(bowl_box, fill=(20, 24, 29, 255), outline=(110, 120, 128, 255), width=3)
        bowl_draw.ellipse(
            (bowl_box[0] + 7, bowl_box[1] + 5, bowl_box[2] - 7, bowl_box[3] - 6),
            fill=(7, 10, 13, 255),
            outline=(150, 160, 166, 255),
            width=2,
        )
        frame = Image.alpha_composite(frame.convert("RGBA"), bowl_back)
        frame.paste(left_transformed, (0, 0), left_alpha)
        frame.paste(right_transformed, (0, 0), right_alpha)

        bowl_front = Image.new("RGBA", reference.size, (0, 0, 0, 0))
        front_draw = ImageDraw.Draw(bowl_front)
        front_draw.pieslice(
            bowl_box,
            0,
            180,
            fill=(30, 35, 41, 255),
            outline=(145, 155, 162, 255),
            width=2,
        )
        frame = Image.alpha_composite(frame, bowl_front).convert("RGB")
        frames.append(frame)
        timeline.append(
            {
                "frame": frame_index,
                "time_s": frame_index / args.fps,
                "image_pixel_frame": {
                    "left_shoulder_xy": list(left_anchor),
                    "left_arm_contact_xy": list(left_target),
                    "bowl_center_xy": [bowl_x, bowl_y],
                    "right_shoulder_xy": list(right_anchor),
                    "right_spoon_tip_xy": list(spoon_target),
                },
                "bowl_table_bottom_y": 241,
                "left_arm_source_pixels_only": True,
                "right_arm_and_spoon_source_pixels_only": True,
            }
        )

    video = output / "original-arms-table-slide.mp4"
    _encode_video(frames, video, ffmpeg, args.fps)
    (output / "timeline.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "coordinate_frame": "image_pixel:448x256",
                "frame_count": args.frames,
                "fps": args.fps,
                "timeline": timeline,
                "invariants": {
                    "original_left_arm_pixels_only": True,
                    "original_right_arm_and_spoon_pixels_only": True,
                    "bowl_table_bottom_y": 241,
                    "bowl_monotonic_right_to_left": True,
                    "spoon_moves_only_after_bowl_stops": True,
                },
                "limitations": [
                    "Two-dimensional similarity transforms are not robot joint kinematics.",
                    "The bowl is procedurally rendered and has no contact dynamics.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    provenance = {
        "reference": str(reference_path),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "output": str(video),
        "output_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "method": "deterministic_original_robot_arm_pixel_animation",
    }
    (output / "manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    left_mask.save(output / "left-arm-mask.png")
    right_mask.save(output / "right-arm-spoon-mask.png")
    Image.fromarray(background).save(output / "background-with-arms-removed.png")
    print(f"VIDEO={video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
