#!/usr/bin/env python3
"""Build a pixel-frame control sequence for a left-hand bowl and spoon placement."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--reference-time-s", type=float)
    return parser


def _ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3 - 2 * value)


def main() -> int:
    args = _parser().parse_args()
    if min(args.width, args.height, args.frames, args.fps) <= 0:
        raise ValueError("dimensions, frames, and FPS must be positive")
    if (args.frames - 1) % 4:
        raise ValueError("frames must satisfy 4n+1")
    reference_video = args.reference_video.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if not reference_video.is_file() or not ffmpeg.is_file():
        raise ValueError("reference video and ffmpeg must exist")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference = output / "reference.png"
    reference_command = [str(ffmpeg), "-v", "error", "-y"]
    if args.reference_time_s is None:
        reference_command.extend(["-sseof", "-0.05"])
    else:
        if args.reference_time_s < 0:
            raise ValueError("reference-time-s must be non-negative")
        reference_command.extend(["-ss", str(args.reference_time_s)])
    reference_command.extend(
        [
            "-i",
            str(reference_video),
            "-frames:v",
            "1",
            "-vf",
            f"scale={args.width}:{args.height}:flags=lanczos",
            str(reference),
        ]
    )
    subprocess.run(reference_command, check=True)

    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    background_edges = np.zeros((args.height, args.width), dtype=np.uint8)

    frames: list[bytes] = []
    timeline: list[dict[str, object]] = []
    spoon_start = (235.0, 224.0)
    bowl_final = (292.0, 218.0)
    bowl_slide_end = round((args.frames - 1) * 0.52)
    table_bottom_y = 241.0
    for index in range(args.frames):
        mask = Image.fromarray(background_edges).convert("L")
        draw = ImageDraw.Draw(mask)

        bowl_phase = _ease(index / bowl_slide_end)
        bowl_x = args.width + 62 + (bowl_final[0] - args.width - 62) * bowl_phase
        bowl_y = bowl_final[1]
        bowl_box = (
            round(bowl_x - 58),
            round(bowl_y - 23),
            round(bowl_x + 58),
            round(table_bottom_y),
        )
        draw.ellipse(bowl_box, outline=255, width=5)
        draw.arc(
            (
                bowl_box[0] + 8,
                bowl_box[1] + 7,
                bowl_box[2] - 8,
                bowl_box[3] - 2,
            ),
            5,
            175,
            fill=230,
            width=3,
        )

        left_shoulder = (346.0, 102.0)
        left_palm = (bowl_x + 48, bowl_y - 5)
        left_elbow = (
            left_shoulder[0] + (left_palm[0] - left_shoulder[0]) * 0.48 + 24,
            left_shoulder[1] + (left_palm[1] - left_shoulder[1]) * 0.48,
        )
        draw.line((*left_shoulder, *left_elbow, *left_palm), fill=255, width=18)
        for finger in range(4):
            offset = (finger - 1.5) * 7
            draw.line(
                (
                    left_palm[0],
                    left_palm[1] + offset,
                    bowl_x + 28,
                    bowl_y - 10 + offset / 2,
                ),
                fill=255,
                width=5,
            )

        if index <= bowl_slide_end:
            spoon_x, spoon_y = spoon_start
            action = "robot_left_hand_slides_bowl_on_table_while_right_hand_holds_spoon"
        else:
            phase = _ease((index - bowl_slide_end) / (args.frames - 1 - bowl_slide_end))
            spoon_x = spoon_start[0] + (bowl_x - spoon_start[0]) * phase
            spoon_y = spoon_start[1] + (bowl_y - spoon_start[1]) * phase
            action = "right_hand_moves_held_spoon_into_bowl"
        spoon_angle = -0.15 + 0.55 * _ease(
            max(0, index - bowl_slide_end) / (args.frames - 1 - bowl_slide_end)
        )
        handle_length = 76
        handle_end = (
            spoon_x - handle_length * math.cos(spoon_angle),
            spoon_y - handle_length * math.sin(spoon_angle),
        )
        draw.line((spoon_x, spoon_y, *handle_end), fill=255, width=8)
        draw.ellipse(
            (spoon_x - 14, spoon_y - 9, spoon_x + 14, spoon_y + 9),
            outline=255,
            width=5,
        )

        grasp = (handle_end[0] + 18, handle_end[1] - 4)
        right_palm = grasp
        right_shoulder = (105.0, 102.0)
        right_elbow = (
            right_shoulder[0] + (right_palm[0] - right_shoulder[0]) * 0.48 - 20,
            right_shoulder[1] + (right_palm[1] - right_shoulder[1]) * 0.48,
        )
        right_wrist = (
            right_shoulder[0] + (right_palm[0] - right_shoulder[0]) * 0.82,
            right_shoulder[1] + (right_palm[1] - right_shoulder[1]) * 0.82,
        )
        draw.line((*right_shoulder, *right_elbow, *right_wrist), fill=255, width=17)
        draw.line((right_wrist, right_palm), fill=255, width=17)
        for finger in range(4):
            offset = (finger - 1.5) * 6
            draw.line(
                (
                    right_palm[0],
                    right_palm[1] + offset,
                    grasp[0],
                    grasp[1] + offset / 3,
                ),
                fill=255,
                width=4,
            )

        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.7))
        rgb = np.repeat(np.asarray(mask)[..., None], 3, axis=2)
        frames.append(rgb.tobytes())
        timeline.append(
            {
                "frame": index,
                "time_s": index / args.fps,
                "action": action,
                "image_pixel_frame": {
                    "bowl_center_xy": [bowl_x, bowl_y],
                    "spoon_center_xy": [spoon_x, spoon_y],
                    "left_palm_xy": list(left_palm),
                    "right_palm_xy": list(right_palm),
                },
                "spoon_in_bowl": index == args.frames - 1,
                "right_hand_spoon_grasp_locked": True,
                "left_hand_bowl_contact_locked": True,
                "bowl_table_bottom_y": table_bottom_y,
            }
        )

    control = output / "control.mp4"
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
            f"{args.width}x{args.height}",
            "-r",
            str(args.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(control),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame)
    process.stdin.close()
    if process.wait():
        raise RuntimeError("ffmpeg failed to encode extension control")
    (output / "timeline.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "coordinate_frame": "image_pixel:448x256",
                "fps": args.fps,
                "frame_count": args.frames,
                "duration_s": args.frames / args.fps,
                "timeline": timeline,
                "acceptance": {
                    "bowl_present_frames": [8, args.frames - 1],
                    "right_hand_spoon_grasp_locked_frames": [0, args.frames - 1],
                    "left_hand_bowl_contact_locked_frames": [0, args.frames - 1],
                    "bowl_table_bottom_y": table_bottom_y,
                    "maximum_bowl_support_error_px": 0.0,
                    "bowl_motion_monotonic_right_to_left": True,
                    "final_spoon_in_bowl": True,
                },
                "limitations": [
                    "The control is screen-space animation, not a physics-verified trajectory.",
                    "The trained VACE model has not seen a real left-hand bowl sequence.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"CONTROL={control}")
    print(f"REFERENCE={reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
