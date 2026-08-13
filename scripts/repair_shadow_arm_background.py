#!/usr/bin/env python3
"""Repair the background and human-arm cleanup in the Shadow Hand showcase.

The original compositor used image inpainting over a mask that sometimes touched
the white strip above the blue-grey wall.  The inpainting consequently pulled
white pixels down into the wall.  This repair keeps source pixels outside an
expanded human-removal mask, fills the mask only from samples on the same image
row, and then re-composites the saved robot RGB/mask layers.

This is deliberately a standalone, CPU-only script.  Pillow and NumPy are
optional runtime dependencies; importing :mod:`phiagent` does not require them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--current-composite", type=Path, required=True)
    parser.add_argument("--replacement-mask", type=Path, required=True)
    parser.add_argument("--robot-rgb", type=Path, required=True)
    parser.add_argument("--robot-mask", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--showcase-template", type=Path)
    parser.add_argument("--showcase-output", type=Path)
    parser.add_argument("--mask-dilation", type=int, default=14)
    parser.add_argument("--wrist-half-width", type=float, default=150.0)
    parser.add_argument("--base-half-width", type=float, default=135.0)
    parser.add_argument("--target-hand-length", type=float, default=148.0)
    parser.add_argument("--target-forearm-half-width", type=float, default=42.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def probe(path: Path) -> VideoInfo:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=float(numerator) / float(denominator),
        frames=int(stream.get("nb_frames") or stream["nb_read_frames"]),
        duration=float(payload["format"]["duration"]),
    )


def reader(path: Path, pixel_format: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            pixel_format,
            "-",
        ],
        stdout=subprocess.PIPE,
    )


def writer(
    path: Path, width: int, height: int, fps: float, pixel_format: str, codec: str
) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pixel_format,
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        codec,
    ]
    if codec == "libx264rgb":
        command.extend(["-preset", "medium", "-crf", "0", "-movflags", "+faststart"])
    elif codec == "ffv1":
        command.extend(["-level", "3"])
    command.append(str(path))
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def read_exact(stream: BinaryIO, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            if not chunks:
                return None
            raise RuntimeError(f"truncated raw frame: expected {size}, got {len(chunks)}")
        chunks.extend(chunk)
    return bytes(chunks)


def binary_dilate(mask: "np.ndarray", radius: int) -> "np.ndarray":
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask.astype(np.uint8), ((radius, radius), (radius, radius)))
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    kernel = 2 * radius + 1
    totals = (
        integral[kernel:, kernel:]
        - integral[:-kernel, kernel:]
        - integral[kernel:, :-kernel]
        + integral[:-kernel, :-kernel]
    )
    return totals > 0


def polygon_mask(
    width: int,
    height: int,
    wrist: "np.ndarray",
    base: "np.ndarray",
    wrist_half_width: float,
    base_half_width: float,
) -> "np.ndarray":
    axis = base - wrist
    length = float(np.linalg.norm(axis))
    if length < 10:
        raise ValueError(f"invalid human forearm axis length: {length}")
    axis /= length
    perpendicular = np.array([-axis[1], axis[0]], dtype=np.float64)
    vertices = np.rint(
        np.array(
            [
                wrist + perpendicular * wrist_half_width,
                wrist - perpendicular * wrist_half_width,
                base - perpendicular * base_half_width,
                base + perpendicular * base_half_width,
            ]
        )
    ).astype(int)
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon([tuple(point) for point in vertices], fill=255)
    return np.asarray(image) > 0


def horizontal_background_fill(
    source: "np.ndarray",
    repair_mask: "np.ndarray",
    sample_width: int = 16,
    vertical_sample_radius: int = 3,
    lower_vertical_sample_radius: int = 35,
    lower_wall_start: int = 430,
    lower_edge_blend_width: int = 0,
) -> "np.ndarray":
    """Fill each masked run from clean pixels in the same-height neighborhood.

    A narrow three-row-radius neighborhood is used near the wall's horizontal
    top edge so white pixels cannot leak into the blue-grey area.  Far below
    that edge, a wider vertical neighborhood suppresses isolated object edges
    and compression noise.
    """

    result = source.copy()
    height, width = repair_mask.shape
    for y in range(height):
        row_mask = repair_mask[y]
        changes = np.diff(np.pad(row_mask.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        for start, end in zip(starts, ends, strict=True):
            left_start = max(0, start - sample_width)
            right_end = min(width, end + 1 + sample_width)

            def sample_colors(radius: int) -> tuple["np.ndarray", "np.ndarray"]:
                sample_top = max(0, y - radius)
                sample_bottom = min(height, y + radius + 1)
                left = source[sample_top:sample_bottom, left_start:start].reshape(-1, 3)
                right = source[sample_top:sample_bottom, end + 1 : right_end].reshape(-1, 3)
                if len(left):
                    left_color = np.median(left, axis=0)
                elif len(right):
                    left_color = np.median(right, axis=0)
                else:
                    raise ValueError(f"row {y} has no clean background samples")
                if len(right):
                    right_color = np.median(right, axis=0)
                else:
                    right_color = left_color
                return left_color, right_color

            edge_left, edge_right = sample_colors(vertical_sample_radius)
            count = end - start + 1
            weights = np.linspace(0.0, 1.0, count + 2, dtype=np.float64)[1:-1]
            edge_values = (
                edge_left[None, :] * (1.0 - weights[:, None])
                + edge_right[None, :] * weights[:, None]
            )
            values = edge_values
            if y >= lower_wall_start:
                smooth_left, smooth_right = sample_colors(lower_vertical_sample_radius)
                smooth_values = (
                    smooth_left[None, :] * (1.0 - weights[:, None])
                    + smooth_right[None, :] * weights[:, None]
                )
                edge_distance = np.minimum(
                    np.arange(1, count + 1), np.arange(count, 0, -1)
                ).astype(np.float64)
                interior_weight = np.clip(
                    edge_distance / max(1, lower_edge_blend_width), 0.0, 1.0
                )
                values = (
                    edge_values * (1.0 - interior_weight[:, None])
                    + smooth_values * interior_weight[:, None]
                )
            result[y, start : end + 1] = np.clip(np.rint(values), 0, 255).astype(
                np.uint8
            )
    return result


def feather_filled_region(
    source: "np.ndarray",
    filled: "np.ndarray",
    repair_mask: "np.ndarray",
    radius: float = 5.0,
) -> "np.ndarray":
    alpha = np.asarray(
        Image.fromarray((repair_mask.astype(np.uint8) * 255), mode="L").filter(
            ImageFilter.GaussianBlur(radius=radius)
        )
    ).astype(np.float32) / 255.0
    # Preserve the strict invariant that pixels outside the audited repair mask
    # are copied bit-for-bit from the source frame.
    alpha[~repair_mask] = 0.0
    return np.clip(
        np.rint(
            filled.astype(np.float32) * alpha[..., None]
            + source.astype(np.float32) * (1.0 - alpha[..., None])
        ),
        0,
        255,
    ).astype(np.uint8)


ROBOT_WRIST = None
ROBOT_MIDDLE_MCP = None
ROBOT_ARM_BASE = None


def similarity_transform(
    source_start: "np.ndarray",
    source_end: "np.ndarray",
    target_start: "np.ndarray",
    target_end: "np.ndarray",
) -> "np.ndarray":
    source_vector = source_end - source_start
    target_vector = target_end - target_start
    source_length = float(np.linalg.norm(source_vector))
    target_length = float(np.linalg.norm(target_vector))
    if source_length < 1 or target_length < 10:
        raise ValueError(f"invalid alignment lengths: {source_length}, {target_length}")
    angle = math.atan2(target_vector[1], target_vector[0]) - math.atan2(
        source_vector[1], source_vector[0]
    )
    scale = target_length / source_length
    linear = scale * np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    translation = target_start - linear @ source_start
    return np.column_stack([linear, translation])


def forearm_transform(
    target_base: "np.ndarray", target_wrist: "np.ndarray", target_half_width: float
) -> "np.ndarray":
    source_vector = ROBOT_WRIST - ROBOT_ARM_BASE
    target_vector = target_wrist - target_base
    source_length = float(np.linalg.norm(source_vector))
    target_length = float(np.linalg.norm(target_vector))
    source_axis = source_vector / source_length
    target_axis = target_vector / target_length
    source_perpendicular = np.array([-source_axis[1], source_axis[0]])
    target_perpendicular = np.array([-target_axis[1], target_axis[0]])
    linear = (
        (target_length / source_length) * np.outer(target_axis, source_axis)
        + (target_half_width / 80.0)
        * np.outer(target_perpendicular, source_perpendicular)
    )
    translation = target_base - linear @ ROBOT_ARM_BASE
    return np.column_stack([linear, translation])


def warp_rgba(
    rgb: "np.ndarray",
    mask: "np.ndarray",
    transform: "np.ndarray",
    width: int,
    height: int,
) -> tuple["np.ndarray", "np.ndarray"]:
    forward = np.vstack([transform, [0.0, 0.0, 1.0]])
    inverse = np.linalg.inv(forward)
    coefficients = tuple(inverse[:2].reshape(-1).tolist())
    rgba = np.dstack([rgb, mask])
    warped = Image.fromarray(rgba, mode="RGBA").transform(
        (width, height),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0, 0),
    )
    warped_array = np.asarray(warped)
    alpha_image = Image.fromarray(warped_array[..., 3], mode="L").filter(
        ImageFilter.GaussianBlur(radius=1.4)
    )
    return warped_array[..., :3], np.asarray(alpha_image).astype(np.float32) / 255.0


def blend(base: "np.ndarray", layer: "np.ndarray", alpha: "np.ndarray") -> "np.ndarray":
    return np.clip(
        np.rint(
            layer.astype(np.float32) * alpha[..., None]
            + base.astype(np.float32) * (1.0 - alpha[..., None])
        ),
        0,
        255,
    ).astype(np.uint8)


def skin_mask(rgb: "np.ndarray") -> "np.ndarray":
    values = rgb.astype(np.float32)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    cb = 128.0 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 128.0 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    return (
        (cb >= 75)
        & (cb <= 132)
        & (cr >= 132)
        & (cr <= 180)
        & (red > green * 1.04)
        & (green > blue * 0.92)
    )


def write_reproducibility_record(
    output_dir: Path, args: argparse.Namespace, inputs: dict[str, str]
) -> None:
    command = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
    (output_dir / "command.txt").write_text(command + "\n")
    (output_dir / "hostname.txt").write_text(socket.gethostname() + "\n")
    (output_dir / "seed.json").write_text(
        json.dumps({"randomness": "none", "seed": None}, indent=2) + "\n"
    )
    git_lines: list[str] = []
    for git_command in (
        ["git", "rev-parse", "HEAD"],
        ["git", "branch", "--show-current"],
        ["git", "status", "--short"],
        ["git", "diff", "--stat"],
    ):
        completed = run(git_command, check=False)
        git_lines.append(f"$ {' '.join(git_command)}\n{completed.stdout}{completed.stderr}")
    (output_dir / "git-state.txt").write_text("\n".join(git_lines))
    ffmpeg_version = run(["ffmpeg", "-version"]).stdout.splitlines()[0]
    packages = {
        "python": sys.version,
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "ffmpeg": ffmpeg_version,
    }
    (output_dir / "packages.json").write_text(json.dumps(packages, indent=2) + "\n")
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "inputs_sha256": inputs,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def create_showcase(template: Path, repaired: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(template),
            "-i",
            str(repaired),
            "-filter_complex",
            "[0:v][1:v]overlay=x=1280:y=48:shortest=1[out]",
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> int:
    global np, Image, ImageDraw, ImageFilter
    global ROBOT_WRIST, ROBOT_MIDDLE_MCP, ROBOT_ARM_BASE

    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError as exc:
        raise SystemExit(
            "repair requires optional dependencies numpy and Pillow; "
            "install them in the script environment"
        ) from exc

    ROBOT_WRIST = np.array([300.0, 320.0])
    ROBOT_MIDDLE_MCP = np.array([300.0, 210.0])
    ROBOT_ARM_BASE = np.array([300.0, 550.0])

    args = parse_args()
    if bool(args.showcase_template) != bool(args.showcase_output):
        raise SystemExit(
            "--showcase-template and --showcase-output must be supplied together"
        )
    input_paths = {
        "source": args.source,
        "current_composite": args.current_composite,
        "replacement_mask": args.replacement_mask,
        "robot_rgb": args.robot_rgb,
        "robot_mask": args.robot_mask,
        "alignment": args.alignment,
    }
    if args.showcase_template:
        input_paths["showcase_template"] = args.showcase_template
    for label, path in input_paths.items():
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repaired_dir = args.output_dir / "repaired"
    repaired_dir.mkdir()
    log_path = args.output_dir / "run.log"

    source_info = probe(args.source)
    current_info = probe(args.current_composite)
    replacement_info = probe(args.replacement_mask)
    robot_info = probe(args.robot_rgb)
    robot_mask_info = probe(args.robot_mask)
    expected = VideoInfo(1280, 720, 30.0, 621, 20.7)
    if source_info != expected or current_info != expected or replacement_info != expected:
        raise SystemExit(
            "unexpected source/composite/mask geometry: "
            f"source={source_info}, composite={current_info}, mask={replacement_info}"
        )
    expected_robot = VideoInfo(600, 600, 30.0, 621, 20.7)
    if robot_info != expected_robot or robot_mask_info != expected_robot:
        raise SystemExit(
            f"unexpected robot layer geometry: rgb={robot_info}, mask={robot_mask_info}"
        )
    alignment = json.loads(args.alignment.read_text())
    if len(alignment) != expected.frames:
        raise SystemExit(f"alignment records: expected {expected.frames}, got {len(alignment)}")

    hashes = {label: sha256(path) for label, path in input_paths.items()}
    write_reproducibility_record(args.output_dir, args, hashes)

    repaired_path = repaired_dir / "five-finger-hand-and-arm-background-repaired.mp4"
    audit_mask_path = repaired_dir / "expanded-human-removal-mask.mkv"
    readers = {
        "source": reader(args.source, "rgb24"),
        "current": reader(args.current_composite, "rgb24"),
        "replacement": reader(args.replacement_mask, "gray"),
        "robot_rgb": reader(args.robot_rgb, "rgb24"),
        "robot_mask": reader(args.robot_mask, "gray"),
    }
    output_writer = writer(
        repaired_path, expected.width, expected.height, expected.fps, "rgb24", "libx264rgb"
    )
    mask_writer = writer(
        audit_mask_path, expected.width, expected.height, expected.fps, "gray", "ffv1"
    )
    for process in readers.values():
        assert process.stdout is not None
    assert output_writer.stdin is not None and mask_writer.stdin is not None

    source_bytes = expected.width * expected.height * 3
    mask_bytes = expected.width * expected.height
    robot_rgb_bytes = expected_robot.width * expected_robot.height * 3
    robot_mask_bytes = expected_robot.width * expected_robot.height
    outside_channel_differences = 0
    white_corruption_before = 0
    white_corruption_after = 0
    human_residual_pixels_before = 0
    human_residual_pixels_after = 0
    source_skin_cleanup_candidates = 0
    robot_alpha_pixels = 0

    for frame_index, record in enumerate(alignment):
        source_raw = read_exact(readers["source"].stdout, source_bytes)
        current_raw = read_exact(readers["current"].stdout, source_bytes)
        replacement_raw = read_exact(readers["replacement"].stdout, mask_bytes)
        robot_rgb_raw = read_exact(readers["robot_rgb"].stdout, robot_rgb_bytes)
        robot_mask_raw = read_exact(readers["robot_mask"].stdout, robot_mask_bytes)
        if any(
            value is None
            for value in (
                source_raw,
                current_raw,
                replacement_raw,
                robot_rgb_raw,
                robot_mask_raw,
            )
        ):
            raise RuntimeError(f"one or more inputs ended at frame {frame_index}")

        source_frame = np.frombuffer(source_raw, dtype=np.uint8).reshape(
            expected.height, expected.width, 3
        )
        current_frame = np.frombuffer(current_raw, dtype=np.uint8).reshape(
            expected.height, expected.width, 3
        )
        original_mask = np.frombuffer(replacement_raw, dtype=np.uint8).reshape(
            expected.height, expected.width
        ) > 127
        robot_frame = np.frombuffer(robot_rgb_raw, dtype=np.uint8).reshape(
            expected_robot.height, expected_robot.width, 3
        )
        full_robot_mask = np.frombuffer(robot_mask_raw, dtype=np.uint8).reshape(
            expected_robot.height, expected_robot.width
        )

        wrist = np.asarray(record["wrist"], dtype=np.float64)
        middle_mcp = np.asarray(record["middle_mcp"], dtype=np.float64)
        source_base = np.asarray(record["source_arm_base"], dtype=np.float64)
        expanded_arm = polygon_mask(
            expected.width,
            expected.height,
            wrist,
            source_base,
            args.wrist_half_width,
            args.base_half_width,
        )
        repair_mask = binary_dilate(original_mask, args.mask_dilation) | expanded_arm
        base = horizontal_background_fill(source_frame, repair_mask)
        base = feather_filled_region(source_frame, base, repair_mask)

        hand_source_mask = full_robot_mask.copy()
        hand_source_mask[350:, :] = 0
        arm_source_mask = full_robot_mask.copy()
        arm_source_mask[:285, :] = 0
        arm_transform = forearm_transform(
            source_base, wrist, args.target_forearm_half_width
        )
        hand_direction = middle_mcp - wrist
        hand_direction /= np.linalg.norm(hand_direction)
        stable_middle_mcp = wrist + hand_direction * args.target_hand_length
        hand_transform = similarity_transform(
            ROBOT_WRIST, ROBOT_MIDDLE_MCP, wrist, stable_middle_mcp
        )
        warped_arm, arm_alpha = warp_rgba(
            robot_frame,
            arm_source_mask,
            arm_transform,
            expected.width,
            expected.height,
        )
        warped_hand, hand_alpha = warp_rgba(
            robot_frame,
            hand_source_mask,
            hand_transform,
            expected.width,
            expected.height,
        )
        repaired = blend(base, warped_arm, arm_alpha)
        repaired = blend(repaired, warped_hand, hand_alpha)
        robot_alpha = np.maximum(arm_alpha, hand_alpha)

        outside = ~repair_mask
        outside_channel_differences += int(
            np.count_nonzero(repaired[outside] != source_frame[outside])
        )
        wall_band = np.zeros_like(repair_mask)
        wall_band[115:430, :] = True
        base_luma = base.mean(axis=2)
        current_luma = current_frame.mean(axis=2)
        repaired_luma = repaired.mean(axis=2)
        corruption_roi = repair_mask & wall_band & (robot_alpha < 0.05)
        white_corruption_before += int(
            np.count_nonzero(corruption_roi & (current_luma > base_luma + 25))
        )
        white_corruption_after += int(
            np.count_nonzero(corruption_roi & (repaired_luma > base_luma + 25))
        )
        cleanup_corridor = repair_mask & (robot_alpha < 0.05)
        source_base_delta = np.abs(
            source_frame.astype(np.int16) - base.astype(np.int16)
        ).mean(axis=2)
        source_skin = (
            skin_mask(source_frame) & cleanup_corridor & (source_base_delta >= 18)
        )
        current_source_delta = np.abs(
            current_frame.astype(np.int16) - source_frame.astype(np.int16)
        ).mean(axis=2)
        repaired_source_delta = np.abs(
            repaired.astype(np.int16) - source_frame.astype(np.int16)
        ).mean(axis=2)
        source_skin_cleanup_candidates += int(np.count_nonzero(source_skin))
        human_residual_pixels_before += int(
            np.count_nonzero(source_skin & (current_source_delta <= 8))
        )
        human_residual_pixels_after += int(
            np.count_nonzero(source_skin & (repaired_source_delta <= 8))
        )
        robot_alpha_pixels += int(np.count_nonzero(robot_alpha >= 0.20))

        output_writer.stdin.write(repaired.tobytes())
        mask_writer.stdin.write((repair_mask.astype(np.uint8) * 255).tobytes())
        if frame_index % 60 == 0 or frame_index + 1 == expected.frames:
            message = f"processed {frame_index + 1}/{expected.frames} frames"
            print(message, flush=True)
            with log_path.open("a") as log:
                log.write(message + "\n")

    for process in readers.values():
        process.stdout.close()
    output_writer.stdin.close()
    mask_writer.stdin.close()
    reader_codes = {label: process.wait() for label, process in readers.items()}
    output_code = output_writer.wait()
    mask_code = mask_writer.wait()
    if any(reader_codes.values()) or output_code or mask_code:
        raise SystemExit(
            f"ffmpeg failure: readers={reader_codes}, output={output_code}, mask={mask_code}"
        )

    audit = {
        "accepted": outside_channel_differences == 0
        and white_corruption_after == 0
        and human_residual_pixels_after == 0,
        "frames": expected.frames,
        "fps": expected.fps,
        "duration_seconds": expected.duration,
        "outside_repair_mask_channel_differences": outside_channel_differences,
        "white_wall_corruption_pixels_before": white_corruption_before,
        "white_wall_corruption_pixels_after": white_corruption_after,
        "source_skin_cleanup_candidates": source_skin_cleanup_candidates,
        "human_residual_pixels_before": human_residual_pixels_before,
        "human_residual_pixels_after": human_residual_pixels_after,
        "robot_alpha_pixels": robot_alpha_pixels,
        "source_info": asdict(source_info),
        "repaired_info": asdict(probe(repaired_path)),
        "repaired_sha256": sha256(repaired_path),
        "repair_mask_sha256": sha256(audit_mask_path),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    if not audit["accepted"]:
        raise SystemExit(f"repair audit failed: {json.dumps(audit, sort_keys=True)}")

    if args.showcase_output:
        create_showcase(args.showcase_template, repaired_path, args.showcase_output)
        showcase_info = probe(args.showcase_output)
        showcase_record = {
            "path": str(args.showcase_output),
            "info": asdict(showcase_info),
            "sha256": sha256(args.showcase_output),
        }
        (args.output_dir / "showcase.json").write_text(
            json.dumps(showcase_record, indent=2) + "\n"
        )

    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
