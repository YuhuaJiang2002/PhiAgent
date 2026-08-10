#!/usr/bin/env python3
"""Prepare one real flower-contact window for localized VACE ablation.

The real scene remains the input video.  A source-driven MuJoCo robot supplies
the geometry control, while the edit mask covers the source person and projected
robot but explicitly excludes the tracked flower union.  This is a bounded
critical-window test; it does not silently expand to the full 660-frame clip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--robot-video", type=Path, required=True)
    parser.add_argument("--robot-mask-video", type=Path, required=True)
    parser.add_argument("--flower-union-masks", type=Path, required=True)
    parser.add_argument("--person-safety-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=272)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--source-frame-step", type=int, default=3)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--robot-scale", type=float, default=1.18)
    parser.add_argument("--robot-translate-x", type=float, default=160.0)
    parser.add_argument("--robot-translate-y", type=float, default=32.0)
    parser.add_argument("--flower-protection-radius", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_indices(start: int, frames: int, step: int, total: int) -> tuple[int, ...]:
    if start < 0 or frames <= 0 or step <= 0 or total <= 0:
        raise ValueError("window indices must be positive and start must be non-negative")
    result = tuple(start + index * step for index in range(frames))
    if result[-1] >= total:
        raise ValueError("selected real window exceeds the source timeline")
    return result


def localized_edit_mask(np: Any, person: Any, robot: Any, protected_flowers: Any) -> Any:
    if not person.shape == robot.shape == protected_flowers.shape:
        raise ValueError("person, robot, and flower masks must have equal shape")
    return ((person > 0) | (robot > 0)) & ~(protected_flowers > 0)


def _load_packed(np: Any, path: Path) -> Any:
    payload = np.load(path)
    height, width = int(payload["height"]), int(payload["width"])
    unpacked = np.unpackbits(
        payload["packed"],
        axis=1,
        bitorder=str(payload["bitorder"]),
    )[:, : height * width]
    return unpacked.reshape(len(payload["packed"]), height, width).astype(np.uint8)


def _encode(ffmpeg: Path, frames: Any, output: Path, fps: int, pixel_format: str) -> None:
    channels = 1 if pixel_format == "gray" else 3
    if frames.ndim != 4 - (channels == 1):
        raise ValueError("unexpected frame array rank")
    height, width = frames.shape[1:3]
    process = subprocess.Popen(
        [
            str(ffmpeg),
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
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.write(frames.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed to encode {output}")


def _seek(capture: Any, cv2: Any, frame_index: int) -> Any:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"video decode failed at frame {frame_index}")
    return frame


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "source_video": args.source_video.expanduser().resolve(),
        "robot_video": args.robot_video.expanduser().resolve(),
        "robot_mask_video": args.robot_mask_video.expanduser().resolve(),
        "flower_union_masks": args.flower_union_masks.expanduser().resolve(),
        "person_safety_mask": args.person_safety_mask.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{name} does not exist or is empty: {path}")
    if args.frames < 9 or (args.frames - 1) % 4:
        raise ValueError("VACE window frames must satisfy 4n+1 and be at least 9")
    if args.width % 16 or args.height % 16:
        raise ValueError("VACE dimensions must be divisible by 16")
    if args.flower_protection_radius < 0:
        raise ValueError("flower protection radius must be non-negative")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite real-window experiment: {output}")
    output.mkdir(parents=True)

    import cv2
    import numpy as np

    captures = {
        name: cv2.VideoCapture(str(paths[name]))
        for name in ("source_video", "robot_video", "robot_mask_video")
    }
    if not all(capture.isOpened() for capture in captures.values()):
        raise RuntimeError("could not open a real-window video input")
    source_total = int(captures["source_video"].get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(captures["source_video"].get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(captures["source_video"].get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (source_total, source_width, source_height) != (660, 832, 480):
        raise RuntimeError("real source must be the aligned 832x480x660 artifact")
    if any(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != 660 for capture in captures.values()):
        raise RuntimeError("source, robot, and robot mask must contain 660 aligned frames")
    indices = selected_indices(
        args.start_frame,
        args.frames,
        args.source_frame_step,
        source_total,
    )
    flower_masks = _load_packed(np, paths["flower_union_masks"])
    if flower_masks.shape != (660, 480, 832):
        raise RuntimeError("flower union masks must have shape 660x480x832")
    person = cv2.imread(str(paths["person_safety_mask"]), cv2.IMREAD_GRAYSCALE)
    if person is None:
        raise RuntimeError("could not decode the person safety mask")
    if person.shape != (480, 832):
        person = cv2.resize(person, (832, 480), interpolation=cv2.INTER_NEAREST)
    transform = np.asarray(
        [
            [args.robot_scale, 0.0, args.robot_translate_x],
            [0.0, args.robot_scale, args.robot_translate_y],
        ],
        dtype=np.float32,
    )
    protection_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.flower_protection_radius * 2 + 1, args.flower_protection_radius * 2 + 1),
    )
    inputs, controls, edit_masks, references = [], [], [], []
    mask_fractions, protected_fractions = [], []
    for frame_index in indices:
        source = _seek(captures["source_video"], cv2, frame_index)
        robot = _seek(captures["robot_video"], cv2, frame_index)
        robot_mask_frame = _seek(captures["robot_mask_video"], cv2, frame_index)
        robot_mask = cv2.cvtColor(robot_mask_frame, cv2.COLOR_BGR2GRAY) >= 127
        placed_robot = cv2.warpAffine(
            robot,
            transform,
            (832, 480),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
        )
        placed_robot_mask = cv2.warpAffine(
            robot_mask.astype(np.uint8) * 255,
            transform,
            (832, 480),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        protected = cv2.dilate(
            flower_masks[frame_index] * 255,
            protection_kernel,
        )
        edit = localized_edit_mask(np, person, placed_robot_mask, protected)
        geometry = ((placed_robot_mask > 0) | (protected > 0)).astype(np.uint8) * 255
        edges = cv2.Canny(geometry, 60, 140)
        control = np.repeat(edges[..., None], 3, axis=2)
        alpha = cv2.GaussianBlur(placed_robot_mask, (3, 3), 0).astype(np.float32) / 255.0
        reference = np.rint(
            placed_robot.astype(np.float32) * alpha[..., None]
            + source.astype(np.float32) * (1.0 - alpha[..., None])
        ).astype(np.uint8)
        reference[protected > 0] = source[protected > 0]
        inputs.append(cv2.resize(source, (args.width, args.height), interpolation=cv2.INTER_AREA))
        controls.append(cv2.resize(control, (args.width, args.height), interpolation=cv2.INTER_AREA))
        edit_masks.append(
            cv2.resize(
                edit.astype(np.uint8) * 255,
                (args.width, args.height),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        references.append(
            cv2.resize(reference, (args.width, args.height), interpolation=cv2.INTER_AREA)
        )
        mask_fractions.append(float(np.mean(edit)))
        protected_fractions.append(float(np.mean(protected > 0)))
    for capture in captures.values():
        capture.release()
    input_array = np.stack(inputs)
    control_array = np.stack(controls)
    mask_array = np.stack(edit_masks)
    input_path = output / "real-input.mp4"
    control_path = output / "robot-flower-control.mp4"
    mask_path = output / "localized-edit-mask.mp4"
    reference_path = output / "robot-reference.png"
    _encode(paths["ffmpeg"], input_array, input_path, 8, "bgr24")
    _encode(paths["ffmpeg"], control_array, control_path, 8, "bgr24")
    _encode(paths["ffmpeg"], mask_array, mask_path, 8, "gray")
    if not cv2.imwrite(str(reference_path), references[0]):
        raise RuntimeError("could not write real-window robot reference")
    manifest = {
        "schema_version": "1.0.0",
        "method": "localized_real_flower_vace_critical_window",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "honest_status": "PARTIAL",
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "seed": args.seed,
        "gpu": {"used": False, "reason": "CPU VACE input preparation"},
        "coordinate_frames": {
            "source": "camera:source_pixels",
            "robot": "robot:base projected into camera:source_pixels",
            "flower": "object:flower union projected into camera:source_pixels",
        },
        "selected_source_frames": list(indices),
        "fps": 8,
        "resolution": [args.width, args.height],
        "robot_transform": {
            "from": "camera:robot_render_pixels",
            "to": "camera:source_pixels",
            "scale": args.robot_scale,
            "translate_x": args.robot_translate_x,
            "translate_y": args.robot_translate_y,
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "input_video": {"path": str(input_path), "sha256": _sha256(input_path)},
            "control_video": {"path": str(control_path), "sha256": _sha256(control_path)},
            "edit_mask": {"path": str(mask_path), "sha256": _sha256(mask_path)},
            "reference": {"path": str(reference_path), "sha256": _sha256(reference_path)},
        },
        "metrics": {
            "edit_mask_fraction_min": min(mask_fractions),
            "edit_mask_fraction_max": max(mask_fractions),
            "protected_flower_fraction_min": min(protected_fractions),
            "protected_flower_fraction_max": max(protected_fractions),
        },
        "limitations": [
            "The source flower evidence is still a union mask and cannot prove active-stem identity.",
            "The MuJoCo robot control follows source wrists but does not enforce contact force.",
            "This 17-frame critical window must pass before any 660-frame expansion.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
