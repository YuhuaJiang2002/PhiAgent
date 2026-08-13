#!/usr/bin/env python3
"""Prepare one held-out >=20s RGB-alpha-contact VACE evaluation window."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.object_factored_long_video import (  # noqa: E402
    SourceResizeCrop,
    binary_dilate_square,
    remap_boolean_mask,
    resolve_flower_visibility,
    source_skin_like,
    strict_flower_seed,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--window-index", type=int, default=-1)
    parser.add_argument("--person-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--limb-masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--late-start", type=int, default=480)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--person-dilation", type=int, default=10)
    parser.add_argument("--skin-negative-dilation", type=int, default=2)
    parser.add_argument("--person-core-negative-erosion", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--mask-frame-name", required=True)
    parser.add_argument("--mask-source-width", type=int, required=True)
    parser.add_argument("--mask-source-height", type=int, required=True)
    parser.add_argument("--mask-scaled-width", type=int, required=True)
    parser.add_argument("--mask-scaled-height", type=int, required=True)
    parser.add_argument("--mask-crop-left", type=int, required=True)
    parser.add_argument("--mask-crop-top", type=int, required=True)
    parser.add_argument("--target-frame-name", required=True)
    parser.add_argument("--target-scaled-width", type=int, required=True)
    parser.add_argument("--target-scaled-height", type=int, required=True)
    parser.add_argument("--target-crop-left", type=int, required=True)
    parser.add_argument("--target-crop-top", type=int, required=True)
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--target-height", type=int, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(np: Any, path: Path, key: str) -> tuple[Any, int, int, str]:
    payload = np.load(path, allow_pickle=False)
    return payload[key], int(payload["height"]), int(payload["width"]), str(payload["bitorder"])


def _unpack(np: Any, payload: tuple[Any, int, int, str], index: int) -> Any:
    packed, height, width, bitorder = payload
    return np.unpackbits(packed[index], bitorder=bitorder)[: height * width].reshape(height, width).astype(bool)


def _read(process: subprocess.Popen[bytes], size: int, index: int) -> bytes:
    assert process.stdout is not None
    value = bytearray()
    while len(value) < size:
        chunk = process.stdout.read(size - len(value))
        if not chunk:
            raise RuntimeError(f"source decoder ended before frame {index}")
        value.extend(chunk)
    return bytes(value)


def _encode(ffmpeg: Path, frames: Any, path: Path, fps: int) -> None:
    process = subprocess.Popen(
        [
            str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{frames.shape[2]}x{frames.shape[1]}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    _, stderr = process.communicate(frames.tobytes())
    if process.returncode:
        raise RuntimeError(stderr.decode(errors="replace")[-2000:])


def main() -> int:
    args = _parser().parse_args()
    import numpy as np
    from PIL import Image

    records = json.loads(args.validation_json.read_text())
    selected = records[args.window_index]
    indices = [int(index) for index in selected["indices"]]
    if min(indices) < args.late_start:
        raise ValueError("held-out validation window begins before the late boundary")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    person_payload = _load(np, args.person_masks, "packed")
    flower_payload = _load(np, args.flower_masks, "packed")
    hand_payload = _load(np, args.limb_masks, "hands_packed")
    _, mask_height, mask_width, _ = person_payload
    mask_frame = SourceResizeCrop(
        args.mask_frame_name,
        args.mask_source_width,
        args.mask_source_height,
        args.mask_scaled_width,
        args.mask_scaled_height,
        args.mask_crop_left,
        args.mask_crop_top,
        mask_width,
        mask_height,
    )
    target = SourceResizeCrop(
        args.target_frame_name,
        args.mask_source_width,
        args.mask_source_height,
        args.target_scaled_width,
        args.target_scaled_height,
        args.target_crop_left,
        args.target_crop_top,
        args.target_width,
        args.target_height,
    )
    command = [
        str(args.ffmpeg), "-v", "error", "-i", str(args.source_video.resolve()), "-vf",
        (
            f"scale={target.scaled_width}:{target.scaled_height}:flags=area,"
            f"crop={target.output_width}:{target.output_height}:{target.crop_left}:{target.crop_top}"
        ),
        "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    decoder = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = target.output_width * target.output_height * 3
    requested = set(indices)
    inputs = {}
    masks = {}
    frame_rows = []
    for index in range(args.expected_frames):
        source = np.frombuffer(_read(decoder, frame_size, index), dtype=np.uint8).reshape(
            target.output_height, target.output_width, 3
        )
        if index not in requested:
            continue
        person = remap_boolean_mask(np, _unpack(np, person_payload, index), source_frame=mask_frame, target_frame=target)
        flower_track = remap_boolean_mask(np, _unpack(np, flower_payload, index), source_frame=mask_frame, target_frame=target)
        hands = remap_boolean_mask(np, _unpack(np, hand_payload, index), source_frame=mask_frame, target_frame=target)
        alpha = binary_dilate_square(np, person, args.person_dilation)
        skin = source_skin_like(np, source) & (person | hands)
        skin = binary_dilate_square(np, skin & alpha, args.skin_negative_dilation)
        flower = resolve_flower_visibility(
            np,
            candidates=flower_track | strict_flower_seed(np, source),
            edit_support=alpha,
            source_person=person,
            source_skin_negative=skin,
            person_core_erosion=args.person_core_negative_erosion,
        )
        edit = alpha & ~flower
        inputs[index] = np.asarray(Image.fromarray(source).resize((args.width, args.height), Image.Resampling.LANCZOS))
        edit_rgb = np.repeat((edit.astype(np.uint8) * 255)[..., None], 3, axis=2)
        masks[index] = np.asarray(Image.fromarray(edit_rgb).resize((args.width, args.height), Image.Resampling.NEAREST))
        frame_rows.append({"frame": index, "edit_fraction": float(edit.mean()), "flower_protected_pixels": int(flower.sum())})
    if decoder.stdout is not None:
        decoder.stdout.close()
    stderr = decoder.stderr.read().decode(errors="replace") if decoder.stderr else ""
    if decoder.wait():
        raise RuntimeError(stderr[-2000:])
    input_path = output / "heldout-input.mp4"
    mask_path = output / "heldout-edit-mask.mp4"
    _encode(args.ffmpeg, np.stack([inputs[index] for index in indices]), input_path, args.fps)
    _encode(args.ffmpeg, np.stack([masks[index] for index in indices]), mask_path, args.fps)
    source_control = Path(selected["control"]).resolve()
    source_reference = Path(selected["reference"]).resolve()
    source_target = Path(selected["target"]).resolve()
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "method": "heldout_after_20s_robot_rgb_alpha_contact_vace_window",
        "selected_validation_index": args.window_index,
        "source_frame_indices": indices,
        "coordinate_frames": {
            "mask": mask_frame.to_dict(),
            "target": target.to_dict(),
            "timeline": f"absolute_frame_index:full_source_{args.expected_frames}",
        },
        "outputs": {
            "input": {"path": str(input_path), "sha256": _sha256(input_path)},
            "edit_mask": {"path": str(mask_path), "sha256": _sha256(mask_path)},
            "control": {"path": str(source_control), "sha256": _sha256(source_control)},
            "reference": {"path": str(source_reference), "sha256": _sha256(source_reference)},
            "teacher_target": {"path": str(source_target), "sha256": _sha256(source_target)},
        },
        "frames": frame_rows,
        "seed": args.seed,
        "hostname": platform.node(),
        "limitations": [
            "The validation window is temporally held out but belongs to the same scene.",
            "The edit mask protects a flower union, not persistent per-stem instances.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "indices": indices}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
