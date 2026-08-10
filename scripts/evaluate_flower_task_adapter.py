#!/usr/bin/env python3
"""Compare zero-shot and task-LoRA outputs on one held-out flower-contact clip."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--zero-shot", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-radius", type=int, default=22)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(cv2: Any, path: Path) -> list[Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video contains no decoded frames: {path}")
    return frames


def _similarity(np: Any, first: Any, second: Any, mask: Any | None = None) -> float:
    delta = np.abs(first.astype(np.float32) - second.astype(np.float32))
    if mask is not None:
        delta = delta[mask]
    return math.exp(-float(delta.mean()) / 32.0)


def _motion_similarity(np: Any, target: list[Any], candidate: list[Any], masks: list[Any]) -> float:
    values = []
    for index in range(1, len(target)):
        target_motion = np.abs(
            target[index].astype(np.float32) - target[index - 1].astype(np.float32)
        )
        candidate_motion = np.abs(
            candidate[index].astype(np.float32) - candidate[index - 1].astype(np.float32)
        )
        union = masks[index] | masks[index - 1]
        residual = np.abs(target_motion[union] - candidate_motion[union])
        values.append(math.exp(-float(residual.mean()) / 32.0))
    return sum(values) / len(values)


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "target": args.target.expanduser().resolve(),
        "zero_shot": args.zero_shot.expanduser().resolve(),
        "adapted": args.adapted.expanduser().resolve(),
        "trajectory": args.trajectory.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{name} does not exist or is empty: {path}")
    if args.contact_radius <= 0:
        raise ValueError("contact radius must be positive")

    import cv2
    import numpy as np

    target = _decode(cv2, paths["target"])
    zero_shot = _decode(cv2, paths["zero_shot"])
    adapted = _decode(cv2, paths["adapted"])
    trajectory = json.loads(paths["trajectory"].read_text())
    rows = trajectory["frames"]
    if not len(target) == len(zero_shot) == len(adapted) == len(rows):
        raise RuntimeError("target, candidates, and trajectory must have equal frame counts")
    shape = target[0].shape
    if any(frame.shape != shape for sequence in (target, zero_shot, adapted) for frame in sequence):
        raise RuntimeError("all validation frames must have the same shape")
    height, width = shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    contact_masks = []
    contact_frames = []
    for index, row in enumerate(rows):
        x, y = row["right_hand_xy"]
        contact_masks.append((xx - x) ** 2 + (yy - y) ** 2 <= args.contact_radius**2)
        if row["right_contact_required"]:
            contact_frames.append(index)
    if not contact_frames:
        raise ValueError("held-out trajectory contains no required contact frames")
    zero_contact = sum(
        _similarity(np, target[index], zero_shot[index], contact_masks[index])
        for index in contact_frames
    ) / len(contact_frames)
    adapted_contact = sum(
        _similarity(np, target[index], adapted[index], contact_masks[index])
        for index in contact_frames
    ) / len(contact_frames)
    zero_global = sum(
        _similarity(np, target_frame, candidate)
        for target_frame, candidate in zip(target, zero_shot)
    ) / len(target)
    adapted_global = sum(
        _similarity(np, target_frame, candidate)
        for target_frame, candidate in zip(target, adapted)
    ) / len(target)
    zero_motion = _motion_similarity(np, target, zero_shot, contact_masks)
    adapted_motion = _motion_similarity(np, target, adapted, contact_masks)
    distinctness = sum(
        float(
            np.mean(
                np.abs(
                    zero_frame.astype(np.float32) - adapted_frame.astype(np.float32)
                )
            )
        )
        for zero_frame, adapted_frame in zip(zero_shot, adapted)
    ) / len(target)
    metrics = {
        "global_similarity": {"zero_shot": zero_global, "adapted": adapted_global},
        "contact_roi_similarity": {
            "zero_shot": zero_contact,
            "adapted": adapted_contact,
        },
        "contact_motion_similarity": {
            "zero_shot": zero_motion,
            "adapted": adapted_motion,
        },
        "adapted_minus_zero_contact": adapted_contact - zero_contact,
        "adapted_minus_zero_motion": adapted_motion - zero_motion,
        "adapted_zero_mean_absolute_difference": distinctness,
    }
    gates = {
        "adapted_output_is_distinct": distinctness >= 0.5,
        "contact_roi_not_regressed": adapted_contact >= zero_contact,
        "contact_motion_not_regressed": adapted_motion >= zero_motion,
    }
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "status": "PARTIAL",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "frame_count": len(target),
        "contact_frames": contact_frames,
        "contact_radius_pixels": args.contact_radius,
        "coordinate_frame": "camera:synthetic_pixels",
        "metrics": metrics,
        "gates": gates,
        "all_proxy_gates_pass": all(gates.values()),
        "limitations": [
            "This held-out clip is synthetic and cannot establish real-video quality.",
            "Pixel similarity near a known contact point is not flower-instance or physics proof.",
            "The smoke LoRA has seen only 12 short procedural training clips.",
        ],
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
