#!/usr/bin/env python3
"""Measure wrist correspondence and contact-conditioned flower proximity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction(values: Any, threshold: float) -> float:
    if not len(values):
        raise ValueError("cannot score an empty measurement")
    return float((values <= threshold).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pose-trajectory", type=Path, required=True)
    parser.add_argument("--wrist-trace", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--translate-x", type=float, required=True)
    parser.add_argument("--translate-y", type=float, required=True)
    parser.add_argument("--motion-threshold-pixels", type=float, default=35.0)
    parser.add_argument("--source-contact-threshold-pixels", type=float, default=25.0)
    parser.add_argument("--robot-contact-threshold-pixels", type=float, default=35.0)
    args = parser.parse_args()

    paths = {
        "candidate": args.candidate.expanduser().resolve(),
        "pose_trajectory": args.pose_trajectory.expanduser().resolve(),
        "wrist_trace": args.wrist_trace.expanduser().resolve(),
        "flower_masks": args.flower_masks.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{name} does not exist: {path}")

    import cv2
    import numpy as np

    pose_payload = json.loads(paths["pose_trajectory"].read_text())
    if pose_payload.get("coordinate_frame") != "camera:source_pixels":
        raise ValueError("pose trajectory must use camera:source_pixels")
    source = np.asarray(pose_payload["robust_median_xy"], dtype=np.float32)[:, [4, 5], :]
    source[..., 0] *= 832.0 / 1280.0
    source[..., 1] *= 480.0 / 720.0
    trace = json.loads(paths["wrist_trace"].read_text())
    robot = np.full(source.shape, np.nan, dtype=np.float32)
    for index, item in enumerate(trace):
        for side_index, side in enumerate(("left", "right")):
            centroid = item["rendered_hand_centroids"][side]
            if centroid is not None:
                robot[index, side_index] = centroid
    robot[..., 0] = robot[..., 0] * args.scale + args.translate_x
    robot[..., 1] = robot[..., 1] * args.scale + args.translate_y

    payload = np.load(paths["flower_masks"])
    height, width = int(payload["height"]), int(payload["width"])
    masks = np.unpackbits(
        payload["packed"], axis=1, bitorder=str(payload["bitorder"])
    )[:, : height * width].reshape(len(payload["packed"]), height, width)
    if source.shape != (660, 2, 2) or robot.shape != source.shape or len(masks) != 660:
        raise RuntimeError("evaluation requires aligned 660-frame wrist and flower tracks")

    def distances(points: Any) -> Any:
        result = np.full((660, 2), np.nan, dtype=np.float32)
        for frame_index, mask in enumerate(masks):
            distance = cv2.distanceTransform(
                (1 - mask.astype(np.uint8)), cv2.DIST_L2, 3
            )
            for side_index in range(2):
                x, y = points[frame_index, side_index]
                if np.isfinite(x) and 0 <= round(x) < width and 0 <= round(y) < height:
                    result[frame_index, side_index] = distance[round(y), round(x)]
        return result

    valid = np.isfinite(robot).all(axis=2)
    motion_error = np.linalg.norm(robot - source, axis=2)
    source_flower_distance = distances(source)
    robot_flower_distance = distances(robot)
    source_contact = source_flower_distance <= args.source_contact_threshold_pixels
    contact_values = robot_flower_distance[source_contact]
    ffmpeg = subprocess.run(
        ["which", "ffmpeg"], check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(paths["candidate"]), "-f", "null", "-"],
        check=True,
    )
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "coordinate_frame": "camera:source_pixels",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "transform": {
            "scale": args.scale,
            "translate_x": args.translate_x,
            "translate_y": args.translate_y,
        },
        "metrics": {
            "motion_within_threshold_fraction": _fraction(
                motion_error[valid], args.motion_threshold_pixels
            ),
            "motion_within_threshold_fraction_by_side": [
                _fraction(motion_error[:, side][valid[:, side]], args.motion_threshold_pixels)
                for side in range(2)
            ],
            "motion_error_median_pixels": float(np.median(motion_error[valid])),
            "motion_error_p90_pixels": float(np.quantile(motion_error[valid], 0.9)),
            "source_contact_observations": int(source_contact.sum()),
            "contact_conditioned_proximity_fraction": _fraction(
                contact_values, args.robot_contact_threshold_pixels
            ),
            "contact_conditioned_distance_median_pixels": float(
                np.nanmedian(contact_values)
            ),
            "contact_conditioned_distance_p90_pixels": float(
                np.nanquantile(contact_values, 0.9)
            ),
        },
        "limitations": [
            "Flower masks are a union of instances, so proximity cannot prove held-instance identity.",
            "The contact proxy is valid only when conditioned on source-contact frames and cannot replace visual/depth review.",
            "No aggregate of these metrics can override morphology, occlusion, or human-preference hard gates.",
        ],
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
