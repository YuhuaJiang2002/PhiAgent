#!/usr/bin/env python3
"""Replace the hand in a same-scene robot reference while preserving arm and object."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from prepare_vendor_hand_target import (
    _overlay,
    _parse_vector,
    _render_hand,
    _transform_foreground,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_polygon(value: str) -> tuple[tuple[int, int], ...]:
    points = tuple(
        tuple(int(coordinate) for coordinate in point.split(","))
        for point in value.split(";")
    )
    if len(points) < 3 or any(len(point) != 2 for point in points):
        raise ValueError("erase-polygon requires at least three x,y points")
    return points


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-license", required=True)
    parser.add_argument("--camera", default="0,0,0.05,0.55,135,-30")
    parser.add_argument("--qpos", default="")
    parser.add_argument("--hand-width", type=int, required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument(
        "--erase-polygon",
        default="275,165;430,155;585,220;575,375;430,390;280,335",
    )
    parser.add_argument("--connector-polygon")
    parser.add_argument("--erase-fill", choices=("inpaint", "right-strip"), default="inpaint")
    args = parser.parse_args()

    reference = args.reference_image.expanduser().resolve()
    model = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for label, path in (("reference image", reference), ("model", model)):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")
    import cv2
    import mujoco
    import numpy as np

    image = cv2.imread(str(reference))
    if image is None:
        raise ValueError(f"failed to decode reference image: {reference}")
    polygon = np.asarray(_parse_polygon(args.erase_polygon), dtype=np.int32)
    erase_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(erase_mask, [polygon], 255)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    apple = (
        ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 170))
        & (hsv[:, :, 1] >= 145)
        & (hsv[:, :, 2] >= 30)
    )
    erase_mask = cv2.dilate(erase_mask, np.ones((5, 5), np.uint8))
    apple_guard = cv2.dilate(apple.astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    erase_mask[apple_guard > 0] = 0
    if args.erase_fill == "inpaint":
        condition = cv2.inpaint(image, erase_mask, 7, cv2.INPAINT_TELEA)
    else:
        condition = image.copy()
        yy, xx = np.nonzero(erase_mask)
        strip_start = round(image.shape[1] * 0.72)
        strip_width = image.shape[1] - strip_start
        source_x = strip_start + (xx % strip_width)
        condition[yy, xx] = image[yy, source_x]
    connector = (
        np.asarray(_parse_polygon(args.connector_polygon), dtype=np.int32)
        if args.connector_polygon
        else None
    )
    if connector is not None:
        cv2.fillConvexPoly(condition, connector, (82, 84, 88), lineType=cv2.LINE_AA)
        cv2.polylines(
            condition, [connector], True, (34, 35, 38), 4, lineType=cv2.LINE_AA
        )
        cv2.line(
            condition,
            tuple(connector[0]),
            tuple(connector[1]),
            (145, 147, 151),
            3,
            lineType=cv2.LINE_AA,
        )

    camera = _parse_vector(args.camera, 6, "camera")
    qpos = tuple(float(item) for item in args.qpos.split(",")) if args.qpos else ()
    placement = tuple(int(item) for item in _parse_vector(args.placement, 2, "placement"))
    rendered, rendered_mask = _render_hand(
        mujoco, np, model, 640, 480, camera, qpos
    )
    hand, hand_mask = _transform_foreground(
        cv2, np, rendered, rendered_mask, args.hand_width, args.rotation_deg
    )
    _overlay(cv2, np, condition, hand, hand_mask, *placement)
    if not cv2.imwrite(str(output), condition):
        raise RuntimeError(f"failed to write condition image: {output}")

    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "same_scene_robot_arm_vendor_hand_condition",
        "vendor": args.vendor,
        "hostname": platform.node(),
        "command": [sys.executable, *sys.argv],
        "inputs": {
            "reference_image": str(reference),
            "reference_sha256": _sha256(reference),
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
            "erase_polygon": polygon.tolist(),
            "connector_polygon": connector.tolist() if connector is not None else None,
            "erase_fill": args.erase_fill,
        },
        "output": str(output),
        "output_sha256": _sha256(output),
        "limitations": [
            "The condition image is a deterministic same-scene composite.",
            "The retained arm is the Sharpa reference arm, not a vendor-specific arm.",
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
