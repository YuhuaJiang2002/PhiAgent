#!/usr/bin/env python3
"""Verify rigid robot-hand morphology and temporal stability in a video ROI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _skeleton_endpoints(binary: np.ndarray) -> int:
    image = (binary > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = eroded
    selected = (skeleton > 0).astype(np.uint8)
    neighbors = cv2.filter2D(selected, -1, np.ones((3, 3), dtype=np.uint8))
    return int(np.sum((selected > 0) & (neighbors == 2)))


def _descriptor(frame: np.ndarray, roi: tuple[float, float, float, float]) -> dict[str, object]:
    height, width = frame.shape[:2]
    x0 = round(roi[0] * width)
    y0 = round(roi[1] * height)
    x1 = round((roi[0] + roi[2]) * width)
    y1 = round((roi[1] + roi[3]) * height)
    crop = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 95)).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    eligible = [
        index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= 20
    ]
    if not eligible:
        return {"detected": False}
    selected = max(eligible, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    component = (labels == selected).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    deep_defects = 0
    if hull_indices is not None and len(hull_indices) >= 4 and len(contour) >= 4:
        defects = cv2.convexityDefects(contour, hull_indices)
        if defects is not None:
            scale = max(component.shape)
            deep_defects = sum(
                depth / 256.0 >= 0.03 * scale for _, _, _, depth in defects[:, 0]
            )
    moments = cv2.HuMoments(cv2.moments(contour)).flatten()
    hu = tuple(
        -math.copysign(1.0, value) * math.log10(abs(value))
        if abs(value) > 1e-12
        else 0.0
        for value in moments
    )
    return {
        "detected": True,
        "area": area,
        "solidity": area / hull_area,
        "deep_convexity_defects": deep_defects,
        "skeleton_endpoints": _skeleton_endpoints(component),
        "hu": hu,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hand-roi",
        type=float,
        nargs=4,
        default=(0.18, 0.13, 0.28, 0.43),
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
    )
    parser.add_argument("--sample-fps", type=float, default=8.0)
    args = parser.parse_args()
    if not args.video.is_file():
        raise ValueError(f"video does not exist: {args.video}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.sample_fps <= 0:
        raise ValueError("sample FPS must be positive")

    capture = cv2.VideoCapture(str(args.video))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    stride = max(1, round(source_fps / args.sample_fps))
    descriptors: list[dict[str, object]] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            descriptors.append(_descriptor(frame, tuple(args.hand_roi)))
        index += 1
    capture.release()
    detected = [item for item in descriptors if item.get("detected")]
    if not descriptors:
        raise ValueError("video has no decodable frames")
    coverage = len(detected) / len(descriptors)
    if detected:
        areas = [float(item["area"]) for item in detected]
        solidities = [float(item["solidity"]) for item in detected]
        endpoints = [int(item["skeleton_endpoints"]) for item in detected]
        defects = [int(item["deep_convexity_defects"]) for item in detected]
        hu_vectors = np.array([item["hu"] for item in detected], dtype=np.float64)
        hu_drift = float(np.mean(np.linalg.norm(hu_vectors - np.median(hu_vectors, axis=0), axis=1)))
        area_ratio = max(areas) / max(min(areas), 1.0)
    else:
        solidities = []
        endpoints = []
        defects = []
        hu_drift = math.inf
        area_ratio = math.inf
    metrics = {
        "tracking_coverage": coverage,
        "area_ratio": area_ratio,
        "minimum_solidity": min(solidities) if solidities else 0.0,
        "maximum_skeleton_endpoints": max(endpoints) if endpoints else None,
        "median_skeleton_endpoints": float(np.median(endpoints)) if endpoints else None,
        "maximum_deep_convexity_defects": max(defects) if defects else None,
        "mean_hu_drift": hu_drift,
    }
    gates = {
        "hand_detected": coverage >= 0.75,
        "area_stable": area_ratio <= 3.0,
        "rigid_solidity": metrics["minimum_solidity"] >= 0.42,
        "limited_branches": bool(endpoints) and max(endpoints) <= 8,
        "limited_finger_gaps": bool(defects) and max(defects) <= 4,
        "shape_stable": hu_drift <= 12.0,
    }
    payload = {
        "schema_version": "1.0.0",
        "video": str(args.video.resolve()),
        "video_sha256": _sha256(args.video),
        "hand_roi": args.hand_roi,
        "sample_fps": args.sample_fps,
        "metrics": metrics,
        "gates": gates,
        "accepted": all(gates.values()),
        "limitations": [
            "The verifier measures the configured active-hand ROI, not full 3D kinematics.",
            "White low-saturation segmentation is specific to Sunday Memo morphology.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
