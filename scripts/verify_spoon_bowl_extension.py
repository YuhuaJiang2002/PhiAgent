#!/usr/bin/env python3
"""Phase-aware verifier for robot hand, bowl, and spoon extension videos."""

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


def _frames(path: Path, width: int, height: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    capture.release()
    if not frames:
        raise ValueError(f"video has no decodable frames: {path}")
    return frames


def _components(mask: np.ndarray, minimum_area: int) -> list[tuple[int, int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [
        tuple(int(value) for value in stats[index])
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area
    ]


def _largest(mask: np.ndarray, minimum_area: int) -> tuple[np.ndarray, tuple[int, int] | None]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    eligible = [
        index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area
    ]
    if not eligible:
        return np.zeros_like(mask, dtype=np.uint8), None
    selected = max(eligible, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    center = tuple(int(round(value)) for value in centroids[selected])
    return (labels == selected).astype(np.uint8), center


def _identity_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    left = left[: round(0.72 * left.shape[0])]
    right = right[: round(0.72 * right.shape[0])]
    left_centered = left - float(left.mean())
    right_centered = right - float(right.mean())
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    correlation = (
        float(np.sum(left_centered * right_centered)) / denominator
        if denominator > 1e-6
        else 0.0
    )
    mae = float(np.mean(np.abs(left - right))) / 255.0
    return max(0.0, min(1.0, 0.6 * ((correlation + 1) / 2) + 0.4 * (1 - mae)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=128)
    args = parser.parse_args()
    for path in (args.candidate, args.reference, args.control):
        if not path.is_file():
            raise ValueError(f"input does not exist: {path}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    candidate = _frames(args.candidate, args.width, args.height)
    control = _frames(args.control, args.width, args.height)
    reference = cv2.imread(str(args.reference), cv2.IMREAD_COLOR)
    if reference is None:
        raise ValueError(f"reference image is invalid: {args.reference}")
    reference = cv2.resize(
        reference, (args.width, args.height), interpolation=cv2.INTER_AREA
    )
    count = min(len(candidate), len(control))
    candidate = candidate[:count]
    control = control[:count]

    bowl_centers: list[tuple[int, int] | None] = []
    spoon_centers: list[tuple[int, int] | None] = []
    bowl_masks: list[np.ndarray] = []
    spoon_masks: list[np.ndarray] = []
    spoon_components: list[int] = []
    spoon_areas: list[int] = []
    hand_contact: list[bool] = []
    hand_spoon_contact: list[bool] = []
    for frame in candidate:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bowl_raw = cv2.inRange(hsv, (12, 65, 55), (42, 255, 255)) > 0
        spoon_raw = cv2.inRange(hsv, (72, 30, 30), (108, 255, 255)) > 0
        interaction_top = round(0.52 * frame.shape[0])
        interaction_left = round(0.18 * frame.shape[1])
        interaction_right = round(0.70 * frame.shape[1])
        for mask in (bowl_raw, spoon_raw):
            mask[:interaction_top] = False
            mask[:, :interaction_left] = False
            mask[:, interaction_right:] = False
        bowl_mask, bowl_center = _largest(bowl_raw, 30)
        spoon_mask, spoon_center = _largest(spoon_raw, 12)
        bowl_masks.append(bowl_mask)
        spoon_masks.append(spoon_mask)
        bowl_centers.append(bowl_center)
        spoon_centers.append(spoon_center)
        components = _components(spoon_raw, 12)
        component_areas = sorted((component[4] for component in components), reverse=True)
        spoon_components.append(
            int(
                len(component_areas) > 1
                and component_areas[1] >= 0.25 * component_areas[0]
            )
        )
        spoon_areas.append(int(spoon_mask.sum()))

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        robot_white = ((saturation < 65) & (value > 150)).astype(np.uint8)
        bowl_ring = cv2.dilate(
            bowl_mask, np.ones((9, 9), dtype=np.uint8), iterations=1
        ) - bowl_mask
        hand_contact.append(bool(np.sum((bowl_ring > 0) & (robot_white > 0)) >= 12))
        spoon_ring = cv2.dilate(
            spoon_mask, np.ones((11, 11), dtype=np.uint8), iterations=1
        )
        hand_spoon_contact.append(
            bool(np.sum((spoon_ring > 0) & (robot_white > 0)) >= 8)
        )

    bowl_coverage = sum(center is not None for center in bowl_centers) / count
    spoon_coverage = sum(center is not None for center in spoon_centers) / count
    pre_insertion_count = max(3, round(0.70 * count))
    positive_areas = [area for area in spoon_areas[:pre_insertion_count] if area]
    spoon_area_ratio = (
        max(positive_areas) / min(positive_areas) if positive_areas else math.inf
    )
    duplicate_spoon_frames = sum(spoon_components)
    final_containment_values: list[bool] = []
    for bowl_mask, spoon_center in zip(bowl_masks[-3:], spoon_centers[-3:]):
        if spoon_center is None:
            final_containment_values.append(False)
            continue
        x, y = spoon_center
        expanded = cv2.dilate(
            bowl_mask, np.ones((21, 21), dtype=np.uint8), iterations=1
        )
        final_containment_values.append(bool(expanded[y, x]))

    identity_scores = [
        _identity_similarity(reference, frame)
        for frame in (candidate[0], candidate[count // 2], candidate[-1])
    ]
    grayscale = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        for frame in candidate
    ]
    temporal_jerk = float(
        np.mean(
            [
                np.mean(np.abs(following - 2 * current + previous))
                for previous, current, following in zip(
                    grayscale, grayscale[1:], grayscale[2:]
                )
            ]
        )
    )
    control_edges = [
        cv2.Canny(frame, 50, 150).astype(np.float32) / 255.0 for frame in control
    ]
    candidate_edges = [
        cv2.Canny(frame, 50, 150).astype(np.float32) / 255.0
        for frame in candidate
    ]
    control_edge_error = float(
        np.mean(
            [
                np.mean(np.abs(left - right))
                for left, right in zip(control_edges, candidate_edges)
            ]
        )
    )

    metrics = {
        "frame_count": count,
        "bowl_tracking_coverage": bowl_coverage,
        "spoon_tracking_coverage": spoon_coverage,
        "spoon_area_ratio": spoon_area_ratio,
        "duplicate_spoon_frames": duplicate_spoon_frames,
        "hand_bowl_contact_fraction": sum(hand_contact) / count,
        "hand_spoon_contact_fraction": sum(hand_spoon_contact) / count,
        "final_spoon_containment_fraction": sum(final_containment_values)
        / len(final_containment_values),
        "minimum_robot_identity": min(identity_scores),
        "temporal_jerk": temporal_jerk,
        "control_edge_error": control_edge_error,
    }
    gates = {
        "bowl_present": bowl_coverage >= 0.45,
        "single_spoon": duplicate_spoon_frames == 0,
        "spoon_persistent": spoon_coverage >= 0.80,
        "spoon_shape_stable": spoon_area_ratio <= 3.0,
        "hand_holds_bowl": sum(hand_contact[-max(3, count // 3) :])
        / max(3, count // 3)
        >= 0.70,
        "hand_moves_spoon": sum(hand_spoon_contact[-max(3, count // 3) :])
        / max(3, count // 3)
        >= 0.50,
        "spoon_in_bowl": all(final_containment_values),
        "same_robot_identity": min(identity_scores) >= 0.78,
        "temporal_stability": temporal_jerk <= 0.055,
        "control_alignment": control_edge_error <= 0.24,
    }
    payload = {
        "schema_version": "1.0.0",
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": _sha256(args.candidate),
        "reference": str(args.reference.resolve()),
        "control": str(args.control.resolve()),
        "metrics": metrics,
        "gates": gates,
        "accepted": all(gates.values()),
        "limitations": [
            "Color-instance gates assume the configured yellow bowl and cyan spoon.",
            "Image-space contact and containment do not prove physical support.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
