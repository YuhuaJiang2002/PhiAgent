#!/usr/bin/env python3
"""Evaluate one AC-WM candidate with explicit proxy and human-review boundaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _largest_yellow(cv2: Any, np: Any, frame: Any) -> tuple[tuple[float, float] | None, int]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (10, 70, 55), (43, 255, 255))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    valid = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= 80]
    if not valid:
        return None, 0
    largest = max(valid, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    center = centroids[largest]
    return (float(center[0]), float(center[1])), len(valid)


def _decode(cv2: Any, path: Path, size: tuple[int, int] | None = None) -> list[Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if size is not None and frame.shape[1::-1] != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    capture.release()
    if len(frames) < 5:
        raise RuntimeError(f"candidate contains too few frames: {path}")
    return frames


def _median_center(np: Any, values: list[tuple[float, float]]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array[:, 0])), float(np.median(array[:, 1]))


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def action_adherence_score(
    expected_delta: tuple[float, float], observed_delta: tuple[float, float]
) -> tuple[float, float, float, float]:
    """Score directional progress without requiring an exact pixel endpoint."""

    expected_norm = math.hypot(*expected_delta)
    observed_norm = math.hypot(*observed_delta)
    if expected_norm < 1e-6 or observed_norm < 1e-6:
        return 0.0, 0.0, math.inf, 0.0
    unit = (expected_delta[0] / expected_norm, expected_delta[1] / expected_norm)
    projected = observed_delta[0] * unit[0] + observed_delta[1] * unit[1]
    lateral = abs(observed_delta[0] * unit[1] - observed_delta[1] * unit[0])
    progress = _clamp(projected / expected_norm)
    lateral_score = math.exp(-lateral / max(20.0, 0.6 * expected_norm))
    cosine = _clamp(projected / observed_norm)
    score = _clamp((0.75 * progress + 0.25 * lateral_score) * cosine)
    return score, projected / expected_norm, lateral, cosine


def _storyboard(cv2: Any, np: Any, frames: list[Any], output: Path) -> None:
    indices = [round(index * (len(frames) - 1) / 5) for index in range(6)]
    cells = []
    for index in indices:
        cell = cv2.resize(frames[index], (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(
            cell,
            f"frame {index}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cells.append(cell)
    sheet = np.vstack((np.hstack(cells[:3]), np.hstack(cells[3:])))
    cv2.imwrite(str(output), sheet)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--first-frame", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--human-review", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    candidate = args.candidate.expanduser().resolve()
    condition_path = args.condition.expanduser().resolve()
    first_frame_path = args.first_frame.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    source = args.source.expanduser().resolve()
    for path in (candidate, condition_path, first_frame_path, metadata_path, source):
        if not path.is_file():
            raise ValueError(f"required evaluation input does not exist: {path}")
    evidence_root = (
        args.evidence_root.expanduser().resolve()
        if args.evidence_root
        else candidate.parent / "evaluation" / candidate.stem
    )
    if evidence_root.exists():
        raise FileExistsError(f"evaluation evidence already exists: {evidence_root}")
    evidence_root.mkdir(parents=True)

    import cv2
    import numpy as np

    first = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"cannot decode first frame: {first_frame_path}")
    size = first.shape[1], first.shape[0]
    frames = _decode(cv2, candidate, size)
    condition = json.loads(condition_path.read_text())
    channels = list(condition["channels"])
    values = condition["values"]
    try:
        object_x = channels.index("object_center_x_px")
        object_y = channels.index("object_center_y_px")
    except ValueError as exc:
        raise ValueError("condition lacks explicit object-center target channels") from exc
    expected_start = (float(values[0][object_x]), float(values[0][object_y]))
    expected_terminal = (float(values[-1][object_x]), float(values[-1][object_y]))

    centers: list[tuple[float, float] | None] = []
    component_counts: list[int] = []
    for frame in frames:
        center, count = _largest_yellow(cv2, np, frame)
        centers.append(center)
        component_counts.append(count)
    valid = [center for center in centers if center is not None]
    detection_fraction = len(valid) / len(frames)
    terminal_valid = [center for center in centers[-max(3, len(frames) // 6) :] if center]
    if terminal_valid:
        observed_terminal = _median_center(np, terminal_valid)
        observed_terminal_payload: tuple[float, float] | None = observed_terminal
        endpoint_error = math.dist(observed_terminal, expected_terminal)
    else:
        observed_terminal = expected_start
        observed_terminal_payload = None
        endpoint_error = math.hypot(*size) * 10.0
    start_valid = [center for center in centers[: max(3, len(frames) // 6)] if center]
    observed_start = _median_center(np, start_valid) if start_valid else expected_start
    expected_delta = (
        expected_terminal[0] - expected_start[0],
        expected_terminal[1] - expected_start[1],
    )
    observed_delta = (
        observed_terminal[0] - observed_start[0],
        observed_terminal[1] - observed_start[1],
    )
    (
        action_adherence,
        projected_progress,
        lateral_error,
        direction_cosine,
    ) = action_adherence_score(expected_delta, observed_delta)
    direction_passed = projected_progress > 0

    duplicate_fraction = sum(count > 1 for count in component_counts) / len(component_counts)
    object_interaction = _clamp(detection_fraction * (1.0 - duplicate_fraction))
    first_background = first[:120]
    first_yellow = cv2.inRange(
        cv2.cvtColor(first_background, cv2.COLOR_BGR2HSV), (10, 70, 55), (43, 255, 255)
    )
    exclusion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    first_yellow = cv2.dilate(first_yellow, exclusion_kernel)
    background_mads = []
    for frame in frames:
        current = frame[:120]
        current_yellow = cv2.inRange(
            cv2.cvtColor(current, cv2.COLOR_BGR2HSV), (10, 70, 55), (43, 255, 255)
        )
        excluded = cv2.bitwise_or(
            first_yellow, cv2.dilate(current_yellow, exclusion_kernel)
        )
        valid_background = excluded == 0
        difference = cv2.absdiff(current, first_background)
        background_mads.append(
            float(np.mean(difference[valid_background]))
            if np.any(valid_background)
            else 255.0
        )
    background_consistency = _clamp(math.exp(-float(np.mean(background_mads)) / 28.0))

    edge_reference = cv2.Canny(first[:, first.shape[1] // 2 :], 70, 150)
    reference_edges = max(1, int(np.count_nonzero(edge_reference)))
    edge_log_errors = []
    for frame in frames:
        edges = cv2.Canny(frame[:, frame.shape[1] // 2 :], 70, 150)
        edge_log_errors.append(
            abs(math.log((max(1, int(np.count_nonzero(edges)))) / reference_edges))
        )
    embodiment_consistency = _clamp(math.exp(-float(np.median(edge_log_errors))))

    differences = [
        float(np.mean(cv2.absdiff(previous, current)))
        for previous, current in zip(frames, frames[1:])
    ]
    median_difference = max(1e-6, float(np.median(differences)))
    transition_ratio = float(np.percentile(differences, 95)) / median_difference
    temporal_consistency = _clamp(1.0 / (1.0 + max(0.0, transition_ratio - 3.0) / 5.0))

    human_review: bool | None = None
    review_payload: dict[str, Any] | None = None
    if args.human_review is not None:
        review_path = args.human_review.expanduser().resolve()
        review_payload = json.loads(review_path.read_text())
        if review_payload.get("case_id") != condition["label"]:
            raise ValueError("human review case_id does not match the condition")
        if review_payload.get("passed") not in {True, False}:
            raise ValueError("human review must contain a boolean passed field")
        human_review = bool(review_payload["passed"])

    storyboard = evidence_root / "storyboard.jpg"
    _storyboard(cv2, np, frames, storyboard)
    diagnoses = []
    if action_adherence < 0.75:
        diagnoses.append("object terminal state does not follow the explicit action target")
    if embodiment_consistency < 0.75:
        diagnoses.append("robot-region edge structure drifts from the first-frame embodiment")
    if object_interaction < 0.75:
        diagnoses.append("yellow object is missing or duplicated in generated frames")
    if temporal_consistency < 0.75:
        diagnoses.append("candidate has high transition outliers")
    if background_consistency < 0.75:
        diagnoses.append("fixed upper background drifts from the real first frame")
    if human_review is None:
        diagnoses.append("mandatory human review is pending")
    elif not human_review:
        diagnoses.append("mandatory human review rejected the visual result")
    evidence = evidence_root / "evaluation.json"
    payload = {
        "schema_version": "1.0.0",
        "evaluator": "phiagent-acwm-bowl-proxy-v2",
        "action_adherence": action_adherence,
        "embodiment_consistency": embodiment_consistency,
        "object_interaction": object_interaction,
        "temporal_consistency": temporal_consistency,
        "background_consistency": background_consistency,
        "human_review_passed": human_review,
        "diagnoses": diagnoses,
        "evidence": str(evidence),
        "storyboard": str(storyboard),
        "metrics": {
            "detected_frame_fraction": detection_fraction,
            "duplicate_component_fraction": duplicate_fraction,
            "expected_start_xy": expected_start,
            "expected_terminal_xy": expected_terminal,
            "observed_start_xy": observed_start,
            "observed_terminal_xy": observed_terminal_payload,
            "terminal_error_pixels": endpoint_error,
            "terminal_direction_passed": direction_passed,
            "projected_progress_ratio": projected_progress,
            "lateral_error_pixels": lateral_error,
            "direction_cosine": direction_cosine,
            "upper_background_mean_mad": float(np.mean(background_mads)),
            "transition_p95_to_median_ratio": transition_ratio,
        },
        "human_review": review_payload,
        "limitations": [
            "Robot edge stability is only an image proxy, not morphology or joint-pose validation.",
            "Yellow-object tracking does not establish grasp force, contact causality, or 3-D lift.",
            "No candidate is accepted unless human_review_passed is explicitly true.",
        ],
    }
    evidence.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
