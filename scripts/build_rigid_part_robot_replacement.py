#!/usr/bin/env python3
"""Drive a rigid robot torso and segmented arms from real pose landmarks.

This is a CPU-only deterministic compositor.  It uses one sharp generated robot
anchor as texture, removes the anchor arms from a rigid clean base, and moves
upper-arm, forearm, and hand pieces with similarity transforms derived from the
source shoulder/elbow/wrist trajectories.  No source pixel is restored inside
the conservative person-clear support.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_multi_anchor_robot_replacement import (  # noqa: E402
    _git_state,
    _package_versions,
    _sha256,
    _source_info,
    _write_json,
    _writer,
)
from scripts.compose_h3_layered_replacement import (  # noqa: E402
    _load_packed,
    _skin_like,
    _strict_flower_seed,
)


POSE_INDICES = {
    "left": (11, 13, 15, 19),
    "right": (12, 14, 16, 20),
}


def _gaussian_smooth(np: Any, values: Any, sigma: float = 2.0) -> Any:
    radius = max(1, round(sigma * 3))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    result = np.empty_like(values, dtype=np.float64)
    for landmark in range(values.shape[1]):
        for coordinate in range(values.shape[2]):
            series = values[:, landmark, coordinate]
            padded = np.pad(series, (radius, radius), mode="edge")
            result[:, landmark, coordinate] = np.convolve(
                padded, kernel, mode="valid"
            )
    return result


def _centered_temporal_median(np: Any, values: Any, radius: int) -> Any:
    if radius < 1:
        raise ValueError("temporal median radius must be positive")
    result = np.empty_like(values, dtype=np.float64)
    for frame in range(len(values)):
        left = max(0, frame - radius)
        right = min(len(values), frame + radius + 1)
        result[frame] = np.median(values[left:right], axis=0)
    return result


def _robust_pose_filter(
    np: Any,
    raw: Any,
    *,
    median_radius: int,
    smoothing_sigma: float,
    outlier_threshold_pixels: float,
) -> tuple[Any, Any, Any]:
    """Reject image-space landmark jumps before zero-phase smoothing."""

    if outlier_threshold_pixels <= 0:
        raise ValueError("outlier threshold must be positive")
    observations = np.array(raw, dtype=np.float64, copy=True)
    if median_radius:
        robust = _centered_temporal_median(np, observations, median_radius)
    else:
        robust = observations.copy()
    difference = np.subtract(observations, robust)
    residual = np.linalg.norm(difference, axis=2)
    outliers = residual > outlier_threshold_pixels
    smoothed = _gaussian_smooth(np, robust, sigma=smoothing_sigma)
    return smoothed, robust, outliers


def _smooth_scalar_series(np: Any, values: Any, sigma: float) -> Any:
    if sigma <= 0:
        raise ValueError("scalar smoothing sigma must be positive")
    radius = max(1, round(sigma * 3))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _zero_phase_bounded_steps(np: Any, values: Any, maximum_step: float) -> Any:
    """Bound scalar frame steps with symmetric passes so no one-way lag is added."""

    if maximum_step <= 0:
        raise ValueError("maximum scalar step must be positive")
    source = np.asarray(values, dtype=np.float64)
    forward = source.copy()
    for frame in range(1, len(forward)):
        forward[frame] = forward[frame - 1] + np.clip(
            source[frame] - forward[frame - 1], -maximum_step, maximum_step
        )
    backward = source.copy()
    for frame in range(len(backward) - 2, -1, -1):
        backward[frame] = backward[frame + 1] + np.clip(
            source[frame] - backward[frame + 1], -maximum_step, maximum_step
        )
    return 0.5 * (forward + backward)


def _zero_phase_bounded_vector_steps(
    np: Any, values: Any, maximum_step: float
) -> Any:
    """Bound 2D displacement norms with symmetric passes and no one-way lag."""

    if maximum_step <= 0:
        raise ValueError("maximum vector step must be positive")
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("vector series must have shape (frames, 2)")

    def bounded(sequence: Any) -> Any:
        result = sequence.copy()
        for frame in range(1, len(result)):
            delta = result[frame] - result[frame - 1]
            length = float(np.linalg.norm(delta))
            if length > maximum_step:
                result[frame] = result[frame - 1] + delta * (
                    maximum_step / length
                )
        return result

    forward = bounded(source)
    backward = bounded(source[::-1])[::-1]
    return 0.5 * (forward + backward)


def _stable_segment_angles(
    np: Any,
    starts: Any,
    ends: Any,
    *,
    minimum_length_pixels: float,
    median_radius: int,
    smoothing_sigma: float,
) -> tuple[Any, dict[str, Any]]:
    """Estimate continuous angles without trusting nearly zero-length vectors."""

    if minimum_length_pixels <= 0:
        raise ValueError("minimum reliable segment length must be positive")
    vectors = np.asarray(ends, dtype=np.float64) - np.asarray(
        starts, dtype=np.float64
    )
    lengths = np.linalg.norm(vectors, axis=1)
    reliable = lengths >= minimum_length_pixels
    if not np.any(reliable):
        raise RuntimeError("no reliable segment directions were observed")
    frames = np.arange(len(vectors), dtype=np.float64)
    raw_angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    reliable_angles = np.unwrap(raw_angles[reliable])
    interpolated = np.interp(frames, frames[reliable], reliable_angles)
    if median_radius:
        robust = np.asarray(
            [
                np.median(
                    interpolated[
                        max(0, frame - median_radius) : min(
                            len(interpolated), frame + median_radius + 1
                        )
                    ]
                )
                for frame in range(len(interpolated))
            ],
            dtype=np.float64,
        )
    else:
        robust = interpolated
    smoothed = _smooth_scalar_series(np, robust, smoothing_sigma)
    return smoothed, {
        "minimum_reliable_length_pixels": minimum_length_pixels,
        "reliable_frames": int(np.count_nonzero(reliable)),
        "interpolated_frames": int(np.count_nonzero(~reliable)),
        "minimum_observed_length_pixels": float(np.min(lengths)),
        "maximum_angle_step_degrees": float(
            np.degrees(np.max(np.abs(np.diff(smoothed))))
        ),
    }


def _stable_hand_angles(
    np: Any,
    tracks: Any,
    *,
    minimum_forearm_length_pixels: float,
    minimum_hand_length_pixels: float,
    median_radius: int,
    smoothing_sigma: float,
    maximum_step_degrees: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep wrist articulation while bridging unreliable wrist-index directions."""

    angles: dict[str, Any] = {}
    records: dict[str, Any] = {}
    for side, (_, elbow, wrist, hand) in POSE_INDICES.items():
        forearm, forearm_record = _stable_segment_angles(
            np,
            tracks[:, elbow],
            tracks[:, wrist],
            minimum_length_pixels=minimum_forearm_length_pixels,
            median_radius=median_radius,
            smoothing_sigma=smoothing_sigma,
        )
        hand_vector = tracks[:, hand] - tracks[:, wrist]
        hand_lengths = np.linalg.norm(hand_vector, axis=1)
        reliable = hand_lengths >= minimum_hand_length_pixels
        if not np.any(reliable):
            raise RuntimeError(f"no reliable {side} hand directions were observed")
        raw_hand = np.arctan2(hand_vector[:, 1], hand_vector[:, 0])
        relative = np.angle(np.exp(1j * (raw_hand - forearm)))
        frames = np.arange(len(tracks), dtype=np.float64)
        relative_interpolated = np.interp(
            frames,
            frames[reliable],
            np.unwrap(relative[reliable]),
        )
        if median_radius:
            relative_robust = np.asarray(
                [
                    np.median(
                        relative_interpolated[
                            max(0, frame - median_radius) : min(
                                len(relative_interpolated),
                                frame + median_radius + 1,
                            )
                        ]
                    )
                    for frame in range(len(relative_interpolated))
                ],
                dtype=np.float64,
            )
        else:
            relative_robust = relative_interpolated
        relative_smoothed = _smooth_scalar_series(
            np, relative_robust, smoothing_sigma
        )
        angles[side] = _zero_phase_bounded_steps(
            np,
            forearm + relative_smoothed,
            math.radians(maximum_step_degrees),
        )
        records[side] = {
            "method": "reliable_wrist_index_relative_to_stable_forearm",
            "forearm": forearm_record,
            "minimum_reliable_hand_length_pixels": minimum_hand_length_pixels,
            "reliable_hand_frames": int(np.count_nonzero(reliable)),
            "interpolated_hand_frames": int(np.count_nonzero(~reliable)),
            "minimum_observed_hand_length_pixels": float(np.min(hand_lengths)),
            "maximum_hand_angle_step_degrees": float(
                np.degrees(np.max(np.abs(np.diff(angles[side]))))
            ),
            "configured_maximum_hand_angle_step_degrees": maximum_step_degrees,
        }
    return angles, records


def _interpolate_missing(
    np: Any, tracks: list[Any | None], landmark_count: int = 33
) -> Any:
    values = np.full((len(tracks), landmark_count, 2), np.nan, dtype=np.float64)
    for frame, points in enumerate(tracks):
        if points is not None:
            values[frame] = points
    frame_axis = np.arange(len(tracks), dtype=np.float64)
    for landmark in range(landmark_count):
        for coordinate in range(2):
            series = values[:, landmark, coordinate]
            valid = np.isfinite(series)
            if not np.count_nonzero(valid):
                raise RuntimeError(f"pose landmark {landmark} was never detected")
            values[:, landmark, coordinate] = np.interp(
                frame_axis, frame_axis[valid], series[valid]
            )
    return values


def _load_reusable_pose_trajectory(
    np: Any,
    trajectory_path: Path,
    tracking_manifest_path: Path,
    *,
    expected_frames: int,
) -> tuple[Any, dict[str, Any], Any, Any]:
    """Reuse an immutable audited trajectory without rerunning its detector."""

    trajectory = json.loads(trajectory_path.read_text())
    manifest = json.loads(tracking_manifest_path.read_text())
    indices = [int(index) for index in trajectory["landmark_indices"]]
    if not set(POSE_INDICES["left"] + POSE_INDICES["right"]).issubset(indices):
        raise ValueError("reusable trajectory is missing an arm or hand landmark")
    selected_target = np.asarray(trajectory["robot_target_xy"], dtype=np.float64)
    selected_raw = np.asarray(trajectory["raw_interpolated_xy"], dtype=np.float64)
    selected_robust = np.asarray(trajectory["robust_median_xy"], dtype=np.float64)
    expected_shape = (expected_frames, len(indices), 2)
    for name, values in (
        ("robot target", selected_target),
        ("raw interpolated", selected_raw),
        ("robust median", selected_robust),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"reusable {name} shape {values.shape} does not match {expected_shape}"
            )
    if trajectory["temporal_index_map"] != list(range(expected_frames)):
        raise ValueError("reusable trajectory does not preserve the source timeline")

    def expand(selected: Any) -> Any:
        full = np.zeros((expected_frames, 33, 2), dtype=np.float64)
        full[:, indices] = selected
        shoulder_width = np.linalg.norm(full[:, 11] - full[:, 12], axis=1)
        torso_drop = np.column_stack(
            (np.zeros(expected_frames), 1.8 * shoulder_width)
        )
        # Hips are used only to form an additional person-clear polygon. The
        # immutable safety union remains the hard source-person exclusion mask.
        full[:, 23] = full[:, 11] + torso_drop
        full[:, 24] = full[:, 12] + torso_drop
        return full

    metrics = manifest["metrics"]
    tracking = {
        key: metrics[key]
        for key in (
            "decoded_frames",
            "missing_pose_frames",
            "low_confidence_frames",
            "smoothing",
            "outlier_repair",
            "correspondence",
        )
    }
    tracking["trajectory_reuse"] = {
        "used": True,
        "trajectory": str(trajectory_path),
        "trajectory_sha256": _sha256(trajectory_path),
        "tracking_manifest": str(tracking_manifest_path),
        "tracking_manifest_sha256": _sha256(tracking_manifest_path),
        "coordinate_frame": trajectory["coordinate_frame"],
        "frame_count": trajectory["frame_count"],
    }
    return (
        expand(selected_target),
        tracking,
        expand(selected_raw),
        expand(selected_robust),
    )


def _fill_missing(
    np: Any,
    tracks: list[Any | None],
    landmark_count: int = 33,
    smoothing_sigma: float = 2.0,
) -> Any:
    if smoothing_sigma <= 0:
        raise ValueError("smoothing sigma must be positive")
    return _gaussian_smooth(
        np,
        _interpolate_missing(np, tracks, landmark_count),
        sigma=smoothing_sigma,
    )


def _trajectory_correspondence_metrics(
    np: Any,
    raw: Any,
    smoothed: Any,
    *,
    width: int,
    height: int,
    smoothing_sigma: float,
    robust: Any | None = None,
    outliers: Any | None = None,
    action_horizon_frames: int = 8,
) -> dict[str, float | int]:
    """Measure same-timeline inlier fidelity and action-scale correspondence."""

    selected = sorted({index for indices in POSE_INDICES.values() for index in indices})
    raw_selected = raw[:, selected]
    smooth_selected = smoothed[:, selected]
    robust_selected = (
        robust[:, selected] if robust is not None else raw_selected
    )
    selected_outliers = (
        outliers[:, selected]
        if outliers is not None
        else np.zeros(raw_selected.shape[:2], dtype=bool)
    )
    inliers = ~selected_outliers
    diagonal = math.hypot(width, height)
    error = np.linalg.norm(smooth_selected - raw_selected, axis=2)
    raw_velocity = np.diff(raw_selected, axis=0)
    smooth_velocity = np.diff(smooth_selected, axis=0)
    raw_speed = np.mean(np.linalg.norm(raw_velocity, axis=2), axis=1)
    smooth_speed = np.mean(np.linalg.norm(smooth_velocity, axis=2), axis=1)
    raw_acceleration = np.diff(raw_velocity, axis=0)
    smooth_acceleration = np.diff(smooth_velocity, axis=0)
    raw_jerk = np.diff(raw_acceleration, axis=0)
    smooth_jerk = np.diff(smooth_acceleration, axis=0)

    horizon = min(max(1, action_horizon_frames), len(raw_selected) - 1)
    raw_action = raw_selected[horizon:] - raw_selected[:-horizon]
    smooth_action = smooth_selected[horizon:] - smooth_selected[:-horizon]
    action_inliers = inliers[horizon:] & inliers[:-horizon]
    numerator = np.sum(raw_action * smooth_action, axis=2)
    raw_action_magnitude = np.linalg.norm(raw_action, axis=2)
    smooth_action_magnitude = np.linalg.norm(smooth_action, axis=2)
    denominator = raw_action_magnitude * smooth_action_magnitude
    active = action_inliers & (denominator > 5.0)
    action_direction_cosine = (
        float(np.mean(numerator[active] / denominator[active]))
        if np.any(active)
        else 1.0
    )
    if (
        np.count_nonzero(active) < 2
        or float(np.std(raw_action_magnitude[active])) < 1e-9
        or float(np.std(smooth_action_magnitude[active])) < 1e-9
    ):
        action_magnitude_correlation = 1.0
    else:
        action_magnitude_correlation = float(
            np.corrcoef(
                raw_action_magnitude[active], smooth_action_magnitude[active]
            )[0, 1]
        )

    max_lag = 6
    centered_raw = raw_speed - np.mean(raw_speed)
    centered_smooth = smooth_speed - np.mean(smooth_speed)
    lag_scores: list[tuple[float, int]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            first, second = centered_raw[-lag:], centered_smooth[:lag]
        elif lag > 0:
            first, second = centered_raw[:-lag], centered_smooth[lag:]
        else:
            first, second = centered_raw, centered_smooth
        denominator_lag = float(np.linalg.norm(first) * np.linalg.norm(second))
        score = float(np.dot(first, second) / denominator_lag) if denominator_lag else 1.0
        lag_scores.append((score, lag))
    best_lag = max(lag_scores)[1]

    neighborhood = max(1, round(3 * smoothing_sigma))
    excess_velocity_frames = 0
    for index, value in enumerate(smooth_speed):
        left = max(0, index - neighborhood)
        right = min(len(raw_speed), index + neighborhood + 1)
        if value > float(np.max(raw_speed[left:right])) + 1e-9:
            excess_velocity_frames += 1

    def maximum_norm(values: Any) -> float:
        if not values.size:
            return 0.0
        return float(np.max(np.linalg.norm(values, axis=2))) / diagonal

    inlier_error = error[inliers]
    robust_error = np.linalg.norm(smooth_selected - robust_selected, axis=2)
    return {
        "inlier_pose_rms_deviation_fraction_of_diagonal": float(
            np.sqrt(np.mean(inlier_error**2)) / diagonal
        ),
        "inlier_pose_maximum_deviation_fraction_of_diagonal": float(
            np.max(inlier_error) / diagonal
        ),
        "robust_target_rms_smoothing_fraction_of_diagonal": float(
            np.sqrt(np.mean(robust_error**2)) / diagonal
        ),
        "detected_outlier_landmark_observations": int(np.count_nonzero(selected_outliers)),
        "detected_outlier_frames": int(
            np.count_nonzero(np.any(selected_outliers, axis=1))
        ),
        "action_horizon_frames": int(horizon),
        "action_direction_cosine": action_direction_cosine,
        "action_magnitude_correlation": action_magnitude_correlation,
        "best_motion_energy_lag_frames": int(best_lag),
        "maximum_raw_velocity_fraction_of_diagonal": maximum_norm(raw_velocity),
        "maximum_smoothed_velocity_fraction_of_diagonal": maximum_norm(smooth_velocity),
        "maximum_raw_acceleration_fraction_of_diagonal": maximum_norm(raw_acceleration),
        "maximum_smoothed_acceleration_fraction_of_diagonal": maximum_norm(
            smooth_acceleration
        ),
        "maximum_raw_jerk_fraction_of_diagonal": maximum_norm(raw_jerk),
        "maximum_smoothed_jerk_fraction_of_diagonal": maximum_norm(smooth_jerk),
        "smoothed_velocity_exceeds_local_source_frames": excess_velocity_frames,
        "temporal_index_map_mismatch_frames": 0,
    }


def _track_pose(
    *,
    cv2: Any,
    np: Any,
    mp: Any,
    source: Path,
    model: Path,
    fps: float,
    width: int,
    height: int,
    smoothing_sigma: float = 2.0,
    temporal_median_radius: int = 0,
    outlier_threshold_pixels: float = 20.0,
    action_horizon_frames: int = 8,
    return_raw_tracks: bool = False,
) -> tuple[Any, dict[str, Any]]:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.30,
        min_pose_presence_confidence=0.30,
        min_tracking_confidence=0.30,
        output_segmentation_masks=False,
    )
    capture = cv2.VideoCapture(str(source))
    tracks: list[Any | None] = []
    missing = 0
    low_confidence = 0
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = round(frame_index * 1000.0 / fps)
                result = landmarker.detect_for_video(image, timestamp)
                if not result.pose_landmarks:
                    tracks.append(None)
                    missing += 1
                else:
                    landmarks = result.pose_landmarks[0]
                    points = np.asarray(
                        [(item.x * width, item.y * height) for item in landmarks],
                        dtype=np.float64,
                    )
                    confidence = min(
                        float(getattr(landmarks[index], "visibility", 1.0))
                        for side in POSE_INDICES.values()
                        for index in side[:3]
                    )
                    if confidence < 0.15:
                        low_confidence += 1
                    tracks.append(points)
                frame_index += 1
    finally:
        capture.release()
    raw = _interpolate_missing(np, tracks).copy()
    smoothed, robust, outliers = _robust_pose_filter(
        np,
        raw,
        median_radius=temporal_median_radius,
        smoothing_sigma=smoothing_sigma,
        outlier_threshold_pixels=outlier_threshold_pixels,
    )
    raw_snapshot = raw.copy()
    selected = sorted({index for indices in POSE_INDICES.values() for index in indices})
    correspondence = _trajectory_correspondence_metrics(
        np,
        raw,
        smoothed,
        width=width,
        height=height,
        smoothing_sigma=smoothing_sigma,
        robust=robust,
        outliers=outliers,
        action_horizon_frames=action_horizon_frames,
    )
    if not np.array_equal(raw, raw_snapshot):
        raise RuntimeError("pose correspondence metrics mutated raw observations")
    record: dict[str, Any] = {
        "decoded_frames": len(tracks),
        "missing_pose_frames": missing,
        "low_confidence_frames": low_confidence,
        "smoothing": {
            "method": "centered_temporal_median_then_zero_phase_gaussian",
            "temporal_median_radius_frames": temporal_median_radius,
            "outlier_threshold_pixels": outlier_threshold_pixels,
            "sigma_frames": smoothing_sigma,
        },
        "outlier_repair": {
            "method": "centered_temporal_median_replacement",
            "detected_landmark_observations": int(
                np.count_nonzero(outliers[:, selected])
            ),
            "detected_frames": int(
                np.count_nonzero(np.any(outliers[:, selected], axis=1))
            ),
            "detected_frame_indices": np.flatnonzero(
                np.any(outliers[:, selected], axis=1)
            ).tolist(),
            "all_detected_outliers_replaced": True,
        },
        "correspondence": correspondence,
    }
    if return_raw_tracks:
        record["_raw_tracks"] = raw
        record["_robust_tracks"] = robust
    return smoothed, record


def _capsule(cv2: Any, np: Any, shape: tuple[int, int], first: Any, second: Any, radius: int) -> Any:
    mask = np.zeros(shape, dtype=np.uint8)
    a = tuple(np.rint(first).astype(int))
    b = tuple(np.rint(second).astype(int))
    cv2.line(mask, a, b, 255, radius * 2, cv2.LINE_AA)
    cv2.circle(mask, a, radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, b, radius, 255, -1, cv2.LINE_AA)
    return mask


def _distance_to_segment_squared(np: Any, reference: Any, first: int, second: int) -> Any:
    height, width = reference["shape"]
    yy, xx = np.mgrid[:height, :width]
    points = np.stack((xx, yy), axis=2).astype(np.float64)
    start = reference["points"][first]
    vector = reference["points"][second] - start
    denominator = max(1e-9, float(np.dot(vector, vector)))
    position = np.clip(
        np.sum((points - start) * vector, axis=2) / denominator,
        0.0,
        1.0,
    )
    projection = start + position[..., None] * vector
    return np.sum((points - projection) ** 2, axis=2)


def _meaningful_component_count(
    cv2: Any, np: Any, mask: Any, *, minimum_area_pixels: int = 20
) -> int:
    binary = (mask >= 16).astype(np.uint8)
    count, _, statistics, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return 0
    return int(
        np.count_nonzero(statistics[1:, cv2.CC_STAT_AREA] >= minimum_area_pixels)
    )


def _piece_mask_overlap_metrics(np: Any, pieces: dict[str, Any]) -> dict[str, int]:
    names = tuple(pieces)
    total_overlap = 0
    cross_side_overlap = 0
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            overlap = int(
                np.count_nonzero((pieces[first] > 0) & (pieces[second] > 0))
            )
            total_overlap += overlap
            if first.split("_", 1)[0] != second.split("_", 1)[0]:
                cross_side_overlap += overlap
    return {
        "pairwise_overlap_pixels": total_overlap,
        "cross_side_overlap_pixels": cross_side_overlap,
    }


def _robot_rig_reference(np: Any, payload: dict[str, Any], fallback: Any) -> Any:
    configured = payload.get("robot_rig_reference_xy")
    if configured is None:
        return np.asarray(fallback, dtype=np.float64)
    reference = np.zeros((33, 2), dtype=np.float64)
    for side, indices in POSE_INDICES.items():
        if side not in configured:
            raise ValueError(f"robot rig reference is missing {side}")
        for label, index in zip(
            ("shoulder", "elbow", "wrist", "hand"), indices, strict=True
        ):
            value = np.asarray(configured[side][label], dtype=np.float64)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid {side} robot rig point {label}")
            reference[index] = value
    return reference


def _piece_masks(
    cv2: Any,
    np: Any,
    anchor_mask: Any,
    reference: Any,
    *,
    disjoint: bool = False,
    joint_overlap_radius_pixels: int = 0,
) -> dict[str, Any]:
    height, width = anchor_mask.shape
    scale = width / 1280.0
    specifications: list[tuple[str, int, int, int]] = []
    for side, (shoulder, elbow, wrist, hand) in POSE_INDICES.items():
        specifications.extend(
            (
                (f"{side}_upper", shoulder, elbow, max(18, round(48 * scale))),
                (f"{side}_lower", elbow, wrist, max(16, round(42 * scale))),
                (f"{side}_hand", wrist, hand, max(18, round(58 * scale))),
            )
        )
    candidates = {
        name: cv2.bitwise_and(
            anchor_mask,
            _capsule(
                cv2,
                np,
                (height, width),
                reference[first],
                reference[second],
                radius,
            ),
        )
        for name, first, second, radius in specifications
    }
    if not disjoint:
        return candidates

    distance_reference = {"shape": (height, width), "points": reference}
    scores = []
    for name, first, second, radius in specifications:
        score = _distance_to_segment_squared(
            np, distance_reference, first, second
        ) / float(radius * radius)
        score[candidates[name] == 0] = np.inf
        scores.append(score)
    stacked = np.stack(scores, axis=0)
    winner = np.argmin(stacked, axis=0)
    masks: dict[str, Any] = {}
    for part_index, (part_name, _, _, _) in enumerate(specifications):
        selected = np.isfinite(stacked[part_index]) & (winner == part_index)
        mask = np.zeros_like(anchor_mask, dtype=np.uint8)
        mask[selected] = anchor_mask[selected]
        masks[part_name] = mask
    if joint_overlap_radius_pixels > 0:
        for side, (_, elbow, wrist, _) in POSE_INDICES.items():
            for joint_index, adjacent in (
                (elbow, (f"{side}_upper", f"{side}_lower")),
                (wrist, (f"{side}_lower", f"{side}_hand")),
            ):
                joint = cv2.bitwise_and(
                    anchor_mask,
                    _capsule(
                        cv2,
                        np,
                        (height, width),
                        reference[joint_index],
                        reference[joint_index],
                        joint_overlap_radius_pixels,
                    ),
                )
                for name in adjacent:
                    masks[name] = cv2.bitwise_or(masks[name], joint)
    return masks


def _similarity(
    np: Any,
    source_a: Any,
    source_b: Any,
    target_a: Any,
    target_b: Any,
    *,
    exact_scale: bool = False,
) -> Any:
    source_vector = source_b - source_a
    target_vector = target_b - target_a
    source_length = max(1e-6, float(np.linalg.norm(source_vector)))
    target_length = max(1e-6, float(np.linalg.norm(target_vector)))
    raw_scale = target_length / source_length
    scale = float(raw_scale if exact_scale else np.clip(raw_scale, 0.72, 1.32))
    source_angle = math.atan2(float(source_vector[1]), float(source_vector[0]))
    target_angle = math.atan2(float(target_vector[1]), float(target_vector[0]))
    angle = target_angle - source_angle
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    matrix = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    translation = target_a - matrix @ source_a
    return np.asarray(
        ((matrix[0, 0], matrix[0, 1], translation[0]),
         (matrix[1, 0], matrix[1, 1], translation[1])),
        dtype=np.float64,
    )


def _anisotropic_segment_transform(
    np: Any,
    source_a: Any,
    source_b: Any,
    target_a: Any,
    target_b: Any,
    *,
    transverse_scale: float,
) -> Any:
    """Map both joints exactly while keeping limb thickness independently fixed."""

    if transverse_scale <= 0:
        raise ValueError("transverse scale must be positive")
    source_vector = source_b - source_a
    target_vector = target_b - target_a
    source_length = max(1e-6, float(np.linalg.norm(source_vector)))
    target_length = max(1e-6, float(np.linalg.norm(target_vector)))
    source_unit = source_vector / source_length
    source_perpendicular = np.asarray(
        (-source_unit[1], source_unit[0]), dtype=np.float64
    )
    target_unit = target_vector / target_length
    target_perpendicular = np.asarray(
        (-target_unit[1], target_unit[0]), dtype=np.float64
    )
    source_basis = np.column_stack((source_unit, source_perpendicular))
    target_basis = np.column_stack(
        (
            target_unit * (target_length / source_length),
            target_perpendicular * transverse_scale,
        )
    )
    matrix = target_basis @ source_basis.T
    translation = target_a - matrix @ source_a
    return np.column_stack((matrix, translation))


def _fixed_scale_hand_transform(
    np: Any,
    source_wrist: Any,
    source_hand: Any,
    target_wrist: Any,
    target_angle_radians: float,
    *,
    scale: float = 1.0,
) -> Any:
    """Map the hand root and angle while preserving calibrated morphology."""

    if scale <= 0:
        raise ValueError("fixed hand scale must be positive")
    source_length = max(
        1e-6, float(np.linalg.norm(source_hand - source_wrist))
    )
    target_hand = target_wrist + source_length * scale * np.asarray(
        (math.cos(target_angle_radians), math.sin(target_angle_radians)),
        dtype=np.float64,
    )
    return _similarity(
        np,
        source_wrist,
        source_hand,
        target_wrist,
        target_hand,
        exact_scale=True,
    )


def _fixed_scale_anchor_transform(
    np: Any,
    source_anchor: Any,
    target_anchor: Any,
    *,
    scale: float,
    angle_degrees: float = 0.0,
) -> Any:
    """Map one object anchor exactly with fixed scale and rotation."""

    if scale <= 0:
        raise ValueError("fixed anchor scale must be positive")
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    matrix = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    translation = np.asarray(target_anchor, dtype=np.float64) - matrix @ np.asarray(
        source_anchor, dtype=np.float64
    )
    return np.column_stack((matrix, translation))


def _series_statistics(np: Any, values: list[float]) -> dict[str, float]:
    series = np.asarray(values, dtype=np.float64)
    if not len(series):
        raise ValueError("cannot summarize an empty series")
    return {
        "minimum": float(np.min(series)),
        "p01": float(np.quantile(series, 0.01)),
        "median": float(np.median(series)),
        "p99": float(np.quantile(series, 0.99)),
        "maximum": float(np.max(series)),
        "p99_to_p01_ratio": float(
            np.quantile(series, 0.99) / max(float(np.quantile(series, 0.01)), 1e-9)
        ),
        "maximum_frame_step": float(
            np.max(np.abs(np.diff(series))) if len(series) > 1 else 0.0
        ),
    }


def _align_rgba_robot(
    cv2: Any,
    np: Any,
    rgba: Any,
    reference_bgr: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Align an extracted RGBA robot back to its full-scene anchor."""

    robot_bgr = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    detector = cv2.ORB_create(
        nfeatures=5000,
        edgeThreshold=15,
        fastThreshold=8,
    )
    robot_gray = cv2.cvtColor(robot_bgr, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    robot_keypoints, robot_descriptors = detector.detectAndCompute(robot_gray, alpha)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(
        reference_gray, None
    )
    if robot_descriptors is None or reference_descriptors is None:
        raise RuntimeError("cannot compute robot-anchor alignment descriptors")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        robot_descriptors, reference_descriptors, k=2
    )
    matches = [first for first, second in pairs if first.distance < 0.7 * second.distance]
    if len(matches) < 30:
        raise RuntimeError(f"only {len(matches)} robot-anchor feature matches")
    robot_points = np.float32(
        [robot_keypoints[match.queryIdx].pt for match in matches]
    )
    reference_points = np.float32(
        [reference_keypoints[match.trainIdx].pt for match in matches]
    )
    transform, inliers = cv2.estimateAffinePartial2D(
        robot_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=10000,
        confidence=0.999,
    )
    if transform is None or inliers is None:
        raise RuntimeError("cannot estimate robot-anchor affine alignment")
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = float(inlier_count / len(matches))
    if inlier_count < 30 or inlier_ratio < 0.80:
        raise RuntimeError(
            f"weak robot-anchor alignment: {inlier_count}/{len(matches)} inliers"
        )
    height, width = reference_bgr.shape[:2]
    aligned_bgr = cv2.warpAffine(
        robot_bgr,
        transform,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
    )
    aligned_alpha = cv2.warpAffine(
        alpha,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    scale = float(math.hypot(float(transform[0, 0]), float(transform[0, 1])))
    angle = float(
        math.degrees(math.atan2(float(transform[1, 0]), float(transform[0, 0])))
    )
    return aligned_bgr, aligned_alpha, {
        "method": "ORB_ratio_test_RANSAC_partial_affine",
        "feature_matches": len(matches),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "transform": transform.tolist(),
        "scale": scale,
        "rotation_degrees": angle,
        "coordinate_frame": "camera:cutout_pixels -> camera:anchor_reference_pixels",
    }


def _warp_piece(cv2: Any, image: Any, mask: Any, transform: Any) -> tuple[Any, Any]:
    height, width = mask.shape
    warped_image = cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT101,
    )
    warped_mask = cv2.warpAffine(
        mask,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return warped_image, warped_mask


def _warp_layer_to_canvas(
    cv2: Any,
    image: Any,
    mask: Any,
    transform: Any,
    *,
    width: int,
    height: int,
) -> tuple[Any, Any]:
    """Warp a smaller RGBA-derived layer directly into the video canvas."""

    warped_image = cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_image, warped_mask


def _overlay(cv2: Any, np: Any, base: Any, image: Any, mask: Any) -> Any:
    alpha = cv2.GaussianBlur(mask, (5, 5), 0.65).astype(np.float32) / 255.0
    return np.rint(
        image.astype(np.float32) * alpha[..., None]
        + base.astype(np.float32) * (1.0 - alpha[..., None])
    ).astype(np.uint8)


def _pose_clear_mask(cv2: Any, np: Any, shape: tuple[int, int], points: Any) -> Any:
    height, width = shape
    scale = width / 1280.0
    result = np.zeros(shape, dtype=np.uint8)
    for shoulder, elbow, wrist, _ in POSE_INDICES.values():
        result = cv2.bitwise_or(
            result,
            _capsule(
                cv2, np, shape, points[shoulder], points[elbow],
                max(20, round(60 * scale)),
            ),
        )
        result = cv2.bitwise_or(
            result,
            _capsule(
                cv2, np, shape, points[elbow], points[wrist],
                max(20, round(58 * scale)),
            ),
        )
    torso_points = np.rint(points[[11, 12, 24, 23]]).astype(np.int32)
    cv2.fillConvexPoly(result, torso_points, 255, cv2.LINE_AA)
    shoulder_center = np.mean(points[[11, 12]], axis=0)
    shoulder_width = float(np.linalg.norm(points[11] - points[12]))
    head_center = shoulder_center + np.asarray((0.0, -0.72 * shoulder_width))
    cv2.circle(
        result,
        tuple(np.rint(head_center).astype(int)),
        max(24, round(0.52 * shoulder_width)),
        255,
        -1,
        cv2.LINE_AA,
    )
    return cv2.dilate(
        result,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )


def _clear_person(
    cv2: Any,
    np: Any,
    source: Any,
    clean_base: Any,
    safety: Any,
    dynamic_clear: Any,
) -> tuple[Any, Any]:
    core = cv2.bitwise_or(safety, dynamic_clear)
    outer = cv2.dilate(
        core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    alpha = cv2.GaussianBlur(outer, (9, 9), 1.1).astype(np.float32) / 255.0
    alpha[outer == 0] = 0.0
    alpha[core > 0] = 1.0
    result = np.rint(
        clean_base.astype(np.float32) * alpha[..., None]
        + source.astype(np.float32) * (1.0 - alpha[..., None])
    ).astype(np.uint8)
    result[core > 0] = clean_base[core > 0]
    return result, outer


def _flower_restore_mask(
    cv2: Any,
    np: Any,
    *,
    source_frame: Any,
    instance_mask: Any,
    safety_mask: Any,
    changed_support: Any,
    skin_dilation_pixels: int,
) -> tuple[Any, dict[str, int]]:
    """Keep tracked task flowers while excluding visible source skin.

    The flower instance track supplies pale petals and thin stems that a color
    seed cannot recover alone.  Strict green/pink/yellow pixels fill tracking
    misses.  The layer is clipped to pixels altered by person clearing or robot
    compositing, so unchanged scene flowers are not needlessly rewritten.
    """

    if skin_dilation_pixels < 0:
        raise ValueError("flower skin-negative dilation must be non-negative")
    height, width = source_frame.shape[:2]
    scale = min(width / instance_mask.shape[1], height / instance_mask.shape[0])
    aligned_width = round(instance_mask.shape[1] * scale)
    aligned_height = round(instance_mask.shape[0] * scale)
    resized_instance = cv2.resize(
        instance_mask.astype(np.uint8),
        (aligned_width, aligned_height),
        interpolation=cv2.INTER_NEAREST,
    )
    aligned_instance = np.zeros((height, width), dtype=bool)
    left = (width - aligned_width) // 2
    top = (height - aligned_height) // 2
    aligned_instance[
        top : top + aligned_height,
        left : left + aligned_width,
    ] = resized_instance.astype(bool)
    safety = safety_mask.astype(bool)
    strict = _strict_flower_seed(cv2, np, source_frame, safety)
    source_skin = np.logical_and(_skin_like(cv2, np, source_frame), safety)
    if skin_dilation_pixels:
        size = skin_dilation_pixels * 2 + 1
        source_skin = cv2.dilate(
            source_skin.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        ) > 0
    candidates = np.logical_or(aligned_instance, strict)
    flowers = np.logical_and(candidates, np.logical_not(source_skin))
    # Materialize the mask before collecting overlap metrics.  Some NumPy/OpenCV
    # array combinations can otherwise leave a temporary-backed boolean result.
    restore = np.array(
        np.logical_and(flowers, changed_support.astype(bool)),
        dtype=bool,
        copy=True,
    )
    restore_skin_overlap_pixels = int(
        np.count_nonzero(np.logical_and(restore, source_skin))
    )
    return restore, {
        "instance_pixels": int(np.count_nonzero(aligned_instance)),
        "strict_seed_pixels": int(np.count_nonzero(strict)),
        "skin_negative_pixels": int(np.count_nonzero(source_skin)),
        "candidate_pixels": int(np.count_nonzero(candidates)),
        "restore_pixels": int(np.count_nonzero(restore)),
        "restore_skin_overlap_pixels": restore_skin_overlap_pixels,
    }


def _measure_encoded_flower_preservation(
    cv2: Any,
    np: Any,
    *,
    source: Path,
    output: Path,
    instance_masks: Any,
    safety_mask: Any,
    skin_dilation_pixels: int,
) -> dict[str, float | int]:
    """Measure decoded source similarity on the independently tracked flowers."""

    source_capture = cv2.VideoCapture(str(source))
    output_capture = cv2.VideoCapture(str(output))
    maes: list[float] = []
    mask_pixels: list[int] = []
    frame = 0
    try:
        while True:
            source_ok, source_frame = source_capture.read()
            output_ok, output_frame = output_capture.read()
            if not source_ok and not output_ok:
                break
            if not source_ok or not output_ok:
                raise RuntimeError("source/output decode lengths differ in flower audit")
            if frame >= len(instance_masks):
                raise RuntimeError("flower instance track is shorter than the video")
            mask, _ = _flower_restore_mask(
                cv2,
                np,
                source_frame=source_frame,
                instance_mask=instance_masks[frame],
                safety_mask=safety_mask,
                changed_support=safety_mask,
                skin_dilation_pixels=skin_dilation_pixels,
            )
            count = int(np.count_nonzero(mask))
            mask_pixels.append(count)
            if count:
                difference = np.abs(
                    output_frame[mask].astype(np.float32)
                    - source_frame[mask].astype(np.float32)
                )
                maes.append(float(np.mean(difference)))
            else:
                maes.append(0.0)
            frame += 1
    finally:
        source_capture.release()
        output_capture.release()
    if frame != len(instance_masks):
        raise RuntimeError(
            f"flower audit decoded {frame}/{len(instance_masks)} expected frames"
        )
    return {
        "frames": frame,
        "minimum_mask_pixels": min(mask_pixels),
        "mean_mask_pixels": float(np.mean(mask_pixels)),
        "mean_rgb_mae": float(np.mean(maes)),
        "p95_rgb_mae": float(np.quantile(maes, 0.95)),
        "maximum_rgb_mae": max(maes),
    }


def _review(ffmpeg: Path, video: Path, output_dir: Path) -> None:
    subprocess.run(
        [
            str(ffmpeg), "-y", "-v", "error", "-i", str(video), "-vf",
            "fps=1/1.7,scale=400:-2,tile=4x4:padding=4:margin=4:color=black",
            "-frames:v", "1", "-q:v", "2", str(output_dir / "storyboard-16.jpg"),
        ],
        check=True,
    )
    early_frames = list(range(0, 56, 2))
    early_expression = "+".join(f"eq(n\\,{frame})" for frame in early_frames)
    subprocess.run(
        [
            str(ffmpeg), "-y", "-v", "error", "-i", str(video), "-vf",
            f"select='{early_expression}',scale=320:-2,"
            "tile=4x7:padding=3:margin=3:color=black",
            "-vsync", "0", "-frames:v", "1", "-q:v", "2",
            str(output_dir / "early-consecutive-review.jpg"),
        ],
        check=True,
    )
    frames = [round(index * 659 / 27) for index in range(28)]
    expression = "+".join(f"eq(n\\,{frame})" for frame in frames)
    subprocess.run(
        [
            str(ffmpeg), "-y", "-v", "error", "-i", str(video), "-vf",
            f"select='{expression}',scale=320:-2,tile=4x7:padding=3:margin=3:color=black",
            "-vsync", "0", "-frames:v", "1", "-q:v", "2",
            str(output_dir / "dense-review.jpg"),
        ],
        check=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--human-review",
        choices=("pending", "passed", "failed"),
        default="pending",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    experiment = args.experiment_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    payload = json.loads(config_path.read_text())
    paths = {
        key: Path(payload[key]).expanduser().resolve()
        for key in (
            "source",
            "robot_anchor",
            "source_anchor",
            "anchor_mask",
            "safety_mask",
            "pose_model",
        )
    }
    if payload.get("clean_base_image"):
        paths["clean_base_image"] = Path(
            payload["clean_base_image"]
        ).expanduser().resolve()
    if payload.get("robot_anchor_alignment_reference"):
        paths["robot_anchor_alignment_reference"] = Path(
            payload["robot_anchor_alignment_reference"]
        ).expanduser().resolve()
    if payload.get("reuse_pose_trajectory"):
        paths["reuse_pose_trajectory"] = Path(
            payload["reuse_pose_trajectory"]
        ).expanduser().resolve()
        paths["reuse_pose_tracking_manifest"] = Path(
            payload["reuse_pose_tracking_manifest"]
        ).expanduser().resolve()
    if payload.get("flower_instance_masks"):
        paths["flower_instance_masks"] = Path(
            payload["flower_instance_masks"]
        ).expanduser().resolve()
    if payload.get("bouquet_texture"):
        paths["bouquet_texture"] = Path(
            payload["bouquet_texture"]
        ).expanduser().resolve()
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    anchor_frame = int(payload["anchor_frame"])
    experiment.mkdir(parents=True, exist_ok=True)
    assets = experiment / "assets"
    final = experiment / "final"
    assets.mkdir(exist_ok=True)
    final.mkdir(exist_ok=True)
    trace_path = experiment / "trace.json"
    import cv2
    import numpy as np

    mp = None
    if "reuse_pose_trajectory" not in paths:
        # MediaPipe's optional documentation hook imports TensorFlow whenever it
        # is merely installed. The pose Tasks API does not need that dependency.
        sys.modules.setdefault("tensorflow", None)
        import mediapipe as mp

    np.random.seed(args.seed)
    source_info = _source_info(cv2, paths["source"])
    width = int(source_info["width"])
    height = int(source_info["height"])
    frame_count = int(source_info["frames"])
    fps = float(source_info["fps"])
    smoothing_sigma = float(payload.get("pose_smoothing_sigma", 2.0))
    temporal_median_radius = int(payload.get("temporal_median_radius", 0))
    outlier_threshold_pixels = float(payload.get("outlier_threshold_pixels", 20.0))
    action_horizon_frames = int(payload.get("action_horizon_frames", 8))
    exact_endpoint_mapping = bool(payload.get("exact_endpoint_mapping", False))
    fixed_hand_scale = float(payload.get("fixed_hand_scale", 1.0))
    fixed_hand_scales = {
        side: float(
            payload.get("fixed_hand_scale_by_side", {}).get(side, fixed_hand_scale)
        )
        for side in POSE_INDICES
    }
    lock_hand_morphology = bool(payload.get("lock_hand_morphology", False))
    disjoint_piece_masks = bool(payload.get("disjoint_piece_masks", False))
    joint_overlap_radius_pixels = int(
        payload.get("joint_overlap_radius_pixels", 0)
    )
    anisotropic_limb_mapping = bool(
        payload.get("anisotropic_limb_mapping", False)
    )
    fixed_arm_thickness_scale = float(
        payload.get("fixed_arm_thickness_scale", 1.0)
    )
    require_flower_protection = bool(
        payload.get("require_flower_protection", False)
    )
    flower_skin_dilation_pixels = int(
        payload.get("flower_skin_dilation_pixels", 2)
    )
    if require_flower_protection and not {
        "flower_instance_masks",
        "bouquet_texture",
    }.intersection(paths):
        raise ValueError(
            "require_flower_protection needs flower_instance_masks or bouquet_texture"
        )
    bouquet_target_height_pixels = float(
        payload.get("bouquet_target_height_pixels", 220.0)
    )
    bouquet_grip_uv = np.asarray(
        payload.get("bouquet_grip_uv", (0.5, 0.65)), dtype=np.float64
    )
    bouquet_target_offset_xy = np.asarray(
        payload.get("bouquet_target_offset_xy", (0.0, 0.0)), dtype=np.float64
    )
    bouquet_target_landmarks = tuple(
        int(index) for index in payload.get("bouquet_target_landmarks", (15, 16))
    )
    bouquet_target_hand_side = payload.get("bouquet_target_hand_side")
    bouquet_target_hand_distance_pixels = float(
        payload.get("bouquet_target_hand_distance_pixels", 0.0)
    )
    bouquet_target_smoothing_sigma = float(
        payload.get("bouquet_target_smoothing_sigma", 0.0)
    )
    bouquet_target_motion_maximum_step_pixels = float(
        payload.get("bouquet_target_motion_maximum_step_pixels", 20.0)
    )
    bouquet_angle_degrees = float(payload.get("bouquet_angle_degrees", 0.0))
    if "bouquet_texture" in paths:
        if bouquet_target_height_pixels <= 0:
            raise ValueError("bouquet target height must be positive")
        if bouquet_grip_uv.shape != (2,) or np.any(
            np.logical_or(bouquet_grip_uv < 0.0, bouquet_grip_uv > 1.0)
        ):
            raise ValueError("bouquet_grip_uv must contain two values in [0, 1]")
        if bouquet_target_offset_xy.shape != (2,):
            raise ValueError("bouquet_target_offset_xy must contain two values")
        if not bouquet_target_landmarks:
            raise ValueError("bouquet_target_landmarks must not be empty")
        if not set(bouquet_target_landmarks).issubset(range(33)):
            raise ValueError("bouquet target landmarks must be MediaPipe pose indices")
        if bouquet_target_hand_side is not None:
            if bouquet_target_hand_side not in POSE_INDICES:
                raise ValueError("bouquet_target_hand_side must be left or right")
            if bouquet_target_hand_distance_pixels < 0:
                raise ValueError("bouquet hand distance must be non-negative")
        if bouquet_target_smoothing_sigma < 0:
            raise ValueError("bouquet target smoothing sigma must be non-negative")
        if bouquet_target_motion_maximum_step_pixels <= 0:
            raise ValueError("bouquet target motion step must be positive")
    hand_orientation_median_radius = int(
        payload.get("hand_orientation_median_radius", temporal_median_radius)
    )
    hand_orientation_smoothing_sigma = float(
        payload.get("hand_orientation_smoothing_sigma", smoothing_sigma)
    )
    minimum_forearm_direction_length_pixels = float(
        payload.get("minimum_forearm_direction_length_pixels", 20.0)
    )
    minimum_hand_direction_length_pixels = float(
        payload.get("minimum_hand_direction_length_pixels", 8.0)
    )
    maximum_hand_angle_step_degrees = float(
        payload.get("maximum_hand_angle_step_degrees", 8.0)
    )
    trace: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "running",
        "honest_status": "PARTIAL",
        "method": "mediapipe_pose_rigid_robot_torso_similarity_transformed_arm_pieces",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            **_package_versions(),
            "mediapipe": importlib.metadata.version("mediapipe"),
            "opencv-contrib-python": importlib.metadata.version("opencv-contrib-python"),
        },
        "seed": args.seed,
        "gpu": {
            "used": False,
            "cuda_visible_devices": None,
            "reason": "MediaPipe Tasks and OpenCV rigid-part compositor run on CPU",
        },
        "git": _git_state(PROJECT_ROOT),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "resolved_config": payload,
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in paths.items()
        },
        "source_video": source_info,
        "coordinate_frames": {
            "source_pose": "camera:source_pixels",
            "robot_pieces": "camera:anchor_pixels",
            "piece_transforms": "camera:anchor_pixels -> camera:current_source_pixels",
        },
    }
    _write_json(trace_path, trace)
    try:
        if "reuse_pose_trajectory" in paths:
            tracks, tracking, raw_tracks, robust_tracks = (
                _load_reusable_pose_trajectory(
                    np,
                    paths["reuse_pose_trajectory"],
                    paths["reuse_pose_tracking_manifest"],
                    expected_frames=frame_count,
                )
            )
        else:
            tracks, tracking = _track_pose(
                cv2=cv2,
                np=np,
                mp=mp,
                source=paths["source"],
                model=paths["pose_model"],
                fps=fps,
                width=width,
                height=height,
                smoothing_sigma=smoothing_sigma,
                temporal_median_radius=temporal_median_radius,
                outlier_threshold_pixels=outlier_threshold_pixels,
                action_horizon_frames=action_horizon_frames,
                return_raw_tracks=True,
            )
            raw_tracks = tracking.pop("_raw_tracks")
            robust_tracks = tracking.pop("_robust_tracks")
        if int(tracking["decoded_frames"]) != frame_count:
            raise RuntimeError("pose tracker did not decode the full source")
        reference = tracks[anchor_frame]
        piece_reference = _robot_rig_reference(np, payload, reference)
        hand_angles, hand_orientation = _stable_hand_angles(
            np,
            tracks,
            minimum_forearm_length_pixels=minimum_forearm_direction_length_pixels,
            minimum_hand_length_pixels=minimum_hand_direction_length_pixels,
            median_radius=hand_orientation_median_radius,
            smoothing_sigma=hand_orientation_smoothing_sigma,
            maximum_step_degrees=maximum_hand_angle_step_degrees,
        )
        bouquet_raw_targets = None
        bouquet_targets = None
        if "bouquet_texture" in paths:
            if bouquet_target_hand_side is not None:
                wrist_index = POSE_INDICES[bouquet_target_hand_side][2]
                angles = hand_angles[bouquet_target_hand_side]
                directions = np.column_stack((np.cos(angles), np.sin(angles)))
                bouquet_raw_targets = (
                    tracks[:, wrist_index]
                    + bouquet_target_hand_distance_pixels * directions
                    + bouquet_target_offset_xy
                )
            else:
                bouquet_raw_targets = (
                    np.mean(tracks[:, list(bouquet_target_landmarks)], axis=1)
                    + bouquet_target_offset_xy
                )
            bouquet_targets = bouquet_raw_targets.copy()
            if bouquet_target_smoothing_sigma > 0:
                for coordinate in range(2):
                    bouquet_targets[:, coordinate] = _smooth_scalar_series(
                        np,
                        bouquet_targets[:, coordinate],
                        bouquet_target_smoothing_sigma,
                    )
            bouquet_targets = _zero_phase_bounded_vector_steps(
                np,
                bouquet_targets,
                bouquet_target_motion_maximum_step_pixels,
            )
        robot_image = cv2.imread(str(paths["robot_anchor"]), cv2.IMREAD_UNCHANGED)
        anchor_mask = cv2.imread(str(paths["anchor_mask"]), cv2.IMREAD_GRAYSCALE)
        safety = cv2.imread(str(paths["safety_mask"]), cv2.IMREAD_GRAYSCALE)
        if robot_image is None or anchor_mask is None or safety is None:
            raise RuntimeError("cannot decode robot or mask assets")
        robot_alignment: dict[str, Any] | None = None
        if robot_image.ndim == 3 and robot_image.shape[2] == 4:
            if payload.get("robot_rig_reference_xy") is not None:
                robot = robot_image[:, :, :3]
                robot_alpha = robot_image[:, :, 3]
            else:
                if "robot_anchor_alignment_reference" not in paths:
                    raise ValueError("RGBA robot anchor requires alignment reference")
                alignment_reference = cv2.imread(
                    str(paths["robot_anchor_alignment_reference"]), cv2.IMREAD_COLOR
                )
                if alignment_reference is None:
                    raise RuntimeError("cannot decode robot alignment reference")
                robot, robot_alpha, robot_alignment = _align_rgba_robot(
                    cv2, np, robot_image, alignment_reference
                )
            if payload.get("use_robot_alpha_mask", True):
                anchor_mask = robot_alpha
        else:
            robot = robot_image[:, :, :3]
        robot = cv2.resize(robot, (width, height), interpolation=cv2.INTER_LANCZOS4)
        anchor_mask = cv2.resize(
            anchor_mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
        cv2.imwrite(
            str(assets / "robot-texture-canvas.png"),
            np.dstack((robot, anchor_mask)),
        )
        safety = cv2.resize(safety, (width, height), interpolation=cv2.INTER_NEAREST)
        safety = cv2.dilate(
            (safety >= 127).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        )
        bouquet_image = None
        bouquet_alpha = None
        bouquet_source_bbox = None
        bouquet_source_anchor = None
        bouquet_scale = None
        if "bouquet_texture" in paths:
            bouquet_rgba = cv2.imread(
                str(paths["bouquet_texture"]), cv2.IMREAD_UNCHANGED
            )
            if (
                bouquet_rgba is None
                or bouquet_rgba.ndim != 3
                or bouquet_rgba.shape[2] != 4
            ):
                raise RuntimeError("bouquet texture must decode as RGBA")
            bouquet_image = bouquet_rgba[:, :, :3]
            bouquet_alpha = bouquet_rgba[:, :, 3]
            visible_y, visible_x = np.nonzero(bouquet_alpha >= 8)
            if not len(visible_x):
                raise RuntimeError("bouquet texture has no visible alpha pixels")
            left = int(np.min(visible_x))
            top = int(np.min(visible_y))
            right = int(np.max(visible_x)) + 1
            bottom = int(np.max(visible_y)) + 1
            bouquet_source_bbox = {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": right - left,
                "height": bottom - top,
            }
            bouquet_source_anchor = np.asarray(
                (
                    left + bouquet_grip_uv[0] * (right - left),
                    top + bouquet_grip_uv[1] * (bottom - top),
                ),
                dtype=np.float64,
            )
            bouquet_scale = bouquet_target_height_pixels / max(1, bottom - top)
            cv2.imwrite(
                str(assets / "bouquet-texture.png"), bouquet_rgba
            )
        flower_instances = None
        if "flower_instance_masks" in paths:
            flower_instances = _load_packed(
                np, paths["flower_instance_masks"], "packed"
            )
            if len(flower_instances) != frame_count:
                raise ValueError(
                    "flower instance frame count does not match source video"
                )
        pieces = _piece_masks(
            cv2,
            np,
            anchor_mask,
            piece_reference,
            disjoint=disjoint_piece_masks,
            joint_overlap_radius_pixels=joint_overlap_radius_pixels,
        )
        piece_mask_overlap = _piece_mask_overlap_metrics(np, pieces)
        source_piece_components = {
            name: _meaningful_component_count(cv2, np, mask)
            for name, mask in pieces.items()
        }
        arm_union = np.zeros((height, width), dtype=np.uint8)
        for name, mask in pieces.items():
            arm_union = cv2.bitwise_or(arm_union, mask)
            cv2.imwrite(str(assets / f"piece-mask-{name}.png"), mask)
        if "clean_base_image" in paths:
            clean_base = cv2.imread(
                str(paths["clean_base_image"]), cv2.IMREAD_COLOR
            )
            if clean_base is None:
                raise RuntimeError("cannot decode clean-base image")
            clean_base = cv2.resize(
                clean_base, (width, height), interpolation=cv2.INTER_LANCZOS4
            )
            clean_base_method = "imagegen_arm_removed_robot_torso_base"
        else:
            inpaint_mask = cv2.dilate(
                arm_union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            )
            clean_base = cv2.inpaint(
                robot, inpaint_mask, 7.0, cv2.INPAINT_TELEA
            )
            clean_base_method = "opencv_telea_arm_inpaint"
        cv2.imwrite(str(assets / "arm-piece-union.png"), arm_union)
        cv2.imwrite(str(assets / "rigid-clean-base.jpg"), clean_base)
        overlay = robot.copy()
        overlay[arm_union > 0] = np.asarray((30, 45, 250), dtype=np.uint8)
        for side, indices in POSE_INDICES.items():
            shoulder, elbow, wrist, hand = indices
            for first, second in ((shoulder, elbow), (elbow, wrist), (wrist, hand)):
                cv2.line(
                    overlay,
                    tuple(np.rint(piece_reference[first]).astype(int)),
                    tuple(np.rint(piece_reference[second]).astype(int)),
                    (40, 235, 70),
                    4,
                    cv2.LINE_AA,
                )
        cv2.imwrite(str(assets / "anchor-rig-overlay.jpg"), overlay)
        trajectory_path = assets / "source-to-robot-pose-trajectory.json"
        _write_json(
            trajectory_path,
            {
                "schema_version": "1.0.0",
                "coordinate_frame": "camera:source_pixels",
                "fps": fps,
                "frame_count": frame_count,
                "landmark_indices": sorted(
                    {index for indices in POSE_INDICES.values() for index in indices}
                ),
                "raw_interpolated_xy": raw_tracks[
                    :,
                    sorted({index for indices in POSE_INDICES.values() for index in indices}),
                ].tolist(),
                "robust_median_xy": robust_tracks[
                    :,
                    sorted({index for indices in POSE_INDICES.values() for index in indices}),
                ].tolist(),
                "robot_target_xy": tracks[
                    :,
                    sorted({index for indices in POSE_INDICES.values() for index in indices}),
                ].tolist(),
                "temporal_index_map": list(range(frame_count)),
                "smoothing": tracking["smoothing"],
                "correspondence": tracking["correspondence"],
            },
        )

        output = final / "robot-motion-replacement-rigged.mp4"
        writer = _writer(ffmpeg, output, width, height, fps)
        capture = cv2.VideoCapture(str(paths["source"]))
        decoded = 0
        background_scores: list[float] = []
        sharpness_scores: list[float] = []
        transition_energy: list[float] = []
        previous_gray = None
        endpoint_mapping_errors: list[float] = []
        hand_root_mapping_errors: list[float] = []
        hand_landmark_mapping_errors: list[float] = []
        piece_scales: dict[str, list[float]] = {
            name: [] for name in pieces
        }
        piece_mask_areas: dict[str, list[float]] = {
            name: [] for name in pieces
        }
        limb_component_counts: dict[str, list[int]] = {
            side: [] for side in POSE_INDICES
        }
        flower_restore_pixels: list[int] = []
        flower_restore_skin_overlap_pixels: list[int] = []
        flower_pre_restore_mae: list[float] = []
        flower_post_restore_difference_pixels: list[int] = []
        bouquet_rendered_areas: list[float] = []
        bouquet_grip_mapping_errors: list[float] = []
        bouquet_target_steps: list[float] = []
        bouquet_hand_overlap_pixels: list[int] = []
        previous_bouquet_target = None
        try:
            while True:
                ok, source_frame = capture.read()
                if not ok:
                    break
                current = tracks[decoded]
                dynamic_clear = _pose_clear_mask(
                    cv2, np, (height, width), current
                )
                candidate, outer = _clear_person(
                    cv2, np, source_frame, clean_base, safety, dynamic_clear
                )
                frame_hand_layers: dict[str, tuple[Any, Any]] = {}
                for side, (shoulder, elbow, wrist, hand) in POSE_INDICES.items():
                    side_union = np.zeros((height, width), dtype=np.uint8)
                    specifications = (
                        (f"{side}_upper", shoulder, elbow),
                        (f"{side}_lower", elbow, wrist),
                        (f"{side}_hand", wrist, hand),
                    )
                    for name, first, second in specifications:
                        is_hand = name.endswith("_hand")
                        if is_hand and lock_hand_morphology:
                            transform = _fixed_scale_hand_transform(
                                np,
                                piece_reference[first],
                                piece_reference[second],
                                current[first],
                                float(hand_angles[side][decoded]),
                                scale=fixed_hand_scales[side],
                            )
                        elif anisotropic_limb_mapping:
                            transform = _anisotropic_segment_transform(
                                np,
                                piece_reference[first],
                                piece_reference[second],
                                current[first],
                                current[second],
                                transverse_scale=fixed_arm_thickness_scale,
                            )
                        else:
                            transform = _similarity(
                                np,
                                piece_reference[first],
                                piece_reference[second],
                                current[first],
                                current[second],
                                exact_scale=exact_endpoint_mapping,
                            )
                        mapped_first = (
                            transform[:, :2] @ piece_reference[first]
                            + transform[:, 2]
                        )
                        mapped_second = (
                            transform[:, :2] @ piece_reference[second]
                            + transform[:, 2]
                        )
                        first_error = float(
                            np.linalg.norm(mapped_first - current[first])
                        )
                        second_error = float(
                            np.linalg.norm(mapped_second - current[second])
                        )
                        if is_hand:
                            hand_root_mapping_errors.append(first_error)
                            hand_landmark_mapping_errors.append(second_error)
                        else:
                            endpoint_mapping_errors.extend(
                                (first_error, second_error)
                            )
                        piece_scales[name].append(
                            float(
                                math.hypot(
                                    float(transform[0, 0]),
                                    float(transform[0, 1]),
                                )
                            )
                        )
                        warped_image, warped_mask = _warp_piece(
                            cv2, robot, pieces[name], transform
                        )
                        if is_hand:
                            frame_hand_layers[side] = (warped_image, warped_mask)
                        side_union = cv2.bitwise_or(side_union, warped_mask)
                        piece_mask_areas[name].append(
                            float(np.sum(warped_mask, dtype=np.float64) / 255.0)
                        )
                        candidate = _overlay(
                            cv2, np, candidate, warped_image, warped_mask
                        )
                        overlay_support = cv2.dilate(
                            warped_mask,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                        )
                        outer = cv2.bitwise_or(outer, overlay_support)
                    limb_component_counts[side].append(
                        _meaningful_component_count(cv2, np, side_union)
                    )
                if (
                    bouquet_image is not None
                    and bouquet_alpha is not None
                    and bouquet_source_anchor is not None
                    and bouquet_scale is not None
                    and bouquet_targets is not None
                ):
                    bouquet_target = bouquet_targets[decoded]
                    bouquet_transform = _fixed_scale_anchor_transform(
                        np,
                        bouquet_source_anchor,
                        bouquet_target,
                        scale=bouquet_scale,
                        angle_degrees=bouquet_angle_degrees,
                    )
                    mapped_grip = (
                        bouquet_transform[:, :2] @ bouquet_source_anchor
                        + bouquet_transform[:, 2]
                    )
                    bouquet_grip_mapping_errors.append(
                        float(np.linalg.norm(mapped_grip - bouquet_target))
                    )
                    if previous_bouquet_target is not None:
                        bouquet_target_steps.append(
                            float(
                                np.linalg.norm(
                                    bouquet_target - previous_bouquet_target
                                )
                            )
                        )
                    previous_bouquet_target = bouquet_target
                    warped_bouquet, warped_bouquet_alpha = _warp_layer_to_canvas(
                        cv2,
                        bouquet_image,
                        bouquet_alpha,
                        bouquet_transform,
                        width=width,
                        height=height,
                    )
                    bouquet_rendered_areas.append(
                        float(
                            np.sum(warped_bouquet_alpha, dtype=np.float64) / 255.0
                        )
                    )
                    candidate = _overlay(
                        cv2,
                        np,
                        candidate,
                        warped_bouquet,
                        warped_bouquet_alpha,
                    )
                    bouquet_support = cv2.dilate(
                        warped_bouquet_alpha,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    )
                    outer = cv2.bitwise_or(outer, bouquet_support)
                    hand_union = np.zeros((height, width), dtype=np.uint8)
                    for _, hand_mask in frame_hand_layers.values():
                        hand_union = cv2.bitwise_or(hand_union, hand_mask)
                    bouquet_hand_overlap_pixels.append(
                        int(
                            np.count_nonzero(
                                np.logical_and(
                                    warped_bouquet_alpha >= 8,
                                    hand_union >= 8,
                                )
                            )
                        )
                    )
                    for hand_image, hand_mask in frame_hand_layers.values():
                        candidate = _overlay(
                            cv2, np, candidate, hand_image, hand_mask
                        )
                if flower_instances is not None:
                    flower_restore, flower_record = _flower_restore_mask(
                        cv2,
                        np,
                        source_frame=source_frame,
                        instance_mask=flower_instances[decoded],
                        safety_mask=safety,
                        changed_support=outer,
                        skin_dilation_pixels=flower_skin_dilation_pixels,
                    )
                    restore_pixels = flower_record["restore_pixels"]
                    flower_restore_pixels.append(restore_pixels)
                    flower_restore_skin_overlap_pixels.append(
                        flower_record["restore_skin_overlap_pixels"]
                    )
                    if restore_pixels:
                        flower_pre_restore_mae.append(
                            float(
                                np.mean(
                                    np.abs(
                                        candidate[flower_restore].astype(np.float32)
                                        - source_frame[flower_restore].astype(np.float32)
                                    )
                                )
                            )
                        )
                        candidate[flower_restore] = source_frame[flower_restore]
                        flower_post_restore_difference_pixels.append(
                            int(
                                np.count_nonzero(
                                    candidate[flower_restore]
                                    != source_frame[flower_restore]
                                )
                            )
                        )
                    else:
                        flower_pre_restore_mae.append(0.0)
                        flower_post_restore_difference_pixels.append(0)
                outside = outer == 0
                background_scores.append(
                    float(
                        np.count_nonzero(
                            np.all(candidate[outside] == source_frame[outside], axis=1)
                        )
                    )
                    / max(1, int(np.count_nonzero(outside)))
                )
                gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
                laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
                sharpness_scores.append(float(np.mean(laplacian[safety > 0])))
                small_gray = cv2.resize(gray, (256, 144), interpolation=cv2.INTER_AREA)
                if previous_gray is not None:
                    transition_energy.append(
                        float(np.mean(cv2.absdiff(small_gray, previous_gray)))
                    )
                previous_gray = small_gray
                assert writer.stdin is not None
                writer.stdin.write(candidate.tobytes())
                decoded += 1
        finally:
            capture.release()
            if writer.stdin is not None:
                writer.stdin.close()
            code = writer.wait()
            if code:
                raise RuntimeError(f"ffmpeg writer failed with code {code}")
        if decoded != frame_count:
            raise RuntimeError(f"decoded {decoded}/{frame_count} frames")
        subprocess.run(
            [str(ffmpeg), "-v", "error", "-i", str(output), "-f", "null", "-"],
            check=True,
        )
        encoded_flower_metrics = None
        if flower_instances is not None:
            encoded_flower_metrics = _measure_encoded_flower_preservation(
                cv2,
                np,
                source=paths["source"],
                output=output,
                instance_masks=flower_instances,
                safety_mask=safety,
                skin_dilation_pixels=flower_skin_dilation_pixels,
            )
        _review(ffmpeg, output, final)
        piece_scale_metrics = {
            name: _series_statistics(np, values)
            for name, values in piece_scales.items()
        }
        piece_area_metrics = {
            name: _series_statistics(np, values)
            for name, values in piece_mask_areas.items()
        }
        hand_scale_metrics = {
            name: statistics
            for name, statistics in piece_scale_metrics.items()
            if name.endswith("_hand")
        }
        hand_area_metrics = {
            name: statistics
            for name, statistics in piece_area_metrics.items()
            if name.endswith("_hand")
        }
        bouquet_enabled = bouquet_image is not None
        bouquet_area_metrics = (
            _series_statistics(np, bouquet_rendered_areas)
            if bouquet_rendered_areas
            else None
        )
        bouquet_metrics = {
            "enabled": bouquet_enabled,
            "required": require_flower_protection,
            "strategy": (
                "complete_object_rgba_bound_inside_rendered_holding_hand"
                if bouquet_enabled and bouquet_target_hand_side is not None
                else "complete_object_rgba_bound_to_source_wrist_midpoint"
                if bouquet_enabled
                else None
            ),
            "source_bbox": bouquet_source_bbox,
            "source_grip_anchor_xy": (
                bouquet_source_anchor.tolist()
                if bouquet_source_anchor is not None
                else None
            ),
            "source_grip_uv": bouquet_grip_uv.tolist(),
            "target_landmarks": list(bouquet_target_landmarks),
            "target_hand_side": bouquet_target_hand_side,
            "target_hand_distance_pixels": bouquet_target_hand_distance_pixels,
            "target_motion_filter": {
                "method": "zero_phase_gaussian_then_symmetric_vector_step_bound",
                "smoothing_sigma_frames": bouquet_target_smoothing_sigma,
                "configured_maximum_step_pixels": (
                    bouquet_target_motion_maximum_step_pixels
                ),
                "raw_maximum_step_pixels": (
                    float(
                        np.max(
                            np.linalg.norm(
                                np.diff(bouquet_raw_targets, axis=0), axis=1
                            )
                        )
                    )
                    if bouquet_raw_targets is not None
                    else 0.0
                ),
                "maximum_filtered_deviation_from_raw_pixels": (
                    float(
                        np.max(
                            np.linalg.norm(
                                bouquet_targets - bouquet_raw_targets, axis=1
                            )
                        )
                    )
                    if bouquet_targets is not None
                    and bouquet_raw_targets is not None
                    else 0.0
                ),
            },
            "target_offset_xy": bouquet_target_offset_xy.tolist(),
            "target_height_pixels": bouquet_target_height_pixels,
            "fixed_scale": bouquet_scale,
            "fixed_angle_degrees": bouquet_angle_degrees,
            "rendered_area_pixels": bouquet_area_metrics,
            "rendered_frames": len(bouquet_rendered_areas),
            "minimum_rendered_area_pixels": (
                min(bouquet_rendered_areas) if bouquet_rendered_areas else 0.0
            ),
            "maximum_grip_mapping_error_pixels": (
                max(bouquet_grip_mapping_errors)
                if bouquet_grip_mapping_errors
                else 0.0
            ),
            "maximum_target_step_pixels": (
                max(bouquet_target_steps) if bouquet_target_steps else 0.0
            ),
            "minimum_hand_occlusion_pixels": (
                min(bouquet_hand_overlap_pixels)
                if bouquet_hand_overlap_pixels
                else 0
            ),
            "mean_hand_occlusion_pixels": (
                float(np.mean(bouquet_hand_overlap_pixels))
                if bouquet_hand_overlap_pixels
                else 0.0
            ),
            "frames_without_required_hand_occlusion": [
                frame
                for frame, pixels in enumerate(bouquet_hand_overlap_pixels)
                if pixels
                < int(payload.get("minimum_bouquet_hand_overlap_pixels", 25))
            ],
        }
        flower_metrics = {
            "enabled": flower_instances is not None,
            "required": require_flower_protection,
            "skin_negative_dilation_pixels": flower_skin_dilation_pixels,
            "minimum_restore_pixels": (
                min(flower_restore_pixels) if flower_restore_pixels else 0
            ),
            "mean_restore_pixels": (
                float(np.mean(flower_restore_pixels))
                if flower_restore_pixels
                else 0.0
            ),
            "frames_without_restore_pixels": sum(
                pixels == 0 for pixels in flower_restore_pixels
            ),
            "mean_pre_restore_rgb_mae": (
                float(np.mean(flower_pre_restore_mae))
                if flower_pre_restore_mae
                else 0.0
            ),
            "maximum_post_restore_difference_pixels": (
                max(flower_post_restore_difference_pixels)
                if flower_post_restore_difference_pixels
                else 0
            ),
            "maximum_restore_skin_overlap_pixels": (
                max(flower_restore_skin_overlap_pixels)
                if flower_restore_skin_overlap_pixels
                else 0
            ),
            "encoded": encoded_flower_metrics,
            "bouquet_layer": bouquet_metrics,
        }
        metrics = {
            "decoded_frames": decoded,
            "background_lock": float(np.mean(background_scores)),
            "source_blend_weight_inside_person_clear": 0.0,
            "robot_identity_sources_per_frame": 1,
            "cross_dissolve": False,
            "hard_anchor_cuts": 0,
            "mean_robot_region_laplacian": float(np.mean(sharpness_scores)),
            "median_full_frame_transition_energy": float(np.median(transition_energy)),
            "maximum_full_frame_transition_ratio": float(
                max(transition_energy) / max(float(np.median(transition_energy)), 1e-6)
            ),
            "person_clear_coverage": float(np.count_nonzero(safety) / safety.size),
            "maximum_segment_endpoint_mapping_error_pixels": max(
                endpoint_mapping_errors
            ),
            "maximum_hand_root_mapping_error_pixels": max(
                hand_root_mapping_errors
            ),
            "maximum_hand_landmark_mapping_error_pixels": max(
                hand_landmark_mapping_errors
            ),
            "piece_transform_scale": piece_scale_metrics,
            "piece_rendered_mask_area_pixels": piece_area_metrics,
            "piece_mask_overlap": piece_mask_overlap,
            "source_piece_component_count": source_piece_components,
            "maximum_limb_component_count": {
                side: max(values) for side, values in limb_component_counts.items()
            },
            "disconnected_limb_frames": {
                side: [
                    frame
                    for frame, count in enumerate(values)
                    if count != 1
                ]
                for side, values in limb_component_counts.items()
            },
            "hand_orientation": hand_orientation,
            "lock_hand_morphology": lock_hand_morphology,
            "fixed_hand_scale": fixed_hand_scales,
            "disjoint_piece_masks": disjoint_piece_masks,
            "anisotropic_limb_mapping": anisotropic_limb_mapping,
            "fixed_arm_thickness_scale": fixed_arm_thickness_scale,
            "flower_preservation": flower_metrics,
            "bouquet_layer": bouquet_metrics,
            "pose_smoothing_sigma_frames": smoothing_sigma,
            "temporal_median_radius_frames": temporal_median_radius,
            "outlier_threshold_pixels": outlier_threshold_pixels,
            "exact_endpoint_mapping": exact_endpoint_mapping,
            "robot_anchor_alignment": robot_alignment,
            "pose_correspondence": tracking["correspondence"],
            **tracking,
        }
        correspondence = tracking["correspondence"]
        acceptance = {
            "full_clip_decoded": decoded == frame_count,
            "background_lock_passed": metrics["background_lock"] >= 0.99999,
            "hard_person_clear_passed": metrics[
                "source_blend_weight_inside_person_clear"
            ] == 0.0,
            "no_cross_dissolve_passed": not metrics["cross_dissolve"],
            "no_anchor_cut_passed": metrics["hard_anchor_cuts"] == 0,
            "transition_passed": metrics["maximum_full_frame_transition_ratio"] <= 4.0,
            "all_pose_frames_observed": tracking["missing_pose_frames"] == 0,
            "temporal_index_exact": correspondence[
                "temporal_index_map_mismatch_frames"
            ]
            == 0,
            "robust_zero_phase_filter_used": tracking["smoothing"]["method"]
            == "centered_temporal_median_then_zero_phase_gaussian",
            "all_detected_outliers_replaced": tracking["outlier_repair"][
                "all_detected_outliers_replaced"
            ],
            "pose_deviation_bounded": correspondence[
                "inlier_pose_rms_deviation_fraction_of_diagonal"
            ]
            <= float(payload.get("maximum_pose_rms_deviation_fraction", 0.01)),
            "action_direction_correspondence_passed": correspondence[
                "action_direction_cosine"
            ]
            >= float(payload.get("minimum_velocity_direction_cosine", 0.80)),
            "action_magnitude_correspondence_passed": correspondence[
                "action_magnitude_correlation"
            ]
            >= float(payload.get("minimum_action_magnitude_correlation", 0.90)),
            "no_excess_velocity_frames": correspondence[
                "smoothed_velocity_exceeds_local_source_frames"
            ]
            == 0,
            "target_step_bounded": correspondence[
                "maximum_smoothed_velocity_fraction_of_diagonal"
            ]
            <= float(payload.get("maximum_target_step_fraction", 0.025)),
            "target_jerk_bounded": correspondence[
                "maximum_smoothed_jerk_fraction_of_diagonal"
            ]
            <= float(payload.get("maximum_target_jerk_fraction", 0.005)),
            "segment_endpoints_exact": metrics[
                "maximum_segment_endpoint_mapping_error_pixels"
            ]
            <= (1e-6 if exact_endpoint_mapping else 64.0),
            "hand_roots_exact": metrics[
                "maximum_hand_root_mapping_error_pixels"
            ]
            <= 1e-6,
            "hand_transform_scale_stable": all(
                statistics["p99_to_p01_ratio"]
                <= float(payload.get("maximum_hand_scale_p99_p01_ratio", 1.001))
                and statistics["maximum_frame_step"]
                <= float(payload.get("maximum_hand_scale_frame_step", 0.001))
                for statistics in hand_scale_metrics.values()
            ),
            "hand_rendered_area_stable": all(
                statistics["p99_to_p01_ratio"]
                <= float(payload.get("maximum_hand_area_p99_p01_ratio", 1.08))
                for statistics in hand_area_metrics.values()
            ),
            "hand_proportion_calibrated": all(
                float(payload.get("minimum_fixed_hand_scale", 0.0))
                <= scale
                <= float(payload.get("maximum_fixed_hand_scale", float("inf")))
                for scale in fixed_hand_scales.values()
            ),
            "flower_layer_present": (
                not require_flower_protection
                or (
                    bouquet_enabled
                    and bouquet_metrics["rendered_frames"] == decoded
                    and bouquet_metrics["minimum_rendered_area_pixels"]
                    >= float(payload.get("minimum_bouquet_area_pixels", 1000.0))
                )
                or (
                    flower_metrics["enabled"]
                    and flower_metrics["frames_without_restore_pixels"]
                    <= int(payload.get("maximum_flower_empty_restore_frames", 0))
                )
            ),
            "flower_source_pixels_preserved": (
                not require_flower_protection
                or bouquet_enabled
                or (
                    flower_metrics["maximum_post_restore_difference_pixels"] == 0
                    and flower_metrics["maximum_restore_skin_overlap_pixels"] == 0
                )
            ),
            "encoded_flower_similarity_passed": (
                not require_flower_protection
                or bouquet_enabled
                or (
                    encoded_flower_metrics is not None
                    and encoded_flower_metrics["mean_rgb_mae"]
                    <= float(payload.get("maximum_encoded_flower_mean_rgb_mae", 5.0))
                    and encoded_flower_metrics["p95_rgb_mae"]
                    <= float(payload.get("maximum_encoded_flower_p95_rgb_mae", 8.0))
                )
            ),
            "bouquet_grip_mapping_exact": (
                not bouquet_enabled
                or bouquet_metrics["maximum_grip_mapping_error_pixels"] <= 1e-6
            ),
            "bouquet_morphology_locked": (
                not bouquet_enabled
                or (
                    bouquet_area_metrics is not None
                    and bouquet_area_metrics["p99_to_p01_ratio"]
                    <= float(payload.get("maximum_bouquet_area_p99_p01_ratio", 1.02))
                )
            ),
            "bouquet_motion_step_bounded": (
                not bouquet_enabled
                or bouquet_metrics["maximum_target_step_pixels"]
                <= float(payload.get("maximum_bouquet_target_step_pixels", 20.0))
            ),
            "bouquet_contact_occluded_by_hands": (
                not bouquet_enabled
                or bouquet_metrics["minimum_hand_occlusion_pixels"]
                >= int(payload.get("minimum_bouquet_hand_overlap_pixels", 25))
            ),
            "hand_orientation_step_bounded": all(
                record["maximum_hand_angle_step_degrees"]
                <= float(payload.get("maximum_hand_angle_step_degrees", 8.0))
                + 1e-9
                for record in hand_orientation.values()
            ),
            "cross_side_piece_masks_disjoint": piece_mask_overlap[
                "cross_side_overlap_pixels"
            ]
            == 0,
            "limb_chain_connected": all(
                count == 1
                for values in limb_component_counts.values()
                for count in values
            ),
            "robot_alpha_alignment_passed": robot_alignment is None
            or (
                robot_alignment["inliers"] >= 30
                and robot_alignment["inlier_ratio"] >= 0.80
            ),
            "human_review_passed": args.human_review == "passed",
        }
        accepted = all(acceptance.values())
        trace.update(
            {
                "status": "accepted" if accepted else "rejected",
                "honest_status": "WORKING" if accepted else "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "clean_base_method": clean_base_method,
                "robot_anchor_alignment": robot_alignment,
                "acceptance": acceptance,
                "outputs": {
                    "video": str(output),
                    "video_sha256": _sha256(output),
                    "storyboard": str(final / "storyboard-16.jpg"),
                    "dense_review": str(final / "dense-review.jpg"),
                    "early_consecutive_review": str(
                        final / "early-consecutive-review.jpg"
                    ),
                    "pose_trajectory": str(trajectory_path),
                    "pose_trajectory_sha256": _sha256(trajectory_path),
                },
                "limitations": [
                    "This is image-space rigid-part pose retargeting, not official PhiZero inference or real-robot execution.",
                    "The torso stays at one camera pose while six 2D arm/hand pieces follow source landmarks.",
                    "Exact mapping covers 2D shoulder, elbow, and wrist joints. Hand roots remain exact while hand size is morphology-locked and unreliable wrist-index directions are bridged continuously.",
                    "The morphology lock deliberately does not force the noisy MediaPipe wrist-index endpoint to coincide with the rendered robot fingertip.",
                    "The complete bouquet is one fixed 2D RGBA object whose grip anchor follows the midpoint of configured source hand landmarks; it does not simulate petal or stem deformation.",
                    "Robot arms and hands are composited after the bouquet so the grippers occlude the stems at contact.",
                    "Regions behind the reference arms are deterministically inpainted from one generated anchor.",
                    "Person-clear and background-lock invariants are measured before lossy H.264 encoding.",
                ],
            }
        )
        _write_json(trace_path, trace)
        _write_json(final / "manifest.json", trace)
        print(
            json.dumps(
                {
                    "experiment": str(experiment),
                    "status": trace["status"],
                    "honest_status": trace["honest_status"],
                    "video": str(output),
                    "metrics": metrics,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if accepted else 2
    except Exception as error:
        trace.update(
            {
                "status": "failed",
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write_json(trace_path, trace)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
