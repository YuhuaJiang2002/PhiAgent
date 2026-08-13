"""Coordinate-explicit metrics for video-to-action labeling experiments."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


POSITION_CHANNELS = (0, 1, 2, 7, 8, 9)
ROTATION_GROUPS = ((3, 4, 5), (10, 11, 12))
GRIPPER_CHANNELS = (6, 13)


def _rows(values: Sequence[Sequence[float]], label: str) -> tuple[tuple[float, ...], ...]:
    result = tuple(tuple(float(value) for value in row) for row in values)
    if not result or any(len(row) != 14 for row in result):
        raise ValueError(f"{label} must be a non-empty sequence of 14-D rows")
    if any(not math.isfinite(value) for row in result for value in row):
        raise ValueError(f"{label} must contain only finite values")
    return result


def wrap_radians(value: float) -> float:
    """Wrap an angle into [-pi, pi)."""

    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def eef_state_deltas(states: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Convert 14-D absolute dual-arm EEF states into frame-to-frame deltas."""

    rows = _rows(states, "states")
    deltas = []
    for previous, current in zip(rows, rows[1:]):
        delta = [current[index] - previous[index] for index in range(14)]
        for group in ROTATION_GROUPS:
            for channel in group:
                delta[channel] = wrap_radians(delta[channel])
        deltas.append(tuple(delta))
    return tuple(deltas)


def integrate_eef_deltas(
    initial_state: Sequence[float],
    deltas: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """Integrate dual-arm EEF deltas from one measured initial state."""

    initial = _rows((initial_state,), "initial_state")[0]
    delta_rows = _rows(deltas, "deltas")
    states = [initial]
    for delta in delta_rows:
        current = [left + right for left, right in zip(states[-1], delta)]
        for group in ROTATION_GROUPS:
            for channel in group:
                current[channel] = wrap_radians(current[channel])
        states.append(tuple(current))
    return tuple(states)


def _rotation_matrix_xyz(euler: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    x, y, z = (float(value) for value in euler)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    return (
        (cy * cz, -cy * sz, sy),
        (cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx),
        (sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy),
    )


def _rotation_geodesic_degrees(first: Sequence[float], second: Sequence[float]) -> float:
    left = _rotation_matrix_xyz(first)
    right = _rotation_matrix_xyz(second)
    trace = sum(
        left[row][column] * right[row][column]
        for row in range(3)
        for column in range(3)
    )
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    if cosine >= 1.0 - 1e-12:
        return 0.0
    return math.degrees(math.acos(cosine))


def video_action_episode_metrics(
    predicted_deltas: Sequence[Sequence[float]],
    target_deltas: Sequence[Sequence[float]],
    predicted_states: Sequence[Sequence[float]],
    target_states: Sequence[Sequence[float]],
    *,
    channel_scale: Sequence[float],
) -> dict[str, float]:
    """Evaluate one clip without treating its frames as independent trials."""

    predicted_delta_rows = _rows(predicted_deltas, "predicted_deltas")
    target_delta_rows = _rows(target_deltas, "target_deltas")
    predicted_state_rows = _rows(predicted_states, "predicted_states")
    target_state_rows = _rows(target_states, "target_states")
    scale = tuple(float(value) for value in channel_scale)
    if len(predicted_delta_rows) != len(target_delta_rows):
        raise ValueError("predicted and target delta counts differ")
    if len(predicted_state_rows) != len(target_state_rows):
        raise ValueError("predicted and target state counts differ")
    if len(predicted_state_rows) != len(predicted_delta_rows) + 1:
        raise ValueError("state sequences must contain one more row than deltas")
    if len(scale) != 14 or any(not math.isfinite(value) or value <= 0 for value in scale):
        raise ValueError("channel_scale must contain 14 finite positive values")
    delta_squared = [
        (predicted[channel] - target[channel]) ** 2
        for predicted, target in zip(predicted_delta_rows, target_delta_rows)
        for channel in range(14)
    ]
    normalized_squared = [
        ((predicted[channel] - target[channel]) / scale[channel]) ** 2
        for predicted, target in zip(predicted_delta_rows, target_delta_rows)
        for channel in range(14)
    ]
    translation_delta_squared = [
        (predicted[channel] - target[channel]) ** 2
        for predicted, target in zip(predicted_delta_rows, target_delta_rows)
        for channel in POSITION_CHANNELS
    ]
    translation_state_squared = [
        (predicted[channel] - target[channel]) ** 2
        for predicted, target in zip(predicted_state_rows, target_state_rows)
        for channel in POSITION_CHANNELS
    ]
    rotation_delta = [
        _rotation_geodesic_degrees(
            tuple(predicted[channel] for channel in group),
            tuple(target[channel] for channel in group),
        )
        for predicted, target in zip(predicted_delta_rows, target_delta_rows)
        for group in ROTATION_GROUPS
    ]
    rotation_state = [
        _rotation_geodesic_degrees(
            tuple(predicted[channel] for channel in group),
            tuple(target[channel] for channel in group),
        )
        for predicted, target in zip(predicted_state_rows, target_state_rows)
        for group in ROTATION_GROUPS
    ]
    gripper_delta = [
        abs(predicted[channel] - target[channel])
        for predicted, target in zip(predicted_delta_rows, target_delta_rows)
        for channel in GRIPPER_CHANNELS
    ]
    gripper_state = [
        abs(predicted[channel] - target[channel])
        for predicted, target in zip(predicted_state_rows, target_state_rows)
        for channel in GRIPPER_CHANNELS
    ]
    endpoint_translation = math.sqrt(
        statistics.fmean(
            (predicted_state_rows[-1][channel] - target_state_rows[-1][channel]) ** 2
            for channel in POSITION_CHANNELS
        )
    )
    return {
        "delta_rmse": math.sqrt(statistics.fmean(delta_squared)),
        "normalized_delta_rmse": math.sqrt(statistics.fmean(normalized_squared)),
        "translation_delta_rmse_cm": 100.0
        * math.sqrt(statistics.fmean(translation_delta_squared)),
        "rotation_delta_geodesic_deg": statistics.fmean(rotation_delta),
        "gripper_delta_mae": statistics.fmean(gripper_delta),
        "absolute_translation_rmse_cm": 100.0
        * math.sqrt(statistics.fmean(translation_state_squared)),
        "absolute_rotation_geodesic_deg": statistics.fmean(rotation_state),
        "absolute_gripper_mae": statistics.fmean(gripper_state),
        "endpoint_translation_rmse_cm": 100.0 * endpoint_translation,
    }


def aggregate_video_action_groups(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Average clips within their physical episode before method aggregation."""

    if not records:
        raise ValueError("video-action aggregation requires records")
    methods: dict[str, dict[str, list[dict[str, float]]]] = {}
    metric_names: set[str] | None = None
    for record in records:
        method = str(record.get("method", "")).strip()
        group = str(record.get("independent_group_id", "")).strip()
        raw_metrics = record.get("metrics")
        if not method or not group or not isinstance(raw_metrics, Mapping):
            raise ValueError("records require method, independent_group_id, and metrics")
        metrics = {str(name): float(value) for name, value in raw_metrics.items()}
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError("video-action metrics must be finite")
        if metric_names is None:
            metric_names = set(metrics)
        elif set(metrics) != metric_names:
            raise ValueError("video-action records must contain identical metrics")
        methods.setdefault(method, {}).setdefault(group, []).append(metrics)
    expected_groups: set[str] | None = None
    result = {}
    for method, groups in sorted(methods.items()):
        if expected_groups is None:
            expected_groups = set(groups)
        elif set(groups) != expected_groups:
            raise ValueError("video-action methods must cover identical physical episodes")
        per_group = {
            group: {
                metric: statistics.fmean(row[metric] for row in rows)
                for metric in sorted(metric_names or ())
            }
            for group, rows in sorted(groups.items())
        }
        result[method] = {
            "independent_groups": len(per_group),
            "raw_clips": sum(len(rows) for rows in groups.values()),
            "per_group": per_group,
            "mean": {
                metric: statistics.fmean(
                    values[metric] for values in per_group.values()
                )
                for metric in sorted(metric_names or ())
            },
        }
    return result
