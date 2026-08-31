"""Dependency-free synchronized action metrics for L3."""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _vector(value: object, width: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ValueError(f"{label} must contain {width} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


@dataclass(frozen=True)
class ActionTrajectory:
    coordinate_frame: str
    timestamps_s: tuple[float, ...]
    eef_positions_m: tuple[tuple[float, float, float], ...]
    eef_quaternions_xyzw: tuple[tuple[float, float, float, float], ...]
    joint_names: tuple[str, ...] = ()
    joint_positions_rad: tuple[tuple[float, ...], ...] = ()
    gripper_width_m: tuple[float, ...] = ()
    contact_state: tuple[bool, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionTrajectory":
        timestamps = tuple(float(value) for value in payload["timestamps_s"])
        if len(timestamps) < 2 or any(not math.isfinite(value) for value in timestamps):
            raise ValueError("action trajectory requires at least two finite timestamps")
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError("action timestamps must be strictly increasing")
        positions = tuple(_vector(row, 3, "eef position") for row in payload["eef_positions_m"])
        quaternions = tuple(
            _vector(row, 4, "eef quaternion") for row in payload["eef_quaternions_xyzw"]
        )
        if len(positions) != len(timestamps) or len(quaternions) != len(timestamps):
            raise ValueError("EEF trajectory fields must match timestamps")
        normalized_quaternions = []
        for quaternion in quaternions:
            norm = math.sqrt(sum(value * value for value in quaternion))
            if norm < 1e-12:
                raise ValueError("EEF quaternion cannot be zero")
            normalized_quaternions.append(tuple(value / norm for value in quaternion))
        joint_names = tuple(str(name) for name in payload.get("joint_names", ()))
        joints = tuple(
            _vector(row, len(joint_names), "joint position")
            for row in payload.get("joint_positions_rad", ())
        )
        if bool(joint_names) != bool(joints) or (joints and len(joints) != len(timestamps)):
            raise ValueError("joint names and samples must be provided together for every timestamp")
        gripper = tuple(float(value) for value in payload.get("gripper_width_m", ()))
        if gripper and (
            len(gripper) != len(timestamps)
            or any(not math.isfinite(value) or value < 0 for value in gripper)
        ):
            raise ValueError("gripper width must be finite, non-negative, and time aligned")
        contact_raw = payload.get("contact_state", ())
        if any(not isinstance(value, bool) for value in contact_raw):
            raise ValueError("contact_state entries must be boolean")
        contact = tuple(contact_raw)
        if contact and len(contact) != len(timestamps):
            raise ValueError("contact_state must align with timestamps")
        frame = str(payload["coordinate_frame"]).strip()
        if not frame:
            raise ValueError("coordinate_frame cannot be empty")
        return cls(
            coordinate_frame=frame,
            timestamps_s=timestamps,
            eef_positions_m=positions,  # type: ignore[arg-type]
            eef_quaternions_xyzw=tuple(normalized_quaternions),  # type: ignore[arg-type]
            joint_names=joint_names,
            joint_positions_rad=joints,
            gripper_width_m=gripper,
            contact_state=contact,
        )

    @classmethod
    def from_json(cls, path: Path) -> "ActionTrajectory":
        payload = json.loads(path.expanduser().resolve().read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("action trajectory JSON must be an object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class MultiArmActionTrajectory:
    """A named collection of arm trajectories in one coordinate frame."""

    coordinate_frame: str
    arms: dict[str, ActionTrajectory]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MultiArmActionTrajectory":
        frame = str(payload["coordinate_frame"]).strip()
        raw_arms = payload.get("arms")
        if not frame or not isinstance(raw_arms, Mapping) or not raw_arms:
            raise ValueError("multi-arm action requires a frame and at least one named arm")
        arms: dict[str, ActionTrajectory] = {}
        for raw_name, raw_trajectory in raw_arms.items():
            name = str(raw_name).strip()
            if not name or not isinstance(raw_trajectory, Mapping):
                raise ValueError("multi-arm action names and trajectories must be valid")
            arm_payload = dict(raw_trajectory)
            declared_frame = arm_payload.setdefault("coordinate_frame", frame)
            if declared_frame != frame:
                raise ValueError("all arm trajectories must use the bundle coordinate frame")
            arms[name] = ActionTrajectory.from_dict(arm_payload)
        return cls(coordinate_frame=frame, arms=arms)

    @classmethod
    def from_json(cls, path: Path) -> "MultiArmActionTrajectory":
        payload = json.loads(path.expanduser().resolve().read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("multi-arm action JSON must be an object")
        return cls.from_dict(payload)


def _bracket(timestamps: tuple[float, ...], timestamp: float) -> tuple[int, int, float]:
    right = bisect.bisect_left(timestamps, timestamp)
    if right == 0:
        return 0, 0, 0.0
    if right == len(timestamps):
        return len(timestamps) - 1, len(timestamps) - 1, 0.0
    if timestamps[right] == timestamp:
        return right, right, 0.0
    left = right - 1
    alpha = (timestamp - timestamps[left]) / (timestamps[right] - timestamps[left])
    return left, right, alpha


def _lerp(left: tuple[float, ...], right: tuple[float, ...], alpha: float) -> tuple[float, ...]:
    return tuple(a + alpha * (b - a) for a, b in zip(left, right))


def _quat_interp(
    left: tuple[float, ...], right: tuple[float, ...], alpha: float
) -> tuple[float, ...]:
    if sum(a * b for a, b in zip(left, right)) < 0:
        right = tuple(-value for value in right)
    mixed = _lerp(left, right, alpha)
    norm = math.sqrt(sum(value * value for value in mixed))
    return tuple(value / norm for value in mixed)


def _sample_vectors(
    trajectory: ActionTrajectory,
    values: tuple[tuple[float, ...], ...],
    timestamp: float,
    *,
    quaternion: bool = False,
) -> tuple[float, ...]:
    left, right, alpha = _bracket(trajectory.timestamps_s, timestamp)
    if left == right:
        return values[left]
    return _quat_interp(values[left], values[right], alpha) if quaternion else _lerp(values[left], values[right], alpha)


def _sample_scalar(
    trajectory: ActionTrajectory, values: tuple[float, ...], timestamp: float
) -> float:
    left, right, alpha = _bracket(trajectory.timestamps_s, timestamp)
    return values[left] if left == right else values[left] + alpha * (values[right] - values[left])


def _orientation_error_deg(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = min(1.0, max(0.0, abs(sum(a * b for a, b in zip(left, right)))))
    return math.degrees(2.0 * math.acos(dot))


def _events(
    timestamps: tuple[float, ...], states: tuple[bool, ...]
) -> list[tuple[float, bool]]:
    return [
        (timestamps[index], states[index])
        for index in range(1, len(states))
        if states[index] != states[index - 1]
    ]


def _event_f1(
    reference: list[tuple[float, bool]],
    candidate: list[tuple[float, bool]],
    tolerance_s: float,
) -> float:
    if not reference and not candidate:
        return 1.0
    matched: set[int] = set()
    true_positive = 0
    for ref_time, ref_state in reference:
        possible = [
            (abs(candidate_time - ref_time), index)
            for index, (candidate_time, candidate_state) in enumerate(candidate)
            if index not in matched
            and candidate_state == ref_state
            and abs(candidate_time - ref_time) <= tolerance_s
        ]
        if possible:
            _, index = min(possible)
            matched.add(index)
            true_positive += 1
    precision = true_positive / len(candidate) if candidate else 0.0
    recall = true_positive / len(reference) if reference else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def compare_action_trajectories(
    reference: ActionTrajectory,
    candidate: ActionTrajectory,
    *,
    gripper_closed_threshold_m: float = 0.01,
    event_tolerance_s: float = 0.15,
) -> dict[str, float]:
    """Align the candidate to reference timestamps and compute L3 raw metrics."""

    if reference.coordinate_frame != candidate.coordinate_frame:
        raise ValueError("action trajectories must use the same named coordinate frame")
    if gripper_closed_threshold_m < 0 or event_tolerance_s < 0:
        raise ValueError("gripper threshold and event tolerance cannot be negative")
    covered = [
        (index, timestamp)
        for index, timestamp in enumerate(reference.timestamps_s)
        if candidate.timestamps_s[0] <= timestamp <= candidate.timestamps_s[-1]
    ]
    coverage = len(covered) / len(reference.timestamps_s)
    if not covered:
        raise ValueError("candidate trajectory does not overlap the reference timeline")
    position_squared: list[float] = []
    orientation_squared: list[float] = []
    joint_squared: list[float] = []
    gripper_absolute: list[float] = []
    for index, timestamp in covered:
        candidate_position = _sample_vectors(candidate, candidate.eef_positions_m, timestamp)
        candidate_quaternion = _sample_vectors(
            candidate, candidate.eef_quaternions_xyzw, timestamp, quaternion=True
        )
        position_squared.append(
            sum(
                (a - b) ** 2
                for a, b in zip(reference.eef_positions_m[index], candidate_position)
            )
        )
        orientation_squared.append(
            _orientation_error_deg(reference.eef_quaternions_xyzw[index], candidate_quaternion) ** 2
        )
        if reference.joint_names and candidate.joint_names:
            if reference.joint_names != candidate.joint_names:
                raise ValueError("joint RMSE is defined only for identical ordered joint names")
            candidate_joints = _sample_vectors(candidate, candidate.joint_positions_rad, timestamp)
            joint_squared.extend(
                (a - b) ** 2
                for a, b in zip(reference.joint_positions_rad[index], candidate_joints)
            )
        if reference.gripper_width_m and candidate.gripper_width_m:
            candidate_width = _sample_scalar(candidate, candidate.gripper_width_m, timestamp)
            gripper_absolute.append(abs(reference.gripper_width_m[index] - candidate_width))

    metrics = {
        "eef_position_rmse_m": math.sqrt(sum(position_squared) / len(position_squared)),
        "eef_orientation_rmse_deg": math.sqrt(
            sum(orientation_squared) / len(orientation_squared)
        ),
        "trajectory_coverage": coverage,
    }
    if joint_squared:
        metrics["joint_rmse_rad"] = math.sqrt(sum(joint_squared) / len(joint_squared))
    if gripper_absolute:
        metrics["gripper_width_mae_m"] = sum(gripper_absolute) / len(gripper_absolute)
        reference_closed = tuple(
            value <= gripper_closed_threshold_m for value in reference.gripper_width_m
        )
        candidate_closed = tuple(
            value <= gripper_closed_threshold_m for value in candidate.gripper_width_m
        )
        metrics["gripper_event_f1"] = _event_f1(
            _events(reference.timestamps_s, reference_closed),
            _events(candidate.timestamps_s, candidate_closed),
            event_tolerance_s,
        )
    if reference.contact_state and candidate.contact_state:
        metrics["contact_event_f1"] = _event_f1(
            _events(reference.timestamps_s, reference.contact_state),
            _events(candidate.timestamps_s, candidate.contact_state),
            event_tolerance_s,
        )
    return metrics


def compare_multi_arm_trajectories(
    reference: MultiArmActionTrajectory,
    candidate: MultiArmActionTrajectory,
    *,
    gripper_closed_threshold_m: float = 0.01,
    event_tolerance_s: float = 0.15,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compare every named arm and conservatively aggregate the weakest arm."""

    if reference.coordinate_frame != candidate.coordinate_frame:
        raise ValueError("multi-arm trajectories must use the same named coordinate frame")
    if set(reference.arms) != set(candidate.arms):
        raise ValueError("reference and candidate must contain the same named arms")
    per_arm = {
        name: compare_action_trajectories(
            reference.arms[name],
            candidate.arms[name],
            gripper_closed_threshold_m=gripper_closed_threshold_m,
            event_tolerance_s=event_tolerance_s,
        )
        for name in sorted(reference.arms)
    }
    common_metrics = set.intersection(*(set(metrics) for metrics in per_arm.values()))
    goodness_metrics = {"gripper_event_f1", "contact_event_f1", "trajectory_coverage"}
    aggregate = {
        metric: (
            min(metrics[metric] for metrics in per_arm.values())
            if metric in goodness_metrics
            else max(metrics[metric] for metrics in per_arm.values())
        )
        for metric in sorted(common_metrics)
    }
    return aggregate, per_arm
