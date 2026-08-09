"""Canonical robot embodiment and trajectory representations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from phiagent.physical_language.schema import FrameKind, PoseSE3


def _finite_tuple(values: tuple[float, ...], label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values")
    return result


@dataclass(frozen=True)
class EmbodimentDescriptor:
    name: str
    joint_names: tuple[str, ...]
    lower_limits_rad: tuple[float, ...]
    upper_limits_rad: tuple[float, ...]
    end_effector_frame: str
    urdf_path: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.end_effector_frame.strip():
            raise ValueError("embodiment name and end_effector_frame must be non-empty")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        dof = len(self.joint_names)
        if len(self.lower_limits_rad) != dof or len(self.upper_limits_rad) != dof:
            raise ValueError("joint names and limit vectors must have the same length")
        lower = _finite_tuple(self.lower_limits_rad, "lower_limits_rad")
        upper = _finite_tuple(self.upper_limits_rad, "upper_limits_rad")
        if any(low >= high for low, high in zip(lower, upper)):
            raise ValueError("every lower joint limit must be below its upper limit")
        object.__setattr__(self, "lower_limits_rad", lower)
        object.__setattr__(self, "upper_limits_rad", upper)

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "joint_names": list(self.joint_names),
            "lower_limits_rad": list(self.lower_limits_rad),
            "upper_limits_rad": list(self.upper_limits_rad),
            "end_effector_frame": self.end_effector_frame,
            "urdf_path": self.urdf_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EmbodimentDescriptor:
        return cls(
            name=str(payload["name"]),
            joint_names=tuple(payload["joint_names"]),
            lower_limits_rad=tuple(payload["lower_limits_rad"]),
            upper_limits_rad=tuple(payload["upper_limits_rad"]),
            end_effector_frame=str(payload["end_effector_frame"]),
            urdf_path=(
                str(payload["urdf_path"]) if payload.get("urdf_path") is not None else None
            ),
        )


@dataclass(frozen=True)
class CanonicalAction:
    """One padded action vector and its embodiment mask."""

    values: tuple[float, ...]
    embodiment_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        values = _finite_tuple(self.values, "canonical action")
        if not values or len(values) != len(self.embodiment_mask):
            raise ValueError("action values and embodiment mask must have equal non-zero length")
        if any(not enabled and abs(value) > 1e-12 for value, enabled in zip(values, self.embodiment_mask)):
            raise ValueError("masked canonical action dimensions must be zero")
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "embodiment_mask": list(self.embodiment_mask),
        }

    @classmethod
    def from_joint_positions(
        cls, joint_positions: tuple[float, ...], canonical_dimension: int
    ) -> CanonicalAction:
        if canonical_dimension < len(joint_positions):
            raise ValueError("canonical dimension is smaller than embodiment DOF")
        padding = canonical_dimension - len(joint_positions)
        return cls(
            values=tuple(joint_positions) + (0.0,) * padding,
            embodiment_mask=(True,) * len(joint_positions) + (False,) * padding,
        )


@dataclass(frozen=True)
class RobotTrajectory:
    schema_version: str
    embodiment: EmbodimentDescriptor
    timestamps_s: tuple[float, ...]
    joint_positions_rad: tuple[tuple[float, ...], ...]
    source_epl: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "0.1.0":
            raise ValueError(f"unsupported trajectory schema {self.schema_version!r}")
        if not self.timestamps_s or len(self.timestamps_s) != len(self.joint_positions_rad):
            raise ValueError("timestamps and joint positions must have equal non-zero length")
        timestamps = _finite_tuple(self.timestamps_s, "timestamps_s")
        if timestamps[0] < 0 or any(
            current <= previous for previous, current in zip(timestamps, timestamps[1:])
        ):
            raise ValueError("trajectory timestamps must be non-negative and strictly increasing")
        positions: list[tuple[float, ...]] = []
        for sample in self.joint_positions_rad:
            values = _finite_tuple(sample, "joint_positions_rad")
            if len(values) != self.embodiment.dof:
                raise ValueError("trajectory sample DOF does not match embodiment")
            positions.append(values)
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "joint_positions_rad", tuple(positions))

    def joint_limit_violations(self, tolerance: float = 1e-9) -> tuple[dict[str, Any], ...]:
        violations: list[dict[str, Any]] = []
        for sample_index, positions in enumerate(self.joint_positions_rad):
            for joint_index, (value, low, high) in enumerate(
                zip(
                    positions,
                    self.embodiment.lower_limits_rad,
                    self.embodiment.upper_limits_rad,
                )
            ):
                if value < low - tolerance or value > high + tolerance:
                    violations.append(
                        {
                            "sample_index": sample_index,
                            "timestamp_s": self.timestamps_s[sample_index],
                            "joint": self.embodiment.joint_names[joint_index],
                            "value_rad": value,
                            "lower_rad": low,
                            "upper_rad": high,
                        }
                    )
        return tuple(violations)

    def canonical_actions(self, dimension: int) -> tuple[CanonicalAction, ...]:
        return tuple(
            CanonicalAction.from_joint_positions(sample, dimension)
            for sample in self.joint_positions_rad
        )

    def resample(self, fps: int) -> RobotTrajectory:
        """Linearly resample the same joint path at a fixed video frame rate."""

        if fps <= 0:
            raise ValueError("resample FPS must be positive")
        start_s = self.timestamps_s[0]
        end_s = self.timestamps_s[-1]
        frame_count = round((end_s - start_s) * fps) + 1
        if frame_count < 2 or abs(start_s + (frame_count - 1) / fps - end_s) > 1e-9:
            raise ValueError("trajectory duration must contain an integer number of video frames")
        timestamps = tuple(start_s + index / fps for index in range(frame_count))
        samples: list[tuple[float, ...]] = []
        segment = 0
        for timestamp in timestamps:
            while (
                segment + 1 < len(self.timestamps_s) - 1
                and self.timestamps_s[segment + 1] < timestamp
            ):
                segment += 1
            left_s = self.timestamps_s[segment]
            right_s = self.timestamps_s[segment + 1]
            alpha = (timestamp - left_s) / (right_s - left_s)
            samples.append(
                tuple(
                    left + alpha * (right - left)
                    for left, right in zip(
                        self.joint_positions_rad[segment],
                        self.joint_positions_rad[segment + 1],
                    )
                )
            )
        return RobotTrajectory(
            schema_version=self.schema_version,
            embodiment=self.embodiment,
            timestamps_s=timestamps,
            joint_positions_rad=tuple(samples),
            source_epl=self.source_epl,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "embodiment": self.embodiment.to_dict(),
            "timestamps_s": list(self.timestamps_s),
            "joint_positions_rad": [list(sample) for sample in self.joint_positions_rad],
            "source_epl": self.source_epl,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RobotTrajectory:
        return cls(
            schema_version=str(payload["schema_version"]),
            embodiment=EmbodimentDescriptor.from_dict(payload["embodiment"]),
            timestamps_s=tuple(payload["timestamps_s"]),
            joint_positions_rad=tuple(
                tuple(sample) for sample in payload["joint_positions_rad"]
            ),
            source_epl=(
                str(payload["source_epl"]) if payload.get("source_epl") is not None else None
            ),
        )

    @classmethod
    def from_json(cls, path: Path) -> RobotTrajectory:
        return cls.from_dict(json.loads(path.read_text()))


@dataclass(frozen=True)
class RigidBodyTrajectory:
    """Frame-explicit sampled poses for one rigid object."""

    schema_version: str
    body_name: str
    timestamps_s: tuple[float, ...]
    poses: tuple[PoseSE3, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "0.1.0":
            raise ValueError(f"unsupported rigid-body trajectory schema {self.schema_version!r}")
        body_name = self.body_name.strip()
        if not body_name:
            raise ValueError("body_name must be non-empty")
        if not self.timestamps_s or len(self.timestamps_s) != len(self.poses):
            raise ValueError("timestamps and poses must have equal non-zero length")
        timestamps = _finite_tuple(self.timestamps_s, "timestamps_s")
        if timestamps[0] < 0 or any(
            current <= previous for previous, current in zip(timestamps, timestamps[1:])
        ):
            raise ValueError("trajectory timestamps must be non-negative and strictly increasing")
        source_frame = self.poses[0].source_frame
        target_frame = self.poses[0].target_frame
        if source_frame.kind is not FrameKind.OBJECT:
            raise ValueError("rigid-body trajectory poses must originate in an object frame")
        if source_frame.name != body_name:
            raise ValueError("body_name must match the object source-frame name")
        if any(
            pose.source_frame != source_frame or pose.target_frame != target_frame
            for pose in self.poses
        ):
            raise ValueError("all rigid-body poses must use the same source and target frames")
        object.__setattr__(self, "body_name", body_name)
        object.__setattr__(self, "timestamps_s", timestamps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_name": self.body_name,
            "timestamps_s": list(self.timestamps_s),
            "poses": [pose.to_dict() for pose in self.poses],
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RigidBodyTrajectory:
        return cls(
            schema_version=str(payload["schema_version"]),
            body_name=str(payload["body_name"]),
            timestamps_s=tuple(payload["timestamps_s"]),
            poses=tuple(PoseSE3.from_dict(pose) for pose in payload["poses"]),
        )

    @classmethod
    def from_json(cls, path: Path) -> RigidBodyTrajectory:
        return cls.from_dict(json.loads(path.read_text()))
