"""Retargeting protocol and an explicit linear differential baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from phiagent.data.schema import EmbodimentDescriptor, RobotTrajectory
from phiagent.physical_language.schema import EPLSequence, QuaternionXYZW


@dataclass(frozen=True)
class RetargetingResult:
    trajectory: RobotTrajectory
    reachability_failures: tuple[dict[str, float | int | str], ...]


class RobotRetargeter(Protocol):
    def retarget(self, epl: EPLSequence) -> RetargetingResult:
        """Map an EPL sequence into one robot embodiment."""


@dataclass(frozen=True)
class LinearRetargetingConfig:
    """Local 6D EEF twist-to-joint-delta map for a measured operating point."""

    embodiment: EmbodimentDescriptor
    initial_joint_positions_rad: tuple[float, ...]
    eef_twist_to_joint_delta: tuple[tuple[float, float, float, float, float, float], ...]

    def __post_init__(self) -> None:
        dof = self.embodiment.dof
        if len(self.initial_joint_positions_rad) != dof:
            raise ValueError("initial joint vector does not match embodiment DOF")
        if len(self.eef_twist_to_joint_delta) != dof:
            raise ValueError("linear retargeting matrix must have one row per joint")
        if any(len(row) != 6 for row in self.eef_twist_to_joint_delta):
            raise ValueError("linear retargeting matrix rows must have six columns")
        if not all(
            math.isfinite(value)
            for row in self.eef_twist_to_joint_delta
            for value in row
        ):
            raise ValueError("linear retargeting matrix must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "embodiment": self.embodiment.to_dict(),
            "initial_joint_positions_rad": list(self.initial_joint_positions_rad),
            "eef_twist_to_joint_delta": [
                list(row) for row in self.eef_twist_to_joint_delta
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LinearRetargetingConfig:
        return cls(
            embodiment=EmbodimentDescriptor.from_dict(payload["embodiment"]),
            initial_joint_positions_rad=tuple(payload["initial_joint_positions_rad"]),
            eef_twist_to_joint_delta=tuple(
                tuple(row) for row in payload["eef_twist_to_joint_delta"]
            ),
        )


def _quaternion_rotation_vector(quaternion: QuaternionXYZW) -> tuple[float, float, float]:
    x, y, z, w = quaternion
    clamped_w = min(1.0, max(-1.0, w))
    angle = 2.0 * math.acos(clamped_w)
    scale = math.sqrt(max(0.0, 1.0 - clamped_w * clamped_w))
    if scale < 1e-8:
        return (2.0 * x, 2.0 * y, 2.0 * z)
    return (angle * x / scale, angle * y / scale, angle * z / scale)


class LinearEPLRetargeter:
    """Deterministic baseline; not a replacement for dex-retargeting or SPIDER."""

    def __init__(self, config: LinearRetargetingConfig) -> None:
        self.config = config

    def retarget(self, epl: EPLSequence) -> RetargetingResult:
        if not epl.chunks:
            raise ValueError("cannot retarget an empty EPL sequence")
        current = tuple(float(value) for value in self.config.initial_joint_positions_rad)
        timestamps = [epl.chunks[0].start_s]
        samples = [current]
        failures: list[dict[str, float | int | str]] = []
        for chunk_index, chunk in enumerate(epl.chunks):
            rotation = _quaternion_rotation_vector(chunk.eef_delta.quaternion_xyzw)
            twist = chunk.eef_delta.translation_m + rotation
            deltas = tuple(
                sum(coefficient * value for coefficient, value in zip(row, twist))
                for row in self.config.eef_twist_to_joint_delta
            )
            proposed = tuple(value + delta for value, delta in zip(current, deltas))
            clamped_values: list[float] = []
            for joint_index, (value, low, high) in enumerate(
                zip(
                    proposed,
                    self.config.embodiment.lower_limits_rad,
                    self.config.embodiment.upper_limits_rad,
                )
            ):
                clamped = min(max(value, low), high)
                if clamped != value:
                    failures.append(
                        {
                            "chunk_index": chunk_index,
                            "timestamp_s": chunk.end_s,
                            "joint": self.config.embodiment.joint_names[joint_index],
                            "proposed_rad": value,
                            "clamped_rad": clamped,
                        }
                    )
                clamped_values.append(clamped)
            current = tuple(clamped_values)
            timestamps.append(chunk.end_s)
            samples.append(current)
        trajectory = RobotTrajectory(
            schema_version="0.1.0",
            embodiment=self.config.embodiment,
            timestamps_s=tuple(timestamps),
            joint_positions_rad=tuple(samples),
            source_epl=epl.source_video,
        )
        return RetargetingResult(trajectory, tuple(failures))


def retarget_json(
    epl_path: Path, output_path: Path, config: LinearRetargetingConfig
) -> RetargetingResult:
    result = LinearEPLRetargeter(config).retarget(EPLSequence.from_json(epl_path))
    result.trajectory.to_json(output_path)
    return result
