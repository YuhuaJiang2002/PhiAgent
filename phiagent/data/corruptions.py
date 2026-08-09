"""Deterministic trajectory corruptions for simulation-grounded repair data."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from phiagent.data.schema import RobotTrajectory


class CorruptionType(str, Enum):
    JOINT_LIMIT = "joint_limit"
    POSITION_NOISE = "position_noise"
    TEMPORAL_SCALE = "temporal_scale"
    TIMING_SHIFT = "timing_shift"


@dataclass(frozen=True)
class CorruptionConfig:
    seed: int = 0
    position_noise_std_rad: float = 0.15
    temporal_scale: float = 1.35
    timing_shift_samples: int = 1

    def __post_init__(self) -> None:
        if self.position_noise_std_rad <= 0:
            raise ValueError("position noise standard deviation must be positive")
        if self.temporal_scale <= 0 or self.temporal_scale == 1.0:
            raise ValueError("temporal scale must be positive and different from one")
        if self.timing_shift_samples < 1:
            raise ValueError("timing shift must be at least one sample")


def corrupt_trajectory(
    trajectory: RobotTrajectory,
    corruption_type: CorruptionType,
    config: CorruptionConfig | None = None,
) -> RobotTrajectory:
    """Create one reproducible negative example without mutating the source."""

    cfg = config or CorruptionConfig()
    positions = trajectory.joint_positions_rad
    timestamps = trajectory.timestamps_s
    if corruption_type is CorruptionType.JOINT_LIMIT:
        sample_index = len(positions) - 1
        joint_index = 0
        high = trajectory.embodiment.upper_limits_rad[joint_index]
        low = trajectory.embodiment.lower_limits_rad[joint_index]
        corrupted = [list(sample) for sample in positions]
        corrupted[sample_index][joint_index] = high + 0.1 * (high - low)
        positions = tuple(tuple(sample) for sample in corrupted)
    elif corruption_type is CorruptionType.POSITION_NOISE:
        generator = random.Random(cfg.seed)
        positions = tuple(
            tuple(
                value + generator.gauss(0.0, cfg.position_noise_std_rad)
                for value in sample
            )
            for sample in positions
        )
    elif corruption_type is CorruptionType.TEMPORAL_SCALE:
        origin = timestamps[0]
        timestamps = tuple(
            origin + (timestamp - origin) * cfg.temporal_scale
            for timestamp in timestamps
        )
    elif corruption_type is CorruptionType.TIMING_SHIFT:
        shift = min(cfg.timing_shift_samples, len(positions) - 1)
        if shift == 0:
            raise ValueError("timing shift requires at least two trajectory samples")
        positions = positions[:1] * shift + positions[:-shift]
    else:  # pragma: no cover - Enum construction normally prevents this branch.
        raise ValueError(f"unsupported corruption type {corruption_type}")
    return RobotTrajectory(
        schema_version=trajectory.schema_version,
        embodiment=trajectory.embodiment,
        timestamps_s=timestamps,
        joint_positions_rad=positions,
        source_epl=trajectory.source_epl,
    )
