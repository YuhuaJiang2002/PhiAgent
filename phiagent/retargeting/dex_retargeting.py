"""Optional dex-retargeting adapter for EPL wrist/fingertip geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phiagent.data.schema import EmbodimentDescriptor, RobotTrajectory
from phiagent.physical_language.schema import EPLChunk, EPLSequence
from phiagent.retargeting.base import RetargetingResult

DEX_RETARGETING_VERSION = "0.4.6"
SUPPORTED_LANDMARK_INDICES = (0, 4, 8, 12, 16, 20)


def epl_landmarks(chunk: EPLChunk) -> dict[int, tuple[float, float, float]]:
    """Return OpenPose-indexed wrist and fingertips available in EPL v0.1."""

    if any(point.frame != chunk.wrist_pose.target_frame for point in chunk.fingertips):
        raise ValueError("EPL wrist and fingertip frames differ")
    return {
        0: chunk.wrist_pose.translation_m,
        **{
            index: point.xyz_m
            for index, point in zip((4, 8, 12, 16, 20), chunk.fingertips)
        },
    }


@dataclass(frozen=True)
class DexRetargetingConfig:
    yaml_path: Path
    embodiment: EmbodimentDescriptor
    fixed_joint_positions: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.yaml_path.is_file():
            raise ValueError(f"dex-retargeting config does not exist: {self.yaml_path}")
        if not all(
            name.strip() and math.isfinite(float(value))
            for name, value in self.fixed_joint_positions.items()
        ):
            raise ValueError("fixed dex-retargeting joints must be named and finite")


class DexEPLRetargeter:
    """Use invariant hand vectors; absolute-position mode is rejected for now."""

    def __init__(self, config: DexRetargetingConfig) -> None:
        self.config = config
        try:
            import dex_retargeting
            import numpy as np
            from dex_retargeting.retargeting_config import RetargetingConfig
        except ImportError as exc:
            raise RuntimeError(
                "DexEPLRetargeter requires dex_retargeting==0.4.6"
            ) from exc
        package_version = getattr(dex_retargeting, "__version__", None)
        if package_version not in (None, DEX_RETARGETING_VERSION):
            raise RuntimeError(
                f"dex-retargeting version mismatch: {package_version} != "
                f"{DEX_RETARGETING_VERSION}"
            )
        raw_config = RetargetingConfig.load_from_file(config.yaml_path)
        self._retargeting = raw_config.build()
        self._np = np
        retargeting_type = self._retargeting.optimizer.retargeting_type
        if retargeting_type == "POSITION":
            raise ValueError(
                "dex-retargeting POSITION mode requires a calibrated camera-to-robot "
                "transform and is intentionally disabled; use VECTOR or DEXPILOT"
            )
        indices = self._retargeting.optimizer.target_link_human_indices.reshape(-1)
        unsupported = sorted(
            {int(index) for index in indices}.difference(SUPPORTED_LANDMARK_INDICES)
        )
        if unsupported:
            raise ValueError(
                "dex-retargeting config requests hand joints not present in EPL v0.1: "
                f"{unsupported}"
            )
        robot_names = tuple(self._retargeting.optimizer.robot.dof_joint_names)
        missing = set(config.embodiment.joint_names).difference(robot_names)
        if missing:
            raise ValueError(
                f"embodiment joints are missing from dex-retargeting URDF: {sorted(missing)}"
            )
        self._robot_names = robot_names

    def _reference(self, chunk: EPLChunk) -> Any:
        landmarks = epl_landmarks(chunk)
        indices = self._retargeting.optimizer.target_link_human_indices
        origins = indices[0, :]
        tasks = indices[1, :]
        return self._np.asarray(
            [
                tuple(
                    task - origin
                    for task, origin in zip(
                        landmarks[int(task_index)], landmarks[int(origin_index)]
                    )
                )
                for origin_index, task_index in zip(origins, tasks)
            ],
            dtype=self._np.float32,
        )

    def retarget(self, epl: EPLSequence) -> RetargetingResult:
        if not epl.chunks:
            raise ValueError("cannot retarget an empty EPL sequence")
        fixed_names = tuple(self._retargeting.optimizer.fixed_joint_names)
        missing_fixed = set(fixed_names).difference(self.config.fixed_joint_positions)
        if missing_fixed:
            raise ValueError(
                "fixed dex-retargeting joints need explicit values: "
                f"{sorted(missing_fixed)}"
            )
        fixed = self._np.asarray(
            [self.config.fixed_joint_positions[name] for name in fixed_names],
            dtype=self._np.float32,
        )
        samples = []
        timestamps = []
        for chunk in epl.chunks:
            qpos = self._retargeting.retarget(self._reference(chunk), fixed)
            by_name = dict(zip(self._robot_names, (float(value) for value in qpos)))
            samples.append(tuple(by_name[name] for name in self.config.embodiment.joint_names))
            timestamps.append(chunk.end_s)
        trajectory = RobotTrajectory(
            schema_version="0.1.0",
            embodiment=self.config.embodiment,
            timestamps_s=tuple(timestamps),
            joint_positions_rad=tuple(samples),
            source_epl=epl.source_video,
        )
        failures = tuple(
            {
                "sample_index": int(item["sample_index"]),
                "timestamp_s": float(item["timestamp_s"]),
                "joint": str(item["joint"]),
                "value_rad": float(item["value_rad"]),
            }
            for item in trajectory.joint_limit_violations()
        )
        return RetargetingResult(trajectory, failures)
