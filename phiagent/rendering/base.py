"""Interfaces shared by video rendering backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from phiagent.agent.verifier import VerificationReport
from phiagent.data.schema import RigidBodyTrajectory, RobotTrajectory
from phiagent.perception.camera import PinholeIntrinsics
from phiagent.physical_language.schema import FrameKind, PoseSE3


@dataclass(frozen=True)
class VisualTransferRequest:
    """Inputs for transferring motion from a human video to a robot image."""

    video: Path
    robot_image: Path
    output: Path
    prompt: str
    experiment_root: Path
    seed: int = 42
    overwrite: bool = False


@dataclass(frozen=True)
class VisualTransferResult:
    """Material outputs of a completed visual-transfer run."""

    output: Path
    experiment_dir: Path
    metadata: Path


class VideoRenderer(Protocol):
    """Backend-independent visual-transfer interface."""

    def render(self, request: VisualTransferRequest) -> VisualTransferResult:
        """Render a robot video and return its persisted artifacts."""


@dataclass(frozen=True)
class TrajectoryConditionedRenderRequest:
    """Inputs that bind rendering to one accepted physical rollout."""

    robot_trajectory: RobotTrajectory
    object_trajectories: tuple[RigidBodyTrajectory, ...]
    control_video: Path
    prompt: str
    camera_intrinsics: PinholeIntrinsics
    camera_T_robot_base: PoseSE3
    scene_assets: tuple[Path, ...]
    verification: VerificationReport
    verification_record: Path
    output: Path
    experiment_root: Path
    seed: int = 42
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.verification.accepted:
            raise ValueError("trajectory-conditioned rendering requires accepted verification")
        if not self.prompt.strip():
            raise ValueError("trajectory-conditioned rendering requires a non-empty prompt")
        if self.control_video.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
            raise ValueError("control_video must use a supported video extension")
        if self.camera_T_robot_base.source_frame.kind is not FrameKind.ROBOT_BASE:
            raise ValueError("camera_T_robot_base must originate in a robot-base frame")
        if self.camera_T_robot_base.target_frame.kind is not FrameKind.CAMERA:
            raise ValueError("camera_T_robot_base must target a camera frame")
        if not self.object_trajectories:
            raise ValueError("at least one object trajectory is required")
        if not self.scene_assets:
            raise ValueError("at least one scene asset is required")
        if len(set(self.scene_assets)) != len(self.scene_assets):
            raise ValueError("scene assets must be unique")
        if len({trajectory.body_name for trajectory in self.object_trajectories}) != len(
            self.object_trajectories
        ):
            raise ValueError("object trajectory body names must be unique")
        for trajectory in self.object_trajectories:
            if trajectory.poses[0].target_frame != self.camera_T_robot_base.source_frame:
                raise ValueError(
                    "object trajectories and camera calibration must share one robot-base frame"
                )
            if trajectory.timestamps_s != self.robot_trajectory.timestamps_s:
                raise ValueError(
                    "object and robot trajectories must have exactly aligned timestamps"
                )
        if self.output.suffix.lower() != ".mp4":
            raise ValueError("trajectory-conditioned output must be an .mp4 file")


@dataclass(frozen=True)
class TrajectoryConditionedRenderResult:
    output: Path
    experiment_dir: Path
    metadata: Path
    alignment_report: Path


class TrajectoryConditionedVideoRenderer(Protocol):
    """Render only from frame-explicit, physically accepted trajectories."""

    def render(
        self, request: TrajectoryConditionedRenderRequest
    ) -> TrajectoryConditionedRenderResult:
        """Render the verified rollout and persist per-frame alignment evidence."""
