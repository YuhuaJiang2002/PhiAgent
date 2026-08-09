from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from phiagent.agent.verifier import VerificationReport
from phiagent.data.schema import (
    EmbodimentDescriptor,
    RigidBodyTrajectory,
    RobotTrajectory,
)
from phiagent.perception.camera import PinholeIntrinsics
from phiagent.physical_language.schema import FrameKind, FrameRef, PoseSE3
from phiagent.rendering.base import TrajectoryConditionedRenderRequest
from phiagent.simulation.base import SimulationResult


def _robot_trajectory() -> RobotTrajectory:
    return RobotTrajectory(
        schema_version="0.1.0",
        embodiment=EmbodimentDescriptor(
            name="test-arm",
            joint_names=("joint",),
            lower_limits_rad=(-1.0,),
            upper_limits_rad=(1.0,),
            end_effector_frame="eef",
        ),
        timestamps_s=(0.0, 0.1),
        joint_positions_rad=((0.0,), (0.2,)),
    )


def _object_trajectory(
    target_frame: FrameRef | None = None,
    timestamps_s: tuple[float, ...] = (0.0, 0.1),
) -> RigidBodyTrajectory:
    object_frame = FrameRef(FrameKind.OBJECT, "block")
    target = target_frame or FrameRef(FrameKind.ROBOT_BASE, "robot")
    return RigidBodyTrajectory(
        schema_version="0.1.0",
        body_name="block",
        timestamps_s=timestamps_s,
        poses=tuple(
            PoseSE3(
                source_frame=object_frame,
                target_frame=target,
                translation_m=(timestamp, 0.0, 0.2),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
            for timestamp in timestamps_s
        ),
    )


def _request(tmp_path: Path) -> TrajectoryConditionedRenderRequest:
    robot_base = FrameRef(FrameKind.ROBOT_BASE, "robot")
    camera = FrameRef(FrameKind.CAMERA, "main")
    return TrajectoryConditionedRenderRequest(
        robot_trajectory=_robot_trajectory(),
        object_trajectories=(_object_trajectory(robot_base),),
        control_video=tmp_path / "control.mp4",
        prompt="A robot transfers a block between grippers.",
        camera_intrinsics=PinholeIntrinsics(500.0, 500.0, 320.0, 240.0, 640, 480),
        camera_T_robot_base=PoseSE3(
            source_frame=robot_base,
            target_frame=camera,
            translation_m=(0.0, 0.0, 1.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        scene_assets=(tmp_path / "scene.usd",),
        verification=VerificationReport(
            accepted=True,
            collision={"passed": True},
            contact={"passed": True},
            reachability={"passed": True},
            diagnoses=(),
        ),
        verification_record=tmp_path / "verification.json",
        output=tmp_path / "render.mp4",
        experiment_root=tmp_path / "experiments",
    )


def test_rigid_body_trajectory_json_round_trip(tmp_path: Path) -> None:
    trajectory = _object_trajectory()
    path = tmp_path / "object-trajectory.json"
    trajectory.to_json(path)
    assert RigidBodyTrajectory.from_json(path) == trajectory


def test_rigid_body_trajectory_rejects_mixed_frames() -> None:
    trajectory = _object_trajectory()
    world = FrameRef(FrameKind.WORLD, "scene")
    poses = trajectory.poses[:-1] + (replace(trajectory.poses[-1], target_frame=world),)
    with pytest.raises(ValueError, match="same source and target frames"):
        replace(trajectory, poses=poses)


def test_render_request_accepts_aligned_verified_trajectories(tmp_path: Path) -> None:
    request = _request(tmp_path)
    assert request.camera_T_robot_base.source_frame == request.object_trajectories[0].poses[
        0
    ].target_frame


def test_render_request_rejects_unverified_rollout(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValueError, match="accepted verification"):
        replace(request, verification=replace(request.verification, accepted=False))


def test_render_request_rejects_missing_prompt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty prompt"):
        replace(_request(tmp_path), prompt=" ")


def test_render_request_rejects_frame_or_timestamp_mismatch(tmp_path: Path) -> None:
    request = _request(tmp_path)
    other_base = FrameRef(FrameKind.ROBOT_BASE, "other")
    with pytest.raises(ValueError, match="share one robot-base frame"):
        replace(request, object_trajectories=(_object_trajectory(other_base),))
    with pytest.raises(ValueError, match="exactly aligned timestamps"):
        replace(
            request,
            object_trajectories=(_object_trajectory(timestamps_s=(0.0, 0.2)),),
        )


def test_simulation_result_samples_frame_explicit_object_trajectory() -> None:
    result = SimulationResult(
        backend="test",
        physically_valid=True,
        task_success=True,
        collision_events=(),
        contact_events=(),
        joint_limit_violations=(),
        reachability_failures=(),
        slip_events=(),
        object_pose_trajectories={
            "block": (
                {
                    "timestamp_s": 0.0,
                    "translation_m": [0.0, 0.0, 0.2],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                {
                    "timestamp_s": 0.1,
                    "translation_m": [0.1, 0.0, 0.2],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
            )
        },
        rendered_rollout=None,
        metrics={},
    )
    trajectory = result.rigid_body_trajectory(
        "block",
        (0.0, 0.1),
        FrameRef(FrameKind.ROBOT_BASE, "robot"),
    )
    assert trajectory.poses[1].translation_m == (0.1, 0.0, 0.2)
    assert trajectory.poses[1].quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)
