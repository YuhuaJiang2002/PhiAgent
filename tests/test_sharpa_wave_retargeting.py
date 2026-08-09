from __future__ import annotations

from pathlib import Path

import pytest

from phiagent.data.schema import EmbodimentDescriptor
from phiagent.perception.schema import HandObservation, PerceptionSequence
from phiagent.physical_language.schema import FrameKind, FrameRef, Point3D, PoseSE3
from phiagent.retargeting.sharpa_wave import (
    SHARPA_WAVE_JOINT_SUFFIXES,
    SharpaWaveRetargeter,
    load_sharpa_wave_embodiment,
)


def _embodiment() -> EmbodimentDescriptor:
    return EmbodimentDescriptor(
        name="sharpa_wave_right_22dof",
        joint_names=tuple(f"right_{name}" for name in SHARPA_WAVE_JOINT_SUFFIXES),
        lower_limits_rad=(-0.5,) * 22,
        upper_limits_rad=(2.0,) * 22,
        end_effector_frame="right_hand_C_MC",
    )


def _hand(timestamp_s: float, curl_z: float) -> HandObservation:
    camera = FrameRef(FrameKind.CAMERA, "front")
    wrist_frame = FrameRef(FrameKind.HUMAN_WRIST, "right")
    points = [
        (0.000, 0.000, 0.000),
        (0.030, 0.018, 0.000),
        (0.050, 0.038, curl_z * 0.3),
        (0.065, 0.054, curl_z * 0.7),
        (0.078, 0.068, curl_z),
        (0.035, 0.045, 0.000),
        (0.036, 0.075, curl_z * 0.3),
        (0.037, 0.100, curl_z * 0.7),
        (0.038, 0.120, curl_z),
        (0.000, 0.050, 0.000),
        (0.000, 0.085, curl_z * 0.3),
        (0.000, 0.113, curl_z * 0.7),
        (0.000, 0.135, curl_z),
        (-0.030, 0.045, 0.000),
        (-0.031, 0.077, curl_z * 0.3),
        (-0.032, 0.102, curl_z * 0.7),
        (-0.033, 0.122, curl_z),
        (-0.052, 0.036, 0.000),
        (-0.053, 0.064, curl_z * 0.3),
        (-0.054, 0.086, curl_z * 0.7),
        (-0.055, 0.103, curl_z),
    ]
    return HandObservation(
        timestamp_s=timestamp_s,
        wrist_pose=PoseSE3(
            wrist_frame, camera, points[0], (0.0, 0.0, 0.0, 1.0)
        ),
        keypoints_3d=tuple(Point3D(camera, point) for point in points),
        articulation=(),
        confidence=1.0,
    )


def test_sharpa_wave_retargeting_produces_bounded_22_dof_motion() -> None:
    observations = PerceptionSequence(
        "0.1.0", (_hand(0.0, 0.0), _hand(0.1, 0.025)), (None, None)
    )
    result = SharpaWaveRetargeter(_embodiment()).retarget(observations)
    assert result.trajectory.embodiment.dof == 22
    assert not result.trajectory.joint_limit_violations()
    assert (
        result.trajectory.joint_positions_rad[1][5]
        > result.trajectory.joint_positions_rad[0][5]
    )


def test_sharpa_wave_rejects_wrong_handedness() -> None:
    hand = _hand(0.0, 0.0)
    left_wrist = FrameRef(FrameKind.HUMAN_WRIST, "left")
    wrong = HandObservation(
        hand.timestamp_s,
        PoseSE3(
            left_wrist,
            hand.wrist_pose.target_frame,
            hand.wrist_pose.translation_m,
            hand.wrist_pose.quaternion_xyzw,
        ),
        hand.keypoints_3d,
        hand.articulation,
        hand.confidence,
    )
    with pytest.raises(ValueError, match="left hand observations"):
        SharpaWaveRetargeter(_embodiment()).retarget(
            PerceptionSequence("0.1.0", (wrong, _hand(0.1, 0.0)), (None, None))
        )


def test_load_sharpa_wave_embodiment_reads_model_ranges(tmp_path: Path) -> None:
    joints = "\n".join(
        f'<joint name="right_{suffix}" range="-0.1 1.1"/>'
        for suffix in SHARPA_WAVE_JOINT_SUFFIXES
    )
    model = tmp_path / "sharpa.xml"
    model.write_text(f"<mujoco><worldbody><body>{joints}</body></worldbody></mujoco>")
    embodiment = load_sharpa_wave_embodiment(model)
    assert embodiment.dof == 22
    assert embodiment.lower_limits_rad == (-0.1,) * 22
    assert embodiment.upper_limits_rad == (1.1,) * 22
