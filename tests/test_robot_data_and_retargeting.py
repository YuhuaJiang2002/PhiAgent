from __future__ import annotations

from pathlib import Path

import pytest

from phiagent.data.schema import CanonicalAction, EmbodimentDescriptor, RobotTrajectory
from phiagent.retargeting.base import LinearEPLRetargeter, LinearRetargetingConfig
from phiagent.retargeting.multi import retarget_multiple
from tests.test_physical_language import _chunk
from phiagent.physical_language.schema import EPLSequence


def _embodiment() -> EmbodimentDescriptor:
    return EmbodimentDescriptor(
        name="test_arm",
        joint_names=("joint1", "joint2"),
        lower_limits_rad=(-1.0, -0.5),
        upper_limits_rad=(1.0, 0.5),
        end_effector_frame="tool",
    )


def test_canonical_action_mask_and_trajectory_round_trip(tmp_path: Path) -> None:
    action = CanonicalAction.from_joint_positions((0.1, -0.2), 5)
    assert action.values == (0.1, -0.2, 0.0, 0.0, 0.0)
    assert action.embodiment_mask == (True, True, False, False, False)
    with pytest.raises(ValueError, match="masked"):
        CanonicalAction((0.1, 1.0), (True, False))

    trajectory = RobotTrajectory(
        "0.1.0", _embodiment(), (0.0, 1.0), ((0.0, 0.0), (0.2, -0.2))
    )
    output = tmp_path / "trajectory.json"
    trajectory.to_json(output)
    assert RobotTrajectory.from_json(output) == trajectory
    assert trajectory.joint_limit_violations() == ()


def test_robot_trajectory_resamples_same_path_at_video_rate() -> None:
    trajectory = RobotTrajectory(
        "0.1.0",
        _embodiment(),
        (0.0, 1.0),
        ((0.0, 0.0), (1.0, -0.5)),
    )
    resampled = trajectory.resample(4)
    assert resampled.timestamps_s == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert resampled.joint_positions_rad[2] == (0.5, -0.25)


def test_trajectory_reports_joint_limit_violations() -> None:
    trajectory = RobotTrajectory(
        "0.1.0", _embodiment(), (0.0,), ((1.1, -0.6),)
    )
    violations = trajectory.joint_limit_violations()
    assert {violation["joint"] for violation in violations} == {"joint1", "joint2"}


def test_linear_retargeter_clamps_and_reports_reachability() -> None:
    chunk = _chunk()
    payload = chunk.to_dict()
    payload["eef_delta"]["translation_m"] = [2.0, 0.0, 0.0]
    sequence = EPLSequence.from_dict(
        {
            "schema_version": "0.1.0",
            "source_video": "human.mp4",
            "chunks": [payload],
        }
    )
    config = LinearRetargetingConfig(
        embodiment=_embodiment(),
        initial_joint_positions_rad=(0.0, 0.0),
        eef_twist_to_joint_delta=(
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.1, 0.0, 0.0, 0.0, 0.0),
        ),
    )
    result = LinearEPLRetargeter(config).retarget(sequence)
    assert result.trajectory.joint_positions_rad[-1] == (1.0, 0.0)
    assert result.reachability_failures[0]["joint"] == "joint1"


def test_one_epl_maps_to_two_masked_embodiments() -> None:
    first = LinearRetargetingConfig(
        embodiment=_embodiment(),
        initial_joint_positions_rad=(0.0, 0.0),
        eef_twist_to_joint_delta=(
            (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.1, 0.0, 0.0, 0.0, 0.0),
        ),
    )
    single = EmbodimentDescriptor(
        "single_joint", ("joint",), (-1.0,), (1.0,), "tool"
    )
    second = LinearRetargetingConfig(
        embodiment=single,
        initial_joint_positions_rad=(0.0,),
        eef_twist_to_joint_delta=((0.2, 0.0, 0.0, 0.0, 0.0, 0.0),),
    )
    sequence = EPLSequence("0.1.0", "human.mp4", (_chunk(),))
    result = retarget_multiple(sequence, (first, second))
    assert result.canonical_dimension == 2
    assert result.canonical_actions["single_joint"][0].embodiment_mask == (
        True,
        False,
    )
