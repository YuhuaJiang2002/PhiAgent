from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from phiagent.data.schema import RobotTrajectory
from phiagent.simulation.base import ObjectPositionGoal, SimulationRequest
from phiagent.simulation.mujoco_backend import MujocoBackend


def test_mujoco_replays_named_actuated_trajectory(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    result = MujocoBackend().simulate(
        SimulationRequest(
            model_xml=root / "demo" / "tabletop_hinge.xml",
            trajectory=RobotTrajectory.from_json(
                root / "demo" / "tabletop_hinge_trajectory.json"
            ),
            object_body_names=("object",),
            render_output=None,
        )
    )
    assert result.physically_valid
    assert result.reachability_failures == ()
    assert result.metrics["simulated_steps"] > 0
    assert len(result.object_pose_trajectories["object"]) > 1
    assert result.collision_events == ()


def test_simulation_task_criteria_are_measured(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    trajectory = RobotTrajectory.from_json(
        root / "demo" / "tabletop_hinge_trajectory.json"
    )
    result = MujocoBackend().simulate(
        SimulationRequest(
            model_xml=root / "demo" / "tabletop_hinge.xml",
            trajectory=trajectory,
            object_body_names=("object",),
            required_contact_pairs=(("table", "object_geom"),),
            object_position_goals=(
                ObjectPositionGoal("object", (0.45, 0.0, 0.04), 0.03),
            ),
        )
    )
    assert result.physically_valid
    assert result.task_success
    assert result.metrics["required_contacts_met"]


def test_tabletop_push_survives_physics_and_moves_object() -> None:
    root = Path(__file__).resolve().parents[1]
    result = MujocoBackend().simulate(
        SimulationRequest(
            model_xml=root / "demo" / "tabletop_push.xml",
            trajectory=RobotTrajectory.from_json(
                root / "demo" / "tabletop_push_trajectory.json"
            ),
            object_body_names=("object",),
            required_contact_pairs=(("arm_link", "object_geom"),),
            object_position_goals=(
                ObjectPositionGoal("object", (0.31, 0.24, 0.04), 0.08),
            ),
        )
    )
    poses = result.object_pose_trajectories["object"]
    displacement = math.dist(
        poses[0]["translation_m"], poses[-1]["translation_m"]
    )
    assert result.physically_valid
    assert result.task_success
    assert displacement > 0.15
    assert result.metrics["object_goal_max_error_m"] < 0.08
