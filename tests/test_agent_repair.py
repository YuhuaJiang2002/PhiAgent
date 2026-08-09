from __future__ import annotations

from pathlib import Path

from phiagent.agent.repair import AgentAction, TrajectoryRepairController
from phiagent.agent.tools import PhysicsTools
from phiagent.agent.verifier import AgentVerifier
from phiagent.data.corruptions import CorruptionType, corrupt_trajectory
from phiagent.data.repair_dataset import (
    RepairExample,
    read_repair_jsonl,
    write_repair_jsonl,
)
from phiagent.data.schema import EmbodimentDescriptor, RobotTrajectory
from phiagent.simulation.base import SimulationRequest, SimulationResult


class LimitCheckingBackend:
    def simulate(self, request: SimulationRequest) -> SimulationResult:
        violations = request.trajectory.joint_limit_violations()
        return SimulationResult(
            backend="test",
            physically_valid=not violations,
            task_success=None,
            collision_events=(),
            contact_events=(),
            joint_limit_violations=violations,
            reachability_failures=(),
            slip_events=(),
            object_pose_trajectories={},
            rendered_rollout=None,
            metrics={"violations": len(violations)},
        )


def _trajectories() -> tuple[RobotTrajectory, RobotTrajectory]:
    embodiment = EmbodimentDescriptor(
        "test",
        ("joint",),
        (-1.0,),
        (1.0,),
        "tool",
    )
    bad = RobotTrajectory("0.1.0", embodiment, (0.0, 1.0), ((0.0,), (2.0,)))
    good = RobotTrajectory("0.1.0", embodiment, (0.0, 1.0), ((0.0,), (1.0,)))
    return bad, good


def test_agent_detects_and_repairs_corrupted_trajectory(tmp_path: Path) -> None:
    bad, good = _trajectories()
    model = tmp_path / "model.xml"
    model.write_text("<mujoco/>")
    controller = TrajectoryRepairController(LimitCheckingBackend())
    outcome = controller.run(
        bad,
        lambda trajectory: SimulationRequest(model_xml=model, trajectory=trajectory),
    )
    assert outcome.accepted
    assert outcome.final_trajectory == good
    assert [step.action for step in outcome.trace] == [
        AgentAction.SIMULATE,
        AgentAction.VERIFY_COLLISION,
        AgentAction.VERIFY_CONTACT,
        AgentAction.VERIFY_REACHABILITY,
        AgentAction.REPLAN,
        AgentAction.SIMULATE,
        AgentAction.VERIFY_COLLISION,
        AgentAction.VERIFY_CONTACT,
        AgentAction.VERIFY_REACHABILITY,
        AgentAction.ACCEPT,
    ]
    trace = tmp_path / "trace.json"
    outcome.to_json(trace)
    assert trace.is_file()


def test_repair_dataset_jsonl_round_trip(tmp_path: Path) -> None:
    bad, good = _trajectories()
    example = RepairExample(
        "0.1.0",
        "joint-limit-0001",
        "epl://demo/0001",
        "joint_limit",
        bad,
        {"joint_limit_violations": 1},
        good,
    )
    path = tmp_path / "repair.jsonl"
    assert write_repair_jsonl([example], path) == 1
    assert read_repair_jsonl(path) == (example,)


def test_corruption_tools_and_verifier_are_simulator_grounded(tmp_path: Path) -> None:
    _, good = _trajectories()
    bad = corrupt_trajectory(good, CorruptionType.JOINT_LIMIT)
    assert bad.joint_limit_violations()
    model = tmp_path / "model.xml"
    model.write_text("<mujoco/>")
    tools = PhysicsTools(
        LimitCheckingBackend(),
        lambda trajectory: SimulationRequest(model_xml=model, trajectory=trajectory),
    )
    result = tools.simulate_trajectory(bad)
    report = AgentVerifier().verify(result)
    assert not report.accepted
    assert not report.reachability["passed"]
    assert tools.get_robot_kinematics(good.embodiment)["dof"] == 1
