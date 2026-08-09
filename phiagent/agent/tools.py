"""Deterministic physics tools exposed to the PhiAgent repair controller."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from phiagent.data.schema import EmbodimentDescriptor, RobotTrajectory
from phiagent.simulation.base import PhysicsBackend, SimulationRequest, SimulationResult


class PhysicsTools:
    """A small explicit tool surface; all checks are grounded in simulator output."""

    def __init__(
        self,
        backend: PhysicsBackend,
        request_factory: Callable[[RobotTrajectory], SimulationRequest],
    ) -> None:
        self.backend = backend
        self.request_factory = request_factory

    @staticmethod
    def get_robot_kinematics(embodiment: EmbodimentDescriptor) -> dict[str, object]:
        return {
            "name": embodiment.name,
            "dof": embodiment.dof,
            "joint_names": list(embodiment.joint_names),
            "lower_limits_rad": list(embodiment.lower_limits_rad),
            "upper_limits_rad": list(embodiment.upper_limits_rad),
            "end_effector_frame": embodiment.end_effector_frame,
            "urdf_path": embodiment.urdf_path,
        }

    def simulate_trajectory(self, trajectory: RobotTrajectory) -> SimulationResult:
        return self.backend.simulate(self.request_factory(trajectory))

    @staticmethod
    def check_collision(result: SimulationResult) -> dict[str, object]:
        return {
            "passed": not result.collision_events,
            "events": [asdict(event) for event in result.collision_events],
        }

    @staticmethod
    def check_contact(result: SimulationResult) -> dict[str, object]:
        return {
            "task_success": result.task_success,
            "events": [asdict(event) for event in result.contact_events],
            "required_contacts_met": result.metrics.get("required_contacts_met"),
        }

    @staticmethod
    def check_reachability(result: SimulationResult) -> dict[str, object]:
        return {
            "passed": not result.reachability_failures
            and not result.joint_limit_violations,
            "reachability_failures": list(result.reachability_failures),
            "joint_limit_violations": list(result.joint_limit_violations),
        }

    def render_simulation(self, trajectory: RobotTrajectory) -> SimulationResult:
        request = self.request_factory(trajectory)
        if request.render_output is None:
            raise ValueError("render_simulation requires request_factory to set render_output")
        return self.backend.simulate(request)
