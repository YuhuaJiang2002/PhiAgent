"""Explicit simulation-tool loop for deterministic first-generation repair."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from phiagent.agent.verifier import AgentVerifier
from phiagent.data.schema import RobotTrajectory
from phiagent.simulation.base import PhysicsBackend, SimulationRequest, SimulationResult


class AgentAction(str, Enum):
    SIMULATE = "<SIMULATE>"
    VERIFY_CONTACT = "<VERIFY_CONTACT>"
    VERIFY_COLLISION = "<VERIFY_COLLISION>"
    VERIFY_REACHABILITY = "<VERIFY_REACHABILITY>"
    REPLAN = "<REPLAN>"
    RENDER = "<RENDER>"
    ACCEPT = "<ACCEPT>"


@dataclass(frozen=True)
class RepairTraceStep:
    iteration: int
    action: AgentAction
    diagnosis: str
    metrics: dict[str, float | int | bool]
    trajectory_changed: bool


@dataclass(frozen=True)
class RepairOutcome:
    accepted: bool
    original_trajectory: RobotTrajectory
    final_trajectory: RobotTrajectory
    final_simulation: SimulationResult
    trace: tuple[RepairTraceStep, ...]

    def to_json(self, path: Path) -> None:
        payload = {
            "accepted": self.accepted,
            "original_trajectory": self.original_trajectory.to_dict(),
            "final_trajectory": self.final_trajectory.to_dict(),
            "final_simulation": self.final_simulation.to_dict(),
            "trace": [
                {
                    **asdict(step),
                    "action": step.action.value,
                }
                for step in self.trace
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def clamp_joint_limits(trajectory: RobotTrajectory) -> RobotTrajectory:
    """Return a new trajectory clamped to its declared embodiment limits."""

    embodiment = trajectory.embodiment
    samples = tuple(
        tuple(
            min(max(value, low), high)
            for value, low, high in zip(
                sample,
                embodiment.lower_limits_rad,
                embodiment.upper_limits_rad,
            )
        )
        for sample in trajectory.joint_positions_rad
    )
    return RobotTrajectory(
        schema_version=trajectory.schema_version,
        embodiment=embodiment,
        timestamps_s=trajectory.timestamps_s,
        joint_positions_rad=samples,
        source_epl=trajectory.source_epl,
    )


class TrajectoryRepairController:
    """Tool loop that currently repairs declared joint-limit violations."""

    def __init__(self, backend: PhysicsBackend, maximum_iterations: int = 3) -> None:
        if maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        self.backend = backend
        self.maximum_iterations = maximum_iterations
        self.verifier = AgentVerifier()

    def run(
        self,
        trajectory: RobotTrajectory,
        request_factory: Callable[[RobotTrajectory], SimulationRequest],
    ) -> RepairOutcome:
        current = trajectory
        trace: list[RepairTraceStep] = []
        final: SimulationResult | None = None
        for iteration in range(self.maximum_iterations):
            trace.append(
                RepairTraceStep(
                    iteration=iteration,
                    action=AgentAction.SIMULATE,
                    diagnosis="simulate current trajectory",
                    metrics={},
                    trajectory_changed=False,
                )
            )
            final = self.backend.simulate(request_factory(current))
            report = self.verifier.verify(final)
            trace.extend(
                [
                    RepairTraceStep(
                        iteration=iteration,
                        action=AgentAction.VERIFY_COLLISION,
                        diagnosis=(
                            "no forbidden collision detected"
                            if report.collision["passed"]
                            else "forbidden collision detected"
                        ),
                        metrics={"events": len(final.collision_events)},
                        trajectory_changed=False,
                    ),
                    RepairTraceStep(
                        iteration=iteration,
                        action=AgentAction.VERIFY_CONTACT,
                        diagnosis=(
                            "task contact criteria satisfied or unspecified"
                            if final.task_success is not False
                            else "task contact/object criteria not satisfied"
                        ),
                        metrics={"events": len(final.contact_events)},
                        trajectory_changed=False,
                    ),
                    RepairTraceStep(
                        iteration=iteration,
                        action=AgentAction.VERIFY_REACHABILITY,
                        diagnosis=(
                            "joint limits and reachability checks passed"
                            if report.reachability["passed"]
                            else "joint-limit or reachability failure detected"
                        ),
                        metrics={
                            "joint_limit_violations": len(
                                final.joint_limit_violations
                            ),
                            "reachability_failures": len(
                                final.reachability_failures
                            ),
                        },
                        trajectory_changed=False,
                    ),
                ]
            )
            if report.accepted:
                trace.append(
                    RepairTraceStep(
                        iteration=iteration,
                        action=AgentAction.ACCEPT,
                        diagnosis="trajectory passed backend physical-validity checks",
                        metrics=final.metrics,
                        trajectory_changed=False,
                    )
                )
                return RepairOutcome(True, trajectory, current, final, tuple(trace))
            if final.joint_limit_violations:
                repaired = clamp_joint_limits(current)
                changed = repaired != current
                trace.append(
                    RepairTraceStep(
                        iteration=iteration,
                        action=AgentAction.REPLAN,
                        diagnosis=(
                            f"clamped {len(final.joint_limit_violations)} "
                            "declared joint-limit violations"
                        ),
                        metrics={"violations": len(final.joint_limit_violations)},
                        trajectory_changed=changed,
                    )
                )
                if not changed:
                    break
                current = repaired
                continue
            trace.append(
                RepairTraceStep(
                    iteration=iteration,
                    action=AgentAction.VERIFY_REACHABILITY,
                    diagnosis="no implemented safe repair for this backend failure",
                    metrics=final.metrics,
                    trajectory_changed=False,
                )
            )
            break
        assert final is not None
        return RepairOutcome(False, trajectory, current, final, tuple(trace))
