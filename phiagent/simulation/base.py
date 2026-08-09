"""Backend-independent simulation request and result schemas."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from phiagent.data.schema import RigidBodyTrajectory, RobotTrajectory
from phiagent.physical_language.schema import FrameKind, FrameRef, PoseSE3


@dataclass(frozen=True)
class SimulationEvent:
    timestamp_s: float
    event_type: str
    entities: tuple[str, ...]
    details: dict[str, Any]

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("event timestamp must be finite and non-negative")
        if not self.event_type.strip() or not self.entities:
            raise ValueError("simulation event requires a type and at least one entity")


def _contact_pair(value: tuple[str, str], label: str) -> tuple[str, str]:
    if len(value) != 2 or not value[0].strip() or not value[1].strip():
        raise ValueError(f"{label} entries must contain two non-empty geom names")
    if value[0] == value[1]:
        raise ValueError(f"{label} cannot contain a self-pair")
    return tuple(sorted(value))  # type: ignore[return-value]


@dataclass(frozen=True)
class ObjectPositionGoal:
    """A measured terminal body-position goal in the MuJoCo world frame."""

    body_name: str
    target_translation_m: tuple[float, float, float]
    tolerance_m: float

    def __post_init__(self) -> None:
        if not self.body_name.strip():
            raise ValueError("object goal body_name cannot be empty")
        target = tuple(float(value) for value in self.target_translation_m)
        if len(target) != 3 or not all(math.isfinite(value) for value in target):
            raise ValueError("object goal target must be a finite three-vector")
        tolerance = float(self.tolerance_m)
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("object goal tolerance must be finite and positive")
        object.__setattr__(self, "target_translation_m", target)
        object.__setattr__(self, "tolerance_m", tolerance)


@dataclass(frozen=True)
class SimulationRequest:
    model_xml: Path
    trajectory: RobotTrajectory
    object_body_names: tuple[str, ...] = ()
    required_contact_pairs: tuple[tuple[str, str], ...] = ()
    forbidden_contact_pairs: tuple[tuple[str, str], ...] = ()
    object_position_goals: tuple[ObjectPositionGoal, ...] = ()
    render_output: Path | None = None
    render_width: int = 640
    render_height: int = 480
    render_fps: int = 30

    def __post_init__(self) -> None:
        if not self.model_xml.is_file():
            raise ValueError(f"MuJoCo model XML does not exist: {self.model_xml}")
        if min(self.render_width, self.render_height, self.render_fps) <= 0:
            raise ValueError("render dimensions and FPS must be positive")
        if self.render_output is not None and self.render_output.suffix.lower() != ".mp4":
            raise ValueError("render_output must be an .mp4 path")
        required = tuple(
            _contact_pair(pair, "required_contact_pairs")
            for pair in self.required_contact_pairs
        )
        forbidden = tuple(
            _contact_pair(pair, "forbidden_contact_pairs")
            for pair in self.forbidden_contact_pairs
        )
        if len(set(required)) != len(required) or len(set(forbidden)) != len(forbidden):
            raise ValueError("contact-pair requirements cannot contain duplicates")
        overlap = set(required).intersection(forbidden)
        if overlap:
            raise ValueError(f"contact pairs cannot be both required and forbidden: {overlap}")
        object.__setattr__(self, "required_contact_pairs", required)
        object.__setattr__(self, "forbidden_contact_pairs", forbidden)
        goal_names = [goal.body_name for goal in self.object_position_goals]
        if len(set(goal_names)) != len(goal_names):
            raise ValueError("object position goals must have unique body names")


@dataclass(frozen=True)
class SimulationResult:
    backend: str
    physically_valid: bool
    task_success: bool | None
    collision_events: tuple[SimulationEvent, ...]
    contact_events: tuple[SimulationEvent, ...]
    joint_limit_violations: tuple[dict[str, Any], ...]
    reachability_failures: tuple[dict[str, Any], ...]
    slip_events: tuple[SimulationEvent, ...]
    object_pose_trajectories: dict[str, tuple[dict[str, Any], ...]]
    rendered_rollout: str | None
    metrics: dict[str, float | int | bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "physically_valid": self.physically_valid,
            "task_success": self.task_success,
            "collision_events": [asdict(event) for event in self.collision_events],
            "contact_events": [asdict(event) for event in self.contact_events],
            "joint_limit_violations": list(self.joint_limit_violations),
            "reachability_failures": list(self.reachability_failures),
            "slip_events": [asdict(event) for event in self.slip_events],
            "object_pose_trajectories": self.object_pose_trajectories,
            "rendered_rollout": self.rendered_rollout,
            "metrics": self.metrics,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SimulationResult:
        def events(name: str) -> tuple[SimulationEvent, ...]:
            return tuple(
                SimulationEvent(
                    timestamp_s=float(event["timestamp_s"]),
                    event_type=str(event["event_type"]),
                    entities=tuple(event["entities"]),
                    details=dict(event["details"]),
                )
                for event in payload[name]
            )

        return cls(
            backend=str(payload["backend"]),
            physically_valid=bool(payload["physically_valid"]),
            task_success=(
                bool(payload["task_success"])
                if payload.get("task_success") is not None
                else None
            ),
            collision_events=events("collision_events"),
            contact_events=events("contact_events"),
            joint_limit_violations=tuple(payload["joint_limit_violations"]),
            reachability_failures=tuple(payload["reachability_failures"]),
            slip_events=events("slip_events"),
            object_pose_trajectories={
                str(name): tuple(samples)
                for name, samples in payload["object_pose_trajectories"].items()
            },
            rendered_rollout=(
                str(payload["rendered_rollout"])
                if payload.get("rendered_rollout") is not None
                else None
            ),
            metrics=dict(payload["metrics"]),
        )

    @classmethod
    def from_json(cls, path: Path) -> SimulationResult:
        return cls.from_dict(json.loads(path.read_text()))

    def rigid_body_trajectory(
        self,
        body_name: str,
        timestamps_s: tuple[float, ...],
        robot_base_frame: FrameRef,
    ) -> RigidBodyTrajectory:
        """Sample measured simulator poses onto explicit render timestamps."""

        if robot_base_frame.kind is not FrameKind.ROBOT_BASE:
            raise ValueError("simulator world must be declared as a robot-base frame")
        samples = self.object_pose_trajectories.get(body_name)
        if not samples:
            raise ValueError(f"simulation has no measured trajectory for body {body_name!r}")
        poses: list[PoseSE3] = []
        for timestamp in timestamps_s:
            sample = min(
                samples,
                key=lambda item: abs(float(item["timestamp_s"]) - timestamp),
            )
            quaternion_wxyz = tuple(float(value) for value in sample["quaternion_wxyz"])
            poses.append(
                PoseSE3(
                    source_frame=FrameRef(FrameKind.OBJECT, body_name),
                    target_frame=robot_base_frame,
                    translation_m=tuple(float(value) for value in sample["translation_m"]),
                    quaternion_xyzw=(
                        quaternion_wxyz[1],
                        quaternion_wxyz[2],
                        quaternion_wxyz[3],
                        quaternion_wxyz[0],
                    ),
                )
            )
        return RigidBodyTrajectory(
            schema_version="0.1.0",
            body_name=body_name,
            timestamps_s=timestamps_s,
            poses=tuple(poses),
        )


class PhysicsBackend(Protocol):
    def simulate(self, request: SimulationRequest) -> SimulationResult:
        """Replay a trajectory and return measured physical events."""
