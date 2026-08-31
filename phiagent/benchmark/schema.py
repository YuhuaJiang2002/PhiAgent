"""Versioned, backend-independent schemas for PhiAgent-Bench.

The benchmark deliberately stores camera, world, and robot-base frame names in
the case manifest. Simulator and hardware integrations remain optional adapters;
importing this module never imports PyTorch, CUDA, MuJoCo, or Isaac Sim.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "0.1.0"
DIMENSIONS = ("l1_visual", "l2_geometry", "l3_action", "l4_sim", "l5_real")
TRACKS = ("h2r_transfer", "action_reconstruction", "cross_embodiment", "policy_utility")
TASK_FAMILIES = (
    "f1_rigid_rearrangement",
    "f2_mechanism_actuation",
    "f3_insertion_assembly",
    "f4_deformable_configuration",
    "f5_bulk_material_transfer",
    "f6_surface_transformation",
)


def _identifier(value: object, label: str) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", text):
        raise ValueError(f"{label} must be a non-empty portable identifier")
    return text


def _finite(value: object, label: str, *, minimum: float | None = None) -> float:
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{label} must be finite" + (f" and >= {minimum}" if minimum is not None else ""))
    return number


def _rate(value: object, label: str) -> float:
    number = _finite(value, label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


@dataclass(frozen=True)
class EmbodimentSpec:
    robot_model: str
    end_effector: str
    arm_count: int
    arm_dof: tuple[int, ...]
    urdf_uri: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmbodimentSpec":
        arm_count = int(payload["arm_count"])
        arm_dof = tuple(int(value) for value in payload["arm_dof"])
        if arm_count <= 0 or len(arm_dof) != arm_count or any(value <= 0 for value in arm_dof):
            raise ValueError("arm_dof must contain one positive DOF count per arm")
        uri = payload.get("urdf_uri")
        return cls(
            robot_model=_identifier(payload["robot_model"], "robot_model"),
            end_effector=_identifier(payload["end_effector"], "end_effector"),
            arm_count=arm_count,
            arm_dof=arm_dof,
            urdf_uri=str(uri).strip() if uri is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_model": self.robot_model,
            "end_effector": self.end_effector,
            "arm_count": self.arm_count,
            "arm_dof": list(self.arm_dof),
            "urdf_uri": self.urdf_uri,
        }


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    task_name: str
    task_family: str
    track: str
    split: str
    source_uri: str
    source_interface: str
    target: EmbodimentSpec
    camera_frame: str
    world_frame: str
    robot_base_frame: str
    required_dimensions: tuple[str, ...]
    required_metrics: dict[str, tuple[str, ...]] = field(default_factory=dict)
    annotation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkCase":
        task_family = str(payload["task_family"])
        track = str(payload["track"])
        required = tuple(str(value) for value in payload["required_dimensions"])
        if task_family not in TASK_FAMILIES:
            raise ValueError(f"unsupported task_family: {task_family}")
        if track not in TRACKS:
            raise ValueError(f"unsupported track: {track}")
        if not required or len(set(required)) != len(required) or any(value not in DIMENSIONS for value in required):
            raise ValueError("required_dimensions must be unique PhiAgent-Bench dimension names")
        raw_metrics = _mapping(payload.get("required_metrics", {}), "required_metrics")
        required_metrics = {
            str(dimension): tuple(str(name) for name in names)
            for dimension, names in raw_metrics.items()
        }
        if any(dimension not in {"l2_geometry", "l3_action"} for dimension in required_metrics):
            raise ValueError("required_metrics currently supports only L2 and L3")
        for dimension, names in required_metrics.items():
            if dimension not in required or not names or len(set(names)) != len(names):
                raise ValueError(f"invalid required metric declaration for {dimension}")
        annotation = _mapping(payload.get("annotation", {}), "annotation")
        return cls(
            case_id=_identifier(payload["case_id"], "case_id"),
            task_name=str(payload["task_name"]).strip(),
            task_family=task_family,
            track=track,
            split=_identifier(payload.get("split", "test"), "split"),
            source_uri=str(payload["source_uri"]).strip(),
            source_interface=_identifier(payload["source_interface"], "source_interface"),
            target=EmbodimentSpec.from_dict(_mapping(payload["target"], "target")),
            camera_frame=_identifier(payload["camera_frame"], "camera_frame"),
            world_frame=_identifier(payload["world_frame"], "world_frame"),
            robot_base_frame=_identifier(payload["robot_base_frame"], "robot_base_frame"),
            required_dimensions=required,
            required_metrics=required_metrics,
            annotation=dict(annotation),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "track": self.track,
            "split": self.split,
            "source_uri": self.source_uri,
            "source_interface": self.source_interface,
            "target": self.target.to_dict(),
            "camera_frame": self.camera_frame,
            "world_frame": self.world_frame,
            "robot_base_frame": self.robot_base_frame,
            "required_dimensions": list(self.required_dimensions),
            "required_metrics": {key: list(value) for key, value in self.required_metrics.items()},
            "annotation": self.annotation,
        }


@dataclass(frozen=True)
class VisualEvidence:
    goal_completion: float
    action_completion: float
    contact_transfer: float
    embodiment_correctness: float
    video_quality: float
    judge_count: int
    evidence_frames: int
    protocol: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "goal_completion",
            "action_completion",
            "contact_transfer",
            "embodiment_correctness",
            "video_quality",
        ):
            _rate(getattr(self, name), name)
        if self.judge_count <= 0 or self.evidence_frames <= 0:
            raise ValueError("visual evidence requires positive judge and frame counts")
        if not self.protocol.strip():
            raise ValueError("visual evidence protocol cannot be empty")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VisualEvidence":
        return cls(
            goal_completion=_rate(payload["goal_completion"], "goal_completion"),
            action_completion=_rate(payload["action_completion"], "action_completion"),
            contact_transfer=_rate(payload["contact_transfer"], "contact_transfer"),
            embodiment_correctness=_rate(payload["embodiment_correctness"], "embodiment_correctness"),
            video_quality=_rate(payload["video_quality"], "video_quality"),
            judge_count=int(payload["judge_count"]),
            evidence_frames=int(payload["evidence_frames"]),
            protocol=str(payload["protocol"]),
            diagnostics=dict(_mapping(payload.get("diagnostics", {}), "visual diagnostics")),
        )

    @property
    def h2r_core(self) -> float:
        return 100.0 * (
            0.15 * self.goal_completion
            + 0.15 * self.action_completion
            + 0.30 * self.contact_transfer
            + 0.30 * self.embodiment_correctness
            + 0.10 * self.video_quality
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_completion": self.goal_completion,
            "action_completion": self.action_completion,
            "contact_transfer": self.contact_transfer,
            "embodiment_correctness": self.embodiment_correctness,
            "video_quality": self.video_quality,
            "judge_count": self.judge_count,
            "evidence_frames": self.evidence_frames,
            "protocol": self.protocol,
            "h2r_core": self.h2r_core,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class ScalarEvidence:
    coordinate_frame: str
    values: dict[str, float]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], label: str) -> "ScalarEvidence":
        values = {
            str(name): _finite(value, f"{label}.{name}", minimum=0.0)
            for name, value in _mapping(payload.get("values", {}), f"{label}.values").items()
        }
        if not values:
            raise ValueError(f"{label}.values cannot be empty")
        return cls(
            coordinate_frame=_identifier(payload["coordinate_frame"], f"{label}.coordinate_frame"),
            values=values,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"coordinate_frame": self.coordinate_frame, "values": self.values}


@dataclass(frozen=True)
class SimulationEvidence:
    backend: str
    attempted: bool
    physical_gate_complete: bool
    physically_valid: bool
    task_success: bool
    stage_success_rate: float
    contact_success_rate: float
    ik_success_rate: float
    joint_limit_violation_rate: float
    velocity_violation_rate: float
    collision_rate: float
    singularity_rate: float
    source_revision: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SimulationEvidence":
        return cls(
            backend=_identifier(payload["backend"], "simulation.backend"),
            attempted=_bool(payload["attempted"], "simulation.attempted"),
            physical_gate_complete=_bool(
                payload["physical_gate_complete"], "simulation.physical_gate_complete"
            ),
            physically_valid=_bool(payload["physically_valid"], "simulation.physically_valid"),
            task_success=_bool(payload["task_success"], "simulation.task_success"),
            stage_success_rate=_rate(payload["stage_success_rate"], "stage_success_rate"),
            contact_success_rate=_rate(payload["contact_success_rate"], "contact_success_rate"),
            ik_success_rate=_rate(payload["ik_success_rate"], "ik_success_rate"),
            joint_limit_violation_rate=_rate(payload["joint_limit_violation_rate"], "joint_limit_violation_rate"),
            velocity_violation_rate=_rate(payload["velocity_violation_rate"], "velocity_violation_rate"),
            collision_rate=_rate(payload["collision_rate"], "collision_rate"),
            singularity_rate=_rate(payload["singularity_rate"], "singularity_rate"),
            source_revision=str(payload["source_revision"]).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class RealEvidence:
    adapter: str
    robot_id: str
    hardware_serial: str
    session_id: str
    attempted: bool
    task_success: bool
    stage_success_rate: float
    safety_violation: bool
    human_intervention: bool
    collision_count: int
    emergency_stop: bool
    force_limit_violation: bool
    blind_review: bool

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RealEvidence":
        collision_count = int(payload["collision_count"])
        if collision_count < 0:
            raise ValueError("collision_count cannot be negative")
        return cls(
            adapter=_identifier(payload["adapter"], "real.adapter"),
            robot_id=_identifier(payload["robot_id"], "real.robot_id"),
            hardware_serial=_identifier(payload["hardware_serial"], "real.hardware_serial"),
            session_id=_identifier(payload["session_id"], "real.session_id"),
            attempted=_bool(payload["attempted"], "real.attempted"),
            task_success=_bool(payload["task_success"], "real.task_success"),
            stage_success_rate=_rate(payload["stage_success_rate"], "real.stage_success_rate"),
            safety_violation=_bool(payload["safety_violation"], "real.safety_violation"),
            human_intervention=_bool(payload["human_intervention"], "real.human_intervention"),
            collision_count=collision_count,
            emergency_stop=_bool(payload["emergency_stop"], "real.emergency_stop"),
            force_limit_violation=_bool(payload["force_limit_violation"], "real.force_limit_violation"),
            blind_review=_bool(payload["blind_review"], "real.blind_review"),
        )

    @property
    def valid_success(self) -> bool:
        return (
            self.attempted
            and self.task_success
            and self.blind_review
            and not self.safety_violation
            and not self.human_intervention
            and self.collision_count == 0
            and not self.emergency_stop
            and not self.force_limit_violation
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "valid_success": self.valid_success}


@dataclass(frozen=True)
class RuntimeEvidence:
    generated_video_seconds: float
    latency_seconds: float
    gpu_hours: float
    peak_vram_gb: float
    candidate_count: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeEvidence":
        candidate_count = int(payload["candidate_count"])
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        return cls(
            generated_video_seconds=_finite(payload["generated_video_seconds"], "generated_video_seconds", minimum=0.0),
            latency_seconds=_finite(payload["latency_seconds"], "latency_seconds", minimum=0.0),
            gpu_hours=_finite(payload["gpu_hours"], "gpu_hours", minimum=0.0),
            peak_vram_gb=_finite(payload["peak_vram_gb"], "peak_vram_gb", minimum=0.0),
            candidate_count=candidate_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PolicyUtilityEvidence:
    real_only_success_rate: float
    real_plus_phiagent_success_rate: float
    evaluation_episodes: int
    matched_training_budget: bool

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyUtilityEvidence":
        episodes = int(payload["evaluation_episodes"])
        if episodes <= 0:
            raise ValueError("evaluation_episodes must be positive")
        return cls(
            real_only_success_rate=_rate(payload["real_only_success_rate"], "real_only_success_rate"),
            real_plus_phiagent_success_rate=_rate(payload["real_plus_phiagent_success_rate"], "real_plus_phiagent_success_rate"),
            evaluation_episodes=episodes,
            matched_training_budget=_bool(payload["matched_training_budget"], "matched_training_budget"),
        )

    @property
    def delta_success_rate(self) -> float:
        return self.real_plus_phiagent_success_rate - self.real_only_success_rate

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "delta_success_rate": self.delta_success_rate}


@dataclass(frozen=True)
class SubmissionRecord:
    case_id: str
    generated_uri: str
    visual: VisualEvidence | None = None
    geometry: ScalarEvidence | None = None
    action: ScalarEvidence | None = None
    simulation: SimulationEvidence | None = None
    real: RealEvidence | None = None
    runtime: RuntimeEvidence | None = None
    policy_utility: PolicyUtilityEvidence | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubmissionRecord":
        def optional(key: str) -> Mapping[str, Any] | None:
            return _mapping(payload[key], key) if payload.get(key) is not None else None

        visual = optional("visual")
        geometry = optional("geometry")
        action = optional("action")
        simulation = optional("simulation")
        real = optional("real")
        runtime = optional("runtime")
        utility = optional("policy_utility")
        return cls(
            case_id=_identifier(payload["case_id"], "case_id"),
            generated_uri=str(payload["generated_uri"]).strip(),
            visual=VisualEvidence.from_dict(visual) if visual is not None else None,
            geometry=ScalarEvidence.from_dict(geometry, "geometry") if geometry is not None else None,
            action=ScalarEvidence.from_dict(action, "action") if action is not None else None,
            simulation=SimulationEvidence.from_dict(simulation) if simulation is not None else None,
            real=RealEvidence.from_dict(real) if real is not None else None,
            runtime=RuntimeEvidence.from_dict(runtime) if runtime is not None else None,
            policy_utility=PolicyUtilityEvidence.from_dict(utility) if utility is not None else None,
        )


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    version: str
    cases: tuple[BenchmarkCase, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkSuite":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported suite schema_version: {payload.get('schema_version')}")
        cases = tuple(BenchmarkCase.from_dict(_mapping(item, "case")) for item in payload["cases"])
        identifiers = [case.case_id for case in cases]
        if not cases or len(set(identifiers)) != len(identifiers):
            raise ValueError("suite must contain cases with unique identifiers")
        return cls(
            name=_identifier(payload["name"], "suite name"),
            version=str(payload["version"]),
            cases=cases,
        )

    @classmethod
    def from_json(cls, path: Path) -> "BenchmarkSuite":
        return cls.from_dict(_mapping(json.loads(path.read_text()), "suite"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "version": self.version,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class Submission:
    method: str
    suite_name: str
    records: tuple[SubmissionRecord, ...]
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Submission":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported submission schema_version: {payload.get('schema_version')}")
        records = tuple(SubmissionRecord.from_dict(_mapping(item, "record")) for item in payload["records"])
        identifiers = [record.case_id for record in records]
        if not records or len(set(identifiers)) != len(identifiers):
            raise ValueError("submission records must have unique case identifiers")
        return cls(
            method=_identifier(payload["method"], "method"),
            suite_name=_identifier(payload["suite_name"], "suite_name"),
            records=records,
            metadata=dict(_mapping(payload.get("metadata", {}), "metadata")),
        )

    @classmethod
    def from_json(cls, path: Path) -> "Submission":
        return cls.from_dict(_mapping(json.loads(path.read_text()), "submission"))
