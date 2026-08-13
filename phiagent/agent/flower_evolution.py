"""Constraint-first self-evolution policy for flower-arranging video transfer.

The policy is intentionally dependency-free.  Renderers and evaluators persist
evidence, while this module decides whether a candidate may be accepted and
which *pipeline family* should be tried next.  Unknown evidence is never treated
as a pass, and an aggregate score can never hide a failed contact or morphology
gate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Mapping, Sequence


class GateVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class PipelineFamily(str, Enum):
    """Increasingly constrained representations available to the planner."""

    FULL_FRAME_GENERATIVE = "full_frame_generative"
    LAYERED_2D = "layered_2d"
    HYBRID_3D_LAYERED = "hybrid_3d_layered"
    ROBOT_CENTRIC_ADAPTED = "robot_centric_adapted"


GATE_NAMES = (
    "background_lock",
    "human_residual_absence",
    "robot_morphology",
    "robot_identity",
    "embodied_motion",
    "flower_integrity",
    "hand_flower_contact",
    "occlusion_order",
    "temporal_consistency",
    "full_video_human_preference",
)

GEOMETRY_GATES = frozenset(
    {
        "robot_morphology",
        "robot_identity",
        "embodied_motion",
        "hand_flower_contact",
        "occlusion_order",
    }
)


@dataclass(frozen=True)
class GateRequirement:
    name: str
    threshold: float

    def __post_init__(self) -> None:
        if self.name not in GATE_NAMES:
            raise ValueError(f"unknown flower acceptance gate: {self.name}")
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("gate threshold must be finite and in [0, 1]")


@dataclass(frozen=True)
class FlowerAcceptanceContract:
    """Hard requirements; there is deliberately no weighted mean."""

    requirements: tuple[GateRequirement, ...]
    coordinate_frame: str = "camera:source_pixels"

    def __post_init__(self) -> None:
        names = tuple(requirement.name for requirement in self.requirements)
        if len(set(names)) != len(names):
            raise ValueError("acceptance contract contains duplicate gates")
        if set(names) != set(GATE_NAMES):
            missing = sorted(set(GATE_NAMES) - set(names))
            extra = sorted(set(names) - set(GATE_NAMES))
            raise ValueError(f"acceptance contract mismatch; missing={missing}, extra={extra}")
        if self.coordinate_frame != "camera:source_pixels":
            raise ValueError("video evidence must be measured in camera:source_pixels")

    @classmethod
    def strict(cls) -> "FlowerAcceptanceContract":
        return cls(
            requirements=(
                GateRequirement("background_lock", 0.999),
                GateRequirement("human_residual_absence", 0.99),
                GateRequirement("robot_morphology", 0.90),
                GateRequirement("robot_identity", 0.85),
                GateRequirement("embodied_motion", 0.80),
                GateRequirement("flower_integrity", 0.95),
                GateRequirement("hand_flower_contact", 0.85),
                GateRequirement("occlusion_order", 0.90),
                GateRequirement("temporal_consistency", 0.85),
                GateRequirement("full_video_human_preference", 1.0),
            )
        )

    @property
    def thresholds(self) -> dict[str, float]:
        return {requirement.name: requirement.threshold for requirement in self.requirements}

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinate_frame": self.coordinate_frame,
            "requirements": [asdict(requirement) for requirement in self.requirements],
        }


@dataclass(frozen=True)
class GateMeasurement:
    name: str
    verdict: GateVerdict
    score: float | None
    evidence: tuple[str, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        if self.name not in GATE_NAMES:
            raise ValueError(f"unknown flower measurement gate: {self.name}")
        if self.score is not None and (
            not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0
        ):
            raise ValueError("gate score must be finite and in [0, 1]")
        if self.verdict is not GateVerdict.UNKNOWN and not self.evidence:
            raise ValueError("PASS and FAIL measurements require evidence")
        if self.verdict is GateVerdict.PASS and self.score is None:
            raise ValueError("PASS measurements require a measured score")
        if any(not item.strip() for item in self.evidence):
            raise ValueError("gate evidence must not contain empty entries")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GateMeasurement":
        score = payload.get("score")
        raw_evidence = payload.get("evidence", ())
        if not isinstance(raw_evidence, (list, tuple)):
            raise ValueError("gate evidence must be an array")
        return cls(
            name=str(payload["name"]),
            verdict=GateVerdict(str(payload["verdict"]).upper()),
            score=None if score is None else float(score),
            evidence=tuple(str(item) for item in raw_evidence),
            notes=str(payload.get("notes", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "verdict": self.verdict.value,
            "score": self.score,
            "evidence": list(self.evidence),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class FlowerCandidateEvaluation:
    candidate_id: str
    measurements: tuple[GateMeasurement, ...]
    evaluated_frames: int
    expected_frames: int
    coordinate_frame: str = "camera:source_pixels"

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id is required")
        if self.evaluated_frames < 1 or self.expected_frames < 1:
            raise ValueError("frame counts must be positive")
        if self.evaluated_frames > self.expected_frames:
            raise ValueError("evaluated_frames cannot exceed expected_frames")
        if self.coordinate_frame != "camera:source_pixels":
            raise ValueError("candidate evidence must use camera:source_pixels")
        names = tuple(measurement.name for measurement in self.measurements)
        if len(set(names)) != len(names):
            raise ValueError("candidate evaluation contains duplicate gates")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FlowerCandidateEvaluation":
        raw = payload.get("measurements")
        if not isinstance(raw, list):
            raise ValueError("candidate measurements must be an array")
        return cls(
            candidate_id=str(payload["candidate_id"]),
            measurements=tuple(GateMeasurement.from_dict(item) for item in raw),
            evaluated_frames=int(payload["evaluated_frames"]),
            expected_frames=int(payload["expected_frames"]),
            coordinate_frame=str(payload.get("coordinate_frame", "camera:source_pixels")),
        )

    @property
    def by_name(self) -> dict[str, GateMeasurement]:
        return {measurement.name: measurement for measurement in self.measurements}

    def failed_gates(self, contract: FlowerAcceptanceContract) -> tuple[str, ...]:
        thresholds = contract.thresholds
        failed = []
        for name in GATE_NAMES:
            measurement = self.by_name.get(name)
            if measurement is None or measurement.verdict is not GateVerdict.PASS:
                failed.append(name)
            elif measurement.score is None or measurement.score < thresholds[name]:
                failed.append(name)
        if self.evaluated_frames != self.expected_frames:
            failed.append("full_frame_coverage")
        return tuple(failed)

    def accepted(self, contract: FlowerAcceptanceContract) -> bool:
        return not self.failed_gates(contract)

    def pass_count(self, contract: FlowerAcceptanceContract) -> int:
        return len(GATE_NAMES) + 1 - len(self.failed_gates(contract))

    def worst_margin(self, contract: FlowerAcceptanceContract) -> float:
        margins = []
        thresholds = contract.thresholds
        for name in GATE_NAMES:
            measurement = self.by_name.get(name)
            if measurement is None or measurement.score is None:
                margins.append(-1.0)
            else:
                margins.append(measurement.score - thresholds[name])
        if self.evaluated_frames != self.expected_frames:
            margins.append(-1.0)
        return min(margins)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "coordinate_frame": self.coordinate_frame,
            "evaluated_frames": self.evaluated_frames,
            "expected_frames": self.expected_frames,
            "measurements": [measurement.to_dict() for measurement in self.measurements],
        }


@dataclass(frozen=True)
class FlowerPipelineConfig:
    family: PipelineFamily
    action_segmentation: bool = False
    explicit_robot_geometry: bool = False
    robot_native_motion: bool = False
    contact_conditioning: bool = False
    depth_layering: bool = False
    local_generation_only: bool = False
    paired_task_adapter: bool = False
    source_frame: str = "camera:source_pixels"
    robot_frame: str = "robot:base"
    flower_frame: str = "object:flower"

    def __post_init__(self) -> None:
        if self.source_frame != "camera:source_pixels":
            raise ValueError("source geometry must be named camera:source_pixels")
        if self.robot_frame != "robot:base":
            raise ValueError("robot geometry must be named robot:base")
        if self.flower_frame != "object:flower":
            raise ValueError("flower geometry must be named object:flower")
        if self.family is PipelineFamily.ROBOT_CENTRIC_ADAPTED and not (
            self.explicit_robot_geometry and self.robot_native_motion and self.paired_task_adapter
        ):
            raise ValueError("robot-centric adapted pipeline requires geometry, native motion, and adapter")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FlowerPipelineConfig":
        fields = {
            "action_segmentation",
            "explicit_robot_geometry",
            "robot_native_motion",
            "contact_conditioning",
            "depth_layering",
            "local_generation_only",
            "paired_task_adapter",
            "source_frame",
            "robot_frame",
            "flower_frame",
        }
        values = {name: payload[name] for name in fields if name in payload}
        return cls(family=PipelineFamily(str(payload["family"])), **values)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["family"] = self.family.value
        return payload


@dataclass(frozen=True)
class FlowerEvolutionDecision:
    status: str
    failed_gates: tuple[str, ...]
    next_config: FlowerPipelineConfig
    actions: tuple[str, ...]
    required_evidence: tuple[str, ...]
    training_required: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "failed_gates": list(self.failed_gates),
            "next_config": self.next_config.to_dict(),
            "actions": list(self.actions),
            "required_evidence": list(self.required_evidence),
            "training_required": self.training_required,
        }


class FlowerEvolutionAgent:
    """Reflective pipeline planner with hard semantic vetoes."""

    def propose(
        self,
        current: FlowerPipelineConfig,
        evaluation: FlowerCandidateEvaluation,
        contract: FlowerAcceptanceContract,
        failure_counts: Mapping[str, int] | None = None,
    ) -> FlowerEvolutionDecision:
        failures = evaluation.failed_gates(contract)
        if not failures:
            return FlowerEvolutionDecision(
                status="WORKING",
                failed_gates=(),
                next_config=current,
                actions=("freeze the immutable candidate; do not apply post-acceptance repair",),
                required_evidence=("persist the complete scorecard and blind preference record",),
            )

        counts = dict(failure_counts or {})
        next_config = current
        actions: list[str] = []
        evidence: list[str] = []
        training_required = False
        semantic_failures = set(failures) & GEOMETRY_GATES

        if semantic_failures:
            repeated = any(counts.get(name, 0) >= 2 for name in semantic_failures)
            if current.family in {
                PipelineFamily.FULL_FRAME_GENERATIVE,
                PipelineFamily.LAYERED_2D,
            }:
                next_config = replace(
                    current,
                    family=PipelineFamily.HYBRID_3D_LAYERED,
                    action_segmentation=True,
                    explicit_robot_geometry=True,
                    robot_native_motion=True,
                    contact_conditioning=True,
                    depth_layering=True,
                    local_generation_only=True,
                )
                actions.extend(
                    (
                        "retarget source joints and end effectors to an explicit articulated robot",
                        "localize approach/grasp/manipulate/release and enforce hand-flower contact",
                        "render robot, flowers, and background as depth-ordered layers",
                        "limit generative repair to residual appearance inside the robot mask",
                    )
                )
            elif current.family is PipelineFamily.HYBRID_3D_LAYERED and repeated:
                next_config = replace(
                    current,
                    family=PipelineFamily.ROBOT_CENTRIC_ADAPTED,
                    action_segmentation=True,
                    explicit_robot_geometry=True,
                    robot_native_motion=True,
                    contact_conditioning=True,
                    depth_layering=True,
                    local_generation_only=True,
                    paired_task_adapter=True,
                )
                training_required = True
                actions.extend(
                    (
                        "build paired synthetic human/humanoid flower-arranging clips",
                        "train a task-specific video-to-video adapter on held-out action phases",
                        "retain explicit geometry and contact controls during adapted generation",
                    )
                )
            else:
                actions.extend(
                    (
                        "repair robot-base kinematics and flower-frame contact constraints",
                        "generate multiple phase-local candidates and run a pairwise tournament",
                    )
                )
            evidence.extend(
                (
                    "per-frame robot joint and end-effector trajectories in robot:base",
                    "per-flower 6-DoF/contact-state trajectories in object:flower",
                    "contact-onset and release timing for every manipulation phase",
                )
            )

        if "human_residual_absence" in failures:
            next_config = replace(next_config, depth_layering=True, local_generation_only=True)
            actions.append("replace the human through a tracked person matte and clean plate")
            evidence.append("full-frame human-remnant segmentation audit over every frame")

        if "background_lock" in failures or "flower_integrity" in failures:
            next_config = replace(next_config, depth_layering=True, local_generation_only=True)
            actions.append("restore immutable source background and flower instances by layer")
            evidence.append("post-decode pixel audit outside the robot/person union mask")

        temporal_only = set(failures) <= {"temporal_consistency", "full_frame_coverage"}
        if temporal_only:
            actions.append("repair only the measured temporal neighborhoods; preserve endpoints")
        if "temporal_consistency" in failures:
            evidence.append("all-frame transition energy plus high-jerk consecutive-frame review")

        if "full_frame_coverage" in failures:
            actions.append("decode and evaluate the complete clip before any acceptance decision")
        if "full_video_human_preference" in failures:
            actions.append("run a blind pairwise full-video review against the strongest baseline")
            evidence.append("reviewer verdict and failure timestamps for the complete video")

        if not actions:
            actions.append("collect the missing hard-gate evidence before mutating the pipeline")
        return FlowerEvolutionDecision(
            status="PARTIAL",
            failed_gates=failures,
            next_config=next_config,
            actions=tuple(dict.fromkeys(actions)),
            required_evidence=tuple(dict.fromkeys(evidence)),
            training_required=training_required,
        )

    def select_parent(
        self,
        candidates: Sequence[FlowerCandidateEvaluation],
        contract: FlowerAcceptanceContract,
    ) -> FlowerCandidateEvaluation:
        """Select lexicographically; failed hard gates dominate any mean score."""

        if not candidates:
            raise ValueError("at least one flower candidate is required")
        return max(
            candidates,
            key=lambda candidate: (
                candidate.accepted(contract),
                candidate.pass_count(contract),
                candidate.worst_margin(contract),
                candidate.candidate_id,
            ),
        )
