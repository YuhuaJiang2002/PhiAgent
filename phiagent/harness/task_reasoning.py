"""Typed physical/task reasoning and language planning for generation harnesses.

The plugin deliberately produces constraints, not robot commands. It keeps image,
world, and robot-base frames distinct and fails closed when metric geometry,
contact force, or calibrated depth are unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any, Mapping, Protocol, Sequence


SCHEMA_VERSION = "1.0.0"
PLUGIN_ENTRYPOINT_GROUP = "phiagent.harness.reasoning_plugins"
OPTICAL_MODULE_TASK = "optical_module_grasp_insert"
TSHIRT_FOLD_TASK = "tshirt_fold_left_right_aside"
_SUPPORTED_TASK_TYPES = {OPTICAL_MODULE_TASK, TSHIRT_FOLD_TASK}
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_LEVELS = {"OBSERVED", "INFERRED", "UNAVAILABLE"}
_VERIFICATION_SOURCES = {"automatic_proxy", "native_resolution_human_review"}
_SPEED_CLASSES = {"stationary", "fine", "slow", "coarse"}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _require_nonempty(values: Sequence[str], field: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{field} must contain non-empty values")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class ReasoningPluginDescriptor:
    name: str
    version: str
    stage: str
    description: str
    capabilities: tuple[str, ...]
    deterministic: bool
    heavyweight: bool

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.name):
            raise ValueError("reasoning plugin name must be filesystem-safe")
        if not self.version.strip():
            raise ValueError("reasoning plugin version is required")
        if self.stage != "reasoning":
            raise ValueError("reasoning plugins must provide the reasoning stage")
        if not self.description.strip():
            raise ValueError("reasoning plugin description is required")
        object.__setattr__(
            self,
            "capabilities",
            _require_nonempty(self.capabilities, "reasoning plugin capabilities"),
        )


@dataclass(frozen=True)
class TaskEntity:
    entity_id: str
    role: str
    description: str

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.entity_id):
            raise ValueError("task entity id must be filesystem-safe")
        if not self.role.strip() or not self.description.strip():
            raise ValueError("task entity role and description are required")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskEntity:
        return cls(
            entity_id=str(payload["entity_id"]),
            role=str(payload["role"]),
            description=str(payload["description"]),
        )


@dataclass(frozen=True)
class TaskReasoningRequest:
    task_id: str
    task_type: str
    instruction: str
    coordinate_frame: str
    duration_seconds: float
    entities: tuple[TaskEntity, ...]
    available_evidence: tuple[str, ...]
    unavailable_evidence: tuple[str, ...]
    user_constraints: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.task_id):
            raise ValueError("task id must be filesystem-safe")
        if self.task_type not in _SUPPORTED_TASK_TYPES:
            raise ValueError(f"unsupported physical task type: {self.task_type}")
        if not self.instruction.strip():
            raise ValueError("task instruction is required")
        if not self.coordinate_frame.startswith("camera:"):
            raise ValueError("visual task reasoning requires a named camera frame")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("task duration must be finite and positive")
        if self.task_type == OPTICAL_MODULE_TASK and len(self.entities) < 3:
            raise ValueError("optical-module planning requires gripper, module, and receptacle")
        if self.task_type == TSHIRT_FOLD_TASK and len(self.entities) < 5:
            raise ValueError(
                "T-shirt folding planning requires two manipulators, two sleeves, and a shirt body"
            )
        entity_ids = tuple(entity.entity_id for entity in self.entities)
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("task entity ids must be unique")
        object.__setattr__(
            self,
            "available_evidence",
            _require_nonempty(self.available_evidence, "available evidence"),
        )
        object.__setattr__(
            self,
            "unavailable_evidence",
            _require_nonempty(self.unavailable_evidence, "unavailable evidence"),
        )
        object.__setattr__(
            self,
            "user_constraints",
            _require_nonempty(self.user_constraints, "user constraints"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskReasoningRequest:
        return cls(
            task_id=str(payload["task_id"]),
            task_type=str(payload["task_type"]),
            instruction=str(payload["instruction"]),
            coordinate_frame=str(payload["coordinate_frame"]),
            duration_seconds=float(payload["duration_seconds"]),
            entities=tuple(TaskEntity.from_dict(item) for item in payload["entities"]),
            available_evidence=tuple(str(item) for item in payload["available_evidence"]),
            unavailable_evidence=tuple(str(item) for item in payload["unavailable_evidence"]),
            user_constraints=tuple(str(item) for item in payload["user_constraints"]),
        )


@dataclass(frozen=True)
class LanguageAnalysis:
    source_language: str
    normalized_instruction: str
    ordered_actions: tuple[str, ...]
    spatial_relations: tuple[str, ...]
    temporal_modifiers: tuple[str, ...]
    ambiguity_resolutions: tuple[str, ...]


@dataclass(frozen=True)
class ReasoningFinding:
    finding_id: str
    category: str
    evidence_level: str
    conclusion: str
    rationale: str
    planning_consequence: str

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.finding_id):
            raise ValueError("finding id must be filesystem-safe")
        if self.evidence_level not in _EVIDENCE_LEVELS:
            raise ValueError(f"unsupported evidence level: {self.evidence_level}")
        for value in (
            self.category,
            self.conclusion,
            self.rationale,
            self.planning_consequence,
        ):
            if not value.strip():
                raise ValueError("reasoning findings require complete text")


@dataclass(frozen=True)
class VerificationGate:
    gate_id: str
    description: str
    evidence_source: str
    fail_closed: bool = True

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.gate_id):
            raise ValueError("verification gate id must be filesystem-safe")
        if not self.description.strip():
            raise ValueError("verification gate description is required")
        if self.evidence_source not in _VERIFICATION_SOURCES:
            raise ValueError(f"unsupported verification source: {self.evidence_source}")
        if not self.fail_closed:
            raise ValueError("physical/task reasoning gates must fail closed")


@dataclass(frozen=True)
class TaskPhase:
    phase_id: str
    start_seconds: float
    end_seconds: float
    objective: str
    language_directive: str
    motion_frame: str
    speed_class: str
    preconditions: tuple[str, ...]
    invariants: tuple[str, ...]
    postconditions: tuple[str, ...]
    gate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.phase_id):
            raise ValueError("phase id must be filesystem-safe")
        if (
            not math.isfinite(self.start_seconds)
            or not math.isfinite(self.end_seconds)
            or self.start_seconds < 0
            or self.end_seconds <= self.start_seconds
        ):
            raise ValueError("phase boundaries must be finite, ordered, and non-negative")
        if not self.objective.strip() or not self.language_directive.strip():
            raise ValueError("phase objective and language directive are required")
        if not self.motion_frame.startswith("camera:"):
            raise ValueError("phase motion must remain in a named camera frame")
        if self.speed_class not in _SPEED_CLASSES:
            raise ValueError(f"unsupported speed class: {self.speed_class}")
        for field, values in (
            ("preconditions", self.preconditions),
            ("invariants", self.invariants),
            ("postconditions", self.postconditions),
            ("gate_ids", self.gate_ids),
        ):
            object.__setattr__(self, field, _require_nonempty(values, field))


@dataclass(frozen=True)
class TaskReasoningPlan:
    schema_version: str
    plugin: ReasoningPluginDescriptor
    task_id: str
    task_type: str
    coordinate_frame: str
    duration_seconds: float
    language_analysis: LanguageAnalysis
    physical_analysis: tuple[ReasoningFinding, ...]
    phases: tuple[TaskPhase, ...]
    global_constraints: tuple[str, ...]
    verification_gates: tuple[VerificationGate, ...]
    claim_boundary: str
    plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskReasoningPlan:
        plugin = payload["plugin"]
        language = payload["language_analysis"]
        plan = cls(
            schema_version=str(payload["schema_version"]),
            plugin=ReasoningPluginDescriptor(
                name=str(plugin["name"]),
                version=str(plugin["version"]),
                stage=str(plugin["stage"]),
                description=str(plugin["description"]),
                capabilities=tuple(str(item) for item in plugin["capabilities"]),
                deterministic=bool(plugin["deterministic"]),
                heavyweight=bool(plugin["heavyweight"]),
            ),
            task_id=str(payload["task_id"]),
            task_type=str(payload["task_type"]),
            coordinate_frame=str(payload["coordinate_frame"]),
            duration_seconds=float(payload["duration_seconds"]),
            language_analysis=LanguageAnalysis(
                source_language=str(language["source_language"]),
                normalized_instruction=str(language["normalized_instruction"]),
                ordered_actions=tuple(str(item) for item in language["ordered_actions"]),
                spatial_relations=tuple(str(item) for item in language["spatial_relations"]),
                temporal_modifiers=tuple(str(item) for item in language["temporal_modifiers"]),
                ambiguity_resolutions=tuple(
                    str(item) for item in language["ambiguity_resolutions"]
                ),
            ),
            physical_analysis=tuple(
                ReasoningFinding(
                    finding_id=str(item["finding_id"]),
                    category=str(item["category"]),
                    evidence_level=str(item["evidence_level"]),
                    conclusion=str(item["conclusion"]),
                    rationale=str(item["rationale"]),
                    planning_consequence=str(item["planning_consequence"]),
                )
                for item in payload["physical_analysis"]
            ),
            phases=tuple(
                TaskPhase(
                    phase_id=str(item["phase_id"]),
                    start_seconds=float(item["start_seconds"]),
                    end_seconds=float(item["end_seconds"]),
                    objective=str(item["objective"]),
                    language_directive=str(item["language_directive"]),
                    motion_frame=str(item["motion_frame"]),
                    speed_class=str(item["speed_class"]),
                    preconditions=tuple(str(value) for value in item["preconditions"]),
                    invariants=tuple(str(value) for value in item["invariants"]),
                    postconditions=tuple(str(value) for value in item["postconditions"]),
                    gate_ids=tuple(str(value) for value in item["gate_ids"]),
                )
                for item in payload["phases"]
            ),
            global_constraints=tuple(str(item) for item in payload["global_constraints"]),
            verification_gates=tuple(
                VerificationGate(
                    gate_id=str(item["gate_id"]),
                    description=str(item["description"]),
                    evidence_source=str(item["evidence_source"]),
                    fail_closed=bool(item["fail_closed"]),
                )
                for item in payload["verification_gates"]
            ),
            claim_boundary=str(payload["claim_boundary"]),
            plan_sha256=str(payload["plan_sha256"]),
        )
        validate_task_reasoning_plan(plan)
        return plan


class TaskReasoningPlugin(Protocol):
    descriptor: ReasoningPluginDescriptor

    def analyze(self, request: TaskReasoningRequest) -> TaskReasoningPlan:
        """Expand a typed task request into a hash-bound, fail-closed phase plan."""


def _detect_language(instruction: str) -> str:
    return "zh-CN" if any("\u4e00" <= character <= "\u9fff" for character in instruction) else "en"


def _scaled_boundaries(duration_seconds: float) -> tuple[float, ...]:
    fractions = (0.0, 0.07, 0.20, 0.30, 0.40, 0.60, 0.75, 0.94, 1.0)
    return tuple(round(duration_seconds * fraction, 6) for fraction in fractions)


class PhysicalTaskReasoningPlugin:
    """Rule-bound planner for visually grounded manipulation generation."""

    descriptor = ReasoningPluginDescriptor(
        name="physical-task-language-planner",
        version="1.0.0",
        stage="reasoning",
        description=(
            "Expands typed language intent into ordered physical preconditions, "
            "camera-frame motion phases, and fail-closed verification gates."
        ),
        capabilities=(
            "language_analysis",
            "task_expansion",
            "physical_reasonableness",
            OPTICAL_MODULE_TASK,
        ),
        deterministic=True,
        heavyweight=False,
    )

    def analyze(self, request: TaskReasoningRequest) -> TaskReasoningPlan:
        if request.task_type != OPTICAL_MODULE_TASK:
            raise ValueError(f"unsupported task type: {request.task_type}")
        frame = request.coordinate_frame
        boundaries = _scaled_boundaries(request.duration_seconds)
        gates = (
            VerificationGate(
                "single_module_identity",
                "Exactly one rigid optical module persists through every phase.",
                "automatic_proxy",
            ),
            VerificationGate(
                "module_static_until_contact",
                "The table-supported module does not translate before both jaws visibly contact it.",
                "automatic_proxy",
            ),
            VerificationGate(
                "downward_approach_precedes_closure",
                "The open gripper first moves toward the table-supported module and reaches grasp height before jaw closure.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "metal_body_grasp",
                "Both jaws close on the rigid metal housing rather than the green pull tab.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "lift_clearance_precedes_transport",
                "The grasped module visibly clears the tabletop before coarse lateral transport.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "preinsert_standoff",
                "Coarse transport stops outside the receptacle before fine alignment begins.",
                "automatic_proxy",
            ),
            VerificationGate(
                "coaxial_and_level_before_insertion",
                "The connector and receptacle centerlines are aligned at a common apparent height before inward motion.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "insertion_slower_than_transport",
                "Median image-plane speed during insertion is no more than 35 percent of coarse transport speed.",
                "automatic_proxy",
            ),
            VerificationGate(
                "axial_insertion_without_sweep",
                "Insertion follows the module long axis without lateral sweeping or a new rotation.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "terminal_seated_hold",
                "The final frames hold one visibly inserted module without rebound or drift.",
                "automatic_proxy",
            ),
            VerificationGate(
                "no_release_without_seating_evidence",
                "The gripper remains closed because this visual path has no force, depth, or electrical seating signal.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "camera_and_background_static",
                "The camera, fixture, receptacle, table, cables, and background remain stable.",
                "automatic_proxy",
            ),
        )
        phases = (
            TaskPhase(
                "observe_and_localize",
                boundaries[0],
                boundaries[1],
                "Freeze the initial state and bind the one gripper, module, and receptacle.",
                (
                    "Hold the exact input frame. Keep the open gripper, table-supported optical "
                    "module, and fixed receptacle motionless while preserving one-object identity."
                ),
                frame,
                "stationary",
                ("The supplied first frame is the visual authority.",),
                ("No object, camera, or background motion.",),
                ("The task entities remain uniquely identifiable.",),
                ("single_module_identity", "camera_and_background_static"),
            ),
            TaskPhase(
                "descend_to_grasp_height",
                boundaries[1],
                boundaries[2],
                "Move the open jaws down toward the tabletop until they straddle the metal housing.",
                (
                    "Move the open two-jaw gripper slightly downward toward the tabletop and the "
                    "module. Keep the module completely still. Stop only when the inner jaw faces "
                    "straddle the rigid silver metal body at the same apparent grasp height; do "
                    "not close while the jaws remain above it."
                ),
                frame,
                "fine",
                ("The module is resting on the tabletop.", "The jaws are open."),
                ("The module and receptacle remain fixed.", "The green pull tab is untouched."),
                ("Both jaw faces bracket the metal housing before closure.",),
                ("downward_approach_precedes_closure", "module_static_until_contact"),
            ),
            TaskPhase(
                "close_on_metal_body",
                boundaries[2],
                boundaries[3],
                "Establish a symmetric two-jaw grasp before moving the module.",
                (
                    "Close both jaws slowly and symmetrically around the silver metal housing. "
                    "Do not grasp, bend, or pull the bright green pull tab. The module must not "
                    "move until both jaws visibly contact the housing."
                ),
                frame,
                "slow",
                ("Both jaw faces bracket the metal housing.",),
                ("The module remains table-supported until visible two-sided contact.",),
                ("A stable two-jaw metal-body grasp is established.",),
                ("metal_body_grasp", "module_static_until_contact"),
            ),
            TaskPhase(
                "lift_for_clearance",
                boundaries[3],
                boundaries[4],
                "Lift the grasped module enough to clear the tabletop.",
                (
                    "Lift the securely grasped module a small visible distance away from the "
                    "tabletop before any upper-left transport. Preserve one rigid module and one "
                    "unchanged grasp."
                ),
                frame,
                "slow",
                ("A stable two-jaw metal-body grasp exists.",),
                ("No lateral transport before visible clearance.",),
                ("The module visibly clears the tabletop.",),
                ("lift_clearance_precedes_transport", "single_module_identity"),
            ),
            TaskPhase(
                "coarse_transport_to_standoff",
                boundaries[4],
                boundaries[5],
                "Carry the lifted module near the receptacle without beginning insertion.",
                (
                    "Carry the lifted module smoothly diagonally upper-left toward the fixed "
                    "receptacle. Keep the connector end leading. Stop outside the mouth with a "
                    "small visible standoff; do not insert during this coarse transport."
                ),
                frame,
                "coarse",
                ("The module has visible tabletop clearance.",),
                ("The grasp and module orientation remain stable.",),
                ("The connector stops outside the receptacle at pre-insertion standoff.",),
                ("preinsert_standoff", "single_module_identity"),
            ),
            TaskPhase(
                "coaxial_preinsert_alignment",
                boundaries[5],
                boundaries[6],
                "Align connector height, centerline, and long axis before inward motion.",
                (
                    "Pause the coarse advance. Use only small fine adjustments to place the "
                    "connector at the same apparent height as the receptacle mouth, center the "
                    "connector on the mouth, and make the module long axis collinear with the "
                    "insertion axis. Hold a short settling dwell while remaining outside the slot."
                ),
                frame,
                "fine",
                ("The connector is stopped at visible standoff.",),
                ("No inward insertion before height and centerline alignment.",),
                ("The connector is level, centered, coaxial, and settled outside the mouth.",),
                ("coaxial_and_level_before_insertion", "preinsert_standoff"),
            ),
            TaskPhase(
                "slow_axial_insertion",
                boundaries[6],
                boundaries[7],
                "Insert slowly along the aligned module axis.",
                (
                    "Insert the connector slowly and incrementally straight along the aligned "
                    "module long axis. Use no more than 35 percent of the coarse transport speed. "
                    "Do not sweep sideways, rotate, pitch, or push before alignment. Increase "
                    "occlusion by the receptacle naturally as the module enters."
                ),
                frame,
                "slow",
                ("The connector is level, centered, coaxial, and settled.",),
                ("Axial motion only; no lateral sweep, new rotation, or identity change.",),
                ("The module reaches a visibly inserted terminal state.",),
                (
                    "insertion_slower_than_transport",
                    "axial_insertion_without_sweep",
                    "single_module_identity",
                ),
            ),
            TaskPhase(
                "seated_hold_without_release",
                boundaries[7],
                boundaries[8],
                "Stop and hold the visually inserted state without claiming physical seating.",
                (
                    "Stop all inward motion and hold the final inserted state motionless. Keep "
                    "both jaws closed on the metal housing because no force, calibrated depth, "
                    "or electrical seating signal is available; do not release or retract."
                ),
                frame,
                "stationary",
                ("The module is visibly inserted.",),
                ("No release, rebound, drift, or receptacle motion.",),
                ("The final inserted visual state remains stable through the last frame.",),
                ("terminal_seated_hold", "no_release_without_seating_evidence"),
            ),
        )
        language = LanguageAnalysis(
            source_language=_detect_language(request.instruction),
            normalized_instruction=(
                "Lower the open gripper to the tabletop-supported optical module, grasp its "
                "metal housing, lift it clear, move to a pre-insertion standoff, align the "
                "connector level and coaxial with the receptacle, then insert slowly along the "
                "module axis and hold without release."
            ),
            ordered_actions=tuple(phase.phase_id for phase in phases),
            spatial_relations=(
                "gripper descends toward the table-supported module before closure",
                "connector stops outside the receptacle at pre-insertion standoff",
                "connector and receptacle share apparent height and centerline before insertion",
                "insertion follows the module long axis",
            ),
            temporal_modifiers=(
                "approach before closure",
                "clearance before transport",
                "alignment and settling before insertion",
                "insertion slower than coarse transport",
                "hold without release after visual insertion",
            ),
            ambiguity_resolutions=(
                (
                    "Downward is a relational move toward the visible tabletop, not a calibrated "
                    "world-z or robot-base command."
                ),
                (
                    "Level and coaxial are apparent camera-frame relations because metric depth "
                    "and receptacle pose are unavailable."
                ),
                (
                    "Seated means visually inserted and stable; it does not mean electrical or "
                    "force-confirmed mating."
                ),
            ),
        )
        findings = (
            ReasoningFinding(
                "table_support_requires_descent",
                "grasp_precondition",
                "INFERRED",
                "The module is table-supported while the gripper is not yet at a secure grasp height.",
                "Closing early can pinch above the object or contact only the pull tab.",
                "Insert a distinct open-jaw downward approach phase before closure.",
            ),
            ReasoningFinding(
                "metal_housing_is_grasp_surface",
                "affordance",
                "OBSERVED",
                "The rigid metal housing is a safer visual grasp surface than the green pull tab.",
                "The pull tab is narrow and visibly deformable.",
                "Require symmetric contact on the metal housing and prohibit pull-tab grasping.",
            ),
            ReasoningFinding(
                "clearance_before_lateral_motion",
                "collision_reasoning",
                "INFERRED",
                "Lateral transport should begin only after visible tabletop clearance.",
                "Dragging the module can cause collision, snagging, or an implausible visual path.",
                "Separate lift-for-clearance from coarse transport.",
            ),
            ReasoningFinding(
                "alignment_before_insertion",
                "insertion_precondition",
                "INFERRED",
                "Connector height, centerline, and long axis must align before inward motion.",
                "Misaligned insertion implies collision with the receptacle rim.",
                "Add a stopped standoff, fine alignment, and settling phase.",
            ),
            ReasoningFinding(
                "slow_insertion_after_coarse_transport",
                "motion_profile",
                "INFERRED",
                "Insertion needs finer motion than free-space transport.",
                "A lower speed reduces overshoot and makes axial engagement visually inspectable.",
                "Cap insertion image-plane speed at 35 percent of coarse transport speed.",
            ),
            ReasoningFinding(
                "seating_signal_unavailable",
                "claim_boundary",
                "UNAVAILABLE",
                "The harness has no calibrated depth, force, tactile, or electrical mating signal.",
                "Visual overlap alone cannot establish physical seating.",
                "Hold the grasp after visual insertion and prohibit automatic release.",
            ),
        )
        unlocked: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "plugin": asdict(self.descriptor),
            "task_id": request.task_id,
            "task_type": request.task_type,
            "coordinate_frame": frame,
            "duration_seconds": request.duration_seconds,
            "language_analysis": asdict(language),
            "physical_analysis": [asdict(item) for item in findings],
            "phases": [asdict(item) for item in phases],
            "global_constraints": [
                "Use only the named camera frame; do not relabel image motion as world-z or robot-base motion.",
                "Preserve exactly one gripper, one rigid optical module, and one fixed receptacle.",
                "The module moves only after visible two-sided jaw contact.",
                "The camera, fixture, table, cables, lighting, and background remain fixed.",
                "No phase may imply force, tactile, electrical, or calibrated depth evidence.",
            ],
            "verification_gates": [asdict(item) for item in gates],
            "claim_boundary": (
                "This plan constrains a language-conditioned visual future. Apparent descent, "
                "clearance, alignment, and insertion are camera-frame relations, not calibrated "
                "3-D geometry, force-controlled insertion, robot commands, or physical success."
            ),
        }
        plan = TaskReasoningPlan.from_dict(
            {**unlocked, "plan_sha256": _canonical_sha256(unlocked)}
        )
        return plan


def _tshirt_fold_boundaries(duration_seconds: float) -> tuple[float, ...]:
    fractions = (0.0, 0.05, 0.15, 0.32, 0.38, 0.48, 0.64, 0.80, 0.90, 0.97, 1.0)
    return tuple(round(duration_seconds * fraction, 6) for fraction in fractions)


class TshirtFoldReasoningPlugin:
    """Fail-closed planner for a two-arm, viewer-relative T-shirt fold."""

    descriptor = ReasoningPluginDescriptor(
        name="tshirt-fold-physical-language-planner",
        version="1.0.0",
        stage="reasoning",
        description=(
            "Expands a T-shirt folding instruction into contact-causal phases and "
            "non-negotiable material-conservation gates."
        ),
        capabilities=(
            "language_analysis",
            "task_expansion",
            "cloth_material_conservation",
            "contact_causality",
            "test_time_scaling_feedback",
            TSHIRT_FOLD_TASK,
        ),
        deterministic=True,
        heavyweight=False,
    )

    def analyze(self, request: TaskReasoningRequest) -> TaskReasoningPlan:
        if request.task_type != TSHIRT_FOLD_TASK:
            raise ValueError(f"unsupported task type: {request.task_type}")
        frame = request.coordinate_frame
        boundaries = _tshirt_fold_boundaries(request.duration_seconds)
        gates = (
            VerificationGate(
                "exact_first_frame",
                "Decoded frame zero preserves the supplied first-frame pixels within the frozen codec tolerance.",
                "automatic_proxy",
            ),
            VerificationGate(
                "single_shirt_identity",
                "Exactly one gray-body black-sleeve T-shirt persists without duplication or material substitution.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "viewer_left_sleeve_length_conserved",
                "Tracked viewer-left cuff-to-shoulder material polyline stays within the frozen length and segment-deformation bounds.",
                "automatic_proxy",
            ),
            VerificationGate(
                "viewer_right_sleeve_length_conserved",
                "Tracked viewer-right cuff-to-shoulder material polyline stays within the frozen length and segment-deformation bounds.",
                "automatic_proxy",
            ),
            VerificationGate(
                "cuff_and_shoulder_identity_persistent",
                "Both cuffs and both shoulder seams keep their material identity through folds and temporary overlap.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "contact_precedes_cloth_motion",
                "A stabilizing contact and a cuff-side grasp are established before the contacted sleeve begins moving.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "viewer_left_fold_precedes_viewer_right_fold",
                "The viewer-left sleeve completes and settles before the viewer-right sleeve starts folding.",
                "automatic_proxy",
            ),
            VerificationGate(
                "no_teleportation_or_crossfade",
                "Tracked material and gripper points move continuously with no hard cut, dissolve, or single-frame jump over the frozen bound.",
                "automatic_proxy",
            ),
            VerificationGate(
                "body_fold_after_both_sleeves",
                "The lower shirt body starts folding only after both sleeves are folded and settled.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "bundle_move_after_body_fold",
                "The compact folded bundle moves viewer-left only after the body fold is complete.",
                "automatic_proxy",
            ),
            VerificationGate(
                "camera_and_background_static",
                "The camera, table, glass partition, cables, surrounding garments, and lighting remain stable.",
                "automatic_proxy",
            ),
            VerificationGate(
                "terminal_compact_bundle_stable",
                "The final compact bundle remains on the viewer-left side without rebound, unfolding, or drift.",
                "native_resolution_human_review",
            ),
        )
        phases = (
            TaskPhase(
                "initial_state_hold",
                boundaries[0],
                boundaries[1],
                "Bind the exact source scene and all material identities before motion.",
                "Hold the exact supplied first frame. Do not move either robot or any part of the shirt.",
                frame,
                "stationary",
                ("The supplied first frame is the only initial-state authority.",),
                ("No camera, background, robot, or cloth motion.",),
                ("Both cuffs, shoulder seams, body, and two manipulators are uniquely bound.",),
                ("exact_first_frame", "single_shirt_identity", "camera_and_background_static"),
            ),
            TaskPhase(
                "establish_viewer_left_two_point_contact",
                boundaries[1],
                boundaries[2],
                "Stabilize the left shoulder region and establish a cuff-side grasp.",
                (
                    "Approach the viewer-left black sleeve without moving it. One gripper stabilizes "
                    "the shoulder/body junction while the other establishes visible cuff-side contact."
                ),
                frame,
                "fine",
                ("The viewer-left sleeve is flat and motionless.",),
                ("No cloth motion before two visible contacts; the viewer-right sleeve stays still.",),
                ("Stable shoulder-side and cuff-side contacts are visible.",),
                ("contact_precedes_cloth_motion", "cuff_and_shoulder_identity_persistent"),
            ),
            TaskPhase(
                "fold_viewer_left_sleeve",
                boundaries[2],
                boundaries[3],
                "Fold the viewer-left sleeve inward without changing its material length.",
                (
                    "Rotate and guide the viewer-left sleeve inward across the gray torso as one "
                    "continuous cloth strip. Preserve the cuff-to-shoulder arclength and every "
                    "material segment; create a moving fold arc, never shrink, grow, dissolve, or teleport it."
                ),
                frame,
                "slow",
                ("Two-point left-sleeve contact is established.",),
                ("Viewer-right sleeve and shirt body stay fixed; both left contacts remain causal.",),
                ("Viewer-left sleeve lies fully inward with unchanged material identity and length.",),
                (
                    "viewer_left_sleeve_length_conserved",
                    "contact_precedes_cloth_motion",
                    "no_teleportation_or_crossfade",
                ),
            ),
            TaskPhase(
                "settle_viewer_left_sleeve",
                boundaries[3],
                boundaries[4],
                "Settle the first sleeve before initiating the second fold.",
                "Hold the folded viewer-left sleeve motionless and keep the viewer-right sleeve completely unchanged.",
                frame,
                "stationary",
                ("The viewer-left sleeve has reached its inward terminal pose.",),
                ("No rebound, unfolding, or viewer-right sleeve motion.",),
                ("The first sleeve is visibly settled before the second approach.",),
                (
                    "viewer_left_sleeve_length_conserved",
                    "viewer_left_fold_precedes_viewer_right_fold",
                ),
            ),
            TaskPhase(
                "establish_viewer_right_two_point_contact",
                boundaries[4],
                boundaries[5],
                "Stabilize the right shoulder region and establish a cuff-side grasp.",
                (
                    "Keep the folded viewer-left sleeve fixed. Approach the viewer-right black sleeve; "
                    "stabilize its shoulder/body junction and establish visible cuff-side contact before motion."
                ),
                frame,
                "fine",
                ("The viewer-left sleeve is settled and the viewer-right sleeve is still flat.",),
                ("No right-sleeve motion before two visible contacts.",),
                ("Stable shoulder-side and cuff-side contacts are visible on the right sleeve.",),
                ("contact_precedes_cloth_motion", "viewer_left_fold_precedes_viewer_right_fold"),
            ),
            TaskPhase(
                "fold_viewer_right_sleeve",
                boundaries[5],
                boundaries[6],
                "Fold the viewer-right sleeve inward without changing its material length.",
                (
                    "Rotate and guide the viewer-right sleeve inward across the gray torso as one "
                    "continuous cloth strip. Preserve cuff-to-shoulder arclength and material segments; "
                    "do not disturb the already folded left sleeve."
                ),
                frame,
                "slow",
                ("Two-point right-sleeve contact is established after the left fold settled.",),
                ("The left fold stays fixed; right contacts remain causal; no crossfade or hard cut.",),
                ("Both sleeves lie inward with unchanged material identities and lengths.",),
                (
                    "viewer_right_sleeve_length_conserved",
                    "cuff_and_shoulder_identity_persistent",
                    "no_teleportation_or_crossfade",
                    "viewer_left_fold_precedes_viewer_right_fold",
                ),
            ),
            TaskPhase(
                "fold_body_bottom_to_top",
                boundaries[6],
                boundaries[7],
                "Fold the lower torso upward only after both sleeves are settled.",
                (
                    "After both sleeves are fully folded, grasp and lift the lower hem, then roll the "
                    "gray body upward in one continuous fold about a horizontal crease. Keep one shirt identity."
                ),
                frame,
                "slow",
                ("Both sleeve folds are complete and settled.",),
                ("No sleeve unfolding, cloth duplication, teleportation, or camera motion.",),
                ("The shirt forms one compact layered rectangle at the center.",),
                (
                    "body_fold_after_both_sleeves",
                    "single_shirt_identity",
                    "no_teleportation_or_crossfade",
                ),
            ),
            TaskPhase(
                "compress_bundle_without_stretch",
                boundaries[7],
                boundaries[8],
                "Settle the layered rectangle without changing sleeve material length.",
                "Use gentle contact to square the folded bundle; do not stretch, shorten, erase, or add cloth.",
                frame,
                "fine",
                ("One layered rectangular bundle exists at the center.",),
                ("Both sleeves retain their bound material length and identity inside the layers.",),
                ("The compact bundle is stable and ready for transport.",),
                (
                    "viewer_left_sleeve_length_conserved",
                    "viewer_right_sleeve_length_conserved",
                    "terminal_compact_bundle_stable",
                ),
            ),
            TaskPhase(
                "move_folded_bundle_viewer_left",
                boundaries[8],
                boundaries[9],
                "Move the completed bundle to the viewer-left side as one object.",
                (
                    "Only after folding is complete, lift or slide the entire compact bundle smoothly "
                    "to the clear viewer-left side. Preserve its shape and keep the background fixed."
                ),
                frame,
                "slow",
                ("The fold is complete and the bundle is stable.",),
                ("The bundle moves as one object without reopening or a single-frame jump.",),
                ("The compact bundle reaches the viewer-left side and clears the center workspace.",),
                (
                    "bundle_move_after_body_fold",
                    "no_teleportation_or_crossfade",
                    "camera_and_background_static",
                ),
            ),
            TaskPhase(
                "terminal_bundle_hold",
                boundaries[9],
                boundaries[10],
                "Hold the completed fold at the side for inspection.",
                "Stop all motion and hold one compact folded shirt on the viewer-left side through the last frame.",
                frame,
                "stationary",
                ("The compact bundle is at the viewer-left side.",),
                ("No drift, rebound, unfolding, identity change, or background motion.",),
                ("The final state remains stable through the final frame.",),
                (
                    "terminal_compact_bundle_stable",
                    "single_shirt_identity",
                    "camera_and_background_static",
                ),
            ),
        )
        language = LanguageAnalysis(
            source_language=_detect_language(request.instruction),
            normalized_instruction=(
                "Preserve the exact first frame; establish two-point contact and fold the "
                "viewer-left sleeve inward without material-length change; settle it; repeat for "
                "the viewer-right sleeve; fold the body bottom-to-top; compact the bundle; move "
                "the completed bundle viewer-left; hold."
            ),
            ordered_actions=tuple(phase.phase_id for phase in phases),
            spatial_relations=(
                "viewer-left and viewer-right are defined in the fixed input camera frame",
                "each cuff remains bound to its original shoulder seam through a material polyline",
                "the lower hem folds upward only after both sleeves settle",
                "the completed bundle moves to the viewer-left side only after body folding",
            ),
            temporal_modifiers=(
                "contact before cloth motion",
                "viewer-left fold and settle before viewer-right fold",
                "both sleeves before body fold",
                "body fold before bundle transport",
                "terminal hold after transport",
            ),
            ambiguity_resolutions=(
                "Left and right mean viewer-relative directions in the named camera frame, not robot-base sides.",
                "Sleeve length means tracked camera-frame material-polyline arclength; it is a strict visual gate, not metric 3-D cloth calibration.",
                "Folded means one visually compact layered shirt; it does not establish force-controlled manipulation or real-robot feasibility.",
            ),
        )
        findings = (
            ReasoningFinding(
                "sleeves_require_material_identity",
                "cloth_conservation",
                "INFERRED",
                "A sleeve cannot physically shrink or grow while being folded.",
                "Crossfades and generative morphing often shorten the black sleeve between key states.",
                "Bind cuff, intermediate seam points, and shoulder; fail on missing tracks or length drift.",
            ),
            ReasoningFinding(
                "two_point_support_reduces_stretch",
                "contact_reasoning",
                "INFERRED",
                "A shoulder-side stabilizer and cuff-side grasp make the intended flat fold observable.",
                "Moving cloth before visible contact breaks causal manipulation and encourages teleportation.",
                "Insert a contact-establishment phase before each sleeve moves.",
            ),
            ReasoningFinding(
                "folds_need_settling_dwell",
                "temporal_reasoning",
                "INFERRED",
                "Each sleeve needs a settled terminal state before the next manipulation.",
                "Overlapping phase motion obscures action order and can create topology changes.",
                "Require an explicit left-sleeve dwell before right-sleeve approach.",
            ),
            ReasoningFinding(
                "physical_measurements_unavailable",
                "claim_boundary",
                "UNAVAILABLE",
                "The visual harness has no calibrated 3-D cloth mesh, force, tactile data, or exact robot trajectory.",
                "Image-space consistency alone cannot prove physical execution.",
                "Treat H3 as a proposal model and keep physical/real-robot claims out of promotion.",
            ),
        )
        unlocked: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "plugin": asdict(self.descriptor),
            "task_id": request.task_id,
            "task_type": request.task_type,
            "coordinate_frame": frame,
            "duration_seconds": request.duration_seconds,
            "language_analysis": asdict(language),
            "physical_analysis": [asdict(item) for item in findings],
            "phases": [asdict(item) for item in phases],
            "global_constraints": [
                "Use only the named camera frame; viewer-left and viewer-right never change meaning.",
                "Preserve one shirt, two original sleeves, both cuffs, both shoulder seams, and two manipulators.",
                "A sleeve must not move before visible contact and its tracked material length must not change.",
                "Never use a mean score, higher seed count, or human preference to override a failed hard gate.",
                "The camera, table, glass partition, cables, surrounding garments, lighting, and background remain fixed.",
                "Do not claim metric geometry, contact force, safety, joint feasibility, or real-robot success."
            ],
            "verification_gates": [asdict(item) for item in gates],
            "claim_boundary": (
                "This plan constrains a generated camera-pixel demonstration. Sleeve length, contact "
                "order, and continuity are fail-closed visual gates; they are not calibrated 3-D cloth "
                "geometry, force evidence, robot joint commands, collision safety, or physical success."
            ),
        }
        return TaskReasoningPlan.from_dict(
            {**unlocked, "plan_sha256": _canonical_sha256(unlocked)}
        )


def validate_task_reasoning_plan(
    plan_or_payload: TaskReasoningPlan | Mapping[str, Any],
) -> TaskReasoningPlan:
    plan = (
        plan_or_payload
        if isinstance(plan_or_payload, TaskReasoningPlan)
        else TaskReasoningPlan.from_dict(plan_or_payload)
    )
    if plan.schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported task reasoning plan schema")
    if not _ID_PATTERN.fullmatch(plan.task_id):
        raise ValueError("task reasoning plan has an invalid task id")
    if plan.task_type not in _SUPPORTED_TASK_TYPES:
        raise ValueError("task reasoning plan has an unsupported task type")
    if not plan.coordinate_frame.startswith("camera:"):
        raise ValueError("task reasoning plan requires a named camera frame")
    if not math.isfinite(plan.duration_seconds) or plan.duration_seconds <= 0:
        raise ValueError("task reasoning plan duration must be finite and positive")
    if not plan.phases:
        raise ValueError("task reasoning plan requires phases")
    expected_start = 0.0
    gate_ids = {gate.gate_id for gate in plan.verification_gates}
    if len(gate_ids) != len(plan.verification_gates):
        raise ValueError("task reasoning plan gate ids must be unique")
    phase_ids: set[str] = set()
    referenced_gates: set[str] = set()
    for phase in plan.phases:
        if phase.phase_id in phase_ids:
            raise ValueError("task reasoning plan phase ids must be unique")
        phase_ids.add(phase.phase_id)
        if not math.isclose(phase.start_seconds, expected_start, abs_tol=1e-6):
            raise ValueError("task reasoning plan phases must be contiguous")
        if phase.motion_frame != plan.coordinate_frame:
            raise ValueError("task reasoning plan phases must use the declared camera frame")
        unknown = set(phase.gate_ids) - gate_ids
        if unknown:
            raise ValueError(f"task reasoning phase references unknown gates: {sorted(unknown)}")
        referenced_gates.update(phase.gate_ids)
        expected_start = phase.end_seconds
    if not math.isclose(expected_start, plan.duration_seconds, abs_tol=1e-6):
        raise ValueError("task reasoning phases must cover the full duration")
    if referenced_gates != gate_ids:
        missing = sorted(gate_ids - referenced_gates)
        raise ValueError(f"task reasoning plan has unreferenced gates: {missing}")
    if not _SHA256_PATTERN.fullmatch(plan.plan_sha256):
        raise ValueError("task reasoning plan requires a lowercase SHA-256")
    unlocked = plan.to_dict()
    expected_sha256 = str(unlocked.pop("plan_sha256"))
    actual_sha256 = _canonical_sha256(unlocked)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"task reasoning plan hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return plan


class ReasoningPluginRegistry:
    """Explicit discovery registry for lightweight or optional reasoning plugins."""

    def __init__(
        self,
        plugins: Sequence[TaskReasoningPlugin] = (
            PhysicalTaskReasoningPlugin(),
            TshirtFoldReasoningPlugin(),
        ),
    ) -> None:
        self._plugins: dict[str, TaskReasoningPlugin] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: TaskReasoningPlugin) -> None:
        descriptor = plugin.descriptor
        if not isinstance(descriptor, ReasoningPluginDescriptor):
            raise TypeError("reasoning plugin must expose ReasoningPluginDescriptor")
        if descriptor.name in self._plugins:
            raise ValueError(f"duplicate reasoning plugin {descriptor.name!r}")
        self._plugins[descriptor.name] = plugin

    def discover(self) -> None:
        entry_points = metadata.entry_points()
        selected = (
            entry_points.select(group=PLUGIN_ENTRYPOINT_GROUP)
            if hasattr(entry_points, "select")
            else entry_points.get(PLUGIN_ENTRYPOINT_GROUP, ())
        )
        for entry_point in selected:
            plugin = entry_point.load()
            if isinstance(plugin, type):
                plugin = plugin()
            self.register(plugin)

    def get(self, name: str) -> TaskReasoningPlugin:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise ValueError(f"unknown reasoning plugin {name!r}") from exc

    def descriptors(self) -> tuple[ReasoningPluginDescriptor, ...]:
        return tuple(
            sorted(
                (plugin.descriptor for plugin in self._plugins.values()),
                key=lambda item: item.name,
            )
        )
