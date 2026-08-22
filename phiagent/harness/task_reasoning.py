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
_MODULE_SUPPORT_STATES = {
    "tabletop_initial_face",
    "tabletop_initial_face_and_gripper",
    "tabletop_initial_face_tail_gripper",
    "table_long_edge_a_tail_gripper",
    "table_long_edge_b_tail_gripper",
    "gripper_suspended",
    "tabletop_opposite_face_and_gripper",
    "tabletop_opposite_face",
    "receptacle_guided_and_gripper",
    "visual_receptacle_hold_and_gripper",
}
_GRIPPER_CONTACT_STATES = {
    "none",
    "approaching_open",
    "approaching_tail_open",
    "bilateral_flip_grasp",
    "bilateral_tail_collar_grasp",
    "maintained_tail_collar_pivot",
    "rigid_coupled_flip_grasp",
    "open_reposition",
    "bilateral_transport_grasp",
    "maintained_transport_grasp",
}
_TSHIRT_COMMON_AUTOMATIC_REQUIRED_GATES = frozenset(
    {
        "exact_first_frame",
        "viewer_left_sleeve_length_conserved",
        "viewer_right_sleeve_length_conserved",
        "viewer_left_sleeve_folds_inward",
        "viewer_right_sleeve_folds_inward",
        "no_teleportation_or_crossfade",
        "body_fold_after_both_sleeves",
        "bundle_move_after_body_fold",
        "bundle_moves_as_one_material",
        "camera_and_background_static",
        "terminal_compact_bundle_stable",
    }
)
_TSHIRT_STRATEGY_ORDER_GATES = frozenset(
    {
        "viewer_left_fold_precedes_viewer_right_fold",
        "viewer_right_fold_precedes_viewer_left_fold",
        "both_sleeves_fold_synchronously",
    }
)
_TSHIRT_MANUAL_REQUIRED_GATES = frozenset(
    {
        "single_shirt_identity",
        "cuff_and_shoulder_identity_persistent",
        "contact_precedes_cloth_motion",
    }
)
_TSHIRT_SINGLE_GRASP_AUTOMATIC_REQUIRED_GATES: frozenset[str] = frozenset()
_TSHIRT_SINGLE_GRASP_MANUAL_REQUIRED_GATES = frozenset(
    {
        "exactly_one_grasp_event_per_active_fold",
        "fold_begins_within_bounded_latency_after_grasp",
        "gripper_rigid_identity_persistent",
    }
)
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
        if self.task_type == TSHIRT_FOLD_TASK:
            role_counts = {
                role: sum(entity.role == role for entity in self.entities)
                for role in ("manipulator", "cloth_part", "cloth_body")
            }
            if (
                role_counts["manipulator"] < 2
                or role_counts["cloth_part"] < 2
                or role_counts["cloth_body"] < 1
            ):
                raise ValueError(
                    "T-shirt folding entities require at least two manipulators, "
                    "two cloth parts, and one cloth body"
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
class PhasePhysicalContract:
    """Contact, support, and rigid-motion contract for one visual phase."""

    module_support: str
    gripper_contact: str
    allowed_module_motion: tuple[str, ...]
    forbidden_module_motion: tuple[str, ...]
    rotation_axis: str | None = None
    rotation_start_degrees: float | None = None
    rotation_end_degrees: float | None = None
    requires_continuous_gripper_contact: bool = False

    def __post_init__(self) -> None:
        if self.module_support not in _MODULE_SUPPORT_STATES:
            raise ValueError(f"unsupported module support state: {self.module_support}")
        if self.gripper_contact not in _GRIPPER_CONTACT_STATES:
            raise ValueError(f"unsupported gripper contact state: {self.gripper_contact}")
        object.__setattr__(
            self,
            "allowed_module_motion",
            _require_nonempty(self.allowed_module_motion, "allowed module motion"),
        )
        object.__setattr__(
            self,
            "forbidden_module_motion",
            _require_nonempty(self.forbidden_module_motion, "forbidden module motion"),
        )
        rotation_values = (
            self.rotation_axis,
            self.rotation_start_degrees,
            self.rotation_end_degrees,
        )
        if all(value is None for value in rotation_values):
            return
        if any(value is None for value in rotation_values):
            raise ValueError("rotation axis, start, and end must be declared together")
        assert self.rotation_axis is not None
        assert self.rotation_start_degrees is not None
        assert self.rotation_end_degrees is not None
        if not self.rotation_axis.strip():
            raise ValueError("rotation axis must be non-empty")
        if (
            not math.isfinite(self.rotation_start_degrees)
            or not math.isfinite(self.rotation_end_degrees)
            or self.rotation_end_degrees <= self.rotation_start_degrees
            or self.rotation_start_degrees < -360.0
            or self.rotation_end_degrees > 360.0
        ):
            raise ValueError("rotation interval must be finite, increasing, and bounded")
        if not self.requires_continuous_gripper_contact:
            raise ValueError("commanded rigid rotation requires continuous gripper contact")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PhasePhysicalContract:
        return cls(
            module_support=str(payload["module_support"]),
            gripper_contact=str(payload["gripper_contact"]),
            allowed_module_motion=tuple(
                str(item) for item in payload["allowed_module_motion"]
            ),
            forbidden_module_motion=tuple(
                str(item) for item in payload["forbidden_module_motion"]
            ),
            rotation_axis=(
                None if payload.get("rotation_axis") is None else str(payload["rotation_axis"])
            ),
            rotation_start_degrees=(
                None
                if payload.get("rotation_start_degrees") is None
                else float(payload["rotation_start_degrees"])
            ),
            rotation_end_degrees=(
                None
                if payload.get("rotation_end_degrees") is None
                else float(payload["rotation_end_degrees"])
            ),
            requires_continuous_gripper_contact=bool(
                payload.get("requires_continuous_gripper_contact", False)
            ),
        )


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
    physical_contract: PhasePhysicalContract | None = None

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
        if self.physical_contract is not None and not isinstance(
            self.physical_contract, PhasePhysicalContract
        ):
            raise ValueError("phase physical contract must use PhasePhysicalContract")


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
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        validation_profile: str = "current",
    ) -> TaskReasoningPlan:
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
                    physical_contract=(
                        None
                        if item.get("physical_contract") is None
                        else PhasePhysicalContract.from_dict(item["physical_contract"])
                    ),
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
        validate_task_reasoning_plan(plan, validation_profile=validation_profile)
        return plan


class TaskReasoningPlugin(Protocol):
    descriptor: ReasoningPluginDescriptor

    def analyze(self, request: TaskReasoningRequest) -> TaskReasoningPlan:
        """Expand a typed task request into a hash-bound, fail-closed phase plan."""


def _detect_language(instruction: str) -> str:
    return "zh-CN" if any("\u4e00" <= character <= "\u9fff" for character in instruction) else "en"


def _optical_flip_insert_boundaries(duration_seconds: float) -> tuple[float, ...]:
    fractions = (
        0.0,
        0.04,
        0.11,
        0.18,
        0.28,
        0.43,
        0.58,
        0.64,
        0.68,
        0.72,
        0.76,
        0.83,
        0.89,
        0.98,
        1.0,
    )
    return tuple(round(duration_seconds * fraction, 6) for fraction in fractions)


class PhysicalTaskReasoningPlugin:
    """Rule-bound planner for visually grounded manipulation generation."""

    descriptor = ReasoningPluginDescriptor(
        name="physical-task-language-planner",
        version="3.0.0",
        stage="reasoning",
        description=(
            "Expands typed language intent into table-supported tail-pivot phases, "
            "camera-frame motion constraints, and fail-closed verification gates."
        ),
        capabilities=(
            "language_analysis",
            "task_expansion",
            "physical_reasonableness",
            "tail_driven_edge_pivot_flip",
            "support_conservation",
            "table_pivot_rotation",
            OPTICAL_MODULE_TASK,
        ),
        deterministic=True,
        heavyweight=False,
    )

    def analyze(self, request: TaskReasoningRequest) -> TaskReasoningPlan:
        if request.task_type != OPTICAL_MODULE_TASK:
            raise ValueError(f"unsupported task type: {request.task_type}")
        if request.duration_seconds < 10.0:
            raise ValueError(
                "tail-pivot optical-module flip and insertion requires at least 10 seconds"
            )
        frame = request.coordinate_frame
        boundaries = _optical_flip_insert_boundaries(request.duration_seconds)
        gates = (
            VerificationGate(
                "exact_first_frame",
                "Decoded frame zero preserves the supplied first-frame appearance within the frozen codec tolerance.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "single_module_identity",
                "Exactly one rigid optical module persists through every phase.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "module_static_until_flip_grasp",
                "The table-supported module does not translate or rotate before both jaws visibly contact the rigid tail collar.",
                "automatic_proxy",
            ),
            VerificationGate(
                "open_approach_precedes_flip_grasp",
                "The open gripper reaches the rigid tail collar immediately inboard of the colored latch and pull loop before jaw closure.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "rigid_tail_collar_flip_grasp",
                "Both jaws close on opposite rigid tail-collar faces; the flexible green pull loop remains outside the jaws.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "table_edge_pivot_before_tail_arc",
                "A named long table edge is visibly loaded as the pivot before the tail begins its turning arc.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "tail_contact_persistence_during_pivot",
                "The two jaws remain closed on the same rigid tail collar with no slip or jaw-topology change throughout the pivot.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "pivot_edge_contact_persistence",
                "One long module edge remains table-supported during each quarter-turn, with exactly one controlled pivot-edge transition near edge-on.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "no_airborne_flip",
                "The module is never wholly suspended during the half-turn; table-edge or broad-face support remains visible at every instant.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "bounded_tail_arc_excursion",
                "The tracked tail follows a bounded local arc around the table pivot rather than translating toward the receptacle.",
                "automatic_proxy",
            ),
            VerificationGate(
                "slow_tail_arc_motion",
                "Median tracked speed during the two tail-pivot phases remains below 75 percent of coarse transport speed.",
                "automatic_proxy",
            ),
            VerificationGate(
                "flip_appearance_progression",
                "The table-pivot interval produces a non-static initial-face to edge-on to opposite-face appearance progression.",
                "automatic_proxy",
            ),
            VerificationGate(
                "monotonic_half_turn_edge_on_opposite_face",
                "Slow tail motion advances monotonically from 0 to 90 to 180 degrees about the table-supported long edge, with no reverse spin.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "no_table_or_gripper_interpenetration",
                "Only the named pivot edge touches the table during turning; the module and jaws never pass through the table, fixture, or receptacle.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "single_gripper_two_jaw_topology",
                "Exactly one gripper body and two attached primary jaws persist throughout approach, pivot, settle, regrasp, and insertion.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "opposite_face_support_before_release",
                "The opposite broad face is visibly supported and settled on the tabletop before the initial flip grasp opens.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "post_flip_settle_before_regrasp",
                "The flipped module remains stable on its opposite face while the open jaws reposition.",
                "automatic_proxy",
            ),
            VerificationGate(
                "post_flip_body_regrasp",
                "After the opposite face settles, both jaws regrasp the rigid metal housing before the second lift.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "lift_clearance_precedes_transport",
                "The regrasped module visibly clears the tabletop before coarse lateral transport.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "contact_persistence_during_transport",
                "The two-jaw body grasp and module-to-gripper relative pose persist throughout lift, transport, alignment, and insertion.",
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
                "Insertion follows the module long axis without lateral sweeping, pitch, yaw, or a new roll.",
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
                (
                    "exact_first_frame",
                    "single_module_identity",
                    "camera_and_background_static",
                ),
                PhasePhysicalContract(
                    module_support="tabletop_initial_face",
                    gripper_contact="none",
                    allowed_module_motion=("remain fully stationary",),
                    forbidden_module_motion=(
                        "translation",
                        "rotation",
                        "deformation",
                        "duplication",
                    ),
                ),
            ),
            TaskPhase(
                "approach_tail_grasp_pose",
                boundaries[1],
                boundaries[2],
                "Move the open jaws around the rigid tail collar without loading the pull loop.",
                (
                    "Approach from the right with both jaws open. Bracket opposite rigid tail-collar "
                    "faces immediately inboard of the blue latch and green pull loop. Keep the module "
                    "fully table-supported and motionless; do not pinch or hook the flexible loop."
                ),
                frame,
                "fine",
                ("The module is resting on its initial broad face.", "The jaws are open."),
                ("The module and receptacle remain fixed.", "The flexible pull loop is untouched."),
                ("Both jaw faces bracket the rigid tail collar before closure.",),
                ("open_approach_precedes_flip_grasp", "module_static_until_flip_grasp"),
                PhasePhysicalContract(
                    module_support="tabletop_initial_face",
                    gripper_contact="approaching_tail_open",
                    allowed_module_motion=("remain fully stationary",),
                    forbidden_module_motion=(
                        "translation before contact",
                        "rotation before contact",
                        "pull-loop deformation",
                    ),
                ),
            ),
            TaskPhase(
                "close_tail_collar_grasp",
                boundaries[2],
                boundaries[3],
                "Establish a symmetric two-jaw pinch on the rigid tail collar.",
                (
                    "Close both jaws slowly on opposite rigid tail-collar faces just inside the "
                    "colored end. Build preload without moving the table-supported module. Keep "
                    "the green pull loop outside the jaws and visibly undeformed."
                ),
                frame,
                "slow",
                ("Both jaw faces bracket the rigid tail collar.",),
                ("The module remains table-supported until visible two-sided contact.",),
                ("A stable bilateral rigid-tail grasp is established.",),
                ("rigid_tail_collar_flip_grasp", "module_static_until_flip_grasp"),
                PhasePhysicalContract(
                    module_support="tabletop_initial_face_tail_gripper",
                    gripper_contact="bilateral_tail_collar_grasp",
                    allowed_module_motion=("jaw preload only after two-sided contact",),
                    forbidden_module_motion=(
                        "object translation before bilateral contact",
                        "object rotation before bilateral contact",
                        "flexible pull-loop grasp",
                    ),
                    requires_continuous_gripper_contact=True,
                ),
            ),
            TaskPhase(
                "establish_table_edge_pivot",
                boundaries[3],
                boundaries[4],
                "Preload the tail and establish one long table edge as the turning pivot.",
                (
                    "Keep the tail collar pinched at low height. Move the tail through a small, "
                    "slow upward-and-sideways arc until one long module edge becomes the sole line "
                    "pivot. Do not lift the whole module clear of the tabletop."
                ),
                frame,
                "slow",
                ("A stable bilateral rigid-tail grasp exists.",),
                (
                    "The initial broad face remains supported until the named long edge is loaded.",
                    "No receptacle-directed translation and no loss of tail contact.",
                ),
                ("One named long edge is visibly established as the table pivot.",),
                (
                    "table_edge_pivot_before_tail_arc",
                    "no_airborne_flip",
                    "single_module_identity",
                ),
                PhasePhysicalContract(
                    module_support="tabletop_initial_face_tail_gripper",
                    gripper_contact="maintained_tail_collar_pivot",
                    allowed_module_motion=(
                        "small tail arc that transfers support from the broad face to long edge A",
                    ),
                    forbidden_module_motion=(
                        "whole-module airborne lift",
                        "translation toward the receptacle",
                        "tail slip inside the jaws",
                        "uncontrolled table slide",
                    ),
                    rotation_axis="table_supported_module_long_edge_a",
                    rotation_start_degrees=0.0,
                    rotation_end_degrees=15.0,
                    requires_continuous_gripper_contact=True,
                ),
            ),
            TaskPhase(
                "tail_arc_to_edge_on",
                boundaries[4],
                boundaries[5],
                "Slowly drive the tail around long edge A until the module is edge-on.",
                (
                    "Maintain the rigid tail pinch and move the tail through a slow monotonic arc. "
                    "Keep long edge A in table contact while the opposite broad face rises. Reach "
                    "one clear edge-on state without free-space roll, slip, or reverse motion."
                ),
                frame,
                "slow",
                ("The tail collar is pinched and long edge A is loaded against the table.",),
                (
                    "Continuous two-jaw tail contact and continuous long-edge-A table support.",
                    "No airborne interval, receptacle-directed translation, or table slide.",
                ),
                ("The same module reaches one visible edge-on 90-degree state.",),
                (
                    "tail_contact_persistence_during_pivot",
                    "pivot_edge_contact_persistence",
                    "no_airborne_flip",
                    "bounded_tail_arc_excursion",
                    "slow_tail_arc_motion",
                    "flip_appearance_progression",
                    "monotonic_half_turn_edge_on_opposite_face",
                    "no_table_or_gripper_interpenetration",
                    "single_gripper_two_jaw_topology",
                    "single_module_identity",
                ),
                PhasePhysicalContract(
                    module_support="table_long_edge_a_tail_gripper",
                    gripper_contact="maintained_tail_collar_pivot",
                    allowed_module_motion=(
                        "slow monotonic table-edge pivot driven by the tail arc",
                        "bounded center arc implied by rolling on long edge A",
                    ),
                    forbidden_module_motion=(
                        "free flight",
                        "tail slip",
                        "reverse rotation",
                        "table penetration",
                        "pivot-edge loss",
                        "translation toward the receptacle",
                    ),
                    rotation_axis="table_supported_module_long_edge_a",
                    rotation_start_degrees=15.0,
                    rotation_end_degrees=90.0,
                    requires_continuous_gripper_contact=True,
                ),
            ),
            TaskPhase(
                "tail_arc_to_opposite_face",
                boundaries[5],
                boundaries[6],
                "Continue the slow tail arc around long edge B toward the opposite face.",
                (
                    "At edge-on, transfer table contact once from long edge A to adjacent long "
                    "edge B. Continue moving the pinched tail slowly from 90 to 180 degrees. "
                    "Keep edge B supported and lower the opposite face without dropping it."
                ),
                frame,
                "slow",
                ("The same module is edge-on with continuous tail contact.",),
                (
                    "Exactly one controlled pivot-edge transition, then persistent edge-B support.",
                    "No free flight, tail slip, reverse spin, or receptacle-directed transport.",
                ),
                ("The opposite broad face reaches the tabletop under tail control.",),
                (
                    "tail_contact_persistence_during_pivot",
                    "pivot_edge_contact_persistence",
                    "no_airborne_flip",
                    "bounded_tail_arc_excursion",
                    "slow_tail_arc_motion",
                    "flip_appearance_progression",
                    "monotonic_half_turn_edge_on_opposite_face",
                    "no_table_or_gripper_interpenetration",
                    "single_gripper_two_jaw_topology",
                    "single_module_identity",
                ),
                PhasePhysicalContract(
                    module_support="table_long_edge_b_tail_gripper",
                    gripper_contact="maintained_tail_collar_pivot",
                    allowed_module_motion=(
                        "slow monotonic table-edge pivot driven by the same tail pinch",
                        "bounded center arc implied by rolling on long edge B",
                    ),
                    forbidden_module_motion=(
                        "free flight",
                        "tail slip",
                        "reverse rotation",
                        "table penetration",
                        "second pivot-edge transition",
                        "translation toward the receptacle",
                    ),
                    rotation_axis="table_supported_module_long_edge_b",
                    rotation_start_degrees=90.0,
                    rotation_end_degrees=180.0,
                    requires_continuous_gripper_contact=True,
                ),
            ),
            TaskPhase(
                "settle_opposite_face_under_tail_control",
                boundaries[6],
                boundaries[7],
                "Finish the tail arc and hold the opposite face motionless before release.",
                (
                    "Keep both jaws closed on the tail collar while the opposite broad face makes "
                    "full table contact. Decelerate the last few degrees, remove rocking, and hold "
                    "the module motionless before opening the jaws."
                ),
                frame,
                "slow",
                ("The opposite broad face has just contacted the tabletop under tail control.",),
                (
                    "No new roll, lateral slide, bounce, or early release.",
                    "The tail pinch remains closed until broad-face support is stable.",
                ),
                ("The opposite broad face is supported and motionless on the tabletop.",),
                (
                    "no_table_or_gripper_interpenetration",
                    "opposite_face_support_before_release",
                    "post_flip_settle_before_regrasp",
                ),
                PhasePhysicalContract(
                    module_support="tabletop_opposite_face_and_gripper",
                    gripper_contact="maintained_tail_collar_pivot",
                    allowed_module_motion=("damped final settling onto opposite-face support",),
                    forbidden_module_motion=(
                        "release before support",
                        "bounce",
                        "lateral slide",
                        "additional rotation",
                    ),
                    requires_continuous_gripper_contact=True,
                ),
            ),
            TaskPhase(
                "release_and_reposition_after_flip",
                boundaries[7],
                boundaries[8],
                "Open only after support, then reposition around the flipped metal housing.",
                (
                    "After the opposite face is visibly settled, open both jaws without moving "
                    "the module. Reposition the open jaws around opposite metal side faces near "
                    "the housing center. Keep the green pull tab untouched."
                ),
                frame,
                "fine",
                ("The flipped module is motionless on its opposite face.",),
                (
                    "The tabletop alone supports the module while the jaws are open.",
                    "No module translation or rotation during repositioning.",
                ),
                ("The open jaws bracket the flipped metal housing for a transport grasp.",),
                ("opposite_face_support_before_release", "post_flip_settle_before_regrasp"),
                PhasePhysicalContract(
                    module_support="tabletop_opposite_face",
                    gripper_contact="open_reposition",
                    allowed_module_motion=("remain fully stationary",),
                    forbidden_module_motion=(
                        "motion while unsupported",
                        "translation during jaw repositioning",
                        "rotation during jaw repositioning",
                        "pull-tab contact",
                    ),
                ),
            ),
            TaskPhase(
                "regrasp_flipped_metal_body",
                boundaries[8],
                boundaries[9],
                "Establish a new central body grasp on the settled opposite-face module.",
                (
                    "Close both jaws symmetrically on the rigid metal housing of the settled "
                    "flipped module. Do not move it until bilateral contact is visible. Do not "
                    "grasp the green pull tab."
                ),
                frame,
                "slow",
                ("The open jaws bracket the flipped metal housing.",),
                ("The tabletop supports the module until two-sided contact is established.",),
                ("A stable bilateral transport grasp exists on the metal housing.",),
                ("post_flip_body_regrasp", "post_flip_settle_before_regrasp"),
                PhasePhysicalContract(
                    module_support="tabletop_opposite_face_and_gripper",
                    gripper_contact="bilateral_transport_grasp",
                    allowed_module_motion=("jaw preload after bilateral body contact",),
                    forbidden_module_motion=(
                        "translation before bilateral contact",
                        "rotation before bilateral contact",
                        "pull-tab grasp",
                    ),
                    requires_continuous_gripper_contact=True,
                ),
            ),
            TaskPhase(
                "lift_for_transport_clearance",
                boundaries[9],
                boundaries[10],
                "Lift the regrasped flipped module clear before transport.",
                (
                    "Lift the securely regrasped opposite-face module a small visible distance "
                    "away from the tabletop. Keep the new body grasp and 180-degree orientation "
                    "fixed. Do not start upper-left transport before clearance."
                ),
                frame,
                "slow",
                ("A stable post-flip bilateral body grasp exists.",),
                (
                    "No lateral transport before visible clearance.",
                    "No relative gripper-module slip or new rotation.",
                ),
                ("The flipped module visibly clears the tabletop.",),
                (
                    "lift_clearance_precedes_transport",
                    "contact_persistence_during_transport",
                    "single_module_identity",
                ),
                PhasePhysicalContract(
                    module_support="gripper_suspended",
                    gripper_contact="maintained_transport_grasp",
                    allowed_module_motion=("small clearance lift normal to the tabletop",),
                    forbidden_module_motion=(
                        "lateral transport before clearance",
                        "relative slip",
                        "new rotation",
                    ),
                    requires_continuous_gripper_contact=True,
                ),
            ),
            TaskPhase(
                "coarse_transport_to_standoff",
                boundaries[10],
                boundaries[11],
                "Carry the flipped, lifted module near the receptacle without inserting.",
                (
                    "Carry the lifted flipped module smoothly diagonally upper-left toward the "
                    "fixed receptacle. Keep the connector end leading and the body grasp rigid. "
                    "Stop outside the mouth with a visible standoff; do not insert yet."
                ),
                frame,
                "coarse",
                ("The flipped module has visible tabletop clearance.",),
                (
                    "The bilateral grasp and gripper-module relative pose remain fixed.",
                    "No insertion, release, or new roll during coarse transport.",
                ),
                ("The connector stops outside the receptacle at pre-insertion standoff.",),
                (
                    "preinsert_standoff",
                    "contact_persistence_during_transport",
                    "single_module_identity",
                ),
                PhasePhysicalContract(
                    module_support="gripper_suspended",
                    gripper_contact="maintained_transport_grasp",
                    allowed_module_motion=("smooth free-space translation to standoff",),
                    forbidden_module_motion=(
                        "insertion before alignment",
                        "relative slip",
                        "new rotation",
                        "release",
                    ),
                    requires_continuous_gripper_contact=True,
                ),
            ),
            TaskPhase(
                "coaxial_preinsert_alignment",
                boundaries[11],
                boundaries[12],
                "Align connector height, centerline, and long axis before inward motion.",
                (
                    "Pause the coarse advance. Use only small fine adjustments to place the "
                    "connector at the same apparent height as the receptacle mouth, center the "
                    "connector on the mouth, and make the module long axis collinear with the "
                    "insertion axis. Hold a short settling dwell outside the slot."
                ),
                frame,
                "fine",
                ("The connector is stopped at visible standoff.",),
                (
                    "No inward insertion before height, centerline, and axis alignment.",
                    "The body grasp and flipped orientation remain unchanged.",
                ),
                ("The connector is level, centered, coaxial, and settled outside the mouth.",),
                (
                    "coaxial_and_level_before_insertion",
                    "preinsert_standoff",
                    "contact_persistence_during_transport",
                ),
                PhasePhysicalContract(
                    module_support="gripper_suspended",
                    gripper_contact="maintained_transport_grasp",
                    allowed_module_motion=("small alignment corrections outside the slot",),
                    forbidden_module_motion=(
                        "inward insertion before settling",
                        "lateral sweep across the receptacle rim",
                        "new rotation",
                        "release",
                    ),
                    requires_continuous_gripper_contact=True,
                ),
            ),
            TaskPhase(
                "slow_axial_insertion",
                boundaries[12],
                boundaries[13],
                "Insert slowly along the aligned module axis.",
                (
                    "Insert the connector slowly and incrementally straight along the aligned "
                    "module long axis. Use no more than 35 percent of the coarse transport speed. "
                    "Do not sweep sideways, pitch, yaw, roll, or push before alignment. Let the "
                    "receptacle occlude the connector naturally as the same module enters."
                ),
                frame,
                "slow",
                ("The connector is level, centered, coaxial, and settled.",),
                (
                    "Axial motion only with continuous body grasp.",
                    "No lateral sweep, new rotation, identity change, or receptacle motion.",
                ),
                ("The module reaches a visibly inserted terminal state.",),
                (
                    "insertion_slower_than_transport",
                    "axial_insertion_without_sweep",
                    "contact_persistence_during_transport",
                    "single_module_identity",
                ),
                PhasePhysicalContract(
                    module_support="receptacle_guided_and_gripper",
                    gripper_contact="maintained_transport_grasp",
                    allowed_module_motion=("slow translation along the aligned insertion axis",),
                    forbidden_module_motion=(
                        "lateral sweep",
                        "pitch",
                        "yaw",
                        "roll",
                        "receptacle penetration outside the mouth",
                        "release",
                    ),
                    requires_continuous_gripper_contact=True,
                ),
            ),
            TaskPhase(
                "seated_hold_without_release",
                boundaries[13],
                boundaries[14],
                "Stop and hold the visually inserted state without claiming physical seating.",
                (
                    "Stop all inward motion and hold the final inserted state motionless. Keep "
                    "both jaws closed on the metal housing because no force, calibrated depth, "
                    "or electrical seating signal is available; do not release or retract."
                ),
                frame,
                "stationary",
                ("The same flipped module is visibly inserted.",),
                ("No release, rebound, drift, rotation, or receptacle motion.",),
                ("The final inserted visual state remains stable through the last frame.",),
                (
                    "terminal_seated_hold",
                    "no_release_without_seating_evidence",
                    "camera_and_background_static",
                ),
                PhasePhysicalContract(
                    module_support="visual_receptacle_hold_and_gripper",
                    gripper_contact="maintained_transport_grasp",
                    allowed_module_motion=("remain fully stationary",),
                    forbidden_module_motion=(
                        "release",
                        "rebound",
                        "drift",
                        "rotation",
                        "receptacle motion",
                    ),
                    requires_continuous_gripper_contact=True,
                ),
            ),
        )
        language = LanguageAnalysis(
            source_language=_detect_language(request.instruction),
            normalized_instruction=(
                "Use the open two-jaw gripper to pinch the rigid tail collar immediately inboard "
                "of the colored latch and flexible pull loop. Keep the module on the tabletop, "
                "establish one long edge as a pivot, and move the tail through one slow monotonic "
                "arc from the initial face through edge-on to the opposite face. Hold the settled "
                "opposite face before release, regrasp the metal housing, lift and transport to a "
                "standoff, align level and coaxial, insert slowly, and hold without release."
            ),
            ordered_actions=tuple(phase.phase_id for phase in phases),
            spatial_relations=(
                "open jaws bracket opposite rigid tail-collar faces before closure",
                "the flexible pull loop remains outside the jaws",
                "one long module edge remains the table pivot during each quarter-turn",
                "the gripper drives a bounded tail arc while the module remains table-supported",
                "the opposite broad face settles on the tabletop before release and regrasp",
                "connector stops outside the receptacle at pre-insertion standoff",
                "connector and receptacle share apparent height and centerline before insertion",
                "insertion follows the module long axis",
            ),
            temporal_modifiers=(
                "tail approach before rigid tail-collar closure",
                "pivot-edge establishment before the tail arc",
                "slow 15-to-90 edge-on pivot before the 90-to-180 opposite-face pivot",
                "opposite-face support before release",
                "post-flip regrasp before the second lift",
                "clearance before transport",
                "alignment and settling before insertion",
                "insertion slower than coarse transport",
                "hold without release after visual insertion",
            ),
            ambiguity_resolutions=(
                (
                    "The requested turn is interpreted as a table-supported long-edge pivot driven "
                    "by a tail arc, not a free-space wrist roll; the single camera still does not "
                    "provide calibrated angle, contact force, or robot commands."
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
                "rigid_tail_collar_is_flip_grasp_surface",
                "affordance",
                "OBSERVED",
                "The rigid tail collar immediately inboard of the colored end provides opposing jaw faces; the green pull loop is flexible.",
                "Pinching the flexible loop can deform or detach it and gives an ill-defined pivot input.",
                "Require bilateral tail-collar contact while keeping the flexible loop outside both jaws.",
            ),
            ReasoningFinding(
                "flip_requires_table_edge_pivot",
                "support_reasoning",
                "INFERRED",
                "A rectangular module can turn realistically by rocking around one long table edge and transferring once to the adjacent long edge near edge-on.",
                "Lifting the whole module and spinning it removes the reaction support needed for a realistic拨动 action.",
                "Keep a named table pivot edge throughout both quarter-turns and prohibit an airborne interval.",
            ),
            ReasoningFinding(
                "tail_arc_must_drive_pivot",
                "contact_causality",
                "INFERRED",
                "The jaws must remain on one rigid tail site while the gripper moves that site along a bounded arc around the table pivot.",
                "A body-wide image rotation without a corresponding tail path, jaw articulation, and table reaction is non-causal.",
                "Require persistent tail contact, continuous two-jaw topology, and monotonic tail motion through both pivot phases.",
            ),
            ReasoningFinding(
                "edge_on_midpoint_disambiguates_half_turn",
                "rotation_reasoning",
                "INFERRED",
                "A physically continuous table roll passes through one edge-on 90-degree state before exposing the opposite face.",
                "A direct initial-to-opposite-face crossfade does not establish rotation.",
                "Require monotonic appearance progression and a native-resolution edge-on midpoint review.",
            ),
            ReasoningFinding(
                "support_before_release",
                "support_conservation",
                "INFERRED",
                "The module needs table support before the flip grasp can open for repositioning.",
                "Opening while suspended would create unsupported free fall.",
                "Decelerate and settle the opposite broad face under tail control before release, then regrasp while table-supported.",
            ),
            ReasoningFinding(
                "post_flip_regrasp_precedes_transport",
                "grasp_precondition",
                "INFERRED",
                "A new bilateral body grasp is required after the module settles on its opposite face.",
                "Transport beginning during open-jaw repositioning would be unsupported and non-causal.",
                "Separate release/reposition, body regrasp, and second clearance lift.",
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
                "Preserve one gripper body and exactly two attached primary jaws throughout every phase.",
                "The module moves only after visible two-sided rigid-tail-collar contact.",
                "The flexible pull loop stays outside the jaws and carries no modeled load.",
                "The module is supported by the tabletop, gripper, receptacle, or an explicit combination at every instant; no unsupported free flight.",
                "The flip remains table-supported: long edge A supports the first quarter-turn and long edge B supports the second.",
                "The tail contact follows one slow bounded arc; free-space wrist spin and broad-face crossfade are forbidden.",
                "The initial flip grasp opens only after the opposite broad face is visibly supported and settled.",
                "The post-flip transport starts only after a new bilateral metal-body grasp and a second clearance lift.",
                "The camera, fixture, table, cables, lighting, and background remain fixed.",
                "No phase may imply force, tactile, electrical, or calibrated depth evidence.",
            ],
            "verification_gates": [asdict(item) for item in gates],
            "claim_boundary": (
                "This plan constrains a language-conditioned visual future with explicit contact, "
                "table-pivot support, and tail-driven rigid-body rotation intent. Apparent edge "
                "contact, tail arc, edge-on rotation, "
                "opposite-face support, alignment, and insertion remain camera-frame evidence, "
                "not calibrated 3-D geometry, friction or force validation, collision checking, "
                "robot wrist commands, electrical mating, or physical success."
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
        version="1.2.0",
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
                "viewer_left_sleeve_folds_inward",
                "The viewer-left sleeve moves a frozen minimum distance toward the original torso center during its assigned fold window.",
                "automatic_proxy",
            ),
            VerificationGate(
                "viewer_right_sleeve_folds_inward",
                "The viewer-right sleeve moves a frozen minimum distance toward the original torso center during its assigned fold window.",
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
                "exactly_one_grasp_event_per_active_fold",
                "Each active folding maneuver contains exactly one visible open-to-closed jaw transition, with no probing close, reopen, or second close.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "fold_begins_within_bounded_latency_after_grasp",
                "Visible non-rigid folding begins one to six native frames after the first stable closed contact.",
                "native_resolution_human_review",
            ),
            VerificationGate(
                "gripper_rigid_identity_persistent",
                "Each gripper preserves one palm, two rigid jaws, dark outer faces, silver inner faces, fixed jaw dimensions, joints, and wrist connection throughout the action.",
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
                "automatic_proxy",
            ),
            VerificationGate(
                "bundle_move_after_body_fold",
                "The compact folded bundle moves viewer-left only after the body fold is complete.",
                "automatic_proxy",
            ),
            VerificationGate(
                "bundle_moves_as_one_material",
                "Both sleeves and the torso undergo one coherent viewer-left transport with bounded component disagreement.",
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
                "automatic_proxy",
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
                    "the shoulder/body junction while the other closes exactly once on the cuff. "
                    "Never probe, reopen, or close a second time."
                ),
                frame,
                "fine",
                ("The viewer-left sleeve is flat and motionless.",),
                ("No cloth motion before two visible contacts; the viewer-right sleeve stays still.",),
                ("Stable shoulder-side and cuff-side contacts are visible.",),
                (
                    "contact_precedes_cloth_motion",
                    "cuff_and_shoulder_identity_persistent",
                    "exactly_one_grasp_event_per_active_fold",
                    "gripper_rigid_identity_persistent",
                ),
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
                    "viewer_left_sleeve_folds_inward",
                    "contact_precedes_cloth_motion",
                    "fold_begins_within_bounded_latency_after_grasp",
                    "gripper_rigid_identity_persistent",
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
                    "stabilize its shoulder/body junction and close exactly once on the cuff before motion. "
                    "Never probe, reopen, or close a second time."
                ),
                frame,
                "fine",
                ("The viewer-left sleeve is settled and the viewer-right sleeve is still flat.",),
                ("No right-sleeve motion before two visible contacts.",),
                ("Stable shoulder-side and cuff-side contacts are visible on the right sleeve.",),
                (
                    "contact_precedes_cloth_motion",
                    "viewer_left_fold_precedes_viewer_right_fold",
                    "exactly_one_grasp_event_per_active_fold",
                    "gripper_rigid_identity_persistent",
                ),
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
                    "viewer_right_sleeve_folds_inward",
                    "cuff_and_shoulder_identity_persistent",
                    "fold_begins_within_bounded_latency_after_grasp",
                    "gripper_rigid_identity_persistent",
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
                    "bundle_moves_as_one_material",
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
                "one close event before cloth motion and folding within six frames of stable grasp",
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
                "For each active fold, close the jaws exactly once, never reopen, and begin folding one to six native frames after stable contact.",
                "Every gripper keeps one palm, two fixed-dimension rigid jaws, dark and silver faces, joints, and wrist connection throughout the timeline.",
                "Each sleeve must move inward by the frozen task displacement and the first sleeve must settle before the second moves.",
                "During final transport, both sleeves and the torso must move viewer-left as one coherent material bundle.",
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
    *,
    validation_profile: str = "current",
) -> TaskReasoningPlan:
    if validation_profile not in {"current", "optical_epoch_v1"}:
        raise ValueError(
            f"unsupported task reasoning validation profile: {validation_profile}"
        )
    plan = (
        plan_or_payload
        if isinstance(plan_or_payload, TaskReasoningPlan)
        else TaskReasoningPlan.from_dict(
            plan_or_payload,
            validation_profile=validation_profile,
        )
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
    if plan.task_type == OPTICAL_MODULE_TASK:
        if any(phase.physical_contract is None for phase in plan.phases):
            raise ValueError("every optical-module phase requires a physical contract")
        roll_intervals = [
            (
                phase.physical_contract.rotation_start_degrees,
                phase.physical_contract.rotation_end_degrees,
            )
            for phase in plan.phases
            if phase.physical_contract is not None
            and phase.physical_contract.rotation_axis is not None
        ]
        expected_roll_intervals = (
            [(0.0, 90.0), (90.0, 180.0)]
            if validation_profile == "optical_epoch_v1"
            else [(0.0, 15.0), (15.0, 90.0), (90.0, 180.0)]
        )
        if roll_intervals != expected_roll_intervals:
            raise ValueError(
                "optical-module rotation phases do not match validation profile "
                f"{validation_profile}: expected {expected_roll_intervals}"
            )
    if plan.task_type == TSHIRT_FOLD_TASK:
        phase_positions = {
            phase.phase_id: index for index, phase in enumerate(plan.phases)
        }
        if "fold_both_sleeves_synchronously" in phase_positions:
            order_gate = "both_sleeves_fold_synchronously"
            strategy_phases = (
                "initial_state_hold",
                "establish_bilateral_cuff_contacts",
                "fold_both_sleeves_synchronously",
                "settle_both_sleeves",
            )
        else:
            sleeve_phase_ids = (
                "fold_viewer_left_sleeve",
                "fold_viewer_right_sleeve",
            )
            if any(phase_id not in phase_positions for phase_id in sleeve_phase_ids):
                raise ValueError(
                    "T-shirt reasoning plan must fold both sleeves or use the "
                    "declared simultaneous fold phase"
                )
            first, second = sorted(
                ("viewer_left", "viewer_right"),
                key=lambda side: phase_positions[f"fold_{side}_sleeve"],
            )
            order_gate = f"{first}_fold_precedes_{second}_fold"
            strategy_phases = (
                "initial_state_hold",
                f"establish_{first}_two_point_contact",
                f"fold_{first}_sleeve",
                f"settle_{first}_sleeve",
                f"establish_{second}_two_point_contact",
                f"fold_{second}_sleeve",
            )
        placement_phases = tuple(
            phase_id
            for phase_id in (
                "move_folded_bundle_viewer_left",
                "move_folded_bundle_viewer_right",
            )
            if phase_id in phase_positions
        )
        if len(placement_phases) != 1:
            raise ValueError(
                "T-shirt reasoning plan must declare exactly one viewer-relative "
                "terminal bundle placement"
            )
        declared_order_gates = gate_ids & _TSHIRT_STRATEGY_ORDER_GATES
        if declared_order_gates != {order_gate}:
            raise ValueError(
                "T-shirt reasoning plan order gate does not match its phase strategy: "
                f"expected {order_gate}, got {sorted(declared_order_gates)}"
            )
        single_grasp_epoch = (
            plan.plugin.name == "tshirt-fold-physical-language-planner"
            and plan.plugin.version == "1.2.0"
        )
        required_automatic_gates = (
            _TSHIRT_COMMON_AUTOMATIC_REQUIRED_GATES
            | {order_gate}
            | (
                _TSHIRT_SINGLE_GRASP_AUTOMATIC_REQUIRED_GATES
                if single_grasp_epoch
                else frozenset()
            )
        )
        required_manual_gates = (
            _TSHIRT_MANUAL_REQUIRED_GATES
            | (
                _TSHIRT_SINGLE_GRASP_MANUAL_REQUIRED_GATES
                if single_grasp_epoch
                else frozenset()
            )
        )
        required_gates = required_automatic_gates | required_manual_gates
        missing_gates = sorted(required_gates - gate_ids)
        if missing_gates:
            raise ValueError(
                f"T-shirt reasoning plan is missing required hard gates: {missing_gates}"
            )
        evidence_by_gate = {
            gate.gate_id: gate.evidence_source for gate in plan.verification_gates
        }
        wrong_automatic_sources = sorted(
            gate_id
            for gate_id in required_automatic_gates
            if evidence_by_gate[gate_id] != "automatic_proxy"
        )
        wrong_manual_sources = sorted(
            gate_id
            for gate_id in required_manual_gates
            if evidence_by_gate[gate_id] != "native_resolution_human_review"
        )
        if wrong_automatic_sources or wrong_manual_sources:
            raise ValueError(
                "T-shirt reasoning plan changed required gate evidence sources: "
                f"automatic={wrong_automatic_sources}, manual={wrong_manual_sources}"
            )
        required_phase_order = (
            *strategy_phases,
            "fold_body_bottom_to_top",
            "compress_bundle_without_stretch",
            placement_phases[0],
            "terminal_bundle_hold",
        )
        missing_phases = [
            phase_id
            for phase_id in required_phase_order
            if phase_id not in phase_positions
        ]
        if missing_phases:
            raise ValueError(
                f"T-shirt reasoning plan is missing required phases: {missing_phases}"
            )
        ordered_positions = [
            phase_positions[phase_id] for phase_id in required_phase_order
        ]
        if ordered_positions != sorted(ordered_positions):
            raise ValueError(
                "T-shirt reasoning plan must keep its declared sleeve strategy, body fold, "
                "bundle transport, and terminal hold in causal order"
            )
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


def rebind_task_reasoning_plan_camera_frame(
    plan_or_payload: TaskReasoningPlan | Mapping[str, Any],
    target_frame: str,
) -> TaskReasoningPlan:
    """Rebind a qualitative phase plan to an explicitly named camera frame.

    Optical-module phase plans contain time, contact, support, and qualitative
    motion contracts rather than pixel coordinates. Numeric image geometry is
    transformed and stored separately by the renderer-specific bundle compiler.
    """

    plan = validate_task_reasoning_plan(plan_or_payload)
    target = str(target_frame).strip()
    if not target.startswith("camera:"):
        raise ValueError("target reasoning frame must be a named camera frame")
    if target == plan.coordinate_frame:
        return plan
    unlocked = plan.to_dict()
    unlocked.pop("plan_sha256")
    unlocked["coordinate_frame"] = target
    for phase in unlocked["phases"]:
        phase["motion_frame"] = target
    return TaskReasoningPlan.from_dict(
        {**unlocked, "plan_sha256": _canonical_sha256(unlocked)}
    )


def validate_task_reasoning_human_review(
    plan_or_payload: TaskReasoningPlan | Mapping[str, Any],
    review: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate a hash-bound, per-gate native-resolution review."""

    plan = validate_task_reasoning_plan(plan_or_payload)
    if review.get("plan_sha256") != plan.plan_sha256:
        raise ValueError("human review plan_sha256 does not match the reasoning plan")
    gate_results = review.get("hard_gate_results")
    if not isinstance(gate_results, Mapping):
        raise ValueError("human review requires a hard_gate_results object")
    required_gate_ids = {
        gate.gate_id
        for gate in plan.verification_gates
        if gate.evidence_source == "native_resolution_human_review"
    }
    if set(gate_results) != required_gate_ids:
        raise ValueError(
            "human review hard_gate_results must exactly cover native-resolution gates"
        )
    if any(type(value) is not bool for value in gate_results.values()):
        raise ValueError("human review hard-gate decisions must be boolean")
    if type(review.get("passed")) is not bool:
        raise ValueError("human review requires a boolean passed field")
    failed = tuple(
        sorted(gate_id for gate_id, passed in gate_results.items() if not passed)
    )
    if review["passed"] is not (not failed):
        raise ValueError("human review passed field disagrees with hard_gate_results")
    return failed


class ReasoningPluginRegistry:
    """Explicit discovery registry for lightweight or optional reasoning plugins."""

    def __init__(
        self,
        plugins: Sequence[TaskReasoningPlugin] | None = None,
    ) -> None:
        if plugins is None:
            from .tshirt_cooperative_repair import TshirtCooperativeJoyAIRepairPlugin
            from .tshirt_fold_strategy import TshirtFoldStrategyReasoningPlugin

            plugins = (
                PhysicalTaskReasoningPlugin(),
                TshirtFoldReasoningPlugin(),
                TshirtFoldStrategyReasoningPlugin(),
                TshirtCooperativeJoyAIRepairPlugin(),
            )
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
