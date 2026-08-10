"""Explicit task contracts for long-horizon flower-arranging adaptation.

This module is dependency-free by design.  GPU renderers may consume the
contracts, but importing :mod:`phiagent` never imports Torch, CUDA, SAM2, or a
video model.  Contact and flower-instance evidence use named coordinate frames
and unknown evidence is always a claim blocker.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Sequence


CAMERA_FRAME = "camera:source_pixels"
ROBOT_FRAME = "robot:base"
FLOWER_FRAME = "object:flower"


class HandSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class HandPhase(str, Enum):
    OBSERVE = "observe"
    APPROACH = "approach"
    GRASP = "grasp"
    HOLD = "hold"
    MANIPULATE = "manipulate"
    RELEASE = "release"
    RETRACT = "retract"


CONTACT_PHASES = frozenset(
    {HandPhase.GRASP, HandPhase.HOLD, HandPhase.MANIPULATE, HandPhase.RELEASE}
)


class OcclusionOrder(str, Enum):
    FLOWER_BEHIND_HAND = "flower_behind_hand"
    FLOWER_IN_FRONT_OF_HAND = "flower_in_front_of_hand"
    DEPTH_TRACK_REQUIRED = "depth_track_required"


class EvidenceKind(str, Enum):
    INSTANCE_MASK = "instance_mask"
    UNION_MASK_PROXY = "union_mask_proxy"
    MANUAL_REVIEW = "manual_review"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FlowerInstanceSpec:
    flower_id: str
    observed_frames: int
    expected_frames: int
    evidence_kind: EvidenceKind
    stable_identity: bool
    coordinate_frame: str = FLOWER_FRAME
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.flower_id.strip():
            raise ValueError("flower_id is required")
        if self.expected_frames < 1 or not 0 <= self.observed_frames <= self.expected_frames:
            raise ValueError("flower frame counts are invalid")
        if self.coordinate_frame != FLOWER_FRAME:
            raise ValueError("flower instances must use object:flower")
        if self.stable_identity and self.evidence_kind is not EvidenceKind.INSTANCE_MASK:
            raise ValueError("stable flower identity requires instance-mask evidence")
        if self.stable_identity and not self.evidence:
            raise ValueError("stable flower identity requires persisted evidence")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FlowerInstanceSpec":
        raw_evidence = payload.get("evidence", ())
        if not isinstance(raw_evidence, (list, tuple)):
            raise ValueError("flower evidence must be an array")
        return cls(
            flower_id=str(payload["flower_id"]),
            observed_frames=int(payload["observed_frames"]),
            expected_frames=int(payload["expected_frames"]),
            evidence_kind=EvidenceKind(str(payload["evidence_kind"])),
            stable_identity=bool(payload["stable_identity"]),
            coordinate_frame=str(payload.get("coordinate_frame", FLOWER_FRAME)),
            evidence=tuple(str(item) for item in raw_evidence),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["evidence_kind"] = self.evidence_kind.value
        return payload


@dataclass(frozen=True)
class ContactConstraint:
    contact_id: str
    hand: HandSide
    flower_id: str
    start_frame: int
    end_frame_exclusive: int
    phase: HandPhase
    evidence_kind: EvidenceKind
    confidence: float
    occlusion_order: OcclusionOrder
    camera_frame: str = CAMERA_FRAME
    robot_frame: str = ROBOT_FRAME
    flower_frame: str = FLOWER_FRAME
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.contact_id.strip() or not self.flower_id.strip():
            raise ValueError("contact_id and flower_id are required")
        if self.start_frame < 0 or self.end_frame_exclusive <= self.start_frame:
            raise ValueError("contact interval is invalid")
        if self.phase not in CONTACT_PHASES:
            raise ValueError("a contact constraint must use a contact phase")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("contact confidence must be finite and in [0, 1]")
        if (self.camera_frame, self.robot_frame, self.flower_frame) != (
            CAMERA_FRAME,
            ROBOT_FRAME,
            FLOWER_FRAME,
        ):
            raise ValueError("contact coordinate frames must be explicit and canonical")
        if self.evidence_kind is EvidenceKind.INSTANCE_MASK and not self.evidence:
            raise ValueError("instance contact evidence must be persisted")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ContactConstraint":
        raw_evidence = payload.get("evidence", ())
        if not isinstance(raw_evidence, (list, tuple)):
            raise ValueError("contact evidence must be an array")
        return cls(
            contact_id=str(payload["contact_id"]),
            hand=HandSide(str(payload["hand"])),
            flower_id=str(payload["flower_id"]),
            start_frame=int(payload["start_frame"]),
            end_frame_exclusive=int(payload["end_frame_exclusive"]),
            phase=HandPhase(str(payload["phase"])),
            evidence_kind=EvidenceKind(str(payload["evidence_kind"])),
            confidence=float(payload["confidence"]),
            occlusion_order=OcclusionOrder(str(payload["occlusion_order"])),
            camera_frame=str(payload.get("camera_frame", CAMERA_FRAME)),
            robot_frame=str(payload.get("robot_frame", ROBOT_FRAME)),
            flower_frame=str(payload.get("flower_frame", FLOWER_FRAME)),
            evidence=tuple(str(item) for item in raw_evidence),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["hand"] = self.hand.value
        payload["phase"] = self.phase.value
        payload["evidence_kind"] = self.evidence_kind.value
        payload["occlusion_order"] = self.occlusion_order.value
        return payload


@dataclass(frozen=True)
class BimanualPhase:
    phase_id: str
    start_frame: int
    end_frame_exclusive: int
    left_phase: HandPhase
    right_phase: HandPhase
    left_flower_id: str | None = None
    right_flower_id: str | None = None

    def __post_init__(self) -> None:
        if not self.phase_id.strip():
            raise ValueError("phase_id is required")
        if self.start_frame < 0 or self.end_frame_exclusive <= self.start_frame:
            raise ValueError("phase interval is invalid")
        for phase, flower_id in (
            (self.left_phase, self.left_flower_id),
            (self.right_phase, self.right_flower_id),
        ):
            if phase in CONTACT_PHASES and not flower_id:
                raise ValueError("contact phases require an explicit flower_id")
            if phase not in CONTACT_PHASES and flower_id is not None:
                raise ValueError("non-contact phases cannot silently retain a flower_id")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BimanualPhase":
        return cls(
            phase_id=str(payload["phase_id"]),
            start_frame=int(payload["start_frame"]),
            end_frame_exclusive=int(payload["end_frame_exclusive"]),
            left_phase=HandPhase(str(payload["left_phase"])),
            right_phase=HandPhase(str(payload["right_phase"])),
            left_flower_id=(
                None if payload.get("left_flower_id") is None else str(payload["left_flower_id"])
            ),
            right_flower_id=(
                None
                if payload.get("right_flower_id") is None
                else str(payload["right_flower_id"])
            ),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["left_phase"] = self.left_phase.value
        payload["right_phase"] = self.right_phase.value
        return payload


@dataclass(frozen=True)
class FlowerTaskContract:
    frame_count: int
    fps: float
    instances: tuple[FlowerInstanceSpec, ...]
    contacts: tuple[ContactConstraint, ...]
    phases: tuple[BimanualPhase, ...]
    adapter_checkpoint_sha256: str | None = None
    coordinate_frame: str = CAMERA_FRAME

    def __post_init__(self) -> None:
        if self.frame_count < 1 or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("task timeline is invalid")
        if self.coordinate_frame != CAMERA_FRAME:
            raise ValueError("task timeline must use camera:source_pixels")
        ids = [instance.flower_id for instance in self.instances]
        if len(ids) != len(set(ids)):
            raise ValueError("flower_id values must be unique")
        known = set(ids)
        contact_ids = [contact.contact_id for contact in self.contacts]
        if len(contact_ids) != len(set(contact_ids)):
            raise ValueError("contact_id values must be unique")
        for contact in self.contacts:
            if contact.flower_id not in known:
                raise ValueError(f"contact references unknown flower {contact.flower_id!r}")
            if contact.end_frame_exclusive > self.frame_count:
                raise ValueError("contact extends beyond the task timeline")
        if not self.phases or self.phases[0].start_frame != 0:
            raise ValueError("phases must start at frame zero")
        expected_start = 0
        for phase in self.phases:
            if phase.start_frame != expected_start:
                raise ValueError("phases must cover the timeline contiguously")
            if phase.end_frame_exclusive > self.frame_count:
                raise ValueError("phase extends beyond the task timeline")
            for flower_id in (phase.left_flower_id, phase.right_flower_id):
                if flower_id is not None and flower_id not in known:
                    raise ValueError(f"phase references unknown flower {flower_id!r}")
            expected_start = phase.end_frame_exclusive
        if expected_start != self.frame_count:
            raise ValueError("phases must cover every task frame")
        if self.adapter_checkpoint_sha256 is not None and (
            len(self.adapter_checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.adapter_checkpoint_sha256)
        ):
            raise ValueError("adapter checkpoint hash must be lowercase SHA-256")

    @property
    def claim_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.adapter_checkpoint_sha256 is None:
            blockers.append("task_adapter_checkpoint_missing")
        for instance in self.instances:
            if not instance.stable_identity:
                blockers.append(f"flower_identity_unverified:{instance.flower_id}")
            if instance.observed_frames != self.frame_count:
                blockers.append(f"flower_track_incomplete:{instance.flower_id}")
        for contact in self.contacts:
            if contact.evidence_kind is not EvidenceKind.INSTANCE_MASK:
                blockers.append(f"contact_is_proxy:{contact.contact_id}")
            if contact.confidence < 0.95:
                blockers.append(f"contact_confidence_below_0.95:{contact.contact_id}")
            if contact.occlusion_order is OcclusionOrder.DEPTH_TRACK_REQUIRED:
                blockers.append(f"occlusion_depth_unverified:{contact.contact_id}")
        required = {
            HandPhase.APPROACH,
            HandPhase.GRASP,
            HandPhase.MANIPULATE,
            HandPhase.RELEASE,
            HandPhase.RETRACT,
        }
        present = {
            phase
            for item in self.phases
            for phase in (item.left_phase, item.right_phase)
        }
        for missing in sorted(required - present, key=lambda item: item.value):
            blockers.append(f"phase_missing:{missing.value}")
        return tuple(dict.fromkeys(blockers))

    @property
    def claim_ready(self) -> bool:
        return not self.claim_blockers

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FlowerTaskContract":
        raw_instances = payload.get("instances")
        raw_contacts = payload.get("contacts")
        raw_phases = payload.get("phases")
        if not isinstance(raw_instances, list):
            raise ValueError("instances must be an array")
        if not isinstance(raw_contacts, list):
            raise ValueError("contacts must be an array")
        if not isinstance(raw_phases, list):
            raise ValueError("phases must be an array")
        return cls(
            frame_count=int(payload["frame_count"]),
            fps=float(payload["fps"]),
            instances=tuple(FlowerInstanceSpec.from_dict(item) for item in raw_instances),
            contacts=tuple(ContactConstraint.from_dict(item) for item in raw_contacts),
            phases=tuple(BimanualPhase.from_dict(item) for item in raw_phases),
            adapter_checkpoint_sha256=(
                None
                if payload.get("adapter_checkpoint_sha256") is None
                else str(payload["adapter_checkpoint_sha256"])
            ),
            coordinate_frame=str(payload.get("coordinate_frame", CAMERA_FRAME)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "coordinate_frame": self.coordinate_frame,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "adapter_checkpoint_sha256": self.adapter_checkpoint_sha256,
            "instances": [instance.to_dict() for instance in self.instances],
            "contacts": [contact.to_dict() for contact in self.contacts],
            "phases": [phase.to_dict() for phase in self.phases],
            "claim_ready": self.claim_ready,
            "claim_blockers": list(self.claim_blockers),
        }


PHASE_CANDIDATE_GATES = (
    "robot_morphology",
    "embodied_motion",
    "flower_instance_identity",
    "contact_attachment",
    "occlusion_order",
    "temporal_consistency",
)


@dataclass(frozen=True)
class PhaseCandidateEvaluation:
    phase_id: str
    candidate_id: str
    scores: Mapping[str, float]
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.phase_id.strip() or not self.candidate_id.strip():
            raise ValueError("phase_id and candidate_id are required")
        if not self.evidence:
            raise ValueError("phase candidate evaluation requires persisted evidence")
        if set(self.scores) != set(PHASE_CANDIDATE_GATES):
            raise ValueError("phase candidate scores must contain every hard gate exactly once")
        if any(
            not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0
            for score in self.scores.values()
        ):
            raise ValueError("phase candidate scores must be finite and in [0, 1]")

    def passes(self, thresholds: Mapping[str, float]) -> bool:
        if set(thresholds) != set(PHASE_CANDIDATE_GATES):
            raise ValueError("phase thresholds must contain every hard gate exactly once")
        return all(float(self.scores[name]) >= float(thresholds[name]) for name in thresholds)

    def worst_margin(self, thresholds: Mapping[str, float]) -> float:
        return min(float(self.scores[name]) - float(thresholds[name]) for name in thresholds)


def select_immutable_phase_candidates(
    evaluations: Sequence[PhaseCandidateEvaluation],
    phase_ids: Sequence[str],
    thresholds: Mapping[str, float],
) -> dict[str, str]:
    """Choose one whole candidate per phase; frame-level candidate mixing is forbidden."""

    selected: dict[str, str] = {}
    for phase_id in phase_ids:
        eligible = [
            item
            for item in evaluations
            if item.phase_id == phase_id and item.passes(thresholds)
        ]
        if not eligible:
            raise ValueError(f"no hard-gate-passing immutable candidate for phase {phase_id!r}")
        winner = max(
            eligible,
            key=lambda item: (item.worst_margin(thresholds), item.candidate_id),
        )
        selected[phase_id] = winner.candidate_id
    return selected
