"""Hash-bound positive-reference conditioning for T-shirt video proposals.

User-approved visual examples are useful proposal priors, but they are not
automatic physical evidence. This module keeps that distinction explicit: a
reference may guide contact cadence and visible choreography while every gate
from the strategy-specific reasoning plan remains non-overridable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from .task_reasoning import TaskReasoningPlan
from .tshirt_fold_strategy import (
    LEFT_THEN_RIGHT,
    RIGHT_THEN_LEFT,
    SIMULTANEOUS,
    TshirtFoldStrategy,
    strategy_from_plan,
)


VISUAL_QUALITY_REVIEW = "visual_generation_quality"
CAMERA_PIXEL_REFERENCE = "camera_pixel_reference_only"
COMPACT_IN_PLACE = "compact_in_place"

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SLEEVE_ORDERS = {LEFT_THEN_RIGHT, RIGHT_THEN_LEFT, SIMULTANEOUS}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonempty_unique(values: Sequence[str], field: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{field} must contain non-empty values")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class PositiveFoldReference:
    """One reviewed visual proposal and its deliberately narrow evidence scope."""

    reference_id: str
    video_path: str
    video_sha256: str
    coordinate_frame: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    sleeve_order: str
    terminal_behavior: str
    review_scope: str
    review_state: str
    observed_strengths: tuple[str, ...]
    unavailable_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.reference_id):
            raise ValueError("positive-reference id must be filesystem-safe")
        relative = PurePosixPath(self.video_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("positive-reference video_path must stay repo-relative")
        if not _SHA256.fullmatch(self.video_sha256):
            raise ValueError("positive-reference video_sha256 must be lowercase SHA-256")
        if not self.coordinate_frame.startswith("camera:"):
            raise ValueError("positive references require one named camera frame")
        if self.width <= 0 or self.height <= 0 or self.frame_count <= 0:
            raise ValueError("positive-reference media dimensions and frames must be positive")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("positive-reference fps must be finite and positive")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("positive-reference duration must be finite and positive")
        frame_duration = self.frame_count / self.fps
        if abs(frame_duration - self.duration_seconds) > (1.0 / self.fps + 1e-6):
            raise ValueError("positive-reference frame count and duration disagree")
        if self.sleeve_order not in _SLEEVE_ORDERS:
            raise ValueError("positive-reference sleeve order is unsupported")
        if self.terminal_behavior != COMPACT_IN_PLACE:
            raise ValueError("positive-reference terminal behavior is unsupported")
        if self.review_scope != VISUAL_QUALITY_REVIEW or self.review_state != "passed":
            raise ValueError("positive references require a passed visual-quality review")
        object.__setattr__(
            self,
            "observed_strengths",
            _nonempty_unique(self.observed_strengths, "observed_strengths"),
        )
        object.__setattr__(
            self,
            "unavailable_evidence",
            _nonempty_unique(self.unavailable_evidence, "unavailable_evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PositiveFoldReference:
        return cls(
            reference_id=str(payload["reference_id"]),
            video_path=str(payload["video_path"]),
            video_sha256=str(payload["video_sha256"]),
            coordinate_frame=str(payload["coordinate_frame"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            fps=float(payload["fps"]),
            frame_count=int(payload["frame_count"]),
            duration_seconds=float(payload["duration_seconds"]),
            sleeve_order=str(payload["sleeve_order"]),
            terminal_behavior=str(payload["terminal_behavior"]),
            review_scope=str(payload["review_scope"]),
            review_state=str(payload["review_state"]),
            observed_strengths=tuple(str(item) for item in payload["observed_strengths"]),
            unavailable_evidence=tuple(
                str(item) for item in payload["unavailable_evidence"]
            ),
        )


@dataclass(frozen=True)
class PositiveReferenceBank:
    """Immutable set of reviewed references used only during proposal generation."""

    bank_id: str
    references: tuple[PositiveFoldReference, ...]

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.bank_id):
            raise ValueError("positive-reference bank id must be filesystem-safe")
        if not self.references:
            raise ValueError("positive-reference bank must not be empty")
        ids = tuple(item.reference_id for item in self.references)
        hashes = tuple(item.video_sha256 for item in self.references)
        if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
            raise ValueError("positive-reference ids and video hashes must be unique")

    @property
    def bank_sha256(self) -> str:
        return _canonical_sha256(
            {
                "bank_id": self.bank_id,
                "references": [item.to_dict() for item in self.references],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "references": [item.to_dict() for item in self.references],
            "bank_sha256": self.bank_sha256,
            "conditioning_scope": CAMERA_PIXEL_REFERENCE,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PositiveReferenceBank:
        bank = cls(
            bank_id=str(payload["bank_id"]),
            references=tuple(
                PositiveFoldReference.from_dict(item) for item in payload["references"]
            ),
        )
        declared_hash = payload.get("bank_sha256")
        if declared_hash is not None and str(declared_hash) != bank.bank_sha256:
            raise ValueError("positive-reference bank hash mismatch")
        declared_scope = payload.get("conditioning_scope")
        if declared_scope is not None and declared_scope != CAMERA_PIXEL_REFERENCE:
            raise ValueError("positive-reference conditioning scope is unsupported")
        return bank

    def validate_files(self, repo_root: Path) -> None:
        root = repo_root.expanduser().resolve()
        for reference in self.references:
            video = (root / reference.video_path).resolve()
            if root not in video.parents or not video.is_file():
                raise FileNotFoundError(video)
            if _file_sha256(video) != reference.video_sha256:
                raise ValueError(
                    f"positive-reference video hash mismatch: {reference.reference_id}"
                )

    def for_sleeve_order(self, sleeve_order: str) -> PositiveFoldReference:
        matches = tuple(
            reference
            for reference in self.references
            if reference.sleeve_order == sleeve_order
        )
        if len(matches) != 1:
            raise ValueError(
                "positive-reference bank requires exactly one example per requested order"
            )
        return matches[0]


@dataclass(frozen=True)
class ReferenceConditioningPlan:
    """Hash-bound reference insertion that cannot alter the evaluation contract."""

    task_plan_sha256: str
    reference_bank_sha256: str
    reference_id: str
    reference_video_sha256: str
    sleeve_order: str
    terminal_strategy_id: str
    prompt_addendum: str
    non_overrideable_hard_gate_ids: tuple[str, ...]
    claim_boundary: str
    conditioning_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_reference_conditioning(
    task_plan: TaskReasoningPlan,
    bank: PositiveReferenceBank,
    *,
    reference_id: str | None = None,
) -> ReferenceConditioningPlan:
    """Bind a compatible reference to one plan without changing any hard gate."""

    strategy: TshirtFoldStrategy = strategy_from_plan(task_plan)
    reference = (
        next(
            (item for item in bank.references if item.reference_id == reference_id),
            None,
        )
        if reference_id is not None
        else bank.for_sleeve_order(strategy.sleeve_order)
    )
    if reference is None:
        raise ValueError("requested positive reference does not exist")
    if reference.sleeve_order != strategy.sleeve_order:
        raise ValueError("positive reference sleeve order does not match the task plan")

    gate_ids = tuple(gate.gate_id for gate in task_plan.verification_gates)
    if not gate_ids or any(not gate.fail_closed for gate in task_plan.verification_gates):
        raise ValueError("task plan contains a missing or relaxable verification gate")
    strengths = "; ".join(reference.observed_strengths)
    prompt = (
        f"Use reviewed visual reference {reference.reference_id} "
        f"(SHA-256 {reference.video_sha256}) only as a camera-pixel motion prior for "
        f"the {strategy.sleeve_order} sleeve cadence. Preserve these visible strengths: "
        f"{strengths}. Do not copy its compact-in-place terminal state; execute the "
        f"task plan's {strategy.strategy_id} terminal placement. The reference review "
        "covers visual generation quality only and supplies no automatic gate result. "
        "Every original hard gate and native-resolution human veto remains unchanged."
    )
    claim_boundary = (
        "Reference conditioning guides a generated camera-pixel proposal. It does not "
        "establish calibrated cloth geometry, contact force, collision safety, joint "
        "feasibility, executable commands, or real-robot task success."
    )
    unlocked: dict[str, Any] = {
        "task_plan_sha256": task_plan.plan_sha256,
        "reference_bank_sha256": bank.bank_sha256,
        "reference_id": reference.reference_id,
        "reference_video_sha256": reference.video_sha256,
        "sleeve_order": reference.sleeve_order,
        "terminal_strategy_id": strategy.strategy_id,
        "prompt_addendum": prompt,
        "non_overrideable_hard_gate_ids": gate_ids,
        "claim_boundary": claim_boundary,
    }
    return ReferenceConditioningPlan(
        **unlocked,
        conditioning_sha256=_canonical_sha256(unlocked),
    )


def compile_reference_conditioning_batch(
    task_plans: Sequence[TaskReasoningPlan],
    bank: PositiveReferenceBank,
) -> tuple[ReferenceConditioningPlan, ...]:
    """Compile a stable, strategy-diverse reference batch."""

    if not task_plans:
        raise ValueError("reference-conditioned task plan batch must not be empty")
    return tuple(compile_reference_conditioning(plan, bank) for plan in task_plans)


def load_positive_reference_bank(
    manifest_path: Path,
    *,
    repo_root: Path,
) -> PositiveReferenceBank:
    """Load a bank from a showcase manifest and verify every referenced byte stream."""

    payload = json.loads(manifest_path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("positive-reference manifest must contain one JSON object")
    raw_bank = payload.get("positive_reference_bank", payload)
    if not isinstance(raw_bank, Mapping):
        raise ValueError("positive_reference_bank must be one JSON object")
    bank = PositiveReferenceBank.from_dict(raw_bank)
    bank.validate_files(repo_root)
    return bank
