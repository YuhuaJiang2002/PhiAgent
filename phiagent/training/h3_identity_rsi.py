"""Contracts for native MiniMax-H3 identity-consistency improvement.

The module is intentionally standard-library only.  Model, CUDA, video, and
DiffSynth imports stay in executable adapters so importing :mod:`phiagent`
remains lightweight.

"RSI" is used in the bounded engineering sense: measure a frozen baseline,
train one declared candidate, reject any capability regression, and use the
failed gates to choose the next candidate from a reviewed search space.  It is
not unrestricted self-modification or self-reported improvement.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


IDENTITY_PLAN_SCHEMA = "1.0.0"
IDENTITY_METRIC_NAMES = (
    "reference_identity_mean",
    "reference_identity_worst",
    "cross_frame_identity",
    "topology_integrity",
    "motion_adherence",
    "action_adherence",
    "scene_preservation",
    "temporal_consistency",
)
TOPOLOGY_FRAME_GATES = (
    "single_robot_subject",
    "single_head_torso_chain",
    "exactly_two_arms",
    "left_shoulder_attachment",
    "right_shoulder_attachment",
    "continuous_arm_segments",
    "stable_robot_proportions",
    "no_extra_or_missing_limbs",
    "no_human_residual",
)
KINEMATIC_TOPOLOGY_FRAME_GATES = (
    "unique_left_shoulder_origin",
    "unique_right_shoulder_origin",
    "arm_roots_clear_of_head_and_neck",
)
SPLITS = ("train", "validation", "test")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,95}$")


def _finite_unit(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return number


def _portable_project_path(value: str, name: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{name} must be a portable project-relative path: {value!r}")
    return path.as_posix()


@dataclass(frozen=True)
class IdentityClipSpec:
    """One rights-attributed source interval for native Ref2VA supervision."""

    clip_id: str
    subject_id: str
    scene_id: str
    split: str
    source_video: str
    prompt: str
    source_start_seconds: float
    reference_frame: int
    license_id: str
    source_uri: str
    review_status: str
    source_crop: tuple[int, int, int, int] | None = None
    curriculum_tags: tuple[str, ...] = ()

    def validate(self) -> None:
        for name, value in (
            ("clip_id", self.clip_id),
            ("subject_id", self.subject_id),
            ("scene_id", self.scene_id),
        ):
            if not _SAFE_ID.fullmatch(value):
                raise ValueError(f"{name} is not filesystem safe: {value!r}")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        _portable_project_path(self.source_video, "source_video")
        if not math.isfinite(self.source_start_seconds) or self.source_start_seconds < 0:
            raise ValueError("source_start_seconds must be finite and non-negative")
        if self.reference_frame < 0:
            raise ValueError("reference_frame must be non-negative")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if not self.license_id.strip() or not self.source_uri.strip():
            raise ValueError("license_id and source_uri must be explicit")
        if self.review_status not in {"accepted", "partial"}:
            raise ValueError("review_status must be accepted or partial")
        invalid_tags = [tag for tag in self.curriculum_tags if not _SAFE_ID.fullmatch(tag)]
        if invalid_tags:
            raise ValueError(f"curriculum tags are not filesystem safe: {invalid_tags}")
        if len(self.curriculum_tags) != len(set(self.curriculum_tags)):
            raise ValueError("curriculum tags must be unique within one clip")
        if self.source_crop is not None:
            x, y, width, height = self.source_crop
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError("source_crop must be non-negative x/y and positive width/height")
            if any(value % 2 for value in self.source_crop):
                raise ValueError("source_crop values must be even for deterministic video encoding")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "IdentityClipSpec":
        crop_payload = payload.get("source_crop")
        tags_payload = payload.get("curriculum_tags", [])
        if crop_payload is not None and (
            not isinstance(crop_payload, list) or len(crop_payload) != 4
        ):
            raise ValueError("source_crop must be a four-integer JSON list")
        if crop_payload is not None and any(
            not isinstance(value, int) or isinstance(value, bool) for value in crop_payload
        ):
            raise ValueError("source_crop must contain only JSON integers")
        if not isinstance(tags_payload, list) or any(
            not isinstance(value, str) for value in tags_payload
        ):
            raise ValueError("curriculum_tags must be a JSON string list")
        item = cls(
            clip_id=str(payload["clip_id"]),
            subject_id=str(payload["subject_id"]),
            scene_id=str(payload["scene_id"]),
            split=str(payload["split"]),
            source_video=str(payload["source_video"]),
            prompt=str(payload["prompt"]),
            source_start_seconds=float(payload.get("source_start_seconds", 0.0)),
            reference_frame=int(payload.get("reference_frame", 0)),
            license_id=str(payload["license_id"]),
            source_uri=str(payload["source_uri"]),
            review_status=str(payload["review_status"]),
            source_crop=(
                tuple(crop_payload)  # type: ignore[arg-type]
                if crop_payload is not None
                else None
            ),
            curriculum_tags=tuple(tags_payload),
        )
        item.validate()
        return item


@dataclass(frozen=True)
class IdentityDatasetPlan:
    """Portable dataset plan with subject-disjoint held-out splits."""

    name: str
    fps: int
    width: int
    height: int
    num_frames: int
    clips: tuple[IdentityClipSpec, ...]
    schema_version: str = IDENTITY_PLAN_SCHEMA

    def validate(self) -> None:
        if self.schema_version != IDENTITY_PLAN_SCHEMA:
            raise ValueError(f"unsupported identity plan schema {self.schema_version!r}")
        if not _SAFE_ID.fullmatch(self.name):
            raise ValueError(f"plan name is not filesystem safe: {self.name!r}")
        if self.fps != 24:
            raise ValueError("MiniMax-H3 identity plans currently require 24 FPS")
        if self.width <= 0 or self.height <= 0 or self.width % 32 or self.height % 32:
            raise ValueError("width and height must be positive multiples of 32")
        if self.num_frames < 5 or (self.num_frames - 5) % 17:
            raise ValueError("num_frames must satisfy MiniMax-H3's 17n+5 contract")
        if not self.clips:
            raise ValueError("identity plan must contain clips")
        for clip in self.clips:
            clip.validate()
            if clip.reference_frame >= self.num_frames:
                raise ValueError(f"{clip.clip_id} reference_frame lies outside the prepared clip")
        ids = [clip.clip_id for clip in self.clips]
        if len(set(ids)) != len(ids):
            raise ValueError("clip_id values must be unique")
        # A source interval may never leak across splits.  Subject overlap is
        # allowed only between train and validation for character-specific
        # tuning; test subjects remain genuinely held out.
        source_splits: dict[str, set[str]] = {}
        subject_splits: dict[str, set[str]] = {}
        for clip in self.clips:
            source_splits.setdefault(clip.source_video, set()).add(clip.split)
            subject_splits.setdefault(clip.subject_id, set()).add(clip.split)
        leaked_sources = sorted(path for path, splits in source_splits.items() if len(splits) > 1)
        if leaked_sources:
            raise ValueError(f"source videos cross dataset splits: {leaked_sources}")
        leaked_test_subjects = sorted(
            subject
            for subject, splits in subject_splits.items()
            if "test" in splits and len(splits) > 1
        )
        if leaked_test_subjects:
            raise ValueError(f"test subjects leak into optimization splits: {leaked_test_subjects}")
        train_subjects = {clip.subject_id for clip in self.clips if clip.split == "train"}
        if len(train_subjects) < 2:
            raise ValueError("native identity training requires at least two training subjects")

    @classmethod
    def load(cls, path: Path) -> "IdentityDatasetPlan":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get("clips"), list):
            raise ValueError("identity plan must be a JSON object containing a clips list")
        plan = cls(
            schema_version=str(payload.get("schema_version", "")),
            name=str(payload["name"]),
            fps=int(payload["fps"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            num_frames=int(payload["num_frames"]),
            clips=tuple(IdentityClipSpec.from_dict(item) for item in payload["clips"]),
        )
        plan.validate()
        return plan

    def split(self, name: str) -> tuple[IdentityClipSpec, ...]:
        if name not in SPLITS:
            raise ValueError(f"unknown split {name!r}")
        return tuple(clip for clip in self.clips if clip.split == name)


@dataclass(frozen=True)
class DomainCurriculumAssessment:
    """Fail-closed coverage result for a domain-diverse RSI curriculum."""

    passed: bool
    gates: tuple[tuple[str, bool], ...]
    training_subjects: tuple[str, ...]
    training_scenes: tuple[str, ...]
    heldout_subjects: tuple[str, ...]
    heldout_scenes: tuple[str, ...]
    training_tags: tuple[str, ...]

    def failed_gates(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.gates if not passed)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["gates"] = dict(self.gates)
        payload["failed_gates"] = list(self.failed_gates())
        return payload


@dataclass(frozen=True)
class DomainCurriculumContract:
    """Minimum diversity required before another learned H3 RSI round.

    This contract exists because increasing LoRA capacity on two synthetic
    render domains did not transfer to the frozen real-scene shoulder failure.
    """

    minimum_training_subjects: int = 3
    minimum_training_scenes: int = 3
    minimum_heldout_subjects: int = 3
    minimum_heldout_scenes: int = 3
    required_training_tags: tuple[str, ...] = (
        "real-background",
        "full-body",
        "unique-shoulder-origins",
        "head-neck-clearance",
        "left-raised-near-head",
        "right-raised-near-head",
        "cross-body",
        "bilateral-reach",
    )

    def __post_init__(self) -> None:
        for name in (
            "minimum_training_subjects",
            "minimum_training_scenes",
            "minimum_heldout_subjects",
            "minimum_heldout_scenes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        invalid_tags = [tag for tag in self.required_training_tags if not _SAFE_ID.fullmatch(tag)]
        if invalid_tags:
            raise ValueError(f"required curriculum tags are not filesystem safe: {invalid_tags}")

    def assess(self, plan: IdentityDatasetPlan) -> DomainCurriculumAssessment:
        training = plan.split("train")
        heldout = (*plan.split("validation"), *plan.split("test"))
        training_subjects = tuple(sorted({clip.subject_id for clip in training}))
        training_scenes = tuple(sorted({clip.scene_id for clip in training}))
        heldout_subjects = tuple(sorted({clip.subject_id for clip in heldout}))
        heldout_scenes = tuple(sorted({clip.scene_id for clip in heldout}))
        training_tags = tuple(sorted({tag for clip in training for tag in clip.curriculum_tags}))
        training_sources = {clip.source_video for clip in training}
        heldout_sources = {clip.source_video for clip in heldout}
        gates = {
            "training_subject_diversity": (
                len(training_subjects) >= self.minimum_training_subjects
            ),
            "training_scene_diversity": len(training_scenes) >= self.minimum_training_scenes,
            "heldout_subject_diversity": len(heldout_subjects) >= self.minimum_heldout_subjects,
            "heldout_scene_diversity": len(heldout_scenes) >= self.minimum_heldout_scenes,
            "subject_disjoint": not set(training_subjects).intersection(heldout_subjects),
            "scene_disjoint": not set(training_scenes).intersection(heldout_scenes),
            "source_disjoint": not training_sources.intersection(heldout_sources),
            "required_training_tags": set(self.required_training_tags).issubset(training_tags),
            "validation_present": bool(plan.split("validation")),
            "test_present": bool(plan.split("test")),
        }
        return DomainCurriculumAssessment(
            passed=all(gates.values()),
            gates=tuple(gates.items()),
            training_subjects=training_subjects,
            training_scenes=training_scenes,
            heldout_subjects=heldout_subjects,
            heldout_scenes=heldout_scenes,
            training_tags=training_tags,
        )


@dataclass(frozen=True)
class IdentityMetrics:
    """Identity metrics plus capabilities that identity tuning must preserve."""

    reference_identity_mean: float
    reference_identity_worst: float
    cross_frame_identity: float
    topology_integrity: float
    motion_adherence: float
    action_adherence: float
    scene_preservation: float
    temporal_consistency: float

    def __post_init__(self) -> None:
        for name in IDENTITY_METRIC_NAMES:
            _finite_unit(getattr(self, name), name)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "IdentityMetrics":
        values = {
            name: _finite_unit(
                payload.get(name, 1.0) if name == "action_adherence" else payload[name],
                name,
            )
            for name in IDENTITY_METRIC_NAMES
        }
        return cls(**values)

    def identity_floor(self) -> float:
        return min(
            self.reference_identity_mean,
            self.reference_identity_worst,
            self.cross_frame_identity,
            self.topology_integrity,
        )


@dataclass(frozen=True)
class TopologyFrameReview:
    """Auditable semantic topology decision for one decoded video frame.

    These gates deliberately describe visual facts instead of a similarity
    proxy.  In particular, a duplicated upper arm or an arm emerging from the
    neck/torso cannot be hidden by averaging a high reference-identity score.
    """

    frame_index: int
    single_robot_subject: bool
    single_head_torso_chain: bool
    exactly_two_arms: bool
    left_shoulder_attachment: bool
    right_shoulder_attachment: bool
    continuous_arm_segments: bool
    stable_robot_proportions: bool
    no_extra_or_missing_limbs: bool
    no_human_residual: bool
    confidence: float
    note: str = ""
    decoded_frame_sha256: str | None = None
    unique_left_shoulder_origin: bool | None = None
    unique_right_shoulder_origin: bool | None = None
    arm_roots_clear_of_head_and_neck: bool | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        _finite_unit(self.confidence, "confidence")
        if self.decoded_frame_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.decoded_frame_sha256
        ):
            raise ValueError("decoded_frame_sha256 must be a lowercase SHA-256 digest")
        non_boolean_kinematic = [
            name
            for name in KINEMATIC_TOPOLOGY_FRAME_GATES
            if getattr(self, name) is not None and not isinstance(getattr(self, name), bool)
        ]
        if non_boolean_kinematic:
            raise ValueError(
                f"kinematic topology gates must be booleans or null: {non_boolean_kinematic}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TopologyFrameReview":
        missing = [name for name in TOPOLOGY_FRAME_GATES if name not in payload]
        if missing:
            raise ValueError(f"topology frame review is missing gates: {missing}")
        non_boolean = [name for name in TOPOLOGY_FRAME_GATES if not isinstance(payload[name], bool)]
        if non_boolean:
            raise ValueError(f"topology frame gates must be JSON booleans: {non_boolean}")
        non_boolean_kinematic = [
            name
            for name in KINEMATIC_TOPOLOGY_FRAME_GATES
            if name in payload and not isinstance(payload[name], bool)
        ]
        if non_boolean_kinematic:
            raise ValueError(
                f"kinematic topology gates must be JSON booleans: {non_boolean_kinematic}"
            )
        return cls(
            frame_index=int(payload["frame_index"]),
            confidence=float(payload["confidence"]),
            note=str(payload.get("note", "")),
            decoded_frame_sha256=(
                str(payload["decoded_frame_sha256"])
                if payload.get("decoded_frame_sha256") is not None
                else None
            ),
            **{
                name: payload.get(name) for name in KINEMATIC_TOPOLOGY_FRAME_GATES
            },
            **{name: payload[name] for name in TOPOLOGY_FRAME_GATES},
        )

    def failed_gates(self) -> tuple[str, ...]:
        return tuple(name for name in TOPOLOGY_FRAME_GATES if not getattr(self, name)) + tuple(
            name
            for name in KINEMATIC_TOPOLOGY_FRAME_GATES
            if getattr(self, name) is False
        )

    def kinematic_detail_complete(self) -> bool:
        return all(getattr(self, name) is not None for name in KINEMATIC_TOPOLOGY_FRAME_GATES)

    def passed(self, minimum_confidence: float) -> bool:
        return self.confidence >= minimum_confidence and not self.failed_gates()


@dataclass(frozen=True)
class TopologyReviewEvidence:
    """Full-video topology evidence bound to an immutable video digest."""

    video_sha256: str
    total_frames: int
    reviewer: str
    review_method: str
    frames: tuple[TopologyFrameReview, ...]
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError(f"unsupported topology evidence schema {self.schema_version!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.video_sha256):
            raise ValueError("video_sha256 must be a lowercase SHA-256 digest")
        if self.total_frames <= 0:
            raise ValueError("total_frames must be positive")
        if not self.reviewer.strip() or not self.review_method.strip():
            raise ValueError("reviewer and review_method must be explicit")
        indices = [frame.frame_index for frame in self.frames]
        if len(indices) != len(set(indices)):
            raise ValueError("topology reviews contain duplicate frame indices")
        if any(index >= self.total_frames for index in indices):
            raise ValueError("topology review frame lies outside the video")

    @classmethod
    def load(cls, path: Path) -> "TopologyReviewEvidence":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
            raise ValueError("topology evidence must contain a frames list")
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            video_sha256=str(payload["video_sha256"]),
            total_frames=int(payload["total_frames"]),
            reviewer=str(payload["reviewer"]),
            review_method=str(payload["review_method"]),
            frames=tuple(TopologyFrameReview.from_dict(item) for item in payload["frames"]),
        )

    def coverage_complete(self) -> bool:
        return {frame.frame_index for frame in self.frames} == set(range(self.total_frames))

    def decoded_frame_digests_complete(self) -> bool:
        """Return true only when every reviewed decoded frame has its own digest."""

        return self.coverage_complete() and all(
            frame.decoded_frame_sha256 is not None for frame in self.frames
        )

    def kinematic_detail_complete(self) -> bool:
        return self.coverage_complete() and all(
            frame.kinematic_detail_complete() for frame in self.frames
        )

    def passing_fraction(self, minimum_confidence: float) -> float:
        if not self.frames:
            return 0.0
        passed = sum(frame.passed(minimum_confidence) for frame in self.frames)
        return passed / self.total_frames

    def failed_frames(self, minimum_confidence: float) -> tuple[int, ...]:
        return tuple(
            frame.frame_index for frame in self.frames if not frame.passed(minimum_confidence)
        )

    def failure_histogram(self, minimum_confidence: float) -> dict[str, int]:
        histogram = {
            name: 0 for name in (*TOPOLOGY_FRAME_GATES, *KINEMATIC_TOPOLOGY_FRAME_GATES)
        }
        histogram["review_confidence"] = 0
        for frame in self.frames:
            for name in frame.failed_gates():
                histogram[name] += 1
            if frame.confidence < minimum_confidence:
                histogram["review_confidence"] += 1
        return {name: count for name, count in histogram.items() if count}


@dataclass(frozen=True)
class IdentityPromotionContract:
    """Hard promotion gate for a native identity candidate."""

    minimum_identity_gain: float = 0.02
    minimum_identity_floor: float = 0.62
    maximum_motion_regression: float = 0.01
    maximum_action_regression: float = 0.01
    maximum_scene_regression: float = 0.005
    maximum_temporal_regression: float = 0.01
    minimum_topology_integrity: float = 1.0
    minimum_topology_review_confidence: float = 0.95

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _finite_unit(value, name)

    def assess(
        self,
        baseline: IdentityMetrics,
        candidate: IdentityMetrics,
        topology_evidence: TopologyReviewEvidence | None = None,
    ) -> "IdentityPromotionAssessment":
        gain = candidate.identity_floor() - baseline.identity_floor()
        topology_score = (
            topology_evidence.passing_fraction(self.minimum_topology_review_confidence)
            if topology_evidence is not None
            else 0.0
        )
        gates = {
            "identity_gain": gain >= self.minimum_identity_gain,
            "identity_floor": candidate.identity_floor() >= self.minimum_identity_floor,
            "topology_evidence": topology_evidence is not None,
            "topology_full_frame_coverage": (
                topology_evidence is not None and topology_evidence.coverage_complete()
            ),
            "topology_decoded_frame_digests": (
                topology_evidence is not None
                and topology_evidence.decoded_frame_digests_complete()
            ),
            "topology_kinematic_detail": (
                topology_evidence is not None
                and topology_evidence.kinematic_detail_complete()
            ),
            "topology": topology_score >= self.minimum_topology_integrity,
            "topology_metric_matches_evidence": math.isclose(
                candidate.topology_integrity,
                topology_score,
                abs_tol=1e-9,
            ),
            "motion_non_regression": (
                candidate.motion_adherence
                >= baseline.motion_adherence - self.maximum_motion_regression
            ),
            "action_non_regression": (
                candidate.action_adherence
                >= baseline.action_adherence - self.maximum_action_regression
            ),
            "scene_non_regression": (
                candidate.scene_preservation
                >= baseline.scene_preservation - self.maximum_scene_regression
            ),
            "temporal_non_regression": (
                candidate.temporal_consistency
                >= baseline.temporal_consistency - self.maximum_temporal_regression
            ),
        }
        return IdentityPromotionAssessment(
            passed=all(gates.values()),
            identity_gain=gain,
            baseline_identity_floor=baseline.identity_floor(),
            candidate_identity_floor=candidate.identity_floor(),
            topology_score=topology_score,
            topology_failed_frames=(
                topology_evidence.failed_frames(self.minimum_topology_review_confidence)
                if topology_evidence is not None
                else ()
            ),
            topology_failure_histogram=(
                tuple(
                    topology_evidence.failure_histogram(
                        self.minimum_topology_review_confidence
                    ).items()
                )
                if topology_evidence is not None
                else ()
            ),
            gates=tuple(gates.items()),
        )


@dataclass(frozen=True)
class IdentityPromotionAssessment:
    passed: bool
    identity_gain: float
    baseline_identity_floor: float
    candidate_identity_floor: float
    topology_score: float
    topology_failed_frames: tuple[int, ...]
    topology_failure_histogram: tuple[tuple[str, int], ...]
    gates: tuple[tuple[str, bool], ...]

    def failed_gates(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.gates if not passed)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "identity_gain": self.identity_gain,
            "baseline_identity_floor": self.baseline_identity_floor,
            "candidate_identity_floor": self.candidate_identity_floor,
            "topology_score": self.topology_score,
            "topology_failed_frames": list(self.topology_failed_frames),
            "topology_failure_histogram": dict(self.topology_failure_histogram),
            "gates": dict(self.gates),
            "failed_gates": list(self.failed_gates()),
        }


@dataclass(frozen=True)
class H3IdentityRound:
    """One reviewed point in the bounded native-LoRA search space."""

    name: str
    lora_rank: int
    learning_rate: float
    dataset_repeat: int
    num_epochs: int
    target_modules: str = "qkv_proj,out_proj"

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.name):
            raise ValueError("round name must be filesystem safe")
        if self.lora_rank not in {4, 8, 16, 32}:
            raise ValueError("lora_rank is outside the reviewed search space")
        if not math.isfinite(self.learning_rate) or not 1e-6 <= self.learning_rate <= 2e-4:
            raise ValueError("learning_rate is outside the reviewed search space")
        if not 1 <= self.dataset_repeat <= 100 or not 1 <= self.num_epochs <= 8:
            raise ValueError("repeat/epoch count is outside the reviewed search space")
        if self.target_modules != "qkv_proj,out_proj":
            raise ValueError("unreviewed H3 LoRA target modules")


RSI_SEARCH_SPACE = (
    H3IdentityRound("r0-smoke-r8", 8, 5e-5, 1, 1),
    H3IdentityRound("r1-identity-r16", 16, 5e-5, 8, 2),
    H3IdentityRound("r2-conservative-r16", 16, 2e-5, 12, 2),
    H3IdentityRound("r3-capacity-r32", 32, 2e-5, 12, 3),
)


def choose_next_round(
    completed: Iterable[tuple[H3IdentityRound, IdentityPromotionAssessment]],
) -> H3IdentityRound | None:
    """Choose the next reviewed candidate from measured failed gates.

    The sequence cannot edit code or invent hyperparameters.  An accepted
    candidate terminates the loop.  Capability regression selects the lower-LR
    conservative round; identity underfit selects the next-capacity round.
    """

    history = tuple(completed)
    if not history:
        return RSI_SEARCH_SPACE[0]
    if history[-1][1].passed:
        return None
    reviewed_names = {round_.name for round_ in RSI_SEARCH_SPACE}
    if history[-1][0].name not in reviewed_names:
        # An out-of-table round is a separately reviewed extension, not an
        # invitation to cycle back through already exhausted r0-r3 settings.
        # Its failure requires a new structural objective/backbone review.
        return None
    tried = {round_.name for round_, _ in history}
    failed = set(history[-1][1].failed_gates())
    preferred = (
        ("r2-conservative-r16", "r3-capacity-r32")
        if failed
        & {
            "motion_non_regression",
            "action_non_regression",
            "scene_non_regression",
            "temporal_non_regression",
        }
        else ("r1-identity-r16", "r3-capacity-r32", "r2-conservative-r16")
    )
    by_name = {round_.name: round_ for round_ in RSI_SEARCH_SPACE}
    for name in preferred:
        if name not in tried:
            return by_name[name]
    return None


def build_diffsynth_metadata(
    clips: Iterable[tuple[IdentityClipSpec, str, str]],
) -> list[dict[str, object]]:
    """Build Ref2VA metadata from prepared relative video/reference paths."""

    rows = []
    for clip, video_path, reference_path in clips:
        clip.validate()
        rows.append(
            {
                "video": _portable_project_path(video_path, "prepared video"),
                "prompt": " ".join(clip.prompt.split()),
                "input_audio": _portable_project_path(video_path, "prepared audio/video"),
                "references": [
                    {
                        "type": "image",
                        "image": _portable_project_path(reference_path, "prepared reference"),
                    }
                ],
                "frame_rate": 24,
            }
        )
    if not rows:
        raise ValueError("cannot build empty DiffSynth metadata")
    return rows
