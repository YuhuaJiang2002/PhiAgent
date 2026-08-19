"""Dependency-free action-intent contracts for causal JoyAI video rendering.

JoyAI-Video-Edit accepts video, text, and an optional reference image.  It does
not expose SC3's numerical forward/inverse action interface.  This module keeps
that boundary explicit: a real demonstration supplies motion, a typed intent
supplies phase semantics, JoyAI proposes pixels, and an independent observer
must recover the intended phases before a candidate can be selected for a
visual demo.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from phiagent.rendering.joyai_video_edit import (
    JOYAI_MODEL_ID,
    JOYAI_MODEL_REVISION,
    JOYAI_REPOSITORY,
    JOYAI_REPOSITORY_REVISION,
)


SCHEMA_VERSION = "1.0.0"
MOTION_AUTHORITY = "source_demonstration"
VALID_ACTIONS = frozenset(
    {
        "observe",
        "approach",
        "grasp",
        "transport",
        "place",
        "release",
        "retract",
        "tool_use",
        "hold",
    }
)
VALID_CONTACT_STATES = frozenset(
    {"free", "approaching", "visual_contact", "held", "released", "supported"}
)
VISUAL_HARD_GATES = (
    "action_consistency",
    "source_motion_preservation",
    "object_identity",
    "embodiment_identity",
    "temporal_continuity",
)
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,95}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_exact_keys(
    data: Mapping[str, Any], *, required: set[str], optional: set[str], label: str
) -> None:
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required - optional)
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _validate_id(value: str, label: str) -> None:
    if not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must contain 2-96 lowercase letters, digits, '.', '_', or '-'")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class IntentObject:
    object_id: str
    description: str
    persistence_invariants: tuple[str, ...]

    def validate(self) -> None:
        _validate_id(self.object_id, "object_id")
        if not self.description.strip():
            raise ValueError("object description must be non-empty")
        if not self.persistence_invariants or any(
            not item.strip() for item in self.persistence_invariants
        ):
            raise ValueError("every object requires non-empty persistence invariants")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntentObject":
        _require_exact_keys(
            data,
            required={"object_id", "description", "persistence_invariants"},
            optional=set(),
            label="intent object",
        )
        value = cls(
            object_id=str(data["object_id"]),
            description=str(data["description"]),
            persistence_invariants=tuple(str(item) for item in data["persistence_invariants"]),
        )
        value.validate()
        return value


@dataclass(frozen=True)
class ActionPhase:
    phase_id: str
    action: str
    start_frame: int
    end_frame: int
    actor: str
    object_id: str | None
    target: str
    contact_state: str
    instruction: str

    def validate(self) -> None:
        _validate_id(self.phase_id, "phase_id")
        if self.action not in VALID_ACTIONS:
            raise ValueError(
                f"unsupported action {self.action!r}; expected one of {sorted(VALID_ACTIONS)}"
            )
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("phase frames must be a non-negative inclusive interval")
        if not self.actor.strip() or not self.target.strip() or not self.instruction.strip():
            raise ValueError("phase actor, target, and instruction must be non-empty")
        if self.object_id is not None:
            _validate_id(self.object_id, "phase object_id")
        if self.contact_state not in VALID_CONTACT_STATES:
            raise ValueError(
                "unsupported contact_state; it must remain an image-space intent label"
            )

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionPhase":
        _require_exact_keys(
            data,
            required={
                "phase_id",
                "action",
                "start_frame",
                "end_frame",
                "actor",
                "object_id",
                "target",
                "contact_state",
                "instruction",
            },
            optional=set(),
            label="action phase",
        )
        value = cls(
            phase_id=str(data["phase_id"]),
            action=str(data["action"]),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            actor=str(data["actor"]),
            object_id=(None if data["object_id"] is None else str(data["object_id"])),
            target=str(data["target"]),
            contact_state=str(data["contact_state"]),
            instruction=str(data["instruction"]),
        )
        value.validate()
        return value


@dataclass(frozen=True)
class JoyAISettings:
    seeds: tuple[int, ...]
    num_inference_steps: int = 2
    output_quality: int = 95
    minimum_phase_confidence: float = 0.75

    def validate(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("JoyAI seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("JoyAI seeds must be non-negative")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if not 1 <= self.output_quality <= 100:
            raise ValueError("output_quality must be in [1, 100]")
        if not 0.0 < self.minimum_phase_confidence <= 1.0:
            raise ValueError("minimum_phase_confidence must be in (0, 1]")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JoyAISettings":
        _require_exact_keys(
            data,
            required={"seeds"},
            optional={
                "num_inference_steps",
                "output_quality",
                "minimum_phase_confidence",
            },
            label="JoyAI settings",
        )
        value = cls(
            seeds=tuple(int(seed) for seed in data["seeds"]),
            num_inference_steps=int(data.get("num_inference_steps", 2)),
            output_quality=int(data.get("output_quality", 95)),
            minimum_phase_confidence=float(data.get("minimum_phase_confidence", 0.75)),
        )
        value.validate()
        return value


@dataclass(frozen=True)
class JoyAIActionIntentConfig:
    task_id: str
    instruction: str
    embodiment: str
    source_video: Path
    source_sha256: str
    reference_image: Path
    reference_sha256: str
    coordinate_frame: str
    width: int
    height: int
    fps: float
    model_frame_count: int
    deliverable_frame_count: int
    motion_authority: str
    objects: tuple[IntentObject, ...]
    phases: tuple[ActionPhase, ...]
    scene_invariants: tuple[str, ...]
    joyai: JoyAISettings
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported action-intent schema {self.schema_version!r}")
        _validate_id(self.task_id, "task_id")
        if not self.instruction.strip() or not self.embodiment.strip():
            raise ValueError("instruction and embodiment must be non-empty")
        if not _SHA256_PATTERN.fullmatch(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if not _SHA256_PATTERN.fullmatch(self.reference_sha256):
            raise ValueError("reference_sha256 must be a lowercase SHA-256 digest")
        if not self.coordinate_frame.startswith("camera:"):
            raise ValueError("coordinate_frame must explicitly name a camera frame")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("width, height, and fps must be positive")
        if not float(self.fps).is_integer():
            raise ValueError("v1 requires integer FPS for the official JoyAI client")
        if self.model_frame_count < 1 or (self.model_frame_count - 1) % 8:
            raise ValueError("model_frame_count must satisfy JoyAI's 1 + 8n contract")
        if not 1 <= self.deliverable_frame_count <= self.model_frame_count:
            raise ValueError("deliverable_frame_count must lie in the model timeline")
        if self.motion_authority != MOTION_AUTHORITY:
            raise ValueError(
                "v1 requires source_demonstration motion authority; JoyAI text is not "
                "a numerical action-conditioning interface"
            )
        if not self.objects or not self.phases or not self.scene_invariants:
            raise ValueError("objects, phases, and scene_invariants must be non-empty")
        for item in self.objects:
            item.validate()
        for phase in self.phases:
            phase.validate()
        object_ids = tuple(item.object_id for item in self.objects)
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("object IDs must be unique")
        phase_ids = tuple(phase.phase_id for phase in self.phases)
        if len(set(phase_ids)) != len(phase_ids):
            raise ValueError("phase IDs must be unique")
        if self.phases[0].start_frame != 0:
            raise ValueError("the first action phase must start at frame 0")
        if self.phases[-1].end_frame != self.deliverable_frame_count - 1:
            raise ValueError("action phases must end on the last deliverable frame")
        for previous, following in zip(self.phases, self.phases[1:]):
            if following.start_frame != previous.end_frame + 1:
                raise ValueError("action phases must be contiguous without gaps or overlap")
        unknown_objects = sorted(
            {
                phase.object_id
                for phase in self.phases
                if phase.object_id is not None and phase.object_id not in object_ids
            }
        )
        if unknown_objects:
            raise ValueError(f"phases reference unknown objects: {unknown_objects}")
        if any(not value.strip() for value in self.scene_invariants):
            raise ValueError("scene invariants must be non-empty")
        self.joyai.validate()

    @property
    def tail_padding_frames(self) -> int:
        return self.model_frame_count - self.deliverable_frame_count

    def verify_inputs(self) -> dict[str, Any]:
        records: dict[str, Any] = {}
        for label, path, expected in (
            ("source_video", self.source_video, self.source_sha256),
            ("reference_image", self.reference_image, self.reference_sha256),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} is missing: {path}")
            observed = sha256_file(path)
            if observed != expected:
                raise ValueError(f"{label} hash mismatch: observed {observed}, expected {expected}")
            records[label] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": observed,
            }
        return records

    def to_manifest(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "embodiment": self.embodiment,
            "source_video": str(self.source_video),
            "source_sha256": self.source_sha256,
            "reference_image": str(self.reference_image),
            "reference_sha256": self.reference_sha256,
            "coordinate_frame": self.coordinate_frame,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "model_frame_count": self.model_frame_count,
            "deliverable_frame_count": self.deliverable_frame_count,
            "tail_padding_frames": self.tail_padding_frames,
            "motion_authority": self.motion_authority,
            "objects": [asdict(item) for item in self.objects],
            "phases": [asdict(phase) for phase in self.phases],
            "scene_invariants": list(self.scene_invariants),
            "joyai": asdict(self.joyai),
            "model": {
                "repository": JOYAI_REPOSITORY,
                "repository_revision": JOYAI_REPOSITORY_REVISION,
                "weights": JOYAI_MODEL_ID,
                "weights_revision": JOYAI_MODEL_REVISION,
                "authority": "visual_proposal_only",
            },
            "physical_evidence": False,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, project_root: Path) -> "JoyAIActionIntentConfig":
        _require_exact_keys(
            data,
            required={
                "schema_version",
                "task_id",
                "instruction",
                "embodiment",
                "source_video",
                "source_sha256",
                "reference_image",
                "reference_sha256",
                "coordinate_frame",
                "width",
                "height",
                "fps",
                "model_frame_count",
                "deliverable_frame_count",
                "motion_authority",
                "objects",
                "phases",
                "scene_invariants",
                "joyai",
            },
            optional=set(),
            label="action-intent config",
        )

        def resolve(value: object) -> Path:
            path = Path(str(value)).expanduser()
            return (project_root / path).resolve() if not path.is_absolute() else path.resolve()

        result = cls(
            schema_version=str(data["schema_version"]),
            task_id=str(data["task_id"]),
            instruction=str(data["instruction"]),
            embodiment=str(data["embodiment"]),
            source_video=resolve(data["source_video"]),
            source_sha256=str(data["source_sha256"]),
            reference_image=resolve(data["reference_image"]),
            reference_sha256=str(data["reference_sha256"]),
            coordinate_frame=str(data["coordinate_frame"]),
            width=int(data["width"]),
            height=int(data["height"]),
            fps=float(data["fps"]),
            model_frame_count=int(data["model_frame_count"]),
            deliverable_frame_count=int(data["deliverable_frame_count"]),
            motion_authority=str(data["motion_authority"]),
            objects=tuple(IntentObject.from_dict(item) for item in data["objects"]),
            phases=tuple(ActionPhase.from_dict(item) for item in data["phases"]),
            scene_invariants=tuple(str(item) for item in data["scene_invariants"]),
            joyai=JoyAISettings.from_dict(data["joyai"]),
        )
        result.validate()
        return result


def load_action_intent_config(path: Path, *, project_root: Path) -> JoyAIActionIntentConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("action-intent config must contain one JSON object")
    return JoyAIActionIntentConfig.from_dict(raw, project_root=project_root)


def compile_action_prompt(config: JoyAIActionIntentConfig) -> str:
    """Compile one immutable full-stream prompt for the official JoyAI protocol."""

    config.validate()
    lines = [
        "Render a causal, reference-guided real-world robot action video.",
        f"GLOBAL ACTION INTENT: {config.instruction.strip()}",
        f"TARGET EMBODIMENT: {config.embodiment.strip()}",
        (
            "MOTION AUTHORITY: The input demonstration is authoritative for camera "
            "motion, timing, actor trajectory, object trajectory, occlusion, and phase "
            "boundaries. Retarget that motion to the target embodiment; do not replace "
            "it with generic or prompt-invented motion."
        ),
        "ACTION TIMELINE (inclusive source frame indices):",
    ]
    for phase in config.phases:
        start_s = phase.start_frame / config.fps
        end_s = (phase.end_frame + 1) / config.fps
        object_text = phase.object_id if phase.object_id is not None else "scene"
        lines.append(
            f"- {phase.phase_id}: frames {phase.start_frame}-{phase.end_frame} "
            f"({start_s:.3f}-{end_s:.3f}s), action={phase.action}, actor={phase.actor}, "
            f"object={object_text}, target={phase.target}, visual_relation={phase.contact_state}. "
            f"{phase.instruction.strip()}"
        )
    lines.append("PERSISTENT OBJECT CONTRACTS:")
    for item in config.objects:
        lines.append(f"- {item.object_id}: {item.description.strip()}")
        lines.extend(f"  * {rule.strip()}" for rule in item.persistence_invariants)
    lines.append("IMMUTABLE SCENE CONTRACTS:")
    lines.extend(f"- {rule.strip()}" for rule in config.scene_invariants)
    lines.extend(
        [
            (
                "Keep one temporally coherent target robot. Preserve handedness, limb "
                "count, tool topology, material, scale, depth ordering, and contact-side "
                "occlusion across every causal chunk."
            ),
            (
                "Do not freeze, teleport, duplicate, merge, or delete the manipulated "
                "objects. Do not add camera cuts, text, new actors, or new tools."
            ),
            (
                "The visual_relation labels above describe intended image-space events. "
                "They are not force, metric contact, calibration, or robot-control evidence."
            ),
        ]
    )
    if config.tail_padding_frames:
        lines.append(
            f"Frames {config.deliverable_frame_count}-{config.model_frame_count - 1} "
            "are cloned protocol tail padding. Keep the final deliverable state unchanged; "
            "these padding frames will be trimmed without interpolation."
        )
    return "\n".join(lines) + "\n"


def candidate_plan(
    config: JoyAIActionIntentConfig,
    *,
    output_dir: Path,
    prompt_file: Path,
    client_script: Path,
    python_executable: str,
    server_url: str,
) -> tuple[dict[str, Any], ...]:
    config.validate()
    if not server_url.startswith(("ws://", "wss://")):
        raise ValueError("server_url must use ws:// or wss://")
    plans = []
    for seed in config.joyai.seeds:
        candidate_id = f"seed-{seed}"
        candidate_dir = output_dir / "candidates" / candidate_id
        command = [
            python_executable,
            str(client_script),
            "--server-url",
            server_url,
            "--input-video",
            str(config.source_video),
            "--output-dir",
            str(candidate_dir),
            "--reference-image",
            str(config.reference_image),
            "--prompt-file",
            str(prompt_file),
            "--width",
            str(config.width),
            "--height",
            str(config.height),
            "--fps",
            f"{config.fps:g}",
            "--expected-frames",
            str(config.model_frame_count),
            "--seed",
            str(seed),
            "--num-inference-steps",
            str(config.joyai.num_inference_steps),
            "--output-quality",
            str(config.joyai.output_quality),
            "--throughput-mode",
        ]
        plans.append(
            {
                "candidate_id": candidate_id,
                "seed": seed,
                "output_dir": str(candidate_dir),
                "command": command,
            }
        )
    return tuple(plans)


def build_candidate_audit_template(
    config: JoyAIActionIntentConfig, candidate_id: str
) -> dict[str, Any]:
    _validate_id(candidate_id, "candidate_id")
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "candidate_path": None,
        "candidate_sha256": None,
        "observer": None,
        "observer_version": None,
        "independent_of_renderer": None,
        "phase_observations": [
            {
                "phase_id": phase.phase_id,
                "intended_action": phase.action,
                "observed_action": None,
                "confidence": None,
                "evidence_frames": [],
            }
            for phase in config.phases
        ],
        "hard_gates": {name: None for name in VISUAL_HARD_GATES},
        "human_native_resolution_veto": None,
        "physical_gates": {
            "metric_camera": False,
            "exact_robot_q": False,
            "persistent_metric_object_geometry": False,
            "sensor_or_solver_force": False,
        },
        "notes": "Fill from an observer that did not generate or tune the candidate.",
    }


def evaluate_candidate_audit(
    config: JoyAIActionIntentConfig, audit: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate a hash-bound inverse-action audit without mean-score overrides."""

    _require_exact_keys(
        audit,
        required={
            "schema_version",
            "candidate_id",
            "candidate_path",
            "candidate_sha256",
            "observer",
            "observer_version",
            "independent_of_renderer",
            "phase_observations",
            "hard_gates",
            "human_native_resolution_veto",
            "physical_gates",
        },
        optional={"notes"},
        label="candidate audit",
    )
    if str(audit["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("candidate audit schema_version does not match the harness")
    candidate_id = str(audit["candidate_id"])
    _validate_id(candidate_id, "candidate_id")
    candidate_path = Path(str(audit["candidate_path"])).expanduser()
    if not candidate_path.is_absolute():
        raise ValueError("candidate_path must be absolute")
    candidate_path = candidate_path.resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(f"audited candidate is missing: {candidate_path}")
    candidate_hash = str(audit["candidate_sha256"])
    if not _SHA256_PATTERN.fullmatch(candidate_hash):
        raise ValueError("candidate audit must bind a lowercase SHA-256 digest")
    observed_candidate_hash = sha256_file(candidate_path)
    if observed_candidate_hash != candidate_hash:
        raise ValueError(
            "candidate audit hash mismatch: "
            f"observed {observed_candidate_hash}, expected {candidate_hash}"
        )
    if audit["independent_of_renderer"] is not True:
        raise ValueError("candidate observer must be independent of the renderer")
    if not str(audit["observer"]).strip() or not str(audit["observer_version"]).strip():
        raise ValueError("observer and observer_version must be non-empty")

    observations = audit["phase_observations"]
    if not isinstance(observations, list) or len(observations) != len(config.phases):
        raise ValueError("phase_observations must match the complete intent timeline")
    phase_results = []
    for phase, raw in zip(config.phases, observations):
        if not isinstance(raw, Mapping):
            raise ValueError("each phase observation must be an object")
        _require_exact_keys(
            raw,
            required={
                "phase_id",
                "intended_action",
                "observed_action",
                "confidence",
                "evidence_frames",
            },
            optional=set(),
            label="phase observation",
        )
        if str(raw["phase_id"]) != phase.phase_id:
            raise ValueError("phase observations must preserve intent order and IDs")
        if str(raw["intended_action"]) != phase.action:
            raise ValueError("audit intended_action does not match the frozen config")
        observed = str(raw["observed_action"])
        confidence = float(raw["confidence"])
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("phase confidence must be finite and in [0, 1]")
        evidence_frames = tuple(int(value) for value in raw["evidence_frames"])
        if not evidence_frames:
            raise ValueError("every phase observation requires evidence frames")
        if tuple(sorted(set(evidence_frames))) != evidence_frames:
            raise ValueError("phase evidence frames must be sorted and unique")
        if any(frame < phase.start_frame or frame > phase.end_frame for frame in evidence_frames):
            raise ValueError("phase evidence frame lies outside the frozen interval")
        matched = observed == phase.action
        passed = matched and confidence >= config.joyai.minimum_phase_confidence
        phase_results.append(
            {
                "phase_id": phase.phase_id,
                "matched": matched,
                "confidence": confidence,
                "passed": passed,
                "evidence_frames": list(evidence_frames),
            }
        )

    hard_gates = audit["hard_gates"]
    if not isinstance(hard_gates, Mapping) or set(hard_gates) != set(VISUAL_HARD_GATES):
        raise ValueError(f"hard_gates must contain exactly {VISUAL_HARD_GATES}")
    if any(type(value) is not bool for value in hard_gates.values()):
        raise ValueError("every hard gate must be a boolean")
    if type(audit["human_native_resolution_veto"]) is not bool:
        raise ValueError("human_native_resolution_veto must be a boolean")

    inverse_action_score = sum(
        result["confidence"] if result["matched"] else 0.0 for result in phase_results
    ) / len(phase_results)
    phase_pass = all(result["passed"] for result in phase_results)
    all_visual_gates = all(bool(hard_gates[name]) for name in VISUAL_HARD_GATES)
    human_pass = not bool(audit["human_native_resolution_veto"])
    visual_eligible = phase_pass and all_visual_gates and human_pass
    failed = [name for name in VISUAL_HARD_GATES if not hard_gates[name]]
    if not phase_pass and "action_consistency" not in failed:
        failed.insert(0, "action_consistency")
    if not human_pass:
        failed.append("human_native_resolution_veto")
    return {
        "candidate_id": candidate_id,
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_hash,
        "phase_results": phase_results,
        "inverse_action_score": inverse_action_score,
        "visual_eligible": visual_eligible,
        "failed_visual_gates": failed,
        "selection_reason": (
            "visual_demo_contract_pass" if visual_eligible else "hard_gate_failed"
        ),
        "physical_promotable": False,
        "physical_reason": "proposal_not_physical_calibration",
    }


def select_visual_candidate(
    config: JoyAIActionIntentConfig, audits: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    evaluations = tuple(evaluate_candidate_audit(config, audit) for audit in audits)
    if not evaluations:
        raise ValueError("at least one candidate audit is required")
    candidate_ids = [str(item["candidate_id"]) for item in evaluations]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate audits must have unique candidate IDs")
    eligible = tuple(item for item in evaluations if item["visual_eligible"])
    if not eligible:
        return {
            "selected_candidate": None,
            "rollback": True,
            "reason": "hard_gate_failed",
            "evaluations": list(evaluations),
            "physical_promotable": False,
        }
    selected = max(
        eligible,
        key=lambda item: (float(item["inverse_action_score"]), str(item["candidate_id"])),
    )
    return {
        "selected_candidate": selected["candidate_id"],
        "selected_candidate_sha256": selected["candidate_sha256"],
        "rollback": False,
        "reason": "visual_demo_contract_pass",
        "evaluations": list(evaluations),
        "physical_promotable": False,
        "physical_reason": "proposal_not_physical_calibration",
    }


__all__ = [
    "ActionPhase",
    "IntentObject",
    "JoyAIActionIntentConfig",
    "JoyAISettings",
    "MOTION_AUTHORITY",
    "VALID_ACTIONS",
    "VISUAL_HARD_GATES",
    "build_candidate_audit_template",
    "candidate_plan",
    "compile_action_prompt",
    "evaluate_candidate_audit",
    "load_action_intent_config",
    "select_visual_candidate",
    "sha256_file",
]
