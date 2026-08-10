"""Agentic generate-evaluate-route loop for action-conditioned world models."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Protocol

from phiagent.acwm.adapters import (
    ACWMRenderRequest,
    ACWMRenderResult,
    ACWMVideoRenderer,
)
from phiagent.acwm.schema import ACWMCase

_SCORE_FIELDS = (
    "action_adherence",
    "embodiment_consistency",
    "object_interaction",
    "temporal_consistency",
    "background_consistency",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class ACWMThresholds:
    action_adherence: float = 0.75
    embodiment_consistency: float = 0.75
    object_interaction: float = 0.75
    temporal_consistency: float = 0.75
    background_consistency: float = 0.75

    def __post_init__(self) -> None:
        for field in _SCORE_FIELDS:
            value = getattr(self, field)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} threshold must be finite and in [0, 1]")


@dataclass(frozen=True)
class ACWMScorecard:
    evaluator: str
    action_adherence: float
    embodiment_consistency: float
    object_interaction: float
    temporal_consistency: float
    background_consistency: float
    human_review_passed: bool | None
    diagnoses: tuple[str, ...] = ()
    evidence: Path | None = None

    def __post_init__(self) -> None:
        if not self.evaluator.strip():
            raise ValueError("AC-WM evaluator name must be non-empty")
        for field in _SCORE_FIELDS:
            value = getattr(self, field)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} score must be finite and in [0, 1]")
        if self.evidence is not None and not self.evidence.is_file():
            raise ValueError(f"AC-WM evidence does not exist: {self.evidence}")

    @property
    def mean_score(self) -> float:
        return sum(getattr(self, field) for field in _SCORE_FIELDS) / len(_SCORE_FIELDS)

    def automatic_gates_pass(self, thresholds: ACWMThresholds) -> bool:
        return all(
            getattr(self, field) >= getattr(thresholds, field) for field in _SCORE_FIELDS
        )

    def accepted(self, thresholds: ACWMThresholds) -> bool:
        return self.automatic_gates_pass(thresholds) and self.human_review_passed is True

    def constraint_margin(self, thresholds: ACWMThresholds) -> float:
        margin = min(
            getattr(self, field) - getattr(thresholds, field) for field in _SCORE_FIELDS
        )
        if self.human_review_passed is False:
            return min(margin, -1.0)
        if self.human_review_passed is None:
            return min(margin, -0.5)
        return margin


@dataclass(frozen=True)
class ACWMEvaluationRequest:
    case: ACWMCase
    result: ACWMRenderResult


class ACWMCandidateEvaluator(Protocol):
    def evaluate(self, request: ACWMEvaluationRequest) -> ACWMScorecard:
        """Score one generated action-conditioned video."""


class CommandACWMEvaluator:
    """Invoke a strict local evaluator without importing its heavy dependencies."""

    _PLACEHOLDERS = {
        "candidate",
        "case_id",
        "condition",
        "first_frame",
        "metadata",
        "source",
    }

    def __init__(self, command: tuple[str, ...]) -> None:
        if not command:
            raise ValueError("AC-WM evaluator command cannot be empty")
        unknown = {
            token[1:-1]
            for argument in command
            for token in argument.split()
            if token.startswith("{") and token.endswith("}")
        } - self._PLACEHOLDERS
        if unknown:
            raise ValueError(f"unknown AC-WM evaluator placeholders: {sorted(unknown)}")
        self.command = command

    def evaluate(self, request: ACWMEvaluationRequest) -> ACWMScorecard:
        condition = request.result.metadata
        try:
            metadata = json.loads(request.result.metadata.read_text())
        except (OSError, json.JSONDecodeError):
            metadata = None
        if isinstance(metadata, dict) and isinstance(metadata.get("condition"), str):
            rendered_condition = Path(metadata["condition"]).expanduser()
            if rendered_condition.is_file():
                condition = rendered_condition.resolve()
        elif request.case.action.visual_condition is not None:
            # This fallback supports older adapters. Native backends should record
            # their canonical action JSON in the per-candidate metadata instead.
            condition = request.case.action.visual_condition
        values = {
            "candidate": str(request.result.output),
            "case_id": request.case.case_id,
            "condition": str(condition),
            "first_frame": str(request.case.first_frame),
            "metadata": str(request.result.metadata),
            "source": str(request.case.source_video),
        }
        command = tuple(
            argument.format_map(values) if "{" in argument else argument
            for argument in self.command
        )
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ValueError("AC-WM evaluator must emit one JSON object")
        missing = set(_SCORE_FIELDS) - payload.keys()
        if missing:
            raise ValueError(f"AC-WM scorecard is missing fields: {sorted(missing)}")
        human = payload.get("human_review_passed")
        if human not in {True, False, None}:
            raise ValueError("human_review_passed must be true, false, or null")
        diagnoses = payload.get("diagnoses", [])
        if not isinstance(diagnoses, list) or not all(
            isinstance(diagnosis, str) for diagnosis in diagnoses
        ):
            raise ValueError("AC-WM diagnoses must be a JSON string array")
        evidence_value = payload.get("evidence")
        return ACWMScorecard(
            evaluator=str(payload.get("evaluator", command[0])),
            **{field: float(payload[field]) for field in _SCORE_FIELDS},
            human_review_passed=human,
            diagnoses=tuple(diagnoses),
            evidence=Path(evidence_value).resolve() if evidence_value else None,
        )


@dataclass(frozen=True)
class ACWMProposal:
    case_id: str
    backend: str
    seed: int
    num_inference_steps: int = 35
    guidance_scale: float = 6.0
    prompt_suffix: str = ""

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.case_id,
            self.backend,
            self.seed,
            self.num_inference_steps,
            self.guidance_scale,
            self.prompt_suffix,
        )


@dataclass(frozen=True)
class ACWMCandidate:
    round_index: int
    candidate_index: int
    proposal: ACWMProposal
    result: ACWMRenderResult
    scorecard: ACWMScorecard


class ACWMRepairAgent(Protocol):
    def propose(
        self,
        *,
        cases: Mapping[str, ACWMCase],
        renderers: Mapping[str, ACWMVideoRenderer],
        history: tuple[ACWMCandidate, ...],
        thresholds: ACWMThresholds,
    ) -> tuple[ACWMProposal, ...]:
        """Route failed cases to a compatible model or revised seed."""


class ModelRoutingRepairAgent:
    """Prefer a new native backend before retrying the same model with feedback."""

    _FEEDBACK = {
        "action_adherence": "Follow the supplied action trajectory exactly frame by frame.",
        "embodiment_consistency": "Keep robot geometry and articulation stable.",
        "object_interaction": "Preserve causal grasp and object contact without duplication.",
        "temporal_consistency": "Avoid flicker, discontinuities, and geometry drift.",
        "background_consistency": "Keep the real camera and background unchanged.",
    }

    def __init__(
        self,
        backend_preference: tuple[str, ...] = ("oscar", "bwm", "kinema4d"),
        seed_stride: int = 1009,
    ) -> None:
        if seed_stride <= 0:
            raise ValueError("seed_stride must be positive")
        self.backend_preference = backend_preference
        self.seed_stride = seed_stride

    def propose(
        self,
        *,
        cases: Mapping[str, ACWMCase],
        renderers: Mapping[str, ACWMVideoRenderer],
        history: tuple[ACWMCandidate, ...],
        thresholds: ACWMThresholds,
    ) -> tuple[ACWMProposal, ...]:
        proposals: list[ACWMProposal] = []
        for case_id, case in cases.items():
            case_history = [item for item in history if item.proposal.case_id == case_id]
            if any(item.scorecard.accepted(thresholds) for item in case_history):
                continue
            best = max(
                case_history,
                key=lambda item: (
                    item.scorecard.constraint_margin(thresholds),
                    item.scorecard.mean_score,
                ),
            )
            if (
                best.scorecard.automatic_gates_pass(thresholds)
                and best.scorecard.human_review_passed is None
            ):
                continue
            tried = {item.proposal.backend for item in case_history}
            fallback = next(
                (
                    name
                    for name in self.backend_preference
                    if name in renderers
                    and name not in tried
                    and renderers[name].supports(case).supported
                ),
                None,
            )
            failed_fields = [
                field
                for field in _SCORE_FIELDS
                if getattr(best.scorecard, field) < getattr(thresholds, field)
            ]
            feedback = " ".join(self._FEEDBACK[field] for field in failed_fields)
            proposals.append(
                ACWMProposal(
                    case_id=case_id,
                    backend=fallback or best.proposal.backend,
                    seed=(best.proposal.seed if fallback else best.proposal.seed + self.seed_stride),
                    num_inference_steps=best.proposal.num_inference_steps,
                    guidance_scale=best.proposal.guidance_scale,
                    prompt_suffix=feedback,
                )
            )
        return tuple(proposals)


@dataclass(frozen=True)
class AgenticACWMRequest:
    cases: tuple[ACWMCase, ...]
    initial_proposals: tuple[ACWMProposal, ...]
    experiment_root: Path
    thresholds: ACWMThresholds = ACWMThresholds()
    maximum_rounds: int = 3

    def __post_init__(self) -> None:
        if not self.cases or not self.initial_proposals:
            raise ValueError("AC-WM workflow requires cases and initial proposals")
        case_ids = {case.case_id for case in self.cases}
        if len(case_ids) != len(self.cases):
            raise ValueError("AC-WM case IDs must be unique")
        if any(proposal.case_id not in case_ids for proposal in self.initial_proposals):
            raise ValueError("AC-WM proposal refers to an unknown case")
        if self.maximum_rounds < 1:
            raise ValueError("maximum_rounds must be positive")


@dataclass(frozen=True)
class AgenticACWMOutcome:
    status: str
    experiment_dir: Path
    candidates: tuple[ACWMCandidate, ...]
    best_by_case: tuple[ACWMCandidate, ...]
    trace_path: Path

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


class AgenticACWMController:
    """Generate, evaluate, route, and stop at mandatory human review."""

    def __init__(
        self,
        renderers: Mapping[str, ACWMVideoRenderer],
        evaluator: ACWMCandidateEvaluator,
        repair_agent: ACWMRepairAgent | None = None,
        project_root: Path | None = None,
    ) -> None:
        if not renderers:
            raise ValueError("at least one AC-WM renderer is required")
        self.renderers = dict(renderers)
        self.evaluator = evaluator
        self.repair_agent = repair_agent or ModelRoutingRepairAgent()
        self.project_root = (
            project_root or Path(__file__).resolve().parents[2]
        ).expanduser().resolve()

    def _new_experiment(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment = root / f"{stamp}-{uuid.uuid4().hex[:8]}"
        experiment.mkdir()
        return experiment

    def _git_state(self) -> dict[str, object]:
        state: dict[str, object] = {}
        for key, command in {
            "head": ["git", "rev-parse", "HEAD"],
            "tracked_changes": [
                "git",
                "status",
                "--short",
                "--untracked-files=no",
            ],
        }.items():
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.project_root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                state[key] = completed.stdout.splitlines() if key != "head" else completed.stdout.strip()
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                state[key] = f"unavailable: {exc}"
        return state

    def _initial_manifest(self, request: AgenticACWMRequest) -> dict[str, object]:
        source_files = [
            self.project_root / "phiagent" / "acwm" / "schema.py",
            self.project_root / "phiagent" / "acwm" / "adapters.py",
            self.project_root / "phiagent" / "agent" / "acwm.py",
            self.project_root / "scripts" / "run_acwm_backend.py",
        ]
        packages = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {
            "schema_version": "1.0.0",
            "status": "running",
            "method": "agentic_native_action_conditioned_world_model",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "command": sys.argv,
            "git": self._git_state(),
            "packages": sorted(line for line in packages if line.strip()),
            "source_files": {str(path.relative_to(self.project_root)): _sha256(path) for path in source_files},
            "thresholds": asdict(request.thresholds),
            "maximum_rounds": request.maximum_rounds,
            "cases": [
                {
                    "case_id": case.case_id,
                    "first_frame": str(case.first_frame.resolve()),
                    "first_frame_sha256": _sha256(case.first_frame),
                    "source_video": str(case.source_video.resolve()),
                    "source_video_sha256": _sha256(case.source_video),
                    "action": case.action.to_dict(),
                    "prompt": case.prompt,
                }
                for case in request.cases
            ],
            "candidates": [],
        }

    @staticmethod
    def _candidate_payload(candidate: ACWMCandidate) -> dict[str, object]:
        return {
            "round_index": candidate.round_index,
            "candidate_index": candidate.candidate_index,
            "proposal": asdict(candidate.proposal),
            "result": {
                "backend": candidate.result.backend,
                "case_id": candidate.result.case_id,
                "output": str(candidate.result.output),
                "output_sha256": _sha256(candidate.result.output),
                "metadata": str(candidate.result.metadata),
            },
            "scorecard": {
                **asdict(candidate.scorecard),
                "evidence": str(candidate.scorecard.evidence) if candidate.scorecard.evidence else None,
                "mean_score": candidate.scorecard.mean_score,
            },
        }

    def run(self, request: AgenticACWMRequest) -> AgenticACWMOutcome:
        cases = {case.case_id: case for case in request.cases}
        for proposal in request.initial_proposals:
            if proposal.backend not in self.renderers:
                raise ValueError(f"unknown AC-WM backend: {proposal.backend}")
            support = self.renderers[proposal.backend].supports(cases[proposal.case_id])
            if not support.supported:
                raise ValueError(
                    f"{proposal.backend} cannot run {proposal.case_id}: {', '.join(support.reasons)}"
                )
        experiment = self._new_experiment(request.experiment_root.expanduser().resolve())
        trace_path = experiment / "trace.json"
        manifest = self._initial_manifest(request)
        _write_json(trace_path, manifest)
        candidates: list[ACWMCandidate] = []
        attempted: set[tuple[object, ...]] = set()
        proposals = request.initial_proposals
        for round_index in range(request.maximum_rounds):
            if not proposals:
                break
            by_backend: dict[str, list[tuple[ACWMProposal, ACWMRenderRequest]]] = {}
            round_dir = experiment / f"round-{round_index:03d}"
            for proposal in proposals:
                if proposal.key in attempted:
                    raise ValueError("AC-WM repair repeated an attempted proposal")
                attempted.add(proposal.key)
                case = cases[proposal.case_id]
                effective_case = case
                if proposal.prompt_suffix.strip():
                    effective_case = replace(
                        case,
                        prompt=f"{case.prompt} {proposal.prompt_suffix.strip()}",
                    )
                output = experiment / "candidates" / (
                    f"{len(candidates) + sum(len(items) for items in by_backend.values()):03d}-"
                    f"{proposal.case_id}-{proposal.backend}.mp4"
                )
                render_request = ACWMRenderRequest(
                    case=effective_case,
                    output=output,
                    experiment_dir=round_dir,
                    seed=proposal.seed,
                    num_inference_steps=proposal.num_inference_steps,
                    guidance_scale=proposal.guidance_scale,
                )
                by_backend.setdefault(proposal.backend, []).append((proposal, render_request))
            for backend, items in by_backend.items():
                results = self.renderers[backend].render_batch([item[1] for item in items])
                if len(results) != len(items):
                    raise RuntimeError(f"{backend} returned the wrong number of candidates")
                for (proposal, render_request), result in zip(items, results):
                    scorecard = self.evaluator.evaluate(
                        ACWMEvaluationRequest(render_request.case, result)
                    )
                    candidates.append(
                        ACWMCandidate(
                            round_index=round_index,
                            candidate_index=len(candidates),
                            proposal=proposal,
                            result=result,
                            scorecard=scorecard,
                        )
                    )
            manifest["candidates"] = [self._candidate_payload(item) for item in candidates]
            _write_json(trace_path, manifest)
            best_by_case = tuple(
                max(
                    (item for item in candidates if item.proposal.case_id == case_id),
                    key=lambda item: (
                        item.scorecard.constraint_margin(request.thresholds),
                        item.scorecard.mean_score,
                    ),
                )
                for case_id in cases
            )
            if all(item.scorecard.accepted(request.thresholds) for item in best_by_case):
                status = "accepted"
                break
            if all(
                item.scorecard.automatic_gates_pass(request.thresholds)
                and item.scorecard.human_review_passed is None
                for item in best_by_case
            ):
                status = "pending_human_review"
                break
            proposals = self.repair_agent.propose(
                cases=cases,
                renderers=self.renderers,
                history=tuple(candidates),
                thresholds=request.thresholds,
            )
        else:
            status = "rejected"
        if "status" not in locals():
            status = "rejected"
        best_by_case = tuple(
            max(
                (item for item in candidates if item.proposal.case_id == case_id),
                key=lambda item: (
                    item.scorecard.constraint_margin(request.thresholds),
                    item.scorecard.mean_score,
                ),
            )
            for case_id in cases
        )
        manifest.update(
            {
                "status": status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "best_candidate_indices": [item.candidate_index for item in best_by_case],
                "human_review_required_for_acceptance": True,
            }
        )
        _write_json(trace_path, manifest)
        return AgenticACWMOutcome(
            status=status,
            experiment_dir=experiment,
            candidates=tuple(candidates),
            best_by_case=best_by_case,
            trace_path=trace_path,
        )
