"""Agentic generate-evaluate-repair loop for approximate visual motion transfer."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Protocol
from uuid import uuid4

from phiagent.rendering.base import VideoRenderer, VisualTransferRequest, VisualTransferResult

_SCORE_FIELDS = (
    "motion_preservation",
    "target_identity",
    "object_consistency",
    "temporal_consistency",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class ProxyThresholds:
    motion_preservation: float = 0.75
    target_identity: float = 0.80
    object_consistency: float = 0.75
    temporal_consistency: float = 0.75

    def __post_init__(self) -> None:
        for field in _SCORE_FIELDS:
            value = getattr(self, field)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} threshold must be finite and in [0, 1]")


@dataclass(frozen=True)
class ProxyProposal:
    backend: str
    target_image: Path
    prompt: str
    seed: int

    def __post_init__(self) -> None:
        if not self.backend.strip() or not self.prompt.strip():
            raise ValueError("proxy backend and prompt must be non-empty")
        if self.seed < 0:
            raise ValueError("proxy seed must be non-negative")
        if not self.target_image.is_file():
            raise ValueError(f"proxy target image does not exist: {self.target_image}")

    @property
    def key(self) -> tuple[str, Path, str, int]:
        return (self.backend, self.target_image.resolve(), self.prompt, self.seed)


@dataclass(frozen=True)
class ProxyScorecard:
    evaluator: str
    motion_preservation: float
    target_identity: float
    object_consistency: float
    temporal_consistency: float
    diagnoses: tuple[str, ...] = ()
    evidence: Path | None = None

    def __post_init__(self) -> None:
        if not self.evaluator.strip():
            raise ValueError("proxy evaluator name must be non-empty")
        for field in _SCORE_FIELDS:
            value = getattr(self, field)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} score must be finite and in [0, 1]")
        if self.evidence is not None and not self.evidence.is_file():
            raise ValueError(f"proxy evaluation evidence does not exist: {self.evidence}")

    @property
    def mean_score(self) -> float:
        return sum(getattr(self, field) for field in _SCORE_FIELDS) / len(_SCORE_FIELDS)

    def accepted(self, thresholds: ProxyThresholds) -> bool:
        return all(
            getattr(self, field) >= getattr(thresholds, field) for field in _SCORE_FIELDS
        )

    def constraint_margin(self, thresholds: ProxyThresholds) -> float:
        """Return the worst score-to-threshold margin across all hard gates."""

        return min(
            getattr(self, field) - getattr(thresholds, field)
            for field in _SCORE_FIELDS
        )


@dataclass(frozen=True)
class ProxyEvaluationRequest:
    source_video: Path
    reference_video: Path
    proposal: ProxyProposal
    result: VisualTransferResult


class ProxyCandidateEvaluator(Protocol):
    def evaluate(self, request: ProxyEvaluationRequest) -> ProxyScorecard:
        """Score a generated candidate from persisted evidence."""


class CommandProxyEvaluator:
    """Run a local evaluator command that emits one strict JSON scorecard."""

    _ALLOWED_PLACEHOLDERS = {
        "candidate",
        "reference",
        "source",
        "target_image",
        "metadata",
    }
    _PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def __init__(self, command: tuple[str, ...]) -> None:
        if not command:
            raise ValueError("evaluator command cannot be empty")
        placeholders = {
            match.group(1)
            for argument in command
            for match in self._PLACEHOLDER_PATTERN.finditer(argument)
        }
        unknown = placeholders - self._ALLOWED_PLACEHOLDERS
        if unknown:
            raise ValueError(f"unknown evaluator command placeholders: {sorted(unknown)}")
        self.command = command

    def evaluate(self, request: ProxyEvaluationRequest) -> ProxyScorecard:
        values = {
            "candidate": str(request.result.output),
            "reference": str(request.reference_video),
            "source": str(request.source_video),
            "target_image": str(request.proposal.target_image),
            "metadata": str(request.result.metadata),
        }
        command = tuple(
            self._PLACEHOLDER_PATTERN.sub(
                lambda match: values.get(match.group(1), match.group(0)),
                argument,
            )
            for argument in self.command
        )
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ValueError("evaluator command must emit one JSON object")
        missing = set(_SCORE_FIELDS) - payload.keys()
        if missing:
            raise ValueError(f"evaluator scorecard is missing fields: {sorted(missing)}")
        diagnoses = payload.get("diagnoses", ())
        if not isinstance(diagnoses, list) or not all(
            isinstance(diagnosis, str) for diagnosis in diagnoses
        ):
            raise ValueError("evaluator diagnoses must be a JSON string array")
        evidence_value = payload.get("evidence")
        return ProxyScorecard(
            evaluator=str(payload.get("evaluator", command[0])),
            motion_preservation=float(payload["motion_preservation"]),
            target_identity=float(payload["target_identity"]),
            object_consistency=float(payload["object_consistency"]),
            temporal_consistency=float(payload["temporal_consistency"]),
            diagnoses=tuple(diagnoses),
            evidence=Path(evidence_value).resolve() if evidence_value is not None else None,
        )


@dataclass(frozen=True)
class ProxyCandidate:
    round_index: int
    candidate_index: int
    proposal: ProxyProposal
    result: VisualTransferResult
    scorecard: ProxyScorecard


class ProxyRepairAgent(Protocol):
    def propose(
        self,
        round_index: int,
        history: tuple[ProxyCandidate, ...],
        thresholds: ProxyThresholds,
    ) -> tuple[ProxyProposal, ...]:
        """Return feedback-conditioned proposals for the next round."""


class SeedFeedbackRepairAgent:
    """Retry the best candidate with a new seed and explicit failure feedback."""

    _FEEDBACK = {
        "motion_preservation": "Preserve the source motion and timing exactly.",
        "target_identity": "Keep the Sharpa dexterous-hand identity and geometry unchanged.",
        "object_consistency": "Preserve the manipulated object and all hand-object contacts.",
        "temporal_consistency": "Avoid flicker, deformation, discontinuities, and identity drift.",
    }

    def __init__(self, seed_stride: int = 1009) -> None:
        if seed_stride <= 0:
            raise ValueError("seed_stride must be positive")
        self.seed_stride = seed_stride

    def propose(
        self,
        round_index: int,
        history: tuple[ProxyCandidate, ...],
        thresholds: ProxyThresholds,
    ) -> tuple[ProxyProposal, ...]:
        if round_index < 1 or not history:
            raise ValueError("repair requires a positive round and candidate history")
        best = max(
            history,
            key=lambda candidate: (
                candidate.scorecard.constraint_margin(thresholds),
                candidate.scorecard.mean_score,
            ),
        )
        feedback = [
            message
            for field, message in self._FEEDBACK.items()
            if getattr(best.scorecard, field) < getattr(thresholds, field)
            and message not in best.proposal.prompt
        ]
        prompt = " ".join((best.proposal.prompt, *feedback))
        return (
            ProxyProposal(
                backend=best.proposal.backend,
                target_image=best.proposal.target_image,
                prompt=prompt,
                seed=best.proposal.seed + self.seed_stride,
            ),
        )


@dataclass(frozen=True)
class AgenticProxyRequest:
    source_video: Path
    reference_video: Path
    initial_proposals: tuple[ProxyProposal, ...]
    experiment_root: Path
    thresholds: ProxyThresholds = ProxyThresholds()
    maximum_rounds: int = 3

    def __post_init__(self) -> None:
        for label, path in (
            ("source video", self.source_video),
            ("reference video", self.reference_video),
        ):
            if not path.is_file():
                raise ValueError(f"proxy {label} does not exist: {path}")
        if not self.initial_proposals:
            raise ValueError("at least one initial proxy proposal is required")
        if self.maximum_rounds < 1:
            raise ValueError("maximum_rounds must be positive")


@dataclass(frozen=True)
class AgenticProxyOutcome:
    accepted: bool
    experiment_dir: Path
    best_candidate: ProxyCandidate
    candidates: tuple[ProxyCandidate, ...]
    trace_path: Path


class AgenticVisualTransferController:
    """Bounded visual candidate generation, evaluation, and feedback repair."""

    def __init__(
        self,
        renderers: Mapping[str, VideoRenderer],
        evaluator: ProxyCandidateEvaluator,
        repair_agent: ProxyRepairAgent | None = None,
        project_root: Path | None = None,
    ) -> None:
        if not renderers:
            raise ValueError("at least one proxy renderer is required")
        if any(not name.strip() for name in renderers):
            raise ValueError("proxy renderer names must be non-empty")
        self.renderers = dict(renderers)
        self.evaluator = evaluator
        self.repair_agent = repair_agent or SeedFeedbackRepairAgent()
        self.project_root = (
            project_root or Path(__file__).resolve().parents[2]
        ).expanduser().resolve()

    def _git_provenance(self) -> dict[str, object]:
        status = subprocess.run(
            ["git", "--no-pager", "status", "--short"],
            cwd=self.project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode != 0:
            return {
                "available": False,
                "head": None,
                "status": [],
                "error": status.stderr.strip(),
            }
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=self.project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "available": True,
            "head": head.stdout.strip() if head.returncode == 0 else "UNBORN",
            "head_error": head.stderr.strip() if head.returncode != 0 else None,
            "status": status.stdout.splitlines(),
        }

    def _source_provenance(self) -> dict[str, str]:
        candidates = set(self.project_root.glob("phiagent/**/*.py"))
        candidates.update(self.project_root.glob("scripts/*.py"))
        pyproject = self.project_root / "pyproject.toml"
        if pyproject.is_file():
            candidates.add(pyproject)
        files = sorted(path for path in candidates if path.is_file())
        if not files:
            raise ValueError(f"no source files found for provenance under {self.project_root}")
        return {
            str(path.relative_to(self.project_root)): _file_sha256(path) for path in files
        }

    @staticmethod
    def _package_versions() -> list[str]:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        return sorted(line for line in completed.stdout.splitlines() if line.strip())

    def _new_experiment(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment = root / f"{stamp}-{uuid4().hex[:8]}"
        experiment.mkdir()
        return experiment

    @staticmethod
    def _candidate_payload(candidate: ProxyCandidate) -> dict[str, object]:
        return {
            "round_index": candidate.round_index,
            "candidate_index": candidate.candidate_index,
            "proposal": {
                **asdict(candidate.proposal),
                "target_image": str(candidate.proposal.target_image),
            },
            "result": {
                "output": str(candidate.result.output),
                "experiment_dir": str(candidate.result.experiment_dir),
                "metadata": str(candidate.result.metadata),
            },
            "scorecard": {
                **asdict(candidate.scorecard),
                "evidence": (
                    str(candidate.scorecard.evidence)
                    if candidate.scorecard.evidence is not None
                    else None
                ),
                "mean_score": candidate.scorecard.mean_score,
            },
        }

    def run(self, request: AgenticProxyRequest) -> AgenticProxyOutcome:
        missing_backends = {
            proposal.backend
            for proposal in request.initial_proposals
            if proposal.backend not in self.renderers
        }
        if missing_backends:
            raise ValueError(f"proxy proposals use unknown backends: {sorted(missing_backends)}")

        experiment = self._new_experiment(request.experiment_root.expanduser().resolve())
        trace_path = experiment / "trace.json"
        candidates: list[ProxyCandidate] = []
        attempted: set[tuple[str, Path, str, int]] = set()
        proposals = request.initial_proposals
        manifest: dict[str, object] = {
            "schema_version": "1.0.0",
            "status": "running",
            "method": "agentic_proxy_not_official_phizero",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_video": str(request.source_video.resolve()),
            "source_sha256": _file_sha256(request.source_video),
            "reference_video": str(request.reference_video.resolve()),
            "reference_sha256": _file_sha256(request.reference_video),
            "thresholds": asdict(request.thresholds),
            "maximum_rounds": request.maximum_rounds,
            "command": sys.argv,
            "host": platform.node(),
            "python": platform.python_version(),
            "packages": self._package_versions(),
            "git": self._git_provenance(),
            "source_files": self._source_provenance(),
            "candidates": [],
        }
        _write_json(trace_path, manifest)

        for round_index in range(request.maximum_rounds):
            if not proposals:
                raise ValueError("proxy repair agent returned no proposals")
            for proposal in proposals:
                if proposal.backend not in self.renderers:
                    raise ValueError(f"proxy repair selected unknown backend: {proposal.backend}")
                if proposal.key in attempted:
                    raise ValueError("proxy repair repeated an already attempted proposal")
                attempted.add(proposal.key)
                candidate_index = len(candidates)
                output = experiment / "candidates" / f"{candidate_index:03d}.mp4"
                render_request = VisualTransferRequest(
                    video=request.source_video,
                    robot_image=proposal.target_image,
                    output=output,
                    prompt=proposal.prompt,
                    experiment_root=experiment / "backend-runs",
                    seed=proposal.seed,
                )
                result = self.renderers[proposal.backend].render(render_request)
                scorecard = self.evaluator.evaluate(
                    ProxyEvaluationRequest(
                        source_video=request.source_video,
                        reference_video=request.reference_video,
                        proposal=proposal,
                        result=result,
                    )
                )
                candidate = ProxyCandidate(
                    round_index=round_index,
                    candidate_index=candidate_index,
                    proposal=proposal,
                    result=result,
                    scorecard=scorecard,
                )
                candidates.append(candidate)
                manifest["candidates"] = [
                    self._candidate_payload(item) for item in candidates
                ]
                _write_json(trace_path, manifest)

            accepted = [
                candidate
                for candidate in candidates
                if candidate.scorecard.accepted(request.thresholds)
            ]
            if accepted:
                best = max(
                    accepted,
                    key=lambda candidate: (
                        candidate.scorecard.constraint_margin(request.thresholds),
                        candidate.scorecard.mean_score,
                    ),
                )
                manifest.update(
                    {
                        "status": "accepted",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "best_candidate_index": best.candidate_index,
                    }
                )
                _write_json(trace_path, manifest)
                return AgenticProxyOutcome(
                    True, experiment, best, tuple(candidates), trace_path
                )
            if round_index + 1 < request.maximum_rounds:
                proposals = self.repair_agent.propose(
                    round_index + 1, tuple(candidates), request.thresholds
                )

        best = max(
            candidates,
            key=lambda candidate: (
                candidate.scorecard.constraint_margin(request.thresholds),
                candidate.scorecard.mean_score,
            ),
        )
        manifest.update(
            {
                "status": "rejected",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "best_candidate_index": best.candidate_index,
            }
        )
        _write_json(trace_path, manifest)
        return AgenticProxyOutcome(False, experiment, best, tuple(candidates), trace_path)
