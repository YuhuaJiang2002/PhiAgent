from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from phiagent.agent.visual_transfer import (
    AgenticProxyRequest,
    AgenticVisualTransferController,
    CommandProxyEvaluator,
    ProxyEvaluationRequest,
    ProxyProposal,
    ProxyScorecard,
    ProxyThresholds,
)
from phiagent.rendering.base import VisualTransferRequest, VisualTransferResult


class _FakeRenderer:
    def render(self, request: VisualTransferRequest) -> VisualTransferResult:
        request.output.parent.mkdir(parents=True, exist_ok=True)
        request.output.write_bytes(f"candidate-{request.seed}".encode())
        experiment = request.experiment_root / f"seed-{request.seed}"
        experiment.mkdir(parents=True)
        metadata = experiment / "metadata.json"
        metadata.write_text(json.dumps({"seed": request.seed}) + "\n")
        return VisualTransferResult(request.output, experiment, metadata)


class _SeedEvaluator:
    def evaluate(self, request: ProxyEvaluationRequest) -> ProxyScorecard:
        temporal_score = 0.9 if request.proposal.seed > 1 else 0.4
        return ProxyScorecard(
            evaluator="seed-test",
            motion_preservation=0.9,
            target_identity=0.9,
            object_consistency=0.9,
            temporal_consistency=temporal_score,
            diagnoses=(() if temporal_score >= 0.8 else ("temporal flicker",)),
        )


class _RejectingEvaluator:
    def evaluate(self, request: ProxyEvaluationRequest) -> ProxyScorecard:
        return ProxyScorecard(
            evaluator="reject-test",
            motion_preservation=0.2,
            target_identity=0.3,
            object_consistency=0.4,
            temporal_consistency=0.5,
            diagnoses=("candidate below proxy thresholds",),
        )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.mp4"
    reference = tmp_path / "reference.mp4"
    target = tmp_path / "sharpa.png"
    source.write_bytes(b"source")
    reference.write_bytes(b"reference")
    target.write_bytes(b"target")
    return source, reference, target


def test_agentic_controller_uses_feedback_and_accepts_repaired_seed(tmp_path) -> None:
    source, reference, target = _inputs(tmp_path)
    controller = AgenticVisualTransferController(
        {"wan": _FakeRenderer()},
        _SeedEvaluator(),
    )
    request = AgenticProxyRequest(
        source_video=source,
        reference_video=reference,
        initial_proposals=(
            ProxyProposal("wan", target, "Transfer the source motion.", seed=1),
        ),
        experiment_root=tmp_path / "experiments",
        thresholds=ProxyThresholds(0.8, 0.8, 0.8, 0.8),
        maximum_rounds=2,
    )

    outcome = controller.run(request)

    assert outcome.accepted
    assert len(outcome.candidates) == 2
    assert outcome.best_candidate.proposal.seed == 1010
    assert "Avoid flicker" in outcome.best_candidate.proposal.prompt
    trace = json.loads(outcome.trace_path.read_text())
    assert trace["status"] == "accepted"
    assert trace["method"] == "agentic_proxy_not_official_phizero"
    assert trace["best_candidate_index"] == 1
    assert trace["source_sha256"] != trace["reference_sha256"]
    assert isinstance(trace["git"]["available"], bool)
    if trace["git"]["available"]:
        assert trace["git"]["head"]
    else:
        assert "not a git repository" in trace["git"]["error"]
    assert "phiagent/agent/visual_transfer.py" in trace["source_files"]


def test_agentic_controller_returns_best_rejected_candidate(tmp_path) -> None:
    source, reference, target = _inputs(tmp_path)
    controller = AgenticVisualTransferController(
        {"wan": _FakeRenderer()},
        _RejectingEvaluator(),
    )
    request = AgenticProxyRequest(
        source_video=source,
        reference_video=reference,
        initial_proposals=(ProxyProposal("wan", target, "Transfer motion.", seed=2),),
        experiment_root=tmp_path / "experiments",
        maximum_rounds=1,
    )

    outcome = controller.run(request)

    assert not outcome.accepted
    assert outcome.best_candidate.candidate_index == 0
    assert json.loads(outcome.trace_path.read_text())["status"] == "rejected"


def test_command_evaluator_reads_strict_json_scorecard(tmp_path) -> None:
    source, reference, target = _inputs(tmp_path)
    candidate = tmp_path / "candidate.mp4"
    metadata = tmp_path / "metadata.json"
    candidate.write_bytes(b"candidate")
    metadata.write_text("{}\n")
    payload = {
        "evaluator": "local-vlm",
        "motion_preservation": 0.8,
        "target_identity": 0.9,
        "object_consistency": 0.7,
        "temporal_consistency": 0.6,
        "diagnoses": ["object drift"],
    }
    evaluator = CommandProxyEvaluator(
        (
            sys.executable,
            "-c",
            f"print({json.dumps(json.dumps(payload))})",
            "{candidate}",
            "{reference}",
        )
    )
    request = ProxyEvaluationRequest(
        source_video=source,
        reference_video=reference,
        proposal=ProxyProposal("wan", target, "Transfer motion.", 3),
        result=VisualTransferResult(candidate, tmp_path, metadata),
    )

    scorecard = evaluator.evaluate(request)

    assert scorecard.evaluator == "local-vlm"
    assert scorecard.diagnoses == ("object drift",)
    assert scorecard.mean_score == pytest.approx(0.75)


def test_command_evaluator_rejects_unknown_placeholder() -> None:
    with pytest.raises(ValueError, match="unknown evaluator"):
        CommandProxyEvaluator(("evaluate", "{secret}"))


def test_constraint_margin_prioritizes_object_success_over_high_mean() -> None:
    thresholds = ProxyThresholds(0.75, 0.75, 0.75, 0.75)
    dropped_object = ProxyScorecard(
        "test",
        motion_preservation=1.0,
        target_identity=1.0,
        object_consistency=0.1,
        temporal_consistency=1.0,
    )
    nearly_valid = ProxyScorecard(
        "test",
        motion_preservation=0.74,
        target_identity=0.74,
        object_consistency=0.70,
        temporal_consistency=0.74,
    )

    assert dropped_object.mean_score > nearly_valid.mean_score
    assert dropped_object.constraint_margin(thresholds) < nearly_valid.constraint_margin(
        thresholds
    )
