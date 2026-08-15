from __future__ import annotations

import math

import pytest

from phiagent.evaluation.acceptance import (
    AcceptanceContract,
    EvaluationRecord,
    GateRequirement,
    IndependentEvaluationUnit,
    evaluate_acceptance,
    summarize_experiment,
    wilson_95_confidence_interval,
)


def _contract() -> AcceptanceContract:
    return AcceptanceContract(
        (
            GateRequirement("task_motion", 0.8, weight=10.0),
            GateRequirement("embodiment", 0.8),
            GateRequirement("interaction", 0.8),
            GateRequirement("video_quality", 0.8),
        )
    )


def _unit(seed: int, *, action: str = "lift", embodiment: str = "arm") -> IndependentEvaluationUnit:
    return IndependentEvaluationUnit(
        scene="table",
        action=action,
        object="cup",
        embodiment=embodiment,
        seed=seed,
    )


def _record(
    seed: int,
    *,
    scores: dict[str, float] | None = None,
    human_review: bool | None = True,
    action: str = "lift",
    embodiment: str = "arm",
) -> EvaluationRecord:
    return EvaluationRecord(
        _unit(seed, action=action, embodiment=embodiment),
        scores
        or {
            "task_motion": 0.9,
            "embodiment": 0.9,
            "interaction": 0.9,
            "video_quality": 0.9,
        },
        human_review,
    )


def test_all_required_gates_and_explicit_human_review_pass() -> None:
    decision = evaluate_acceptance(_contract(), _record(1))

    assert decision.accepted
    assert decision.mean_score == pytest.approx(0.9)
    assert decision.weighted_score == pytest.approx(0.9)
    assert not decision.validation_errors


def test_high_mean_never_compensates_for_failed_hard_gate() -> None:
    decision = evaluate_acceptance(
        _contract(),
        _record(
            1,
            scores={
                "task_motion": 0.79,
                "embodiment": 1.0,
                "interaction": 1.0,
                "video_quality": 1.0,
            },
        ),
    )

    assert decision.mean_score == pytest.approx(0.9475)
    assert not decision.accepted
    assert decision.gate_failure_names == ("task_motion",)


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.01, 1.01])
def test_invalid_gate_values_fail_validation(value: float) -> None:
    with pytest.raises(ValueError, match="finite number in \\[0, 1\\]"):
        _record(1, scores={"task_motion": value})


def test_missing_gate_fails_closed_with_a_precise_validation_error() -> None:
    decision = evaluate_acceptance(_contract(), _record(1, scores={"task_motion": 0.9}))

    assert not decision.accepted
    assert decision.mean_score is None
    assert "missing required gate: embodiment" in decision.validation_errors


@pytest.mark.parametrize("review", [None, False])
def test_pending_or_rejected_human_review_cannot_pass(review: bool | None) -> None:
    decision = evaluate_acceptance(_contract(), _record(1, human_review=review))

    assert not decision.accepted
    assert "human review" in decision.validation_errors[-1]


def test_duplicate_independent_unit_rejects_frame_count_inflation() -> None:
    first = _record(1)
    repeated_view = EvaluationRecord(first.unit, first.gate_scores, True)

    with pytest.raises(ValueError, match="duplicate independent evaluation unit"):
        summarize_experiment(_contract(), (first, repeated_view), grouping_key="action")


def test_wilson_interval_known_boundary_cases() -> None:
    assert wilson_95_confidence_interval(0, 1) == pytest.approx((0.0, 0.7934506856))
    assert wilson_95_confidence_interval(1, 1) == pytest.approx((0.2065493144, 1.0))


def test_per_action_and_embodiment_rates_report_worst_group() -> None:
    records = (
        _record(1, action="lift", embodiment="arm"),
        _record(
            2,
            action="push",
            embodiment="gripper",
            scores={
                "task_motion": 0.2,
                "embodiment": 0.9,
                "interaction": 0.9,
                "video_quality": 0.9,
            },
        ),
        _record(3, action="push", embodiment="gripper"),
    )

    by_action = summarize_experiment(_contract(), records, grouping_key="action")
    by_embodiment = summarize_experiment(_contract(), records, grouping_key="embodiment")

    assert by_action.valid_transfer_rate.passed == 2
    assert by_action.valid_transfer_rate.total == 3
    assert by_action.worst_group.group == "push"
    assert by_embodiment.worst_group.group == "gripper"
    assert by_action.gate_failure_counts == {
        "embodiment": 0,
        "interaction": 0,
        "task_motion": 1,
        "video_quality": 0,
    }


def test_empty_and_unknown_grouping_keys_fail_precisely() -> None:
    with pytest.raises(ValueError, match="empty experiment"):
        summarize_experiment(_contract(), (), grouping_key="action")
    with pytest.raises(ValueError, match="unknown grouping key"):
        summarize_experiment(_contract(), (_record(1),), grouping_key="frame")


def test_statistics_serialization_is_deterministic_and_counts_human_status() -> None:
    stats = summarize_experiment(
        _contract(),
        (_record(1, human_review=None), _record(2, human_review=False)),
        grouping_key="action",
    )

    assert stats.human_pending_count == 1
    assert stats.human_rejected_count == 1
    assert stats.to_json() == stats.to_json()
