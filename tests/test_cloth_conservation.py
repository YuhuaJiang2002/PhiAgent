from __future__ import annotations

from phiagent.evaluation.cloth_conservation import (
    SleeveLengthThresholds,
    SleeveObservation,
    TshirtFoldCandidateScore,
    rank_tshirt_fold_candidates,
    score_sleeve_length_conservation,
)


def _track(*, scale: float = 1.0, frames: int = 10, missing: int | None = None):
    result = []
    for frame_index in range(frames):
        current_scale = scale if frame_index >= frames // 2 else 1.0
        result.append(
            SleeveObservation(
                frame_index=frame_index,
                polyline_xy=(
                    (0.0, 0.0),
                    (10.0 * current_scale, 0.0),
                    (20.0 * current_scale, 0.0),
                ),
                confidence=0.0 if frame_index == missing else 0.95,
                visible=frame_index != missing,
            )
        )
    return result


def _candidate(seed: int, sleeve_passed: bool, soft: float):
    sleeve = score_sleeve_length_conservation(
        "left",
        _track(scale=1.0 if sleeve_passed else 0.65),
        expected_frames=10,
        thresholds=SleeveLengthThresholds(minimum_tracking_fraction=0.9),
    )
    return TshirtFoldCandidateScore(
        seed=seed,
        sleeve_scores={"left": sleeve},
        frame_zero_score=soft,
        background_score=soft,
        action_completion_score=soft,
        temporal_score=soft,
        minimum_frame_zero_score=0.5,
        minimum_background_score=0.5,
        minimum_action_completion_score=0.5,
        minimum_temporal_score=0.5,
        human_review_passed=True,
    )


def test_constant_sleeve_length_clears_the_hard_gate() -> None:
    score = score_sleeve_length_conservation(
        "viewer-left",
        _track(),
        expected_frames=10,
    )

    assert score.passed
    assert score.maximum_relative_deviation == 0.0
    assert score.terminal_relative_deviation == 0.0


def test_shortened_sleeve_is_rejected_even_when_tracking_is_confident() -> None:
    score = score_sleeve_length_conservation(
        "viewer-right",
        _track(scale=0.60),
        expected_frames=10,
    )

    assert not score.passed
    assert "length_relative_deviation" in score.failures
    assert "terminal_relative_deviation" in score.failures


def test_tracker_loss_is_a_failure_instead_of_hiding_length_change() -> None:
    score = score_sleeve_length_conservation(
        "viewer-left",
        _track(missing=5),
        expected_frames=10,
    )

    assert not score.passed
    assert "tracking_fraction" in score.failures


def test_high_soft_score_cannot_override_a_failed_sleeve_gate() -> None:
    failed = _candidate(seed=1, sleeve_passed=False, soft=0.99)
    passed = _candidate(seed=2, sleeve_passed=True, soft=0.70)

    ranked = rank_tshirt_fold_candidates(
        (failed, passed),
        require_human_review=True,
    )

    assert [item.seed for item in ranked] == [2]
    assert failed.soft_score > passed.soft_score


def test_human_review_is_a_veto_after_automatic_gates() -> None:
    candidate = _candidate(seed=3, sleeve_passed=True, soft=0.90)
    pending = TshirtFoldCandidateScore(
        **{
            **candidate.__dict__,
            "human_review_passed": None,
        }
    )

    assert rank_tshirt_fold_candidates(
        (pending,),
        require_human_review=False,
    )
    assert not rank_tshirt_fold_candidates(
        (pending,),
        require_human_review=True,
    )
