from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from phiagent.evaluation.task_motion import (
    ContactEvent,
    ContactTimeline,
    EPLPhase,
    EPLTimeline,
    TaskMotionInput,
    Trajectory,
    evaluate_task_motion,
)


def _input(
    points: tuple[tuple[float, ...], ...] = ((0.0,), (0.5,), (1.0,)),
    times: tuple[float, ...] = (0.0, 0.5, 1.0),
    *,
    frame: str = "robot_base",
    phases: EPLTimeline | None = None,
    contacts: ContactTimeline | None = None,
) -> TaskMotionInput:
    return TaskMotionInput(
        Trajectory(times, points, frame),
        phases
        or EPLTimeline(
            (
                EPLPhase("approach", times[0], times[1]),
                EPLPhase("place", times[1], times[-1]),
            )
        ),
        contacts or ContactTimeline((ContactEvent("gripper:object", times[1], times[-1]),)),
    )


def _evaluate(reference: TaskMotionInput, candidate: TaskMotionInput):
    return evaluate_task_motion(reference, candidate, spatial_normalization=1.0)


def test_exact_match_is_perfect_and_immutable() -> None:
    reference = _input()

    scorecard = _evaluate(reference, reference)

    assert scorecard.overall_score == pytest.approx(1.0)
    assert scorecard.trajectory_similarity == pytest.approx(1.0)
    assert scorecard.phase_macro_f1 == pytest.approx(1.0)
    assert scorecard.contact_state_f1 == pytest.approx(1.0)
    with pytest.raises(FrozenInstanceError):
        scorecard.overall_score = 0.0  # type: ignore[misc]


def test_smooth_resampling_retains_action_adherence() -> None:
    reference = _input(
        ((0.0,), (0.25,), (0.5,), (0.75,), (1.0,)),
        (0.0, 0.25, 0.5, 0.75, 1.0),
    )
    candidate = _input(
        ((0.0,), (0.5,), (1.0,)),
        (0.0, 0.5, 1.0),
        phases=EPLTimeline((EPLPhase("approach", 0.0, 0.25), EPLPhase("place", 0.25, 1.0))),
        contacts=ContactTimeline((ContactEvent("gripper:object", 0.25, 1.0),)),
    )

    scorecard = _evaluate(reference, candidate)

    assert scorecard.trajectory_similarity > 0.9
    assert scorecard.direction_progress_adherence == pytest.approx(1.0)
    assert scorecard.overall_score > 0.9


def test_wrong_direction_is_not_hidden_by_other_metrics() -> None:
    reference = _input()
    candidate = _input(((1.0,), (0.5,), (0.0,)))

    scorecard = _evaluate(reference, candidate)

    assert scorecard.direction_progress_adherence == 0.0
    assert scorecard.overall_score == 0.0


def test_correct_endpoint_with_wrong_path_fails_full_horizon_similarity() -> None:
    reference = _input(((0.0, 0.0), (0.5, 0.0), (1.0, 0.0)))
    candidate = _input(((0.0, 0.0), (0.5, 2.0), (1.0, 0.0)))

    scorecard = _evaluate(reference, candidate)

    assert scorecard.terminal_state_accuracy == pytest.approx(1.0)
    assert scorecard.trajectory_similarity < 0.6
    assert scorecard.overall_score < 0.6


def test_shifted_and_missing_phases_reject_phase_agreement() -> None:
    reference = _input()
    shifted = _input(
        phases=EPLTimeline((EPLPhase("approach", 0.0, 0.8), EPLPhase("place", 0.8, 1.0)))
    )
    missing = _input(phases=EPLTimeline((EPLPhase("approach", 0.0, 1.0),)))

    shifted_score = _evaluate(reference, shifted)
    missing_score = _evaluate(reference, missing)

    assert shifted_score.phase_boundary_score < 0.8
    assert missing_score.phase_macro_f1 < 0.7
    assert missing_score.overall_score < 0.7


def test_missing_contact_rejects_contact_agreement() -> None:
    reference = _input()
    candidate = _input(contacts=ContactTimeline())

    scorecard = _evaluate(reference, candidate)

    assert scorecard.contact_state_f1 == 0.0
    assert scorecard.contact_timing_score < 0.5
    assert scorecard.overall_score == 0.0


def test_incomplete_horizon_rejects_coverage() -> None:
    reference = _input()
    candidate = _input(
        ((0.0,), (0.4,), (0.8,)),
        (0.0, 0.4, 0.8),
        phases=EPLTimeline((EPLPhase("approach", 0.0, 0.4), EPLPhase("place", 0.4, 0.8))),
        contacts=ContactTimeline((ContactEvent("gripper:object", 0.4, 0.8),)),
    )

    scorecard = _evaluate(reference, candidate)

    assert scorecard.horizon_coverage == pytest.approx(0.8)
    assert scorecard.overall_score <= 0.8


def test_frame_mismatch_has_precise_failure() -> None:
    with pytest.raises(ValueError, match="trajectory frame mismatch"):
        _evaluate(_input(frame="world"), _input(frame="camera"))


@pytest.mark.parametrize(
    ("timestamps", "positions", "message"),
    (
        ((0.0,), ((0.0,),), "at least two states"),
        ((0.0, 0.0), ((0.0,), (1.0,)), "strictly monotonic"),
        ((0.0, 1.0), ((0.0,), (0.5,), (1.0,)), "aligned lengths"),
        ((0.0, 1.0), ((nan,), (1.0,)), "must be finite"),
        ((0.0, inf), ((0.0,), (1.0,)), "must be finite"),
    ),
)
def test_invalid_and_nonfinite_trajectory_inputs_fail_closed(
    timestamps: tuple[float, ...], positions: tuple[tuple[float, ...], ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Trajectory(timestamps, positions, "world")


def test_invalid_normalization_and_phase_coverage_are_rejected() -> None:
    reference = _input()
    with pytest.raises(ValueError, match="spatial_normalization must be positive"):
        evaluate_task_motion(reference, reference, spatial_normalization=0.0)
    uncovered = _input(phases=EPLTimeline((EPLPhase("approach", 0.1, 1.0),)))
    with pytest.raises(ValueError, match="cover its trajectory horizon exactly"):
        _evaluate(reference, uncovered)


def test_numerically_equivalent_timeline_boundaries_are_accepted() -> None:
    rounded = 0.3
    computed = 0.1 + 0.2
    evidence = TaskMotionInput(
        Trajectory((0.0, computed, 0.6), ((0.0,), (0.5,), (1.0,)), "robot_base"),
        EPLTimeline(
            (
                EPLPhase("approach", 0.0, computed),
                EPLPhase("place", rounded, 0.6),
            )
        ),
        ContactTimeline(
            (
                ContactEvent("gripper:object", 0.0, computed),
                ContactEvent("gripper:object", rounded, 0.6),
            )
        ),
    )

    scorecard = _evaluate(evidence, evidence)

    assert scorecard.overall_score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("role", "contacts"),
    (
        ("reference", ContactTimeline((ContactEvent("gripper:object", -0.1, 0.5),))),
        ("candidate", ContactTimeline((ContactEvent("gripper:object", 0.5, 1.1),))),
    ),
)
def test_out_of_horizon_contact_evidence_is_rejected(role: str, contacts: ContactTimeline) -> None:
    reference = _input()
    candidate = _input()
    if role == "reference":
        reference = _input(contacts=contacts)
    else:
        candidate = _input(contacts=contacts)

    with pytest.raises(ValueError, match=f"{role} contact event .*trajectory horizon"):
        _evaluate(reference, candidate)


def test_horizon_coverage_is_clamped_for_candidate_spanning_reference() -> None:
    reference = _input()
    candidate = _input(
        ((-1.0,), (0.5,), (2.0,)),
        (-1.0, 0.5, 2.0),
        phases=EPLTimeline((EPLPhase("approach", -1.0, 0.5), EPLPhase("place", 0.5, 2.0))),
        contacts=ContactTimeline((ContactEvent("gripper:object", 0.5, 2.0),)),
    )

    scorecard = _evaluate(reference, candidate)

    assert 0.0 <= scorecard.horizon_coverage <= 1.0
    assert scorecard.horizon_coverage == pytest.approx(1.0)
