from __future__ import annotations

import pytest

from phiagent.evaluation.interaction import (
    FramePoint3D,
    InteractionEvaluationConfig,
    InteractionExpectation,
    InteractionFrame,
    InteractionTrace,
    TrackedObjectObservation,
    evaluate_interaction,
)


FRAME = "robot_base"


def _point(x: float, y: float = 0.0, z: float = 0.0, *, frame: str = FRAME) -> FramePoint3D:
    return FramePoint3D(frame, x, y, z)


def _trace(
    hand_xs: tuple[float, ...] = (0.0, 0.0, 0.1, 0.2),
    object_xs: tuple[float | None, ...] = (0.10, 0.02, 0.12, 0.22),
    *,
    states: tuple[str, ...] = ("table", "grasped", "grasped", "lifted"),
    duplicate_at: int | None = None,
    frame: str = FRAME,
) -> InteractionTrace:
    frames = []
    for index, (hand_x, object_x, state) in enumerate(zip(hand_xs, object_xs, states)):
        objects: tuple[TrackedObjectObservation, ...]
        if object_x is None:
            objects = ()
        else:
            item = TrackedObjectObservation("cup", _point(object_x, frame=frame), state=state)
            objects = (item, item) if index == duplicate_at else (item,)
        frames.append(InteractionFrame(index * 0.1, _point(hand_x, frame=frame), objects))
    return InteractionTrace("cup", frame, tuple(frames))


def _expectation() -> InteractionExpectation:
    return InteractionExpectation(_trace())


def test_correct_grasp_and_lift_passes() -> None:
    scorecard = evaluate_interaction(_trace(), _expectation())

    assert scorecard.passed
    assert scorecard.manipulation_score == 1.0
    assert scorecard.relative_trajectory_score == 1.0


def test_visible_but_never_contacted_fails() -> None:
    candidate = _trace(object_xs=(0.10, 0.10, 0.20, 0.30))

    scorecard = evaluate_interaction(candidate, _expectation())

    assert not scorecard.passed
    assert scorecard.manipulation_score == 0.0
    assert scorecard.contact_agreement_score == 0.0


def test_object_motion_before_contact_fails_causal_score() -> None:
    candidate = _trace(
        hand_xs=(0.0, 0.0, 0.0, 0.2),
        object_xs=(0.10, 0.20, 0.02, 0.22),
    )

    scorecard = evaluate_interaction(candidate, _expectation())

    assert scorecard.causal_order_score < 1.0
    assert not scorecard.passed


def test_causal_degradation_uses_seconds_not_spatial_normalization() -> None:
    candidate = _trace(
        hand_xs=(0.0, 0.0, 0.0, 0.2),
        object_xs=(0.10, 0.20, 0.02, 0.22),
    )
    scaled_candidate = _trace(
        hand_xs=(0.0, 0.0, 0.0, 2.0),
        object_xs=(1.0, 2.0, 0.2, 2.2),
    )
    scaled_expectation = InteractionExpectation(
        _trace(
            hand_xs=(0.0, 0.0, 1.0, 2.0),
            object_xs=(1.0, 0.2, 1.2, 2.2),
        )
    )
    scorecard = evaluate_interaction(
        candidate,
        _expectation(),
        InteractionEvaluationConfig(normalization_m=0.001, causal_score_decay_s=0.1),
    )
    scaled_scorecard = evaluate_interaction(
        scaled_candidate,
        scaled_expectation,
        InteractionEvaluationConfig(
            contact_distance_m=0.3,
            trajectory_tolerance_m=0.3,
            distance_tolerance_m=0.2,
            coupling_tolerance_m=0.2,
            terminal_tolerance_m=0.3,
            motion_threshold_m=0.1,
            maximum_object_step_m=2.0,
            normalization_m=10.0,
            causal_score_decay_s=0.1,
        ),
    )

    assert scorecard.causal_order_score == pytest.approx(0.3)
    assert scaled_scorecard.causal_order_score == pytest.approx(scorecard.causal_order_score)


def test_uncoupled_object_drift_while_grasped_fails() -> None:
    candidate = _trace(object_xs=(0.10, 0.02, 0.02, 0.02))

    scorecard = evaluate_interaction(candidate, _expectation())

    assert scorecard.motion_coupling_score < 0.75
    assert not scorecard.passed


@pytest.mark.parametrize(
    ("object_xs", "duplicate_at"),
    [((0.10, None, 0.12, 0.22), None), ((0.10, 0.02, 0.12, 0.22), 2)],
)
def test_dropout_or_duplication_fails_coverage_and_identity(
    object_xs: tuple[float | None, ...], duplicate_at: int | None
) -> None:
    candidate = _trace(object_xs=object_xs, duplicate_at=duplicate_at)
    scorecard = evaluate_interaction(candidate, _expectation())

    assert scorecard.identity_score < 1.0
    assert scorecard.visibility_coverage < 1.0
    assert not scorecard.passed


def test_teleport_fails_continuity() -> None:
    candidate = _trace(object_xs=(0.10, 0.02, 1.0, 0.22))

    scorecard = evaluate_interaction(candidate, _expectation())

    assert scorecard.continuity_score == 0.0
    assert scorecard.diagnostics.teleport_frames == (2, 3)
    assert not scorecard.passed


def test_wrong_terminal_state_fails() -> None:
    candidate = _trace(states=("table", "grasped", "grasped", "table"))

    scorecard = evaluate_interaction(candidate, _expectation())

    assert scorecard.terminal_state_score == 0.0
    assert not scorecard.passed


def test_frame_mismatch_and_invalid_input_raise() -> None:
    with pytest.raises(ValueError, match="coordinate frame"):
        InteractionTrace(
            "cup",
            FRAME,
            (
                InteractionFrame(
                    0.0,
                    _point(0.0, frame="camera"),
                    (TrackedObjectObservation("cup", _point(0.1)),),
                ),
                InteractionFrame(
                    0.1,
                    _point(0.0, frame="camera"),
                    (TrackedObjectObservation("cup", _point(0.1)),),
                ),
            ),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        InteractionTrace(
            "cup",
            FRAME,
            (
                InteractionFrame(0.0, _point(0.0), ()),
                InteractionFrame(0.0, _point(0.0), ()),
            ),
        )
    with pytest.raises(ValueError, match="positive"):
        InteractionEvaluationConfig(normalization_m=0.0)
    with pytest.raises(ValueError, match="positive"):
        InteractionEvaluationConfig(causal_score_decay_s=0.0)
