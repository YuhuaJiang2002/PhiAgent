from __future__ import annotations

import math

import pytest

from scripts.build_epic_ego_bottle_action_controls import (
    _clean_interaction_frame,
    _extract_bottle_patch,
    _parse_progress_range,
    ego_bottle_state,
)


def _state(label: str, progress: float):
    return ego_bottle_state(
        label,
        progress,
        width=1280,
        height=720,
        start_x=640,
        start_y=430,
    )


def test_ego_bottle_actions_have_distinct_midpoint_states() -> None:
    pour = _state("pour-bottle", 0.55)
    shake = _state("shake-bottle", 0.55)
    handover = _state("handover-bottle", 0.55)

    assert pour.holder == "right_hand"
    assert shake.holder == "both_hands"
    assert handover.holder == "both_hands"
    assert pour.bottle_rotation_degrees > 60
    assert abs(shake.bottle_x - handover.bottle_x) > 20


def test_ego_bottle_actions_have_declared_terminal_holders() -> None:
    assert all(_state(label, 0.0).holder == "right_hand" for label in (
        "pour-bottle", "shake-bottle", "handover-bottle"
    ))
    assert _state("pour-bottle", 1.0).holder == "counter"
    assert _state("shake-bottle", 1.0).holder == "counter"
    assert _state("handover-bottle", 1.0).holder == "left_hand"


def test_shake_action_completes_multiple_lateral_cycles() -> None:
    samples = [_state("shake-bottle", index / 200).bottle_x for index in range(61, 136)]
    directions = [math.copysign(1, second - first) for first, second in zip(samples, samples[1:]) if second != first]
    reversals = sum(first != second for first, second in zip(directions, directions[1:]))

    assert reversals >= 6


def test_multitask_actions_have_distinct_semantics_and_terminal_holders() -> None:
    rack = _state("place-bottle-rack", 1.0)
    cap = _state("unscrew-bottle-cap", 1.0)
    rinse = _state("rinse-bottle", 1.0)

    assert rack.holder == "dish_rack"
    assert not rack.left_grasp and not rack.right_grasp
    assert cap.holder == "right_hand" and cap.left_grasp and cap.right_grasp
    assert rinse.holder == "right_hand" and rinse.right_grasp
    assert abs(rack.bottle_x - cap.bottle_x) > 100
    assert abs(cap.left_wrist_x - rinse.left_wrist_x) > 50


def test_unscrew_and_rinse_encode_repeated_rotation_cycles() -> None:
    unscrew = [_state("unscrew-bottle-cap", index / 240).left_wrist_x for index in range(92, 168)]
    rinse = [_state("rinse-bottle", index / 240).bottle_rotation_degrees for index in range(92, 172)]

    def reversals(values: list[float]) -> int:
        directions = [
            math.copysign(1, second - first)
            for first, second in zip(values, values[1:])
            if second != first
        ]
        return sum(first != second for first, second in zip(directions, directions[1:]))

    assert reversals(unscrew) >= 5
    assert reversals(rinse) >= 5


def test_continuation_progress_range_is_bounded_and_action_specific() -> None:
    assert _parse_progress_range("unscrew-bottle-cap=0.72,1.0") == (
        "unscrew-bottle-cap",
        0.72,
        1.0,
    )
    with pytest.raises(Exception, match="0 <= start < end <= 1"):
        _parse_progress_range("rinse-bottle=0.9,0.5")
    with pytest.raises(Exception, match="unsupported progress-range label"):
        _parse_progress_range("unknown-task=0.5,1.0")


def test_bottle_patch_uses_object_matte_instead_of_box_border() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.full((100, 140, 3), 220, dtype=np.uint8)
    cv2.rectangle(frame, (58, 24), (81, 77), (25, 45, 160), -1)

    patch, mask = _extract_bottle_patch(cv2, np, frame, (48, 14, 44, 74))

    assert patch.shape[:2] == (74, 44)
    assert mask[37, 22] == 255
    assert mask[0, 0] == 0
    assert 0.1 < float(np.mean(mask > 0)) < 0.7


def test_blue_bottle_matte_rejects_dark_counter_rectangle() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.full((100, 140, 3), (28, 28, 28), dtype=np.uint8)
    cv2.ellipse(frame, (70, 55), (17, 31), 0, 0, 360, (180, 95, 25), -1)

    _, mask = _extract_bottle_patch(cv2, np, frame, (45, 15, 50, 75))

    assert mask[40, 25] == 255
    assert mask[0, 0] == 0
    assert mask[-1, -1] == 0
    assert float(np.mean(mask > 0)) < 0.6


def test_interaction_softening_preserves_top_of_ego_frame() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.full((100, 140, 3), 210, dtype=np.uint8)
    frame[55:88, 45:90] = (80, 125, 190)
    frame[67:77, 57:67] = (10, 10, 10)

    cleaned = _clean_interaction_frame(cv2, np, frame)

    assert np.array_equal(cleaned[:30], frame[:30])
    assert float(np.mean(np.abs(cleaned[55:88].astype(float) - frame[55:88]))) > 0
