from __future__ import annotations

from scripts.evaluate_acwm_bowl_action_variants import (
    classify_terminal_state,
    pairwise_endpoint_floor,
)


def test_terminal_direction_gates_are_mutually_readable() -> None:
    start = (416.0, 300.0)
    assert classify_terminal_state("slide-left", start, (250.0, 310.0), 832, 480) is True
    assert classify_terminal_state("slide-right", start, (580.0, 310.0), 832, 480)
    assert classify_terminal_state("lift-up", start, (430.0, 180.0), 832, 480)
    assert not classify_terminal_state("slide-left", start, (430.0, 180.0), 832, 480)
    assert not classify_terminal_state("lift-up", start, (250.0, 310.0), 832, 480)


def test_pairwise_endpoint_floor_uses_all_counterfactuals() -> None:
    floor = pairwise_endpoint_floor(
        {"slide-left": (150.0, 320.0), "slide-right": (650.0, 320.0), "lift-up": (400.0, 120.0)}
    )
    assert 300.0 < floor < 330.0
