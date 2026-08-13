from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_routed_vfm_long_demo.py"
SPEC = importlib.util.spec_from_file_location("build_routed_vfm_long_demo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_route_strength_is_bounded_and_reaches_full_weight() -> None:
    weights = [MODULE.route_strength(i, 10, 30, 3, 99) for i in range(8, 33)]
    assert weights[:2] == [0.0, 0.0]
    assert weights[2:6] == [0.25, 0.5, 0.75, 1.0]
    assert weights[-6:] == [1.0, 0.75, 0.5, 0.25, 0.0, 0.0]
    assert all(0.0 <= value <= 1.0 for value in weights)


def test_terminal_window_does_not_fade_to_incumbent() -> None:
    assert MODULE.route_strength(95, 90, 99, 3, 99) == 1.0
    assert MODULE.route_strength(99, 90, 99, 3, 99) == 1.0


def test_switch_search_interval_prevents_provider_gap() -> None:
    routes = [
        {"start": 470, "end": 558},
        {"start": 485, "end": 524},
        {"start": 514, "end": 558},
        {"start": 542, "end": 630},
        {"start": 571, "end": 659},
    ]
    assert MODULE.switch_search_interval(routes, 0) == (470, 484)
    assert MODULE.switch_search_interval(routes, 2) == (514, 525)
    assert MODULE.switch_search_interval(routes, 3) == (542, 559)
    assert MODULE.switch_search_interval(routes, 4) == (571, 631)


def test_invalid_route_contract_fails_closed() -> None:
    try:
        MODULE.route_strength(5, 7, 4, 3, 10)
    except ValueError as exc:
        assert "invalid route" in str(exc)
    else:
        raise AssertionError("invalid route must fail")


def test_locked_and_editable_regions_are_disjoint_and_do_not_mutate_inputs() -> None:
    support = np.asarray([[True, True, False], [False, True, False]])
    flower = np.asarray([[False, True, False], [True, False, False]])
    support_before = support.copy()
    flower_before = flower.copy()

    background, editable = MODULE._locked_and_editable_regions(np, support, flower)

    assert np.array_equal(support, support_before)
    assert np.array_equal(flower, flower_before)
    assert np.array_equal(background, [[False, False, True], [False, False, True]])
    assert np.array_equal(editable, [[True, False, False], [False, True, False]])
    assert not np.any(np.logical_and(background, editable))
