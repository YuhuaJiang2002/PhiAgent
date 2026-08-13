from __future__ import annotations

import pytest

from scripts.build_h3_continuous_task_candidates import (
    _decayed_affine,
    _nearest_indices,
    _parse_action,
    _select_seam_frame,
)


def test_nearest_retiming_has_exact_endpoints_without_interpolation() -> None:
    np = pytest.importorskip("numpy")
    indices = _nearest_indices(45, 123, 117, np)

    assert len(indices) == 117
    assert int(indices[0]) == 45
    assert int(indices[-1]) == 123
    assert bool(np.all(indices[1:] >= indices[:-1]))
    assert set(np.diff(indices)).issubset({0, 1})


def test_state_valid_interval_selects_lowest_interaction_mad() -> None:
    np = pytest.importorskip("numpy")
    previous = np.zeros((10, 12, 3), dtype=np.uint8)
    following = [np.full_like(previous, value) for value in (30, 10, 20)]

    selected, candidates = _select_seam_frame(previous, following, 0, 2, np)

    assert selected == 1
    assert len(candidates) == 3


def test_camera_alignment_decays_exactly_to_identity() -> None:
    np = pytest.importorskip("numpy")
    warp = np.asarray([[0.99, 0.05, 12.0], [-0.05, 0.99, -8.0]], dtype=np.float32)

    assert np.allclose(_decayed_affine(warp, 0, 5, np), warp)
    assert np.allclose(_decayed_affine(warp, 4, 5, np), np.eye(2, 3))


def test_action_parser_rejects_reversed_interval() -> None:
    assert _parse_action("unscrew-bottle-cap=45,70") == (
        "unscrew-bottle-cap",
        45,
        70,
    )
    with pytest.raises(Exception, match="0 <= START <= END"):
        _parse_action("rinse-bottle=30,10")
