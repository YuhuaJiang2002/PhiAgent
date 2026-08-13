from __future__ import annotations

import numpy as np
import pytest

from scripts.stitch_strict_flower_expansion import compose_frames


def _frames(values: list[int]) -> list[np.ndarray]:
    return [np.full((2, 3, 3), value, dtype=np.uint8) for value in values]


def test_compose_preserves_global_mapping_and_uses_bounded_fade() -> None:
    result, record = compose_frames(
        np, _frames([10, 11, 12]), _frames([20, 21, 22, 23, 24]), _frames([30, 31, 32, 33]),
        prefix_global_start=100, left_global_start=102, right_global_start=104,
        left_right_cut_global=105, fade_weights=[0.75, 0.25],
    )
    assert [int(frame[0, 0, 0]) for frame in result] == [10, 11, 12, 14, 20, 23, 32, 33]
    assert record["global_range_inclusive"] == [100, 107]
    assert record["fade_global_range_inclusive"] == [103, 104]


def test_compose_rejects_increasing_fade_weights() -> None:
    with pytest.raises(ValueError, match="monotonically"):
        compose_frames(
            np, _frames([1]), _frames([2, 3]), _frames([4, 5]),
            prefix_global_start=0, left_global_start=1, right_global_start=2,
            left_right_cut_global=2, fade_weights=[0.2, 0.5],
        )
