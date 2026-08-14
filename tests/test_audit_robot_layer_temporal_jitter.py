from __future__ import annotations

import pytest

from scripts.audit_robot_layer_temporal_jitter import _high_jitter_count


def test_high_jitter_count_uses_strict_frozen_threshold() -> None:
    assert _high_jitter_count([19.9, 20.0, 20.1, 25.0], 20.0) == 2


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("inf"), float("nan")])
def test_high_jitter_count_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _high_jitter_count([1.0], threshold)
