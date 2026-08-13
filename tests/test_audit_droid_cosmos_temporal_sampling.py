from __future__ import annotations

import pytest

from scripts.audit_droid_cosmos_temporal_sampling import (
    near_duplicate_fraction,
    percentile_nearest_rank,
)


def test_duplicate_fraction_detects_systematic_pairs() -> None:
    mads = [0.05, 2.0, 0.04, 1.8, 0.10, 3.0]
    assert near_duplicate_fraction(mads, 0.25) == pytest.approx(0.5)


def test_duplicate_fraction_rejects_single_frame() -> None:
    with pytest.raises(ValueError, match="at least one"):
        near_duplicate_fraction([], 0.25)


def test_nearest_rank_p90() -> None:
    assert percentile_nearest_rank([0.0, 0.1, 0.2, 0.3, 0.4], 0.9) == 0.4
