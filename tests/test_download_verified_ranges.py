from __future__ import annotations

import pytest

from scripts.download_verified_ranges import plan_ranges


def test_plan_ranges_covers_bytes_without_gaps() -> None:
    ranges = plan_ranges(10, 3)
    assert ranges == ((0, 3), (4, 6), (7, 9))
    assert sum(end - start + 1 for start, end in ranges) == 10


def test_plan_ranges_caps_workers_to_bytes() -> None:
    assert plan_ranges(3, 8) == ((0, 0), (1, 1), (2, 2))


def test_plan_ranges_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        plan_ranges(0, 2)
