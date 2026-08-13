from __future__ import annotations

import pytest

from scripts.repair_robot_hand_layer import expanded_repair_frames


def test_repair_intervals_group_nearby_failures_and_add_padding() -> None:
    assert expanded_repair_frames(
        [10, 12, 20], total_frames=30, padding=2, maximum_gap=3
    ) == tuple(list(range(8, 15)) + list(range(18, 23)))


def test_repair_intervals_reject_out_of_timeline_frame() -> None:
    with pytest.raises(ValueError, match="outside"):
        expanded_repair_frames(
            [30], total_frames=30, padding=1, maximum_gap=3
        )
