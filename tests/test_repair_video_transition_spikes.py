from __future__ import annotations

import pytest

from scripts.repair_video_transition_spikes import (
    _energy_summary,
    group_transition_frames,
    intervals_overlap,
    motion_bridge,
)


def test_group_transition_frames_collapses_consecutive_values() -> None:
    assert group_transition_frames([645, 228, 227, 498, 498]) == [
        (227, 228),
        (498, 498),
        (645, 645),
    ]


def test_group_transition_frames_rejects_zero() -> None:
    with pytest.raises(ValueError, match="positive"):
        group_transition_frames([0])


def test_half_open_interval_overlap() -> None:
    assert intervals_overlap(284, 289, 259, 297)
    assert not intervals_overlap(297, 302, 259, 297)
    assert not intervals_overlap(250, 259, 259, 297)


def test_crossfade_bridge_preserves_shape_and_orders_intensity() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    first = np.zeros((8, 10, 3), dtype=np.uint8)
    second = np.full((8, 10, 3), 100, dtype=np.uint8)

    frames = motion_bridge(cv2, np, first, second, 3, mode="crossfade")

    assert len(frames) == 3
    assert all(frame.shape == first.shape for frame in frames)
    means = [float(frame.mean()) for frame in frames]
    assert means == sorted(means)
    assert 0 < means[0] < means[-1] < 100


def test_energy_summary_uses_one_based_transition_frames() -> None:
    np = pytest.importorskip("numpy")

    summary = _energy_summary(np, [0.5, 2.0, 1.0], [2])

    assert summary["median_transition"] == 1.0
    assert summary["maximum_transition"] == 2.0
    assert summary["maximum_transition_ratio"] == 2.0
    assert summary["selected_transition_energy"] == {"2": 2.0}
