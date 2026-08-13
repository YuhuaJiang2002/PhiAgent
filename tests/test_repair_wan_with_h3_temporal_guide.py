from __future__ import annotations

import pytest

from scripts.repair_wan_with_h3_temporal_guide import (
    detect_guided_anomalies,
    guided_crossfade,
    guided_progress,
    merge_overlapping_repair_groups,
)


def test_detector_selects_candidate_only_spike() -> None:
    candidate = [1.0] * 20
    guide = [1.0] * 20
    source = [1.0] * 20
    candidate[9] = 5.0

    selected, diagnostics = detect_guided_anomalies(
        candidate,
        guide,
        source,
        analysis_end_frame=20,
        repair_radius=2,
        minimum_guide_score=2.0,
        minimum_candidate_ratio=1.8,
        minimum_candidate_energy=1.0,
    )

    assert selected == [10]
    assert diagnostics[0]["guide_score"] == pytest.approx(5.0)


def test_merge_groups_only_when_changed_intervals_overlap() -> None:
    assert merge_overlapping_repair_groups([(31, 31), (35, 36)], 2) == [
        (31, 31),
        (35, 36),
    ]
    assert merge_overlapping_repair_groups([(216, 216), (219, 219), (223, 223)], 2) == [
        (216, 219),
        (223, 223),
    ]


def test_guided_progress_and_crossfade_are_monotonic() -> None:
    np = pytest.importorskip("numpy")
    progress = guided_progress(np, [1.0, 4.0, 1.0], [1.0, 1.0, 1.0])
    first = np.zeros((4, 5, 3), dtype=np.uint8)
    second = np.full_like(first, 100)

    result = guided_crossfade(np, first, second, progress)

    assert len(result) == 2
    means = [float(frame.mean()) for frame in result]
    assert means == sorted(means)
    assert 0 < means[0] < means[-1] < 100
