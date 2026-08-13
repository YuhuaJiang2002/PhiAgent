from __future__ import annotations

import pytest

from phiagent.rendering.h3_long_video import (
    apply_subject_color_offset,
    estimate_subject_color_offset,
    merge_at_masked_seam,
    overlap_continuity_metrics,
    select_masked_seam,
)


def test_subject_offset_never_changes_background_or_protected_objects() -> None:
    np = pytest.importorskip("numpy")
    masks = [np.zeros((10, 12), dtype=np.uint8) for _ in range(5)]
    objects = [np.zeros((10, 12), dtype=np.uint8) for _ in range(5)]
    for mask, protected in zip(masks, objects):
        mask[:, 2:11] = 255
        protected[1:3, 3] = 255
    reference = [np.full((10, 12, 3), 100, dtype=np.uint8) for _ in range(4)]
    candidate = [np.full((10, 12, 3), 90, dtype=np.uint8) for _ in range(4)]

    offset = estimate_subject_color_offset(
        np,
        reference=reference,
        reference_start=0,
        candidate=candidate,
        candidate_start=1,
        subject_masks=masks,
        object_masks=objects,
    )
    aligned = apply_subject_color_offset(
        np,
        frames=candidate,
        start_frame=1,
        subject_masks=masks,
        object_masks=objects,
        offset=offset,
    )

    assert offset == pytest.approx((10.0, 10.0, 10.0))
    assert int(aligned[0][0, 2, 0]) == 100
    assert int(aligned[0][0, 0, 0]) == 90
    assert int(aligned[0][1, 3, 0]) == 90


def test_masked_seam_selects_the_continuous_robot_transition() -> None:
    np = pytest.importorskip("numpy")
    current = [np.full((3, 4, 3), value, dtype=np.uint8) for value in range(8)]
    following = [
        np.full((3, 4, 3), value, dtype=np.uint8)
        for value in (4, 100, 5, 100, 8, 9, 10, 11)
    ]
    source = [np.full((3, 4, 3), value, dtype=np.uint8) for value in range(12)]
    masks = [np.full((3, 4), 255, dtype=np.uint8) for _ in range(12)]

    seam, cost = select_masked_seam(
        np,
        current=current,
        current_start=0,
        following=following,
        following_start=4,
        source=source,
        subject_masks=masks,
    )
    merged, record = merge_at_masked_seam(
        np,
        current=current,
        current_start=0,
        following=following,
        following_start=4,
        source=source,
        subject_masks=masks,
    )

    assert seam == 6
    assert cost == pytest.approx(0.75)
    assert record["seam_frame"] == 6
    assert len(merged) == 12


def test_overlap_continuity_metrics_reports_best_seam() -> None:
    np = pytest.importorskip("numpy")
    previous = [np.full((4, 5, 3), value, dtype=np.uint8) for value in range(8)]
    following = [
        np.full((4, 5, 3), value, dtype=np.uint8)
        for value in (40, 5, 30, 7, 8, 9, 10, 11)
    ]
    mask = np.full((4, 5), 255, dtype=np.uint8)

    metrics = overlap_continuity_metrics(
        np,
        previous=previous,
        previous_start=0,
        following=following,
        following_start=4,
        subject_mask=mask,
    )

    assert metrics["overlap_frames"] == 4
    assert metrics["best_seam_frame"] == 5
    assert metrics["best_seam_subject_mad"] == pytest.approx(1.0)
