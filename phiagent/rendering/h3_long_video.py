"""Dependency-light seam and color helpers for overlapping H3 video windows."""

from __future__ import annotations

from typing import Any


def overlap_interval(
    first_start: int,
    first_count: int,
    second_start: int,
    second_count: int,
) -> tuple[int, int]:
    """Return the non-empty half-open overlap of two frame intervals."""

    start = max(first_start, second_start)
    end = min(first_start + first_count, second_start + second_count)
    if end <= start:
        raise ValueError("frame intervals do not overlap")
    return start, end


def estimate_subject_color_offset(
    np: Any,
    *,
    reference: list[Any],
    reference_start: int,
    candidate: list[Any],
    candidate_start: int,
    subject_masks: list[Any],
    object_masks: list[Any],
    maximum_offset: float = 12.0,
) -> tuple[float, float, float]:
    """Estimate a bounded BGR offset only on the robot support in the overlap."""

    start, end = overlap_interval(
        reference_start,
        len(reference),
        candidate_start,
        len(candidate),
    )
    samples = []
    for frame in range(start, end, 3):
        mask = (subject_masks[frame] > 0) & ~(object_masks[frame] > 0)
        if np.count_nonzero(mask) < 64:
            continue
        first = reference[frame - reference_start][mask].astype(np.float32)
        second = candidate[frame - candidate_start][mask].astype(np.float32)
        samples.append(np.median(first - second, axis=0))
    if not samples:
        return (0.0, 0.0, 0.0)
    offset = np.clip(np.median(np.asarray(samples), axis=0), -maximum_offset, maximum_offset)
    return tuple(float(value) for value in offset)


def apply_subject_color_offset(
    np: Any,
    *,
    frames: list[Any],
    start_frame: int,
    subject_masks: list[Any],
    object_masks: list[Any],
    offset: tuple[float, float, float],
) -> list[Any]:
    """Apply an offset inside the subject while leaving background/objects exact."""

    correction = np.asarray(offset, dtype=np.float32)
    result = []
    for local, frame in enumerate(frames):
        absolute = start_frame + local
        mask = (subject_masks[absolute] > 0) & ~(object_masks[absolute] > 0)
        adjusted = frame.copy()
        adjusted[mask] = np.clip(
            frame[mask].astype(np.float32) + correction,
            0,
            255,
        ).astype(np.uint8)
        result.append(adjusted)
    return result


def select_masked_seam(
    np: Any,
    *,
    current: list[Any],
    current_start: int,
    following: list[Any],
    following_start: int,
    source: list[Any],
    subject_masks: list[Any],
    minimum_frame: int | None = None,
    maximum_frame_exclusive: int | None = None,
) -> tuple[int, float]:
    """Choose a hard seam with minimum robot jump and source-motion error."""

    start, end = overlap_interval(
        current_start,
        len(current),
        following_start,
        len(following),
    )
    start = max(start + 1, minimum_frame if minimum_frame is not None else start + 1)
    end = min(end, maximum_frame_exclusive if maximum_frame_exclusive is not None else end)
    if end <= start:
        raise ValueError("seam search range is empty")
    scored = []
    for seam in range(start, end):
        mask = (subject_masks[seam - 1] > 0) | (subject_masks[seam] > 0)
        if np.count_nonzero(mask) < 1:
            mask = np.ones(subject_masks[seam].shape, dtype=bool)
        before = current[seam - 1 - current_start].astype(np.float32)
        after = following[seam - following_start].astype(np.float32)
        source_before = source[seam - 1].astype(np.float32)
        source_after = source[seam].astype(np.float32)
        candidate_motion = np.abs(after - before)[mask]
        source_motion = np.abs(source_after - source_before)[mask]
        jump = float(np.mean(candidate_motion))
        motion_error = float(np.mean(np.abs(candidate_motion - source_motion)))
        scored.append((jump + 0.75 * motion_error, seam))
    cost, seam = min(scored)
    return seam, cost


def merge_at_masked_seam(
    np: Any,
    *,
    current: list[Any],
    current_start: int,
    following: list[Any],
    following_start: int,
    source: list[Any],
    subject_masks: list[Any],
    minimum_frame: int | None = None,
    maximum_frame_exclusive: int | None = None,
) -> tuple[list[Any], dict[str, float | int]]:
    """Merge overlapping clips without cross-dissolving robot identities."""

    seam, cost = select_masked_seam(
        np,
        current=current,
        current_start=current_start,
        following=following,
        following_start=following_start,
        source=source,
        subject_masks=subject_masks,
        minimum_frame=minimum_frame,
        maximum_frame_exclusive=maximum_frame_exclusive,
    )
    return (
        current[: seam - current_start] + following[seam - following_start :],
        {
            "current_start": current_start,
            "following_start": following_start,
            "seam_frame": seam,
            "seam_cost": cost,
        },
    )


def overlap_continuity_metrics(
    np: Any,
    *,
    previous: list[Any],
    previous_start: int,
    following: list[Any],
    following_start: int,
    subject_mask: Any,
) -> dict[str, Any]:
    """Measure same-time disagreement and the least-visible hard transition."""

    start, end = overlap_interval(
        previous_start,
        len(previous),
        following_start,
        len(following),
    )
    selected = subject_mask > 0
    if np.count_nonzero(selected) < 1:
        raise ValueError("subject_mask must contain selected pixels")
    same_time = []
    for frame in range(start, end):
        first = previous[frame - previous_start].astype(np.float32)
        second = following[frame - following_start].astype(np.float32)
        same_time.append(float(np.mean(np.abs(first[selected] - second[selected]))))
    seams = []
    for seam in range(start + 1, end):
        before = previous[seam - 1 - previous_start].astype(np.float32)
        after = following[seam - following_start].astype(np.float32)
        seams.append(
            {
                "frame": seam,
                "subject_mad": float(np.mean(np.abs(before[selected] - after[selected]))),
            }
        )
    best = min(seams, key=lambda item: item["subject_mad"])
    return {
        "overlap_start": start,
        "overlap_end_exclusive": end,
        "overlap_frames": end - start,
        "mean_same_time_subject_mad": float(np.mean(same_time)),
        "maximum_same_time_subject_mad": max(same_time),
        "best_seam_frame": best["frame"],
        "best_seam_subject_mad": best["subject_mad"],
        "seams": seams,
    }
