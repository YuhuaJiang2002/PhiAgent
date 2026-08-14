"""Part-local temporal appearance and explicit flower/robot z-order contracts."""

from __future__ import annotations

from math import isfinite
from typing import Any


def _odd_positive(value: int, label: str) -> None:
    if value < 1 or value % 2 == 0:
        raise ValueError(f"{label} must be a positive odd integer")


def right_arm_flower_partition(
    cv2: Any,
    np: Any,
    *,
    right_arm: Any,
    flower_visible: Any,
    hand_support: Any,
    corridor_dilation_pixels: int,
    hand_dilation_pixels: int,
) -> tuple[Any, Any, Any]:
    """Partition one camera frame into editable arm and source-flower owner.

    The resolved source-visible flower is authoritative inside the arm/flower
    corridor.  The robot arm owns the remaining tracked arm support, except for
    a protected hand band whose projected contact must not be rewritten.
    """

    if not (
        right_arm.shape == flower_visible.shape == hand_support.shape
        and right_arm.ndim == 2
    ):
        raise ValueError("right arm, flower, and hand masks must share one image plane")
    _odd_positive(corridor_dilation_pixels, "corridor dilation")
    _odd_positive(hand_dilation_pixels, "hand dilation")
    corridor_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (corridor_dilation_pixels, corridor_dilation_pixels),
    )
    hand_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (hand_dilation_pixels, hand_dilation_pixels),
    )
    arm = np.asarray(right_arm, dtype=np.bool_)
    flower = np.asarray(flower_visible, dtype=np.bool_)
    hand = np.asarray(hand_support, dtype=np.bool_)
    corridor = cv2.dilate(arm.astype(np.uint8), corridor_kernel) > 0
    flower_owner = np.logical_and(flower, corridor)
    protected_hand = cv2.dilate(hand.astype(np.uint8), hand_kernel) > 0
    flower_guard = cv2.dilate(flower_owner.astype(np.uint8), corridor_kernel) > 0
    arm_editable = np.logical_and.reduce(
        (arm, np.logical_not(protected_hand), np.logical_not(flower_guard))
    )
    return arm_editable, flower_owner, protected_hand


def source_owned_flower_restore_mask(
    cv2: Any,
    np: Any,
    *,
    flower_owner: Any,
    person: Any,
    hand_core: Any,
    protected_hand: Any,
    clean_plate_padding_pixels: int,
    sample_footprint_pixels: int,
) -> Any:
    """Return a sampling-safe source-flower ownership mask.

    The full reconstruction footprint around an owned flower sample is
    authoritative even where person or hand masks overlap.  Wider codec
    padding remains conservative and excludes those protected regions.
    """

    if not (
        flower_owner.shape == person.shape == hand_core.shape == protected_hand.shape
        and flower_owner.ndim == 2
    ):
        raise ValueError(
            "flower, person, hand core, and protected hand must share an image plane"
        )
    _odd_positive(clean_plate_padding_pixels, "clean-plate padding")
    _odd_positive(sample_footprint_pixels, "sample footprint")
    owner = np.asarray(flower_owner, dtype=np.bool_)
    padding_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (clean_plate_padding_pixels, clean_plate_padding_pixels),
    )
    footprint_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (sample_footprint_pixels, sample_footprint_pixels),
    )
    safe_padding = np.logical_and.reduce(
        (
            cv2.dilate(owner.astype(np.uint8), padding_kernel) > 0,
            np.logical_not(np.asarray(person, dtype=np.bool_)),
            np.logical_not(np.asarray(protected_hand, dtype=np.bool_)),
        )
    )
    sample_footprint = (
        cv2.dilate(owner.astype(np.uint8), footprint_kernel) > 0
    )
    sample_footprint = np.logical_and(
        sample_footprint,
        np.logical_not(np.asarray(hand_core, dtype=np.bool_)),
    )
    return np.logical_or.reduce((owner, sample_footprint, safe_padding))


def projected_contact_evidence_lock(
    cv2: Any,
    np: Any,
    *,
    candidate: Any,
    source: Any,
    hand_core: Any,
    flower_owner: Any,
    replacement_threshold: float,
    contact_dilation_pixels: int,
) -> Any:
    """Lock existing generated evidence in a source-observed contact corridor."""

    if candidate.shape != source.shape or candidate.ndim != 3:
        raise ValueError("candidate and source must share one HxWxC frame")
    if not (
        hand_core.shape == flower_owner.shape == candidate.shape[:2]
        and hand_core.ndim == 2
    ):
        raise ValueError("hand and flower masks must match the video image plane")
    if not isfinite(replacement_threshold) or replacement_threshold <= 0:
        raise ValueError("replacement threshold must be finite and positive")
    _odd_positive(contact_dilation_pixels, "contact dilation")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (contact_dilation_pixels, contact_dilation_pixels),
    )
    hand_band = cv2.dilate(
        np.asarray(hand_core, dtype=np.uint8), kernel
    ) > 0
    flower_band = cv2.dilate(
        np.asarray(flower_owner, dtype=np.uint8), kernel
    ) > 0
    replaced = np.max(
        np.abs(candidate.astype(np.int16) - source.astype(np.int16)),
        axis=2,
    ) > replacement_threshold
    return np.logical_and.reduce(
        (
            hand_band,
            flower_band,
            replaced,
            np.logical_not(np.asarray(flower_owner, dtype=np.bool_)),
        )
    )


def source_motion_residual_median_update(
    np: Any,
    *,
    current_candidate: Any,
    current_residual: Any,
    warped_previous_residual: Any,
    warped_next_residual: Any,
    reliable: Any,
    maximum_residual_delta: float,
) -> tuple[Any, dict[str, float]]:
    """Replace a part-local temporal residual extremum by a bounded median."""

    expected = current_candidate.shape
    for label, value in (
        ("current residual", current_residual),
        ("warped previous residual", warped_previous_residual),
        ("warped next residual", warped_next_residual),
    ):
        if value.shape != expected:
            raise ValueError(f"{label} must match the HxWx3 candidate")
    if reliable.shape != expected[:2]:
        raise ValueError("reliable mask must match the candidate image plane")
    if not isfinite(maximum_residual_delta) or maximum_residual_delta <= 0:
        raise ValueError("maximum residual delta must be finite and positive")

    current = current_residual.astype(np.float32)
    target = np.median(
        np.stack(
            (
                warped_previous_residual.astype(np.float32),
                current,
                warped_next_residual.astype(np.float32),
            ),
            axis=0,
        ),
        axis=0,
    )
    active = np.asarray(reliable, dtype=np.bool_)
    correction = np.clip(
        target - current,
        -maximum_residual_delta,
        maximum_residual_delta,
    )
    repaired = current_candidate.copy()
    updated = np.clip(
        np.rint(current_candidate.astype(np.float32) + correction), 0, 255
    ).astype(np.uint8)
    repaired[active] = updated[active]
    applied = np.abs(correction)[active]
    baseline = np.abs(current - target)[active]
    repaired_residual = current.copy()
    repaired_residual[active] += correction[active]
    post = np.abs(repaired_residual - target)[active]
    return repaired, {
        "active_fraction": float(np.mean(active)),
        "mean_abs_applied_correction": (
            float(np.mean(applied)) if applied.size else 0.0
        ),
        "maximum_abs_applied_correction": (
            float(np.max(applied)) if applied.size else 0.0
        ),
        "baseline_temporal_extremum_mae": (
            float(np.mean(baseline)) if baseline.size else 0.0
        ),
        "repaired_temporal_extremum_mae": (
            float(np.mean(post)) if post.size else 0.0
        ),
    }
