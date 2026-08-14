"""Part-local temporal appearance and explicit flower/robot z-order contracts."""

from __future__ import annotations

from math import isfinite
from typing import Any

from phiagent.rendering.object_factored_long_video import binary_dilate_square
from phiagent.rendering.robot_layer_contract import (
    mask_chebyshev_distance,
    replacement_mask,
)


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


def evidence_ordered_flower_front(
    np: Any,
    *,
    candidate: Any,
    source: Any,
    resolved_flower: Any,
    tracked_flower: Any,
    contested_support: Any,
    replacement_threshold: float,
) -> tuple[Any, Any]:
    """Resolve flower/robot depth from two independent image observations.

    Outside an explicit robot/object conflict corridor, the persistent object
    track owns the front layer.  Inside that corridor, conservative source
    visibility is authoritative and the wider track is allowed in front only
    where the incumbent composite still agrees with the source.  A track alone
    therefore cannot pull a flower through a generated robot arm, while real
    foreground flowers remain coloured over the torso and background.
    """

    rendered = np.asarray(candidate)
    reference = np.asarray(source)
    resolved = np.asarray(resolved_flower, dtype=np.bool_)
    tracked = np.asarray(tracked_flower, dtype=np.bool_)
    contested = np.asarray(contested_support, dtype=np.bool_)
    if not (
        rendered.shape == reference.shape
        and rendered.ndim == 3
        and rendered.shape[2] == 3
        and resolved.shape == tracked.shape == contested.shape == rendered.shape[:2]
    ):
        raise ValueError("videos and flower masks must share one image plane")
    if not isfinite(replacement_threshold) or replacement_threshold <= 0:
        raise ValueError("replacement threshold must be finite and positive")
    source_retained = np.max(
        np.abs(rendered.astype(np.int16) - reference.astype(np.int16)), axis=2
    ) <= replacement_threshold
    conservative_front = np.logical_or(
        resolved, np.logical_and(tracked, source_retained)
    )
    front = np.where(contested, conservative_front, tracked)
    return front, source_retained


def projected_contact_evidence_lock(
    cv2: Any,
    np: Any,
    *,
    candidate: Any,
    source: Any,
    hand_core: Any,
    tracked_object: Any,
    replacement_threshold: float,
    contact_radius: int,
    maximum_source_occlusion_gap: int,
) -> Any:
    """Lock generated evidence in an independently tracked contact corridor.

    ``tracked_object`` is deliberately the persistent object-state channel, not
    the resolved visible flower layer.  Visibility resolution removes object
    pixels hidden by the source hand, while those hidden pixels are exactly
    what define the bounded grasp corridor that must survive a later material
    projection.
    """

    if candidate.shape != source.shape or candidate.ndim != 3:
        raise ValueError("candidate and source must share one HxWxC frame")
    if not (
        hand_core.shape == tracked_object.shape == candidate.shape[:2]
        and hand_core.ndim == 2
    ):
        raise ValueError("hand and flower masks must match the video image plane")
    if not isfinite(replacement_threshold) or replacement_threshold <= 0:
        raise ValueError("replacement threshold must be finite and positive")
    if contact_radius < 0 or maximum_source_occlusion_gap < contact_radius:
        raise ValueError("invalid contact radius or maximum source occlusion gap")
    hand = np.asarray(hand_core, dtype=np.bool_)
    corridor = projected_contact_corridor(
        np,
        hand_core=hand,
        tracked_object=tracked_object,
        contact_radius=contact_radius,
        maximum_source_occlusion_gap=maximum_source_occlusion_gap,
    )
    replaced = replacement_mask(
        np,
        candidate,
        source,
        threshold=replacement_threshold,
    )
    return np.logical_and(corridor, replaced)


def projected_contact_corridor(
    np: Any,
    *,
    hand_core: Any,
    tracked_object: Any,
    contact_radius: int,
    maximum_source_occlusion_gap: int,
) -> Any:
    """Return the same bounded hand/object corridor as the frozen grasp audit."""

    hand = np.asarray(hand_core, dtype=np.bool_)
    tracked = np.asarray(tracked_object, dtype=np.bool_)
    if hand.ndim != 2 or tracked.shape != hand.shape:
        raise ValueError("hand and tracked object must share one 2-D image plane")
    if contact_radius < 0 or maximum_source_occlusion_gap < contact_radius:
        raise ValueError("invalid contact radius or maximum source occlusion gap")
    source_gap = mask_chebyshev_distance(
        np,
        hand,
        tracked,
        maximum_radius=maximum_source_occlusion_gap,
    )
    if source_gap is None:
        return np.zeros_like(hand)
    return np.logical_and(
        hand,
        binary_dilate_square(np, tracked, source_gap + contact_radius),
    )


def propagate_robot_material_residual(
    np: Any,
    *,
    projected: Any,
    candidate: Any,
    source: Any,
    corridor: Any,
    seed_mask: Any,
    replacement_threshold: float,
) -> tuple[Any, dict[str, float]]:
    """Complete source-hand leakage with the nearest observed robot residual.

    The bounded occlusion corridor comes from independent hand and object
    observations.  Missing pixels inherit the RGB *residual* of the nearest
    already-generated robot-hand seed instead of a fixed colour or a fitted
    score target.  This converts a partial hand/object bridge into one coherent
    material layer without changing any audit threshold.
    """

    result = np.asarray(projected, dtype=np.uint8).copy()
    incumbent = np.asarray(candidate, dtype=np.uint8)
    reference = np.asarray(source, dtype=np.uint8)
    corridor_value = np.asarray(corridor, dtype=np.bool_)
    seeds = np.asarray(seed_mask, dtype=np.bool_)
    if not (
        result.shape == incumbent.shape == reference.shape
        and result.ndim == 3
        and result.shape[2] == 3
        and corridor_value.shape == seeds.shape == result.shape[:2]
    ):
        raise ValueError("video planes, corridor, and seed mask must align")
    if not isfinite(replacement_threshold) or replacement_threshold <= 0:
        raise ValueError("replacement threshold must be finite and positive")
    current_residual = np.mean(
        np.abs(result.astype(np.int16) - reference.astype(np.int16)), axis=2
    )
    missing = np.logical_and(corridor_value, current_residual < replacement_threshold)
    seed_value = np.logical_and(seeds, corridor_value)
    target_coordinates = np.argwhere(missing)
    seed_coordinates = np.argwhere(seed_value)
    if target_coordinates.size and seed_coordinates.size:
        seed_residual = (
            incumbent.astype(np.int16) - reference.astype(np.int16)
        )[seed_value]
        for target in target_coordinates:
            squared_distance = np.sum(
                np.square(seed_coordinates - target[None, :]), axis=1
            )
            nearest = int(np.argmin(squared_distance))
            y, x = (int(target[0]), int(target[1]))
            result[y, x] = np.clip(
                reference[y, x].astype(np.int16) + seed_residual[nearest],
                0,
                255,
            ).astype(np.uint8)
    completed_residual = np.mean(
        np.abs(result.astype(np.int16) - reference.astype(np.int16)), axis=2
    )
    unresolved = np.logical_and(
        missing, completed_residual < replacement_threshold
    )
    return result, {
        "corridor_pixels": float(np.count_nonzero(corridor_value)),
        "seed_pixels": float(np.count_nonzero(seed_value)),
        "propagated_pixels": float(target_coordinates.shape[0]),
        "unresolved_source_like_pixels": float(np.count_nonzero(unresolved)),
    }


def reinforce_projected_contact_evidence(
    np: Any,
    *,
    projected: Any,
    candidate: Any,
    source: Any,
    evidence_mask: Any,
    replacement_threshold: float,
    codec_error_margin: float,
) -> tuple[Any, dict[str, float]]:
    """Give existing bridge pixels a bounded residual margin before encoding.

    The direction comes from the incumbent generated residual, so the method
    does not invent a new colour or contact.  Only evidence already above the
    frozen replacement threshold is eligible; its mean RGB residual is raised
    to ``threshold + margin`` when necessary so one lossy review encode cannot
    erase a marginal bridge pixel.
    """

    result = np.asarray(projected, dtype=np.uint8).copy()
    incumbent = np.asarray(candidate, dtype=np.uint8)
    reference = np.asarray(source, dtype=np.uint8)
    evidence = np.asarray(evidence_mask, dtype=np.bool_)
    if not (
        result.shape == incumbent.shape == reference.shape
        and result.ndim == 3
        and result.shape[2] == 3
        and evidence.shape == result.shape[:2]
    ):
        raise ValueError("projected, candidate, source, and evidence must share one image plane")
    if not isfinite(replacement_threshold) or replacement_threshold <= 0:
        raise ValueError("replacement threshold must be finite and positive")
    if not isfinite(codec_error_margin) or codec_error_margin < 0:
        raise ValueError("codec error margin must be finite and non-negative")
    residual = incumbent.astype(np.float32) - reference.astype(np.float32)
    magnitude = np.mean(np.abs(residual), axis=2)
    eligible = np.logical_and(evidence, magnitude >= replacement_threshold)
    target = replacement_threshold + codec_error_margin
    needs_margin = np.logical_and(eligible, magnitude < target)
    if np.any(needs_margin):
        scale = target / np.maximum(magnitude[needs_margin], 1e-6)
        reinforced = reference[needs_margin].astype(np.float32) + (
            residual[needs_margin] * scale[:, None]
        )
        result[needs_margin] = np.clip(np.rint(reinforced), 0, 255).astype(np.uint8)
    return result, {
        "eligible_pixels": float(np.count_nonzero(eligible)),
        "reinforced_pixels": float(np.count_nonzero(needs_margin)),
        "target_mean_rgb_residual": float(target),
    }


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
