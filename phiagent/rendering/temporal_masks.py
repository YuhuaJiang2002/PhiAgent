"""Pure mask contracts shared by long-video temporal appearance tools."""

from __future__ import annotations

from typing import Any


def _odd_positive_kernel(cv2: Any, pixels: int, label: str) -> Any:
    if pixels < 1 or pixels % 2 == 0:
        raise ValueError(f"{label} must be a positive odd integer")
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels, pixels))


def build_limb_contact_locks(
    cv2: Any,
    np: Any,
    *,
    limbs: Any,
    flower: Any,
    limb_dilation_pixels: int,
    contact_dilation_pixels: int,
) -> tuple[Any, Any]:
    """Build incumbent-exact arm/hand and hand-flower contact masks."""

    if limbs.shape != flower.shape:
        raise ValueError("limb and flower masks must share one camera-pixel frame")
    limb_kernel = _odd_positive_kernel(
        cv2, limb_dilation_pixels, "limb dilation"
    )
    contact_kernel = _odd_positive_kernel(
        cv2, contact_dilation_pixels, "contact dilation"
    )
    limb_lock = cv2.dilate(limbs.astype(np.uint8), limb_kernel) > 0
    contact_lock = np.logical_and(
        cv2.dilate(limbs.astype(np.uint8), contact_kernel) > 0,
        cv2.dilate(flower.astype(np.uint8), contact_kernel) > 0,
    )
    return limb_lock, contact_lock


def build_torso_head_whitelist(
    cv2: Any,
    np: Any,
    *,
    robot: Any,
    limbs: Any,
    flower: Any,
    limb_dilation_pixels: int,
    torso_erosion_pixels: int,
    contact_dilation_pixels: int,
) -> tuple[Any, Any, Any]:
    """Return editable head/torso plus explicit limb and contact locks."""

    if robot.shape != limbs.shape or robot.shape != flower.shape:
        raise ValueError(
            "robot, limb, and flower masks must share one named camera-pixel frame"
        )
    torso_kernel = _odd_positive_kernel(
        cv2, torso_erosion_pixels, "torso erosion"
    )
    limb_lock, contact_lock = build_limb_contact_locks(
        cv2,
        np,
        limbs=limbs,
        flower=flower,
        limb_dilation_pixels=limb_dilation_pixels,
        contact_dilation_pixels=contact_dilation_pixels,
    )
    robot_core = cv2.erode(robot.astype(np.uint8), torso_kernel) > 0
    editable = np.logical_and.reduce(
        (
            robot_core,
            np.logical_not(limb_lock),
            np.logical_not(contact_lock),
            np.logical_not(flower),
        )
    )
    return editable, limb_lock, contact_lock


def apply_temporal_lock_envelope(
    np: Any,
    *,
    editable: Any,
    adjacent_locked_masks: list[Any],
) -> Any:
    """Exclude adjacent-frame locks so protected temporal deltas stay exact."""

    if not adjacent_locked_masks:
        return editable.copy()
    if any(mask.shape != editable.shape for mask in adjacent_locked_masks):
        raise ValueError("temporal lock masks must match the editable camera frame")
    envelope = np.logical_or.reduce(adjacent_locked_masks)
    return np.logical_and(editable, np.logical_not(envelope))
