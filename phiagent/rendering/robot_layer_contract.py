"""Explicit RGB-alpha-contact contract for long robot-replacement video.

This module intentionally receives a NumPy-like module from callers.  Merely
importing :mod:`phiagent` therefore does not pull in NumPy, OpenCV, Torch, or a
GPU runtime.

The contract separates three quantities that an RGB-only video generator tends
to entangle:

* robot appearance (RGB inside a named alpha support),
* robot topology (replacement coverage on tracked body/arm/hand supports), and
* interaction state (a projected hand/object contact event).

The projected contact check is a 2D image-space invariant.  It is deliberately
not described as force, collision, or executable-robot validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phiagent.rendering.object_factored_long_video import (
    binary_dilate_square,
    binary_erode_square,
    rgb_to_opencv_hsv,
    source_skin_like,
)


@dataclass(frozen=True)
class RobotLayerContract:
    """Names the camera and timeline frames of one RGB-alpha-contact layer."""

    camera_frame: str
    timeline: str
    width: int
    height: int
    contact_radius_pixels: int = 3

    def validate(self) -> None:
        if not self.camera_frame.strip():
            raise ValueError("robot layer camera frame must be named")
        if not self.timeline.strip():
            raise ValueError("robot layer timeline must be named")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("robot layer dimensions must be positive")
        if self.contact_radius_pixels < 0:
            raise ValueError("contact radius must be non-negative")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return {
            "camera_frame": self.camera_frame,
            "timeline": self.timeline,
            "width": self.width,
            "height": self.height,
            "contact_radius_pixels": self.contact_radius_pixels,
            "control_channel_r": "robot_alpha",
            "control_channel_g": "object_boundary",
            "control_channel_b": "required_contact_region",
        }


def _as_mask(np: Any, value: Any, shape: tuple[int, int], label: str) -> Any:
    result = np.asarray(value, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"{label} shape {result.shape} does not match {shape}")
    return result


def object_boundary(np: Any, object_mask: Any, radius: int = 1) -> Any:
    """Return a scale-explicit two-sided binary boundary."""

    if radius < 1:
        raise ValueError("object boundary radius must be at least one pixel")
    mask = np.asarray(object_mask, dtype=bool)
    return np.logical_xor(
        binary_dilate_square(np, mask, radius),
        binary_erode_square(np, mask, radius),
    )


def projected_contact_required(
    np: Any,
    hand_mask: Any,
    object_mask: Any,
    *,
    radius: int,
) -> bool:
    """Whether tracked hand and object supports require projected contact."""

    hand = np.asarray(hand_mask, dtype=bool)
    object_value = np.asarray(object_mask, dtype=bool)
    if hand.shape != object_value.shape:
        raise ValueError("hand and object masks must share one camera frame")
    overlap = np.logical_and(binary_dilate_square(np, hand, radius), object_value)
    return bool(np.any(overlap))


def make_state_control(
    np: Any,
    *,
    robot_alpha: Any,
    hand_mask: Any,
    object_mask: Any,
    contact_radius: int,
    marker_radius: int = 2,
) -> Any:
    """Encode alpha, object boundary, and contact state as an RGB control.

    Red is the robot alpha support, green is the object boundary, and blue is
    the hand/object neighbourhood only on frames where contact is required.
    This is a stable training/inference interface rather than an aesthetic edge
    map whose semantics change from clip to clip.
    """

    alpha = np.asarray(robot_alpha, dtype=bool)
    if alpha.ndim != 2:
        raise ValueError("robot alpha must be a two-dimensional mask")
    hand = _as_mask(np, hand_mask, alpha.shape, "hand")
    object_value = _as_mask(np, object_mask, alpha.shape, "object")
    boundary = object_boundary(np, object_value)
    contact_neighbourhood = np.logical_and(
        binary_dilate_square(np, hand, contact_radius),
        binary_dilate_square(np, object_value, contact_radius),
    )
    if np.any(contact_neighbourhood):
        contact_neighbourhood = binary_dilate_square(
            np, contact_neighbourhood, marker_radius
        )
    control = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    control[..., 0] = alpha.astype(np.uint8) * 255
    control[..., 1] = boundary.astype(np.uint8) * 255
    control[..., 2] = contact_neighbourhood.astype(np.uint8) * 255
    return control


def replacement_mask(
    np: Any,
    candidate_rgb: Any,
    source_rgb: Any,
    *,
    threshold: float,
) -> Any:
    """Return pixels whose mean absolute RGB change reaches ``threshold``."""

    candidate = np.asarray(candidate_rgb)
    source = np.asarray(source_rgb)
    if candidate.shape != source.shape or candidate.ndim != 3 or candidate.shape[2] != 3:
        raise ValueError("candidate and source must be matching HxWx3 frames")
    delta = np.abs(candidate.astype(np.int16) - source.astype(np.int16))
    return delta.mean(axis=2) >= float(threshold)


def canonical_palette_histogram(
    np: Any,
    frame_rgb: Any,
    mask: Any,
    *,
    bins: int = 8,
    pseudocount: float = 1.0,
) -> Any:
    """Build a smoothed quantized RGB distribution for robot identity."""

    if bins < 2 or bins > 32:
        raise ValueError("palette bins must be between 2 and 32")
    frame = np.asarray(frame_rgb, dtype=np.uint8)
    selected = _as_mask(np, mask, frame.shape[:2], "palette")
    if not np.any(selected):
        raise ValueError("canonical palette mask is empty")
    quantized = np.minimum(frame[selected].astype(np.int64) * bins // 256, bins - 1)
    indices = quantized[:, 0] * bins * bins + quantized[:, 1] * bins + quantized[:, 2]
    counts = np.bincount(indices, minlength=bins**3).astype(np.float64)
    counts += float(pseudocount)
    return counts / counts.sum()


def palette_surprisal(
    np: Any,
    frame_rgb: Any,
    mask: Any,
    palette: Any,
) -> float:
    """Mean negative log likelihood under a canonical RGB palette."""

    frame = np.asarray(frame_rgb, dtype=np.uint8)
    selected = _as_mask(np, mask, frame.shape[:2], "palette evaluation")
    if not np.any(selected):
        return float("inf")
    distribution = np.asarray(palette, dtype=np.float64)
    bins_float = round(float(len(distribution)) ** (1.0 / 3.0))
    bins = int(bins_float)
    if bins**3 != len(distribution) or np.any(distribution <= 0):
        raise ValueError("palette must be a positive cubic RGB histogram")
    quantized = np.minimum(frame[selected].astype(np.int64) * bins // 256, bins - 1)
    indices = quantized[:, 0] * bins * bins + quantized[:, 1] * bins + quantized[:, 2]
    return float((-np.log(distribution[indices])).mean())


def _coverage(np: Any, replacement: Any, support: Any) -> float:
    denominator = int(np.count_nonzero(support))
    if denominator == 0:
        return 1.0
    overlap = np.logical_and(replacement, support)
    return float(np.count_nonzero(overlap) / denominator)


def _spatial_chroma_tv(np: Any, frame_rgb: Any, mask: Any) -> float:
    _, saturation, _ = rgb_to_opencv_hsv(np, frame_rgb)
    horizontal = np.logical_and(mask[:, 1:], mask[:, :-1])
    vertical = np.logical_and(mask[1:, :], mask[:-1, :])
    values = []
    if np.any(horizontal):
        values.append(np.abs(saturation[:, 1:] - saturation[:, :-1])[horizontal])
    if np.any(vertical):
        values.append(np.abs(saturation[1:, :] - saturation[:-1, :])[vertical])
    if not values:
        return 0.0
    return float(np.concatenate(values).mean())


def _spatial_luma_edge_energy(np: Any, frame_rgb: Any, mask: Any) -> float:
    frame = np.asarray(frame_rgb, dtype=np.float32)
    luma = 0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2]
    horizontal = np.logical_and(mask[:, 1:], mask[:, :-1])
    vertical = np.logical_and(mask[1:, :], mask[:-1, :])
    values = []
    if np.any(horizontal):
        values.append(np.abs(luma[:, 1:] - luma[:, :-1])[horizontal])
    if np.any(vertical):
        values.append(np.abs(luma[1:, :] - luma[:-1, :])[vertical])
    if not values:
        return 0.0
    return float(np.concatenate(values).mean())


def _grid_topology_coverage(
    np: Any,
    replacement: Any,
    support: Any,
    *,
    rows: int = 4,
    columns: int = 4,
    cell_threshold: float = 0.45,
) -> float:
    ys, xs = np.nonzero(support)
    if not len(xs):
        return 1.0
    y_edges = np.linspace(int(ys.min()), int(ys.max()) + 1, rows + 1, dtype=int)
    x_edges = np.linspace(int(xs.min()), int(xs.max()) + 1, columns + 1, dtype=int)
    present = 0
    accepted = 0
    for row in range(rows):
        for column in range(columns):
            cell_support = support[
                y_edges[row] : y_edges[row + 1],
                x_edges[column] : x_edges[column + 1],
            ]
            if int(np.count_nonzero(cell_support)) < 4:
                continue
            cell_replacement = replacement[
                y_edges[row] : y_edges[row + 1],
                x_edges[column] : x_edges[column + 1],
            ]
            present += 1
            accepted += _coverage(np, cell_replacement, cell_support) >= cell_threshold
    return float(accepted / present) if present else 1.0


def frame_contract_metrics(
    np: Any,
    *,
    candidate_rgb: Any,
    source_rgb: Any,
    robot_alpha: Any,
    arm_support: Any,
    hand_support: Any,
    object_mask: Any,
    palette: Any,
    replacement_threshold: float = 12.0,
    contact_radius: int = 3,
) -> dict[str, float | bool]:
    """Measure appearance, topology, and projected contact for one frame."""

    candidate = np.asarray(candidate_rgb, dtype=np.uint8)
    source = np.asarray(source_rgb, dtype=np.uint8)
    shape = candidate.shape[:2]
    alpha = _as_mask(np, robot_alpha, shape, "robot alpha")
    arms = _as_mask(np, arm_support, shape, "arm support")
    hands = _as_mask(np, hand_support, shape, "hand support")
    object_value = _as_mask(np, object_mask, shape, "object")
    evaluation = np.logical_and(alpha, np.logical_not(object_value))
    replaced = np.logical_and(
        replacement_mask(np, candidate, source, threshold=replacement_threshold),
        evaluation,
    )
    _, saturation, value = rgb_to_opencv_hsv(np, candidate)
    high_chroma = np.logical_and(saturation >= 110.0, value >= 48.0)
    contact_required = projected_contact_required(
        np, hands, object_value, radius=contact_radius
    )
    robot_hand = np.logical_and(replaced, hands)
    contact_observed = bool(
        np.any(
            np.logical_and(
                binary_dilate_square(np, robot_hand, contact_radius), object_value
            )
        )
    )
    skin = np.logical_and(source_skin_like(np, candidate), evaluation)
    hand_evaluation = np.logical_and(hands, evaluation)
    hand_edge_energy = _spatial_luma_edge_energy(
        np, candidate, hand_evaluation
    )
    denominator = max(1, int(np.count_nonzero(evaluation)))
    return {
        "palette_surprisal": palette_surprisal(
            np, candidate, evaluation, palette
        ),
        "high_chroma_fraction": float(
            np.count_nonzero(np.logical_and(high_chroma, evaluation)) / denominator
        ),
        "skin_like_fraction": float(np.count_nonzero(skin) / denominator),
        "spatial_chroma_tv": _spatial_chroma_tv(np, candidate, evaluation),
        # Two names intentionally expose one two-sided morphology statistic to
        # the generic one-sided audit-limit machinery.
        "hand_edge_energy_lower_gate": hand_edge_energy,
        "hand_edge_energy_upper_gate": hand_edge_energy,
        "replacement_coverage": _coverage(np, replaced, evaluation),
        "arm_replacement_coverage": _coverage(
            np, replaced, np.logical_and(arms, evaluation)
        ),
        "hand_replacement_coverage": _coverage(np, replaced, hand_evaluation),
        "grid_topology_coverage": _grid_topology_coverage(
            np, replaced, evaluation
        ),
        "contact_required": contact_required,
        "contact_observed": contact_observed,
        "contact_pass": (not contact_required) or contact_observed,
    }


def merge_missing_replacement(
    np: Any,
    *,
    base_rgb: Any,
    donor_rgb: Any,
    source_rgb: Any,
    hand_support: Any,
    protected_object: Any,
    replacement_threshold: float,
    expansion_radius: int = 1,
    maximum_color_offset: float = 30.0,
) -> tuple[Any, Any, tuple[float, float, float]]:
    """Fill only missing hand replacement pixels from a second candidate.

    Donor pixels are eligible only where the donor replaces the source and the
    base does not.  A bounded per-channel offset is estimated on pixels where
    both candidates already replace the hand, avoiding an unconstrained global
    colour transfer.  The protected object is never touched.
    """

    base = np.asarray(base_rgb, dtype=np.uint8)
    donor = np.asarray(donor_rgb, dtype=np.uint8)
    source = np.asarray(source_rgb, dtype=np.uint8)
    if base.shape != donor.shape or base.shape != source.shape:
        raise ValueError("base, donor, and source RGB frames must have equal shape")
    hand = _as_mask(np, hand_support, base.shape[:2], "hand support")
    protected = _as_mask(np, protected_object, base.shape[:2], "protected object")
    base_replaced = replacement_mask(
        np, base, source, threshold=replacement_threshold
    )
    donor_replaced = replacement_mask(
        np, donor, source, threshold=replacement_threshold
    )
    missing = np.logical_and.reduce(
        (hand, donor_replaced, np.logical_not(base_replaced), np.logical_not(protected))
    )
    if expansion_radius:
        missing = np.logical_and.reduce(
            (
                binary_dilate_square(np, missing, expansion_radius),
                hand,
                donor_replaced,
                np.logical_not(protected),
            )
        )
    overlap = np.logical_and.reduce(
        (hand, donor_replaced, base_replaced, np.logical_not(protected))
    )
    if int(np.count_nonzero(overlap)) >= 8:
        offset = np.median(base[overlap].astype(np.float32), axis=0) - np.median(
            donor[overlap].astype(np.float32), axis=0
        )
        offset = np.clip(offset, -maximum_color_offset, maximum_color_offset)
    else:
        offset = np.zeros(3, dtype=np.float32)
    result = base.copy()
    adjusted = np.clip(donor.astype(np.float32) + offset, 0, 255).astype(np.uint8)
    result[missing] = adjusted[missing]
    return result, missing, tuple(float(value) for value in offset)


def project_missing_contact(
    np: Any,
    *,
    candidate_rgb: Any,
    source_rgb: Any,
    hand_support: Any,
    protected_object: Any,
    replacement_threshold: float,
    contact_radius: int,
    maximum_bridge_steps: int = 6,
) -> tuple[Any, Any, int, bool]:
    """Extend an existing generated hand through its source-supported contact fringe.

    The projection is deliberately narrow: it can only grow already replaced
    pixels through tracked hand support, never writes the protected object, and
    stops as soon as the fixed projected-contact invariant is satisfied.  It is
    an image-space continuity repair, not depth or force evidence.
    """

    candidate = np.asarray(candidate_rgb, dtype=np.uint8)
    source = np.asarray(source_rgb, dtype=np.uint8)
    if candidate.shape != source.shape or candidate.ndim != 3 or candidate.shape[2] != 3:
        raise ValueError("candidate and source RGB frames must be matching HxWx3 frames")
    if contact_radius < 0 or maximum_bridge_steps < 0:
        raise ValueError("contact radius and maximum bridge steps must be non-negative")
    hand = _as_mask(np, hand_support, candidate.shape[:2], "hand support")
    protected = _as_mask(np, protected_object, candidate.shape[:2], "protected object")
    result = candidate.copy()
    added = np.zeros(candidate.shape[:2], dtype=bool)
    required = projected_contact_required(
        np, hand, protected, radius=contact_radius
    )

    def replacement(value: Any) -> Any:
        return np.logical_and(
            replacement_mask(np, value, source, threshold=replacement_threshold),
            np.logical_and(hand, np.logical_not(protected)),
        )

    current = replacement(result)
    observed = bool(
        np.any(
            np.logical_and(
                binary_dilate_square(np, current, contact_radius), protected
            )
        )
    )
    if not required or observed or maximum_bridge_steps == 0:
        return result, added, 0, observed

    near_object = binary_dilate_square(
        np, protected, contact_radius + maximum_bridge_steps + 1
    )
    current = np.logical_and(current, near_object)
    corridor = np.logical_and.reduce((hand, np.logical_not(protected), near_object))
    if not np.any(current):
        return result, added, 0, False

    steps_used = 0
    for step in range(1, maximum_bridge_steps + 1):
        frontier = np.logical_and.reduce(
            (binary_dilate_square(np, current, 1), corridor, np.logical_not(current))
        )
        if not np.any(frontier):
            break
        padded_rgb = np.pad(result.astype(np.float32), ((1, 1), (1, 1), (0, 0)))
        padded_current = np.pad(current, 1)
        colour_sum = np.zeros_like(result, dtype=np.float32)
        count = np.zeros(result.shape[:2], dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                neighbour = padded_current[dy : dy + result.shape[0], dx : dx + result.shape[1]]
                colour_sum += (
                    padded_rgb[dy : dy + result.shape[0], dx : dx + result.shape[1]]
                    * neighbour[..., None]
                )
                count += neighbour
        writable = np.logical_and(frontier, count > 0)
        if not np.any(writable):
            break
        colours = np.rint(
            colour_sum[writable] / count[writable, None]
        ).clip(0, 255).astype(np.uint8)
        result[writable] = colours
        # Averaging may accidentally approach the source colour.  Reuse the
        # median nearby generated-hand colour in only those invalid pixels.
        changed = replacement_mask(
            np, result, source, threshold=replacement_threshold
        )
        invalid = np.logical_and(writable, np.logical_not(changed))
        if np.any(invalid):
            palette_pixels = result[current]
            if len(palette_pixels):
                fallback = np.median(palette_pixels, axis=0).clip(0, 255).astype(np.uint8)
                result[invalid] = fallback
        changed = np.logical_and(replacement(result), near_object)
        added |= np.logical_and(writable, changed)
        current |= changed
        steps_used = step
        observed = bool(
            np.any(
                np.logical_and(
                    binary_dilate_square(np, current, contact_radius), protected
                )
            )
        )
        if observed:
            break
    return result, added, steps_used, observed


def robust_limit(
    np: Any,
    values: Any,
    *,
    direction: str,
    mad_scale: float = 6.0,
    minimum_margin: float = 1e-6,
) -> float:
    """Fit an auditable one-sided limit from accepted anchor observations."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("robust limit expects finite one-dimensional observations")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    margin = max(float(minimum_margin), float(mad_scale) * mad)
    if direction == "upper":
        return median + margin
    if direction == "lower":
        return median - margin
    raise ValueError("robust limit direction must be 'upper' or 'lower'")
