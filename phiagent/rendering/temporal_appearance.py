"""Confidence-gated temporal state for a bounded generated appearance residual."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class FlowState:
    map_x: Any
    map_y: Any
    confidence: Any
    cycle_error: Any
    photometric_error: Any


@dataclass(frozen=True)
class ResidualConsensus:
    value: Any
    reliable: Any
    support_count: Any
    weight_sum: Any
    maximum_channel_mad: Any


def bidirectional_flow_state(
    cv2: Any,
    np: Any,
    previous_reference: Any,
    current_reference: Any,
    *,
    scale: float = 0.5,
    cycle_sigma: float = 0.75,
    maximum_cycle_error: float = 1.5,
    photometric_sigma: float = 8.0,
    maximum_photometric_error: float = 24.0,
) -> FlowState:
    """Estimate current-to-previous maps and reject inconsistent correspondences."""

    if previous_reference.shape != current_reference.shape:
        raise ValueError("flow reference frames must share one image shape")
    if previous_reference.ndim != 3 or previous_reference.shape[2] != 3:
        raise ValueError("flow reference frames must be HxWx3")
    for name, value in (
        ("scale", scale),
        ("cycle_sigma", cycle_sigma),
        ("maximum_cycle_error", maximum_cycle_error),
        ("photometric_sigma", photometric_sigma),
        ("maximum_photometric_error", maximum_photometric_error),
    ):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    height, width = previous_reference.shape[:2]
    flow_width = max(16, int(round(width * scale)))
    flow_height = max(16, int(round(height * scale)))
    previous_gray = cv2.resize(
        cv2.cvtColor(previous_reference, cv2.COLOR_BGR2GRAY),
        (flow_width, flow_height),
        interpolation=cv2.INTER_AREA,
    )
    current_gray = cv2.resize(
        cv2.cvtColor(current_reference, cv2.COLOR_BGR2GRAY),
        (flow_width, flow_height),
        interpolation=cv2.INTER_AREA,
    )
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    estimator.setUseSpatialPropagation(True)
    forward = estimator.calc(previous_gray, current_gray, None)
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    estimator.setUseSpatialPropagation(True)
    backward = estimator.calc(current_gray, previous_gray, None)

    low_y, low_x = np.mgrid[0:flow_height, 0:flow_width].astype(np.float32)
    previous_x = low_x + backward[..., 0]
    previous_y = low_y + backward[..., 1]
    sampled_forward = cv2.remap(
        forward,
        previous_x,
        previous_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1e6,
    )
    cycle_low = np.linalg.norm(backward + sampled_forward, axis=2)
    in_bounds_low = (
        (previous_x >= 0)
        & (previous_x <= flow_width - 1)
        & (previous_y >= 0)
        & (previous_y <= flow_height - 1)
    )

    backward_full = cv2.resize(
        backward,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    backward_full[..., 0] *= width / flow_width
    backward_full[..., 1] *= height / flow_height
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x = grid_x + backward_full[..., 0]
    map_y = grid_y + backward_full[..., 1]
    warped_previous_gray = cv2.remap(
        cv2.cvtColor(previous_reference, cv2.COLOR_BGR2GRAY),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    current_gray_full = cv2.cvtColor(current_reference, cv2.COLOR_BGR2GRAY)
    photometric_error = np.abs(
        current_gray_full.astype(np.float32)
        - warped_previous_gray.astype(np.float32)
    )
    cycle_error = cv2.resize(
        cycle_low,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    in_bounds = cv2.resize(
        in_bounds_low.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    confidence = np.exp(-0.5 * (cycle_error / cycle_sigma) ** 2)
    confidence *= np.exp(-0.5 * (photometric_error / photometric_sigma) ** 2)
    confidence *= (
        in_bounds
        & (cycle_error <= maximum_cycle_error)
        & (photometric_error <= maximum_photometric_error)
    )
    return FlowState(
        map_x=map_x,
        map_y=map_y,
        confidence=confidence.astype(np.float32),
        cycle_error=cycle_error.astype(np.float32),
        photometric_error=photometric_error.astype(np.float32),
    )


def warp_with_flow(cv2: Any, value: Any, flow: FlowState, *, nearest: bool = False) -> Any:
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(
        value,
        flow.map_x,
        flow.map_y,
        interpolation,
        borderMode=cv2.BORDER_REFLECT101,
    )


def weighted_residual_consensus(
    np: Any,
    *,
    residuals: Any,
    weights: Any,
    minimum_observations: int,
    maximum_channel_mad: float,
) -> ResidualConsensus:
    """Aggregate aligned observations without averaging incompatible values."""

    values = np.asarray(residuals, dtype=np.float32)
    confidence = np.asarray(weights, dtype=np.float32)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("residuals must have shape KxHxWx3")
    if confidence.shape != values.shape[:3]:
        raise ValueError("weights must have shape KxHxW")
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least two")
    if minimum_observations > values.shape[0]:
        raise ValueError("minimum_observations exceeds the observation count")
    if not isfinite(maximum_channel_mad) or maximum_channel_mad <= 0:
        raise ValueError("maximum_channel_mad must be finite and positive")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(confidence)):
        raise ValueError("residual observations and weights must be finite")
    if np.any(confidence < 0):
        raise ValueError("consensus weights must be non-negative")

    support_count = np.count_nonzero(confidence > 0, axis=0)
    weight_sum = np.sum(confidence, axis=0)
    median = np.zeros(values.shape[1:], dtype=np.float32)
    mad = np.zeros(values.shape[1:], dtype=np.float32)
    for channel in range(3):
        channel_values = values[..., channel]
        order = np.argsort(channel_values, axis=0)
        ordered_values = np.take_along_axis(channel_values, order, axis=0)
        ordered_weights = np.take_along_axis(confidence, order, axis=0)
        cumulative = np.cumsum(ordered_weights, axis=0)
        threshold = weight_sum * 0.5
        median_index = np.argmax(cumulative >= threshold[None, ...], axis=0)
        median[..., channel] = np.take_along_axis(
            ordered_values,
            median_index[None, ...],
            axis=0,
        )[0]

        deviations = np.abs(channel_values - median[..., channel][None, ...])
        deviation_order = np.argsort(deviations, axis=0)
        ordered_deviations = np.take_along_axis(
            deviations,
            deviation_order,
            axis=0,
        )
        deviation_weights = np.take_along_axis(
            confidence,
            deviation_order,
            axis=0,
        )
        deviation_cumulative = np.cumsum(deviation_weights, axis=0)
        mad_index = np.argmax(
            deviation_cumulative >= threshold[None, ...],
            axis=0,
        )
        mad[..., channel] = np.take_along_axis(
            ordered_deviations,
            mad_index[None, ...],
            axis=0,
        )[0]

    max_mad = np.max(mad, axis=2)
    reliable = np.asarray(
        (support_count >= minimum_observations)
        & (weight_sum > 0)
        & (max_mad <= maximum_channel_mad),
        dtype=np.bool_,
    )
    return ResidualConsensus(
        value=median,
        reliable=reliable,
        support_count=support_count,
        weight_sum=weight_sum,
        maximum_channel_mad=max_mad,
    )


def residual_state_update(
    cv2: Any,
    np: Any,
    *,
    current_incumbent: Any,
    current_candidate: Any,
    warped_previous_incumbent: Any,
    warped_previous_state: Any,
    confidence: Any,
    reliable: Any,
    strength: float,
    gaussian_sigma: float,
    maximum_residual_delta: float,
) -> tuple[Any, dict[str, float]]:
    """Correct only low-frequency appearance residual; keep current geometry."""

    shapes = {
        current_incumbent.shape,
        current_candidate.shape,
        warped_previous_incumbent.shape,
        warped_previous_state.shape,
    }
    if len(shapes) != 1 or current_candidate.ndim != 3:
        raise ValueError("appearance frames must share one HxWx3 shape")
    if confidence.shape != reliable.shape or confidence.shape != current_candidate.shape[:2]:
        raise ValueError("confidence and reliable mask must match the frame")
    for name, value in (
        ("strength", strength),
        ("gaussian_sigma", gaussian_sigma),
        ("maximum_residual_delta", maximum_residual_delta),
    ):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if strength > 1:
        raise ValueError("strength must not exceed one")

    current_residual = (
        current_candidate.astype(np.float32) - current_incumbent.astype(np.float32)
    )
    previous_residual = (
        warped_previous_state.astype(np.float32)
        - warped_previous_incumbent.astype(np.float32)
    )
    current_low = cv2.GaussianBlur(
        current_residual,
        (0, 0),
        gaussian_sigma,
    )
    previous_low = cv2.GaussianBlur(
        previous_residual,
        (0, 0),
        gaussian_sigma,
    )
    correction = np.clip(
        previous_low - current_low,
        -maximum_residual_delta,
        maximum_residual_delta,
    )
    weight = np.clip(confidence, 0.0, 1.0) * reliable.astype(np.float32)
    repaired = np.clip(
        np.rint(
            current_candidate.astype(np.float32)
            + correction * (strength * weight[..., None])
        ),
        0,
        255,
    ).astype(np.uint8)
    immutable = np.logical_not(np.asarray(reliable, dtype=np.bool_))
    repaired[immutable] = current_candidate[immutable]
    active = reliable & (weight > 0)
    if not np.any(active):
        return repaired, {
            "active_fraction": 0.0,
            "mean_abs_applied_correction": 0.0,
            "maximum_abs_applied_correction": 0.0,
        }
    applied = np.abs(
        repaired.astype(np.float32) - current_candidate.astype(np.float32)
    )
    return repaired, {
        "active_fraction": float(np.mean(active)),
        "mean_abs_applied_correction": float(np.mean(applied[active])),
        "maximum_abs_applied_correction": float(np.max(applied[active])),
    }
