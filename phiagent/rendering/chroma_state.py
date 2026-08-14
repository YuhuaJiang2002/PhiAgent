"""Masked chroma-state projection for long-video robot layers.

Geometry and contact boundaries are primarily carried by luma.  This module
regularizes only the two chroma channels inside an explicit editable mask,
leaving every pixel outside that mask byte-identical to the input frame.
"""

from __future__ import annotations

from typing import Any


def restore_masked_luma_carrier(
    cv2: Any,
    np: Any,
    frame_bgr: Any,
    target_bgr: Any,
    mask: Any,
    *,
    maximum_iterations: int = 3,
) -> tuple[Any, dict[str, float]]:
    """Project integer BGR pixels back onto the target OpenCV luma plane."""

    result = np.asarray(frame_bgr, dtype=np.uint8).copy()
    target = np.asarray(target_bgr, dtype=np.uint8)
    support = np.asarray(mask, dtype=np.bool_)
    if result.shape != target.shape or result.ndim != 3 or result.shape[2] != 3:
        raise ValueError("frame and luma target must be matching HxWx3 images")
    if support.shape != result.shape[:2]:
        raise ValueError("luma mask must match the image plane")
    if maximum_iterations < 1:
        raise ValueError("maximum iterations must be positive")
    target_luma = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.int16)
    for _ in range(maximum_iterations):
        current_luma = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)[..., 0].astype(
            np.int16
        )
        error = target_luma - current_luma
        active = np.logical_and(support, error != 0)
        if not np.any(active):
            break
        corrected = result[active].astype(np.int16) + error[active][:, None]
        result[active] = np.clip(corrected, 0, 255).astype(np.uint8)
    final_luma = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.int16)
    selected = np.abs(final_luma[support] - target_luma[support])
    return result, {
        "mean_abs_luma_delta": float(selected.mean()) if selected.size else 0.0,
        "maximum_abs_luma_delta": float(selected.max()) if selected.size else 0.0,
        "nonzero_luma_pixels": float(np.count_nonzero(selected)),
    }


def spatial_chroma_tv(np: Any, saturation: Any, mask: Any) -> float:
    """Measure mean spatial total variation on an OpenCV saturation plane."""

    values = np.asarray(saturation, dtype=np.float32)
    support = np.asarray(mask, dtype=np.bool_)
    if values.ndim != 2 or support.shape != values.shape:
        raise ValueError("saturation and mask must share one 2-D image plane")
    horizontal = np.logical_and(support[:, 1:], support[:, :-1])
    vertical = np.logical_and(support[1:, :], support[:-1, :])
    differences = []
    if np.any(horizontal):
        differences.append(np.abs(values[:, 1:] - values[:, :-1])[horizontal])
    if np.any(vertical):
        differences.append(np.abs(values[1:, :] - values[:-1, :])[vertical])
    return float(np.concatenate(differences).mean()) if differences else 0.0


def normalized_mask_blur(
    cv2: Any,
    np: Any,
    values: Any,
    mask: Any,
    *,
    kernel_size: int,
) -> Any:
    """Blur ``values`` without allowing colours outside ``mask`` to bleed in."""

    array = np.asarray(values, dtype=np.float32)
    support = np.asarray(mask, dtype=np.bool_)
    if array.ndim != 2 or support.shape != array.shape:
        raise ValueError("values and mask must share one 2-D image plane")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel size must be an odd integer of at least three")
    weights = cv2.GaussianBlur(
        support.astype(np.float32),
        (kernel_size, kernel_size),
        0,
        borderType=cv2.BORDER_REFLECT101,
    )
    numerator = cv2.GaussianBlur(
        array * support.astype(np.float32),
        (kernel_size, kernel_size),
        0,
        borderType=cv2.BORDER_REFLECT101,
    )
    result = array.copy()
    valid = np.logical_and(support, weights > 1e-6)
    result[valid] = numerator[valid] / weights[valid]
    return result


def project_masked_chroma_state(
    cv2: Any,
    np: Any,
    frame_bgr: Any,
    editable_mask: Any,
    *,
    kernel_size: int,
    strength: float,
    maximum_chroma_delta: float,
) -> tuple[Any, dict[str, float]]:
    """Project chroma to a spatially coherent state while preserving luma.

    The projection smooths HSV saturation directly, retains hue/value, then
    restores the original Y channel with a neutral RGB offset.  This is a
    state-representation change rather than a gate or threshold relaxation.
    """

    result, metrics = project_masked_multiscale_chroma_state(
        cv2,
        np,
        frame_bgr,
        editable_mask,
        kernel_sizes=(kernel_size,),
        strength=strength,
        maximum_chroma_delta=maximum_chroma_delta,
        saturation_scale=1.0,
    )
    first_pass = metrics["passes"][0]
    return result, {
        "editable_pixels": metrics["editable_pixels"],
        "mean_abs_chroma_delta": first_pass["mean_abs_chroma_delta"],
        "maximum_abs_chroma_delta": first_pass["maximum_abs_chroma_delta"],
        "mean_abs_luma_delta": metrics["mean_abs_luma_delta"],
    }


def project_masked_multiscale_chroma_state(
    cv2: Any,
    np: Any,
    frame_bgr: Any,
    editable_mask: Any,
    *,
    kernel_sizes: tuple[int, ...],
    strength: float,
    maximum_chroma_delta: float,
    saturation_scale: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Apply a coarse-to-fine projection with one colour-space round trip.

    Reusing one HSV state is both more faithful and substantially cheaper than
    repeatedly quantizing BGR to HSV and back for every spatial scale.
    """

    frame = np.asarray(frame_bgr, dtype=np.uint8)
    mask = np.asarray(editable_mask, dtype=np.bool_)
    if frame.ndim != 3 or frame.shape[2] != 3 or mask.shape != frame.shape[:2]:
        raise ValueError("frame must be HxWx3 and mask must be HxW")
    if not 0.0 < strength <= 1.0:
        raise ValueError("strength must be in (0, 1]")
    if maximum_chroma_delta <= 0:
        raise ValueError("maximum chroma delta must be positive")
    if not 0.0 <= saturation_scale <= 1.0:
        raise ValueError("saturation scale must be in [0, 1]")
    kernels = tuple(int(value) for value in kernel_sizes)
    if not kernels:
        raise ValueError("at least one kernel size is required")
    if any(value < 3 or value % 2 == 0 for value in kernels):
        raise ValueError("kernel sizes must be odd integers of at least three")
    if not np.any(mask):
        return frame.copy(), {
            "editable_pixels": 0.0,
            "mean_abs_luma_delta": 0.0,
            "chroma_tv_before": 0.0,
            "chroma_tv_after": 0.0,
            "passes": [
                {
                    "kernel_size": value,
                    "mean_abs_chroma_delta": 0.0,
                    "maximum_abs_chroma_delta": 0.0,
                }
                for value in kernels
            ],
        }

    original_y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    original_saturation = hsv[..., 1].copy()
    projected_saturation = original_saturation.copy()
    pass_metrics = []
    for kernel_size in kernels:
        target = normalized_mask_blur(
            cv2,
            np,
            projected_saturation,
            mask,
            kernel_size=kernel_size,
        )
        saturation_delta = np.clip(
            (target - projected_saturation) * strength,
            -maximum_chroma_delta,
            maximum_chroma_delta,
        )
        projected_saturation[mask] = np.clip(
            projected_saturation[mask] + saturation_delta[mask], 0.0, 255.0
        )
        selected_delta = np.abs(saturation_delta[mask])
        pass_metrics.append(
            {
                "kernel_size": kernel_size,
                "mean_abs_chroma_delta": float(selected_delta.mean()),
                "maximum_abs_chroma_delta": float(selected_delta.max()),
            }
        )
    # The edited body is a silver/black robot.  A bounded projection toward
    # the achromatic canonical material prior prevents 4:2:0 subsampling from
    # turning luma edges into false coloured patches.  Flowers are excluded by
    # the caller's object mask and therefore retain their chroma.
    achromatic_delta = np.clip(
        projected_saturation * (saturation_scale - 1.0),
        -maximum_chroma_delta,
        maximum_chroma_delta,
    )
    projected_saturation[mask] = np.clip(
        projected_saturation[mask] + achromatic_delta[mask], 0.0, 255.0
    )
    selected_achromatic_delta = np.abs(achromatic_delta[mask])
    pass_metrics.append(
        {
            "kernel_size": 0,
            "projection": "canonical_achromatic_material_prior",
            "saturation_scale": saturation_scale,
            "mean_abs_chroma_delta": float(selected_achromatic_delta.mean()),
            "maximum_abs_chroma_delta": float(selected_achromatic_delta.max()),
        }
    )

    projected = hsv.copy()
    projected[..., 1][mask] = projected_saturation[mask]
    result_float = cv2.cvtColor(
        np.clip(np.rint(projected), 0, 255).astype(np.uint8),
        cv2.COLOR_HSV2BGR,
    ).astype(np.float32)
    projected_y = cv2.cvtColor(
        result_float.astype(np.uint8), cv2.COLOR_BGR2YCrCb
    )[..., 0].astype(np.float32)
    # A neutral RGB offset restores the original Y carrier without changing
    # hue direction.  Clipping is bounded and audited below.
    result_float += (original_y - projected_y)[..., None]
    result = np.clip(np.rint(result_float), 0, 255).astype(np.uint8)
    result[np.logical_not(mask)] = frame[np.logical_not(mask)]
    result_luma = cv2.cvtColor(
        result, cv2.COLOR_BGR2YCrCb
    )[..., 0].astype(np.float32)
    result_saturation = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)[..., 1].astype(
        np.float32
    )
    return result, {
        "editable_pixels": float(np.count_nonzero(mask)),
        "mean_abs_luma_delta": float(
            np.abs(result_luma[mask] - original_y[mask]).mean()
        ),
        "chroma_tv_before": spatial_chroma_tv(np, original_saturation, mask),
        "chroma_tv_after": spatial_chroma_tv(np, result_saturation, mask),
        "passes": pass_metrics,
    }
