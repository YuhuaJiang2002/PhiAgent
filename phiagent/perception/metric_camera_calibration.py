"""Robust sparse-observation calibration for foundation-model camera geometry.

The numerical routines accept a NumPy-compatible module so importing PhiAgent
does not require NumPy, PyTorch, CUDA, a sensor SDK, or a checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from phiagent.perception.foundation_contact import EvidenceClass


@dataclass(frozen=True)
class MetricDepthCalibrationContract:
    """Frozen acceptance contract for one camera/environment calibration."""

    camera_frame: str
    world_frame: str
    timeline: str
    source_video_sha256: str
    minimum_anchors: int = 20
    minimum_independent_groups: int = 2
    maximum_anchor_relative_error_p95: float = 0.04
    maximum_group_holdout_relative_error_p95: float = 0.06
    maximum_scale_standard_deviation_fraction: float = 0.02
    minimum_robust_inlier_fraction: float = 0.80
    maximum_unscaled_camera_motion_m: float = 0.01
    maximum_exact_asset_reprojection_rmse_px: float = 8.0
    bootstrap_samples: int = 512
    allowed_exact_asset_sha256: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.camera_frame.strip() or not self.world_frame.strip():
            raise ValueError("camera and world frames must be named")
        if not self.timeline.strip():
            raise ValueError("calibration timeline must be named")
        if len(self.source_video_sha256) != 64:
            raise ValueError("source video requires a SHA-256 digest")
        if self.minimum_anchors < 4:
            raise ValueError("minimum_anchors must be at least four")
        if self.minimum_independent_groups < 2:
            raise ValueError("calibration requires at least two independent groups")
        fractions = (
            self.maximum_anchor_relative_error_p95,
            self.maximum_group_holdout_relative_error_p95,
            self.maximum_scale_standard_deviation_fraction,
            self.minimum_robust_inlier_fraction,
        )
        if any(not math.isfinite(value) or value <= 0 for value in fractions):
            raise ValueError("calibration thresholds must be finite and positive")
        if self.minimum_robust_inlier_fraction > 1:
            raise ValueError("minimum inlier fraction cannot exceed one")
        if self.maximum_unscaled_camera_motion_m <= 0:
            raise ValueError("camera-motion threshold must be positive")
        if self.bootstrap_samples < 32:
            raise ValueError("at least 32 bootstrap samples are required")
        if any(len(value) != 64 for value in self.allowed_exact_asset_sha256):
            raise ValueError("allowed exact assets must use SHA-256 digests")


def _weighted_affine_fit(np: Any, x: Any, y: Any, weights: Any) -> tuple[Any, Any]:
    design = np.stack((x, np.ones_like(x)), axis=1)
    weighted_design = design * weights[:, None]
    normal = design.T @ weighted_design
    if int(np.linalg.matrix_rank(normal)) < 2:
        raise ValueError("metric anchors do not span enough depth to identify scale and shift")
    parameters = np.linalg.solve(normal, design.T @ (weights * y))
    covariance = np.linalg.inv(normal)
    return parameters, covariance


def _robust_inverse_depth_fit(
    np: Any,
    predicted_depth_m: Any,
    metric_depth_m: Any,
    metric_depth_std_m: Any,
    *,
    maximum_iterations: int = 20,
) -> dict[str, Any]:
    """Fit metric inverse depth = a / predicted depth + b using Huber IRLS."""

    predicted = np.asarray(predicted_depth_m, dtype=np.float64)
    metric = np.asarray(metric_depth_m, dtype=np.float64)
    sigma_m = np.asarray(metric_depth_std_m, dtype=np.float64)
    if predicted.ndim != 1 or metric.shape != predicted.shape or sigma_m.shape != predicted.shape:
        raise ValueError("predicted depth, metric depth, and uncertainty must align")
    if len(predicted) < 4:
        raise ValueError("at least four metric anchors are required")
    if not bool(
        np.all(np.isfinite(predicted))
        and np.all(np.isfinite(metric))
        and np.all(np.isfinite(sigma_m))
        and np.all(predicted > 0)
        and np.all(metric > 0)
        and np.all(sigma_m > 0)
    ):
        raise ValueError("metric anchors and uncertainties must be finite and positive")

    x = 1.0 / predicted
    y = 1.0 / metric
    sigma_y = sigma_m / np.square(metric)
    base_weights = 1.0 / np.maximum(np.square(sigma_y), 1e-16)
    robust_weights = np.ones_like(base_weights)
    parameters = np.asarray((1.0, 0.0), dtype=np.float64)
    covariance = np.eye(2, dtype=np.float64)
    for _ in range(maximum_iterations):
        previous = parameters.copy()
        parameters, covariance = _weighted_affine_fit(
            np, x, y, base_weights * robust_weights
        )
        residual_sigma = (y - (parameters[0] * x + parameters[1])) / sigma_y
        absolute = np.abs(residual_sigma)
        huber_delta = 1.345
        robust_weights = np.where(
            absolute <= huber_delta,
            1.0,
            huber_delta / np.maximum(absolute, 1e-12),
        )
        if float(np.max(np.abs(parameters - previous))) <= 1e-12:
            break

    fitted_inverse = parameters[0] * x + parameters[1]
    fitted_depth = np.where(fitted_inverse > 0, 1.0 / fitted_inverse, np.nan)
    relative_error = np.abs(fitted_depth - metric) / metric
    residual_sigma = (y - fitted_inverse) / sigma_y
    robust_inliers = np.abs(residual_sigma) <= 2.5
    degrees_of_freedom = max(len(x) - 2, 1)
    reduced_chi_square = float(
        np.sum((residual_sigma * np.sqrt(robust_weights)) ** 2) / degrees_of_freedom
    )
    covariance = covariance * max(reduced_chi_square, 1e-12)
    return {
        "parameters": parameters,
        "covariance": covariance,
        "fitted_depth_m": fitted_depth,
        "relative_error": relative_error,
        "robust_inliers": robust_inliers,
        "reduced_chi_square": reduced_chi_square,
    }


def _anchor_evidence_allowed(
    evidence: str,
    *,
    complete_q: bool,
    asset_sha256: str,
    reprojection_rmse_px: float | None,
    maximum_reprojection_rmse_px: float,
    allowed_exact_asset_sha256: tuple[str, ...],
) -> bool:
    evidence_class = EvidenceClass(evidence)
    if evidence_class in {
        EvidenceClass.SENSOR_MEASUREMENT,
        EvidenceClass.CALIBRATED_GEOMETRY,
    }:
        return True
    if evidence_class is EvidenceClass.EXACT_ASSET:
        return bool(
            complete_q
            and asset_sha256 in allowed_exact_asset_sha256
            and reprojection_rmse_px is not None
            and math.isfinite(reprojection_rmse_px)
            and reprojection_rmse_px <= maximum_reprojection_rmse_px
        )
    return False


def _group_holdout_errors(
    np: Any,
    *,
    predicted_depth_m: Any,
    metric_depth_m: Any,
    metric_depth_std_m: Any,
    group_ids: Any,
) -> tuple[list[dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    all_errors: list[float] = []
    for group in sorted(str(value) for value in np.unique(group_ids)):
        holdout = group_ids == group
        train = ~holdout
        try:
            fit = _robust_inverse_depth_fit(
                np,
                predicted_depth_m[train],
                metric_depth_m[train],
                metric_depth_std_m[train],
            )
            parameters = fit["parameters"]
            inverse = parameters[0] / predicted_depth_m[holdout] + parameters[1]
            valid = inverse > 0
            estimates = np.where(valid, 1.0 / inverse, np.nan)
            errors = np.abs(estimates - metric_depth_m[holdout]) / metric_depth_m[holdout]
            finite_errors = errors[np.isfinite(errors)]
            p95 = float(np.percentile(finite_errors, 95)) if len(finite_errors) else float("inf")
            all_errors.extend(float(value) for value in finite_errors)
            rows.append(
                {
                    "group_id": group,
                    "anchors": int(np.count_nonzero(holdout)),
                    "relative_error_p95": p95,
                    "fit_identifiable": True,
                }
            )
        except ValueError:
            rows.append(
                {
                    "group_id": group,
                    "anchors": int(np.count_nonzero(holdout)),
                    "relative_error_p95": float("inf"),
                    "fit_identifiable": False,
                }
            )
    maximum = max((float(row["relative_error_p95"]) for row in rows), default=float("inf"))
    return rows, maximum


def _bootstrap_scale_uncertainty(
    np: Any,
    *,
    predicted_depth_m: Any,
    metric_depth_m: Any,
    metric_depth_std_m: Any,
    group_ids: Any,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    groups = [str(value) for value in np.unique(group_ids)]
    group_indices = {group: np.flatnonzero(group_ids == group) for group in groups}
    scales: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_parts = []
        for group in groups:
            indices = group_indices[group]
            sampled_parts.append(rng.choice(indices, size=len(indices), replace=True))
        sampled = np.concatenate(sampled_parts)
        try:
            fit = _robust_inverse_depth_fit(
                np,
                predicted_depth_m[sampled],
                metric_depth_m[sampled],
                metric_depth_std_m[sampled],
            )
        except ValueError:
            continue
        parameters = fit["parameters"]
        inverse = parameters[0] / predicted_depth_m + parameters[1]
        valid = inverse > 0
        if not bool(np.all(valid)):
            continue
        calibrated = 1.0 / inverse
        scales.append(float(np.median(calibrated / predicted_depth_m)))
    if len(scales) < max(32, bootstrap_samples // 4):
        return {
            "successful_samples": float(len(scales)),
            "median_scale": float("nan"),
            "scale_standard_deviation_fraction": float("inf"),
        }
    values = np.asarray(scales, dtype=np.float64)
    median = float(np.median(values))
    return {
        "successful_samples": float(len(values)),
        "median_scale": median,
        "scale_standard_deviation_fraction": float(np.std(values, ddof=1) / median),
    }


def calibrate_metric_camera_sequence(
    np: Any,
    *,
    contract: MetricDepthCalibrationContract,
    frame_indices: Any,
    intrinsics_px: Any,
    world_from_camera: Any,
    predicted_depth_m: Any,
    depth_confidence: Any,
    anchor_frame_indices: Any,
    anchor_xy_px: Any,
    anchor_metric_depth_m: Any,
    anchor_metric_depth_std_m: Any,
    anchor_group_ids: Sequence[str],
    anchor_evidence_classes: Sequence[str],
    anchor_complete_q: Any | None = None,
    anchor_asset_sha256: Sequence[str] | None = None,
    anchor_reprojection_rmse_px: Any | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Calibrate learned depth from sparse independent metric observations.

    The returned arrays are suitable for persistence only when ``passed`` is
    true.  Learned/model-only anchors are retained in diagnostics but cannot
    satisfy the physical evidence gate.
    """

    contract.validate()
    frames = np.asarray(frame_indices, dtype=np.int64)
    intrinsics = np.asarray(intrinsics_px, dtype=np.float64)
    poses = np.asarray(world_from_camera, dtype=np.float64)
    predicted = np.asarray(predicted_depth_m, dtype=np.float64)
    confidence = np.asarray(depth_confidence, dtype=np.float64)
    anchor_frames = np.asarray(anchor_frame_indices, dtype=np.int64)
    anchor_xy = np.asarray(anchor_xy_px, dtype=np.float64)
    metric = np.asarray(anchor_metric_depth_m, dtype=np.float64)
    metric_std = np.asarray(anchor_metric_depth_std_m, dtype=np.float64)
    group_ids = np.asarray(tuple(str(value) for value in anchor_group_ids))
    evidence = tuple(str(value) for value in anchor_evidence_classes)
    count = len(anchor_frames)

    if predicted.ndim != 3:
        raise ValueError("predicted depth must have shape TxHxW")
    expected_depth_shape = (len(frames), predicted.shape[1], predicted.shape[2])
    if predicted.shape != expected_depth_shape:
        raise ValueError("predicted depth must have shape TxHxW")
    if confidence.shape != predicted.shape:
        raise ValueError("depth confidence must align with predicted depth")
    if intrinsics.shape != (len(frames), 3, 3) or poses.shape != (len(frames), 4, 4):
        raise ValueError("camera intrinsics and poses must align with sampled frames")
    if anchor_xy.shape != (count, 2):
        raise ValueError("anchor pixels must have shape Nx2 in x,y order")
    if metric.shape != (count,) or metric_std.shape != (count,):
        raise ValueError("anchor metric depths and standard deviations must have shape N")
    if len(group_ids) != count or len(evidence) != count:
        raise ValueError("anchor groups and evidence classes must align with anchors")

    complete_q = (
        np.zeros(count, dtype=bool)
        if anchor_complete_q is None
        else np.asarray(anchor_complete_q, dtype=bool)
    )
    reprojection = (
        np.full(count, np.nan, dtype=np.float64)
        if anchor_reprojection_rmse_px is None
        else np.asarray(anchor_reprojection_rmse_px, dtype=np.float64)
    )
    asset_sha256 = (
        tuple("" for _ in range(count))
        if anchor_asset_sha256 is None
        else tuple(str(value) for value in anchor_asset_sha256)
    )
    if (
        complete_q.shape != (count,)
        or reprojection.shape != (count,)
        or len(asset_sha256) != count
    ):
        raise ValueError("exact-asset support metadata must align with anchors")

    frame_lookup = {int(frame): index for index, frame in enumerate(frames)}
    frame_valid = np.asarray([int(frame) in frame_lookup for frame in anchor_frames])
    height, width = predicted.shape[1:]
    pixel_valid = (
        np.isfinite(anchor_xy).all(axis=1)
        & (anchor_xy[:, 0] >= 0)
        & (anchor_xy[:, 0] < width)
        & (anchor_xy[:, 1] >= 0)
        & (anchor_xy[:, 1] < height)
    )
    observation_valid = frame_valid & pixel_valid
    sampled_predicted = np.full(count, np.nan, dtype=np.float64)
    for index in np.flatnonzero(observation_valid):
        frame_slot = frame_lookup[int(anchor_frames[index])]
        x = int(round(float(anchor_xy[index, 0])))
        y = int(round(float(anchor_xy[index, 1])))
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        x0, x1 = max(0, x - 1), min(width, x + 2)
        y0, y1 = max(0, y - 1), min(height, y + 2)
        patch = predicted[frame_slot, y0:y1, x0:x1]
        valid_patch = patch[np.isfinite(patch) & (patch > 0)]
        if len(valid_patch):
            sampled_predicted[index] = float(np.median(valid_patch))
    observation_valid &= np.isfinite(sampled_predicted)

    evidence_allowed = np.asarray(
        [
            _anchor_evidence_allowed(
                row,
                complete_q=bool(complete_q[index]),
                asset_sha256=asset_sha256[index],
                reprojection_rmse_px=(
                    float(reprojection[index]) if np.isfinite(reprojection[index]) else None
                ),
                maximum_reprojection_rmse_px=(
                    contract.maximum_exact_asset_reprojection_rmse_px
                ),
                allowed_exact_asset_sha256=contract.allowed_exact_asset_sha256,
            )
            for index, row in enumerate(evidence)
        ],
        dtype=bool,
    )
    valid = observation_valid & evidence_allowed
    independent_groups = sorted(str(value) for value in np.unique(group_ids[valid]))
    gates: dict[str, bool] = {
        "source_video_hash_bound": len(contract.source_video_sha256) == 64,
        "anchor_pixels_and_frames_valid": bool(np.all(observation_valid)) if count else False,
        "anchor_evidence_independent_of_foundation_model": (
            bool(np.all(evidence_allowed)) if count else False
        ),
        "minimum_anchor_count": int(np.count_nonzero(valid)) >= contract.minimum_anchors,
        "minimum_independent_groups": (
            len(independent_groups) >= contract.minimum_independent_groups
        ),
    }
    report: dict[str, object] = {
        "anchors_total": count,
        "anchors_admissible": int(np.count_nonzero(valid)),
        "independent_group_ids": independent_groups,
        "gates": gates,
        "passed": False,
    }
    if not all(gates.values()):
        report["reasons"] = [name for name, passed in gates.items() if not passed]
        return report

    selected_predicted = sampled_predicted[valid]
    selected_metric = metric[valid]
    selected_std = metric_std[valid]
    selected_groups = group_ids[valid]
    fit = _robust_inverse_depth_fit(
        np, selected_predicted, selected_metric, selected_std
    )
    parameters = fit["parameters"]
    inverse_all = parameters[0] / predicted + parameters[1]
    positive_mapping = bool(np.all(np.isfinite(inverse_all)) and np.all(inverse_all > 0))
    calibrated_depth = np.where(positive_mapping, 1.0 / inverse_all, np.nan)
    anchor_error_p95 = float(np.percentile(fit["relative_error"], 95))
    inlier_fraction = float(np.mean(fit["robust_inliers"]))
    group_rows, holdout_p95 = _group_holdout_errors(
        np,
        predicted_depth_m=selected_predicted,
        metric_depth_m=selected_metric,
        metric_depth_std_m=selected_std,
        group_ids=selected_groups,
    )
    bootstrap = _bootstrap_scale_uncertainty(
        np,
        predicted_depth_m=selected_predicted,
        metric_depth_m=selected_metric,
        metric_depth_std_m=selected_std,
        group_ids=selected_groups,
        bootstrap_samples=contract.bootstrap_samples,
        seed=seed,
    )
    translations = poses[:, :3, 3]
    camera_motion = float(np.max(np.linalg.norm(translations - translations[0], axis=1)))
    fixed_camera = camera_motion <= contract.maximum_unscaled_camera_motion_m
    gates.update(
        {
            "inverse_depth_affine_mapping_positive": positive_mapping,
            "anchor_relative_error_bounded": (
                anchor_error_p95 <= contract.maximum_anchor_relative_error_p95
            ),
            "group_holdout_error_bounded": (
                holdout_p95 <= contract.maximum_group_holdout_relative_error_p95
            ),
            "robust_inlier_fraction_bounded": (
                inlier_fraction >= contract.minimum_robust_inlier_fraction
            ),
            "scale_uncertainty_bounded": (
                bootstrap["scale_standard_deviation_fraction"]
                <= contract.maximum_scale_standard_deviation_fraction
            ),
            "camera_motion_static_or_independently_scaled": fixed_camera,
        }
    )
    effective_scale = float(bootstrap["median_scale"])
    calibrated_poses = poses.copy()
    calibrated_poses[:, :3, 3] = (
        translations[0] + effective_scale * (translations - translations[0])
    )
    report.update(
        {
            "model": "robust_affine_inverse_depth",
            "parameters": {
                "inverse_depth_scale": float(parameters[0]),
                "inverse_depth_shift_per_m": float(parameters[1]),
                "covariance": [
                    [float(value) for value in row] for row in fit["covariance"]
                ],
            },
            "anchor_relative_error_p95": anchor_error_p95,
            "group_holdout": group_rows,
            "group_holdout_relative_error_p95_max": holdout_p95,
            "robust_inlier_fraction": inlier_fraction,
            "reduced_chi_square": float(fit["reduced_chi_square"]),
            "effective_depth_scale": effective_scale,
            "scale_standard_deviation_fraction": float(
                bootstrap["scale_standard_deviation_fraction"]
            ),
            "bootstrap_successful_samples": int(bootstrap["successful_samples"]),
            "unscaled_camera_motion_m_max": camera_motion,
            "gates": gates,
            "passed": all(gates.values()),
        }
    )
    if not report["passed"]:
        report["reasons"] = [name for name, passed in gates.items() if not passed]
        return report
    report["calibrated_depth_m"] = calibrated_depth.astype(np.float32)
    report["calibrated_world_from_camera"] = calibrated_poses.astype(np.float32)
    report["intrinsics_px"] = intrinsics.astype(np.float32)
    report["confidence"] = confidence.astype(np.float32)
    return report
