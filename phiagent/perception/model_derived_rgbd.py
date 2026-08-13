"""Contracts for model-derived RGB-D and calibrated virtual camera views.

The functions in this module are NumPy-agnostic at import time.  A caller passes
its NumPy module explicitly.  Virtual views have exact *constructed* extrinsics,
but remain derived from the same visible source surfaces and therefore are not
independent physical camera observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelDerivedRGBDContract:
    """Lineage and acceptance bounds for a model-derived RGB-D proposal."""

    source_video_sha256: str
    timeline: str
    fps: float
    model_name: str
    model_revision: str
    checkpoint_sha256: str
    source_group_frames: tuple[str, ...]
    virtual_camera_frame: str
    maximum_frame_gap: int = 6
    minimum_valid_depth_fraction: float = 0.99
    minimum_mean_virtual_view_coverage: float = 0.80
    maximum_cycle_depth_relative_error_p95: float = 0.01
    maximum_cross_run_median_depth_fraction: float = 0.02

    def validate(self) -> None:
        named = (
            self.timeline,
            self.model_name,
            self.model_revision,
            self.virtual_camera_frame,
            *self.source_group_frames,
        )
        if any(not value.strip() for value in named):
            raise ValueError("model RGB-D coordinate frames and provenance must be named")
        if len(self.source_video_sha256) != 64 or len(self.checkpoint_sha256) != 64:
            raise ValueError("model RGB-D source and checkpoint require SHA-256 digests")
        if len(set(self.source_group_frames)) != len(self.source_group_frames):
            raise ValueError("each model RGB-D run requires a distinct world frame")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("model RGB-D FPS must be finite and positive")
        if self.maximum_frame_gap <= 0:
            raise ValueError("maximum frame gap must be positive")
        fractions = (
            self.minimum_valid_depth_fraction,
            self.minimum_mean_virtual_view_coverage,
            self.maximum_cycle_depth_relative_error_p95,
            self.maximum_cross_run_median_depth_fraction,
        )
        if any(not math.isfinite(value) or value < 0 for value in fractions):
            raise ValueError("model RGB-D thresholds must be finite and non-negative")


def depth_splat_rgbd(
    np: Any,
    *,
    source_rgb: Any,
    source_depth_m: Any,
    source_confidence: Any,
    intrinsics_px: Any,
    target_camera_from_source_camera: Any,
) -> dict[str, Any]:
    """Z-buffer one RGB-D visible surface into a named target camera frame."""

    rgb = np.asarray(source_rgb, dtype=np.uint8)
    depth = np.asarray(source_depth_m, dtype=np.float64)
    confidence = np.asarray(source_confidence, dtype=np.float64)
    intrinsics = np.asarray(intrinsics_px, dtype=np.float64)
    transform = np.asarray(target_camera_from_source_camera, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("source RGB must have shape HxWx3")
    height, width = depth.shape
    if rgb.shape[:2] != (height, width) or confidence.shape != depth.shape:
        raise ValueError("source RGB, depth, and confidence must be pixel aligned")
    if intrinsics.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("intrinsics and camera transform have invalid shapes")
    if not bool(np.all(np.isfinite(intrinsics))) or not bool(
        np.all(np.isfinite(transform))
    ):
        raise ValueError("intrinsics and camera transform must be finite")
    expected_bottom = np.asarray([0.0, 0.0, 0.0, 1.0])
    if float(np.max(np.abs(transform[3] - expected_bottom))) > 1e-8:
        raise ValueError("target camera transform must be homogeneous")

    y, x = np.mgrid[0:height, 0:width]
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    z = depth[valid]
    points = np.stack(
        (
            (x[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (y[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
            np.ones_like(z),
        ),
        axis=0,
    )
    # NumPy 2.2 on Apple Accelerate emits a false divide-by-zero warning for
    # ``matmul`` on this 4xN layout even though both operands are finite.  dot
    # uses the same explicit linear transform without that backend defect.
    target_points = np.dot(transform, points)
    target_z = target_points[2]
    in_front = target_z > 1e-6
    target_points = target_points[:, in_front]
    target_z = target_z[in_front]
    colors = rgb[valid][in_front]
    confidences = confidence[valid][in_front]
    source_flat = np.flatnonzero(valid)[in_front]
    u = np.rint(
        intrinsics[0, 0] * target_points[0] / target_z + intrinsics[0, 2]
    ).astype(np.int64)
    v = np.rint(
        intrinsics[1, 1] * target_points[1] / target_z + intrinsics[1, 2]
    ).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[inside]
    v = v[inside]
    target_z = target_z[inside]
    colors = colors[inside]
    confidences = confidences[inside]
    source_flat = source_flat[inside]
    target_flat = v * width + u
    order = np.argsort(target_z, kind="stable")
    sorted_flat = target_flat[order]
    _, first = np.unique(sorted_flat, return_index=True)
    selected = order[first]
    target_flat = target_flat[selected]

    output_rgb = np.zeros((height * width, 3), dtype=np.uint8)
    output_depth = np.full(height * width, np.nan, dtype=np.float32)
    output_confidence = np.zeros(height * width, dtype=np.float32)
    output_source_index = np.full(height * width, -1, dtype=np.int32)
    output_rgb[target_flat] = colors[selected]
    output_depth[target_flat] = target_z[selected].astype(np.float32)
    output_confidence[target_flat] = confidences[selected].astype(np.float32)
    output_source_index[target_flat] = source_flat[selected].astype(np.int32)
    output_valid = output_source_index >= 0
    return {
        "rgb": output_rgb.reshape(height, width, 3),
        "depth_m": output_depth.reshape(height, width),
        "confidence": output_confidence.reshape(height, width),
        "valid_mask": output_valid.reshape(height, width),
        "source_flat_index": output_source_index.reshape(height, width),
    }


def audit_model_derived_rgbd(
    np: Any,
    *,
    contract: ModelDerivedRGBDContract,
    source_frame_indices: Any,
    source_group_indices: Any,
    depth_m: Any,
    virtual_view_coverage: Any,
    cycle_depth_relative_error_p95: Any,
    group_median_depth_m: Any,
) -> dict[str, object]:
    """Validate proposal utility without promoting a same-video view to calibration."""

    contract.validate()
    frames = np.asarray(source_frame_indices, dtype=np.int64)
    group_indices = np.asarray(source_group_indices, dtype=np.int64)
    depth = np.asarray(depth_m, dtype=np.float64)
    coverage = np.asarray(virtual_view_coverage, dtype=np.float64)
    cycle_error = np.asarray(cycle_depth_relative_error_p95, dtype=np.float64)
    medians = np.asarray(group_median_depth_m, dtype=np.float64)
    if frames.ndim != 1 or len(frames) < 2 or bool(np.any(np.diff(frames) <= 0)):
        raise ValueError("combined model RGB-D frames must be unique and increasing")
    if group_indices.shape != frames.shape:
        raise ValueError("one source group index is required per RGB-D frame")
    if depth.ndim != 3 or depth.shape[0] != len(frames):
        raise ValueError("combined depth must have shape TxHxW")
    if coverage.ndim != 1 or cycle_error.shape != coverage.shape:
        raise ValueError("virtual-view metrics must be aligned vectors")
    if len(medians) != len(contract.source_group_frames):
        raise ValueError("one median depth is required per model run")
    if bool(np.any(group_indices < 0)) or bool(
        np.any(group_indices >= len(contract.source_group_frames))
    ):
        raise ValueError("source group index lies outside named group frames")

    valid_depth_fraction = float(np.mean(np.isfinite(depth) & (depth > 0)))
    maximum_gap = int(np.max(np.diff(frames)))
    mean_coverage = float(np.mean(coverage))
    maximum_cycle_error = float(np.max(cycle_error))
    median_reference = float(np.median(medians))
    cross_run_fraction = float(
        (np.max(medians) - np.min(medians)) / max(abs(median_reference), 1e-12)
    )
    diagnostic_gates = {
        "dense_interleaved_timeline": maximum_gap <= contract.maximum_frame_gap,
        "valid_positive_depth": (
            valid_depth_fraction >= contract.minimum_valid_depth_fraction
        ),
        "virtual_view_visible_surface_coverage": (
            mean_coverage >= contract.minimum_mean_virtual_view_coverage
        ),
        "virtual_view_cycle_depth_bounded": (
            maximum_cycle_error
            <= contract.maximum_cycle_depth_relative_error_p95
        ),
        "same_model_cross_run_scale_stable": (
            cross_run_fraction
            <= contract.maximum_cross_run_median_depth_fraction
        ),
        "group_coordinate_frames_are_explicit": (
            len(contract.source_group_frames) == len(set(group_indices.tolist()))
        ),
    }
    physical_gates = {
        "independent_absolute_scale_observation": False,
        "physically_measured_extra_view": False,
        "newly_observed_occluded_surfaces": False,
    }
    return {
        "proposal_passed": all(diagnostic_gates.values()),
        "physical_calibration_passed": False,
        "evidence_class": "foundation_model_estimate",
        "independent_physical_groups": 0,
        "diagnostic_gates": diagnostic_gates,
        "physical_gates": physical_gates,
        "metrics": {
            "samples": int(len(frames)),
            "maximum_frame_gap": maximum_gap,
            "effective_proposal_hz": float(contract.fps / maximum_gap),
            "valid_depth_fraction": valid_depth_fraction,
            "mean_virtual_view_coverage": mean_coverage,
            "maximum_cycle_depth_relative_error_p95": maximum_cycle_error,
            "cross_run_median_depth_fraction": cross_run_fraction,
            "group_median_depth_m": medians.tolist(),
        },
        "reason": "model_derived_same_video_not_independent_calibration",
    }
