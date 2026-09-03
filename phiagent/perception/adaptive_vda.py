"""Adaptive VDA depth fusion for Stage-3-BIR hand trajectories.

This module contains only NumPy geometry.  VDA inference, H3MR, MANO, and the
penetration detector remain external adapters so importing :mod:`phiagent`
does not require CUDA or model checkpoints.

Coordinate convention:

* ``delta_z_camera_m`` is a translation along the camera-frame +Z axis.
* ``camera_R_c2w`` rotates camera-frame vectors into world-frame vectors.
* ``world_shift_m`` is applied to one complete hand; local MANO geometry and
  camera rotation are never changed here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


HANDS = ("left", "right")
VERSION = "adaptive-vda-v2-hand-root-depth-v1"


@dataclass(frozen=True)
class AdaptiveVDAConfig:
    """Frozen V1/V2 routing and V2 fusion thresholds."""

    hard_minimum_valid_keyframes: int = 4
    hard_scale_relative_mad: float = 0.20
    wrist_patch_radius_px: int = 5
    wrist_patch_percentile: float = 25.0
    minimum_valid_patch_pixels: int = 10
    minimum_robust_anchors: int = 4
    temporal_outlier_sigma_mad: float = 3.0
    temporal_outlier_floor_m: float = 0.02
    residual_clip_m: float = 0.04
    fusion_beta: float = 0.5
    absolute_center_gate_m: float = 0.15
    bir_transition_frames: int = 15

    def __post_init__(self) -> None:
        if self.hard_minimum_valid_keyframes < 1:
            raise ValueError("hard_minimum_valid_keyframes must be positive")
        if self.hard_scale_relative_mad <= 0:
            raise ValueError("hard_scale_relative_mad must be positive")
        if self.wrist_patch_radius_px < 0:
            raise ValueError("wrist_patch_radius_px cannot be negative")
        if not 0 <= self.wrist_patch_percentile <= 100:
            raise ValueError("wrist_patch_percentile must be in [0, 100]")
        if self.minimum_valid_patch_pixels < 1 or self.minimum_robust_anchors < 2:
            raise ValueError("support thresholds are too small")
        if self.residual_clip_m <= 0 or self.absolute_center_gate_m <= 0:
            raise ValueError("metric bounds must be positive")
        if not 0 <= self.fusion_beta <= 1:
            raise ValueError("fusion_beta must be in [0, 1]")
        if self.bir_transition_frames < 1:
            raise ValueError("bir_transition_frames must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def is_hard_sequence(
    valid_keyframes: int,
    scale_relative_mad: float,
    config: AdaptiveVDAConfig,
) -> bool:
    """Return the frozen, GT-free V1 dense-recovery routing decision."""

    return (
        int(valid_keyframes) < config.hard_minimum_valid_keyframes
        or float(scale_relative_mad) > config.hard_scale_relative_mad
    )


def map_relative_depth(relative: np.ndarray, mapping: Mapping[str, Any]) -> np.ndarray:
    """Map VDA relative depth to metric depth using a frozen calibration."""

    linear = (
        np.asarray(relative, dtype=np.float64) * float(mapping["slope"])
        + float(mapping["intercept"])
    )
    kind = str(mapping["kind"])
    if kind == "direct_depth":
        return linear
    if kind == "inverse_depth":
        return np.divide(
            1.0,
            linear,
            out=np.full_like(linear, np.nan),
            where=linear > 1e-8,
        )
    raise ValueError(f"unknown VDA depth mapping {kind!r}")


def project_camera_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project camera-frame XYZ points to pixels without changing coordinates."""

    points = np.asarray(points_camera, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    z = points[..., 2]
    result = np.full(points.shape[:-1] + (2,), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=-1) & (z > 1e-8)
    if matrix.ndim == 2:
        result[..., 0] = matrix[0, 0] * points[..., 0] / z + matrix[0, 2]
        result[..., 1] = matrix[1, 1] * points[..., 1] / z + matrix[1, 2]
    elif matrix.ndim == 3 and points.ndim >= 3 and len(matrix) == len(points):
        result[..., 0] = (
            matrix[:, 0, 0, None] * points[..., 0] / z + matrix[:, 0, 2, None]
        )
        result[..., 1] = (
            matrix[:, 1, 1, None] * points[..., 1] / z + matrix[:, 1, 2, None]
        )
    else:
        raise ValueError(
            f"intrinsics shape {matrix.shape} is incompatible with points {points.shape}"
        )
    result[~valid] = np.nan
    return result


def estimate_wrist_patch_residual(
    metric_depth: np.ndarray,
    wrist_camera: np.ndarray,
    intrinsics: np.ndarray,
    config: AdaptiveVDAConfig,
) -> tuple[float, dict[str, float | int]] | None:
    """Estimate observed-minus-predicted wrist depth from a robust image patch."""

    point = np.asarray(wrist_camera, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all() or point[2] <= 0.05:
        return None
    uv = project_camera_points(point[None], intrinsics)[0]
    if not np.isfinite(uv).all():
        return None
    x, y = np.rint(uv).astype(int)
    depth = np.asarray(metric_depth, dtype=np.float64)
    height, width = depth.shape
    if not (0 <= x < width and 0 <= y < height):
        return None
    radius = config.wrist_patch_radius_px
    patch = depth[
        max(0, y - radius) : min(height, y + radius + 1),
        max(0, x - radius) : min(width, x + radius + 1),
    ]
    values = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 10.0)]
    if len(values) < config.minimum_valid_patch_pixels:
        return None
    observed = float(np.percentile(values, config.wrist_patch_percentile))
    q25, q75 = np.percentile(values, (25, 75))
    return observed - float(point[2]), {
        "valid_pixels": int(len(values)),
        "patch_iqr_m": float(q75 - q25),
    }


def robust_temporal_correction(
    frames: int,
    estimates: Sequence[tuple[int, float, Mapping[str, float | int]]],
    config: AdaptiveVDAConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reject residual outliers, remove global bias, interpolate, and clip."""

    if frames < 1:
        raise ValueError("frames must be positive")
    if len(estimates) < config.minimum_robust_anchors:
        return np.zeros(frames, dtype=np.float64), {
            "accepted": False,
            "reason": "insufficient_anchors",
            "raw_anchors": int(len(estimates)),
        }
    ordered = sorted(estimates, key=lambda item: int(item[0]))
    times = np.asarray([item[0] for item in ordered], dtype=np.int64)
    values = np.asarray([item[1] for item in ordered], dtype=np.float64)
    if times[0] < 0 or times[-1] >= frames or np.any(np.diff(times) <= 0):
        raise ValueError("anchor frame indices must be unique, increasing, and in range")
    center = float(np.median(values))
    temporal_mad = float(np.median(np.abs(values - center)))
    threshold = max(
        config.temporal_outlier_sigma_mad * 1.4826 * temporal_mad,
        config.temporal_outlier_floor_m,
    )
    keep = np.abs(values - center) <= threshold
    kept_times = times[keep]
    kept_values = values[keep]
    if len(kept_values) < config.minimum_robust_anchors:
        return np.zeros(frames, dtype=np.float64), {
            "accepted": False,
            "reason": "insufficient_robust_anchors",
            "raw_anchors": int(len(values)),
            "kept_anchors": int(len(kept_values)),
        }
    center = float(np.median(kept_values))
    centered = np.clip(
        kept_values - center,
        -config.residual_clip_m,
        config.residual_clip_m,
    )
    correction = np.interp(np.arange(frames), kept_times, centered)
    patch_iqr = [float(item[2]["patch_iqr_m"]) for item in ordered]
    return correction, {
        "accepted": True,
        "raw_anchors": int(len(values)),
        "kept_anchors": int(len(kept_values)),
        "center_m": center,
        "temporal_mad_m": temporal_mad,
        "outlier_threshold_m": float(threshold),
        "correction_abs_p95_before_beta_m": float(
            np.percentile(np.abs(correction), 95)
        ),
        "mean_patch_iqr_m": float(np.mean(patch_iqr)),
    }


def interaction_protection_weight(
    payloads: Mapping[str, Mapping[str, np.ndarray]],
    frames: int,
    config: AdaptiveVDAConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend both hands to a common shift around Stage-3-BIR interaction frames."""

    core = np.zeros(frames, dtype=bool)
    for hand in HANDS:
        payload = payloads[hand]
        core |= np.asarray(payload["stage3_bir_adjusted_mask"], dtype=bool)
        core |= np.asarray(payload["stage3_bir_penetration_detected_after"], dtype=bool)
    if not core.any():
        return core, np.zeros(frames, dtype=np.float64)
    positions = np.flatnonzero(core)
    distance = np.min(
        np.abs(np.arange(frames)[:, None] - positions[None, :]), axis=1
    )
    weight = np.clip(1.0 - distance / config.bir_transition_frames, 0.0, 1.0)
    return core, weight


def camera_z_to_world(
    camera_r_c2w: np.ndarray,
    delta_z_camera_m: np.ndarray,
) -> np.ndarray:
    """Rotate a camera +Z translation into the world frame."""

    rotation = np.asarray(camera_r_c2w, dtype=np.float64)
    delta = np.asarray(delta_z_camera_m, dtype=np.float64)
    if rotation.shape != (len(delta), 3, 3):
        raise ValueError("camera_R_c2w must have shape (T, 3, 3)")
    camera_shift = np.zeros((len(delta), 3), dtype=np.float64)
    camera_shift[:, 2] = delta
    return np.einsum("tij,tj->ti", rotation, camera_shift)


def apply_root_depth_correction(
    payload: Mapping[str, np.ndarray],
    delta_z_camera_m: np.ndarray,
    detector: Mapping[str, np.ndarray],
    config: AdaptiveVDAConfig,
) -> dict[str, np.ndarray]:
    """Apply one bounded root-depth correction while preserving local geometry."""

    frames = len(np.asarray(payload["frame_index"]))
    delta = np.asarray(delta_z_camera_m, dtype=np.float64)
    if delta.shape != (frames,) or not np.isfinite(delta).all():
        raise ValueError("delta_z_camera_m must be a finite (T,) array")
    if np.max(np.abs(delta), initial=0.0) > config.residual_clip_m * config.fusion_beta + 1e-10:
        raise ValueError("depth correction exceeds the configured fused bound")
    world_shift = camera_z_to_world(payload["camera_R_c2w"], delta)
    camera_shift = np.zeros((frames, 3), dtype=np.float64)
    camera_shift[:, 2] = delta
    result = {key: np.asarray(value).copy() for key, value in payload.items()}
    if np.any(delta != 0.0):
        result["transl"] = (
            np.asarray(payload["transl"], dtype=np.float64) + world_shift
        ).astype(np.asarray(payload["transl"]).dtype)
        for key in ("joints_3d_world", "vertices_world"):
            result[key] = (
                np.asarray(payload[key], dtype=np.float64) + world_shift[:, None, :]
            ).astype(np.asarray(payload[key]).dtype)
        for key in ("joints_3d_camera", "vertices_camera"):
            result[key] = (
                np.asarray(payload[key], dtype=np.float64) + camera_shift[:, None, :]
            ).astype(np.asarray(payload[key]).dtype)
        projected = project_camera_points(
            result["joints_3d_camera"], np.asarray(payload["camera_intrinsics"])
        )
        result["joints_2d"] = projected.astype(np.asarray(payload["joints_2d"]).dtype)
        width, height = map(int, np.asarray(payload["image_size"]).reshape(2))
        result["joints_in_frame"] = (
            np.isfinite(projected).all(axis=-1)
            & (result["joints_3d_camera"][..., 2] > 0)
            & (projected[..., 0] >= 0)
            & (projected[..., 0] < width)
            & (projected[..., 1] >= 0)
            & (projected[..., 1] < height)
        )
    result.update(
        {
            "adaptive_vda_version": np.asarray(VERSION),
            "adaptive_vda_config_sha256": np.asarray(config.sha256()),
            "adaptive_vda_candidate_generated_without_gt": np.asarray(True),
            "adaptive_vda_depth_correction_camera_z_m": delta.astype(np.float32),
            "adaptive_vda_world_shift_m": world_shift.astype(np.float32),
            "adaptive_vda_penetration_detected": np.asarray(
                detector["penetration_detected"], dtype=bool
            ),
            "adaptive_vda_penetration_energy": np.asarray(
                detector["penetration_energy"], dtype=np.float32
            ),
            "adaptive_vda_max_depth_proxy_m": np.asarray(
                detector["max_nearest_vertex_depth_proxy_m"], dtype=np.float32
            ),
        }
    )
    return result


def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Strict equality helper that treats same-position NaNs as equal."""

    first = np.asarray(left)
    second = np.asarray(right)
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    if np.issubdtype(first.dtype, np.inexact):
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))
