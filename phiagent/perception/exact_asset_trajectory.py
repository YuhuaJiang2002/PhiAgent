"""Exact-asset articulated trajectory fitting and fail-closed validation.

Foundation models may propose 2-D robot landmarks, masks, and initialization.
Only a hash-bound kinematic model, pinhole reprojection, held-out observations,
and an identifiable full-q posterior can promote the result. Numerical routines
accept a NumPy-compatible module so importing :mod:`phiagent` stays lightweight.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExactAssetTrajectoryContract:
    """Frozen acceptance contract for camera-relative full-q reconstruction."""

    embodiment_id: str
    camera_frame: str
    robot_base_frame: str
    timeline: str
    source_video_sha256: str
    fps: float
    joint_names: tuple[str, ...]
    joint_limits_rad: tuple[tuple[float, float], ...]
    asset_sha256: Mapping[str, str]
    expected_asset_sha256: Mapping[str, str]
    minimum_visible_keypoints_per_frame: int = 12
    minimum_heldout_frames: int = 4
    minimum_heldout_groups: int = 2
    maximum_reprojection_rmse_px_p95: float = 8.0
    minimum_silhouette_iou_p05: float = 0.65
    maximum_joint_standard_deviation_rad: float = 0.08
    maximum_base_translation_standard_deviation_m: float = 0.02
    minimum_alternative_asset_error_margin_px_p05: float = 4.0
    maximum_joint_velocity_rad_s: float = 12.0

    def validate(self) -> None:
        labels = (
            self.embodiment_id,
            self.camera_frame,
            self.robot_base_frame,
            self.timeline,
        )
        if any(not value.strip() for value in labels):
            raise ValueError("embodiment, frames, and timeline must be named")
        if len(self.source_video_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_video_sha256.lower()
        ):
            raise ValueError("source video must use a SHA-256 digest")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("trajectory FPS must be finite and positive")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint names must be non-empty and unique")
        if len(self.joint_limits_rad) != len(self.joint_names):
            raise ValueError("every joint requires one limit interval")
        if any(lower > upper for lower, upper in self.joint_limits_rad):
            raise ValueError("joint lower limit exceeds upper limit")
        all_hashes = tuple(self.expected_asset_sha256.values()) + tuple(
            self.asset_sha256.values()
        )
        if not self.expected_asset_sha256 or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
            for value in all_hashes
        ):
            raise ValueError("expected assets require SHA-256 digests")
        positive = (
            self.minimum_visible_keypoints_per_frame,
            self.minimum_heldout_frames,
            self.minimum_heldout_groups,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("observation counts must be positive")
        finite_positive = (
            self.maximum_reprojection_rmse_px_p95,
            self.minimum_silhouette_iou_p05,
            self.maximum_joint_standard_deviation_rad,
            self.maximum_base_translation_standard_deviation_m,
            self.minimum_alternative_asset_error_margin_px_p05,
            self.maximum_joint_velocity_rad_s,
        )
        if any(not math.isfinite(value) or value <= 0 for value in finite_positive):
            raise ValueError("trajectory thresholds must be finite and positive")
        if self.minimum_silhouette_iou_p05 > 1:
            raise ValueError("silhouette IoU threshold cannot exceed one")


def _rodrigues(np: Any, rotation_vector: Any) -> Any:
    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        skew = np.asarray(
            (
                (0.0, -vector[2], vector[1]),
                (vector[2], 0.0, -vector[0]),
                (-vector[1], vector[0], 0.0),
            ),
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + skew
    axis = vector / angle
    skew = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        ),
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def _project_points(
    np: Any,
    *,
    points_robot_base_m: Any,
    intrinsics_px: Any,
    rotation_vector: Any,
    translation_m: Any,
) -> tuple[Any, Any]:
    rotation = _rodrigues(np, rotation_vector)
    camera_points = (rotation @ points_robot_base_m.T).T + translation_m[None, :]
    depth = camera_points[:, 2]
    safe_depth = np.where(np.abs(depth) > 1e-9, depth, np.nan)
    normalized = camera_points[:, :2] / safe_depth[:, None]
    pixels = np.empty_like(normalized)
    pixels[:, 0] = (
        intrinsics_px[0, 0] * normalized[:, 0] + intrinsics_px[0, 2]
    )
    pixels[:, 1] = (
        intrinsics_px[1, 1] * normalized[:, 1] + intrinsics_px[1, 2]
    )
    return pixels, depth


def fit_articulated_keypoints_frame(
    np: Any,
    *,
    forward_keypoints_robot_base_m: Callable[[Any], Any],
    intrinsics_px: Any,
    observed_keypoints_px: Any,
    keypoint_confidence: Any,
    initial_joint_positions_rad: Any,
    joint_limits_rad: Any,
    initial_rotation_vector: Any,
    initial_translation_m: Any,
    minimum_keypoint_confidence: float = 0.2,
    huber_delta_px: float = 6.0,
    prior_weight: float = 1e-4,
    finite_difference_step: float = 1e-5,
    maximum_iterations: int = 40,
    maximum_data_jacobian_condition_number: float = 1e10,
) -> dict[str, object]:
    """Fit one exact articulated model to 2-D proposals with finite-difference LM.

    The forward callback is supplied by an optional exact-asset adapter such as
    MuJoCo. Priors may stabilize optimization but never contribute to the
    reported data-Jacobian rank or posterior observability.
    """

    intrinsics = np.asarray(intrinsics_px, dtype=np.float64)
    observed = np.asarray(observed_keypoints_px, dtype=np.float64)
    confidence = np.asarray(keypoint_confidence, dtype=np.float64)
    initial_q = np.asarray(initial_joint_positions_rad, dtype=np.float64)
    limits = np.asarray(joint_limits_rad, dtype=np.float64)
    rotation = np.asarray(initial_rotation_vector, dtype=np.float64)
    translation = np.asarray(initial_translation_m, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape 3x3")
    if observed.ndim != 2 or observed.shape[1] != 2:
        raise ValueError("observed keypoints must have shape Kx2")
    if confidence.shape != (len(observed),):
        raise ValueError("keypoint confidence must have shape K")
    if limits.shape != (len(initial_q), 2):
        raise ValueError("joint limits must have shape Jx2")
    if rotation.shape != (3,) or translation.shape != (3,):
        raise ValueError("base rotation and translation must have shape 3")
    if not 0 <= minimum_keypoint_confidence <= 1:
        raise ValueError("minimum keypoint confidence must lie in [0,1]")
    if (
        maximum_iterations < 1
        or finite_difference_step <= 0
        or maximum_data_jacobian_condition_number <= 1
    ):
        raise ValueError("solver iterations and finite-difference step must be positive")

    valid = (
        np.all(np.isfinite(observed), axis=1)
        & np.isfinite(confidence)
        & (confidence >= minimum_keypoint_confidence)
    )
    if int(np.count_nonzero(valid)) < 4:
        raise ValueError("at least four visible keypoints are required")
    parameters = np.concatenate((rotation, translation, initial_q))
    lower = limits[:, 0]
    upper = limits[:, 1]

    def data_residual(value: Any) -> tuple[Any, Any, Any]:
        q = value[6:]
        points = np.asarray(forward_keypoints_robot_base_m(q), dtype=np.float64)
        if points.shape != (len(observed), 3):
            raise ValueError("forward keypoints must have shape Kx3")
        pixels, depth = _project_points(
            np,
            points_robot_base_m=points,
            intrinsics_px=intrinsics,
            rotation_vector=value[:3],
            translation_m=value[3:6],
        )
        selected_depth = depth[valid]
        if not bool(
            np.all(np.isfinite(selected_depth)) and np.all(selected_depth > 1e-3)
        ):
            return np.full(int(np.count_nonzero(valid)) * 2, 1e6), pixels, depth
        residual = pixels[valid] - observed[valid]
        if not bool(np.all(np.isfinite(residual))):
            return np.full(int(np.count_nonzero(valid)) * 2, 1e6), pixels, depth
        return np.clip(residual.reshape(-1), -1e6, 1e6), pixels, depth

    damping = 1e-2
    iterations = 0
    converged = False
    for iteration in range(maximum_iterations):
        residual, _, _ = data_residual(parameters)
        point_error = np.linalg.norm(residual.reshape(-1, 2), axis=1)
        robust = np.where(
            point_error <= huber_delta_px,
            1.0,
            huber_delta_px / np.maximum(point_error, 1e-12),
        )
        coordinate_weights = np.repeat(
            np.sqrt(robust * confidence[valid]), 2
        )
        jacobian = np.empty((len(residual), len(parameters)), dtype=np.float64)
        for column in range(len(parameters)):
            shifted = parameters.copy()
            shifted[column] += finite_difference_step
            shifted_residual, _, _ = data_residual(shifted)
            jacobian[:, column] = (
                shifted_residual - residual
            ) / finite_difference_step
        weighted_residual = residual * coordinate_weights
        weighted_jacobian = jacobian * coordinate_weights[:, None]
        prior_residual = math.sqrt(prior_weight) * (parameters[6:] - initial_q)
        prior_jacobian = np.zeros(
            (len(initial_q), len(parameters)), dtype=np.float64
        )
        prior_jacobian[:, 6:] = math.sqrt(prior_weight) * np.eye(len(initial_q))
        system_jacobian = np.concatenate((weighted_jacobian, prior_jacobian), axis=0)
        system_residual = np.concatenate((weighted_residual, prior_residual))
        normal = np.einsum("ki,kj->ij", system_jacobian, system_jacobian)
        gradient = np.einsum("ki,k->i", system_jacobian, system_residual)
        diagonal = np.maximum(np.diag(normal), 1.0)
        try:
            update = np.linalg.solve(normal + damping * np.diag(diagonal), -gradient)
        except np.linalg.LinAlgError:
            update = np.linalg.lstsq(
                normal + damping * np.diag(diagonal), -gradient, rcond=None
            )[0]
        bounded_update = update.copy()
        bounded_update[:3] = np.clip(bounded_update[:3], -0.25, 0.25)
        bounded_update[3:6] = np.clip(bounded_update[3:6], -0.25, 0.25)
        bounded_update[6:] = np.clip(bounded_update[6:], -0.35, 0.35)
        challenger = parameters + bounded_update
        challenger[:3] = np.clip(challenger[:3], -math.pi, math.pi)
        challenger[3:5] = np.clip(challenger[3:5], -10.0, 10.0)
        challenger[6:] = np.clip(challenger[6:], lower, upper)
        challenger[5] = float(np.clip(challenger[5], 1e-3, 100.0))
        next_residual, _, _ = data_residual(challenger)
        current_cost = float(np.mean(np.square(weighted_residual)))
        next_cost = float(
            np.mean(np.square(next_residual * coordinate_weights))
        )
        if next_cost < current_cost:
            parameters = challenger
            damping = max(damping / 3.0, 1e-8)
            if float(np.linalg.norm(bounded_update)) <= 1e-7:
                converged = True
                iterations = iteration + 1
                break
        else:
            damping = min(damping * 10.0, 1e10)
        iterations = iteration + 1

    residual, rendered, depth = data_residual(parameters)
    final_jacobian = np.empty((len(residual), len(parameters)), dtype=np.float64)
    for column in range(len(parameters)):
        shifted = parameters.copy()
        shifted[column] += finite_difference_step
        shifted_residual, _, _ = data_residual(shifted)
        final_jacobian[:, column] = (
            shifted_residual - residual
        ) / finite_difference_step
    _, singular_values, right_singular_vectors = np.linalg.svd(
        final_jacobian, full_matrices=False
    )
    tolerance = (
        max(final_jacobian.shape)
        * np.finfo(np.float64).eps
        * float(singular_values[0])
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) == len(parameters) and singular_values[-1] > 0
        else float("inf")
    )
    identifiable = (
        rank == len(parameters)
        and condition_number <= maximum_data_jacobian_condition_number
    )
    if identifiable:
        degrees_of_freedom = max(len(residual) - len(parameters), 1)
        variance = float(np.sum(np.square(residual)) / degrees_of_freedom)
        inverse_normal = np.einsum(
            "ki,k,kj->ij",
            right_singular_vectors,
            1.0 / np.square(singular_values),
            right_singular_vectors,
        )
        covariance = variance * inverse_normal
        parameter_std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    else:
        covariance = np.full(
            (len(parameters), len(parameters)), np.nan, dtype=np.float64
        )
        parameter_std = np.full(len(parameters), np.inf, dtype=np.float64)
    camera_from_robot_base = np.eye(4, dtype=np.float64)
    camera_from_robot_base[:3, :3] = _rodrigues(np, parameters[:3])
    camera_from_robot_base[:3, 3] = parameters[3:6]
    point_error = np.linalg.norm(residual.reshape(-1, 2), axis=1)
    return {
        "joint_positions_rad": parameters[6:],
        "camera_from_robot_base": camera_from_robot_base,
        "rendered_keypoints_px": rendered,
        "keypoint_depth_m": depth,
        "reprojection_rmse_px": float(np.sqrt(np.mean(np.square(point_error)))),
        "reprojection_error_px_p95": float(np.percentile(point_error, 95)),
        "data_jacobian_rank": rank,
        "data_jacobian_condition_number": condition_number,
        "parameter_count": len(parameters),
        "identifiable": identifiable,
        "parameter_standard_deviation": parameter_std,
        "covariance": covariance,
        "iterations": iterations,
        "converged": converged,
        "visible_keypoints": int(np.count_nonzero(valid)),
    }


def validate_exact_asset_trajectory(
    np: Any,
    *,
    contract: ExactAssetTrajectoryContract,
    evidence_source_video_sha256: str,
    frame_indices: Any,
    joint_positions_rad: Any,
    camera_from_robot_base: Any,
    observed_keypoints_px: Any,
    rendered_keypoints_px: Any,
    keypoint_confidence: Any,
    fit_frame_mask: Any,
    heldout_group_ids: Sequence[str],
    silhouette_iou: Any,
    joint_standard_deviation_rad: Any,
    base_translation_standard_deviation_m: Any,
    alternative_asset_reprojection_rmse_px: Any,
    joint_velocities_rad_s: Any | None = None,
    minimum_keypoint_confidence: float = 0.2,
) -> dict[str, object]:
    """Validate a complete trajectory without letting priors fill unseen joints."""

    contract.validate()
    frames = np.asarray(frame_indices, dtype=np.int64)
    q = np.asarray(joint_positions_rad, dtype=np.float64)
    poses = np.asarray(camera_from_robot_base, dtype=np.float64)
    observed = np.asarray(observed_keypoints_px, dtype=np.float64)
    rendered = np.asarray(rendered_keypoints_px, dtype=np.float64)
    confidence = np.asarray(keypoint_confidence, dtype=np.float64)
    fit_mask = np.asarray(fit_frame_mask, dtype=bool)
    silhouette = np.asarray(silhouette_iou, dtype=np.float64)
    q_std = np.asarray(joint_standard_deviation_rad, dtype=np.float64)
    translation_std = np.asarray(
        base_translation_standard_deviation_m, dtype=np.float64
    )
    alternatives = np.asarray(
        alternative_asset_reprojection_rmse_px, dtype=np.float64
    )
    groups = np.asarray(tuple(str(value) for value in heldout_group_ids))
    frame_count = len(frames)
    joint_count = len(contract.joint_names)

    if frames.ndim != 1 or frame_count < 2 or bool(np.any(np.diff(frames) <= 0)):
        raise ValueError("frame indices must be increasing and contain two frames")
    if q.shape != (frame_count, joint_count):
        raise ValueError("joint positions must contain complete TxJ generalized coordinates")
    if poses.shape != (frame_count, 4, 4):
        raise ValueError("camera_from_robot_base must have shape Tx4x4")
    if observed.ndim != 3 or observed.shape[0] != frame_count or observed.shape[2] != 2:
        raise ValueError("observed keypoints must have shape TxKx2")
    if rendered.shape != observed.shape or confidence.shape != observed.shape[:2]:
        raise ValueError("rendered keypoints and confidence must align with observations")
    if fit_mask.shape != (frame_count,) or len(groups) != frame_count:
        raise ValueError("fit mask and heldout groups must align with frames")
    if silhouette.shape != (frame_count,) or q_std.shape != q.shape:
        raise ValueError("silhouette and q uncertainty must align with frames")
    if translation_std.shape != (frame_count, 3):
        raise ValueError("base translation uncertainty must have shape Tx3")
    if alternatives.ndim != 2 or alternatives.shape[0] != frame_count:
        raise ValueError("alternative-asset errors must have shape TxA")

    finite_q = bool(np.all(np.isfinite(q)))
    limits = np.asarray(contract.joint_limits_rad, dtype=np.float64)
    limits_passed = bool(
        finite_q
        and np.all(q >= limits[None, :, 0])
        and np.all(q <= limits[None, :, 1])
    )
    if joint_velocities_rad_s is None:
        delta_t = np.diff(frames).astype(np.float64) / contract.fps
        velocity = np.diff(q, axis=0) / delta_t[:, None]
    else:
        velocity = np.asarray(joint_velocities_rad_s, dtype=np.float64)
        if velocity.shape != q.shape:
            raise ValueError("joint velocities must have shape TxJ")
    velocity_passed = bool(
        np.all(np.isfinite(velocity))
        and np.max(np.abs(velocity)) <= contract.maximum_joint_velocity_rad_s
    )

    bottom = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    rotations = poses[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)
    orthogonality = float(
        np.max(
            np.linalg.norm(
                np.swapaxes(rotations, 1, 2) @ rotations - identity,
                axis=(1, 2),
            )
        )
    )
    proper_se3 = bool(
        np.all(np.isfinite(poses))
        and np.max(np.abs(poses[:, 3, :] - bottom[None, :])) <= 1e-6
        and orthogonality <= 1e-3
        and np.max(np.abs(np.linalg.det(rotations) - 1.0)) <= 1e-3
    )

    visible = (
        np.all(np.isfinite(observed), axis=2)
        & np.all(np.isfinite(rendered), axis=2)
        & np.isfinite(confidence)
        & (confidence >= minimum_keypoint_confidence)
    )
    visible_count = np.count_nonzero(visible, axis=1)
    frame_rmse = np.full(frame_count, np.inf, dtype=np.float64)
    for index in range(frame_count):
        if visible_count[index]:
            error = rendered[index, visible[index]] - observed[index, visible[index]]
            frame_rmse[index] = float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))
    heldout = ~fit_mask
    heldout_frames = int(np.count_nonzero(heldout))
    heldout_groups = sorted(
        str(value) for value in np.unique(groups[heldout]) if str(value).strip()
    )
    group_rows = []
    for group in heldout_groups:
        selected = heldout & (groups == group)
        group_rows.append(
            {
                "group_id": group,
                "frames": int(np.count_nonzero(selected)),
                "reprojection_rmse_px_p95": float(
                    np.percentile(frame_rmse[selected], 95)
                ),
            }
        )
    heldout_reprojection_p95 = (
        float(np.percentile(frame_rmse[heldout], 95))
        if heldout_frames
        else float("inf")
    )
    silhouette_p05 = (
        float(np.percentile(silhouette[heldout], 5))
        if heldout_frames
        else float("-inf")
    )
    q_std_max = float(np.max(q_std)) if q_std.size else float("inf")
    translation_std_max = (
        float(np.max(translation_std)) if translation_std.size else float("inf")
    )
    if alternatives.shape[1] > 0 and heldout_frames:
        identity_margin = np.min(alternatives, axis=1) - frame_rmse
        identity_margin_p05 = float(np.percentile(identity_margin[heldout], 5))
    else:
        identity_margin_p05 = float("-inf")
    asset_hashes_match = bool(
        dict(contract.asset_sha256) == dict(contract.expected_asset_sha256)
    )
    gates = {
        "source_video_hash_bound": (
            evidence_source_video_sha256 == contract.source_video_sha256
        ),
        "exact_asset_hashes_match_registry": asset_hashes_match,
        "complete_finite_q": finite_q,
        "joint_limits_passed": limits_passed,
        "joint_velocity_passed": velocity_passed,
        "proper_camera_from_robot_base_se3": proper_se3,
        "visible_keypoint_coverage": bool(
            np.all(visible_count >= contract.minimum_visible_keypoints_per_frame)
        ),
        "heldout_frame_count": heldout_frames >= contract.minimum_heldout_frames,
        "heldout_group_count": len(heldout_groups) >= contract.minimum_heldout_groups,
        "heldout_reprojection_bounded": (
            heldout_reprojection_p95 <= contract.maximum_reprojection_rmse_px_p95
        ),
        "heldout_silhouette_iou_bounded": (
            silhouette_p05 >= contract.minimum_silhouette_iou_p05
        ),
        "full_q_posterior_observable": (
            bool(np.all(np.isfinite(q_std)))
            and q_std_max <= contract.maximum_joint_standard_deviation_rad
        ),
        "metric_base_translation_observable": (
            bool(np.all(np.isfinite(translation_std)))
            and translation_std_max
            <= contract.maximum_base_translation_standard_deviation_m
        ),
        "selected_asset_beats_alternatives": (
            identity_margin_p05
            >= contract.minimum_alternative_asset_error_margin_px_p05
        ),
    }
    proposal_gate_names = {
        "complete_finite_q",
        "joint_limits_passed",
        "joint_velocity_passed",
        "proper_camera_from_robot_base_se3",
        "visible_keypoint_coverage",
    }
    return {
        "frames": frame_count,
        "joints": joint_count,
        "visible_keypoints_per_frame_min": int(np.min(visible_count)),
        "heldout_frames": heldout_frames,
        "heldout_groups": heldout_groups,
        "heldout_group_metrics": group_rows,
        "heldout_reprojection_rmse_px_p95": heldout_reprojection_p95,
        "heldout_silhouette_iou_p05": silhouette_p05,
        "joint_standard_deviation_rad_max": q_std_max,
        "base_translation_standard_deviation_m_max": translation_std_max,
        "alternative_asset_error_margin_px_p05": identity_margin_p05,
        "rotation_orthogonality_error": orthogonality,
        "gates": gates,
        "proposal_passed": all(gates[name] for name in proposal_gate_names),
        "passed": all(gates.values()),
        "reasons": [name for name, passed in gates.items() if not passed],
    }
