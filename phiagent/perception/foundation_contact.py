"""Fail-closed contracts for foundation-model-assisted physical reconstruction.

Foundation models are useful proposal mechanisms, but a plausible prediction is
not a measurement.  This module keeps that distinction machine-readable while
remaining importable without NumPy, PyTorch, CUDA, simulators, or checkpoints.
Callers pass a NumPy-compatible module to the numerical validators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class EvidenceClass(str, Enum):
    """How an estimate was obtained, ordered by semantics rather than confidence."""

    SENSOR_MEASUREMENT = "sensor_measurement"
    CALIBRATED_GEOMETRY = "calibrated_geometry"
    EXACT_ASSET = "exact_asset"
    FOUNDATION_MODEL_ESTIMATE = "foundation_model_estimate"
    PHYSICS_SOLVER_ESTIMATE = "physics_solver_estimate"
    HEURISTIC_PROXY = "heuristic_proxy"


@dataclass(frozen=True)
class ModelProvenance:
    name: str
    revision: str
    repository: str
    license: str
    purpose: str

    def validate(self) -> None:
        values = (self.name, self.revision, self.repository, self.license, self.purpose)
        if any(not value.strip() for value in values):
            raise ValueError("model provenance fields must be non-empty")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "name": self.name,
            "revision": self.revision,
            "repository": self.repository,
            "license": self.license,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class MetricCameraContract:
    """Metric camera/depth evidence for a sampled video timeline."""

    camera_frame: str
    world_frame: str
    timeline: str
    fps: float
    image_width: int
    image_height: int
    intrinsics_evidence: EvidenceClass
    depth_evidence: EvidenceClass
    metric_scale_source: str
    learned_context_scale_variation_fraction: float | None = None
    absolute_scale_standard_deviation_fraction: float | None = None
    independent_calibration_groups: int = 0
    calibration_report_sha256: str | None = None

    def validate_metadata(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.camera_frame,
                self.world_frame,
                self.timeline,
                self.metric_scale_source,
            )
        ):
            raise ValueError("camera frames, timeline, and scale source must be named")
        if self.fps <= 0 or not math.isfinite(self.fps):
            raise ValueError("camera FPS must be finite and positive")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("camera image dimensions must be positive")
        if self.learned_context_scale_variation_fraction is not None and not (
            math.isfinite(self.learned_context_scale_variation_fraction)
            and self.learned_context_scale_variation_fraction >= 0
        ):
            raise ValueError("learned context-scale variation must be finite and non-negative")
        if self.absolute_scale_standard_deviation_fraction is not None and not (
            math.isfinite(self.absolute_scale_standard_deviation_fraction)
            and self.absolute_scale_standard_deviation_fraction >= 0
        ):
            raise ValueError("absolute scale uncertainty must be finite and non-negative")
        if self.independent_calibration_groups < 0:
            raise ValueError("independent calibration group count cannot be negative")
        if self.calibration_report_sha256 is not None and len(
            self.calibration_report_sha256
        ) != 64:
            raise ValueError("calibration report digest must be SHA-256")


@dataclass(frozen=True)
class RobotTrajectoryContract:
    """Full generalized-coordinate trajectory tied to immutable robot assets."""

    embodiment_id: str
    robot_base_frame: str
    timeline: str
    fps: float
    joint_names: tuple[str, ...]
    joint_limits_rad: tuple[tuple[float, float], ...]
    asset_sha256: Mapping[str, str]
    trajectory_evidence: EvidenceClass

    def validate_metadata(self) -> None:
        if any(
            not value.strip()
            for value in (self.embodiment_id, self.robot_base_frame, self.timeline)
        ):
            raise ValueError("robot embodiment, frame, and timeline must be named")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("robot trajectory FPS must be finite and positive")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("robot joint names must be non-empty and unique")
        if len(self.joint_limits_rad) != len(self.joint_names):
            raise ValueError("every robot joint requires one limit interval")
        if any(lower > upper for lower, upper in self.joint_limits_rad):
            raise ValueError("robot joint lower limit exceeds upper limit")
        if not self.asset_sha256 or any(
            not name.strip() or len(digest) != 64
            for name, digest in self.asset_sha256.items()
        ):
            raise ValueError("robot assets require named SHA-256 digests")


@dataclass(frozen=True)
class StemCenterlineContract:
    """Persistent, metric, per-instance deformable centerline observations."""

    instance_ids: tuple[str, ...]
    coordinate_frame: str
    timeline: str
    nodes_per_stem: int
    geometry_evidence: EvidenceClass

    def validate_metadata(self) -> None:
        if not self.instance_ids or len(set(self.instance_ids)) != len(self.instance_ids):
            raise ValueError("stem instance IDs must be non-empty and unique")
        if any(not value.strip() for value in self.instance_ids):
            raise ValueError("stem instance IDs cannot be blank")
        if not self.coordinate_frame.strip() or not self.timeline.strip():
            raise ValueError("stem coordinate frame and timeline must be named")
        if self.nodes_per_stem < 3:
            raise ValueError("each stem centerline requires at least three nodes")


@dataclass(frozen=True)
class ContactForceContract:
    """Force evidence whose provenance cannot be a visual foundation model."""

    coordinate_frame: str
    timeline: str
    instance_ids: tuple[str, ...]
    force_evidence: EvidenceClass
    source_name: str

    def validate_metadata(self) -> None:
        if not self.coordinate_frame.strip() or not self.timeline.strip():
            raise ValueError("force coordinate frame and timeline must be named")
        if not self.source_name.strip() or not self.instance_ids:
            raise ValueError("force source and instance IDs must be named")
        allowed = {
            EvidenceClass.SENSOR_MEASUREMENT,
            EvidenceClass.PHYSICS_SOLVER_ESTIMATE,
        }
        if self.force_evidence not in allowed:
            raise ValueError(
                "contact force must come from a sensor or physics solver, never a visual model"
            )


def validate_metric_camera_sequence(
    np: Any,
    *,
    contract: MetricCameraContract,
    frame_indices: Any,
    intrinsics_px: Any,
    world_from_camera: Any,
    depth_m: Any | None,
    depth_confidence: Any | None,
    minimum_valid_depth_fraction: float = 0.90,
    maximum_rotation_orthogonality_error: float = 1e-3,
    maximum_absolute_scale_standard_deviation_fraction: float = 0.02,
    minimum_independent_calibration_groups: int = 2,
) -> dict[str, object]:
    """Validate metric depth and SE(3) poses without upgrading learned scale to calibration."""

    contract.validate_metadata()
    frames = np.asarray(frame_indices, dtype=np.int64)
    intrinsics = np.asarray(intrinsics_px, dtype=np.float64)
    poses = np.asarray(world_from_camera, dtype=np.float64)
    if frames.ndim != 1 or len(frames) == 0 or bool(np.any(np.diff(frames) <= 0)):
        raise ValueError("camera frame indices must be a non-empty increasing vector")
    count = len(frames)
    if intrinsics.shape == (3, 3):
        intrinsics = np.repeat(intrinsics[None, :, :], count, axis=0)
    if intrinsics.shape != (count, 3, 3):
        raise ValueError("camera intrinsics must have shape 3x3 or Tx3x3")
    if poses.shape != (count, 4, 4):
        raise ValueError("world-from-camera poses must have shape Tx4x4")
    finite = bool(np.all(np.isfinite(intrinsics)) and np.all(np.isfinite(poses)))
    focal_positive = bool(np.all(intrinsics[:, 0, 0] > 0) and np.all(intrinsics[:, 1, 1] > 0))
    principal_points_inside = bool(
        np.all((intrinsics[:, 0, 2] >= 0) & (intrinsics[:, 0, 2] < contract.image_width))
        and np.all((intrinsics[:, 1, 2] >= 0) & (intrinsics[:, 1, 2] < contract.image_height))
    )
    bottom = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    homogeneous = bool(np.max(np.abs(poses[:, 3, :] - bottom[None, :])) <= 1e-6)
    rotations = poses[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)
    orthogonality_error = float(
        np.max(np.linalg.norm(np.swapaxes(rotations, 1, 2) @ rotations - identity, axis=(1, 2)))
    )
    determinants = np.linalg.det(rotations)
    proper_rotations = bool(
        orthogonality_error <= maximum_rotation_orthogonality_error
        and np.max(np.abs(determinants - 1.0)) <= maximum_rotation_orthogonality_error
    )
    valid_depth_fraction = None
    confidence_aligned = depth_confidence is None
    if depth_m is not None:
        depth = np.asarray(depth_m, dtype=np.float64)
        expected = (count, contract.image_height, contract.image_width)
        if depth.shape != expected:
            raise ValueError(f"metric depth must have shape {expected}")
        valid_depth_fraction = float(np.mean(np.isfinite(depth) & (depth > 0)))
        if depth_confidence is not None:
            confidence = np.asarray(depth_confidence, dtype=np.float64)
            if confidence.shape != depth.shape:
                raise ValueError("depth confidence must align with metric depth")
            confidence_aligned = bool(
                np.all(np.isfinite(confidence)) and np.all(confidence >= 0)
            )
    direct_sensor_scale = contract.depth_evidence is EvidenceClass.SENSOR_MEASUREMENT
    bridged_calibrated_scale = (
        contract.depth_evidence is EvidenceClass.CALIBRATED_GEOMETRY
        and contract.absolute_scale_standard_deviation_fraction is not None
        and contract.absolute_scale_standard_deviation_fraction
        <= maximum_absolute_scale_standard_deviation_fraction
        and contract.independent_calibration_groups
        >= minimum_independent_calibration_groups
        and contract.calibration_report_sha256 is not None
    )
    calibrated_scale = direct_sensor_scale or bridged_calibrated_scale
    bounded_learned_context_scale = (
        contract.depth_evidence is EvidenceClass.FOUNDATION_MODEL_ESTIMATE
        and contract.learned_context_scale_variation_fraction is not None
    )
    gates = {
        "finite_camera_state": finite,
        "positive_focal_lengths": focal_positive,
        "principal_points_inside_image": principal_points_inside,
        "proper_se3_poses": homogeneous and proper_rotations,
        "metric_depth_available": valid_depth_fraction is not None,
        "metric_depth_coverage": (
            valid_depth_fraction is not None
            and valid_depth_fraction >= minimum_valid_depth_fraction
        ),
        "depth_confidence_aligned": confidence_aligned,
        "context_scale_stability_bounded_or_calibrated": (
            calibrated_scale or bounded_learned_context_scale
        ),
        "absolute_metric_scale_calibrated": calibrated_scale,
    }
    proposal_gates = {
        name: value
        for name, value in gates.items()
        if name != "absolute_metric_scale_calibrated"
    }
    return {
        "frames": count,
        "valid_depth_fraction": valid_depth_fraction,
        "rotation_orthogonality_error": orthogonality_error,
        "tested_context_scale_variation_fraction_p95": (
            contract.learned_context_scale_variation_fraction
        ),
        "absolute_scale_standard_deviation_fraction": (
            contract.absolute_scale_standard_deviation_fraction
        ),
        "independent_calibration_groups": contract.independent_calibration_groups,
        "calibration_report_sha256": contract.calibration_report_sha256,
        "calibrated_scale": calibrated_scale,
        "proposal_passed": all(proposal_gates.values()),
        "proposal_scope": (
            "learned metric geometry with bounded tested context sensitivity; "
            "not an independent absolute-scale calibration"
        ),
        "gates": gates,
        "passed": all(gates.values()),
    }


def validate_robot_trajectory(
    np: Any,
    *,
    contract: RobotTrajectoryContract,
    frame_indices: Any,
    joint_positions_rad: Any,
    joint_velocities_rad_s: Any | None = None,
    reprojection_rmse_px: Any | None = None,
    maximum_joint_velocity_rad_s: float = 12.0,
    maximum_reprojection_rmse_px: float = 8.0,
) -> dict[str, object]:
    """Reject partial targets and validate a complete URDF-coordinate trajectory."""

    contract.validate_metadata()
    frames = np.asarray(frame_indices, dtype=np.int64)
    positions = np.asarray(joint_positions_rad, dtype=np.float64)
    expected = (len(frames), len(contract.joint_names))
    if frames.ndim != 1 or len(frames) < 2 or bool(np.any(np.diff(frames) <= 0)):
        raise ValueError("robot frame indices must be increasing and contain two frames")
    if positions.shape != expected:
        raise ValueError(f"joint positions must have shape {expected}")
    finite = bool(np.all(np.isfinite(positions)))
    lower = np.asarray([limit[0] for limit in contract.joint_limits_rad])[None, :]
    upper = np.asarray([limit[1] for limit in contract.joint_limits_rad])[None, :]
    limit_violations = int(np.count_nonzero((positions < lower) | (positions > upper)))
    velocity = None
    if joint_velocities_rad_s is not None:
        velocity = np.asarray(joint_velocities_rad_s, dtype=np.float64)
        if velocity.shape != expected:
            raise ValueError("joint velocities must align with joint positions")
    else:
        dt = np.diff(frames).astype(np.float64) / contract.fps
        velocity = np.zeros_like(positions)
        velocity[1:] = np.diff(positions, axis=0) / dt[:, None]
        velocity[0] = velocity[1]
    maximum_velocity = float(np.max(np.abs(velocity)))
    reprojection = None
    if reprojection_rmse_px is not None:
        reprojection_values = np.asarray(reprojection_rmse_px, dtype=np.float64)
        if reprojection_values.shape != (len(frames),):
            raise ValueError("reprojection RMSE must have one value per robot frame")
        reprojection = float(np.percentile(reprojection_values, 95))
    gates = {
        "finite_full_q_sequence": finite,
        "joint_limits_respected": limit_violations == 0,
        "joint_velocity_bounded": maximum_velocity <= maximum_joint_velocity_rad_s,
        "render_reprojection_validated": (
            reprojection is not None and reprojection <= maximum_reprojection_rmse_px
        ),
        "exact_asset_bound": bool(contract.asset_sha256),
    }
    return {
        "frames": len(frames),
        "joints": len(contract.joint_names),
        "joint_limit_violations": limit_violations,
        "maximum_joint_velocity_rad_s": maximum_velocity,
        "reprojection_rmse_px_p95": reprojection,
        "gates": gates,
        "passed": all(gates.values()),
    }


def validate_stem_centerlines(
    np: Any,
    *,
    contract: StemCenterlineContract,
    frame_indices: Any,
    centerlines_m: Any,
    confidence: Any,
    minimum_visible_fraction: float = 0.80,
    maximum_segment_length_cv: float = 0.12,
) -> dict[str, object]:
    """Validate TxSxNx3 metric centerlines with persistent stem identities."""

    contract.validate_metadata()
    frames = np.asarray(frame_indices, dtype=np.int64)
    centers = np.asarray(centerlines_m, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    expected = (len(frames), len(contract.instance_ids), contract.nodes_per_stem, 3)
    if centers.shape != expected:
        raise ValueError(f"stem centerlines must have shape {expected}")
    if conf.shape != expected[:-1]:
        raise ValueError("stem confidence must have shape TxSxN")
    visible = np.isfinite(centers).all(axis=-1) & np.isfinite(conf) & (conf > 0)
    visible_fraction = np.mean(visible, axis=(0, 2))
    segment_cvs = []
    for stem_index in range(len(contract.instance_ids)):
        segments = np.linalg.norm(
            centers[:, stem_index, 1:] - centers[:, stem_index, :-1], axis=-1
        )
        segment_visible = visible[:, stem_index, 1:] & visible[:, stem_index, :-1]
        for segment_index in range(contract.nodes_per_stem - 1):
            values = segments[:, segment_index][segment_visible[:, segment_index]]
            if len(values) < 2 or float(np.mean(values)) <= 1e-8:
                segment_cvs.append(float("inf"))
            else:
                segment_cvs.append(float(np.std(values) / np.mean(values)))
    maximum_cv = max(segment_cvs, default=float("inf"))
    gates = {
        "persistent_instance_ids": len(set(contract.instance_ids)) == len(contract.instance_ids),
        "finite_confidence": bool(np.all(np.isfinite(conf)) and np.all(conf >= 0)),
        "visible_coverage": bool(np.all(visible_fraction >= minimum_visible_fraction)),
        "segment_lengths_temporally_rigid": maximum_cv <= maximum_segment_length_cv,
        "metric_not_pixel_geometry": contract.geometry_evidence is not EvidenceClass.HEURISTIC_PROXY,
    }
    return {
        "frames": len(frames),
        "stems": len(contract.instance_ids),
        "nodes_per_stem": contract.nodes_per_stem,
        "visible_fraction_by_stem": [float(value) for value in visible_fraction],
        "maximum_segment_length_cv": maximum_cv,
        "gates": gates,
        "passed": all(gates.values()),
    }


def validate_contact_force_sequence(
    np: Any,
    *,
    contract: ContactForceContract,
    forces_n: Any,
    solver_residual_n: Any,
    covariance_n2: Any | None,
    maximum_solver_residual_n: float = 0.08,
) -> dict[str, object]:
    """Validate force provenance, units, uncertainty, and dynamics residual."""

    contract.validate_metadata()
    forces = np.asarray(forces_n, dtype=np.float64)
    residual = np.asarray(solver_residual_n, dtype=np.float64)
    if forces.ndim != 4 or forces.shape[1] != len(contract.instance_ids) or forces.shape[-1] != 3:
        raise ValueError("contact forces must have shape TxSxCx3")
    if residual.shape != forces.shape[:2]:
        raise ValueError("solver residual must have shape TxS")
    covariance_valid = covariance_n2 is not None
    if covariance_n2 is not None:
        covariance = np.asarray(covariance_n2, dtype=np.float64)
        if covariance.shape != forces.shape + (3,):
            raise ValueError("force covariance must have shape TxSxCx3x3")
        symmetric = np.max(np.abs(covariance - np.swapaxes(covariance, -1, -2))) <= 1e-8
        eigenvalues = np.linalg.eigvalsh(covariance)
        covariance_valid = bool(symmetric and np.all(eigenvalues >= -1e-10))
    residual_p95 = float(np.percentile(residual, 95))
    gates = {
        "finite_force_state": bool(np.all(np.isfinite(forces)) and np.all(np.isfinite(residual))),
        "physical_force_source": contract.force_evidence in {
            EvidenceClass.SENSOR_MEASUREMENT,
            EvidenceClass.PHYSICS_SOLVER_ESTIMATE,
        },
        "dynamics_residual_bounded": residual_p95 <= maximum_solver_residual_n,
        "uncertainty_propagated": covariance_valid,
    }
    return {
        "frames": forces.shape[0],
        "stems": forces.shape[1],
        "contacts_per_stem": forces.shape[2],
        "solver_residual_n_p95": residual_p95,
        "gates": gates,
        "passed": all(gates.values()),
    }


def decide_foundation_contact_status(
    stage_reports: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, object]:
    """Compute an honest end-to-end status from required physical stages."""

    required = ("metric_camera", "robot_trajectory", "stem_centerlines", "contact_forces")
    gates = {
        name: bool(stage_reports.get(name) and stage_reports[name].get("passed"))
        for name in required
    }
    missing = [name for name, passed in gates.items() if not passed]
    return {
        "status": "WORKING" if not missing else "PARTIAL",
        "gates": gates,
        "missing_or_rejected_stages": missing,
        "rule": "WORKING requires every metric, kinematic, deformable, and force gate",
    }


def model_registry() -> tuple[ModelProvenance, ...]:
    """Pinned proposal models selected for this pipeline revision."""

    return (
        ModelProvenance(
            name="DA3NESTED-GIANT-LARGE-1.1",
            revision="41736238f5bced4debf3f2a12375d2466874866d",
            repository="https://github.com/ByteDance-Seed/Depth-Anything-3",
            license="CC-BY-NC-4.0",
            purpose="primary learned-metric depth, intrinsics, and long-video camera proposal",
        ),
        ModelProvenance(
            name="UniDepthV2-ViT-S14",
            revision="8d8cfe4c7ee15297099983607febf0d4f32eb3d6",
            repository="https://github.com/lpiccinelli-eth/UniDepth",
            license="CC-BY-NC-4.0",
            purpose="single-frame metric depth, points, intrinsics, and confidence proposal",
        ),
        ModelProvenance(
            name="V-DPM",
            revision="5e2a57cf6007dfb0511a8b396a0805089b9edcc4",
            repository="https://github.com/eldar/vdpm",
            license="MIT code; VGGT model license for inherited weights",
            purpose="rolling-window camera pose and dynamic point-map proposal",
        ),
        ModelProvenance(
            name="HaMeR",
            revision="3a01849f4148352e9260b69bf28b65d1671a4905",
            repository="https://github.com/geopavlakos/hamer",
            license="research-only dependencies include separately licensed MANO",
            purpose="source human hand articulation proposal before robot retargeting",
        ),
        ModelProvenance(
            name="PhysTwin",
            revision="54106c6357e369955bc21ea77f012fbd5867165c",
            repository="https://github.com/Jianghanxiao/PhysTwin",
            license="MIT",
            purpose="RGB-D deformable inverse-physics reference; never a monocular force oracle",
        ),
    )
