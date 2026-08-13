"""Fail-closed 4-D hand/contact and deformable-stem contracts.

The functions in this module receive a NumPy-like module from their caller so
that importing :mod:`phiagent` remains lightweight.  The central rule is that
image-space adjacency is never promoted to metric contact.  Metric contact
requires named 3-D frames, calibrated depth, explicit articulated geometry,
and force evidence.  A deterministic damped-rod simulator is included as a
physics backbone; a learned model may predict residuals, but may not replace
the conservation and evidence gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class InteractionFrameContract:
    """Coordinate and sampling contract for one reconstructed interaction."""

    camera_frame: str
    metric_frame: str
    timeline: str
    fps: float
    fx_pixels: float | None = None
    fy_pixels: float | None = None
    cx_pixels: float | None = None
    cy_pixels: float | None = None
    metric_scale_source: str | None = None

    @property
    def has_metric_camera(self) -> bool:
        values = (self.fx_pixels, self.fy_pixels, self.cx_pixels, self.cy_pixels)
        return (
            all(value is not None for value in values)
            and self.metric_scale_source is not None
            and bool(self.metric_scale_source.strip())
        )

    def validate(self) -> None:
        for label, value in (
            ("camera_frame", self.camera_frame),
            ("metric_frame", self.metric_frame),
            ("timeline", self.timeline),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be named")
        if self.fps <= 0:
            raise ValueError("interaction FPS must be positive")
        intrinsics = (self.fx_pixels, self.fy_pixels, self.cx_pixels, self.cy_pixels)
        supplied = [value is not None for value in intrinsics]
        if any(supplied) and not all(supplied):
            raise ValueError("camera intrinsics must be supplied together")
        if all(supplied) and (self.fx_pixels <= 0 or self.fy_pixels <= 0):
            raise ValueError("camera focal lengths must be positive")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "camera_frame": self.camera_frame,
            "metric_frame": self.metric_frame,
            "timeline": self.timeline,
            "fps": self.fps,
            "intrinsics_pixels": {
                "fx": self.fx_pixels,
                "fy": self.fy_pixels,
                "cx": self.cx_pixels,
                "cy": self.cy_pixels,
            },
            "metric_scale_source": self.metric_scale_source,
            "has_metric_camera": self.has_metric_camera,
        }


@dataclass(frozen=True)
class ArticulatedHandContract:
    """Immutable joint tree and limits for one robot hand embodiment."""

    embodiment_id: str
    coordinate_frame: str
    joint_names: tuple[str, ...]
    parent_indices: tuple[int, ...]
    joint_limits_rad: tuple[tuple[float, float], ...]
    fingertip_indices: tuple[int, ...]
    palm_index: int

    def validate(self) -> None:
        if not self.embodiment_id.strip() or not self.coordinate_frame.strip():
            raise ValueError("hand embodiment and coordinate frame must be named")
        count = len(self.joint_names)
        if count < 3:
            raise ValueError("articulated hand must contain at least three joints")
        if len(set(self.joint_names)) != count:
            raise ValueError("hand joint names must be unique")
        if len(self.parent_indices) != count or len(self.joint_limits_rad) != count:
            raise ValueError("joint tree and limits must match joint names")
        roots = [index for index, parent in enumerate(self.parent_indices) if parent == -1]
        if roots != [0]:
            raise ValueError("hand must have exactly one root at joint zero")
        for index, parent in enumerate(self.parent_indices[1:], 1):
            if parent < 0 or parent >= index:
                raise ValueError("joint parents must precede their child")
        for lower, upper in self.joint_limits_rad:
            if lower > upper:
                raise ValueError("joint lower limit exceeds upper limit")
        if self.palm_index < 0 or self.palm_index >= count:
            raise ValueError("palm index is outside the hand tree")
        if (
            len(self.fingertip_indices) < 2
            or len(set(self.fingertip_indices)) != len(self.fingertip_indices)
            or any(index <= 0 or index >= count for index in self.fingertip_indices)
        ):
            raise ValueError("at least two distinct fingertip indices are required")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "embodiment_id": self.embodiment_id,
            "coordinate_frame": self.coordinate_frame,
            "joint_names": list(self.joint_names),
            "parent_indices": list(self.parent_indices),
            "joint_limits_rad": [list(value) for value in self.joint_limits_rad],
            "fingertip_indices": list(self.fingertip_indices),
            "palm_index": self.palm_index,
        }


@dataclass(frozen=True)
class StemRodContract:
    """Physical backbone for one flower represented as a deformable rod."""

    instance_id: str
    coordinate_frame: str
    node_count: int
    root_node: int
    linear_density_kg_m: float
    axial_stiffness_n_m: float
    bending_stiffness_n_m: float
    damping_n_s_m: float

    def validate(self) -> None:
        if not self.instance_id.strip() or not self.coordinate_frame.strip():
            raise ValueError("stem identity and coordinate frame must be named")
        if self.node_count < 3:
            raise ValueError("deformable stem requires at least three nodes")
        if self.root_node not in (0, self.node_count - 1):
            raise ValueError("stem root must be one endpoint")
        values = (
            self.linear_density_kg_m,
            self.axial_stiffness_n_m,
            self.bending_stiffness_n_m,
            self.damping_n_s_m,
        )
        if any(value <= 0 for value in values):
            raise ValueError("stem material parameters must be positive")


@dataclass(frozen=True)
class MetricContactContract:
    """Evidence thresholds with physical units, not pixel distances."""

    maximum_surface_gap_m: float = 0.003
    minimum_normal_force_n: float = 0.02
    friction_coefficient: float = 0.35
    maximum_force_balance_residual_n: float = 0.08
    maximum_moment_balance_residual_nm: float = 0.01
    minimum_distinct_fingertips: int = 2
    friction_cone_edges: int = 8
    minimum_grasp_matrix_singular_value: float = 1e-4
    maximum_force_closure_origin_residual: float = 1e-6
    minimum_positive_wrench_weight: float = 1e-8

    def validate(self) -> None:
        if self.maximum_surface_gap_m <= 0:
            raise ValueError("maximum surface gap must be positive")
        if self.minimum_normal_force_n <= 0:
            raise ValueError("minimum normal force must be positive")
        if self.friction_coefficient <= 0:
            raise ValueError("friction coefficient must be positive")
        if self.maximum_force_balance_residual_n <= 0:
            raise ValueError("force residual limit must be positive")
        if self.maximum_moment_balance_residual_nm <= 0:
            raise ValueError("moment residual limit must be positive")
        if self.minimum_distinct_fingertips < 2:
            raise ValueError("force closure requires at least two fingertips")
        if self.friction_cone_edges < 4:
            raise ValueError("friction cone requires at least four edges")
        if self.minimum_grasp_matrix_singular_value <= 0:
            raise ValueError("grasp matrix singular-value floor must be positive")
        if self.maximum_force_closure_origin_residual <= 0:
            raise ValueError("force-closure origin residual limit must be positive")
        if self.minimum_positive_wrench_weight < 0:
            raise ValueError("positive wrench weight floor must be non-negative")


def _project_probability_simplex(np: Any, values: Any) -> Any:
    """Project a finite vector onto non-negative weights that sum to one."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0 or not bool(np.all(np.isfinite(vector))):
        raise ValueError("simplex projection requires one non-empty finite vector")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, len(vector) + 1, dtype=np.float64)
    admissible = ordered - (cumulative - 1.0) / indices > 0.0
    if not bool(np.any(admissible)):
        raise RuntimeError("probability-simplex projection found no active weight")
    rho = int(np.flatnonzero(admissible)[-1])
    threshold = float((cumulative[rho] - 1.0) / (rho + 1))
    return np.maximum(vector - threshold, 0.0)


def _force_closure_certificate(
    np: Any,
    *,
    points: Any,
    unit_normals: Any,
    center: Any,
    contract: MetricContactContract,
) -> dict[str, object]:
    """Compute a conservative linearized 6-D frictional force-closure proof.

    The friction cone at every point contact is discretized into edges.  Full
    row rank plus a strictly positive null-vector is a sufficient certificate
    that the positive span of contact wrenches covers the local 6-D wrench
    space.  Moments are normalized by grasp radius before the SVD so force and
    torque units are not compared directly.
    """

    radial = points - center[None, :]
    radius = max(float(np.max(np.linalg.norm(radial, axis=1))), 1e-6)
    wrenches = []
    for point, normal in zip(points, unit_normals):
        axis = np.zeros(3, dtype=np.float64)
        axis[int(np.argmin(np.abs(normal)))] = 1.0
        tangent_a = np.cross(normal, axis)
        tangent_a /= max(float(np.linalg.norm(tangent_a)), 1e-12)
        tangent_b = np.cross(normal, tangent_a)
        for edge in range(contract.friction_cone_edges):
            angle = 2.0 * np.pi * edge / contract.friction_cone_edges
            force = -normal + contract.friction_coefficient * (
                np.cos(angle) * tangent_a + np.sin(angle) * tangent_b
            )
            force /= max(float(np.linalg.norm(force)), 1e-12)
            moment = np.cross(point - center, force) / radius
            wrenches.append(np.concatenate((force, moment)))
    matrix = np.asarray(wrenches, dtype=np.float64).T
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    minimum_singular = float(singular_values[-1]) if len(singular_values) >= 6 else 0.0
    rank = int(
        np.count_nonzero(singular_values >= contract.minimum_grasp_matrix_singular_value)
    )
    weights = np.full(
        matrix.shape[1],
        1.0 / matrix.shape[1],
        dtype=np.float64,
    )
    accelerated = weights.copy()
    best_weights = weights.copy()
    best_residual = float(np.linalg.norm(matrix @ weights))
    momentum = 1.0
    lipschitz = max(float(singular_values[0] ** 2), 1e-12)
    iterations = 0
    for iterations in range(1, 5001):
        gradient = matrix.T @ (matrix @ accelerated)
        candidate = _project_probability_simplex(
            np,
            accelerated - gradient / lipschitz,
        )
        next_momentum = 0.5 * (1.0 + (1.0 + 4.0 * momentum * momentum) ** 0.5)
        accelerated = candidate + (
            (momentum - 1.0) / next_momentum
        ) * (candidate - weights)
        delta = float(np.linalg.norm(candidate - weights))
        candidate_residual = float(np.linalg.norm(matrix @ candidate))
        if candidate_residual < best_residual:
            best_residual = candidate_residual
            best_weights = candidate.copy()
        weights = candidate
        momentum = next_momentum
        if (
            delta <= 1e-12
            or best_residual
            <= contract.maximum_force_closure_origin_residual * 0.1
        ):
            break
    weights = best_weights
    origin_residual = best_residual
    active = weights > contract.minimum_positive_wrench_weight
    positive_count = int(np.count_nonzero(active))
    positive_rank = (
        int(np.linalg.matrix_rank(matrix[:, active]))
        if positive_count
        else 0
    )
    minimum_weight = (
        float(np.min(weights[active])) if positive_count else 0.0
    )
    positive_origin = (
        origin_residual <= contract.maximum_force_closure_origin_residual
        and positive_count >= 7
        and positive_rank == 6
        and minimum_weight > contract.minimum_positive_wrench_weight
    )
    return {
        "passed": rank == 6 and positive_origin,
        "linearized_grasp_matrix_rank": rank,
        "minimum_grasp_matrix_singular_value": minimum_singular,
        "force_closure_origin_residual": origin_residual,
        "minimum_positive_wrench_weight": minimum_weight,
        "positive_wrench_count": positive_count,
        "positive_wrench_rank": positive_rank,
        "nonnegative_solver_iterations": iterations,
        "nonnegative_solver": "fista_probability_simplex",
        "friction_cone_edges_per_contact": contract.friction_cone_edges,
        "moment_normalization_radius_m": radius,
    }


def validate_kinematic_sequence(
    np: Any,
    *,
    joints_xyz_m: Any,
    joint_angles_rad: Any,
    contract: ArticulatedHandContract,
    maximum_bone_length_cv: float = 0.015,
) -> dict[str, object]:
    """Validate topology, finite metric geometry, bone rigidity, and limits."""

    contract.validate()
    joints = np.asarray(joints_xyz_m, dtype=np.float64)
    angles = np.asarray(joint_angles_rad, dtype=np.float64)
    frames = joints.shape[0] if joints.ndim else 0
    expected_joints = len(contract.joint_names)
    if joints.ndim != 3 or joints.shape[1:] != (expected_joints, 3):
        raise ValueError("joints must have shape TxJx3 matching the hand contract")
    if angles.shape != (frames, expected_joints):
        raise ValueError("joint angles must have shape TxJ matching the hand contract")
    finite = bool(np.all(np.isfinite(joints)) and np.all(np.isfinite(angles)))
    bone_cvs = []
    collapsed_bones = 0
    for child, parent in enumerate(contract.parent_indices):
        if parent < 0:
            continue
        lengths = np.linalg.norm(joints[:, child] - joints[:, parent], axis=1)
        mean = float(np.mean(lengths))
        if mean <= 1e-8:
            collapsed_bones += 1
            bone_cvs.append(float("inf"))
        else:
            bone_cvs.append(float(np.std(lengths) / mean))
    limit_violations = 0
    for joint_index, (lower, upper) in enumerate(contract.joint_limits_rad):
        limit_violations += int(
            np.count_nonzero(
                (angles[:, joint_index] < lower) | (angles[:, joint_index] > upper)
            )
        )
    maximum_cv = max(bone_cvs, default=0.0)
    gates = {
        "finite_metric_geometry": finite,
        "fixed_joint_count": joints.shape[1] == expected_joints,
        "no_collapsed_bones": collapsed_bones == 0,
        "bone_lengths_rigid": maximum_cv <= maximum_bone_length_cv,
        "joint_limits_respected": limit_violations == 0,
    }
    return {
        "frames": frames,
        "joints": expected_joints,
        "maximum_bone_length_cv": maximum_cv,
        "collapsed_bones": collapsed_bones,
        "joint_limit_violations": limit_violations,
        "gates": gates,
        "passed": all(gates.values()),
    }


def assess_metric_force_closure(
    np: Any,
    *,
    contact_points_m: Any | None,
    surface_gaps_m: Any | None,
    contact_normals: Any | None,
    contact_forces_n: Any | None,
    object_center_m: Any | None,
    external_force_n: Any | None,
    external_moment_nm: Any | None,
    fingertip_indices: Iterable[int] | None,
    frame_contract: InteractionFrameContract,
    contact_contract: MetricContactContract,
    depth_source: str | None,
    force_source: str | None,
    occlusion_order_known: bool,
) -> dict[str, object]:
    """Fail closed unless metric geometry and measured/simulated forces exist."""

    frame_contract.validate()
    contact_contract.validate()
    reasons = []
    if not frame_contract.has_metric_camera:
        reasons.append("missing_metric_camera")
    if not depth_source:
        reasons.append("missing_depth_source")
    if not force_source:
        reasons.append("missing_force_source")
    if not occlusion_order_known:
        reasons.append("unknown_occlusion_order")
    if any(
        value is None
        for value in (
            contact_points_m,
            surface_gaps_m,
            contact_normals,
            contact_forces_n,
            object_center_m,
            external_force_n,
            external_moment_nm,
        )
    ):
        reasons.append("missing_contact_geometry_or_forces")
        return {
            "passed": False,
            "reasons": reasons,
            "distinct_fingertips": 0,
            "force_balance_residual_n": None,
            "moment_balance_residual_nm": None,
            "friction_cone_violations": None,
            "maximum_surface_gap_m": None,
            "force_closure": None,
        }
    points = np.asarray(contact_points_m, dtype=np.float64)
    gaps = np.asarray(surface_gaps_m, dtype=np.float64)
    normals = np.asarray(contact_normals, dtype=np.float64)
    forces = np.asarray(contact_forces_n, dtype=np.float64)
    center = np.asarray(object_center_m, dtype=np.float64)
    external_force = np.asarray(external_force_n, dtype=np.float64)
    external_moment = np.asarray(external_moment_nm, dtype=np.float64)
    fingers = tuple(int(value) for value in (fingertip_indices or ()))
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape or forces.shape != points.shape:
        raise ValueError("contact points, normals, and forces must share shape Nx3")
    if (
        gaps.shape != (len(points),)
        or center.shape != (3,)
        or external_force.shape != (3,)
        or external_moment.shape != (3,)
        or len(fingers) != len(points)
    ):
        raise ValueError("object center must be 3-D and each contact must name a fingertip")
    if not (
        np.all(np.isfinite(points))
        and np.all(np.isfinite(normals))
        and np.all(np.isfinite(forces))
        and np.all(np.isfinite(gaps))
        and np.all(np.isfinite(center))
        and np.all(np.isfinite(external_force))
        and np.all(np.isfinite(external_moment))
    ):
        reasons.append("non_finite_contact_state")
    maximum_gap = float(np.max(gaps)) if len(gaps) else float("inf")
    if bool(np.any(gaps < 0)) or maximum_gap > contact_contract.maximum_surface_gap_m:
        reasons.append("surface_gap_violation")
    normal_norm = np.linalg.norm(normals, axis=1)
    valid_normals = normal_norm > 1e-8
    unit_normals = np.zeros_like(normals)
    unit_normals[valid_normals] = normals[valid_normals] / normal_norm[valid_normals, None]
    normal_load = -np.sum(forces * unit_normals, axis=1)
    tangential = forces + normal_load[:, None] * unit_normals
    tangential_load = np.linalg.norm(tangential, axis=1)
    point_friction_violation = (
        (normal_load < -1e-12)
        | (
            tangential_load
            > contact_contract.friction_coefficient
            * np.maximum(normal_load, 0.0)
            + 1e-12
        )
        | ~valid_normals
    )
    finger_normal_loads = {}
    for fingertip_index, load in zip(fingers, normal_load):
        finger_normal_loads[fingertip_index] = (
            finger_normal_loads.get(fingertip_index, 0.0) + float(load)
        )
    underloaded_fingertips = sum(
        load + 1e-12 < contact_contract.minimum_normal_force_n
        for load in finger_normal_loads.values()
    )
    net_force = np.sum(forces, axis=0) + external_force
    net_moment = (
        np.sum(np.cross(points - center[None, :], forces), axis=0) + external_moment
    )
    force_residual = float(np.linalg.norm(net_force))
    moment_residual = float(np.linalg.norm(net_moment))
    distinct = len(set(fingers))
    if distinct < contact_contract.minimum_distinct_fingertips:
        reasons.append("insufficient_distinct_fingertips")
    if bool(np.any(point_friction_violation)) or underloaded_fingertips:
        reasons.append("friction_or_compression_violation")
    if force_residual > contact_contract.maximum_force_balance_residual_n:
        reasons.append("force_balance_violation")
    if moment_residual > contact_contract.maximum_moment_balance_residual_nm:
        reasons.append("moment_balance_violation")
    force_closure = _force_closure_certificate(
        np,
        points=points,
        unit_normals=unit_normals,
        center=center,
        contract=contact_contract,
    )
    if not force_closure["passed"]:
        reasons.append("force_closure_certificate_failed")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "distinct_fingertips": distinct,
        "force_balance_residual_n": force_residual,
        "moment_balance_residual_nm": moment_residual,
        "friction_cone_violations": (
            int(np.count_nonzero(point_friction_violation))
            + underloaded_fingertips
        ),
        "normal_load_by_fingertip_n": {
            str(index): load for index, load in finger_normal_loads.items()
        },
        "maximum_surface_gap_m": maximum_gap,
        "force_closure": force_closure,
    }


def couple_contact_patch_to_required_wrench(
    np: Any,
    *,
    contact_points_m: Any,
    contact_normals: Any,
    fingertip_indices: Iterable[int],
    object_center_m: Any,
    required_force_n: Any,
    required_moment_nm: Any,
    friction_coefficient: float = 0.35,
    minimum_normal_force_n: float = 0.02,
    friction_cone_edges: int = 8,
) -> dict[str, Any]:
    """Solve exact-pad contact forces against one inverse-dynamics rod wrench."""

    points = np.asarray(contact_points_m, dtype=np.float64)
    normals = np.asarray(contact_normals, dtype=np.float64)
    center = np.asarray(object_center_m, dtype=np.float64)
    required_force = np.asarray(required_force_n, dtype=np.float64)
    required_moment = np.asarray(required_moment_nm, dtype=np.float64)
    fingers = tuple(int(value) for value in fingertip_indices)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or normals.shape != points.shape
        or len(fingers) != len(points)
    ):
        raise ValueError("coupled contacts require aligned points, normals, and fingertips")
    if center.shape != (3,) or required_force.shape != (3,) or required_moment.shape != (3,):
        raise ValueError("coupled contact center, force, and moment must be three-vectors")
    if not (
        np.all(np.isfinite(points))
        and np.all(np.isfinite(normals))
        and np.all(np.isfinite(center))
        and np.all(np.isfinite(required_force))
        and np.all(np.isfinite(required_moment))
    ):
        raise ValueError("coupled contact state must be finite")
    if (
        friction_coefficient <= 0
        or minimum_normal_force_n <= 0
        or friction_cone_edges < 4
    ):
        raise ValueError("coupled friction and load parameters must be positive")
    normal_norms = np.linalg.norm(normals, axis=1)
    if bool(np.any(normal_norms <= 1e-10)):
        raise ValueError("coupled contact normals must be nonzero")
    unit_normals = normals / normal_norms[:, None]
    counts = {
        fingertip: fingers.count(fingertip)
        for fingertip in set(fingers)
    }
    contact_forces = np.asarray(
        [
            -minimum_normal_force_n / counts[fingertip] * normal
            for fingertip, normal in zip(fingers, unit_normals)
        ],
        dtype=np.float64,
    )
    radius = max(
        float(np.max(np.linalg.norm(points - center[None, :], axis=1))),
        1e-6,
    )
    edge_forces = []
    edge_contact_indices = []
    wrench_columns = []
    for contact_index, (point, normal) in enumerate(zip(points, unit_normals)):
        axis = np.zeros(3, dtype=np.float64)
        axis[int(np.argmin(np.abs(normal)))] = 1.0
        tangent_a = np.cross(normal, axis)
        tangent_a /= max(float(np.linalg.norm(tangent_a)), 1e-12)
        tangent_b = np.cross(normal, tangent_a)
        for edge in range(friction_cone_edges):
            angle = 2.0 * np.pi * edge / friction_cone_edges
            force = -normal + friction_coefficient * (
                np.cos(angle) * tangent_a
                + np.sin(angle) * tangent_b
            )
            force /= max(float(np.linalg.norm(force)), 1e-12)
            moment = np.cross(point - center, force)
            edge_forces.append(force)
            edge_contact_indices.append(contact_index)
            wrench_columns.append(
                np.concatenate((force, moment / radius))
            )
    matrix = np.asarray(wrench_columns, dtype=np.float64).T
    preload_force = np.sum(contact_forces, axis=0)
    preload_moment = np.sum(
        np.cross(points - center[None, :], contact_forces),
        axis=0,
    )
    target = np.concatenate(
        (
            required_force - preload_force,
            (required_moment - preload_moment) / radius,
        )
    )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    lipschitz = max(float(singular_values[0] ** 2), 1e-12)
    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    accelerated = weights.copy()
    best_weights = weights.copy()
    best_residual = float(np.linalg.norm(target))
    momentum = 1.0
    iterations = 0
    for iterations in range(1, 5001):
        gradient = matrix.T @ (matrix @ accelerated - target)
        candidate = np.maximum(accelerated - gradient / lipschitz, 0.0)
        next_momentum = 0.5 * (1.0 + (1.0 + 4.0 * momentum * momentum) ** 0.5)
        accelerated = candidate + (
            (momentum - 1.0) / next_momentum
        ) * (candidate - weights)
        residual = float(np.linalg.norm(matrix @ candidate - target))
        if residual < best_residual:
            best_residual = residual
            best_weights = candidate.copy()
        delta = float(np.linalg.norm(candidate - weights))
        weights = candidate
        momentum = next_momentum
        if delta <= 1e-12 or best_residual <= 1e-8:
            break
    for weight, force, contact_index in zip(
        best_weights,
        edge_forces,
        edge_contact_indices,
    ):
        contact_forces[contact_index] += weight * force
    achieved_force = np.sum(contact_forces, axis=0)
    achieved_moment = np.sum(
        np.cross(points - center[None, :], contact_forces),
        axis=0,
    )
    force_residual = float(np.linalg.norm(achieved_force - required_force))
    moment_residual = float(np.linalg.norm(achieved_moment - required_moment))
    return {
        "contact_forces_n": contact_forces,
        "external_force_n": -required_force,
        "external_moment_nm": -required_moment,
        "required_force_n": required_force,
        "required_moment_nm": required_moment,
        "achieved_force_n": achieved_force,
        "achieved_moment_nm": achieved_moment,
        "coupled_force_residual_n": force_residual,
        "coupled_moment_residual_nm": moment_residual,
        "nonnegative_solver_residual_normalized": best_residual,
        "nonnegative_solver_iterations": iterations,
        "force_source": "exact-pad-friction-cone-coupled-to-inverse-rod-wrench",
    }


def simulate_damped_stem(
    np: Any,
    *,
    rest_nodes_m: Any,
    contact_targets_m: Any,
    contact_active: Any,
    contact_node: int,
    contract: StemRodContract,
    fps: float,
) -> dict[str, Any]:
    """Roll out a stable spring/rod backbone with an explicit contact actuator.

    The integration substep is derived from mass and stiffness.  It is not a
    visual smoothing knob.  A neural residual can later correct the resulting
    state, but the rooted endpoint and finite-energy checks remain mandatory.
    """

    contract.validate()
    if fps <= 0:
        raise ValueError("simulation FPS must be positive")
    rest = np.asarray(rest_nodes_m, dtype=np.float64)
    targets = np.asarray(contact_targets_m, dtype=np.float64)
    active = np.asarray(contact_active, dtype=bool)
    if rest.ndim != 2 or rest.shape != (contract.node_count, 3):
        raise ValueError("rest stem must have shape Nx3 matching the rod contract")
    if targets.ndim != 2 or targets.shape[1] != 3 or active.shape != (len(targets),):
        raise ValueError("contact target and activity arrays must have shapes Tx3 and T")
    if contact_node == contract.root_node or not 0 <= contact_node < contract.node_count:
        raise ValueError("contact node must be a non-root stem node")
    segment_rest = np.linalg.norm(rest[1:] - rest[:-1], axis=1)
    if np.any(segment_rest <= 1e-8):
        raise ValueError("rest stem contains a collapsed segment")
    mean_segment = float(np.mean(segment_rest))
    mass = max(1e-8, contract.linear_density_kg_m * mean_segment)
    dt = 1.0 / fps
    critical = (mass / max(contract.axial_stiffness_n_m, contract.bending_stiffness_n_m)) ** 0.5
    substeps = max(1, int(np.ceil(dt / max(1e-6, 0.2 * critical))))
    h = dt / substeps
    positions = rest.copy()
    velocities = np.zeros_like(rest)
    trajectory = []
    contact_forces = []
    energy = []
    for target, is_active in zip(targets, active):
        frame_contact_force = np.zeros(3, dtype=np.float64)
        for _ in range(substeps):
            force = -contract.damping_n_s_m * velocities
            for left in range(contract.node_count - 1):
                right = left + 1
                delta = positions[right] - positions[left]
                length = float(np.linalg.norm(delta))
                if length <= 1e-10:
                    continue
                direction = delta / length
                spring = contract.axial_stiffness_n_m * (length - segment_rest[left]) * direction
                force[left] += spring
                force[right] -= spring
            for index in range(1, contract.node_count - 1):
                current_curvature = positions[index - 1] - 2.0 * positions[index] + positions[index + 1]
                rest_curvature = rest[index - 1] - 2.0 * rest[index] + rest[index + 1]
                bend = contract.bending_stiffness_n_m * (rest_curvature - current_curvature)
                force[index] += bend
                force[index - 1] -= 0.5 * bend
                force[index + 1] -= 0.5 * bend
            if is_active:
                actuator = contract.axial_stiffness_n_m * 0.5 * (target - positions[contact_node])
                force[contact_node] += actuator
                frame_contact_force = -actuator
            force[contract.root_node] = 0.0
            velocities += h * force / mass
            positions += h * velocities
            positions[contract.root_node] = rest[contract.root_node]
            velocities[contract.root_node] = 0.0
        trajectory.append(positions.copy())
        contact_forces.append(frame_contact_force.copy())
        kinetic = 0.5 * mass * float(np.sum(velocities**2))
        axial = 0.0
        for left in range(contract.node_count - 1):
            length = float(np.linalg.norm(positions[left + 1] - positions[left]))
            axial += 0.5 * contract.axial_stiffness_n_m * (length - segment_rest[left]) ** 2
        energy.append(kinetic + axial)
    trajectory_value = np.asarray(trajectory)
    finite = bool(np.all(np.isfinite(trajectory_value)) and np.all(np.isfinite(energy)))
    root_error = float(
        np.max(np.linalg.norm(trajectory_value[:, contract.root_node] - rest[contract.root_node], axis=1))
    )
    return {
        "nodes_m": trajectory_value,
        "contact_forces_n": np.asarray(contact_forces),
        "energy_j": np.asarray(energy),
        "integration_substeps": substeps,
        "finite": finite,
        "maximum_root_error_m": root_error,
        "passed": finite and root_error <= 1e-9,
    }


def infer_stem_contact_forces(
    np: Any,
    *,
    nodes_m: Any,
    position_sigma_m: Any,
    contact_nodes: Any,
    contact_active: Any,
    contract: StemRodContract,
    fps: float,
    gravity_m_s2: Any = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Invert a metric rod trajectory into contact forces and uncertainty.

    This is a dynamics residual, not a visual force predictor.  The observed
    acceleration is compared with axial, bending, damping, and gravity terms.
    The remaining nodal wrench is assigned only to explicitly active contact
    nodes; unexplained force on every other movable node remains in the solver
    residual and can therefore reject a visually plausible but non-physical
    reconstruction.
    """

    contract.validate()
    if fps <= 0:
        raise ValueError("inverse dynamics FPS must be positive")
    nodes = np.asarray(nodes_m, dtype=np.float64)
    sigma = np.asarray(position_sigma_m, dtype=np.float64)
    contacts = np.asarray(contact_nodes, dtype=np.int64)
    active = np.asarray(contact_active, dtype=bool)
    gravity = np.asarray(gravity_m_s2, dtype=np.float64)
    if nodes.ndim != 3 or nodes.shape[1:] != (contract.node_count, 3):
        raise ValueError("observed stem nodes must have shape TxNx3")
    frames = nodes.shape[0]
    if frames < 3:
        raise ValueError("inverse rod dynamics requires at least three frames")
    if sigma.shape != nodes.shape[:2]:
        raise ValueError("position uncertainty must have shape TxN")
    if contacts.ndim == 1:
        contacts = contacts[:, None]
    if active.ndim == 1:
        active = active[:, None]
    if contacts.shape != active.shape or contacts.shape[0] != frames:
        raise ValueError("contact nodes and activity must align as TxC")
    if gravity.shape != (3,):
        raise ValueError("gravity must be a 3-D vector")
    if not (
        np.all(np.isfinite(nodes))
        and np.all(np.isfinite(sigma))
        and np.all(sigma >= 0)
        and np.all(np.isfinite(gravity))
    ):
        raise ValueError("inverse rod state must be finite with non-negative uncertainty")
    valid_contact_indices = contacts[active]
    if bool(
        np.any(valid_contact_indices < 0)
        or np.any(valid_contact_indices >= contract.node_count)
        or np.any(valid_contact_indices == contract.root_node)
    ):
        raise ValueError("active contacts must name non-root stem nodes")
    for frame_contacts, frame_active in zip(contacts, active):
        selected = frame_contacts[frame_active]
        if len(selected) != len(set(int(value) for value in selected)):
            raise ValueError("one frame cannot assign two contacts to the same stem node")

    dt = 1.0 / fps
    velocity = np.zeros_like(nodes)
    velocity[1:-1] = (nodes[2:] - nodes[:-2]) / (2.0 * dt)
    velocity[0] = (nodes[1] - nodes[0]) / dt
    velocity[-1] = (nodes[-1] - nodes[-2]) / dt
    acceleration = np.zeros_like(nodes)
    acceleration[1:-1] = (nodes[2:] - 2.0 * nodes[1:-1] + nodes[:-2]) / (dt * dt)
    acceleration[0] = acceleration[1]
    acceleration[-1] = acceleration[-2]

    rest = nodes[0].copy()
    segment_rest = np.linalg.norm(rest[1:] - rest[:-1], axis=1)
    if bool(np.any(segment_rest <= 1e-8)):
        raise ValueError("initial stem centerline contains a collapsed segment")
    nodal_length = np.zeros(contract.node_count, dtype=np.float64)
    nodal_length[:-1] += 0.5 * segment_rest
    nodal_length[1:] += 0.5 * segment_rest
    mass = contract.linear_density_kg_m * nodal_length
    passive = np.zeros_like(nodes)
    passive += mass[None, :, None] * gravity[None, None, :]
    passive -= contract.damping_n_s_m * velocity
    for frame_index in range(frames):
        position = nodes[frame_index]
        for left in range(contract.node_count - 1):
            right = left + 1
            delta = position[right] - position[left]
            length = float(np.linalg.norm(delta))
            if length <= 1e-10:
                continue
            direction = delta / length
            spring = (
                contract.axial_stiffness_n_m
                * (length - segment_rest[left])
                * direction
            )
            passive[frame_index, left] += spring
            passive[frame_index, right] -= spring
        for index in range(1, contract.node_count - 1):
            current_curvature = position[index - 1] - 2.0 * position[index] + position[index + 1]
            rest_curvature = rest[index - 1] - 2.0 * rest[index] + rest[index + 1]
            bend = contract.bending_stiffness_n_m * (rest_curvature - current_curvature)
            passive[frame_index, index] += bend
            passive[frame_index, index - 1] -= 0.5 * bend
            passive[frame_index, index + 1] -= 0.5 * bend
    required_external = mass[None, :, None] * acceleration - passive
    required_external[:, contract.root_node] = 0.0

    contact_forces = np.zeros((*contacts.shape, 3), dtype=np.float64)
    covariance = np.zeros((*contacts.shape, 3, 3), dtype=np.float64)
    residual = np.zeros(frames, dtype=np.float64)
    force_sensitivity = (
        mass[None, :] * (fps**2) * (6.0**0.5)
        + 2.0 * contract.axial_stiffness_n_m
        + 6.0 * contract.bending_stiffness_n_m
        + contract.damping_n_s_m * fps * (2.0**0.5)
    )
    for frame_index in range(frames):
        unexplained = required_external[frame_index].copy()
        for slot, (node, is_active) in enumerate(
            zip(contacts[frame_index], active[frame_index])
        ):
            if not is_active:
                continue
            node_index = int(node)
            contact_forces[frame_index, slot] = required_external[frame_index, node_index]
            force_sigma = float(force_sensitivity[0, node_index] * sigma[frame_index, node_index])
            covariance[frame_index, slot] = np.eye(3, dtype=np.float64) * force_sigma**2
            unexplained[node_index] = 0.0
        unexplained[contract.root_node] = 0.0
        residual[frame_index] = float(np.linalg.norm(unexplained))
    finite = bool(
        np.all(np.isfinite(contact_forces))
        and np.all(np.isfinite(covariance))
        and np.all(np.isfinite(residual))
    )
    return {
        "hand_on_stem_forces_n": contact_forces,
        "force_covariance_n2": covariance,
        "unexplained_force_residual_n": residual,
        "required_external_force_by_node_n": required_external,
        "velocity_m_s": velocity,
        "acceleration_m_s2": acceleration,
        "finite": finite,
        "semantics": "physics_solver_estimate; force exerted by hand on stem",
    }


def causal_motion_audit(
    np: Any,
    *,
    grasp_active: Any,
    hand_speed: Any,
    stem_speed: Any,
    hand_motion_floor: float,
    stem_motion_floor: float,
    maximum_response_lag_frames: int,
    maximum_frozen_run_frames: int,
) -> dict[str, object]:
    """Detect a grasped stem that stays frozen while its driving hand moves."""

    grasp = np.asarray(grasp_active, dtype=bool)
    hand = np.asarray(hand_speed, dtype=np.float64)
    stem = np.asarray(stem_speed, dtype=np.float64)
    if grasp.ndim != 1 or hand.shape != grasp.shape or stem.shape != grasp.shape:
        raise ValueError("grasp, hand speed, and stem speed must be aligned vectors")
    if hand_motion_floor < 0 or stem_motion_floor < 0:
        raise ValueError("motion floors must be non-negative")
    if maximum_response_lag_frames < 0 or maximum_frozen_run_frames < 0:
        raise ValueError("lag and run limits must be non-negative")
    active_driver = grasp & (hand > hand_motion_floor)
    responded = np.zeros_like(grasp)
    latencies = []
    for index in np.flatnonzero(active_driver):
        end = min(len(stem), int(index) + maximum_response_lag_frames + 1)
        candidates = np.flatnonzero(stem[int(index) : end] > stem_motion_floor)
        if len(candidates):
            latency = int(candidates[0])
            responded[int(index)] = True
            latencies.append(latency)
    frozen = active_driver & ~responded
    runs = []
    start = None
    for index, value in enumerate(frozen.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    longest = max((end - begin + 1 for begin, end in runs), default=0)
    gates = {
        "has_grasped_hand_motion": bool(np.any(active_driver)),
        "all_driven_frames_respond": not bool(np.any(frozen)),
        "maximum_frozen_run": longest <= maximum_frozen_run_frames,
    }
    return {
        "driver_frames": int(np.count_nonzero(active_driver)),
        "responded_frames": int(np.count_nonzero(responded)),
        "frozen_driver_frames": int(np.count_nonzero(frozen)),
        "frozen_runs": [
            {"start_frame": begin, "end_frame": end, "frames": end - begin + 1}
            for begin, end in runs
        ],
        "maximum_frozen_run_frames": longest,
        "response_latency_frames_median": (
            float(np.median(latencies)) if latencies else None
        ),
        "gates": gates,
        "passed": all(gates.values()),
    }
