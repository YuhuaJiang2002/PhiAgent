"""Metric schedules and geometry for a calibrated flower-manipulation simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricFlowerSimulationContract:
    """Immutable timeline and geometry contract for one long simulated rollout."""

    frames: int = 660
    fps: float = 24.0
    nodes_per_stem: int = 12
    contact_node: int = 7
    approach_end_frame: int = 180
    release_frame: int = 480
    right_pad_offset_robot_base_m: tuple[float, float, float] = (
        0.08053,
        0.00033,
        -0.06185,
    )

    def validate(self) -> None:
        if self.frames < 9:
            raise ValueError("flower simulation requires at least nine frames")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("flower simulation FPS must be finite and positive")
        if self.nodes_per_stem < 3:
            raise ValueError("flower stem requires at least three nodes")
        if not 0 < self.contact_node < self.nodes_per_stem:
            raise ValueError("contact node must be a non-root stem node")
        if not 2 <= self.approach_end_frame < self.release_frame < self.frames - 2:
            raise ValueError("contact interval must be strictly inside the rollout")
        if (
            len(self.right_pad_offset_robot_base_m) != 3
            or any(
                not math.isfinite(value)
                for value in self.right_pad_offset_robot_base_m
            )
        ):
            raise ValueError("right pad offset must be a finite robot-base three-vector")


def smoothstep(np: Any, value: Any) -> Any:
    """Return a clipped cubic smoothstep for scalar or array inputs."""

    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def build_metric_flower_schedule(
    np: Any,
    *,
    rest_nodes_m: Any,
    contract: MetricFlowerSimulationContract,
) -> dict[str, Any]:
    """Build contact, wrist, and grasp schedules in the robot-base metric frame."""

    contract.validate()
    rest = np.asarray(rest_nodes_m, dtype=np.float64)
    if rest.shape != (contract.nodes_per_stem, 3):
        raise ValueError("rest stem must match the simulation node contract")
    contact_origin = rest[contract.contact_node]
    frame_indices = np.arange(contract.frames, dtype=np.int64)
    contact_active = (
        (frame_indices >= contract.approach_end_frame)
        & (frame_indices < contract.release_frame)
    )
    targets = np.repeat(contact_origin[None, :], contract.frames, axis=0)

    approach = smoothstep(
        np,
        frame_indices[: contract.approach_end_frame]
        / max(1, contract.approach_end_frame - 1),
    )
    approach_offset = np.asarray((-0.035, -0.11, -0.045), dtype=np.float64)
    targets[: contract.approach_end_frame] += (
        (1.0 - approach)[:, None] * approach_offset[None, :]
    )

    contact_count = contract.release_frame - contract.approach_end_frame
    phase = np.linspace(0.0, 1.0, contact_count, endpoint=False)
    envelope = np.sin(np.pi * phase) ** 2
    motion_scale = min(1.0, contact_count / (5.0 * contract.fps))
    targets[contract.approach_end_frame : contract.release_frame, 0] += (
        motion_scale * 0.045 * envelope * np.sin(2.0 * np.pi * phase)
    )
    targets[contract.approach_end_frame : contract.release_frame, 1] += (
        motion_scale * 0.025 * envelope * np.sin(4.0 * np.pi * phase)
    )
    targets[contract.approach_end_frame : contract.release_frame, 2] += (
        motion_scale * 0.035 * envelope
    )

    retract_count = contract.frames - contract.release_frame
    retract = smoothstep(np, np.arange(retract_count) / max(1, retract_count - 1))
    retract_offset = np.asarray((-0.025, -0.13, 0.065), dtype=np.float64)
    targets[contract.release_frame :] += retract[:, None] * retract_offset[None, :]

    left = np.repeat(
        np.asarray((0.30, 0.18, 0.88), dtype=np.float64)[None, :],
        contract.frames,
        axis=0,
    )
    timeline = frame_indices / contract.fps
    left[:, 0] += 0.008 * np.sin(2.0 * np.pi * timeline / 9.0)
    left[:, 2] += 0.006 * np.sin(2.0 * np.pi * timeline / 7.0)

    closure = np.zeros(contract.frames, dtype=np.float64)
    close_frames = max(
        4,
        min(
            max(8, round(contract.fps * 0.5)),
            contract.approach_end_frame // 2,
        ),
    )
    close_start = contract.approach_end_frame - close_frames
    close = smoothstep(
        np,
        np.arange(contract.approach_end_frame - close_start)
        / max(1, contract.approach_end_frame - close_start - 1),
    )
    closure[close_start : contract.approach_end_frame] = close
    closure[contract.approach_end_frame : contract.release_frame] = 1.0
    open_end = min(
        contract.frames - 2,
        contract.release_frame + max(8, round(contract.fps * 0.5)),
    )
    opening = smoothstep(
        np,
        np.arange(open_end - contract.release_frame)
        / max(1, open_end - contract.release_frame - 1),
    )
    closure[contract.release_frame : open_end] = 1.0 - opening

    phases = np.full(contract.frames, "idle", dtype="<U10")
    phases[: contract.approach_end_frame] = "approach"
    phases[contract.approach_end_frame : contract.release_frame] = "manipulate"
    phases[contract.release_frame :] = "retract"
    phases[close_start : contract.approach_end_frame] = "grasp"
    phases[contract.release_frame:open_end] = "release"
    return {
        "frame_indices": frame_indices,
        "phases": phases,
        "contact_active": contact_active,
        "contact_targets_m": targets,
        "left_wrist_targets_m": left,
        "right_wrist_targets_m": targets
        - np.asarray(
            contract.right_pad_offset_robot_base_m,
            dtype=np.float64,
        )[None, :],
        "right_hand_closure": closure,
        "right_pad_offset_robot_base_m": np.asarray(
            contract.right_pad_offset_robot_base_m,
            dtype=np.float64,
        ),
    }


def articulated_hand_points(np: Any, closure: float) -> Any:
    """Create 21 finite landmarks whose bends close monotonically around a stem."""

    amount = float(np.clip(closure, 0.0, 1.0))
    points = np.zeros((21, 3), dtype=np.float64)
    finger_indices = (
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    )
    for finger, indices in enumerate(finger_indices):
        lateral = (finger - 2) * 0.018
        base = np.asarray((lateral, 0.020, 0.003 * finger), dtype=np.float64)
        points[indices[0]] = base
        current = base
        heading = 0.0
        for segment, index in enumerate(indices[1:], 1):
            heading += amount * (0.48 + 0.08 * segment)
            step = np.asarray(
                (
                    0.0,
                    0.028 * math.cos(heading),
                    -0.028 * math.sin(heading),
                ),
                dtype=np.float64,
            )
            current = current + step
            points[index] = current
    return points


def camera_calibration_from_mujoco_scene(
    np: Any,
    *,
    scene_camera: Any,
    width: int,
    height: int,
    vertical_fov_degrees: float,
) -> tuple[Any, Any]:
    """Recover pinhole intrinsics and world-from-camera from a MuJoCo scene camera."""

    if width <= 0 or height <= 0:
        raise ValueError("camera dimensions must be positive")
    if not 0.0 < vertical_fov_degrees < 180.0:
        raise ValueError("vertical field of view must be in (0, 180)")
    positions = np.asarray([camera.pos for camera in scene_camera], dtype=np.float64)
    forwards = np.asarray([camera.forward for camera in scene_camera], dtype=np.float64)
    ups = np.asarray([camera.up for camera in scene_camera], dtype=np.float64)
    position = np.mean(positions, axis=0)
    forward = np.mean(forwards, axis=0)
    forward /= np.linalg.norm(forward)
    up = np.mean(ups, axis=0)
    up -= forward * float(np.dot(up, forward))
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    rotation = np.stack((right, -up, forward), axis=1)
    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, :3] = rotation
    world_from_camera[:3, 3] = position

    focal = 0.5 * height / math.tan(math.radians(vertical_fov_degrees) * 0.5)
    intrinsics = np.asarray(
        (
            (focal, 0.0, (width - 1) * 0.5),
            (0.0, focal, (height - 1) * 0.5),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return intrinsics, world_from_camera


def project_world_points(
    np: Any,
    *,
    points_world_m: Any,
    intrinsics_px: Any,
    world_from_camera: Any,
) -> tuple[Any, Any]:
    """Project named world-frame points into a calibrated pixel frame."""

    points = np.asarray(points_world_m, dtype=np.float64)
    intrinsics = np.asarray(intrinsics_px, dtype=np.float64)
    transform = np.asarray(world_from_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("world points must have shape Nx3")
    if intrinsics.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("camera matrices must have shapes 3x3 and 4x4")
    camera_from_world = np.linalg.inv(transform)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera = (camera_from_world @ homogeneous.T).T[:, :3]
    depth = camera[:, 2]
    if bool(np.any(depth <= 0)):
        raise ValueError("all projected points must lie in front of the camera")
    pixels_h = (intrinsics @ camera.T).T
    return pixels_h[:, :2] / pixels_h[:, 2:3], depth


def closest_points_to_polyline(
    np: Any,
    *,
    points_m: Any,
    polyline_m: Any,
) -> tuple[Any, Any]:
    """Return the closest point and distance from each point to a 3-D polyline."""

    points = np.asarray(points_m, dtype=np.float64)
    polyline = np.asarray(polyline_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("query points must have shape Nx3")
    if polyline.ndim != 2 or polyline.shape[1] != 3 or len(polyline) < 2:
        raise ValueError("polyline must have shape Mx3 with at least two nodes")
    if not bool(np.all(np.isfinite(points)) and np.all(np.isfinite(polyline))):
        raise ValueError("polyline distance inputs must be finite")
    starts = polyline[:-1]
    segments = polyline[1:] - starts
    lengths_squared = np.sum(segments * segments, axis=1)
    if bool(np.any(lengths_squared <= 1e-16)):
        raise ValueError("polyline contains a collapsed segment")
    relative = points[:, None, :] - starts[None, :, :]
    fractions = np.sum(relative * segments[None, :, :], axis=2) / lengths_squared[None, :]
    fractions = np.clip(fractions, 0.0, 1.0)
    candidates = starts[None, :, :] + fractions[..., None] * segments[None, :, :]
    distances = np.linalg.norm(points[:, None, :] - candidates, axis=2)
    selected = np.argmin(distances, axis=1)
    rows = np.arange(len(points))
    return candidates[rows, selected], distances[rows, selected]


def exact_pad_stem_contact_state(
    np: Any,
    *,
    pad_vertices_by_fingertip: dict[int, Any],
    stem_nodes_m: Any,
    stem_radius_m: float = 0.0015,
    maximum_surface_gap_m: float = 0.003,
    maximum_rest_mesh_penetration_m: float = 0.0015,
    maximum_contacts_per_fingertip: int = 24,
) -> dict[str, Any]:
    """Derive contacts only from exact fingertip-mesh vertices near the metric stem."""

    if not pad_vertices_by_fingertip:
        raise ValueError("at least one exact fingertip mesh is required")
    if min(
        stem_radius_m,
        maximum_surface_gap_m,
        maximum_rest_mesh_penetration_m,
    ) <= 0:
        raise ValueError("contact geometry and penalty parameters must be positive")
    if maximum_contacts_per_fingertip <= 0:
        raise ValueError("maximum contacts per fingertip must be positive")
    stem = np.asarray(stem_nodes_m, dtype=np.float64)
    contact_points = []
    gaps = []
    normals = []
    forces = []
    fingertip_indices = []
    closest_centerline_points = []
    rest_mesh_penetrations = []
    for fingertip_index, raw_vertices in pad_vertices_by_fingertip.items():
        vertices = np.asarray(raw_vertices, dtype=np.float64)
        closest, distances = closest_points_to_polyline(
            np,
            points_m=vertices,
            polyline_m=stem,
        )
        raw_gaps = distances - stem_radius_m
        eligible = np.flatnonzero(
            (raw_gaps >= -maximum_rest_mesh_penetration_m)
            & (raw_gaps <= maximum_surface_gap_m)
            & (distances > 1e-10)
        )
        if not len(eligible):
            continue
        first = int(eligible[np.argmin(np.abs(raw_gaps[eligible]))])
        selected_vertices = [first]
        while len(selected_vertices) < min(
            maximum_contacts_per_fingertip,
            len(eligible),
        ):
            remaining = [
                int(index)
                for index in eligible
                if int(index) not in selected_vertices
            ]
            next_vertex = max(
                remaining,
                key=lambda index: min(
                    float(np.linalg.norm(vertices[index] - vertices[selected]))
                    for selected in selected_vertices
                ),
            )
            selected_vertices.append(next_vertex)
        for selected in selected_vertices:
            distance = float(distances[selected])
            raw_gap = float(raw_gaps[selected])
            reported_gap = max(0.0, raw_gap)
            direction = vertices[selected] - closest[selected]
            direction /= distance
            contact_point = closest[selected] + stem_radius_m * direction
            contact_points.append(contact_point)
            gaps.append(reported_gap)
            normals.append(direction)
            forces.append(np.zeros(3, dtype=np.float64))
            fingertip_indices.append(int(fingertip_index))
            closest_centerline_points.append(closest[selected])
            rest_mesh_penetrations.append(max(0.0, -raw_gap))
    points_array = np.asarray(contact_points, dtype=np.float64).reshape(-1, 3)
    normals_array = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    forces_array = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
    center = (
        np.mean(np.asarray(closest_centerline_points), axis=0)
        if closest_centerline_points
        else np.mean(stem, axis=0)
    )
    return {
        "contact_points_m": points_array,
        "surface_gaps_m": np.asarray(gaps, dtype=np.float64),
        "contact_normals": normals_array,
        "contact_forces_n": forces_array,
        "object_center_m": center,
        "external_force_n": np.zeros(3, dtype=np.float64),
        "external_moment_nm": np.zeros(3, dtype=np.float64),
        "fingertip_indices": tuple(fingertip_indices),
        "candidate_fingertips": len(pad_vertices_by_fingertip),
        "contacting_fingertips": len(set(fingertip_indices)),
        "contact_patch_points": len(fingertip_indices),
        "maximum_rest_mesh_penetration_m": max(
            rest_mesh_penetrations,
            default=0.0,
        ),
        "contact_source": "exact_sharpa_elastomer_mesh_vertices",
        "force_source": None,
    }
