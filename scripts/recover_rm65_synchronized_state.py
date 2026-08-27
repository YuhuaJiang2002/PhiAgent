#!/usr/bin/env python3
"""Recover a source-timed RM65/AG2F90-C visual replay state from RGB anchors.

The source video already contains two RM65 arms, so this tool does not infer a
human pose.  It combines reviewed fingertip anchors with local optical flow,
the reviewed table homography, source-specific grasp/release events and the
official robot kinematic model.  The result is a per-frame q/g trajectory for
same-frame MuJoCo replay.  It is visual state estimation, not encoder ground
truth or a calibrated robot-control trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from render_realman_rm65_visual_replay import (
    _gripper_qpos,
    _joint_dofs,
    build_model,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--eef-anchor-overrides", type=Path, help="Reviewed visible fingertip anchors that replace contact-target points in the legacy provenance.")
    parser.add_argument("--rm65-urdf", type=Path, required=True)
    parser.add_argument("--ag2f90c-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-base", nargs=3, type=float, default=(0.05, -0.18, 0.0))
    parser.add_argument("--left-base-yaw", type=float, default=math.pi / 2)
    parser.add_argument("--right-base", nargs=3, type=float, default=(0.50, 0.24, 0.0))
    parser.add_argument("--right-base-yaw", type=float, default=math.pi)
    parser.add_argument("--left-base-roll", type=float, default=0.0)
    parser.add_argument("--left-base-pitch", type=float, default=0.0)
    parser.add_argument("--right-base-roll", type=float, default=0.0)
    parser.add_argument("--right-base-pitch", type=float, default=0.0)
    parser.add_argument("--table-half-size", nargs=2, type=float, default=(0.72, 0.52))
    parser.add_argument("--table-center-y", type=float, default=0.14)
    parser.add_argument("--direct-plane-scale", type=float, default=0.0, help="If positive, map table homography coordinates directly into the robot world at this scale.")
    parser.add_argument("--plane-offset", nargs=2, type=float, default=(0.0, 0.08))
    parser.add_argument("--z-scale", type=float, default=0.8)
    parser.add_argument("--z-offset", type=float, default=0.02)
    parser.add_argument("--rotation-weight", type=float, default=0.005)
    parser.add_argument("--left-seed", nargs=6, type=float, default=(0.0, -0.55, 0.75, 0.0, 0.55, 0.0))
    parser.add_argument("--right-seed", nargs=6, type=float, default=(0.0, -0.55, 0.75, 0.0, 0.55, 0.0))
    parser.add_argument("--posture-weight", type=float, default=0.001)
    parser.add_argument(
        "--backproject-camera",
        nargs=6,
        type=float,
        metavar=("AZIMUTH", "ELEVATION", "DISTANCE", "LOOK_X", "LOOK_Y", "LOOK_Z"),
        help="Backproject each reviewed EEF pixel to the camera ray and select the point nearest its planar 3-D proposal.",
    )
    parser.add_argument(
        "--backproject-blend",
        type=float,
        default=1.0,
        help="Blend from the planar 3-D proposal toward its source-pixel camera ray (0 keeps the proposal, 1 reaches the ray).",
    )
    parser.add_argument(
        "--backproject-z-range",
        nargs=2,
        type=float,
        metavar=("MIN_Z", "MAX_Z"),
        help="Optional workspace-height clamp applied after camera-ray blending.",
    )
    parser.add_argument(
        "--joint-anchor-file",
        type=Path,
        help="Optional reviewed/multistart IK branch anchors with frames, left_q and right_q.",
    )
    parser.add_argument(
        "--opposed-grippers",
        action="store_true",
        help="Constrain the two gripper longitudinal axes to face one another at every frame.",
    )
    return parser.parse_args()


def _read_gray(video: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read {video}")
    return frames, fps


def _local_flow_step(previous: np.ndarray, current: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, float]:
    """Track a distal gripper point using robust local feature motion."""
    height, width = previous.shape
    radius = 72
    x0, y0 = np.round(point).astype(int)
    mask = np.zeros_like(previous)
    cv2.circle(mask, (int(np.clip(x0, 0, width - 1)), int(np.clip(y0, 0, height - 1))), radius, 255, -1)
    # Prefer high-gradient, dark robot/gripper features over the low-texture table.
    gradient = cv2.magnitude(cv2.Sobel(previous, cv2.CV_32F, 1, 0), cv2.Sobel(previous, cv2.CV_32F, 0, 1))
    mask[(gradient < 18) | (previous > 205)] = 0
    features = cv2.goodFeaturesToTrack(previous, 100, 0.01, 5, mask=mask, blockSize=5)
    if features is None or len(features) < 6:
        return point.copy(), 0.0
    tracked, status, error = cv2.calcOpticalFlowPyrLK(
        previous, current, features, None,
        winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
    )
    keep = status.reshape(-1).astype(bool) & (error.reshape(-1) < 30)
    source = features.reshape(-1, 2)[keep]
    target = tracked.reshape(-1, 2)[keep]
    if len(source) < 5:
        return point.copy(), 0.0
    transform, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=1000
    )
    if transform is None:
        displacement = np.median(target - source, axis=0)
        return point + displacement, min(1.0, len(source) / 25.0)
    prediction = transform[:, :2] @ point + transform[:, 2]
    confidence = float(np.mean(inliers)) if inliers is not None else min(1.0, len(source) / 25.0)
    return prediction, confidence


def _dense_track(frames: list[np.ndarray], anchors: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    count = len(frames)
    output = np.zeros((count, 2), dtype=np.float64)
    confidence = np.zeros(count, dtype=np.float64)
    keys = sorted(anchors)
    for start, stop in zip(keys[:-1], keys[1:]):
        length = stop - start + 1
        linear = np.linspace(anchors[start], anchors[stop], length)
        energies = [0.0]
        for local, index in enumerate(range(start + 1, stop + 1), start=1):
            center = np.round(linear[local]).astype(int)
            x0, y0 = center
            radius = 54
            x1, x2 = max(0, x0 - radius), min(frames[index].shape[1], x0 + radius)
            y1, y2 = max(0, y0 - radius), min(frames[index].shape[0], y0 + radius)
            difference = cv2.absdiff(frames[index - 1][y1:y2, x1:x2], frames[index][y1:y2, x1:x2])
            energies.append(float(np.percentile(difference, 75)) + 0.5)
        energies = gaussian_filter1d(np.asarray(energies), sigma=1.0, mode="nearest")
        cumulative = np.cumsum(energies)
        cumulative = (cumulative - cumulative[0]) / max(cumulative[-1] - cumulative[0], 1e-8)
        uniform = np.linspace(0.0, 1.0, length)
        progress = 0.55 * cumulative + 0.45 * uniform
        progress = progress * progress * (3.0 - 2.0 * progress)
        predicted_array = anchors[start][None, :] * (1.0 - progress[:, None]) + anchors[stop][None, :] * progress[:, None]
        scores = np.clip(energies / max(np.percentile(energies, 90), 1e-8), 0.0, 1.0)
        output[start : stop + 1] = predicted_array
        confidence[start : stop + 1] = scores
    output[keys[-1] :] = anchors[keys[-1]]
    confidence[keys[-1] :] = 1.0
    # Small smoothing is followed by exact anchor restoration.
    output = gaussian_filter1d(output, sigma=0.65, axis=0, mode="nearest")
    for index, value in anchors.items():
        output[index] = value
        confidence[index] = 1.0
    return output, confidence


def _homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _backproject_nearest_to_reference(
    points_px: np.ndarray,
    reference_xyz: np.ndarray,
    model: mujoco.MjModel,
    camera_parameters: tuple[float, ...],
) -> np.ndarray:
    """Lift pixels onto camera rays without inventing an arbitrary fixed depth.

    Monocular RGB does not determine depth.  The minimum-change choice used
    here is the point on each source pixel ray closest to the existing planar
    3-D proposal.  This preserves the proposal scale while making image-space
    EEF reprojection exact for the fixed visual-replay camera.
    """
    azimuth, elevation, distance, look_x, look_y, look_z = camera_parameters
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance
    camera.lookat[:] = (look_x, look_y, look_z)
    with mujoco.Renderer(model, height=768, width=1024) as renderer:
        renderer.update_scene(data, camera=camera)
        scene_camera = renderer.scene.camera[0]
        camera_position = np.asarray(scene_camera.pos, dtype=np.float64)
        forward = np.asarray(scene_camera.forward, dtype=np.float64)
        up = np.asarray(scene_camera.up, dtype=np.float64)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        tan_y = float(scene_camera.frustum_top / scene_camera.frustum_near)
    tan_x = tan_y * (1024.0 / 768.0)
    normalized_x = (points_px[:, 0] - 512.0) / 512.0
    normalized_y = (384.0 - points_px[:, 1]) / 384.0
    rays = (
        forward[None, :]
        + normalized_x[:, None] * tan_x * right[None, :]
        + normalized_y[:, None] * tan_y * up[None, :]
    )
    ray_parameter = np.sum((reference_xyz - camera_position) * rays, axis=1) / np.sum(rays * rays, axis=1)
    return camera_position[None, :] + ray_parameter[:, None] * rays


def _event_command(frames: int) -> np.ndarray:
    """Two source-visible grasp/release cycles, with finite transition time."""
    keyframes = np.asarray((0, 24, 40, 64, 76, 96, 112, 150, 162, frames - 1), dtype=int)
    values = np.asarray((0, 0, 1, 1, 0, 0, 1, 1, 0, 0), dtype=np.float64)
    command = np.interp(np.arange(frames), keyframes, values)
    command = gaussian_filter1d(command, sigma=0.8, mode="nearest")
    return np.clip(command, 0.0, 1.0)


def _site_target_rotation(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    direction = target[:2] - base[:2]
    direction /= max(np.linalg.norm(direction), 1e-8)
    site_z = np.asarray((direction[0], direction[1], -0.12), dtype=np.float64)
    site_z /= np.linalg.norm(site_z)
    site_x = np.asarray((0.12 * direction[0], 0.12 * direction[1], 1.0), dtype=np.float64)
    site_x -= site_z * np.dot(site_x, site_z)
    site_x /= np.linalg.norm(site_x)
    site_y = np.cross(site_z, site_x)
    return np.column_stack((site_x, site_y, site_z))


def _opposed_site_rotation(target: np.ndarray, counterpart: np.ndarray) -> np.ndarray:
    """Point the gripper longitudinal axis toward its opposite gripper.

    The AG2F90-C EEF site is located on local +Z, so the third rotation column
    controls the visible gripper approach axis.  A small downward component
    preserves the source video's cloth-facing wrist pitch without turning the
    side-mounted arms back into top-down tabletop arms.
    """
    site_z = np.asarray(counterpart - target, dtype=np.float64)
    site_z[2] -= 0.08
    site_z /= max(np.linalg.norm(site_z), 1e-8)
    site_x = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    site_x -= site_z * np.dot(site_x, site_z)
    site_x /= max(np.linalg.norm(site_x), 1e-8)
    site_y = np.cross(site_z, site_x)
    return np.column_stack((site_x, site_y, site_z))


def _solve_pose_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    dofs: np.ndarray,
    site: int,
    target_xyz: np.ndarray,
    target_rotation: np.ndarray,
    rotation_weight: float,
    preferred_q: np.ndarray,
    posture_weight: float,
) -> tuple[float, float]:
    for _ in range(72):
        mujoco.mj_forward(model, data)
        position_error = target_xyz - data.site_xpos[site]
        current_rotation = data.site_xmat[site].reshape(3, 3)
        rotation_error = Rotation.from_matrix(target_rotation @ current_rotation.T).as_rotvec()
        if np.linalg.norm(position_error) < 0.0018 and np.linalg.norm(rotation_error) < 0.05:
            break
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site)
        jacobian = np.vstack((jacp[:, dofs], rotation_weight * jacr[:, dofs]))
        error = np.r_[position_error, rotation_weight * rotation_error]
        current_q = data.qpos[qpos]
        lhs = jacobian.T @ jacobian + (5e-4 + posture_weight) * np.eye(len(qpos))
        rhs = jacobian.T @ error + posture_weight * (preferred_q - current_q)
        step = np.linalg.solve(lhs, rhs)
        data.qpos[qpos] += np.clip(step, -0.07, 0.07)
        for dof, index in zip(dofs, qpos):
            joint = model.dof_jntid[dof]
            data.qpos[index] = np.clip(data.qpos[index], *model.jnt_range[joint])
    mujoco.mj_forward(model, data)
    position = float(np.linalg.norm(target_xyz - data.site_xpos[site]))
    current_rotation = data.site_xmat[site].reshape(3, 3)
    angle = float(np.linalg.norm(Rotation.from_matrix(target_rotation @ current_rotation.T).as_rotvec()))
    return position, angle


def _recover_q(
    model: mujoco.MjModel,
    targets: np.ndarray,
    prefix: str,
    base: np.ndarray,
    rotation_weight: float,
    seed: np.ndarray,
    posture_weight: float,
    target_rotations: np.ndarray | None = None,
    q_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    qpos, dofs = _joint_dofs(model, prefix)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_eef")
    # A bent seed avoids the straight-arm singularity at q=0.
    seed = np.asarray(seed, dtype=np.float64)
    data.qpos[qpos] = q_reference[0] if q_reference is not None else seed
    recovered, pos_errors, rot_errors = [], [], []
    previous_q: np.ndarray | None = None
    for frame_index, target in enumerate(targets):
        preferred_q = q_reference[frame_index] if q_reference is not None else seed
        rotation = (
            target_rotations[frame_index]
            if target_rotations is not None
            else _site_target_rotation(target, base)
        )
        pos_error, rot_error = _solve_pose_ik(
            model, data, qpos, dofs, site, target, rotation, rotation_weight,
            preferred_q, posture_weight,
        )
        if pos_error > 0.025:
            pos_error, rot_error = _solve_pose_ik(
                model, data, qpos, dofs, site, target, rotation, 0.0, preferred_q, posture_weight
            )
        candidate = data.qpos[qpos].copy()
        if previous_q is not None:
            candidate = np.clip(candidate, previous_q - 0.16, previous_q + 0.16)
            data.qpos[qpos] = candidate
            mujoco.mj_forward(model, data)
            pos_error = float(np.linalg.norm(target - data.site_xpos[site]))
        recovered.append(candidate)
        previous_q = candidate
        pos_errors.append(pos_error); rot_errors.append(rot_error)
    q = np.asarray(recovered)
    # Temporal acceleration prior; preserve the fit through a short refinement.
    q = gaussian_filter1d(q, sigma=1.15, axis=0, mode="nearest")
    refined, pos_errors, rot_errors = [], [], []
    previous_q = None
    for index, target in enumerate(targets):
        data.qpos[qpos] = q[index]
        preferred_q = q_reference[index] if q_reference is not None else seed
        rotation = (
            target_rotations[index]
            if target_rotations is not None
            else _site_target_rotation(target, base)
        )
        pos_error, rot_error = _solve_pose_ik(
            model, data, qpos, dofs, site, target, rotation, rotation_weight,
            preferred_q, posture_weight,
        )
        if pos_error > 0.025:
            pos_error, rot_error = _solve_pose_ik(
                model, data, qpos, dofs, site, target, rotation, 0.0, preferred_q, posture_weight
            )
        candidate = data.qpos[qpos].copy()
        if previous_q is not None:
            candidate = np.clip(candidate, previous_q - 0.14, previous_q + 0.14)
            data.qpos[qpos] = candidate
            mujoco.mj_forward(model, data)
            pos_error = float(np.linalg.norm(target - data.site_xpos[site]))
        refined.append(candidate)
        previous_q = candidate
        pos_errors.append(pos_error); rot_errors.append(rot_error)
    q = gaussian_filter1d(np.asarray(refined), sigma=0.55, axis=0, mode="nearest")
    return q, np.asarray(pos_errors), np.asarray(rot_errors)


def main() -> None:
    args = _args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = json.loads(args.provenance.read_text())
    frames, fps = _read_gray(args.video)
    anchors = provenance["anchors"]
    left_anchor = {int(item["frame"]): np.asarray(item["left_tip_px"], dtype=float) for item in anchors}
    right_anchor = {int(item["frame"]): np.asarray(item["right_tip_px"], dtype=float) for item in anchors}
    if args.eef_anchor_overrides:
        overrides = json.loads(args.eef_anchor_overrides.read_text())
        left_anchor.update({int(frame): np.asarray(value, dtype=float) for frame, value in overrides["left_tip_px"].items()})
        right_anchor.update({int(frame): np.asarray(value, dtype=float) for frame, value in overrides["right_tip_px"].items()})
    left_px, left_confidence = _dense_track(frames, left_anchor)
    right_px, right_confidence = _dense_track(frames, right_anchor)
    matrix = np.asarray(provenance["calibration"]["homography_image_px_to_aloha_xy"], dtype=float)
    initial = provenance["calibration"]["target_initial_eef_xyz_m"]

    def target(points_px: np.ndarray, side: str) -> np.ndarray:
        plane = _homography(points_px, matrix)
        anchor_frames = np.asarray([item["frame"] for item in anchors])
        anchor_z = np.asarray([item[f"{side}_z_m"] for item in anchors])
        if args.direct_plane_scale > 0:
            xyz = np.zeros((len(points_px), 3), dtype=np.float64)
            xyz[:, :2] = plane * args.direct_plane_scale + np.asarray(args.plane_offset)
            xyz[:, 2] = args.z_scale * np.interp(np.arange(len(points_px)), anchor_frames, anchor_z) + args.z_offset
            return xyz
        start_plane = _homography(points_px[:1], matrix)[0]
        xyz = np.zeros((len(points_px), 3), dtype=np.float64)
        xyz[:, :2] = np.asarray(initial[side])[:2] + plane - start_plane
        xyz[:, 2] = float(initial[side][2]) + np.interp(np.arange(len(points_px)), anchor_frames, anchor_z - anchor_z[0])
        # Match the source-conditioned metric placement used by the previous
        # renderer while retaining the new dense RGB time parameterization.
        xyz[:, 0] *= 1.25
        xyz[:, 1] += 0.12
        xyz[:, 2] -= 0.11
        xyz[:, 0] += -0.03 if side == "left" else 0.03
        return xyz

    left_target = target(left_px, "left")
    right_target = target(right_px, "right")
    left_rotations = right_rotations = None
    if args.opposed_grippers:
        left_rotations = np.asarray([
            _opposed_site_rotation(left, right)
            for left, right in zip(left_target, right_target)
        ])
        right_rotations = np.asarray([
            _opposed_site_rotation(right, left)
            for left, right in zip(left_target, right_target)
        ])
    model = build_model(
        args.rm65_urdf, args.ag2f90c_dir, False,
        tuple(args.left_base), args.left_base_yaw,
        tuple(args.right_base), args.right_base_yaw,
        args.left_base_roll, args.left_base_pitch,
        args.right_base_roll, args.right_base_pitch,
        tuple(args.table_half_size), args.table_center_y,
    )
    if args.backproject_camera:
        if not 0.0 <= args.backproject_blend <= 1.0:
            raise ValueError("--backproject-blend must be in [0, 1]")
        left_ray_target = _backproject_nearest_to_reference(
            left_px, left_target, model, tuple(args.backproject_camera)
        )
        right_ray_target = _backproject_nearest_to_reference(
            right_px, right_target, model, tuple(args.backproject_camera)
        )
        left_target += args.backproject_blend * (left_ray_target - left_target)
        right_target += args.backproject_blend * (right_ray_target - right_target)
        if args.backproject_z_range:
            min_z, max_z = args.backproject_z_range
            if min_z > max_z:
                raise ValueError("--backproject-z-range MIN_Z must not exceed MAX_Z")
            left_target[:, 2] = np.clip(left_target[:, 2], min_z, max_z)
            right_target[:, 2] = np.clip(right_target[:, 2], min_z, max_z)
    left_q_reference = right_q_reference = None
    if args.joint_anchor_file:
        joint_anchors = json.loads(args.joint_anchor_file.read_text())
        joint_frames = np.asarray(joint_anchors["frames"], dtype=int)
        if joint_frames[0] != 0 or joint_frames[-1] != len(frames) - 1:
            raise ValueError("joint anchor frames must include the first and last video frames")

        def interpolate_joint_anchors(key: str) -> np.ndarray:
            values = np.asarray(joint_anchors[key], dtype=np.float64)
            if values.shape != (len(joint_frames), 6):
                raise ValueError(f"{key} must have shape ({len(joint_frames)}, 6)")
            return np.column_stack([
                np.interp(np.arange(len(frames)), joint_frames, values[:, joint])
                for joint in range(6)
            ])

        left_q_reference = interpolate_joint_anchors("left_q")
        right_q_reference = interpolate_joint_anchors("right_q")
    left_q, left_position_error, left_rotation_error = _recover_q(
        model, left_target, "left", np.asarray(args.left_base), args.rotation_weight,
        np.asarray(args.left_seed), args.posture_weight, left_rotations, left_q_reference,
    )
    right_q, right_position_error, right_rotation_error = _recover_q(
        model, right_target, "right", np.asarray(args.right_base), args.rotation_weight,
        np.asarray(args.right_seed), args.posture_weight, right_rotations, right_q_reference,
    )
    command = _event_command(len(frames))
    phase = np.empty(len(frames), dtype="U64")
    anchor_frames = [int(item["frame"]) for item in anchors]
    for index, item in enumerate(anchors):
        stop = anchor_frames[index + 1] if index + 1 < len(anchors) else len(frames)
        phase[int(item["frame"]) : stop] = item["phase"]
    state_path = args.output_dir / "rm65_synchronized_state.npz"
    np.savez_compressed(
        state_path,
        left_q=left_q, right_q=right_q,
        left_gripper_command=command, right_gripper_command=command.copy(),
        left_target_xyz=left_target, right_target_xyz=right_target,
        left_tip_px=left_px, right_tip_px=right_px,
        left_track_confidence=left_confidence, right_track_confidence=right_confidence,
        phase=phase, fps=fps,
    )
    velocity_left = np.diff(left_q, axis=0) * fps
    velocity_right = np.diff(right_q, axis=0) * fps
    acceleration_left = np.diff(velocity_left, axis=0) * fps
    acceleration_right = np.diff(velocity_right, axis=0) * fps
    manifest = {
        "schema_version": "phiagent-rm65-synchronized-state/1.0",
        "source_video": str(args.video.resolve()),
        "eef_anchor_overrides": str(args.eef_anchor_overrides.resolve()) if args.eef_anchor_overrides else None,
        "frames": len(frames), "fps": fps,
        "observation_layers": {
            "camera_plane": "reviewed four-corner table homography",
            "eef_image_motion": "reviewed distal anchors + interval-local RGB motion-energy time warp (path remains anchor-bounded)",
            "robot_bases": "source-conditioned visible-base estimates; not survey-grade extrinsics",
            "gripper": "two RGB-visible grasp/release cycles with smoothed event constraints",
        },
        "base_poses": {"left_xyz_rpy": [*args.left_base, args.left_base_roll, args.left_base_pitch, args.left_base_yaw], "right_xyz_rpy": [*args.right_base, args.right_base_roll, args.right_base_pitch, args.right_base_yaw]},
        "world_mapping": {"direct_plane_scale": args.direct_plane_scale, "plane_offset": args.plane_offset, "z_scale": args.z_scale, "z_offset": args.z_offset},
        "camera_ray_backprojection": {
            "enabled": bool(args.backproject_camera),
            "camera": args.backproject_camera,
            "blend": args.backproject_blend if args.backproject_camera else None,
            "z_range": args.backproject_z_range if args.backproject_camera else None,
            "depth_rule": "point on source pixel ray nearest the planar 3-D proposal" if args.backproject_camera else None,
        },
        "posture_prior": {"left_seed": args.left_seed, "right_seed": args.right_seed, "weight": args.posture_weight},
        "joint_anchor_file": str(args.joint_anchor_file.resolve()) if args.joint_anchor_file else None,
        "wrist_orientation": "opposed gripper approach axes" if args.opposed_grippers else "base-to-target approach axes",
        "track_confidence_mean": {"left": float(left_confidence.mean()), "right": float(right_confidence.mean())},
        "ik_position_error_m": {
            "left_mean": float(left_position_error.mean()), "left_max": float(left_position_error.max()),
            "right_mean": float(right_position_error.mean()), "right_max": float(right_position_error.max()),
        },
        "ik_orientation_error_deg": {
            "left_mean": float(np.degrees(left_rotation_error).mean()), "left_max": float(np.degrees(left_rotation_error).max()),
            "right_mean": float(np.degrees(right_rotation_error).mean()), "right_max": float(np.degrees(right_rotation_error).max()),
        },
        "temporal": {
            "left_max_joint_velocity_rad_s": float(np.abs(velocity_left).max()),
            "right_max_joint_velocity_rad_s": float(np.abs(velocity_right).max()),
            "left_max_joint_acceleration_rad_s2": float(np.abs(acceleration_left).max()),
            "right_max_joint_acceleration_rad_s2": float(np.abs(acceleration_right).max()),
            "grasp_intervals_frames": [[40, 64], [112, 150]],
            "release_transitions_frames": [[64, 76], [150, 162]],
        },
        "claim_boundary": "same-frame source-conditioned visual state estimate; not joint encoder GT, metric camera/base calibration, collision proof, or executable controller output",
        "state_npz": str(state_path.resolve()),
    }
    (args.output_dir / "state_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
