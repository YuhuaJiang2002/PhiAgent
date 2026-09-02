#!/usr/bin/env python3
"""Fuse a posture-stable RM65 proposal with a camera-ray-aligned proposal.

This is intentionally a refinement stage: monocular RGB fixes an image ray but
does not determine metric depth.  Blending the ray solution with the stable
planar proposal retains the visible motion while regularising the hidden depth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from recover_rm65_synchronized_state import _recover_q
from render_realman_rm65_visual_replay import _joint_dofs, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-state", type=Path, required=True)
    parser.add_argument("--ray-state", type=Path, required=True)
    parser.add_argument("--rm65-urdf", type=Path, required=True)
    parser.add_argument("--ag2f90c-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blend", type=float, required=True)
    parser.add_argument(
        "--target-z-offset",
        type=float,
        default=0.0,
        help="Vertical EEF offset applied after stable/ray fusion and before workspace clipping.",
    )
    parser.add_argument(
        "--left-target-z-offset",
        type=float,
        help="Override the common EEF Z offset for the left arm.",
    )
    parser.add_argument(
        "--right-target-z-offset",
        type=float,
        help="Override the common EEF Z offset for the right arm.",
    )
    parser.add_argument(
        "--eef-z-keyframes",
        type=Path,
        help=(
            "Reviewed frame/left_z/right_z anchors that override monocular EEF height "
            "while retaining its XY path."
        ),
    )
    parser.add_argument("--z-range", nargs=2, type=float, default=(0.04, 0.24))
    parser.add_argument("--left-base", nargs=3, type=float, required=True)
    parser.add_argument("--right-base", nargs=3, type=float, required=True)
    parser.add_argument("--left-base-rpy", nargs=3, type=float, required=True)
    parser.add_argument("--right-base-rpy", nargs=3, type=float, required=True)
    parser.add_argument("--left-seed", nargs=6, type=float, required=True)
    parser.add_argument("--right-seed", nargs=6, type=float, required=True)
    parser.add_argument("--rotation-weight", type=float, default=0.02)
    parser.add_argument(
        "--left-rotation-weight",
        type=float,
        help="Override --rotation-weight for the left arm.",
    )
    parser.add_argument(
        "--right-rotation-weight",
        type=float,
        help="Override --rotation-weight for the right arm.",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=("full", "axis", "none"),
        default="full",
        help="Use axis-only orientation for monocular RGB to avoid an unobserved wrist-roll flip.",
    )
    parser.add_argument("--posture-weight", type=float, default=5e-6)
    parser.add_argument(
        "--left-tool-roll-offset-deg",
        type=float,
        default=0.0,
        help="Fixed joint-6/tool-roll mounting offset applied after axis-only IK.",
    )
    parser.add_argument(
        "--right-tool-roll-offset-deg",
        type=float,
        default=0.0,
        help="Fixed joint-6/tool-roll mounting offset applied after axis-only IK.",
    )
    parser.add_argument("--table-half-size", nargs=2, type=float, default=(0.55, 0.35))
    parser.add_argument("--table-center-y", type=float, default=0.10)
    parser.add_argument(
        "--camera",
        nargs=6,
        type=float,
        metavar=("AZIMUTH", "ELEVATION", "DISTANCE", "LOOK_X", "LOOK_Y", "LOOK_Z"),
        help="MuJoCo review camera used to lift an observed 2-D gripper axis into 3-D.",
    )
    parser.add_argument(
        "--left-screen-axis",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "VIEW_DEPTH"),
        help="Visible wrist-to-tip image direction plus its unobserved camera-depth component.",
    )
    parser.add_argument(
        "--right-screen-axis",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "VIEW_DEPTH"),
        help="Visible wrist-to-tip image direction plus its unobserved camera-depth component.",
    )
    parser.add_argument(
        "--screen-axis-keyframes",
        type=Path,
        help=(
            "Reviewed frame-wise left/right wrist-to-tip image axes; overrides "
            "constant screen axes."
        ),
    )
    parser.add_argument(
        "--project-gripper-axis-to-table-plane",
        action="store_true",
        help=(
            "Resolve the monocular camera-depth component analytically so the "
            "wrist-to-tip axis lies in the world XY/table plane."
        ),
    )
    parser.add_argument(
        "--joint-anchor-file",
        type=Path,
        help="Multistart IK branch anchors used as a temporal posture prior.",
    )
    parser.add_argument(
        "--fit-screen-width-roll",
        choices=("none", "left", "right", "both"),
        default="none",
        help=(
            "Fit joint 6 after axis IK so the projected gripper-width axis is "
            "perpendicular to the reviewed wrist-to-tip image axis."
        ),
    )
    parser.add_argument(
        "--screen-width-roll-smoothing-window",
        type=int,
        default=1,
        help=(
            "Odd triangular smoothing window applied only to the fitted joint-6 roll; "
            "1 disables smoothing."
        ),
    )
    return parser.parse_args()


def target_rotation_from_axis(axis: np.ndarray) -> np.ndarray:
    site_z = np.asarray(axis, dtype=np.float64)
    site_z /= np.linalg.norm(site_z)
    site_x = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    site_x -= site_z * np.dot(site_x, site_z)
    if np.linalg.norm(site_x) < 0.1:
        site_x = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        site_x -= site_z * np.dot(site_x, site_z)
    site_x /= np.linalg.norm(site_x)
    return np.column_stack((site_x, np.cross(site_z, site_x), site_z))


def observed_axis_rotations(
    model: mujoco.MjModel,
    frames: int,
    camera_values: list[float],
    image_axis: np.ndarray | list[float],
    plane_normal: np.ndarray | None = None,
) -> np.ndarray:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth, camera.elevation, camera.distance = camera_values[:3]
    camera.lookat[:] = camera_values[3:]
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, height=768, width=1024) as renderer:
        renderer.update_scene(data, camera=camera)
        scene_camera = renderer.scene.camera[0]
        forward = np.asarray(scene_camera.forward, dtype=np.float64)
        up = np.asarray(scene_camera.up, dtype=np.float64)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
    axes = np.asarray(image_axis, dtype=np.float64)
    if axes.shape == (3,):
        axes = np.repeat(axes[None, :], frames, axis=0)
    if axes.shape != (frames, 3):
        raise ValueError(f"screen axes must have shape (3,) or ({frames}, 3), got {axes.shape}")
    rotations = []
    for dx, dy, view_depth in axes:
        screen_axis = dx * right - dy * up
        norm = np.linalg.norm(screen_axis)
        if norm < 1e-8:
            raise ValueError("screen-axis DX/DY must not both be zero")
        screen_axis /= norm
        if plane_normal is not None:
            normal = np.asarray(plane_normal, dtype=np.float64)
            normal /= np.linalg.norm(normal)
            denominator = float(np.dot(forward, normal))
            if abs(denominator) < 1e-8:
                raise ValueError("camera forward is parallel to the requested gripper-axis plane")
            view_depth = -float(np.dot(screen_axis, normal)) / denominator
        world_axis = screen_axis + view_depth * forward
        rotations.append(target_rotation_from_axis(world_axis))
    return np.asarray(rotations)


def temporal_metrics(q: np.ndarray, fps: float) -> dict[str, float]:
    velocity = np.diff(q, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    return {
        "max_joint_velocity_rad_s": float(np.abs(velocity).max()),
        "max_joint_acceleration_rad_s2": float(np.abs(acceleration).max()),
    }


def canonicalize_periodic_joints(
    model: mujoco.MjModel,
    q: np.ndarray,
    side: str,
    reference: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Choose an in-limit, temporally continuous representative of periodic joints.

    MuJoCo's IK can return a revolute angle outside the URDF interval even when
    an exactly equivalent ``q + 2 k pi`` value is legal.  RM65 joint 6 spans
    almost two complete turns, so rejecting that raw value would incorrectly
    discard a valid wrist pose.  Prefer the reviewed branch prior when one is
    available and otherwise the previous frame.  Narrow-range joints are never
    wrapped because their limits describe a real mechanical restriction.
    """

    output = np.asarray(q, dtype=np.float64).copy()
    _, dofs = _joint_dofs(model, side)
    limits = np.asarray([model.jnt_range[model.dof_jntid[dof]] for dof in dofs])
    periodic = (limits[:, 1] - limits[:, 0]) >= (2.0 * np.pi - 1e-2)
    changed = np.zeros(output.shape[1], dtype=int)
    for frame in range(len(output)):
        for joint in np.flatnonzero(periodic):
            candidates = output[frame, joint] + 2.0 * np.pi * np.arange(-4, 5)
            legal = candidates[
                (candidates >= limits[joint, 0] - 1e-9)
                & (candidates <= limits[joint, 1] + 1e-9)
            ]
            if legal.size == 0:
                continue
            if reference is not None:
                target = float(reference[frame, joint])
            elif frame:
                target = float(output[frame - 1, joint])
            else:
                target = float(np.clip(output[frame, joint], *limits[joint]))
            selected = float(legal[np.argmin(np.abs(legal - target))])
            if not np.isclose(selected, output[frame, joint], atol=1e-10):
                changed[joint] += 1
            output[frame, joint] = selected
    return output, {
        "periodic_joint_indices_1based": (np.flatnonzero(periodic) + 1).tolist(),
        "canonicalized_frames_by_joint_1based": {
            str(joint + 1): int(count)
            for joint, count in enumerate(changed)
            if count
        },
        "selection_prior": "reviewed_joint_branch" if reference is not None else "previous_frame",
    }


def _camera_screen_basis(
    model: mujoco.MjModel,
    camera_values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth, camera.elevation, camera.distance = camera_values[:3]
    camera.lookat[:] = camera_values[3:]
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, height=768, width=1024) as renderer:
        renderer.update_scene(data, camera=camera)
        scene_camera = renderer.scene.camera[0]
        forward = np.asarray(scene_camera.forward, dtype=np.float64)
        up = np.asarray(scene_camera.up, dtype=np.float64)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
    return right, up


def triangular_smooth(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a 1-D trajectory with an edge-padded odd triangular window."""

    values = np.asarray(values, dtype=np.float64)
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd integer")
    if window == 1:
        return values.copy(), np.asarray((1.0,), dtype=np.float64)
    radius = window // 2
    kernel = np.concatenate(
        (
            np.arange(1, radius + 2, dtype=np.float64),
            np.arange(radius, 0, -1, dtype=np.float64),
        )
    )
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid"), kernel


def fit_screen_width_roll(
    model: mujoco.MjModel,
    q: np.ndarray,
    side: str,
    camera_values: list[float],
    source_long_axes: np.ndarray,
    smoothing_window: int = 1,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit wrist roll without changing EEF position or its longitudinal axis.

    A two-finger gripper exposes a second visual cue that axis-only IK ignores:
    the line joining its fingers.  With no calibrated 3-D labels, use the image
    perpendicular to the reviewed wrist-to-tip line as an explicit proxy.  The
    proxy is undirected (a parallel-jaw gripper is visually pi-symmetric), and a
    small continuity cost prevents frame-wise wrist-flip branch changes.
    """

    output = np.asarray(q, dtype=np.float64).copy()
    qpos_indices, _ = _joint_dofs(model, side)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_eef")
    joint_id = model.dof_jntid[_joint_dofs(model, side)[1][5]]
    lower, upper = model.jnt_range[joint_id]
    right, up = _camera_screen_basis(model, camera_values)
    data = mujoco.MjData(model)
    selected_delta: list[float] = []
    previous = float(output[0, 5])
    for frame, (dx, dy, _) in enumerate(source_long_axes):
        target = np.asarray((dy, -dx), dtype=np.float64)
        target /= np.linalg.norm(target)
        centre = float(output[frame, 5])
        candidates = centre + np.deg2rad(np.arange(-180.0, 180.0 + 1e-6, 1.0))
        candidates = candidates[(candidates >= lower) & (candidates <= upper)]
        best: tuple[float, float, float] | None = None
        for candidate in candidates:
            data.qpos[qpos_indices] = output[frame]
            data.qpos[qpos_indices[5]] = candidate
            mujoco.mj_forward(model, data)
            rotation = data.site_xmat[site_id].reshape(3, 3)
            width_world = rotation[:, 1]
            projected = np.asarray(
                (np.dot(width_world, right), -np.dot(width_world, up)),
                dtype=np.float64,
            )
            norm = np.linalg.norm(projected)
            if norm < 1e-8:
                continue
            projected /= norm
            angle = float(np.arccos(np.clip(abs(np.dot(projected, target)), 0.0, 1.0)))
            # Image alignment remains the primary objective.  The two small
            # terms only break visually equivalent or near-equivalent ties.
            cost = angle + 0.002 * abs(candidate - centre) + 0.004 * abs(candidate - previous)
            if best is None or cost < best[0]:
                best = (cost, float(candidate), angle)
        if best is None:
            raise RuntimeError(f"no legal joint-6 roll candidate for {side} frame {frame}")
        output[frame, 5] = best[1]
        selected_delta.append(best[1] - centre)
        previous = best[1]

    output[:, 5], smoothing_kernel = triangular_smooth(output[:, 5], smoothing_window)

    # Report the visual residual after smoothing, not the lower raw grid-search
    # residual.  This exposes the small alignment/temporal-smoothness trade-off.
    errors: list[float] = []
    for frame, (dx, dy, _) in enumerate(source_long_axes):
        target = np.asarray((dy, -dx), dtype=np.float64)
        target /= np.linalg.norm(target)
        data.qpos[qpos_indices] = output[frame]
        mujoco.mj_forward(model, data)
        width_world = data.site_xmat[site_id].reshape(3, 3)[:, 1]
        projected = np.asarray(
            (np.dot(width_world, right), -np.dot(width_world, up)),
            dtype=np.float64,
        )
        projected /= np.linalg.norm(projected)
        errors.append(
            float(np.arccos(np.clip(abs(np.dot(projected, target)), 0.0, 1.0)))
        )

    frame_delta = np.diff(output[:, 5])
    return output, {
        "target": "image-perpendicular proxy to reviewed wrist-to-tip axis",
        "source_width_axis_is_directly_observed": False,
        "screen_width_axis_error_deg": {
            "mean": float(np.degrees(errors).mean()),
            "max": float(np.degrees(errors).max()),
        },
        "joint_6_delta_deg": {
            "min": float(np.degrees(selected_delta).min()),
            "max": float(np.degrees(selected_delta).max()),
        },
        "joint_6_range_rad": [float(output[:, 5].min()), float(output[:, 5].max())],
        "max_frame_joint_6_delta_rad": (
            float(np.abs(frame_delta).max()) if len(frame_delta) else 0.0
        ),
        "grid_resolution_deg": 1.0,
        "smoothing_window_frames": smoothing_window,
        "smoothing_kernel": smoothing_kernel.tolist(),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.blend <= 1.0:
        raise ValueError("--blend must be in [0, 1]")
    if (
        args.screen_width_roll_smoothing_window < 1
        or args.screen_width_roll_smoothing_window % 2 == 0
    ):
        raise ValueError("--screen-width-roll-smoothing-window must be a positive odd integer")
    stable = np.load(args.stable_state)
    ray = np.load(args.ray_state)
    if stable["left_target_xyz"].shape != ray["left_target_xyz"].shape:
        raise ValueError("candidate state shapes differ")

    targets = {}
    for side in ("left", "right"):
        key = f"{side}_target_xyz"
        targets[side] = stable[key] + args.blend * (ray[key] - stable[key])
        side_offset = getattr(args, f"{side}_target_z_offset")
        targets[side][:, 2] += args.target_z_offset if side_offset is None else side_offset
        targets[side][:, 2] = np.clip(targets[side][:, 2], *args.z_range)
    if args.eef_z_keyframes:
        height_anchors = json.loads(args.eef_z_keyframes.read_text())
        anchor_frames = np.asarray(height_anchors["frames"], dtype=int)
        if anchor_frames[0] != 0 or anchor_frames[-1] != len(targets["left"]) - 1:
            raise ValueError("EEF Z keyframes must include the first and last state frames")
        if np.any(np.diff(anchor_frames) <= 0):
            raise ValueError("EEF Z keyframe frames must be strictly increasing")
        for side in ("left", "right"):
            values = np.asarray(height_anchors[f"{side}_z_m"], dtype=np.float64)
            if values.shape != anchor_frames.shape:
                raise ValueError(f"{side}_z_m must match frames")
            targets[side][:, 2] = np.interp(
                np.arange(len(targets[side])), anchor_frames, values
            )

    model = build_model(
        args.rm65_urdf,
        args.ag2f90c_dir,
        False,
        tuple(args.left_base),
        args.left_base_rpy[2],
        tuple(args.right_base),
        args.right_base_rpy[2],
        args.left_base_rpy[0],
        args.left_base_rpy[1],
        args.right_base_rpy[0],
        args.right_base_rpy[1],
        tuple(args.table_half_size),
        args.table_center_y,
    )
    if bool(args.left_screen_axis) != bool(args.right_screen_axis):
        raise ValueError("left and right screen-axis constraints must be supplied together")
    if args.screen_axis_keyframes and args.left_screen_axis:
        raise ValueError("use either constant screen axes or --screen-axis-keyframes, not both")
    if (args.left_screen_axis or args.screen_axis_keyframes) and not args.camera:
        raise ValueError("--camera is required with screen-axis constraints")
    left_rotations = right_rotations = None
    frame_axis: dict[str, np.ndarray] | None = None
    axis_plane_normal = (
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        if args.project_gripper_axis_to_table_plane
        else None
    )
    screen_axis_manifest: dict[str, object] | None = None
    if args.screen_axis_keyframes:
        keyframes = json.loads(args.screen_axis_keyframes.read_text())
        keyframe_indices = np.asarray(keyframes["frames"], dtype=int)
        if keyframe_indices[0] != 0 or keyframe_indices[-1] != len(targets["left"]) - 1:
            raise ValueError("screen-axis keyframes must include the first and final state frames")
        if np.any(np.diff(keyframe_indices) <= 0):
            raise ValueError("screen-axis keyframe indices must be strictly increasing")
        frame_axis = {}
        for side in ("left", "right"):
            values = np.asarray(keyframes[f"{side}_dx_dy_view_depth"], dtype=np.float64)
            if values.shape != (len(keyframe_indices), 3):
                raise ValueError(
                    f"{side} screen-axis values must have shape "
                    f"({len(keyframe_indices)}, 3)"
                )
            frame_axis[side] = np.column_stack([
                np.interp(np.arange(len(targets[side])), keyframe_indices, values[:, component])
                for component in range(3)
            ])
        left_rotations = observed_axis_rotations(
            model, len(targets["left"]), args.camera, frame_axis["left"], axis_plane_normal
        )
        right_rotations = observed_axis_rotations(
            model, len(targets["right"]), args.camera, frame_axis["right"], axis_plane_normal
        )
        screen_axis_manifest = {
            "keyframes": str(args.screen_axis_keyframes),
            "frames": keyframe_indices.tolist(),
        }
    elif args.left_screen_axis:
        left_rotations = observed_axis_rotations(
            model, len(targets["left"]), args.camera, args.left_screen_axis, axis_plane_normal
        )
        right_rotations = observed_axis_rotations(
            model, len(targets["right"]), args.camera, args.right_screen_axis, axis_plane_normal
        )
        screen_axis_manifest = {
            "left_screen_dx_dy_view_depth": args.left_screen_axis,
            "right_screen_dx_dy_view_depth": args.right_screen_axis,
        }
    left_q_reference = right_q_reference = None
    if args.joint_anchor_file:
        joint_anchors = json.loads(args.joint_anchor_file.read_text())
        joint_frames = np.asarray(joint_anchors["frames"], dtype=int)
        if joint_frames[0] != 0 or joint_frames[-1] != len(targets["left"]) - 1:
            raise ValueError("joint anchor frames must include the first and final state frames")
        if np.any(np.diff(joint_frames) <= 0):
            raise ValueError("joint anchor frames must be strictly increasing")

        def interpolate_joint_anchors(side: str) -> np.ndarray:
            values = np.asarray(joint_anchors[f"{side}_q"], dtype=np.float64)
            if values.shape != (len(joint_frames), 6):
                raise ValueError(f"{side}_q must have shape ({len(joint_frames)}, 6)")
            return np.column_stack([
                np.interp(np.arange(len(targets[side])), joint_frames, values[:, joint])
                for joint in range(6)
            ])

        left_q_reference = interpolate_joint_anchors("left")
        right_q_reference = interpolate_joint_anchors("right")
    left_q, left_pos, left_rot = _recover_q(
        model, targets["left"], "left", np.asarray(args.left_base),
        args.rotation_weight if args.left_rotation_weight is None else args.left_rotation_weight,
        np.asarray(args.left_seed), args.posture_weight,
        target_rotations=left_rotations,
        q_reference=left_q_reference,
        orientation_mode=args.orientation_mode,
    )
    right_q, right_pos, right_rot = _recover_q(
        model, targets["right"], "right", np.asarray(args.right_base),
        args.rotation_weight if args.right_rotation_weight is None else args.right_rotation_weight,
        np.asarray(args.right_seed), args.posture_weight,
        target_rotations=right_rotations,
        q_reference=right_q_reference,
        orientation_mode=args.orientation_mode,
    )
    if args.orientation_mode == "full" and (
        args.left_tool_roll_offset_deg or args.right_tool_roll_offset_deg
    ):
        raise ValueError("tool-roll offsets require axis-only or position-only orientation")
    left_q[:, 5] += np.deg2rad(args.left_tool_roll_offset_deg)
    right_q[:, 5] += np.deg2rad(args.right_tool_roll_offset_deg)
    screen_width_roll_manifest: dict[str, object] = {}
    fit_sides = {
        "none": (),
        "left": ("left",),
        "right": ("right",),
        "both": ("left", "right"),
    }[args.fit_screen_width_roll]
    if fit_sides:
        if frame_axis is None or args.camera is None:
            raise ValueError(
                "--fit-screen-width-roll requires --screen-axis-keyframes and --camera"
            )
        for side in fit_sides:
            fitted, fit_manifest = fit_screen_width_roll(
                model,
                left_q if side == "left" else right_q,
                side,
                args.camera,
                frame_axis[side],
                args.screen_width_roll_smoothing_window,
            )
            if side == "left":
                left_q = fitted
            else:
                right_q = fitted
            screen_width_roll_manifest[side] = fit_manifest
    left_q, left_periodic_manifest = canonicalize_periodic_joints(
        model, left_q, "left", left_q_reference
    )
    right_q, right_periodic_manifest = canonicalize_periodic_joints(
        model, right_q, "right", right_q_reference
    )
    for side, q in (("left", left_q), ("right", right_q)):
        _, dofs = _joint_dofs(model, side)
        limits = np.asarray([model.jnt_range[model.dof_jntid[dof]] for dof in dofs])
        # The damped IK update clips at the URDF endpoints; roundoff can leave
        # values such as 2.2689000000000004 for a 2.2689 upper bound.  Snap
        # only those sub-nanoradian endpoint differences before the real check.
        q[:] = np.where(
            (q < limits[:, 0]) & (q >= limits[:, 0] - 1e-9),
            limits[:, 0],
            q,
        )
        q[:] = np.where(
            (q > limits[:, 1]) & (q <= limits[:, 1] + 1e-9),
            limits[:, 1],
            q,
        )
        outside = (q < limits[:, 0]) | (q > limits[:, 1])
        if np.any(outside):
            frames, joints = np.where(outside)
            details = [
                {
                    "frame": int(frame),
                    "joint_1based": int(joint + 1),
                    "q": float(q[frame, joint]),
                    "range": limits[joint].tolist(),
                }
                for frame, joint in zip(frames[:8], joints[:8])
            ]
            raise ValueError(
                f"{side} trajectory exceeds RM65 joint limits after tool-roll offset: {details}"
            )
    fps = float(stable["fps"])
    output = args.output_dir / "rm65_synchronized_state.npz"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {key: stable[key] for key in stable.files}
    payload.update(
        left_q=left_q,
        right_q=right_q,
        left_target_xyz=targets["left"],
        right_target_xyz=targets["right"],
    )
    np.savez_compressed(output, **payload)
    manifest = {
        "schema_version": "phiagent-rm65-depth-regularized-state/1.0",
        "stable_state": str(args.stable_state),
        "ray_state": str(args.ray_state),
        "blend": args.blend,
        "target_z_offset_m": args.target_z_offset,
        "left_target_z_offset_m": args.left_target_z_offset,
        "right_target_z_offset_m": args.right_target_z_offset,
        "eef_z_keyframes": str(args.eef_z_keyframes) if args.eef_z_keyframes else None,
        "z_range_m": args.z_range,
        "frames": int(len(left_q)),
        "fps": fps,
        "base_poses": {
            "left_xyz_rpy": [*args.left_base, *args.left_base_rpy],
            "right_xyz_rpy": [*args.right_base, *args.right_base_rpy],
        },
        "ik_position_error_m": {
            "left_mean": float(left_pos.mean()), "left_max": float(left_pos.max()),
            "right_mean": float(right_pos.mean()), "right_max": float(right_pos.max()),
        },
        "ik_orientation_error_deg": {
            "left_mean": float(np.degrees(left_rot).mean()),
            "left_max": float(np.degrees(left_rot).max()),
            "right_mean": float(np.degrees(right_rot).mean()),
            "right_max": float(np.degrees(right_rot).max()),
        },
        "orientation_mode": args.orientation_mode,
        "rotation_weight": {
            "left": (
                args.rotation_weight
                if args.left_rotation_weight is None
                else args.left_rotation_weight
            ),
            "right": (
                args.rotation_weight
                if args.right_rotation_weight is None
                else args.right_rotation_weight
            ),
        },
        "tool_roll_offset_deg": {
            "left": args.left_tool_roll_offset_deg,
            "right": args.right_tool_roll_offset_deg,
        },
        "periodic_joint_canonicalization": {
            "left": left_periodic_manifest,
            "right": right_periodic_manifest,
        },
        "screen_width_roll_fit": screen_width_roll_manifest,
        "joint_anchor_file": str(args.joint_anchor_file) if args.joint_anchor_file else None,
        "rgb_gripper_axis_constraint": {
            "camera": args.camera,
            "input": screen_axis_manifest,
            "camera_depth_resolution": (
                "analytic intersection with world XY/table plane"
                if args.project_gripper_axis_to_table_plane
                else "provided monocular regularizer; not observed"
            ),
            "target_axis_abs_world_z_max": {
                "left": (
                    float(np.abs(left_rotations[:, 2, 2]).max())
                    if left_rotations is not None
                    else None
                ),
                "right": (
                    float(np.abs(right_rotations[:, 2, 2]).max())
                    if right_rotations is not None
                    else None
                ),
            },
            "semantics": (
                "source-visible wrist-to-tip image axis with an explicit monocular "
                "depth assumption"
            ),
        },
        "temporal": {
            "left": temporal_metrics(left_q, fps),
            "right": temporal_metrics(right_q, fps),
        },
        "state_npz": str(output),
        "claim_boundary": (
            "source-conditioned monocular visual replay; depth is regularized, "
            "not measured metric ground truth"
        ),
    }
    (args.output_dir / "state_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
