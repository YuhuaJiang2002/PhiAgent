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
    parser.add_argument("--left-target-z-offset", type=float, help="Override the common EEF Z offset for the left arm.")
    parser.add_argument("--right-target-z-offset", type=float, help="Override the common EEF Z offset for the right arm.")
    parser.add_argument(
        "--eef-z-keyframes",
        type=Path,
        help="Reviewed frame/left_z/right_z anchors that override monocular EEF height while retaining its XY path.",
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
        help="Reviewed frame-wise left/right wrist-to-tip image axes; overrides constant screen axes.",
    )
    parser.add_argument(
        "--joint-anchor-file",
        type=Path,
        help="Multistart IK branch anchors used as a temporal posture prior.",
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


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.blend <= 1.0:
        raise ValueError("--blend must be in [0, 1]")
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
    screen_axis_manifest: dict[str, object] | None = None
    if args.screen_axis_keyframes:
        keyframes = json.loads(args.screen_axis_keyframes.read_text())
        keyframe_indices = np.asarray(keyframes["frames"], dtype=int)
        if keyframe_indices[0] != 0 or keyframe_indices[-1] != len(targets["left"]) - 1:
            raise ValueError("screen-axis keyframes must include the first and final state frames")
        if np.any(np.diff(keyframe_indices) <= 0):
            raise ValueError("screen-axis keyframe indices must be strictly increasing")
        frame_axis: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            values = np.asarray(keyframes[f"{side}_dx_dy_view_depth"], dtype=np.float64)
            if values.shape != (len(keyframe_indices), 3):
                raise ValueError(f"{side} screen-axis values must have shape ({len(keyframe_indices)}, 3)")
            frame_axis[side] = np.column_stack([
                np.interp(np.arange(len(targets[side])), keyframe_indices, values[:, component])
                for component in range(3)
            ])
        left_rotations = observed_axis_rotations(model, len(targets["left"]), args.camera, frame_axis["left"])
        right_rotations = observed_axis_rotations(model, len(targets["right"]), args.camera, frame_axis["right"])
        screen_axis_manifest = {
            "keyframes": str(args.screen_axis_keyframes),
            "frames": keyframe_indices.tolist(),
        }
    elif args.left_screen_axis:
        left_rotations = observed_axis_rotations(
            model, len(targets["left"]), args.camera, args.left_screen_axis
        )
        right_rotations = observed_axis_rotations(
            model, len(targets["right"]), args.camera, args.right_screen_axis
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
        args.rotation_weight, np.asarray(args.left_seed), args.posture_weight,
        target_rotations=left_rotations,
        q_reference=left_q_reference,
        orientation_mode=args.orientation_mode,
    )
    right_q, right_pos, right_rot = _recover_q(
        model, targets["right"], "right", np.asarray(args.right_base),
        args.rotation_weight, np.asarray(args.right_seed), args.posture_weight,
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
    for side, q in (("left", left_q), ("right", right_q)):
        _, dofs = _joint_dofs(model, side)
        limits = np.asarray([model.jnt_range[model.dof_jntid[dof]] for dof in dofs])
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            raise ValueError(f"{side} trajectory exceeds RM65 joint limits after tool-roll offset")
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
            "left_mean": float(np.degrees(left_rot).mean()), "left_max": float(np.degrees(left_rot).max()),
            "right_mean": float(np.degrees(right_rot).mean()), "right_max": float(np.degrees(right_rot).max()),
        },
        "orientation_mode": args.orientation_mode,
        "tool_roll_offset_deg": {
            "left": args.left_tool_roll_offset_deg,
            "right": args.right_tool_roll_offset_deg,
        },
        "joint_anchor_file": str(args.joint_anchor_file) if args.joint_anchor_file else None,
        "rgb_gripper_axis_constraint": {
            "camera": args.camera,
            "input": screen_axis_manifest,
            "semantics": "source-visible wrist-to-tip axis; camera-depth component is regularized, not observed",
        },
        "temporal": {
            "left": temporal_metrics(left_q, fps),
            "right": temporal_metrics(right_q, fps),
        },
        "state_npz": str(output),
        "claim_boundary": "source-conditioned monocular visual replay; depth is regularized, not measured metric ground truth",
    }
    (args.output_dir / "state_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
