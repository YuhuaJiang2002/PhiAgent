#!/usr/bin/env python3
"""Audit FK, timing and image-plane alignment of an RM65 visual replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from render_realman_rm65_visual_replay import _joint_dofs, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-npz", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--comparison-video", type=Path, required=True)
    parser.add_argument("--rm65-urdf", type=Path, required=True)
    parser.add_argument("--ag2f90c-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-base", nargs=3, type=float, required=True)
    parser.add_argument("--right-base", nargs=3, type=float, required=True)
    parser.add_argument("--left-base-rpy", nargs=3, type=float, required=True)
    parser.add_argument("--right-base-rpy", nargs=3, type=float, required=True)
    parser.add_argument("--table-half-size", nargs=2, type=float, default=(0.55, 0.35))
    parser.add_argument("--table-center-y", type=float, default=0.10)
    parser.add_argument("--table-top-z", type=float, default=0.014)
    parser.add_argument("--camera", nargs=6, type=float, required=True)
    return parser.parse_args()


def video_metadata(path: Path) -> dict[str, float | int | str]:
    capture = cv2.VideoCapture(str(path))
    result = {
        "path": str(path),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return result


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    state = np.load(args.state_npz)
    model = build_model(
        args.rm65_urdf, args.ag2f90c_dir, False,
        tuple(args.left_base), args.left_base_rpy[2],
        tuple(args.right_base), args.right_base_rpy[2],
        args.left_base_rpy[0], args.left_base_rpy[1],
        args.right_base_rpy[0], args.right_base_rpy[1],
        tuple(args.table_half_size), args.table_center_y,
    )
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    azimuth, elevation, distance, look_x, look_y, look_z = args.camera
    camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
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

    def project(xyz: np.ndarray) -> np.ndarray:
        offset = xyz - camera_position
        depth = offset @ forward
        x = 512.0 + 512.0 * (offset @ right) / (depth * tan_x)
        y = 384.0 - 384.0 * (offset @ up) / (depth * tan_y)
        return np.column_stack((x, y))

    left_qpos, _ = _joint_dofs(model, "left")
    right_qpos, _ = _joint_dofs(model, "right")
    left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_eef")
    right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_eef")
    actual = {"left": [], "right": []}
    for left_q, right_q in zip(state["left_q"], state["right_q"]):
        data.qpos[left_qpos] = left_q
        data.qpos[right_qpos] = right_q
        mujoco.mj_forward(model, data)
        actual["left"].append(data.site_xpos[left_site].copy())
        actual["right"].append(data.site_xpos[right_site].copy())
    actual = {side: np.asarray(xyz) for side, xyz in actual.items()}

    fps = float(state["fps"])
    metrics: dict[str, object] = {
        "schema_version": "phiagent-rm65-replay-audit/1.0",
        "state": str(args.state_npz),
        "source_video": video_metadata(args.source_video),
        "comparison_video": video_metadata(args.comparison_video),
        "finite_state": bool(all(np.isfinite(state[key]).all() for key in ("left_q", "right_q", "left_gripper_command", "right_gripper_command"))),
        "camera": args.camera,
        "eef_fk_residual_m": {},
        "eef_source_projection_error_px": {},
        "eef_table_clearance_m": {},
        "event_eef_table_clearance_m": {},
        "temporal": {},
        "gripper": {},
    }
    projection_errors = []
    for side in ("left", "right"):
        residual = np.linalg.norm(actual[side] - state[f"{side}_target_xyz"], axis=1)
        pixel_error = np.linalg.norm(project(actual[side]) - state[f"{side}_tip_px"], axis=1)
        clearance = actual[side][:, 2] - args.table_top_z
        projection_errors.append(pixel_error)
        metrics["eef_fk_residual_m"][side] = distribution(residual)
        metrics["eef_source_projection_error_px"][side] = distribution(pixel_error)
        metrics["eef_table_clearance_m"][side] = distribution(clearance)
        q = state[f"{side}_q"]
        velocity = np.diff(q, axis=0) * fps
        acceleration = np.diff(velocity, axis=0) * fps
        metrics["temporal"][side] = {
            "max_frame_joint_delta_rad": float(np.abs(np.diff(q, axis=0)).max()),
            "max_joint_velocity_rad_s": float(np.abs(velocity).max()),
            "max_joint_acceleration_rad_s2": float(np.abs(acceleration).max()),
        }
        command = state[f"{side}_gripper_command"]
        crossings = np.flatnonzero(np.diff(command >= 0.5)) + 1
        metrics["gripper"][side] = {
            "range": [float(command.min()), float(command.max())],
            "half_command_crossing_frames": crossings.tolist(),
        }
        reviewed_frames = (0, 32, 40, 64, 71, 96, 104, 112, 150, 157, 191)
        metrics["event_eef_table_clearance_m"][side] = {
            str(frame): float(clearance[frame]) for frame in reviewed_frames
        }
    combined = np.concatenate(projection_errors)
    metrics["eef_source_projection_error_px"]["combined"] = distribution(combined)
    metrics["claim_boundary"] = "2-D source-conditioned FK replay audit; source pixels are reviewed RGB observations, not encoder/calibration ground truth"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
