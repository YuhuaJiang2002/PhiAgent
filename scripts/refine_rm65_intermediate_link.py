#!/usr/bin/env python3
"""Refine an RM65 wrist configuration under source-visible 2-D constraints.

The endpoint of a monocular replay does not uniquely determine the visible
intermediate-link configuration.  This stage changes one reviewed joint branch
while preserving the projected EEF position, gripper longitudinal/width axes,
and upstream joint centres.  It is a visual refinement, not encoder recovery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from render_realman_rm65_visual_replay import _joint_dofs, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-npz", type=Path, required=True)
    parser.add_argument("--rm65-urdf", type=Path, required=True)
    parser.add_argument("--ag2f90c-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--joint", type=int, choices=range(1, 7), default=4)
    parser.add_argument("--reviewed-offset-deg", type=float, required=True)
    parser.add_argument("--left-base", nargs=3, type=float, required=True)
    parser.add_argument("--right-base", nargs=3, type=float, required=True)
    parser.add_argument("--left-base-rpy", nargs=3, type=float, required=True)
    parser.add_argument("--right-base-rpy", nargs=3, type=float, required=True)
    parser.add_argument("--table-half-size", nargs=2, type=float, default=(0.55, 0.35))
    parser.add_argument("--table-center-y", type=float, default=0.10)
    parser.add_argument(
        "--camera",
        nargs=6,
        type=float,
        required=True,
        metavar=("AZIMUTH", "ELEVATION", "DISTANCE", "LOOK_X", "LOOK_Y", "LOOK_Z"),
    )
    parser.add_argument("--smooth-sigma", type=float, default=0.65)
    parser.add_argument(
        "--metric-eef-position-scale-m",
        type=float,
        default=0.005,
        help="Robustness scale for retaining the input EEF metric position.",
    )
    parser.add_argument(
        "--metric-eef-rotation-scale-deg",
        type=float,
        default=3.0,
        help="Robustness scale for retaining the input EEF metric orientation.",
    )
    return parser.parse_args()


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("cannot normalize a near-zero projected axis")
    return vector / norm


def _angle_deg(first: np.ndarray, second: np.ndarray, undirected: bool = False) -> float:
    dot = float(np.dot(_unit(first), _unit(second)))
    if undirected:
        dot = abs(dot)
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


class ImageObservation:
    """Project MuJoCo sites, body origins and local axes into the review camera."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_values: list[float],
        width: int = 1024,
        height: int = 768,
    ) -> None:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.azimuth, camera.elevation, camera.distance = camera_values[:3]
        camera.lookat[:] = camera_values[3:]
        data = mujoco.MjData(model)
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera=camera)
            scene_camera = renderer.scene.camera[0]
            self.position = np.asarray(scene_camera.pos, dtype=np.float64).copy()
            self.forward = np.asarray(scene_camera.forward, dtype=np.float64).copy()
            self.up = np.asarray(scene_camera.up, dtype=np.float64).copy()
            self.right = _unit(np.cross(self.forward, self.up))
            self.tan_y = float(scene_camera.frustum_top / scene_camera.frustum_near)
        self.width = width
        self.height = height
        self.tan_x = self.tan_y * width / height

    def project(self, xyz: np.ndarray) -> np.ndarray:
        offset = np.asarray(xyz, dtype=np.float64) - self.position
        depth = float(np.dot(offset, self.forward))
        if depth <= 1e-8:
            raise ValueError("point lies behind the review camera")
        return np.asarray(
            (
                self.width / 2 + self.width / 2 * np.dot(offset, self.right) / (depth * self.tan_x),
                self.height / 2 - self.height / 2 * np.dot(offset, self.up) / (depth * self.tan_y),
            )
        )


def _observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: ImageObservation,
    side: str,
    site_id: int,
) -> dict[str, np.ndarray]:
    position = data.site_xpos[site_id].copy()
    rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    eef_px = camera.project(position)
    longitudinal = _unit(camera.project(position + 0.12 * rotation[:, 2]) - eef_px)
    width = _unit(camera.project(position + 0.08 * rotation[:, 1]) - eef_px)
    joint_pixels = []
    for joint in (3, 4):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint_{joint}"
        )
        joint_pixels.append(camera.project(data.xpos[model.jnt_bodyid[joint_id]]))
    return {
        "eef_px": eef_px,
        "longitudinal": longitudinal,
        "width": width,
        "joint_pixels": np.asarray(joint_pixels),
        "eef_xyz": position,
        "eef_rotation": rotation,
    }


def _trajectory_metrics(q: np.ndarray, fps: float) -> dict[str, float]:
    velocity = np.diff(q, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    return {
        "max_frame_joint_delta_rad": float(np.abs(np.diff(q, axis=0)).max()),
        "max_joint_velocity_rad_s": float(np.abs(velocity).max()),
        "max_joint_acceleration_rad_s2": float(np.abs(acceleration).max()),
    }


def main() -> None:
    args = parse_args()
    if args.smooth_sigma < 0:
        raise ValueError("--smooth-sigma must be non-negative")
    if args.metric_eef_position_scale_m <= 0:
        raise ValueError("--metric-eef-position-scale-m must be positive")
    if args.metric_eef_rotation_scale_deg <= 0:
        raise ValueError("--metric-eef-rotation-scale-deg must be positive")
    with np.load(args.state_npz) as loaded:
        payload = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
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
    data = mujoco.MjData(model)
    qpos, dofs = _joint_dofs(model, args.side)
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"{args.side}_eef"
    )
    camera = ImageObservation(model, args.camera)
    base_q = np.asarray(payload[f"{args.side}_q"], dtype=np.float64)
    limits = np.asarray(
        [model.jnt_range[model.dof_jntid[dof]] for dof in dofs], dtype=np.float64
    )
    joint_index = args.joint - 1
    reviewed_target = np.clip(
        base_q[:, joint_index] + np.deg2rad(args.reviewed_offset_deg),
        limits[joint_index, 0] + 1e-5,
        limits[joint_index, 1] - 1e-5,
    )

    base_observations: list[dict[str, np.ndarray]] = []
    for q in base_q:
        data.qpos[qpos] = q
        mujoco.mj_forward(model, data)
        base_observations.append(_observation(model, data, camera, args.side, site_id))

    refined: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for frame, (reference_q, target) in enumerate(zip(base_q, base_observations)):
        expected_motion = np.zeros(6) if frame == 0 else reference_q - base_q[frame - 1]
        temporal_target = reference_q if previous is None else previous + expected_motion

        def residual(candidate: np.ndarray) -> np.ndarray:
            data.qpos[qpos] = candidate
            mujoco.mj_forward(model, data)
            current = _observation(model, data, camera, args.side, site_id)
            metric_rotation_delta = Rotation.from_matrix(
                current["eef_rotation"] @ target["eef_rotation"].T
            ).as_rotvec()
            return np.concatenate(
                (
                    (current["eef_px"] - target["eef_px"]) / 3.0,
                    (current["longitudinal"] - target["longitudinal"]) / 0.02,
                    (current["width"] - target["width"]) / 0.02,
                    (current["joint_pixels"] - target["joint_pixels"]).ravel() / 4.0,
                    np.asarray(((candidate[joint_index] - reviewed_target[frame]) / 0.03,)),
                    (candidate[:3] - reference_q[:3]) / 0.12,
                    (candidate - temporal_target) / 0.30,
                    (current["eef_xyz"] - target["eef_xyz"])
                    / args.metric_eef_position_scale_m,
                    metric_rotation_delta
                    / np.deg2rad(args.metric_eef_rotation_scale_deg),
                )
            )

        initial = reference_q.copy() if previous is None else temporal_target.copy()
        initial[joint_index] = reviewed_target[frame]
        solution = least_squares(
            residual,
            np.clip(initial, limits[:, 0] + 1e-6, limits[:, 1] - 1e-6),
            bounds=(limits[:, 0] + 1e-7, limits[:, 1] - 1e-7),
            max_nfev=500,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        ).x
        refined.append(solution)
        previous = solution

    refined_q = np.asarray(refined)
    if args.smooth_sigma:
        refined_q = gaussian_filter1d(
            refined_q, sigma=args.smooth_sigma, axis=0, mode="nearest"
        )
        refined_q = np.clip(refined_q, limits[:, 0], limits[:, 1])

    eef_px_delta = []
    longitudinal_error = []
    width_error = []
    joint_pixel_delta = []
    eef_xyz_delta = []
    eef_rotation_delta = []
    final_xyz = []
    for q, target in zip(refined_q, base_observations):
        data.qpos[qpos] = q
        mujoco.mj_forward(model, data)
        current = _observation(model, data, camera, args.side, site_id)
        eef_px_delta.append(np.linalg.norm(current["eef_px"] - target["eef_px"]))
        longitudinal_error.append(
            _angle_deg(current["longitudinal"], target["longitudinal"])
        )
        width_error.append(_angle_deg(current["width"], target["width"], undirected=True))
        joint_pixel_delta.append(
            np.linalg.norm(current["joint_pixels"] - target["joint_pixels"], axis=1)
        )
        eef_xyz_delta.append(np.linalg.norm(current["eef_xyz"] - target["eef_xyz"]))
        eef_rotation_delta.append(
            np.degrees(
                np.linalg.norm(
                    Rotation.from_matrix(
                        current["eef_rotation"] @ target["eef_rotation"].T
                    ).as_rotvec()
                )
            )
        )
        final_xyz.append(current["eef_xyz"])

    payload[f"{args.side}_q"] = refined_q
    payload[f"{args.side}_target_xyz"] = np.asarray(final_xyz)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "rm65_synchronized_state.npz"
    np.savez_compressed(output, **payload)

    def distribution(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "p90": float(np.percentile(values, 90)),
            "max": float(values.max()),
        }

    achieved = np.degrees(refined_q[:, joint_index] - base_q[:, joint_index])
    manifest = {
        "schema_version": "phiagent-rm65-intermediate-link-refinement/1.0",
        "input_state": str(args.state_npz),
        "output_state": str(output),
        "side": args.side,
        "reviewed_joint_1based": args.joint,
        "requested_offset_deg": args.reviewed_offset_deg,
        "metric_regularization": {
            "eef_position_scale_m": args.metric_eef_position_scale_m,
            "eef_rotation_scale_deg": args.metric_eef_rotation_scale_deg,
        },
        "achieved_offset_deg": distribution(achieved),
        "limit_clamped_frames": int(
            np.count_nonzero(
                np.isclose(reviewed_target, limits[joint_index, 0] + 1e-5)
                | np.isclose(reviewed_target, limits[joint_index, 1] - 1e-5)
            )
        ),
        "projected_eef_delta_px": distribution(np.asarray(eef_px_delta)),
        "projected_longitudinal_axis_delta_deg": distribution(
            np.asarray(longitudinal_error)
        ),
        "projected_width_axis_delta_deg": distribution(np.asarray(width_error)),
        "projected_joint_3_4_delta_px": {
            "joint_3": distribution(np.asarray(joint_pixel_delta)[:, 0]),
            "joint_4": distribution(np.asarray(joint_pixel_delta)[:, 1]),
        },
        "metric_eef_delta_from_v21_m": distribution(np.asarray(eef_xyz_delta)),
        "eef_rotation_delta_from_v21_deg": distribution(
            np.asarray(eef_rotation_delta)
        ),
        "temporal": _trajectory_metrics(refined_q, float(payload["fps"])),
        "camera": args.camera,
        "claim_boundary": (
            "reviewed image-space intermediate-link refinement; preserves projected "
            "gripper cues but does not recover calibrated metric 6-D ground truth"
        ),
    }
    (args.output_dir / "state_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
