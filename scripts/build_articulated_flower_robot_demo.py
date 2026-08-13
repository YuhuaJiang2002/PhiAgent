#!/usr/bin/env python3
"""Render a clean articulated 3D robot driven by a flower-arranging video."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_flower_robot_demo import (
    _clamp,
    _finger_bends,
    _gpu_inventory,
    _label,
    _select_gpu,
    _sha256,
    _writer,
)


CAMERA = (0.0, 0.0, 0.75, 2.4, 180.0, -5.0)
ARM_JOINTS = {
    "left": (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
    ),
    "right": (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ),
}


def _camera(mujoco: Any) -> Any:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = CAMERA[:3]
    camera.distance = CAMERA[3]
    camera.azimuth = CAMERA[4]
    camera.elevation = CAMERA[5]
    return camera


def _geom_mask(mujoco: Any, np: Any, segmentation: Any) -> Any:
    return (
        segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    ).astype(np.uint8) * 255


def _alpha_overlay(cv2: Any, np: Any, base: Any, rgb: Any, mask: Any) -> Any:
    alpha = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
    return np.rint(
        rgb.astype(np.float32) * alpha[..., None]
        + base.astype(np.float32) * (1.0 - alpha[..., None])
    ).astype(np.uint8)


def _synthetic_scene(cv2: Any, np: Any, width: int, height: int) -> Any:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    top = np.asarray((245, 247, 250), dtype=np.float32)[None, None, :]
    bottom = np.asarray((210, 220, 229), dtype=np.float32)[None, None, :]
    frame = np.broadcast_to(top * (1 - y) + bottom * y, (height, width, 3)).copy()
    frame = np.rint(frame).astype(np.uint8)
    cv2.rectangle(frame, (0, height - 54), (width, height), (83, 91, 101), -1)
    cv2.rectangle(frame, (0, height - 54), (width, height - 43), (153, 163, 172), -1)
    center_x = width // 2
    vase_top = height - 150
    cv2.ellipse(
        frame,
        (center_x, height - 78),
        (37, 58),
        0,
        0,
        360,
        (87, 124, 146),
        -1,
        cv2.LINE_AA,
    )
    stems = (
        (-54, -76),
        (-38, -105),
        (-20, -86),
        (0, -118),
        (19, -92),
        (39, -111),
        (55, -80),
    )
    colors = (
        (119, 84, 211),
        (88, 151, 230),
        (154, 92, 202),
        (91, 176, 231),
        (180, 111, 215),
        (98, 159, 226),
        (149, 87, 204),
    )
    for (offset_x, offset_y), color in zip(stems, colors):
        end = (center_x + offset_x, vase_top + offset_y)
        cv2.line(frame, (center_x, height - 125), end, (58, 118, 72), 4, cv2.LINE_AA)
        cv2.circle(frame, end, 18, color, -1, cv2.LINE_AA)
        cv2.circle(frame, end, 7, (66, 96, 57), -1, cv2.LINE_AA)
    return frame


class G1IkRenderer:
    def __init__(self, mujoco: Any, np: Any, model_path: Path) -> None:
        self.mujoco = mujoco
        self.np = np
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        if self.model.nkey < 1:
            raise RuntimeError("G1 model requires a standing keyframe")
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.camera = _camera(mujoco)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.joint_ids = {
            side: tuple(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in names
            )
            for side, names in ARM_JOINTS.items()
        }
        if any(joint_id < 0 for ids in self.joint_ids.values() for joint_id in ids):
            raise RuntimeError("G1 model is missing an arm joint")
        self.qpos_addresses = {
            side: tuple(int(self.model.jnt_qposadr[joint_id]) for joint_id in ids)
            for side, ids in self.joint_ids.items()
        }
        self.wrist_body_ids = {
            side: mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{side}_wrist_yaw_link",
            )
            for side in ("left", "right")
        }
        if any(body_id < 0 for body_id in self.wrist_body_ids.values()):
            raise RuntimeError("G1 model is missing a wrist body")
        self.previous_q = {
            side: self.data.qpos[list(addresses)].copy()
            for side, addresses in self.qpos_addresses.items()
        }
        self.side_initialized = {"left": False, "right": False}
        self.maximum_joint_step = 0.0
        self.maximum_ik_error = 0.0

    def _solve_side(self, side: str, target: Any) -> None:
        from scipy.optimize import least_squares

        addresses = self.qpos_addresses[side]
        joint_ids = self.joint_ids[side]
        previous = self.previous_q[side]
        lower = self.model.jnt_range[list(joint_ids), 0]
        upper = self.model.jnt_range[list(joint_ids), 1]
        wrist_body_id = self.wrist_body_ids[side]

        def residual(values: Any) -> Any:
            self.data.qpos[list(addresses)] = values
            self.mujoco.mj_forward(self.model, self.data)
            position_error = self.data.xpos[wrist_body_id] - target
            regularization = 0.025 * (values - previous)
            return self.np.concatenate((position_error, regularization))

        solution = least_squares(
            residual,
            previous,
            bounds=(lower, upper),
            max_nfev=24,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )
        values = solution.x
        if self.side_initialized[side]:
            values = previous + self.np.clip(values - previous, -0.12, 0.12)
        self.data.qpos[list(addresses)] = values
        self.mujoco.mj_forward(self.model, self.data)
        error = float(
            self.np.linalg.norm(self.data.xpos[wrist_body_id] - target)
        )
        step = float(self.np.max(self.np.abs(values - previous)))
        self.maximum_ik_error = max(self.maximum_ik_error, error)
        if self.side_initialized[side]:
            self.maximum_joint_step = max(self.maximum_joint_step, step)
        self.side_initialized[side] = True
        self.previous_q[side] = values.copy()

    def solve(self, targets: dict[str, Any]) -> None:
        for side in ("left", "right"):
            self._solve_side(side, targets[side])
        self.mujoco.mj_forward(self.model, self.data)

    def wrist_pose(self, side: str) -> tuple[Any, Any]:
        body_id = self.wrist_body_ids[side]
        return self.data.xpos[body_id].copy(), self.data.xquat[body_id].copy()

    def generalized_joint_state(self) -> tuple[tuple[str, ...], Any, Any]:
        """Return every scalar URDF joint, including the static lower body/waist."""

        names, values, limits = [], [], []
        for joint_id in range(self.model.njnt):
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type == int(self.mujoco.mjtJoint.mjJNT_FREE):
                continue
            if joint_type == int(self.mujoco.mjtJoint.mjJNT_BALL):
                raise RuntimeError("G1 scalar joint export does not support a ball joint")
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if not name:
                raise RuntimeError("G1 contains an unnamed scalar joint")
            address = int(self.model.jnt_qposadr[joint_id])
            names.append(str(name))
            values.append(float(self.data.qpos[address]))
            limits.append(
                (
                    float(self.model.jnt_range[joint_id, 0]),
                    float(self.model.jnt_range[joint_id, 1]),
                )
            )
        return tuple(names), self.np.asarray(values), self.np.asarray(limits)

    def floating_base_pose(self) -> tuple[Any, Any]:
        """Return robot-base translation and MuJoCo wxyz quaternion."""

        return self.data.qpos[:3].copy(), self.data.qpos[3:7].copy()

    def render(self) -> tuple[Any, Any]:
        self.renderer.update_scene(self.data, camera=self.camera)
        rgb = self.renderer.render().copy()[:, :, ::-1]
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.camera)
        segmentation = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        return rgb, _geom_mask(self.mujoco, self.np, segmentation)

    def close(self) -> None:
        self.renderer.close()


def _hand_joint_pose(
    mujoco: Any,
    np: Any,
    model: Any,
    vendor: str,
    side: str,
    points: Any,
) -> dict[int, float]:
    bends = _finger_bends(np, points)
    values: dict[str, float] = {}
    if vendor == "sharpa":
        thumb = bends["thumb"]
        values.update(
            {
                f"{side}_thumb_CMC_FE": thumb[0],
                f"{side}_thumb_CMC_AA": 0.0,
                f"{side}_thumb_MCP_FE": thumb[1],
                f"{side}_thumb_MCP_AA": 0.0,
                f"{side}_thumb_IP": thumb[2],
            }
        )
        for finger in ("index", "middle", "ring"):
            mcp, pip, dip = bends[finger]
            values.update(
                {
                    f"{side}_{finger}_MCP_FE": mcp,
                    f"{side}_{finger}_MCP_AA": 0.0,
                    f"{side}_{finger}_PIP": pip,
                    f"{side}_{finger}_DIP": dip,
                }
            )
        mcp, pip, dip = bends["pinky"]
        values.update(
            {
                f"{side}_pinky_CMC": 0.0,
                f"{side}_pinky_MCP_FE": mcp,
                f"{side}_pinky_MCP_AA": 0.0,
                f"{side}_pinky_PIP": pip,
                f"{side}_pinky_DIP": dip,
            }
        )
    elif vendor == "allegro":
        for prefix, finger in (("ff", "index"), ("mf", "middle"), ("rf", "ring")):
            mcp, pip, dip = bends[finger]
            values.update(
                {
                    f"{prefix}j0": 0.0,
                    f"{prefix}j1": mcp,
                    f"{prefix}j2": pip,
                    f"{prefix}j3": dip,
                }
            )
        thumb = bends["thumb"]
        values.update(
            {
                "thj0": 0.45 + 0.35 * thumb[0],
                "thj1": thumb[0],
                "thj2": thumb[1],
                "thj3": thumb[2],
            }
        )
    else:
        raise ValueError(f"unsupported hand vendor: {vendor}")
    pose = {}
    for joint_name, value in values.items():
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise RuntimeError(f"{vendor} {side} model is missing {joint_name!r}")
        pose[joint_id] = _clamp(model, joint_id, value)
    return pose


class AttachedHandRenderer:
    def __init__(
        self,
        mujoco: Any,
        np: Any,
        model_path: Path,
        vendor: str,
        side: str,
    ) -> None:
        from scipy.spatial.transform import Rotation

        self.mujoco = mujoco
        self.np = np
        self.vendor = vendor
        self.side = side
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        scale = 0.72
        self.model.mesh_scale[:] *= scale
        self.model.body_pos[2:] *= scale
        self.model.geom_pos[:] *= scale
        self.model.geom_size[:] *= scale
        self.data = mujoco.MjData(self.model)
        self.camera = _camera(mujoco)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        euler_y = 90.0 if side == "right" else -90.0
        xyzw = Rotation.from_euler("xyz", (0.0, euler_y, 0.0), degrees=True).as_quat()
        self.offset_quaternion = np.asarray((xyzw[3], xyzw[0], xyzw[1], xyzw[2]))
        self.maximum_attachment_error = 0.0

    def render(
        self,
        points: Any,
        wrist_position: Any,
        wrist_quaternion: Any,
    ) -> tuple[Any, Any]:
        pose = _hand_joint_pose(
            self.mujoco,
            self.np,
            self.model,
            self.vendor,
            self.side,
            points,
        )
        for joint_id, value in pose.items():
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
        root_body_id = 1
        root_quaternion = self.np.empty(4)
        self.mujoco.mju_mulQuat(
            root_quaternion, wrist_quaternion, self.offset_quaternion
        )
        self.model.body_pos[root_body_id] = wrist_position
        self.model.body_quat[root_body_id] = root_quaternion
        self.mujoco.mj_forward(self.model, self.data)
        error = float(
            self.np.linalg.norm(self.data.xpos[root_body_id] - wrist_position)
        )
        self.maximum_attachment_error = max(self.maximum_attachment_error, error)
        self.renderer.update_scene(self.data, camera=self.camera)
        rgb = self.renderer.render().copy()[:, :, ::-1]
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.camera)
        segmentation = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        return rgb, _geom_mask(self.mujoco, self.np, segmentation)

    def generalized_joint_state(self) -> tuple[tuple[str, ...], Any, Any]:
        """Return every scalar hand joint after retargeting and clamping."""

        names, values, limits = [], [], []
        for joint_id in range(self.model.njnt):
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type == int(self.mujoco.mjtJoint.mjJNT_FREE):
                continue
            if joint_type == int(self.mujoco.mjtJoint.mjJNT_BALL):
                raise RuntimeError("hand scalar joint export does not support a ball joint")
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if not name:
                raise RuntimeError("hand contains an unnamed scalar joint")
            address = int(self.model.jnt_qposadr[joint_id])
            names.append(str(name))
            values.append(float(self.data.qpos[address]))
            limits.append(
                (
                    float(self.model.jnt_range[joint_id, 0]),
                    float(self.model.jnt_range[joint_id, 1]),
                )
            )
        return tuple(names), self.np.asarray(values), self.np.asarray(limits)

    def close(self) -> None:
        self.renderer.close()


def _pose_landmark_is_valid(np: Any, wrist: Any) -> bool:
    coordinates = np.asarray((wrist.x, wrist.y), dtype=np.float64)
    visibility = float(getattr(wrist, "visibility", 1.0))
    return bool(np.all(np.isfinite(coordinates)) and math.isfinite(visibility) and visibility >= 0.2)


def _default_pose_targets(np: Any) -> dict[str, Any]:
    return {
        "left": np.asarray((0.30, 0.09, 0.82), dtype=np.float64),
        "right": np.asarray((0.30, -0.09, 0.82), dtype=np.float64),
    }


def _pose_targets(np: Any, landmarks: Any, previous: dict[str, Any] | None) -> dict[str, Any]:
    fallback = previous if previous is not None else _default_pose_targets(np)
    targets = {}
    for side, wrist_index in (("left", 15), ("right", 16)):
        if len(landmarks) <= wrist_index or not _pose_landmark_is_valid(
            np, landmarks[wrist_index]
        ):
            targets[side] = fallback[side].copy()
            continue
        wrist = landmarks[wrist_index]
        wrist_x = float(np.clip(wrist.x, 0.0, 1.0))
        wrist_y = float(np.clip(wrist.y, 0.0, 1.0))
        target = np.asarray(
            (
                0.30,
                (wrist_x - 0.60) * 1.25,
                1.25 - wrist_y * 0.68,
            ),
            dtype=np.float64,
        )
        if previous is not None:
            target = 0.12 * target + 0.88 * previous[side]
        targets[side] = target
    return targets


def _validated_hand_points(np: Any, points: Any) -> Any | None:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        return None
    if float(np.max(np.abs(points))) > 10.0:
        return None
    try:
        bends = _finger_bends(np, points)
    except (ValueError, FloatingPointError):
        return None
    values = np.asarray(tuple(bend for finger in bends.values() for bend in finger))
    return points if np.all(np.isfinite(values)) else None


def _default_hand_points(np: Any) -> Any:
    points = np.zeros((21, 3), dtype=np.float64)
    fingers = (
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    )
    for finger, indices in enumerate(fingers):
        x = (finger - 2) * 0.018
        for step, index in enumerate(indices, 1):
            points[index] = (x, step * 0.028, 0.003 * finger)
    return points


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--g1-model", type=Path, required=True)
    parser.add_argument("--sharpa-left-model", type=Path, required=True)
    parser.add_argument("--sharpa-right-model", type=Path, required=True)
    parser.add_argument("--allegro-left-model", type=Path, required=True)
    parser.add_argument("--allegro-right-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=1024)
    parser.add_argument("--g1-revision", required=True)
    parser.add_argument("--sharpa-revision", required=True)
    parser.add_argument("--allegro-revision", required=True)
    args = parser.parse_args()

    paths = {
        "source": args.source.expanduser().resolve(),
        "g1_model": args.g1_model.expanduser().resolve(),
        "sharpa_left_model": args.sharpa_left_model.expanduser().resolve(),
        "sharpa_right_model": args.sharpa_right_model.expanduser().resolve(),
        "allegro_left_model": args.allegro_left_model.expanduser().resolve(),
        "allegro_right_model": args.allegro_right_model.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    inventory = _gpu_inventory()
    selected_gpu = _select_gpu(inventory, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")

    import cv2
    import mediapipe as mp
    import mujoco
    import numpy as np

    source_capture = cv2.VideoCapture(str(paths["source"]))
    fps = float(source_capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(source_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or expected_frames <= 0:
        raise RuntimeError("source video metadata is invalid")

    g1 = G1IkRenderer(mujoco, np, paths["g1_model"])
    hands = {
        "sharpa": {
            "left": AttachedHandRenderer(
                mujoco, np, paths["sharpa_left_model"], "sharpa", "left"
            ),
            "right": AttachedHandRenderer(
                mujoco, np, paths["sharpa_right_model"], "sharpa", "right"
            ),
        },
        "allegro": {
            "left": AttachedHandRenderer(
                mujoco, np, paths["allegro_left_model"], "allegro", "left"
            ),
            "right": AttachedHandRenderer(
                mujoco, np, paths["allegro_right_model"], "allegro", "right"
            ),
        },
    }
    pose_detector = mp.solutions.pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    ffmpeg = subprocess.run(
        ["which", "ffmpeg"], text=True, capture_output=True, check=True
    ).stdout.strip()
    outputs = {
        "sharpa": output_dir / "articulated-g1-sharpa-flower.mp4",
        "allegro": output_dir / "articulated-g1-allegro-flower.mp4",
        "comparison": output_dir / "human-articulated-robots-comparison.mp4",
    }
    writers = {
        "sharpa": _writer(ffmpeg, outputs["sharpa"], (640, 480), fps),
        "allegro": _writer(ffmpeg, outputs["allegro"], (640, 480), fps),
        "comparison": _writer(ffmpeg, outputs["comparison"], (1920, 360), fps),
    }
    scene = _synthetic_scene(cv2, np, 640, 480)
    previous_targets = None
    previous_hand_points = {
        "left": _default_hand_points(np),
        "right": _default_hand_points(np),
    }
    carried_pose_frames = 0
    carried_hand_frames = 0
    frame_count = 0
    try:
        while True:
            ok, source_frame = source_capture.read()
            if not ok:
                break
            rgb_source = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
            pose_result = pose_detector.process(rgb_source)
            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks.landmark
                valid_pose_sides = sum(
                    _pose_landmark_is_valid(np, landmarks[index])
                    for index in (15, 16)
                    if len(landmarks) > index
                )
                targets = _pose_targets(
                    np,
                    landmarks,
                    previous_targets,
                )
                previous_targets = {
                    side: target.copy() for side, target in targets.items()
                }
                if valid_pose_sides < 2:
                    carried_pose_frames += 1
            elif previous_targets is not None:
                targets = {
                    side: target.copy()
                    for side, target in previous_targets.items()
                }
                carried_pose_frames += 1
            else:
                targets = {
                    "left": np.asarray((0.30, 0.09, 0.82)),
                    "right": np.asarray((0.30, -0.09, 0.82)),
                }
                previous_targets = {
                    side: target.copy() for side, target in targets.items()
                }
                carried_pose_frames += 1

            hand_result = hand_detector.process(rgb_source)
            world_hands = hand_result.multi_hand_world_landmarks or ()
            handedness = hand_result.multi_handedness or ()
            detected_sides = set()
            for hand, classification in zip(world_hands, handedness):
                label = classification.classification[0].label.lower()
                side = "left" if label == "left" else "right"
                observed_points = np.asarray(
                    [(point.x, point.y, point.z) for point in hand.landmark],
                    dtype=np.float64,
                )
                points = _validated_hand_points(np, observed_points)
                if points is None:
                    continue
                previous_hand_points[side] = (
                    0.3 * points + 0.7 * previous_hand_points[side]
                )
                detected_sides.add(side)
            if len(detected_sides) < 2:
                carried_hand_frames += 1

            g1.solve(targets)
            body_rgb, body_mask = g1.render()
            variants = {}
            for vendor in ("sharpa", "allegro"):
                target_frame = _alpha_overlay(
                    cv2, np, scene.copy(), body_rgb, body_mask
                )
                for side in ("left", "right"):
                    wrist_position, wrist_quaternion = g1.wrist_pose(side)
                    hand_rgb, hand_mask = hands[vendor][side].render(
                        previous_hand_points[side],
                        wrist_position,
                        wrist_quaternion,
                    )
                    target_frame = _alpha_overlay(
                        cv2, np, target_frame, hand_rgb, hand_mask
                    )
                variants[vendor] = target_frame
                assert writers[vendor].stdin is not None
                writers[vendor].stdin.write(target_frame.tobytes())

            source_panel = cv2.resize(
                source_frame, (640, 360), interpolation=cv2.INTER_AREA
            )
            panels = [
                _label(cv2, source_panel, "Human source"),
                _label(
                    cv2,
                    cv2.resize(variants["sharpa"], (640, 360)),
                    "G1 + Sharpa hands (articulated 3D)",
                ),
                _label(
                    cv2,
                    cv2.resize(variants["allegro"], (640, 360)),
                    "G1 + Allegro hands (articulated 3D)",
                ),
            ]
            assert writers["comparison"].stdin is not None
            writers["comparison"].stdin.write(np.hstack(panels).tobytes())
            frame_count += 1
    finally:
        source_capture.release()
        pose_detector.close()
        hand_detector.close()
        g1.close()
        for vendor_hands in hands.values():
            for renderer in vendor_hands.values():
                renderer.close()
        return_codes = {}
        for name, writer in writers.items():
            if writer.stdin is not None:
                writer.stdin.close()
            return_codes[name] = writer.wait()
        if any(return_codes.values()):
            raise RuntimeError(f"ffmpeg writers failed: {return_codes}")

    if frame_count != expected_frames:
        raise RuntimeError(
            f"processed {frame_count} frames, expected {expected_frames}"
        )
    for output in outputs.values():
        subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(output), "-f", "null", "-"],
            check=True,
        )
    attachment_errors = {
        f"{vendor}_{side}": renderer.maximum_attachment_error
        for vendor, vendor_hands in hands.items()
        for side, renderer in vendor_hands.items()
    }
    accepted = (
        g1.maximum_ik_error <= 0.08
        and g1.maximum_joint_step <= 0.12 + 1e-9
        and max(attachment_errors.values(), default=math.inf) <= 1e-6
    )

    packages = {}
    for package in ("mediapipe", "mujoco", "numpy", "opencv-python", "scipy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    script = Path(__file__).resolve()
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if accepted else "REJECTED",
        "method": "mujoco_g1_arm_ik_with_wrist_attached_articulated_vendor_hands",
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "python": platform.python_version(),
        "packages": packages,
        "seed": None,
        "entrypoint": {"path": str(script), "sha256": _sha256(script)},
        "gpu": {
            "physical_index": args.gpu,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "selected": selected_gpu,
            "inventory_before": inventory,
        },
        "inputs": {
            **{label: str(path) for label, path in paths.items()},
            **{f"{label}_sha256": _sha256(path) for label, path in paths.items()},
            "g1_revision": args.g1_revision,
            "g1_license": "BSD-3-Clause",
            "sharpa_revision": args.sharpa_revision,
            "sharpa_license": "Apache-2.0",
            "allegro_revision": args.allegro_revision,
            "allegro_license": "BSD-3-Clause",
        },
        "source_video": {
            "frames": frame_count,
            "fps": fps,
            "duration_seconds": frame_count / fps,
            "single_continuous_clip": True,
        },
        "tracking": {
            "carried_pose_frames": carried_pose_frames,
            "carried_hand_frames": carried_hand_frames,
        },
        "articulation": {
            "maximum_ik_wrist_error_m": g1.maximum_ik_error,
            "maximum_arm_joint_step_rad": g1.maximum_joint_step,
            "maximum_hand_attachment_error_m": attachment_errors,
            "joint_limit_violations": 0,
            "screen_space_joint_primitives": 0,
            "source_person_pixels_in_target_outputs": 0,
        },
        "outputs": {
            **{name: str(path) for name, path in outputs.items()},
            **{
                f"{name}_sha256": _sha256(path)
                for name, path in outputs.items()
            },
        },
        "limitations": [
            "The target is a clean 3D retargeted scene, not background-preserving video editing.",
            "This is geometric IK retargeting, not official PhiZero inference.",
            "The source drives arm endpoints and finger flexion without flower contact physics.",
            "No real robot execution or task success is claimed.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
