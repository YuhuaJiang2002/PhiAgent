#!/usr/bin/env python3
"""Generate a 660-frame flower manipulation with complete metric ground truth."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from build_articulated_flower_robot_demo import (  # noqa: E402
    AttachedHandRenderer,
    G1IkRenderer,
)
from build_flower_robot_demo import _gpu_inventory, _select_gpu  # noqa: E402
from phiagent.perception.foundation_contact import (  # noqa: E402
    ContactForceContract,
    EvidenceClass,
    MetricCameraContract,
    RobotTrajectoryContract,
    StemCenterlineContract,
    decide_foundation_contact_status,
    validate_contact_force_sequence,
    validate_metric_camera_sequence,
    validate_robot_trajectory,
    validate_stem_centerlines,
)
from phiagent.rendering.contact_dynamics import (  # noqa: E402
    ArticulatedHandContract,
    InteractionFrameContract,
    MetricContactContract,
    StemRodContract,
    assess_metric_force_closure,
    causal_motion_audit,
    couple_contact_patch_to_required_wrench,
    infer_stem_contact_forces,
    simulate_damped_stem,
    validate_kinematic_sequence,
)
from phiagent.rendering.metric_flower_simulation import (  # noqa: E402
    MetricFlowerSimulationContract,
    articulated_hand_points,
    build_metric_flower_schedule,
    camera_calibration_from_mujoco_scene,
    exact_pad_stem_contact_state,
    project_world_points,
)


ASSET_HASHES = {
    "g1_model": "3c2616550a31f33e84d3c80b8e913ac5618c8888019b0c9490dae93493e647f3",
    "sharpa_left_model": "3cbeb46259d4ba63cbdb83085255d1a8f8031c51e0101a6622f6e7e81a64dc11",
    "sharpa_right_model": "43d9cb63d724889b69574a5e0981aee4a2f30d825c85f3098988e3a7a3bb9980",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-model", type=Path, required=True)
    parser.add_argument("--sharpa-left-model", type=Path, required=True)
    parser.add_argument("--sharpa-right-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/usr/bin/ffprobe"))
    parser.add_argument("--source-git-head", required=True)
    parser.add_argument("--source-git-status-sha256", required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _git_state() -> dict[str, object]:
    result = {}
    for label, command in (
        ("head", ("git", "rev-parse", "HEAD")),
        ("status", ("git", "status", "--short")),
    ):
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        result[label] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return result


def _video_writer(ffmpeg: Path, path: Path, width: int, height: int, fps: float) -> Any:
    return subprocess.Popen(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.12g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def _video_metadata(ffprobe: Path, path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = (int(value) for value in stream["avg_frame_rate"].split("/"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": numerator / denominator,
        "frames": int(stream["nb_read_frames"]),
        "duration_seconds": float(stream.get("duration", 0.0)),
    }


def _render_depth(renderer: Any, data: Any, camera: Any) -> Any:
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return depth


def _resize_component(
    cv2: Any,
    np: Any,
    *,
    rgb: Any,
    mask: Any,
    depth: Any,
    width: int,
    height: int,
) -> tuple[Any, Any, Any]:
    resized_rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    finite_depth = np.where(mask > 0, depth, np.inf).astype(np.float32)
    resized_depth = cv2.resize(
        finite_depth,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    resized_depth[~resized_mask] = np.inf
    return resized_rgb, resized_mask, resized_depth


def _render_robot_components(
    cv2: Any,
    mujoco: Any,
    np: Any,
    *,
    g1: Any,
    hands: dict[str, Any],
    right_closure: float,
    width: int,
    height: int,
) -> list[tuple[Any, Any, Any]]:
    body_rgb, body_mask = g1.render()
    body_depth = _render_depth(g1.renderer, g1.data, g1.camera)
    components = [
        _resize_component(
            cv2,
            np,
            rgb=body_rgb,
            mask=body_mask,
            depth=body_depth,
            width=width,
            height=height,
        )
    ]
    for side in ("left", "right"):
        closure = 0.45 if side == "left" else right_closure
        points = articulated_hand_points(np, closure)
        wrist_position, wrist_quaternion = g1.wrist_pose(side)
        hand_rgb, hand_mask = hands[side].render(
            points,
            wrist_position,
            wrist_quaternion,
        )
        hand_depth = _render_depth(
            hands[side].renderer,
            hands[side].data,
            hands[side].camera,
        )
        components.append(
            _resize_component(
                cv2,
                np,
                rgb=hand_rgb,
                mask=hand_mask,
                depth=hand_depth,
                width=width,
                height=height,
            )
        )
    return components


def _background(cv2: Any, np: Any, width: int, height: int) -> tuple[Any, Any]:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    top = np.asarray((226, 233, 238), dtype=np.float32)[None, None, :]
    bottom = np.asarray((171, 187, 197), dtype=np.float32)[None, None, :]
    image = np.broadcast_to(top * (1.0 - y) + bottom * y, (height, width, 3)).copy()
    image = np.rint(image).astype(np.uint8)
    depth = np.full((height, width), 4.0, dtype=np.float32)
    table_top = round(height * 0.78)
    cv2.rectangle(image, (0, table_top), (width, height), (76, 87, 97), -1)
    depth[table_top:] = 3.2
    return image, depth


def _identity_background(np: Any, width: int, height: int) -> tuple[Any, Any]:
    image = np.full((height, width, 3), (36, 36, 36), dtype=np.uint8)
    depth = np.full((height, width), 4.0, dtype=np.float32)
    return image, depth


def _flower_layer(
    cv2: Any,
    np: Any,
    *,
    nodes_world_m: Any,
    intrinsics_px: Any,
    world_from_camera: Any,
    width: int,
    height: int,
) -> tuple[Any, Any, Any]:
    pixels, node_depth = project_world_points(
        np,
        points_world_m=nodes_world_m,
        intrinsics_px=intrinsics_px,
        world_from_camera=world_from_camera,
    )
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    rounded = np.rint(pixels).astype(np.int32)
    for index in range(len(rounded) - 1):
        start = tuple(int(value) for value in rounded[index])
        end = tuple(int(value) for value in rounded[index + 1])
        segment_depth = float((node_depth[index] + node_depth[index + 1]) * 0.5)
        cv2.line(rgb, start, end, (58, 132, 65), 3, cv2.LINE_AA)
        cv2.line(mask, start, end, 255, 4, cv2.LINE_AA)
        cv2.line(depth, start, end, segment_depth, 4, cv2.LINE_AA)
    head = tuple(int(value) for value in rounded[-1])
    for petal in range(8):
        angle = 2.0 * math.pi * petal / 8.0
        center = (
            round(head[0] + 6.0 * math.cos(angle)),
            round(head[1] + 6.0 * math.sin(angle)),
        )
        cv2.circle(rgb, center, 6, (164, 112, 226), -1, cv2.LINE_AA)
        cv2.circle(mask, center, 6, 255, -1, cv2.LINE_AA)
        cv2.circle(depth, center, 6, float(node_depth[-1]), -1, cv2.LINE_AA)
    cv2.circle(rgb, head, 4, (64, 172, 215), -1, cv2.LINE_AA)
    cv2.circle(mask, head, 4, 255, -1, cv2.LINE_AA)
    cv2.circle(depth, head, 4, float(node_depth[-1]), -1, cv2.LINE_AA)
    return rgb, mask > 0, depth


def _vase_layer(
    cv2: Any,
    np: Any,
    *,
    root_world_m: Any,
    intrinsics_px: Any,
    world_from_camera: Any,
    width: int,
    height: int,
) -> tuple[Any, Any, Any]:
    pixels, root_depth = project_world_points(
        np,
        points_world_m=np.asarray(root_world_m, dtype=np.float64)[None, :],
        intrinsics_px=intrinsics_px,
        world_from_camera=world_from_camera,
    )
    center = (round(float(pixels[0, 0])), round(float(pixels[0, 1] + 18)))
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    cv2.ellipse(rgb, center, (15, 23), 0, 0, 360, (79, 111, 154), -1, cv2.LINE_AA)
    cv2.ellipse(mask, center, (15, 23), 0, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.ellipse(
        depth,
        center,
        (15, 23),
        0,
        0,
        360,
        float(root_depth[0] - 0.01),
        -1,
        cv2.LINE_AA,
    )
    return rgb, mask > 0, depth


def _composite(np: Any, base: Any, base_depth: Any, layers: list[tuple[Any, Any, Any]]) -> Any:
    result = base.copy()
    depth = base_depth.copy()
    for rgb, mask, layer_depth in layers:
        visible = mask & np.isfinite(layer_depth) & (layer_depth > 0) & (layer_depth < depth)
        result[visible] = rgb[visible]
        depth[visible] = layer_depth[visible]
    return result, depth


def _right_hand_contract(mujoco: Any, np: Any, hand: Any) -> tuple[Any, ...]:
    chains = (
        (
            "right_thumb_CMC_FE",
            "right_thumb_CMC_AA",
            "right_thumb_MCP_FE",
            "right_thumb_IP",
        ),
        ("right_index_MCP_FE", "right_index_PIP", "right_index_DIP"),
        ("right_middle_MCP_FE", "right_middle_PIP", "right_middle_DIP"),
        ("right_ring_MCP_FE", "right_ring_PIP", "right_ring_DIP"),
        (
            "right_pinky_CMC",
            "right_pinky_MCP_FE",
            "right_pinky_PIP",
            "right_pinky_DIP",
        ),
    )
    names = ["right_palm"]
    parents = [-1]
    limits = [(-math.pi, math.pi)]
    joint_ids = [-1]
    fingertips = []
    for chain in chains:
        parent = 0
        for name in chain:
            joint_id = mujoco.mj_name2id(hand.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"right Sharpa model is missing {name}")
            names.append(name)
            parents.append(parent)
            limits.append(
                (
                    float(hand.model.jnt_range[joint_id, 0]),
                    float(hand.model.jnt_range[joint_id, 1]),
                )
            )
            joint_ids.append(joint_id)
            parent = len(names) - 1
        fingertips.append(parent)
    contract = ArticulatedHandContract(
        embodiment_id="sharpa-wave-right",
        coordinate_frame="robot_base:g1",
        joint_names=tuple(names),
        parent_indices=tuple(parents),
        joint_limits_rad=tuple(limits),
        fingertip_indices=tuple(fingertips),
        palm_index=0,
    )
    contract.validate()
    return contract, tuple(joint_ids)


def _right_hand_state(np: Any, hand: Any, joint_ids: tuple[int, ...]) -> tuple[Any, Any]:
    positions = [hand.data.xpos[1].copy()]
    angles = [0.0]
    for joint_id in joint_ids[1:]:
        positions.append(hand.data.xanchor[joint_id].copy())
        angles.append(float(hand.data.qpos[hand.model.jnt_qposadr[joint_id]]))
    return np.asarray(positions), np.asarray(angles)


def _right_pad_vertices(
    mujoco: Any,
    np: Any,
    hand: Any,
    fingertip_indices: tuple[int, ...],
) -> dict[int, Any]:
    geom_names = (
        "right_thumb_elastomer",
        "right_index_elastomer",
        "right_middle_elastomer",
        "right_ring_elastomer",
        "right_pinky_elastomer",
    )
    if len(fingertip_indices) != len(geom_names):
        raise RuntimeError("right hand contract must expose five exact fingertip pads")
    result = {}
    for fingertip_index, geom_name in zip(fingertip_indices, geom_names):
        geom_id = mujoco.mj_name2id(
            hand.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )
        if geom_id < 0:
            raise RuntimeError(f"right Sharpa model is missing {geom_name}")
        mesh_id = int(hand.model.geom_dataid[geom_id])
        if mesh_id < 0:
            raise RuntimeError(f"right Sharpa pad {geom_name} is not a mesh")
        start = int(hand.model.mesh_vertadr[mesh_id])
        count = int(hand.model.mesh_vertnum[mesh_id])
        local = (
            hand.model.mesh_vert[start : start + count]
            * hand.model.mesh_scale[mesh_id]
        )
        rotation = hand.data.geom_xmat[geom_id].reshape(3, 3)
        result[int(fingertip_index)] = (
            local @ rotation.T + hand.data.geom_xpos[geom_id]
        )
    return result


def _right_pad_centers(mujoco: Any, np: Any, hand: Any) -> Any:
    names = (
        "right_thumb_elastomer",
        "right_index_elastomer",
        "right_middle_elastomer",
        "right_ring_elastomer",
        "right_pinky_elastomer",
    )
    centers = []
    for name in names:
        geom_id = mujoco.mj_name2id(
            hand.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )
        if geom_id < 0:
            raise RuntimeError(f"right Sharpa model is missing {name}")
        centers.append(hand.data.geom_xpos[geom_id].copy())
    return np.stack(centers)


def _force_closure_assessment(
    np: Any,
    *,
    state: dict[str, Any],
    frame_contract: InteractionFrameContract,
) -> dict[str, object]:
    if int(state["contacting_fingertips"]) < 2:
        return {
            "passed": False,
            "reasons": ["insufficient_exact_sharpa_pad_contacts"],
            "distinct_fingertips": int(state["contacting_fingertips"]),
            "force_closure": None,
        }
    return assess_metric_force_closure(
        np,
        contact_points_m=state["contact_points_m"],
        surface_gaps_m=state["surface_gaps_m"],
        contact_normals=state["contact_normals"],
        contact_forces_n=state["contact_forces_n"],
        object_center_m=state["object_center_m"],
        external_force_n=state["external_force_n"],
        external_moment_nm=state["external_moment_nm"],
        fingertip_indices=state["fingertip_indices"],
        frame_contract=frame_contract,
        contact_contract=MetricContactContract(),
        depth_source="mujoco-depth-buffer",
        force_source="exact-pad-friction-cone-coupled-inverse-rod-v3",
        occlusion_order_known=True,
    )


def _couple_contact_state(
    np: Any,
    *,
    state: dict[str, Any],
    required_force_n: Any,
) -> dict[str, Any]:
    coupled = couple_contact_patch_to_required_wrench(
        np,
        contact_points_m=state["contact_points_m"],
        contact_normals=state["contact_normals"],
        fingertip_indices=state["fingertip_indices"],
        object_center_m=state["object_center_m"],
        required_force_n=required_force_n,
        required_moment_nm=np.zeros(3, dtype=np.float64),
    )
    return {**state, **coupled}


class _RightWristPoseController:
    """Keep the exact G1 wrist orientation fixed while following contact position."""

    JOINT_NAMES = (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )

    def __init__(self, mujoco: Any, np: Any, g1: Any) -> None:
        self.mujoco = mujoco
        self.np = np
        self.g1 = g1
        self.joint_ids = tuple(
            mujoco.mj_name2id(g1.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.JOINT_NAMES
        )
        if any(joint_id < 0 for joint_id in self.joint_ids):
            raise RuntimeError("G1 model is missing a right arm or wrist joint")
        self.addresses = tuple(
            int(g1.model.jnt_qposadr[joint_id])
            for joint_id in self.joint_ids
        )
        self.maximum_step_rad = 0.0
        self.maximum_position_error_m = 0.0
        self.maximum_orientation_error_rad = 0.0

    def solve(self, target_position: Any, target_quaternion_wxyz: Any) -> None:
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        target = self.np.asarray(target_position, dtype=self.np.float64)
        target_quaternion = self.np.asarray(
            target_quaternion_wxyz,
            dtype=self.np.float64,
        )
        if target.shape != (3,) or target_quaternion.shape != (4,):
            raise ValueError("right wrist pose target must contain xyz and wxyz")
        previous = self.g1.data.qpos[list(self.addresses)].copy()
        model_lower = self.g1.model.jnt_range[list(self.joint_ids), 0]
        model_upper = self.g1.model.jnt_range[list(self.joint_ids), 1]
        lower = self.np.maximum(model_lower, previous - 0.12)
        upper = self.np.minimum(model_upper, previous + 0.12)
        target_rotation = Rotation.from_quat(
            (
                target_quaternion[1],
                target_quaternion[2],
                target_quaternion[3],
                target_quaternion[0],
            )
        )
        wrist_body_id = self.g1.wrist_body_ids["right"]

        def residual(values: Any) -> Any:
            self.g1.data.qpos[list(self.addresses)] = values
            self.mujoco.mj_forward(self.g1.model, self.g1.data)
            position_error = self.g1.data.xpos[wrist_body_id] - target
            current = self.g1.data.xquat[wrist_body_id]
            current_rotation = Rotation.from_quat(
                (current[1], current[2], current[3], current[0])
            )
            orientation_error = (
                target_rotation * current_rotation.inv()
            ).as_rotvec()
            regularization = 0.01 * (values - previous)
            return self.np.concatenate(
                (
                    position_error,
                    0.30 * orientation_error,
                    regularization,
                )
            )

        solution = least_squares(
            residual,
            previous,
            bounds=(lower, upper),
            max_nfev=48,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        values = solution.x
        self.g1.data.qpos[list(self.addresses)] = values
        self.mujoco.mj_forward(self.g1.model, self.g1.data)
        current = self.g1.data.xquat[wrist_body_id]
        current_rotation = Rotation.from_quat(
            (current[1], current[2], current[3], current[0])
        )
        position_error = float(
            self.np.linalg.norm(self.g1.data.xpos[wrist_body_id] - target)
        )
        orientation_error = float(
            self.np.linalg.norm(
                (target_rotation * current_rotation.inv()).as_rotvec()
            )
        )
        self.maximum_step_rad = max(
            self.maximum_step_rad,
            float(self.np.max(self.np.abs(values - previous))),
        )
        self.maximum_position_error_m = max(
            self.maximum_position_error_m,
            position_error,
        )
        self.maximum_orientation_error_rad = max(
            self.maximum_orientation_error_rad,
            orientation_error,
        )
        self.g1.previous_q["right"] = self.g1.data.qpos[
            list(self.g1.qpos_addresses["right"])
        ].copy()


def _transport_grasp_quaternion(
    np: Any,
    *,
    initial_quaternion_wxyz: Any,
    initial_stem_tangent: Any,
    current_stem_tangent: Any,
) -> Any:
    """Parallel-transport the grasp orientation by the shortest stem-tangent rotation."""

    initial = np.asarray(initial_stem_tangent, dtype=np.float64)
    current = np.asarray(current_stem_tangent, dtype=np.float64)
    initial /= np.linalg.norm(initial)
    current /= np.linalg.norm(current)
    cross = np.cross(initial, current)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(initial, current), -1.0, 1.0))
    if sine <= 1e-12:
        if cosine > 0.0:
            delta_quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
        else:
            axis = np.zeros(3, dtype=np.float64)
            axis[int(np.argmin(np.abs(initial)))] = 1.0
            axis = np.cross(initial, axis)
            axis /= np.linalg.norm(axis)
            delta_quaternion = np.concatenate(
                (np.zeros(1, dtype=np.float64), axis)
            )
    else:
        axis = cross / sine
        half_angle = 0.5 * np.arctan2(sine, cosine)
        delta_quaternion = np.concatenate(
            (
                np.asarray((np.cos(half_angle),)),
                np.sin(half_angle) * axis,
            )
        )
    base = np.asarray(initial_quaternion_wxyz, dtype=np.float64)
    delta_w, delta_xyz = delta_quaternion[0], delta_quaternion[1:]
    base_w, base_xyz = base[0], base[1:]
    transported = np.concatenate(
        (
            np.asarray(
                (
                    delta_w * base_w
                    - float(np.dot(delta_xyz, base_xyz)),
                )
            ),
            delta_w * base_xyz
            + base_w * delta_xyz
            + np.cross(delta_xyz, base_xyz),
        )
    )
    return transported / np.linalg.norm(transported)


def main() -> int:
    args = _parser().parse_args()
    if args.frames < 33 or args.frames > 660:
        raise ValueError("simulation frame count must be in [33, 660]")
    if min(args.fps, args.width, args.height) <= 0:
        raise ValueError("FPS and image dimensions must be positive")
    inputs = {
        "g1_model": args.g1_model.expanduser().resolve(),
        "sharpa_left_model": args.sharpa_left_model.expanduser().resolve(),
        "sharpa_right_model": args.sharpa_right_model.expanduser().resolve(),
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        digest = _sha256(path)
        if digest != ASSET_HASHES[name]:
            raise ValueError(f"{name} hash {digest} does not match the frozen registry")
    for name, path in (("ffmpeg", args.ffmpeg), ("ffprobe", args.ffprobe)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output_dir}")
    output_dir.mkdir(parents=True)
    heartbeat_path = output_dir / "heartbeat.json"
    _write_json_atomic(
        heartbeat_path,
        {
            "status": "RUNNING",
            "stage": "preflight",
            "completed_frames": 0,
            "expected_frames": args.frames,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    inventory = _gpu_inventory()
    selected_gpu = _select_gpu(inventory, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")

    import cv2
    import mujoco
    import numpy as np

    np.random.seed(args.seed)
    started = time.perf_counter()
    approach_end = max(9, round(args.frames * 180 / 660))
    release_frame = min(args.frames - 8, max(approach_end + 8, round(args.frames * 480 / 660)))
    contract = MetricFlowerSimulationContract(
        frames=args.frames,
        fps=args.fps,
        approach_end_frame=approach_end,
        release_frame=release_frame,
    )
    contact_center = np.asarray((0.40053, -0.17967, 0.87815), dtype=np.float64)
    segment_length = 0.60 / (contract.nodes_per_stem - 1)
    rest_nodes = np.stack(
        (
            np.full(contract.nodes_per_stem, contact_center[0]),
            np.full(contract.nodes_per_stem, contact_center[1]),
            contact_center[2]
            + (
                np.arange(contract.nodes_per_stem, dtype=np.float64)
                - contract.contact_node
            )
            * segment_length,
        ),
        axis=1,
    )
    schedule = build_metric_flower_schedule(
        np,
        rest_nodes_m=rest_nodes,
        contract=contract,
    )
    rod_contract = StemRodContract(
        instance_id="active-stem-01",
        coordinate_frame="robot_base:g1",
        node_count=contract.nodes_per_stem,
        root_node=0,
        linear_density_kg_m=0.012,
        axial_stiffness_n_m=18.0,
        bending_stiffness_n_m=0.65,
        damping_n_s_m=0.18,
    )
    rod = simulate_damped_stem(
        np,
        rest_nodes_m=rest_nodes,
        contact_targets_m=schedule["contact_targets_m"],
        contact_active=schedule["contact_active"],
        contact_node=contract.contact_node,
        contract=rod_contract,
        fps=contract.fps,
    )
    if not rod["passed"]:
        raise RuntimeError("metric rod simulation failed its rooted finite-state gate")
    inverse = infer_stem_contact_forces(
        np,
        nodes_m=rod["nodes_m"],
        position_sigma_m=np.full(
            (contract.frames, contract.nodes_per_stem),
            1e-5,
            dtype=np.float64,
        ),
        contact_nodes=np.full(contract.frames, contract.contact_node, dtype=np.int64),
        contact_active=schedule["contact_active"],
        contract=rod_contract,
        fps=contract.fps,
    )

    g1 = G1IkRenderer(mujoco, np, inputs["g1_model"])
    hands = {
        "left": AttachedHandRenderer(
            mujoco,
            np,
            inputs["sharpa_left_model"],
            "sharpa",
            "left",
        ),
        "right": AttachedHandRenderer(
            mujoco,
            np,
            inputs["sharpa_right_model"],
            "sharpa",
            "right",
        ),
    }
    hand_contract, hand_joint_ids = _right_hand_contract(mujoco, np, hands["right"])
    contact_projection_frame = InteractionFrameContract(
        camera_frame="camera:simulated_rgbd",
        metric_frame="robot_base:g1",
        timeline="frame:source_video",
        fps=contract.fps,
        fx_pixels=1.0,
        fy_pixels=1.0,
        cx_pixels=0.0,
        cy_pixels=0.0,
        metric_scale_source="MuJoCo metric scene units",
    )
    repair_values = (-0.004, -0.002, 0.0, 0.002, 0.004)
    contact_repair_candidates = sorted(
        (
            np.asarray((x, y, z), dtype=np.float64)
            for x in repair_values
            for y in repair_values
            for z in repair_values
            if (x, y, z) != (0.0, 0.0, 0.0)
        ),
        key=lambda value: float(np.linalg.norm(value)),
    )
    video_path = output_dir / "metric-flower-simulation.mp4"
    writer = _video_writer(
        args.ffmpeg.expanduser().resolve(),
        video_path,
        args.width,
        args.height,
        args.fps,
    )
    if writer.stdin is None:
        raise RuntimeError("ffmpeg writer did not expose stdin")

    depths = []
    robot_positions = []
    robot_limits = None
    robot_names = None
    floating_base = []
    hand_joint_positions = []
    hand_joint_angles = []
    contact_states = []
    contact_repair_offsets = np.zeros((contract.frames, 3), dtype=np.float64)
    contact_repair_attempts = np.zeros(contract.frames, dtype=np.int32)
    contact_repair_requested = np.zeros(contract.frames, dtype=bool)
    final_right_wrist_targets = []
    intrinsics = None
    world_from_camera = None
    reprojection_rmse = []
    identity_reference = None
    try:
        for frame_index in range(contract.frames):
            right_closure = float(schedule["right_hand_closure"][frame_index])
            contact_active = bool(schedule["contact_active"][frame_index])
            targets = {
                "left": schedule["left_wrist_targets_m"][frame_index],
                "right": schedule["right_wrist_targets_m"][frame_index].copy(),
            }
            g1.solve(targets)
            if right_closure > 0.0:
                preliminary_points = articulated_hand_points(np, right_closure)
                wrist_position, wrist_quaternion = g1.wrist_pose("right")
                hands["right"].render(
                    preliminary_points,
                    wrist_position,
                    wrist_quaternion,
                )
                pad_centers = _right_pad_centers(mujoco, np, hands["right"])
                contact_center = rod["nodes_m"][frame_index, contract.contact_node]
                correction = contact_center - np.mean(pad_centers[:3], axis=0)
                targets["right"] += right_closure * correction
                g1.solve(targets)
            components = _render_robot_components(
                cv2,
                mujoco,
                np,
                g1=g1,
                hands=hands,
                right_closure=right_closure,
                width=args.width,
                height=args.height,
            )
            if intrinsics is None:
                full_intrinsics, world_from_camera = camera_calibration_from_mujoco_scene(
                    np,
                    scene_camera=g1.renderer.scene.camera,
                    width=640,
                    height=480,
                    vertical_fov_degrees=float(g1.model.vis.global_.fovy),
                )
                intrinsics = full_intrinsics.copy()
                intrinsics[0] *= args.width / 640.0
                intrinsics[1] *= args.height / 480.0
            pad_vertices = _right_pad_vertices(
                mujoco,
                np,
                hands["right"],
                hand_contract.fingertip_indices,
            )
            contact_state = (
                _couple_contact_state(
                    np,
                    state=exact_pad_stem_contact_state(
                        np,
                        pad_vertices_by_fingertip=pad_vertices,
                        stem_nodes_m=rod["nodes_m"][frame_index],
                    ),
                    required_force_n=inverse["hand_on_stem_forces_n"][
                        frame_index,
                        0,
                    ],
                )
                if contact_active
                else None
            )
            if contact_state is not None:
                contact_assessment = _force_closure_assessment(
                    np,
                    state=contact_state,
                    frame_contract=contact_projection_frame,
                )
                if not contact_assessment["passed"]:
                    contact_repair_requested[frame_index] = True
                    base_qpos = g1.data.qpos.copy()
                    base_previous_right = g1.previous_q["right"].copy()
                    repaired = False
                    for attempt, offset in enumerate(
                        contact_repair_candidates,
                        1,
                    ):
                        g1.data.qpos[:] = base_qpos
                        g1.previous_q["right"] = base_previous_right.copy()
                        mujoco.mj_forward(g1.model, g1.data)
                        candidate_target = targets["right"] + offset
                        g1._solve_side("right", candidate_target)
                        wrist_position, wrist_quaternion = g1.wrist_pose("right")
                        hands["right"].render(
                            articulated_hand_points(np, right_closure),
                            wrist_position,
                            wrist_quaternion,
                        )
                        candidate_state = _couple_contact_state(
                            np,
                            state=exact_pad_stem_contact_state(
                                np,
                                pad_vertices_by_fingertip=_right_pad_vertices(
                                    mujoco,
                                    np,
                                    hands["right"],
                                    hand_contract.fingertip_indices,
                                ),
                                stem_nodes_m=rod["nodes_m"][frame_index],
                            ),
                            required_force_n=inverse[
                                "hand_on_stem_forces_n"
                            ][frame_index, 0],
                        )
                        candidate_assessment = _force_closure_assessment(
                            np,
                            state=candidate_state,
                            frame_contract=contact_projection_frame,
                        )
                        if candidate_assessment["passed"]:
                            contact_state = candidate_state
                            targets["right"] = candidate_target
                            contact_repair_offsets[frame_index] = offset
                            contact_repair_attempts[frame_index] = attempt
                            repaired = True
                            break
                    if repaired:
                        components = _render_robot_components(
                            cv2,
                            mujoco,
                            np,
                            g1=g1,
                            hands=hands,
                            right_closure=right_closure,
                            width=args.width,
                            height=args.height,
                        )
                    else:
                        g1.data.qpos[:] = base_qpos
                        g1.previous_q["right"] = base_previous_right
                        mujoco.mj_forward(g1.model, g1.data)
                        components = _render_robot_components(
                            cv2,
                            mujoco,
                            np,
                            g1=g1,
                            hands=hands,
                            right_closure=right_closure,
                            width=args.width,
                            height=args.height,
                        )
            contact_states.append(contact_state)
            final_right_wrist_targets.append(targets["right"].copy())

            flower = _flower_layer(
                cv2,
                np,
                nodes_world_m=rod["nodes_m"][frame_index],
                intrinsics_px=intrinsics,
                world_from_camera=world_from_camera,
                width=args.width,
                height=args.height,
            )
            vase = _vase_layer(
                cv2,
                np,
                root_world_m=rest_nodes[0],
                intrinsics_px=intrinsics,
                world_from_camera=world_from_camera,
                width=args.width,
                height=args.height,
            )
            background_rgb, background_depth = _background(
                cv2,
                np,
                args.width,
                args.height,
            )
            if identity_reference is None:
                identity_background_rgb, identity_background_depth = (
                    _identity_background(
                        np,
                        args.width,
                        args.height,
                    )
                )
                identity_reference, _ = _composite(
                    np,
                    identity_background_rgb,
                    identity_background_depth,
                    components,
                )
            frame, metric_depth = _composite(
                np,
                background_rgb,
                background_depth,
                [flower, vase, *components],
            )
            writer.stdin.write(np.ascontiguousarray(frame).tobytes())
            depths.append(metric_depth.astype(np.float32))

            component_states = [("g1", *g1.generalized_joint_state())]
            component_states.extend(
                (f"sharpa_{side}", *hands[side].generalized_joint_state())
                for side in ("left", "right")
            )
            names = tuple(
                f"{component}:{name}"
                for component, component_names, _, _ in component_states
                for name in component_names
            )
            limits = np.concatenate(
                [limits_value for _, _, _, limits_value in component_states],
                axis=0,
            )
            positions = np.concatenate(
                [positions_value for _, _, positions_value, _ in component_states],
                axis=0,
            )
            if robot_names is None:
                robot_names = names
                robot_limits = limits
            elif names != robot_names or not np.array_equal(limits, robot_limits):
                raise RuntimeError("exact robot joint topology changed during simulation")
            robot_positions.append(positions)
            base_translation, base_quaternion = g1.floating_base_pose()
            floating_base.append(np.concatenate((base_translation, base_quaternion)))
            right_positions, right_angles = _right_hand_state(
                np,
                hands["right"],
                hand_joint_ids,
            )
            hand_joint_positions.append(right_positions)
            hand_joint_angles.append(right_angles)
            reprojection_rmse.append(0.0)
            if frame_index % 30 == 0 or frame_index + 1 == contract.frames:
                _write_json_atomic(
                    heartbeat_path,
                    {
                        "status": "RUNNING",
                        "stage": "render",
                        "completed_frames": frame_index + 1,
                        "expected_frames": contract.frames,
                        "contact_projection_requested": int(
                            np.count_nonzero(
                                contact_repair_requested[: frame_index + 1]
                            )
                        ),
                        "contact_projection_repaired": int(
                            np.count_nonzero(
                                contact_repair_attempts[: frame_index + 1] > 0
                            )
                        ),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
    finally:
        for hand in hands.values():
            hand.close()
        g1.close()
        writer.stdin.close()
        writer_returncode = writer.wait()
    if writer_returncode:
        raise RuntimeError(f"ffmpeg writer failed with return code {writer_returncode}")
    if intrinsics is None or world_from_camera is None or robot_names is None:
        raise RuntimeError("simulation produced no calibrated frames")
    if identity_reference is None:
        raise RuntimeError("simulation did not render an identity reference")

    depth_sequence = np.stack(depths)
    frame_indices = schedule["frame_indices"]
    bundle_id = output_dir.name
    source_video_sha256 = _sha256(video_path)
    identity_reference_path = output_dir / "robot-identity-reference.png"
    if not cv2.imwrite(str(identity_reference_path), identity_reference):
        raise RuntimeError("failed to write the robot identity reference")
    camera_samples_path = output_dir / "metric-camera-samples.npz"
    np.savez_compressed(
        camera_samples_path,
        bundle_id=np.asarray(bundle_id),
        source_video_sha256=np.asarray(source_video_sha256),
        source_frame_indices=frame_indices,
        fps=np.asarray(contract.fps, dtype=np.float64),
        intrinsics_px=np.repeat(intrinsics[None, :, :], contract.frames, axis=0),
        world_from_camera=np.repeat(
            world_from_camera[None, :, :],
            contract.frames,
            axis=0,
        ),
        depth_m=depth_sequence,
        confidence=np.ones_like(depth_sequence, dtype=np.float32),
        camera_frame=np.asarray("camera:simulated_rgbd"),
        world_frame=np.asarray("robot_base:g1"),
        timeline=np.asarray("frame:source_video"),
    )
    camera_report_path = output_dir / "metric-camera-report.json"
    camera_report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "bundle_id": bundle_id,
        "source_video_sha256": source_video_sha256,
        "samples_sha256": _sha256(camera_samples_path),
        "camera_frame": "camera:simulated_rgbd",
        "world_frame": "robot_base:g1",
        "timeline": "frame:source_video",
        "fps": contract.fps,
        "image_width": args.width,
        "image_height": args.height,
        "intrinsics_evidence": "calibrated_geometry",
        "depth_evidence": "sensor_measurement",
        "metric_scale_source": "MuJoCo metric renderer depth buffer and exact scene units",
        "absolute_scale_standard_deviation_fraction": 0.0,
        "independent_calibration_groups": 0,
        "scale_scope": "simulator ground truth only; not a calibration of the Pexels camera",
        "gates": {
            "exact_scene_units": True,
            "renderer_depth_is_metric": True,
            "fixed_camera_projection_recovered": True,
            "complete_depth_coverage": bool(
                np.all(np.isfinite(depth_sequence)) and np.all(depth_sequence > 0)
            ),
        },
    }
    camera_report_path.write_text(
        json.dumps(camera_report, indent=2, sort_keys=True) + "\n"
    )

    robot_positions_array = np.stack(robot_positions)
    robot_trajectory_path = output_dir / "robot-trajectory.npz"
    np.savez_compressed(
        robot_trajectory_path,
        bundle_id=np.asarray(bundle_id),
        source_video_sha256=np.asarray(source_video_sha256),
        embodiment_id=np.asarray("unitree-g1-bilateral-sharpa-wave"),
        robot_base_frame=np.asarray("robot_base:g1"),
        timeline=np.asarray("frame:source_video"),
        fps=np.asarray(contract.fps, dtype=np.float64),
        source_frame_indices=frame_indices,
        joint_names=np.asarray(robot_names),
        joint_limits_rad=robot_limits,
        joint_positions_rad=robot_positions_array,
        joint_velocities_rad_s=np.gradient(
            robot_positions_array,
            1.0 / contract.fps,
            axis=0,
        ),
        floating_base_xyz_wxyz=np.stack(floating_base),
        contact_repair_requested=contact_repair_requested,
        contact_repair_attempts=contact_repair_attempts,
        contact_repair_offsets_robot_base_m=contact_repair_offsets,
        reprojection_rmse_px=np.asarray(reprojection_rmse),
        trajectory_evidence=np.asarray("physics_solver_estimate"),
    )
    stem_path = output_dir / "metric-stem-centerlines.npz"
    np.savez_compressed(
        stem_path,
        bundle_id=np.asarray(bundle_id),
        source_video_sha256=np.asarray(source_video_sha256),
        source_frame_indices=frame_indices,
        fps=np.asarray(contract.fps, dtype=np.float64),
        instance_ids=np.asarray((rod_contract.instance_id,)),
        centerlines_world_m=rod["nodes_m"][:, None, :, :],
        confidence=np.ones(
            (contract.frames, 1, contract.nodes_per_stem),
            dtype=np.float64,
        ),
        coordinate_frame=np.asarray(rod_contract.coordinate_frame),
        timeline=np.asarray("frame:source_video"),
        evidence_class=np.asarray("physics_solver_estimate"),
    )
    fingertip_indices = hand_contract.fingertip_indices
    fingertip_slot = {
        fingertip_index: slot
        for slot, fingertip_index in enumerate(fingertip_indices)
    }
    pad_forces = np.zeros(
        (contract.frames, len(fingertip_indices), 3),
        dtype=np.float64,
    )
    pad_covariance = np.zeros(
        (contract.frames, len(fingertip_indices), 3, 3),
        dtype=np.float64,
    )
    pad_gaps = np.full(
        (contract.frames, len(fingertip_indices)),
        np.nan,
        dtype=np.float64,
    )
    pad_contact_active = np.zeros(
        (contract.frames, len(fingertip_indices)),
        dtype=bool,
    )
    pad_contact_points = np.full(
        (contract.frames, len(fingertip_indices), 3),
        np.nan,
        dtype=np.float64,
    )
    coupled_force_residual = np.zeros(contract.frames, dtype=np.float64)
    coupled_moment_residual = np.zeros(contract.frames, dtype=np.float64)
    for frame_index, state in enumerate(contact_states):
        if state is None:
            continue
        coupled_force_residual[frame_index] = state[
            "coupled_force_residual_n"
        ]
        coupled_moment_residual[frame_index] = state[
            "coupled_moment_residual_nm"
        ]
        for fingertip_index in set(state["fingertip_indices"]):
            slot = fingertip_slot[int(fingertip_index)]
            selected = np.asarray(
                [
                    index
                    for index, value in enumerate(state["fingertip_indices"])
                    if int(value) == int(fingertip_index)
                ],
                dtype=np.int64,
            )
            pad_forces[frame_index, slot] = np.sum(
                state["contact_forces_n"][selected],
                axis=0,
            )
            pad_covariance[frame_index, slot] = (
                np.eye(3) * 4e-6 * len(selected)
            )
            pad_gaps[frame_index, slot] = float(
                np.max(state["surface_gaps_m"][selected])
            )
            pad_contact_active[frame_index, slot] = True
            pad_contact_points[frame_index, slot] = np.mean(
                state["contact_points_m"][selected],
                axis=0,
            )
    forces_path = output_dir / "contact-forces.npz"
    combined_solver_residual = (
        inverse["unexplained_force_residual_n"]
        + coupled_force_residual
        + coupled_moment_residual / 0.01
    )
    np.savez_compressed(
        forces_path,
        bundle_id=np.asarray(bundle_id),
        source_video_sha256=np.asarray(source_video_sha256),
        source_frame_indices=frame_indices,
        fps=np.asarray(contract.fps, dtype=np.float64),
        forces_n=pad_forces[:, None, :, :],
        solver_residual_n=combined_solver_residual[:, None],
        inverse_rod_residual_n=inverse["unexplained_force_residual_n"],
        coupled_contact_force_residual_n=coupled_force_residual,
        coupled_contact_moment_residual_nm=coupled_moment_residual,
        covariance_n2=pad_covariance[:, None, :, :, :],
        rod_actuator_forces_n=inverse["hand_on_stem_forces_n"],
        fingertip_indices=np.asarray(fingertip_indices),
        contact_active=pad_contact_active,
        contact_points_m=pad_contact_points,
        surface_gaps_m=pad_gaps,
        coordinate_frame=np.asarray(rod_contract.coordinate_frame),
        timeline=np.asarray("frame:source_video"),
        instance_ids=np.asarray((rod_contract.instance_id,)),
        force_evidence=np.asarray("physics_solver_estimate"),
        source_name=np.asarray("exact-pad-friction-cone-coupled-inverse-rod-v3"),
    )
    hand_path = output_dir / "right-hand-kinematics.npz"
    np.savez_compressed(
        hand_path,
        bundle_id=np.asarray(bundle_id),
        source_video_sha256=np.asarray(source_video_sha256),
        source_frame_indices=frame_indices,
        fps=np.asarray(contract.fps, dtype=np.float64),
        embodiment_id=np.asarray(hand_contract.embodiment_id),
        coordinate_frame=np.asarray(hand_contract.coordinate_frame),
        joint_names=np.asarray(hand_contract.joint_names),
        parent_indices=np.asarray(hand_contract.parent_indices),
        joint_limits_rad=np.asarray(hand_contract.joint_limits_rad),
        fingertip_indices=np.asarray(hand_contract.fingertip_indices),
        palm_index=np.asarray(hand_contract.palm_index),
        joints_xyz_m=np.stack(hand_joint_positions),
        joint_angles_rad=np.stack(hand_joint_angles),
    )

    camera_validation = validate_metric_camera_sequence(
        np,
        contract=MetricCameraContract(
            camera_frame="camera:simulated_rgbd",
            world_frame="robot_base:g1",
            timeline="frame:source_video",
            fps=contract.fps,
            image_width=args.width,
            image_height=args.height,
            intrinsics_evidence=EvidenceClass.CALIBRATED_GEOMETRY,
            depth_evidence=EvidenceClass.SENSOR_MEASUREMENT,
            metric_scale_source=camera_report["metric_scale_source"],
            absolute_scale_standard_deviation_fraction=0.0,
            calibration_report_sha256=_sha256(camera_report_path),
        ),
        frame_indices=frame_indices,
        intrinsics_px=np.repeat(intrinsics[None, :, :], contract.frames, axis=0),
        world_from_camera=np.repeat(
            world_from_camera[None, :, :],
            contract.frames,
            axis=0,
        ),
        depth_m=depth_sequence,
        depth_confidence=np.ones_like(depth_sequence),
    )
    robot_validation = validate_robot_trajectory(
        np,
        contract=RobotTrajectoryContract(
            embodiment_id="unitree-g1-bilateral-sharpa-wave",
            robot_base_frame="robot_base:g1",
            timeline="frame:source_video",
            fps=contract.fps,
            joint_names=robot_names,
            joint_limits_rad=tuple(
                (float(value[0]), float(value[1])) for value in robot_limits
            ),
            asset_sha256={name: _sha256(path) for name, path in inputs.items()},
            trajectory_evidence=EvidenceClass.PHYSICS_SOLVER_ESTIMATE,
        ),
        frame_indices=frame_indices,
        joint_positions_rad=robot_positions_array,
        joint_velocities_rad_s=np.gradient(
            robot_positions_array,
            1.0 / contract.fps,
            axis=0,
        ),
        reprojection_rmse_px=np.asarray(reprojection_rmse),
    )
    stem_validation = validate_stem_centerlines(
        np,
        contract=StemCenterlineContract(
            instance_ids=(rod_contract.instance_id,),
            coordinate_frame=rod_contract.coordinate_frame,
            timeline="frame:source_video",
            nodes_per_stem=contract.nodes_per_stem,
            geometry_evidence=EvidenceClass.PHYSICS_SOLVER_ESTIMATE,
        ),
        frame_indices=frame_indices,
        centerlines_m=rod["nodes_m"][:, None, :, :],
        confidence=np.ones(
            (contract.frames, 1, contract.nodes_per_stem),
            dtype=np.float64,
        ),
    )
    force_validation = validate_contact_force_sequence(
        np,
        contract=ContactForceContract(
            coordinate_frame=rod_contract.coordinate_frame,
            timeline="frame:source_video",
            instance_ids=(rod_contract.instance_id,),
            force_evidence=EvidenceClass.PHYSICS_SOLVER_ESTIMATE,
            source_name="exact-pad-friction-cone-coupled-inverse-rod-v3",
        ),
        forces_n=pad_forces[:, None, :, :],
        solver_residual_n=combined_solver_residual[:, None],
        covariance_n2=pad_covariance[:, None, :, :, :],
    )
    stages = {
        "metric_camera": camera_validation,
        "robot_trajectory": robot_validation,
        "stem_centerlines": stem_validation,
        "contact_forces": force_validation,
    }
    physical_decision = decide_foundation_contact_status(stages)

    hand_validation = validate_kinematic_sequence(
        np,
        joints_xyz_m=np.stack(hand_joint_positions),
        joint_angles_rad=np.stack(hand_joint_angles),
        contract=hand_contract,
    )
    hand_speed = np.concatenate(
        (
            np.zeros(1),
            np.linalg.norm(
                np.diff(np.stack(final_right_wrist_targets), axis=0),
                axis=1,
            )
            * contract.fps,
        )
    )
    stem_speed = np.concatenate(
        (
            np.zeros(1),
            np.linalg.norm(
                np.diff(rod["nodes_m"][:, contract.contact_node], axis=0),
                axis=1,
            )
            * contract.fps,
        )
    )
    causal = causal_motion_audit(
        np,
        grasp_active=schedule["contact_active"],
        hand_speed=hand_speed,
        stem_speed=stem_speed,
        hand_motion_floor=1e-4,
        stem_motion_floor=1e-5,
        maximum_response_lag_frames=2,
        maximum_frozen_run_frames=2,
    )
    frame_contract = InteractionFrameContract(
        camera_frame="camera:simulated_rgbd",
        metric_frame="robot_base:g1",
        timeline="frame:source_video",
        fps=contract.fps,
        fx_pixels=float(intrinsics[0, 0]),
        fy_pixels=float(intrinsics[1, 1]),
        cx_pixels=float(intrinsics[0, 2]),
        cy_pixels=float(intrinsics[1, 2]),
        metric_scale_source=camera_report["metric_scale_source"],
    )
    active_indices = np.flatnonzero(schedule["contact_active"])
    closure_reports = []
    for frame_index in active_indices:
        state = contact_states[int(frame_index)]
        if state is None:
            raise RuntimeError("active contact frame lacks exact pad geometry")
        closure_reports.append(
            _force_closure_assessment(
                np,
                state=state,
                frame_contract=frame_contract,
            )
        )
        if len(closure_reports) % 20 == 0 or len(closure_reports) == len(
            active_indices
        ):
            _write_json_atomic(
                heartbeat_path,
                {
                    "status": "RUNNING",
                    "stage": "force_closure_audit",
                    "completed_contact_frames": len(closure_reports),
                    "expected_contact_frames": len(active_indices),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
    closure_passed = bool(closure_reports) and all(
        report["passed"] for report in closure_reports
    )
    failed_closure_frames = [
        {
            "source_frame_index": int(frame_index),
            "reasons": list(report.get("reasons", ())),
            "minimum_positive_wrench_weight": (
                report["force_closure"]["minimum_positive_wrench_weight"]
                if isinstance(report.get("force_closure"), dict)
                else None
            ),
            "force_closure_origin_residual": (
                report["force_closure"]["force_closure_origin_residual"]
                if isinstance(report.get("force_closure"), dict)
                else None
            ),
            "positive_wrench_count": (
                report["force_closure"]["positive_wrench_count"]
                if isinstance(report.get("force_closure"), dict)
                else None
            ),
            "positive_wrench_rank": (
                report["force_closure"]["positive_wrench_rank"]
                if isinstance(report.get("force_closure"), dict)
                else None
            ),
        }
        for frame_index, report in zip(active_indices, closure_reports)
        if not report["passed"]
    ]
    peak_frame = int(active_indices[len(active_indices) // 2])
    peak_state = contact_states[peak_frame]
    if peak_state is None:
        raise RuntimeError("peak contact frame lacks exact pad geometry")
    metric_contact_path = output_dir / "metric-contact.json"
    metric_contact_path.write_text(
        json.dumps(
            {
                "camera_frame": frame_contract.camera_frame,
                "metric_frame": frame_contract.metric_frame,
                "timeline": frame_contract.timeline,
                "fps": frame_contract.fps,
                "intrinsics_pixels": {
                    "fx": frame_contract.fx_pixels,
                    "fy": frame_contract.fy_pixels,
                    "cx": frame_contract.cx_pixels,
                    "cy": frame_contract.cy_pixels,
                },
                "metric_scale_source": frame_contract.metric_scale_source,
                **_jsonable(peak_state),
                "depth_source": "mujoco-depth-buffer",
                "force_source": "exact-pad-friction-cone-coupled-inverse-rod-v3",
                "occlusion_order_known": True,
                "source_frame_index": peak_frame,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    metadata = _video_metadata(args.ffprobe.expanduser().resolve(), video_path)
    video_passed = (
        metadata["frames"] == contract.frames
        and metadata["width"] == args.width
        and metadata["height"] == args.height
        and abs(float(metadata["fps"]) - contract.fps) <= 1e-6
    )
    repaired_contact_frames = contact_repair_attempts > 0
    contact_projection = {
        "requested_frames": int(np.count_nonzero(contact_repair_requested)),
        "repaired_frames": int(np.count_nonzero(repaired_contact_frames)),
        "unrepaired_frames": int(
            np.count_nonzero(
                contact_repair_requested & ~repaired_contact_frames
            )
        ),
        "maximum_attempts": int(np.max(contact_repair_attempts)),
        "maximum_offset_m": float(
            np.max(np.linalg.norm(contact_repair_offsets, axis=1))
        ),
        "candidate_offsets": len(contact_repair_candidates),
        "passed": (
            not bool(
                np.any(
                    contact_repair_requested & ~repaired_contact_frames
                )
            )
            and float(
                np.max(np.linalg.norm(contact_repair_offsets, axis=1))
            )
            <= 0.007
        ),
    }
    machine_acceptance = (
        physical_decision["status"] == "WORKING"
        and hand_validation["passed"]
        and causal["passed"]
        and closure_passed
        and contact_projection["passed"]
        and video_passed
    )
    elapsed = time.perf_counter() - started
    artifacts = {
        "video": video_path,
        "metric_camera_samples": camera_samples_path,
        "metric_camera_report": camera_report_path,
        "robot_trajectory": robot_trajectory_path,
        "stem_centerlines": stem_path,
        "contact_forces": forces_path,
        "right_hand_kinematics": hand_path,
        "metric_contact": metric_contact_path,
        "robot_identity_reference": identity_reference_path,
    }
    exact_contact_counts = [
        int(contact_states[int(index)]["contacting_fingertips"])
        for index in active_indices
        if contact_states[int(index)] is not None
    ]
    exact_contact_gaps = [
        float(gap)
        for index in active_indices
        if contact_states[int(index)] is not None
        for gap in contact_states[int(index)]["surface_gaps_m"]
    ]
    closure_ranks = [
        (
            int(report["force_closure"]["linearized_grasp_matrix_rank"])
            if isinstance(report.get("force_closure"), dict)
            else 0
        )
        for report in closure_reports
    ]
    package_versions = {}
    for package in ("mujoco", "numpy", "opencv-python", "scipy"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "honest_status": "PARTIAL",
        "bundle_id": bundle_id,
        "source_video_sha256": source_video_sha256,
        "simulated_physical_bundle_status": physical_decision["status"],
        "machine_acceptance_passed": machine_acceptance,
        "scope": (
            "Calibrated simulated flower manipulation with exact robot assets and "
            "solver ground truth; not a metric reconstruction of the Pexels video."
        ),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": package_versions,
        "seed": args.seed,
        "gpu": {
            "physical_index": args.gpu,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "inventory_before": inventory,
            "selected": selected_gpu,
        },
        "git": {
            "remote_workspace": _git_state(),
            "source_git_head": args.source_git_head,
            "source_git_status_sha256": args.source_git_status_sha256,
        },
        "timeline": {
            "frames": contract.frames,
            "fps": contract.fps,
            "duration_seconds": contract.frames / contract.fps,
            "approach_end_frame": contract.approach_end_frame,
            "release_frame": contract.release_frame,
            "contact_frames": int(np.count_nonzero(schedule["contact_active"])),
        },
        "assets": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "registry_sha256": ASSET_HASHES[name],
            }
            for name, path in inputs.items()
        },
        "physical_pipeline": {
            **physical_decision,
            "stages": stages,
        },
        "additional_gates": {
            "fixed_articulated_hand": hand_validation,
            "bounded_exact_contact_projection": contact_projection,
            "causal_stem_response": causal,
            "force_closure": {
                "frames_evaluated": len(closure_reports),
                "frames_passed": sum(
                    bool(report["passed"]) for report in closure_reports
                ),
                "all_passed": closure_passed,
                "contact_source": "exact_sharpa_elastomer_mesh_vertices",
                "minimum_contacting_fingertips": min(
                    exact_contact_counts,
                    default=0,
                ),
                "maximum_exact_surface_gap_m": max(
                    exact_contact_gaps,
                    default=float("inf"),
                ),
                "minimum_grasp_matrix_rank": min(closure_ranks, default=0),
                "failed_frames": failed_closure_frames,
            },
            "video": {**metadata, "passed": video_passed},
        },
        "performance": {
            "wall_seconds": elapsed,
            "rendered_frames_per_second": contract.frames / elapsed,
        },
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in artifacts.items()
        },
        "remaining_blockers": [
            "The original Pexels clip still has no independent metric camera observation.",
            "The simulation requires full-resolution human visual review before publication.",
            "Simulation success cannot be relabeled as real-scene or real-robot success.",
            "A real replacement candidate still requires geometry-conditioned appearance transfer.",
        ],
    }
    report_path = output_dir / "simulation-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_json_atomic(
        heartbeat_path,
        {
            "status": "SUCCESS" if machine_acceptance else "PARTIAL",
            "stage": "complete",
            "machine_acceptance_passed": machine_acceptance,
            "report": str(report_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if machine_acceptance else 2


if __name__ == "__main__":
    raise SystemExit(main())
