#!/usr/bin/env python3
"""Render a source-timed dual RealMan RM65-B visual replay in MuJoCo.

This is intentionally a visual comparison tool.  It turns reviewed 2-D/video
anchors into a smooth Cartesian proxy, solves only position IK for the official
RM65-B visual meshes, and renders the supplied ChangingTek AG2F90-C asset
silhouette.  It is not a cloth simulator, robot controller, or physical
validation result.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np


def _vec(value: str | None, default: str = "0 0 0") -> str:
    return value if value else default


def _mesh_path(urdf: Path, filename: str) -> Path:
    marker = "meshes/"
    if marker not in filename:
        raise ValueError(f"Unsupported RM65 mesh reference: {filename}")
    relative = Path("meshes") / filename.split(marker, 1)[1]
    # Official RM65 packages have appeared both as RM65-B/urdf/RM65-B.urdf
    # and RM65-B/urdf/RM65-B/RM65-B.urdf.  Resolve from the closest package
    # ancestor rather than assuming one particular exported layout.
    for parent in (urdf.parent, *urdf.parents):
        candidate = parent / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve {filename} relative to {urdf}")


def _build_arm(
    root: ET.Element,
    urdf: Path,
    prefix: str,
    base_pos: tuple[float, float, float],
    base_rpy: tuple[float, float, float],
    asset_names: dict[Path, str],
    gripper_meshes: dict[str, Path],
) -> None:
    robot = ET.parse(urdf).getroot()
    links = {link.attrib["name"]: link for link in robot.findall("link")}
    children: dict[str, list[ET.Element]] = {}
    child_names: set[str] = set()
    for joint in robot.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        children.setdefault(parent.attrib["link"], []).append(joint)
        child_names.add(child.attrib["link"])
    base_link = next(name for name in links if name not in child_names)

    def add_visuals(body: ET.Element, link_name: str) -> None:
        for visual in links[link_name].findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None or "filename" not in mesh.attrib:
                continue
            path = _mesh_path(urdf, mesh.attrib["filename"])
            name = asset_names.setdefault(path, f"mesh_{len(asset_names):03d}")
            origin = visual.find("origin")
            ET.SubElement(
                body,
                "geom",
                {
                    "type": "mesh",
                    "mesh": name,
                    "pos": _vec(origin.attrib.get("xyz") if origin is not None else None),
                    "euler": _vec(origin.attrib.get("rpy") if origin is not None else None),
                    "rgba": "0.93 0.93 0.94 1",
                    "contype": "0",
                    "conaffinity": "0",
                },
            )

    def visit(parent: ET.Element, link_name: str) -> None:
        add_visuals(parent, link_name)
        for joint in children.get(link_name, []):
            child_name = joint.find("child").attrib["link"]
            origin = joint.find("origin")
            body = ET.SubElement(
                parent,
                "body",
                {
                    "name": f"{prefix}_{child_name}",
                    "pos": _vec(origin.attrib.get("xyz") if origin is not None else None),
                    "euler": _vec(origin.attrib.get("rpy") if origin is not None else None),
                },
            )
            if joint.attrib.get("type") == "revolute":
                limit = joint.find("limit")
                axis = joint.find("axis")
                ET.SubElement(
                    body,
                    "joint",
                    {
                        "name": f"{prefix}_{joint.attrib['name']}",
                        "type": "hinge",
                        "axis": _vec(axis.attrib.get("xyz") if axis is not None else None, "0 0 1"),
                        "range": f"{limit.attrib['lower']} {limit.attrib['upper']}",
                        "damping": "1.0",
                    },
                )
            visit(body, child_name)

    arm = ET.SubElement(
        root,
        "body",
        {
            "name": f"{prefix}_base",
            "pos": " ".join(str(item) for item in base_pos),
            "euler": " ".join(str(item) for item in base_rpy),
        },
    )
    visit(arm, base_link)
    wrist = root.find(f".//body[@name='{prefix}_link_6']")
    if wrist is None:
        raise RuntimeError("RM65 URDF does not contain link_6")
    # AG2F90-C's gripper_base is already expressed at its mounting origin.
    # The former 85 mm offset left a visible gap between RM65 link_6 and the
    # gripper body.  No optional flange mesh is rendered here, so attach the
    # gripper base directly to the RM65 wrist origin.
    gripper = ET.SubElement(wrist, "body", {"name": f"{prefix}_ag2f90c", "pos": "0 0 0"})

    def add_gripper_geom(body: ET.Element, mesh_key: str) -> None:
        path = gripper_meshes[mesh_key]
        name = asset_names.setdefault(path, f"mesh_{len(asset_names):03d}")
        ET.SubElement(
            body,
            "geom",
            {
                "type": "mesh",
                "mesh": name,
                "rgba": "0.06 0.07 0.08 1",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    def hinged_body(parent: ET.Element, suffix: str, pos: str, euler: str, axis: str, mesh_key: str) -> ET.Element:
        body = ET.SubElement(parent, "body", {"name": f"{prefix}_{suffix}", "pos": pos, "euler": euler})
        ET.SubElement(body, "joint", {"name": f"{prefix}_gripper_{suffix}", "type": "hinge", "axis": axis, "range": "0 1", "damping": "0.1"})
        add_gripper_geom(body, mesh_key)
        return body

    add_gripper_geom(gripper, "base")
    left_1 = hinged_body(gripper, "left_1", "0 -0.030141 0.090147", f"{math.pi / 2} 0 0", "1 0 0", "link_1")
    hinged_body(left_1, "left_in", "0 0.023914 -0.0020253", "0 0 0", "-1 0 0", "in_link")
    left_up = hinged_body(left_1, "left_up", "0 0.054804 -0.0046413", "0 0 0", "-1 0 0", "up_link")
    left_pad = ET.SubElement(left_up, "body", {"name": f"{prefix}_left_pad", "pos": "0 0.04125 -0.02425"})
    add_gripper_geom(left_pad, "pad_link")
    hinged_body(gripper, "left_2", "0 -0.016 0.107", f"{math.pi / 2} 0 0", "1 0 0", "link_2")
    right_1 = hinged_body(gripper, "right_1", "0 0.030141 0.090147", f"{math.pi / 2} 0 {math.pi}", "1 0 0", "link_1")
    hinged_body(right_1, "right_in", "0 0.023914 -0.0020253", "0 0 0", "-1 0 0", "in_link")
    right_up = hinged_body(right_1, "right_up", "0 0.054804 -0.0046413", "0 0 0", "-1 0 0", "up_link")
    right_pad = ET.SubElement(right_up, "body", {"name": f"{prefix}_right_pad", "pos": "0 0.04125 -0.02425"})
    add_gripper_geom(right_pad, "pad_link")
    hinged_body(gripper, "right_2", "0 0.016 0.107", f"{math.pi / 2} 0 {math.pi}", "1 0 0", "link_2")
    ET.SubElement(gripper, "site", {"name": f"{prefix}_eef", "pos": "0 0 0.19", "size": "0.008", "rgba": "0 1 0 1"})


def build_model(
    urdf: Path,
    gripper_dir: Path,
    show_cloth_proxy: bool,
    left_base: tuple[float, float, float] = (0.05, -0.18, 0.0),
    left_base_yaw: float = math.pi / 2,
    right_base: tuple[float, float, float] = (0.50, 0.24, 0.0),
    right_base_yaw: float = math.pi,
    left_base_roll: float = 0.0,
    left_base_pitch: float = 0.0,
    right_base_roll: float = 0.0,
    right_base_pitch: float = 0.0,
    table_half_size: tuple[float, float] = (0.72, 0.52),
    table_center_y: float = 0.14,
) -> mujoco.MjModel:
    mujoco_root = ET.Element("mujoco", {"model": "phiagent_realman_rm65_visual"})
    # URDF origin rpy is a fixed-axis roll/pitch/yaw convention.  MuJoCo uses
    # uppercase letters for fixed axes; lowercase xyz would rotate about the
    # moving body axes and corrupt both the RM65 and AG2F90-C chains.
    ET.SubElement(mujoco_root, "compiler", {"angle": "radian", "eulerseq": "XYZ"})
    ET.SubElement(mujoco_root, "option", {"timestep": "0.01"})
    visual = ET.SubElement(mujoco_root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1024", "offheight": "768"})
    assets = ET.SubElement(mujoco_root, "asset")
    world = ET.SubElement(mujoco_root, "worldbody")
    ET.SubElement(world, "geom", {"type": "plane", "size": "3 3 0.1", "rgba": "0.10 0.13 0.16 1"})
    table_x, table_y = table_half_size
    ET.SubElement(world, "geom", {"type": "box", "pos": f"0 {table_center_y} -0.055", "size": f"{table_x} {table_y} 0.055", "rgba": "0.78 0.76 0.72 1"})
    ET.SubElement(world, "geom", {"type": "box", "pos": f"0 {table_center_y} 0.006", "size": f"{max(table_x - 0.04, 0.05)} {max(table_y - 0.04, 0.05)} 0.008", "rgba": "0.93 0.93 0.92 1"})
    if show_cloth_proxy:
        ET.SubElement(world, "geom", {"name": "cloth_body", "type": "box", "pos": "0 0.16 0.018", "size": "0.23 0.27 0.004", "rgba": "0.69 0.69 0.70 1"})
        ET.SubElement(world, "geom", {"name": "cloth_left_sleeve", "type": "box", "pos": "-0.31 0.20 0.020", "size": "0.12 0.12 0.004", "rgba": "0.06 0.07 0.08 1"})
        ET.SubElement(world, "geom", {"name": "cloth_right_sleeve", "type": "box", "pos": "0.31 0.20 0.020", "size": "0.12 0.12 0.004", "rgba": "0.06 0.07 0.08 1"})
    asset_names: dict[Path, str] = {}
    gripper_meshes = {
        key: gripper_dir / "meshes" / "AG2F90-C" / "collision" / f"{name}.stl"
        for key, name in {
            "base": "base", "link_1": "link_1", "link_2": "link_2",
            "in_link": "in_link", "up_link": "up_link", "pad_link": "pad_link",
        }.items()
    }
    # The source installation is not a symmetric ALOHA layout.  One RM65 is
    # mounted at the near edge and the other at the far-right edge.  Keep the
    # source-conditioned base poses configurable so the same renderer can be
    # used after a real calibration is available.
    _build_arm(world, urdf, "left", left_base, (left_base_roll, left_base_pitch, left_base_yaw), asset_names, gripper_meshes)
    _build_arm(world, urdf, "right", right_base, (right_base_roll, right_base_pitch, right_base_yaw), asset_names, gripper_meshes)
    ET.SubElement(world, "light", {"pos": "0 -0.4 1.8", "dir": "0 0 -1", "diffuse": "1 1 1"})
    for path, name in asset_names.items():
        if not path.exists():
            raise FileNotFoundError(path)
        ET.SubElement(assets, "mesh", {"name": name, "file": str(path)})
    return mujoco.MjModel.from_xml_string(ET.tostring(mujoco_root, encoding="unicode"))


def _joint_dofs(model: mujoco.MjModel, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_joint_{index}") for index in range(1, 7)]
    if any(index < 0 for index in ids):
        raise RuntimeError(f"missing RM65 joints for {prefix}")
    return np.asarray([model.jnt_qposadr[index] for index in ids]), np.asarray([model.jnt_dofadr[index] for index in ids])


def _gripper_qpos(model: mujoco.MjModel, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    suffixes = ("left_1", "left_in", "left_up", "left_2", "right_1", "right_in", "right_up", "right_2")
    multipliers = np.asarray((1.0, 0.1, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0), dtype=np.float64)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_gripper_{suffix}") for suffix in suffixes]
    if any(index < 0 for index in ids):
        raise RuntimeError(f"missing AG2F90-C joints for {prefix}")
    return np.asarray([model.jnt_qposadr[index] for index in ids]), multipliers


def _set_gripper(data: mujoco.MjData, qpos: np.ndarray, multipliers: np.ndarray, command: float) -> None:
    # Public AG2F90-C Xacro: joint angle 0 is closed (~2.5 mm pad-centre
    # separation) and angle 1 is open (~99 mm).  Expose the more convenient
    # command_closed_01 convention used by the recovered source events.
    data.qpos[qpos] = (1.0 - float(np.clip(command, 0.0, 1.0))) * multipliers


def _solve_position_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_indices: np.ndarray,
    dof_indices: np.ndarray,
    site_id: int,
    target: np.ndarray,
) -> None:
    for _ in range(48):
        mujoco.mj_forward(model, data)
        error = target - data.site_xpos[site_id]
        if np.linalg.norm(error) < 0.002:
            break
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacobian, None, site_id)
        active = jacobian[:, dof_indices]
        step = active.T @ np.linalg.solve(active @ active.T + 2e-4 * np.eye(3), error)
        data.qpos[qpos_indices] += np.clip(step, -0.08, 0.08)
        for joint_id, qpos_index in zip(
            [model.dof_jntid[dof_index] for dof_index in dof_indices], qpos_indices
        ):
            data.qpos[qpos_index] = np.clip(
                data.qpos[qpos_index], model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1]
            )


def _targets(proposal: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(proposal, allow_pickle=False)
    left = np.asarray(payload["left_pos"], dtype=np.float64)
    right = np.asarray(payload["right_pos"], dtype=np.float64)
    left_grip = np.asarray(payload["left_gripper"], dtype=np.float64)
    right_grip = np.asarray(payload["right_gripper"], dtype=np.float64)
    phases = np.asarray(payload["phase"]).astype(str)
    # The previous planar proxy is used only as a reviewed video-coordinate
    # curve.  This affine placement is tuned for the two real RM65 bases above.
    def convert(points: np.ndarray, side: str) -> np.ndarray:
        output = points.copy()
        output[:, 0] *= 1.25
        output[:, 1] = points[:, 1] + 0.12
        output[:, 2] = points[:, 2] - 0.11
        output[:, 0] += -0.03 if side == "left" else 0.03
        return output
    return convert(left, "left"), convert(right, "right"), left_grip, right_grip, phases


def _animate_tshirt_proxy(model: mujoco.MjModel, frame: int, frames: int) -> None:
    """Make the scene state follow the reviewed source phase boundaries.

    This is deliberately only a visual garment proxy: a grey torso and two
    dark sleeves fold inward, then become a compact bundle.  The robot pose
    remains driven by the reviewed source-time Cartesian proposal.
    """
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cloth_body")
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cloth_left_sleeve")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cloth_right_sleeve")
    phase = min(frame / max(frames - 1, 1), 1.0)
    sleeve_fold = np.clip((phase - 0.37) / 0.18, 0.0, 1.0)
    body_fold = np.clip((phase - 0.62) / 0.16, 0.0, 1.0)
    bundle = np.clip((phase - 0.79) / 0.15, 0.0, 1.0)
    sleeve_x = (1.0 - sleeve_fold) * 0.275 + sleeve_fold * 0.118
    model.geom_pos[left] = (-sleeve_x * (1.0 - bundle), 0.205 + 0.035 * bundle, 0.020)
    model.geom_pos[right] = (sleeve_x * (1.0 - bundle), 0.205 + 0.035 * bundle, 0.020)
    model.geom_size[left] = (0.115 * (1.0 - 0.25 * sleeve_fold), 0.12, 0.004)
    model.geom_size[right] = (0.115 * (1.0 - 0.25 * sleeve_fold), 0.12, 0.004)
    model.geom_pos[body] = (0.0, 0.16 + 0.075 * body_fold, 0.018)
    model.geom_size[body] = (0.23, 0.27 * (1.0 - 0.46 * body_fold), 0.004)


def _camera_frame(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    azimuth: float,
    elevation: float,
    distance: float,
    lookat: tuple[float, float, float],
) -> np.ndarray:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rm65-urdf", type=Path, required=True)
    parser.add_argument("--ag2f90c-dir", type=Path, required=True)
    parser.add_argument("--source-proposal", type=Path)
    parser.add_argument("--state-npz", type=Path, help="Recovered per-frame left/right q and gripper commands.")
    parser.add_argument("--source-video", type=Path, help="If supplied, also write a same-frame side-by-side comparison.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    # Approximate the observed reference view: near-left arm in the foreground
    # and far-right arm entering from the opposite side, rather than a frontal
    # symmetric tabletop view.
    parser.add_argument("--camera-azimuth", type=float, default=45.0)
    parser.add_argument("--camera-elevation", type=float, default=-35.0)
    parser.add_argument("--camera-distance", type=float, default=1.45)
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
    parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.0, 0.12, 0.18))
    parser.add_argument("--show-cloth-proxy", action="store_true", help="Render the non-physical garment proxy; off by default.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.state_npz is None and args.source_proposal is None:
        raise SystemExit("one of --state-npz or --source-proposal is required")
    model = build_model(
        args.rm65_urdf,
        args.ag2f90c_dir,
        args.show_cloth_proxy,
        tuple(args.left_base),
        args.left_base_yaw,
        tuple(args.right_base),
        args.right_base_yaw,
        args.left_base_roll,
        args.left_base_pitch,
        args.right_base_roll,
        args.right_base_pitch,
        tuple(args.table_half_size),
        args.table_center_y,
    )
    data = mujoco.MjData(model)
    state = np.load(args.state_npz, allow_pickle=False) if args.state_npz else None
    if state is None:
        left, right, left_grip, right_grip, phases = _targets(args.source_proposal)
        left_q = right_q = left_command = right_command = None
    else:
        left_q = np.asarray(state["left_q"], dtype=np.float64)
        right_q = np.asarray(state["right_q"], dtype=np.float64)
        left_command = np.asarray(state["left_gripper_command"], dtype=np.float64)
        right_command = np.asarray(state["right_gripper_command"], dtype=np.float64)
        phases = np.asarray(state["phase"]).astype(str)
        left = np.asarray(state["left_target_xyz"], dtype=np.float64)
        right = np.asarray(state["right_target_xyz"], dtype=np.float64)
        left_grip = right_grip = None
        left_tip_px = right_tip_px = None
    if state is not None:
        left_tip_px = np.asarray(state["left_tip_px"], dtype=np.float64) if "left_tip_px" in state.files else None
        right_tip_px = np.asarray(state["right_tip_px"], dtype=np.float64) if "right_tip_px" in state.files else None
    left_qpos, left_dofs = _joint_dofs(model, "left")
    right_qpos, right_dofs = _joint_dofs(model, "right")
    left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_eef")
    right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_eef")
    left_grip_qpos, left_grip_multipliers = _gripper_qpos(model, "left")
    right_grip_qpos, right_grip_multipliers = _gripper_qpos(model, "right")
    left_residuals: list[float] = []
    right_residuals: list[float] = []
    source_cap = cv2.VideoCapture(str(args.source_video)) if args.source_video else None
    comparison_writer = None
    if source_cap is not None:
        comparison_writer = cv2.VideoWriter(
            str(args.output_dir / "source_vs_realman_synchronized_replay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2048, 768)
        )
    with mujoco.Renderer(model, height=768, width=1024) as renderer:
        writer = imageio.get_writer(args.output_dir / "realman_rm65_ag2f90c_visual_replay.mp4", fps=args.fps, codec="libx264")
        try:
            for index, (left_target, right_target) in enumerate(zip(left, right)):
                if state is None:
                    _solve_position_ik(model, data, left_qpos, left_dofs, left_site, left_target)
                    _solve_position_ik(model, data, right_qpos, right_dofs, right_site, right_target)
                else:
                    data.qpos[left_qpos] = left_q[index]
                    data.qpos[right_qpos] = right_q[index]
                if args.show_cloth_proxy:
                    _animate_tshirt_proxy(model, index, len(left))
                current_left_command = left_command[index] if state is not None else (0.035 - left_grip[index]) / (0.035 - 0.0084)
                current_right_command = right_command[index] if state is not None else (0.035 - right_grip[index]) / (0.035 - 0.0084)
                _set_gripper(data, left_grip_qpos, left_grip_multipliers, current_left_command)
                _set_gripper(data, right_grip_qpos, right_grip_multipliers, current_right_command)
                mujoco.mj_forward(model, data)
                left_residuals.append(float(np.linalg.norm(data.site_xpos[left_site] - left_target)))
                right_residuals.append(float(np.linalg.norm(data.site_xpos[right_site] - right_target)))
                frame = _camera_frame(
                    renderer,
                    data,
                    args.camera_azimuth,
                    args.camera_elevation,
                    args.camera_distance,
                    tuple(args.camera_lookat),
                )
                frame = cv2.putText(frame, "RealMan RM65-B + ChangingTek AG2F90-C | visual replay", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
                frame = cv2.putText(frame, f"source phase: {phases[index]}", (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 230, 255), 1, cv2.LINE_AA)
                writer.append_data(frame)
                if comparison_writer is not None:
                    ok, source_frame = source_cap.read()
                    if ok:
                        cv2.putText(source_frame, f"source | frame {index:03d}", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
                        if left_tip_px is not None and right_tip_px is not None:
                            cv2.circle(source_frame, tuple(np.round(left_tip_px[index]).astype(int)), 7, (0, 220, 0), -1)
                            cv2.circle(source_frame, tuple(np.round(right_tip_px[index]).astype(int)), 7, (0, 80, 255), -1)
                        cv2.putText(source_frame, f"grasp state L/R: {current_left_command:.2f} / {current_right_command:.2f}", (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
                        rendered_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.putText(rendered_bgr, f"MuJoCo q/g replay | frame {index:03d}", (24, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(rendered_bgr, f"closed L/R: {current_left_command:.2f} / {current_right_command:.2f}", (24, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
                        comparison_writer.write(np.hstack((source_frame, rendered_bgr)))
        finally:
            writer.close()
            if source_cap is not None:
                source_cap.release()
            if comparison_writer is not None:
                comparison_writer.release()
    probe = mujoco.MjData(model)
    open_qpos, open_multipliers = _gripper_qpos(model, "left")
    left_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_left_pad")
    right_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_right_pad")
    pad_separation = {}
    for label, command in (("open_command_0", 0.0), ("closed_command_1", 1.0)):
        _set_gripper(probe, open_qpos, open_multipliers, command)
        mujoco.mj_forward(model, probe)
        pad_separation[label] = float(np.linalg.norm(probe.xpos[left_pad] - probe.xpos[right_pad]))
    manifest = {
        "schema_version": "phiagent-realman-visual-replay/1.0",
        "source_proposal": str(args.source_proposal.resolve()) if args.source_proposal else None,
        "state_npz": str(args.state_npz.resolve()) if args.state_npz else None,
        "robot": "official RealMan RM65-B URDF visual meshes",
        "gripper": "ChangingTek AG2F90-C public mesh assets",
        "gripper_pad_center_separation_m": pad_separation,
        "timing": {"frames": int(len(left)), "fps": args.fps},
        "camera": {"azimuth": args.camera_azimuth, "elevation": args.camera_elevation, "distance": args.camera_distance, "lookat": args.camera_lookat},
        "base_poses": {"left_xyz_rpy": [*args.left_base, args.left_base_roll, args.left_base_pitch, args.left_base_yaw], "right_xyz_rpy": [*args.right_base, args.right_base_roll, args.right_base_pitch, args.right_base_yaw]},
        "table": {"half_size_xy": args.table_half_size, "center_y": args.table_center_y},
        "show_cloth_proxy": args.show_cloth_proxy,
        "visual_ik_position_residual_m": {
            "left_mean": float(np.mean(left_residuals)),
            "left_max": float(np.max(left_residuals)),
            "right_mean": float(np.mean(right_residuals)),
            "right_max": float(np.max(right_residuals)),
        },
        "claim_boundary": "source-conditioned visual replay only; no cloth dynamics, contact, collision, or real-robot execution claim",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
