#!/usr/bin/env python3
"""Render an alpha-matted 3D robot with contact-oriented wrist calibration."""

from __future__ import annotations

import argparse
import hashlib
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from build_articulated_flower_robot_demo import (  # noqa: E402
    AttachedHandRenderer,
    G1IkRenderer,
    _default_hand_points,
)
from build_flower_robot_demo import _gpu_inventory, _select_gpu  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose_landmark_is_valid(np: Any, wrist: Any) -> bool:
    coordinates = np.asarray((wrist.x, wrist.y), dtype=np.float64)
    visibility = float(getattr(wrist, "visibility", 1.0))
    return bool(
        np.all(np.isfinite(coordinates))
        and math.isfinite(visibility)
        and visibility >= 0.2
    )


def _validated_hand_points(np: Any, points: Any) -> Any | None:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        return None
    return points if float(np.max(np.abs(points))) <= 10.0 else None


def _writer(ffmpeg: str, path: Path, fps: float, *, lossless: bool = False) -> Any:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "640x480",
        "-r",
        f"{fps:.12g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "0" if lossless else "14",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def _pose_targets(
    np: Any,
    landmarks: Any,
    previous: dict[str, Any] | None,
    *,
    center_x: float,
    horizontal_gain: float,
    vertical_origin: float,
    vertical_gain: float,
    observation_weight: float,
) -> dict[str, Any]:
    defaults = {
        "left": np.asarray((0.30, 0.18, 0.90), dtype=np.float64),
        "right": np.asarray((0.30, -0.18, 0.90), dtype=np.float64),
    }
    fallback = previous if previous is not None else defaults
    targets = {}
    for side, wrist_index in (("left", 15), ("right", 16)):
        if len(landmarks) <= wrist_index or not _pose_landmark_is_valid(
            np, landmarks[wrist_index]
        ):
            targets[side] = fallback[side].copy()
            continue
        wrist = landmarks[wrist_index]
        target = np.asarray(
            (
                0.30,
                (float(np.clip(wrist.x, 0.0, 1.0)) - center_x) * horizontal_gain,
                vertical_origin
                - float(np.clip(wrist.y, 0.0, 1.0)) * vertical_gain,
            ),
            dtype=np.float64,
        )
        if previous is not None:
            target = observation_weight * target + (1.0 - observation_weight) * previous[side]
        targets[side] = target
    return targets


def _alpha_overlay(np: Any, base: Any, rgb: Any, mask: Any) -> Any:
    alpha = mask.astype(np.float32) / 255.0
    return np.rint(
        rgb.astype(np.float32) * alpha[..., None]
        + base.astype(np.float32) * (1.0 - alpha[..., None])
    ).astype(np.uint8)


def _centroid(np: Any, mask: Any) -> list[float] | None:
    rows, columns = np.nonzero(mask > 0)
    if not len(rows):
        return None
    return [float(np.mean(columns)), float(np.mean(rows))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--g1-model", type=Path, required=True)
    parser.add_argument("--sharpa-left-model", type=Path, required=True)
    parser.add_argument("--sharpa-right-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=1024)
    parser.add_argument("--center-x", type=float, default=0.66)
    parser.add_argument("--horizontal-gain", type=float, default=2.0)
    parser.add_argument("--vertical-origin", type=float, default=1.45)
    parser.add_argument("--vertical-gain", type=float, default=0.95)
    parser.add_argument("--observation-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    inputs = {
        "source": args.source.expanduser().resolve(),
        "g1_model": args.g1_model.expanduser().resolve(),
        "sharpa_left_model": args.sharpa_left_model.expanduser().resolve(),
        "sharpa_right_model": args.sharpa_right_model.expanduser().resolve(),
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise ValueError(f"{name} does not exist: {path}")
    if not 0.0 < args.observation_weight <= 1.0:
        raise ValueError("observation weight must be in (0, 1]")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    inventory = _gpu_inventory()
    selected = _select_gpu(inventory, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")

    import cv2
    import mediapipe as mp
    import mujoco
    import numpy as np

    capture = cv2.VideoCapture(str(inputs["source"]))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or fps <= 0 or expected_frames <= 0:
        raise RuntimeError("source video metadata is invalid")
    g1 = G1IkRenderer(mujoco, np, inputs["g1_model"])
    hands = {
        "left": AttachedHandRenderer(
            mujoco, np, inputs["sharpa_left_model"], "sharpa", "left"
        ),
        "right": AttachedHandRenderer(
            mujoco, np, inputs["sharpa_right_model"], "sharpa", "right"
        ),
    }
    pose_detector = mp.solutions.pose.Pose(
        model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    hand_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    ffmpeg = subprocess.run(
        ["which", "ffmpeg"], check=True, capture_output=True, text=True
    ).stdout.strip()
    outputs = {
        "rgb": output_dir / "robot-rgb-black.mp4",
        "mask": output_dir / "robot-mask.mp4",
    }
    writers = {
        "rgb": _writer(ffmpeg, outputs["rgb"], fps),
        "mask": _writer(ffmpeg, outputs["mask"], fps, lossless=True),
    }
    previous_targets = None
    previous_hand_points = {
        "left": _default_hand_points(np),
        "right": _default_hand_points(np),
    }
    wrist_trace = []
    joint_names: tuple[str, ...] | None = None
    joint_limits = None
    joint_trajectory = []
    floating_base_trajectory = []
    carried_pose_frames = 0
    carried_hand_frames = 0
    frame_count = 0
    try:
        while True:
            ok, source_frame = capture.read()
            if not ok:
                break
            rgb_source = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
            pose_result = pose_detector.process(rgb_source)
            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks.landmark
                targets = _pose_targets(
                    np,
                    landmarks,
                    previous_targets,
                    center_x=args.center_x,
                    horizontal_gain=args.horizontal_gain,
                    vertical_origin=args.vertical_origin,
                    vertical_gain=args.vertical_gain,
                    observation_weight=args.observation_weight,
                )
                valid_sides = sum(
                    _pose_landmark_is_valid(np, landmarks[index])
                    for index in (15, 16)
                    if len(landmarks) > index
                )
                if valid_sides < 2:
                    carried_pose_frames += 1
            elif previous_targets is not None:
                targets = {side: value.copy() for side, value in previous_targets.items()}
                carried_pose_frames += 1
            else:
                targets = {
                    "left": np.asarray((0.30, 0.18, 0.90)),
                    "right": np.asarray((0.30, -0.18, 0.90)),
                }
                carried_pose_frames += 1
            previous_targets = {side: value.copy() for side, value in targets.items()}

            hand_result = hand_detector.process(rgb_source)
            detected_sides = set()
            for hand, classification in zip(
                hand_result.multi_hand_world_landmarks or (),
                hand_result.multi_handedness or (),
            ):
                side = classification.classification[0].label.lower()
                if side not in {"left", "right"}:
                    continue
                observed = np.asarray(
                    [(point.x, point.y, point.z) for point in hand.landmark],
                    dtype=np.float64,
                )
                points = _validated_hand_points(np, observed)
                if points is None:
                    continue
                previous_hand_points[side] = 0.35 * points + 0.65 * previous_hand_points[side]
                detected_sides.add(side)
            if len(detected_sides) < 2:
                carried_hand_frames += 1

            g1.solve(targets)
            body_rgb, body_mask = g1.render()
            robot_rgb = _alpha_overlay(np, np.zeros_like(body_rgb), body_rgb, body_mask)
            robot_mask = body_mask.copy()
            hand_centroids = {}
            for side in ("left", "right"):
                wrist_position, wrist_quaternion = g1.wrist_pose(side)
                hand_rgb, hand_mask = hands[side].render(
                    previous_hand_points[side], wrist_position, wrist_quaternion
                )
                robot_rgb = _alpha_overlay(np, robot_rgb, hand_rgb, hand_mask)
                robot_mask = np.maximum(robot_mask, hand_mask)
                hand_centroids[side] = _centroid(np, hand_mask)
            component_states = [("g1", *g1.generalized_joint_state())]
            for side in ("left", "right"):
                component_states.append(
                    (f"sharpa_{side}", *hands[side].generalized_joint_state())
                )
            frame_names = tuple(
                f"{component}:{name}"
                for component, names, _, _ in component_states
                for name in names
            )
            frame_limits = np.concatenate(
                [limits for _, _, _, limits in component_states], axis=0
            )
            frame_positions = np.concatenate(
                [values for _, _, values, _ in component_states], axis=0
            )
            if joint_names is None:
                joint_names = frame_names
                joint_limits = frame_limits
            elif frame_names != joint_names or not np.array_equal(frame_limits, joint_limits):
                raise RuntimeError("robot generalized-coordinate topology changed across frames")
            joint_trajectory.append(frame_positions)
            base_translation, base_quaternion_wxyz = g1.floating_base_pose()
            floating_base_trajectory.append(
                np.concatenate((base_translation, base_quaternion_wxyz))
            )
            mask_bgr = np.repeat(robot_mask[..., None], 3, axis=2)
            for name, frame in (("rgb", robot_rgb), ("mask", mask_bgr)):
                if writers[name].stdin is None:
                    raise RuntimeError(f"{name} writer stdin closed")
                writers[name].stdin.write(frame.tobytes())
            wrist_trace.append(
                {
                    "frame": frame_count,
                    "source_frame": "camera:source_pixels",
                    "robot_frame": "robot:base",
                    "targets_robot_base": {
                        side: [float(value) for value in targets[side]]
                        for side in ("left", "right")
                    },
                    "rendered_hand_centroids": hand_centroids,
                }
            )
            frame_count += 1
    finally:
        capture.release()
        pose_detector.close()
        hand_detector.close()
        g1.close()
        for hand in hands.values():
            hand.close()
        return_codes = {}
        for name, writer in writers.items():
            if writer.stdin is not None:
                writer.stdin.close()
            return_codes[name] = writer.wait()
    if any(return_codes.values()):
        raise RuntimeError(f"ffmpeg writers failed: {return_codes}")
    if frame_count != expected_frames:
        raise RuntimeError(f"processed {frame_count} frames, expected {expected_frames}")
    for output in outputs.values():
        subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(output), "-f", "null", "-"], check=True
        )
    (output_dir / "wrist-trace.json").write_text(
        json.dumps(wrist_trace, indent=2, sort_keys=True) + "\n"
    )
    if joint_names is None or joint_limits is None or not joint_trajectory:
        raise RuntimeError("no generalized-coordinate trajectory was exported")
    joint_positions = np.stack(joint_trajectory).astype(np.float64)
    joint_velocities = np.gradient(joint_positions, 1.0 / fps, axis=0)
    robot_trajectory_path = output_dir / "robot-trajectory.npz"
    np.savez_compressed(
        robot_trajectory_path,
        embodiment_id=np.asarray("unitree-g1-sharpa-wave"),
        robot_base_frame=np.asarray("robot_base:g1"),
        timeline=np.asarray("frame:source_video"),
        fps=np.asarray(fps, dtype=np.float64),
        source_frame_indices=np.arange(frame_count, dtype=np.int32),
        joint_names=np.asarray(joint_names),
        joint_limits_rad=joint_limits,
        joint_positions_rad=joint_positions,
        joint_velocities_rad_s=joint_velocities,
        floating_base_xyz_wxyz=np.stack(floating_base_trajectory),
        trajectory_evidence=np.asarray("physics_solver_estimate"),
        reprojection_evidence=np.asarray("NOT_AVAILABLE"),
    )
    packages = {}
    for package in ("mediapipe", "mujoco", "numpy", "opencv-python", "scipy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    accepted_structure = (
        g1.maximum_ik_error <= 0.12
        and g1.maximum_joint_step <= 0.12 + 1e-9
        and all(math.isfinite(value) for value in (g1.maximum_ik_error, g1.maximum_joint_step))
    )
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "honest_status": "PARTIAL",
        "method": "contact_calibrated_source_wrist_to_mujoco_robot_ik",
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "seed": args.seed,
        "gpu": {
            "physical_index": args.gpu,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "inventory_before": inventory,
            "selected": selected,
        },
        "coordinate_frames": {
            "source": "camera:source_pixels",
            "kinematics": "robot:base",
            "render": "camera:robot_render_pixels",
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "parameters": {
            "center_x": args.center_x,
            "horizontal_gain": args.horizontal_gain,
            "vertical_origin": args.vertical_origin,
            "vertical_gain": args.vertical_gain,
            "observation_weight": args.observation_weight,
        },
        "tracking": {
            "frames": frame_count,
            "carried_pose_frames": carried_pose_frames,
            "carried_hand_frames": carried_hand_frames,
        },
        "kinematics": {
            "maximum_ik_error_m": g1.maximum_ik_error,
            "maximum_joint_step_rad": g1.maximum_joint_step,
            "structural_gate_passed": accepted_structure,
        },
        "outputs": {
            **{
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in outputs.items()
            },
            "wrist_trace": str(output_dir / "wrist-trace.json"),
            "robot_trajectory": {
                "path": str(robot_trajectory_path),
                "sha256": _sha256(robot_trajectory_path),
                "joint_count": len(joint_names),
                "frames": frame_count,
                "includes_static_lower_body_and_waist": True,
                "floating_base_pose_exported_separately": True,
                "trajectory_evidence": "physics_solver_estimate",
                "reprojection_evidence": "NOT_AVAILABLE",
            },
        },
        "entrypoint": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "limitations": [
            "Source wrist observations condition IK, but flower contact and collision are not yet enforced.",
            "The complete solver q sequence is exported, but it remains rejected until render reprojection is validated against metric observations.",
            "The render is an intermediate alpha layer, not a visually accepted scene replacement.",
            "WORKING is forbidden until the final full scene passes every semantic hard gate.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if accepted_structure else 2


if __name__ == "__main__":
    raise SystemExit(main())
