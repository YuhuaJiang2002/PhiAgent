#!/usr/bin/env python3
"""Build a same-scene flower-arranging hand-retargeting visualization."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_inventory() -> list[dict[str, int | str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    inventory = []
    for line in result.stdout.splitlines():
        index, name, total, used, free, utilization = (
            item.strip() for item in line.split(",")
        )
        inventory.append(
            {
                "physical_index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
                "utilization_percent": int(utilization),
            }
        )
    return inventory


def _select_gpu(
    inventory: list[dict[str, int | str]], physical_index: int, minimum_free_mib: int
) -> dict[str, int | str]:
    selected = next(
        (gpu for gpu in inventory if gpu["physical_index"] == physical_index), None
    )
    if selected is None:
        raise RuntimeError(f"physical GPU {physical_index} is not present")
    if int(selected["memory_free_mib"]) < minimum_free_mib:
        raise RuntimeError(
            f"physical GPU {physical_index} has only "
            f"{selected['memory_free_mib']} MiB free; need {minimum_free_mib} MiB"
        )
    return selected


def _unit(np: Any, vector: Any) -> Any:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("hand geometry contains a zero-length segment")
    return vector / norm


def _bend(np: Any, first: Any, middle: Any, last: Any) -> float:
    incoming = _unit(np, middle - first)
    outgoing = _unit(np, last - middle)
    return math.acos(float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0)))


def _finger_bends(np: Any, points: Any) -> dict[str, tuple[float, float, float]]:
    return {
        "thumb": (
            _bend(np, points[0], points[1], points[2]),
            _bend(np, points[1], points[2], points[3]),
            _bend(np, points[2], points[3], points[4]),
        ),
        "index": (
            _bend(np, points[0], points[5], points[6]),
            _bend(np, points[5], points[6], points[7]),
            _bend(np, points[6], points[7], points[8]),
        ),
        "middle": (
            _bend(np, points[0], points[9], points[10]),
            _bend(np, points[9], points[10], points[11]),
            _bend(np, points[10], points[11], points[12]),
        ),
        "ring": (
            _bend(np, points[0], points[13], points[14]),
            _bend(np, points[13], points[14], points[15]),
            _bend(np, points[14], points[15], points[16]),
        ),
        "pinky": (
            _bend(np, points[0], points[17], points[18]),
            _bend(np, points[17], points[18], points[19]),
            _bend(np, points[18], points[19], points[20]),
        ),
    }


def _clamp(model: Any, joint_id: int, value: float) -> float:
    if not model.jnt_limited[joint_id]:
        return value
    low, high = model.jnt_range[joint_id]
    return float(max(low, min(high, value)))


def _pose_from_landmarks(
    mujoco: Any, np: Any, model: Any, vendor: str, points: Any
) -> dict[int, float]:
    bends = _finger_bends(np, points)
    values: dict[str, float] = {}
    if vendor == "sharpa":
        thumb = bends["thumb"]
        values.update(
            {
                "right_thumb_CMC_FE": thumb[0],
                "right_thumb_CMC_AA": 0.0,
                "right_thumb_MCP_FE": thumb[1],
                "right_thumb_MCP_AA": 0.0,
                "right_thumb_IP": thumb[2],
            }
        )
        for finger in ("index", "middle", "ring"):
            mcp, pip, dip = bends[finger]
            values.update(
                {
                    f"right_{finger}_MCP_FE": mcp,
                    f"right_{finger}_MCP_AA": 0.0,
                    f"right_{finger}_PIP": pip,
                    f"right_{finger}_DIP": dip,
                }
            )
        mcp, pip, dip = bends["pinky"]
        values.update(
            {
                "right_pinky_CMC": 0.0,
                "right_pinky_MCP_FE": mcp,
                "right_pinky_MCP_AA": 0.0,
                "right_pinky_PIP": pip,
                "right_pinky_DIP": dip,
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
        raise ValueError(f"unsupported vendor: {vendor}")

    pose = {}
    for joint_name, value in values.items():
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise RuntimeError(f"{vendor} model is missing joint {joint_name!r}")
        pose[joint_id] = _clamp(model, joint_id, value)
    return pose


class HandRenderer:
    def __init__(
        self,
        mujoco: Any,
        np: Any,
        model_path: Path,
        vendor: str,
        camera: tuple[float, float, float, float, float, float],
    ) -> None:
        self.mujoco = mujoco
        self.np = np
        self.vendor = vendor
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = camera[:3]
        self.camera.distance = camera[3]
        self.camera.azimuth = camera[4]
        self.camera.elevation = camera[5]
        self.hand_geom_ids = self._hand_geom_ids()

    def _hand_geom_ids(self) -> set[int]:
        connected_bodies: set[int] = set()
        for joint_id in range(self.model.njnt):
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if not name:
                continue
            body_id = int(self.model.jnt_bodyid[joint_id])
            while body_id > 0:
                connected_bodies.add(body_id)
                body_id = int(self.model.body_parentid[body_id])
        return {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in connected_bodies
        }

    def render(self, points_3d: Any) -> tuple[Any, Any]:
        pose = _pose_from_landmarks(
            self.mujoco, self.np, self.model, self.vendor, points_3d
        )
        for joint_id, value in pose.items():
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        rgb = self.renderer.render().copy()[:, :, ::-1]
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.camera)
        segmentation = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        mask = self.np.isin(segmentation[:, :, 0], tuple(self.hand_geom_ids))
        if int(mask.sum()) < 100:
            raise RuntimeError(f"{self.vendor} hand segmentation is empty")
        return rgb, mask.astype(self.np.uint8) * 255

    def close(self) -> None:
        self.renderer.close()


def _source_hand_mask(cv2: Any, np: Any, connections: Any, points: Any, size: tuple[int, int]) -> Any:
    width, height = size
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    thickness = max(16, round(palm_width * 0.62))
    mask = np.zeros((height, width), dtype=np.uint8)
    for start, end in connections:
        cv2.line(
            mask,
            tuple(np.rint(points[start]).astype(int)),
            tuple(np.rint(points[end]).astype(int)),
            255,
            thickness,
            cv2.LINE_AA,
        )
    for point in points:
        cv2.circle(
            mask,
            tuple(np.rint(point).astype(int)),
            thickness // 2,
            255,
            -1,
            cv2.LINE_AA,
        )
    palm = np.rint(points[[0, 1, 5, 9, 13, 17]]).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(palm), 255, cv2.LINE_AA)
    kernel_size = max(9, 2 * round(thickness * 0.3) + 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.dilate(mask, kernel)


def _warp_robot(
    cv2: Any,
    np: Any,
    rgb: Any,
    mask: Any,
    target_points: Any,
    vendor: str,
    output_size: tuple[int, int],
) -> tuple[Any, Any]:
    ys, xs = np.nonzero(mask)
    horizontal_span = float(xs.max() - xs.min())
    vertical_span = float(ys.max() - ys.min())
    if vertical_span >= horizontal_span:
        anchor = np.array([float(np.median(xs)), float(ys.max())])
        tip = np.array([float(np.median(xs)), float(ys.min())])
    else:
        anchor = np.array([float(xs.min()), float(np.median(ys))])
        tip = np.array([float(xs.max()), float(np.median(ys))])
    if vendor == "allegro":
        anchor, tip = tip, anchor
    source_vector = tip - anchor
    target_anchor = target_points[0]
    target_tip = max(
        (target_points[index] for index in (8, 12, 16)),
        key=lambda point: float(np.linalg.norm(point - target_anchor)),
    )
    target_vector = target_tip - target_anchor
    source_length = float(np.linalg.norm(source_vector))
    target_length = float(np.linalg.norm(target_vector))
    if source_length < 1 or target_length < 8:
        raise ValueError("cannot align robot hand to degenerate image landmarks")
    angle = math.atan2(target_vector[1], target_vector[0]) - math.atan2(
        source_vector[1], source_vector[0]
    )
    scale = 1.72 * target_length / source_length
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    linear = np.array(((cosine, -sine), (sine, cosine)))
    translation = target_anchor - linear @ anchor
    transform = np.column_stack((linear, translation))
    warped_rgb = cv2.warpAffine(
        rgb,
        transform,
        output_size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_mask = cv2.warpAffine(
        mask,
        transform,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return warped_rgb, warped_mask


def _composite(
    cv2: Any,
    np: Any,
    source: Any,
    source_mask: Any,
    robot_rgb: Any,
    robot_mask: Any,
) -> Any:
    cleaned = cv2.inpaint(source, source_mask, 7, cv2.INPAINT_TELEA)
    base = source.copy()
    base[source_mask > 0] = cleaned[source_mask > 0]
    alpha = cv2.GaussianBlur(robot_mask, (7, 7), 0).astype(np.float32) / 255.0
    result = np.rint(
        robot_rgb.astype(np.float32) * alpha[..., None]
        + base.astype(np.float32) * (1.0 - alpha[..., None])
    ).astype(np.uint8)

    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    saturated_flower = (
        (((hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 95)) & (hsv[:, :, 1] >= 50))
        | ((hsv[:, :, 0] >= 145) & (hsv[:, :, 1] >= 95))
    )
    changed = (source_mask > 0) | (robot_mask > 0)
    preserve = saturated_flower & changed
    result[preserve] = source[preserve]
    return result


def _writer(ffmpeg: str, output: Path, size: tuple[int, int], fps: float) -> subprocess.Popen[bytes]:
    width, height = size
    process = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("failed to open ffmpeg input pipe")
    return process


def _label(cv2: Any, frame: Any, text: str) -> Any:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (18, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sharpa-model", type=Path, required=True)
    parser.add_argument("--allegro-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=1024)
    parser.add_argument("--working-width", type=int, default=960)
    parser.add_argument("--sharpa-revision", required=True)
    parser.add_argument("--allegro-revision", required=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    sharpa_model = args.sharpa_model.expanduser().resolve()
    allegro_model = args.allegro_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for label, path in (
        ("source", source),
        ("Sharpa model", sharpa_model),
        ("Allegro model", allegro_model),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    if args.working_width <= 0:
        raise ValueError("working width must be positive")
    output_dir.mkdir(parents=True)

    inventory = _gpu_inventory()
    selected_gpu = _select_gpu(inventory, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")

    import cv2
    import mediapipe as mp
    import mujoco
    import numpy as np

    ffmpeg = subprocess.run(
        ["which", "ffmpeg"], text=True, capture_output=True, check=True
    ).stdout.strip()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")

    capture = cv2.VideoCapture(str(source))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_width <= 0 or source_height <= 0 or fps <= 0 or expected_frames <= 0:
        raise RuntimeError("source video metadata is invalid")
    working_height = round(source_height * args.working_width / source_width)
    if working_height % 2:
        working_height += 1
    working_size = (args.working_width, working_height)
    comparison_size = (1920, 360)

    sharpa_path = output_dir / "sharpa-flower-arranging.mp4"
    allegro_path = output_dir / "allegro-flower-arranging.mp4"
    comparison_path = output_dir / "human-sharpa-allegro-flower-demo.mp4"
    sharpa_writer = _writer(ffmpeg, sharpa_path, working_size, fps)
    allegro_writer = _writer(ffmpeg, allegro_path, working_size, fps)
    comparison_writer = _writer(ffmpeg, comparison_path, comparison_size, fps)

    sharpa = HandRenderer(
        mujoco, np, sharpa_model, "sharpa", (0, 0, 0.05, 0.55, 135, -30)
    )
    allegro = HandRenderer(
        mujoco, np, allegro_model, "allegro", (0, 0, 0.05, 0.55, 180, -30)
    )
    detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    connections = tuple(mp.solutions.hands.HAND_CONNECTIONS)

    previous_2d = None
    previous_3d = None
    one_hand_frames = 0
    two_hand_frames = 0
    frame_count = 0
    try:
        while True:
            ok, source_frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(
                source_frame, working_size, interpolation=cv2.INTER_AREA
            )
            detection = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_hands = detection.multi_hand_landmarks or ()
            world_hands = detection.multi_hand_world_landmarks or ()
            if not image_hands or len(image_hands) != len(world_hands):
                raise RuntimeError(f"hand detection failed at frame {frame_count}")
            if len(image_hands) == 1:
                one_hand_frames += 1
            else:
                two_hand_frames += 1
            candidates = [
                np.asarray(
                    [
                        (landmark.x * working_size[0], landmark.y * working_size[1])
                        for landmark in image_hand.landmark
                    ],
                    dtype=np.float64,
                )
                for image_hand in image_hands
            ]
            candidates_3d = [
                np.asarray(
                    [(landmark.x, landmark.y, landmark.z) for landmark in hand.landmark],
                    dtype=np.float64,
                )
                for hand in world_hands
            ]
            if previous_2d is None:
                selected_index = max(
                    range(len(candidates)), key=lambda index: candidates[index][0, 0]
                )
            else:
                selected_index = min(
                    range(len(candidates)),
                    key=lambda index: float(
                        np.linalg.norm(candidates[index][0] - previous_2d[0])
                    ),
                )
            points_2d = candidates[selected_index]
            points_3d = candidates_3d[selected_index]
            if previous_2d is not None and previous_3d is not None:
                points_2d = 0.4 * points_2d + 0.6 * previous_2d
                points_3d = 0.4 * points_3d + 0.6 * previous_3d
            previous_2d = points_2d
            previous_3d = points_3d

            source_mask = _source_hand_mask(
                cv2, np, connections, points_2d, working_size
            )
            sharpa_rgb, sharpa_mask = sharpa.render(points_3d)
            allegro_rgb, allegro_mask = allegro.render(points_3d)
            sharpa_rgb, sharpa_mask = _warp_robot(
                cv2,
                np,
                sharpa_rgb,
                sharpa_mask,
                points_2d,
                "sharpa",
                working_size,
            )
            allegro_rgb, allegro_mask = _warp_robot(
                cv2,
                np,
                allegro_rgb,
                allegro_mask,
                points_2d,
                "allegro",
                working_size,
            )
            sharpa_frame = _composite(
                cv2, np, frame, source_mask, sharpa_rgb, sharpa_mask
            )
            allegro_frame = _composite(
                cv2, np, frame, source_mask, allegro_rgb, allegro_mask
            )
            assert sharpa_writer.stdin is not None
            assert allegro_writer.stdin is not None
            assert comparison_writer.stdin is not None
            sharpa_writer.stdin.write(sharpa_frame.tobytes())
            allegro_writer.stdin.write(allegro_frame.tobytes())
            panels = [
                _label(
                    cv2,
                    cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA),
                    "Human source",
                ),
                _label(
                    cv2,
                    cv2.resize(
                        sharpa_frame, (640, 360), interpolation=cv2.INTER_AREA
                    ),
                    "Sharpa Wave (geometric)",
                ),
                _label(
                    cv2,
                    cv2.resize(
                        allegro_frame, (640, 360), interpolation=cv2.INTER_AREA
                    ),
                    "Wonik Allegro (geometric)",
                ),
            ]
            comparison_writer.stdin.write(np.hstack(panels).tobytes())
            frame_count += 1
    finally:
        capture.release()
        detector.close()
        sharpa.close()
        allegro.close()
        return_codes = {}
        for name, writer in (
            ("sharpa", sharpa_writer),
            ("allegro", allegro_writer),
            ("comparison", comparison_writer),
        ):
            if writer.stdin is not None:
                writer.stdin.close()
            return_codes[name] = writer.wait()
        if any(return_codes.values()):
            raise RuntimeError(f"ffmpeg writers failed: {return_codes}")

    if frame_count != expected_frames:
        raise RuntimeError(
            f"processed {frame_count} frames, expected {expected_frames}"
        )
    for output in (sharpa_path, allegro_path, comparison_path):
        subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(output), "-f", "null", "-"],
            check=True,
        )

    packages = {}
    for package in ("mediapipe", "mujoco", "numpy", "opencv-python"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    root = Path(__file__).resolve().parents[1]
    git_commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    git_status_result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    git_available = (
        git_commit_result.returncode == 0 and git_status_result.returncode == 0
    )
    git_state = {
        "available": git_available,
        "commit": git_commit_result.stdout.strip() if git_available else None,
        "status": git_status_result.stdout.splitlines() if git_available else None,
        "error": None
        if git_available
        else (
            git_commit_result.stderr.strip()
            or git_status_result.stderr.strip()
            or "git state unavailable"
        ),
    }
    duration_s = frame_count / fps
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "mediapipe_tracked_mujoco_joint_animation_and_screen_space_replacement",
        "status": "WORKING" if duration_s > 10 else "REJECTED",
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "git": git_state,
        "entrypoint": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "python": platform.python_version(),
        "packages": packages,
        "seed": None,
        "gpu": {
            "physical_index": args.gpu,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "selected": selected_gpu,
            "inventory_before": inventory,
        },
        "coordinate_frames": {
            "mediapipe_landmarks": "camera_normalized_and_camera_relative_metric",
            "compositing": f"image_pixel:{working_size[0]}x{working_size[1]}",
            "robot_joint_animation": "robot_base",
            "alignment": "explicit_camera_relative_landmarks_to_image_pixel_similarity_transform",
        },
        "inputs": {
            "source": str(source),
            "source_sha256": _sha256(source),
            "sharpa_model": str(sharpa_model),
            "sharpa_model_sha256": _sha256(sharpa_model),
            "sharpa_revision": args.sharpa_revision,
            "sharpa_license": "Apache-2.0",
            "allegro_model": str(allegro_model),
            "allegro_model_sha256": _sha256(allegro_model),
            "allegro_revision": args.allegro_revision,
            "allegro_license": "BSD-3-Clause",
        },
        "source_video": {
            "frames": frame_count,
            "fps": fps,
            "duration_seconds": duration_s,
            "source_size": [source_width, source_height],
            "working_size": list(working_size),
            "single_continuous_clip": True,
        },
        "tracking": {
            "strategy": "temporally_nearest_primary_hand_initialized_from_rightmost_wrist",
            "one_hand_frames": one_hand_frames,
            "two_or_more_hand_frames": two_hand_frames,
            "lost_hand_frames": 0,
        },
        "outputs": {
            "sharpa": str(sharpa_path),
            "sharpa_sha256": _sha256(sharpa_path),
            "allegro": str(allegro_path),
            "allegro_sha256": _sha256(allegro_path),
            "comparison": str(comparison_path),
            "comparison_sha256": _sha256(comparison_path),
        },
        "limitations": [
            "This is geometric retargeting and screen-space compositing, not official PhiZero inference.",
            "Only the tracked primary manipulating hand is replaced; the support hand remains human.",
            "The robot hands are driven by landmark-derived joint angles without physical contact simulation.",
            "Flower pixels are color-preserved where possible, but occlusion is not depth-calibrated.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "command.txt").write_text(
        subprocess.list2cmdline([sys.executable, *sys.argv]) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "WORKING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
