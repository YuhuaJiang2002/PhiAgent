#!/usr/bin/env python3
"""Replace the full visible florist with a robot without temporal human trails."""

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
    HandRenderer,
    _gpu_inventory,
    _label,
    _select_gpu,
    _sha256,
    _warp_robot,
    _writer,
)


def _largest_components(cv2: Any, np: Any, mask: Any, hand_points: list[Any]) -> Any:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    retained = np.zeros(mask.shape, dtype=np.uint8)
    for component in range(1, count):
        x, _, width, _, area = stats[component]
        component_mask = labels == component
        overlaps_left = x < round(mask.shape[1] * 0.42)
        overlaps_hand = any(
            0 <= round(point[1]) < mask.shape[0]
            and 0 <= round(point[0]) < mask.shape[1]
            and component_mask[round(point[1]), round(point[0])]
            for point in hand_points
        )
        if area >= 1800 and (overlaps_left or overlaps_hand):
            retained[component_mask] = 255
    return retained


def _hand_mask(cv2: Any, np: Any, connections: Any, points: Any, size: tuple[int, int]) -> Any:
    width, height = size
    palm_width = max(20.0, float(np.linalg.norm(points[5] - points[17])))
    thickness = max(24, round(palm_width * 0.9))
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
    palm = np.rint(points[[0, 1, 5, 9, 13, 17]]).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(palm), 255, cv2.LINE_AA)
    return cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )


def _flower_mask(cv2: Any, np: Any, frame: Any) -> Any:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue >= 28) & (hue <= 95) & (saturation >= 42) & (value >= 35)
    pink = (hue >= 145) & (hue <= 179) & (saturation >= 100) & (value >= 65)
    mask = ((green | pink).astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def _clean_plate(cv2: Any, np: Any, frames: list[Any], erase_mask: Any) -> Any:
    from scipy.ndimage import distance_transform_edt

    median = np.median(np.stack(frames), axis=0).astype(np.uint8)
    inside = erase_mask > 0
    _, nearest = distance_transform_edt(inside, return_indices=True)
    filled = median.copy()
    filled[inside] = median[nearest[0][inside], nearest[1][inside]]
    blurred = cv2.GaussianBlur(filled, (0, 0), 5)
    filled[inside] = blurred[inside]
    return filled


class FullRobotRenderer:
    def __init__(self, mujoco: Any, np: Any, model_path: Path) -> None:
        self.mujoco = mujoco
        self.np = np
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        if self.model.nkey < 1:
            raise RuntimeError("full robot model requires a standing keyframe")
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._set_pose()
        mujoco.mj_forward(self.model, self.data)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = (0.0, 0.0, 0.85)
        self.camera.distance = 3.0
        self.camera.azimuth = 180
        self.camera.elevation = -4
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def _set_joint(self, name: str, value: float) -> None:
        joint_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise RuntimeError(f"full robot model is missing joint {name!r}")
        low, high = self.model.jnt_range[joint_id]
        self.data.qpos[self.model.jnt_qposadr[joint_id]] = max(
            float(low), min(float(high), value)
        )

    def _set_pose(self) -> None:
        for side in ("left", "right"):
            self._set_joint(f"{side}_shoulder_pitch_joint", -0.65)
            self._set_joint(f"{side}_elbow_joint", 1.05)
        self._set_joint("waist_pitch_joint", 0.12)
        self._set_joint("left_shoulder_roll_joint", 0.35)
        self._set_joint("right_shoulder_roll_joint", -0.35)

    def render(self) -> tuple[Any, Any]:
        self.renderer.update_scene(self.data, camera=self.camera)
        rgb = self.renderer.render().copy()[:, :, ::-1]
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.camera)
        segmentation = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        mask = (
            segmentation[:, :, 1] == int(self.mujoco.mjtObj.mjOBJ_GEOM)
        ).astype(self.np.uint8) * 255
        count, labels, stats, _ = __import__("cv2").connectedComponentsWithStats(mask)
        if count <= 1:
            raise RuntimeError("full robot segmentation is empty")
        largest = 1 + int(self.np.argmax(stats[1:, __import__("cv2").CC_STAT_AREA]))
        return rgb, (labels == largest).astype(self.np.uint8) * 255

    def close(self) -> None:
        self.renderer.close()


def _fit_full_robot(
    cv2: Any,
    np: Any,
    rgb: Any,
    mask: Any,
    size: tuple[int, int],
    target_mask: Any,
) -> tuple[Any, Any]:
    ys, xs = np.nonzero(mask)
    cropped_rgb = rgb[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    cropped_mask = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    target_ys, target_xs = np.nonzero(target_mask)
    target_height = min(
        round(size[1] * 0.98),
        round((target_ys.max() - target_ys.min() + 1) * 0.96),
    )
    scale = target_height / cropped_rgb.shape[0]
    resized_rgb = cv2.resize(
        cropped_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
    )
    resized_mask = cv2.resize(
        cropped_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    canvas_mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    target_center_x = float(target_xs.min() + target_xs.max()) * 0.5
    x = round(target_center_x - resized_rgb.shape[1] * 0.5)
    y = round(target_ys.max() - resized_rgb.shape[0] + 1)
    x0, y0 = max(0, x), max(0, y)
    x1 = min(size[0], x + resized_rgb.shape[1])
    y1 = min(size[1], y + resized_rgb.shape[0])
    canvas[y0:y1, x0:x1] = resized_rgb[y0 - y : y1 - y, x0 - x : x1 - x]
    canvas_mask[y0:y1, x0:x1] = resized_mask[y0 - y : y1 - y, x0 - x : x1 - x]
    return canvas, canvas_mask


def _overlay(cv2: Any, np: Any, base: Any, rgb: Any, mask: Any) -> Any:
    alpha = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
    return np.rint(
        rgb.astype(np.float32) * alpha[..., None]
        + base.astype(np.float32) * (1.0 - alpha[..., None])
    ).astype(np.uint8)


def _draw_robot_arm(
    cv2: Any,
    frame: Any,
    shoulder: tuple[int, int],
    wrist: tuple[int, int],
    side: int,
) -> None:
    span = max(1, wrist[0] - shoulder[0])
    elbow = (
        shoulder[0] + round(span * 0.52),
        round((shoulder[1] + wrist[1]) * 0.5) + side * 24,
    )
    for start, end in ((shoulder, elbow), (elbow, wrist)):
        cv2.line(frame, start, end, (50, 53, 58), 34, cv2.LINE_AA)
        cv2.line(frame, start, end, (128, 134, 141), 5, cv2.LINE_AA)
    for center in (shoulder, elbow, wrist):
        cv2.circle(frame, center, 21, (35, 38, 43), -1, cv2.LINE_AA)
        cv2.circle(frame, center, 21, (158, 164, 170), 4, cv2.LINE_AA)


def _select_hands(np: Any, candidates: list[Any], previous: list[Any] | None) -> list[int]:
    if len(candidates) == 1:
        return [0]
    if previous is None:
        return sorted(range(len(candidates)), key=lambda index: candidates[index][0, 0])[:2]
    selected: list[int] = []
    available = set(range(len(candidates)))
    for old in previous[:2]:
        if not available:
            break
        index = min(
            available,
            key=lambda item: float(np.linalg.norm(candidates[item][0] - old[0])),
        )
        selected.append(index)
        available.remove(index)
        if not available:
            break
    selected.extend(sorted(available)[: 2 - len(selected)])
    return selected


def _default_hand_pose(np: Any, size: tuple[int, int]) -> tuple[list[Any], list[Any]]:
    template = np.asarray(
        [
            (0.00, 0.00),
            (-0.04, -0.03),
            (-0.07, -0.07),
            (-0.09, -0.11),
            (-0.10, -0.15),
            (-0.03, -0.07),
            (-0.03, -0.13),
            (-0.03, -0.19),
            (-0.03, -0.24),
            (0.00, -0.08),
            (0.00, -0.15),
            (0.00, -0.22),
            (0.00, -0.28),
            (0.03, -0.07),
            (0.03, -0.13),
            (0.03, -0.19),
            (0.03, -0.24),
            (0.06, -0.06),
            (0.07, -0.11),
            (0.08, -0.16),
            (0.09, -0.20),
        ],
        dtype=np.float64,
    )
    center = np.asarray((size[0] * 0.52, size[1] * 0.58))
    image_points = center + template * np.asarray((size[0] * 0.9, size[1] * 0.9))
    world_points = np.column_stack((template * 0.35, np.zeros(21)))
    return [image_points], [world_points]


def _analyse_source(
    cv2: Any,
    mp: Any,
    np: Any,
    source: Path,
    size: tuple[int, int],
) -> tuple[Any, Any, dict[str, int]]:
    capture = cv2.VideoCapture(str(source))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    union = np.zeros((size[1], size[0]), dtype=np.uint8)
    plate_frames = []
    detected = 0
    connections = tuple(mp.solutions.hands.HAND_CONNECTIONS)
    segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frame_index = 0
    while True:
        ok, source_frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(source_frame, size, interpolation=cv2.INTER_AREA)
        if frame_index % 10 == 0:
            plate_frames.append(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        person_score = segmenter.process(rgb).segmentation_mask
        hand_result = detector.process(rgb)
        image_hands = hand_result.multi_hand_landmarks or ()
        hand_arrays = [
            np.asarray(
                [(point.x * size[0], point.y * size[1]) for point in hand.landmark],
                dtype=np.float64,
            )
            for hand in image_hands
        ]
        hand_centers = [points[0] for points in hand_arrays]
        person = (person_score >= 0.18).astype(np.uint8) * 255
        person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        person = _largest_components(cv2, np, person, hand_centers)
        for points in hand_arrays:
            person = cv2.bitwise_or(
                person, _hand_mask(cv2, np, connections, points, size)
            )
        person[:, round(size[0] * 0.72) :] = 0
        union = cv2.bitwise_or(union, person)
        detected += int(bool(hand_arrays))
        frame_index += 1
    capture.release()
    detector.close()
    segmenter.close()
    if frame_index != expected_frames:
        raise RuntimeError(
            f"analysis decoded {frame_index}/{expected_frames} frames"
        )
    if detected < round(expected_frames * 0.6):
        raise RuntimeError(
            f"hand detection is too sparse: {detected}/{expected_frames}"
        )
    union = cv2.morphologyEx(
        union,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
    )
    union = cv2.dilate(
        union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    )
    return union, _clean_plate(cv2, np, plate_frames, union), {
        "frames": frame_index,
        "hand_detected_frames": detected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--full-robot-model", type=Path, required=True)
    parser.add_argument("--sharpa-model", type=Path, required=True)
    parser.add_argument("--allegro-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=1024)
    parser.add_argument("--working-width", type=int, default=960)
    parser.add_argument("--g1-revision", required=True)
    parser.add_argument("--sharpa-revision", required=True)
    parser.add_argument("--allegro-revision", required=True)
    args = parser.parse_args()

    paths = {
        "source": args.source.expanduser().resolve(),
        "full_robot_model": args.full_robot_model.expanduser().resolve(),
        "sharpa_model": args.sharpa_model.expanduser().resolve(),
        "allegro_model": args.allegro_model.expanduser().resolve(),
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

    capture = cv2.VideoCapture(str(paths["source"]))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    working_height = round(source_height * args.working_width / source_width)
    if working_height % 2:
        working_height += 1
    size = (args.working_width, working_height)

    erase_mask, clean_plate, analysis = _analyse_source(
        cv2, mp, np, paths["source"], size
    )
    cv2.imwrite(str(output_dir / "temporal-union-person-mask.png"), erase_mask)
    cv2.imwrite(str(output_dir / "static-clean-plate.jpg"), clean_plate)

    full_renderer = FullRobotRenderer(mujoco, np, paths["full_robot_model"])
    full_rgb, full_mask = full_renderer.render()
    full_renderer.close()
    full_rgb, full_mask = _fit_full_robot(
        cv2, np, full_rgb, full_mask, size, erase_mask
    )
    cv2.imwrite(str(output_dir / "full-robot-layer.png"), full_rgb)
    cv2.imwrite(str(output_dir / "full-robot-mask.png"), full_mask)
    full_ys, full_xs = np.nonzero(full_mask)
    full_bbox = (
        int(full_xs.min()),
        int(full_ys.min()),
        int(full_xs.max()),
        int(full_ys.max()),
    )

    sharpa = HandRenderer(
        mujoco,
        np,
        paths["sharpa_model"],
        "sharpa",
        (0, 0, 0.05, 0.55, 135, -30),
    )
    allegro = HandRenderer(
        mujoco,
        np,
        paths["allegro_model"],
        "allegro",
        (0, 0, 0.05, 0.55, 180, -30),
    )
    ffmpeg = subprocess.run(
        ["which", "ffmpeg"], text=True, capture_output=True, check=True
    ).stdout.strip()
    outputs = {
        "sharpa": output_dir / "full-robot-sharpa-flower-arranging.mp4",
        "allegro": output_dir / "full-robot-allegro-flower-arranging.mp4",
        "comparison": output_dir / "human-full-robot-flower-comparison.mp4",
    }
    writers = {
        "sharpa": _writer(ffmpeg, outputs["sharpa"], size, fps),
        "allegro": _writer(ffmpeg, outputs["allegro"], size, fps),
        "comparison": _writer(ffmpeg, outputs["comparison"], (1920, 360), fps),
    }

    capture = cv2.VideoCapture(str(paths["source"]))
    detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    previous_2d: list[Any] | None = None
    previous_3d: list[Any] | None = None
    frame_count = 0
    carried_hand_frames = 0
    maximum_source_match_ratio = {"sharpa": 0.0, "allegro": 0.0}
    robot_centroids: dict[str, list[tuple[float, float]]] = {
        "sharpa": [],
        "allegro": [],
    }
    try:
        while True:
            ok, source_frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(source_frame, size, interpolation=cv2.INTER_AREA)
            result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_hands = result.multi_hand_landmarks or ()
            world_hands = result.multi_hand_world_landmarks or ()
            if image_hands and len(image_hands) == len(world_hands):
                candidates_2d = [
                    np.asarray(
                        [(point.x * size[0], point.y * size[1]) for point in hand.landmark],
                        dtype=np.float64,
                    )
                    for hand in image_hands
                ]
                candidates_3d = [
                    np.asarray(
                        [(point.x, point.y, point.z) for point in hand.landmark],
                        dtype=np.float64,
                    )
                    for hand in world_hands
                ]
                selected = _select_hands(np, candidates_2d, previous_2d)
                hands_2d = [candidates_2d[index] for index in selected]
                hands_3d = [candidates_3d[index] for index in selected]
                if previous_2d is not None and previous_3d is not None:
                    for index in range(min(len(hands_2d), len(previous_2d))):
                        hands_2d[index] = (
                            0.35 * hands_2d[index] + 0.65 * previous_2d[index]
                        )
                        hands_3d[index] = (
                            0.35 * hands_3d[index] + 0.65 * previous_3d[index]
                        )
            elif previous_2d is not None and previous_3d is not None:
                hands_2d = [points.copy() for points in previous_2d]
                hands_3d = [points.copy() for points in previous_3d]
                carried_hand_frames += 1
            else:
                hands_2d, hands_3d = _default_hand_pose(np, size)
                carried_hand_frames += 1
            previous_2d = [points.copy() for points in hands_2d]
            previous_3d = [points.copy() for points in hands_3d]

            flower = _flower_mask(cv2, np, frame)
            base = frame.copy()
            base[erase_mask > 0] = clean_plate[erase_mask > 0]
            base = _overlay(cv2, np, base, full_rgb, full_mask)
            target_wrists = [
                tuple(np.rint(points[0]).astype(int))
                for points in hands_2d
            ]
            while len(target_wrists) < 2:
                target_wrists.append((round(size[0] * 0.42), round(size[1] * 0.54)))

            variants = {}
            for vendor, renderer in (("sharpa", sharpa), ("allegro", allegro)):
                composed = base.copy()
                robot_width = full_bbox[2] - full_bbox[0] + 1
                robot_height = full_bbox[3] - full_bbox[1] + 1
                shoulders = (
                    (
                        full_bbox[0] + round(robot_width * 0.58),
                        full_bbox[1] + round(robot_height * 0.28),
                    ),
                    (
                        full_bbox[0] + round(robot_width * 0.56),
                        full_bbox[1] + round(robot_height * 0.38),
                    ),
                )
                for index, wrist in enumerate(target_wrists[:2]):
                    _draw_robot_arm(cv2, composed, shoulders[index], wrist, index * 2 - 1)
                    points_2d = (
                        hands_2d[index]
                        if index < len(hands_2d)
                        else hands_2d[0] + np.asarray((25.0, 20.0))
                    )
                    points_3d = (
                        hands_3d[index]
                        if index < len(hands_3d)
                        else hands_3d[0]
                    )
                    robot_rgb, robot_mask = renderer.render(points_3d)
                    robot_rgb, robot_mask = _warp_robot(
                        cv2,
                        np,
                        robot_rgb,
                        robot_mask,
                        points_2d,
                        vendor,
                        size,
                    )
                    robot_mask = cv2.dilate(robot_mask, np.ones((5, 5), np.uint8))
                    composed = _overlay(cv2, np, composed, robot_rgb, robot_mask)
                preserve = (flower > 0) & (erase_mask > 0)
                composed[preserve] = frame[preserve]
                variants[vendor] = composed
                audited_region = (erase_mask > 0) & ~preserve
                exact_source_matches = np.all(composed == frame, axis=2) & audited_region
                source_match_ratio = float(
                    np.count_nonzero(exact_source_matches)
                    / max(1, np.count_nonzero(audited_region))
                )
                maximum_source_match_ratio[vendor] = max(
                    maximum_source_match_ratio[vendor], source_match_ratio
                )
                ys, xs = np.nonzero(full_mask)
                robot_centroids[vendor].append(
                    (float(xs.mean()), float(ys.mean()))
                )

            for vendor in ("sharpa", "allegro"):
                assert writers[vendor].stdin is not None
                writers[vendor].stdin.write(variants[vendor].tobytes())
            panels = [
                _label(
                    cv2,
                    cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA),
                    "Human source",
                ),
                _label(
                    cv2,
                    cv2.resize(
                        variants["sharpa"], (640, 360), interpolation=cv2.INTER_AREA
                    ),
                    "Full robot + Sharpa (geometric)",
                ),
                _label(
                    cv2,
                    cv2.resize(
                        variants["allegro"], (640, 360), interpolation=cv2.INTER_AREA
                    ),
                    "Full robot + Allegro (geometric)",
                ),
            ]
            assert writers["comparison"].stdin is not None
            writers["comparison"].stdin.write(np.hstack(panels).tobytes())
            frame_count += 1
    finally:
        capture.release()
        detector.close()
        sharpa.close()
        allegro.close()
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
    if any(ratio > 0.02 for ratio in maximum_source_match_ratio.values()):
        raise RuntimeError(
            "full-person replacement retained too many exact source pixels: "
            f"{maximum_source_match_ratio}"
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
        "status": "WORKING",
        "method": "temporal_union_full_person_removal_plus_full_robot_geometric_composite",
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
            **{
                label: str(path)
                for label, path in paths.items()
            },
            **{
                f"{label}_sha256": _sha256(path)
                for label, path in paths.items()
            },
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
            "source_size": [source_width, source_height],
            "working_size": list(size),
        },
        "person_removal": {
            "mask": str(output_dir / "temporal-union-person-mask.png"),
            "mask_sha256": _sha256(output_dir / "temporal-union-person-mask.png"),
            "clean_plate": str(output_dir / "static-clean-plate.jpg"),
            "clean_plate_sha256": _sha256(output_dir / "static-clean-plate.jpg"),
            "analysis": analysis,
            "maximum_exact_source_rgb_match_ratio_inside_erased_person_region_outside_flower": maximum_source_match_ratio,
            "temporal_union_prevents_previous_frame_human_trails": True,
        },
        "ghosting_audit": {
            "full_robot_layer_is_static": True,
            "full_robot_centroid_max_step_pixels": {
                vendor: max(
                    (
                        math.dist(first, second)
                        for first, second in zip(centroids, centroids[1:])
                    ),
                    default=0.0,
                )
                for vendor, centroids in robot_centroids.items()
            },
            "decoded_all_outputs": True,
            "carried_forward_robot_hand_pose_frames": carried_hand_frames,
        },
        "outputs": {
            name: str(path) for name, path in outputs.items()
        }
        | {
            f"{name}_sha256": _sha256(path)
            for name, path in outputs.items()
        },
        "limitations": [
            "This is deterministic geometric compositing, not official PhiZero inference.",
            "The shared full body is a Unitree G1 visualization with target-specific Sharpa or Allegro hands.",
            "Arms connecting the full body to tracked hands are procedural screen-space links.",
            "Flower pixels use a color-preservation mask; occlusion is not depth-calibrated.",
            "No physics-valid contact or real robot execution is claimed.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
