#!/usr/bin/env python3
"""Compile three counterfactual bowl actions into explicit real-scene controls.

The controls are intermediate conditioning videos, not model outputs.  They
start from the same frame of the real Hand2Dex-2 laboratory video and the same
reviewed robot-transfer frame.  Only the commanded hand path and rigid yellow
bowl terminal state differ.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAMERA_PIXEL_FRAME = "camera:hand2dex_2_reference_pixels"


@dataclass(frozen=True)
class BowlActionPlan:
    label: str
    target_center_x: float
    target_center_y: float
    target_scale: float = 1.0
    coordinate_frame: str = CAMERA_PIXEL_FRAME

    def validate(self, width: int = 896, height: int = 512) -> None:
        if self.coordinate_frame != CAMERA_PIXEL_FRAME:
            raise ValueError(f"plan frame must be {CAMERA_PIXEL_FRAME}")
        if not 0 <= self.target_center_x < width or not 0 <= self.target_center_y < height:
            raise ValueError("target center must be inside the authored camera frame")
        if not math.isfinite(self.target_scale) or not 0.6 <= self.target_scale <= 1.4:
            raise ValueError("target scale must be finite and in [0.6, 1.4]")


def default_bowl_action_plans() -> tuple[BowlActionPlan, ...]:
    plans = (
        BowlActionPlan("slide-left", 175.0, 330.0, 1.0),
        BowlActionPlan("slide-right", 610.0, 330.0, 1.0),
        BowlActionPlan("lift-up", 365.0, 135.0, 1.08),
    )
    for plan in plans:
        plan.validate()
    return plans


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def action_progress(frame_index: int, num_frames: int) -> float:
    """Hold the common start, move, then hold the mutually exclusive endpoint."""

    normalized = frame_index / max(1, num_frames - 1)
    return smoothstep((normalized - 0.18) / 0.56)


def similarity_affine(
    np: Any,
    anchor_xy: tuple[float, float],
    source_contact_xy: tuple[float, float],
    target_contact_xy: tuple[float, float],
) -> Any:
    """Map one contact point while keeping the arm-base camera pixel fixed."""

    anchor = np.asarray(anchor_xy, dtype=np.float64)
    source = np.asarray(source_contact_xy, dtype=np.float64) - anchor
    target = np.asarray(target_contact_xy, dtype=np.float64) - anchor
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm < 1e-6 or target_norm < 1e-6:
        raise ValueError("arm contact vectors must be non-zero")
    scale = target_norm / source_norm
    source_angle = math.atan2(float(source[1]), float(source[0]))
    target_angle = math.atan2(float(target[1]), float(target[0]))
    angle = target_angle - source_angle
    linear = scale * np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
        dtype=np.float64,
    )
    translation = anchor - linear @ anchor
    return np.hstack((linear, translation[:, None]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _decode(cv2: Any, path: Path) -> list[Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < 3:
        raise RuntimeError(f"decoded too few frames from {path}")
    return frames


def _writer(ffmpeg: Path, output: Path, width: int, height: int, fps: int) -> Any:
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
            "-c:v", "libx264", "-crf", "12", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _largest_component(cv2: Any, np: Any, mask: Any) -> Any:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        raise RuntimeError("foreground segmentation found no component")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == index, 255, 0).astype(np.uint8)


def _overlay(cv2: Any, np: Any, base: Any, layer: Any, mask: Any) -> Any:
    alpha = cv2.GaussianBlur(mask, (0, 0), 0.9).astype(np.float32) / 255.0
    return np.rint(
        base.astype(np.float32) * (1.0 - alpha[..., None])
        + layer.astype(np.float32) * alpha[..., None]
    ).astype(np.uint8)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--robot-video", type=Path, required=True)
    parser.add_argument("--action-manifest", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


def main() -> int:
    args = _parser().parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    manifest_path = experiment / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"control experiment already exists: {manifest_path}")
    paths = {
        "source_video": args.source_video.expanduser().resolve(),
        "robot_video": args.robot_video.expanduser().resolve(),
        "action_manifest": args.action_manifest.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required input does not exist or is empty: {path}")
    requested = json.loads(paths["action_manifest"].read_text())
    actions = {item["label"]: item for item in requested["actions"]}
    plans = default_bowl_action_plans()
    if set(actions) != {plan.label for plan in plans}:
        raise ValueError("action manifest labels do not match bowl control plans")
    experiment.mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np

    np.random.seed(args.seed)
    source_frames = _decode(cv2, paths["source_video"])
    robot_frames = _decode(cv2, paths["robot_video"])
    source = source_frames[0]
    robot = robot_frames[0]
    if source.shape != robot.shape:
        raise RuntimeError("real source and robot reference frames are not aligned")
    height, width = robot.shape[:2]
    for plan in plans:
        plan.validate(width, height)

    hsv = cv2.cvtColor(robot, cv2.COLOR_BGR2HSV)
    bowl_mask = cv2.inRange(hsv, (10, 75, 70), (42, 255, 255))
    bowl_mask[:150] = 0
    bowl_mask = cv2.morphologyEx(
        bowl_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    bowl_mask = _largest_component(cv2, np, bowl_mask)
    difference = np.max(cv2.absdiff(robot, source), axis=2)
    arm_mask = np.where(difference >= 18, 255, 0).astype(np.uint8)
    arm_mask[:175] = 0
    arm_mask[:, :430] = 0
    arm_mask = cv2.morphologyEx(
        arm_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    )
    arm_mask = cv2.dilate(
        arm_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    arm_mask[bowl_mask > 0] = 0
    combined = cv2.dilate(
        cv2.max(arm_mask, bowl_mask),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    clean = cv2.inpaint(robot, combined, 9, cv2.INPAINT_TELEA)
    moments = cv2.moments(bowl_mask)
    if moments["m00"] <= 0:
        raise RuntimeError("yellow bowl mask is empty")
    start_center = np.asarray(
        (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
        dtype=np.float64,
    )
    source_contact = np.asarray((520.0, 285.0), dtype=np.float64)
    arm_anchor = (875.0, 505.0)

    input_dir = experiment / "input"
    input_dir.mkdir(parents=True)
    robot_reference = input_dir / "robot-reference.png"
    scene_anchor = input_dir / "real-scene-frame-000.png"
    clean_path = input_dir / "control-clean-plate.png"
    bowl_mask_path = input_dir / "bowl-mask.png"
    arm_mask_path = input_dir / "robot-arm-mask.png"
    cv2.imwrite(str(robot_reference), robot)
    cv2.imwrite(str(scene_anchor), source)
    cv2.imwrite(str(clean_path), clean)
    cv2.imwrite(str(bowl_mask_path), bowl_mask)
    cv2.imwrite(str(arm_mask_path), arm_mask)

    aligned_source = input_dir / "real-scene-source-124f.mp4"
    source_writer = _writer(paths["ffmpeg"], aligned_source, width, height, args.fps)
    try:
        for index in range(args.num_frames):
            source_index = round(index * (len(source_frames) - 1) / (args.num_frames - 1))
            assert source_writer.stdin is not None
            source_writer.stdin.write(source_frames[source_index].tobytes())
    finally:
        if source_writer.stdin is not None:
            source_writer.stdin.close()
        if source_writer.wait():
            raise RuntimeError("ffmpeg failed for aligned real-scene source")

    records = []
    endpoints = []
    for plan in plans:
        output = experiment / "variants" / plan.label / "action-control.mp4"
        writer = _writer(paths["ffmpeg"], output, width, height, args.fps)
        trace = []
        target_center = np.asarray(
            (plan.target_center_x, plan.target_center_y), dtype=np.float64
        )
        try:
            for frame_index in range(args.num_frames):
                progress = action_progress(frame_index, args.num_frames)
                center = start_center + (target_center - start_center) * progress
                scale = 1.0 + (plan.target_scale - 1.0) * progress
                bowl_transform = np.asarray(
                    (
                        (scale, 0.0, center[0] - scale * start_center[0]),
                        (0.0, scale, center[1] - scale * start_center[1]),
                    ),
                    dtype=np.float64,
                )
                bowl_layer = cv2.warpAffine(
                    robot, bowl_transform, (width, height), flags=cv2.INTER_LANCZOS4
                )
                moved_bowl_mask = cv2.warpAffine(
                    bowl_mask, bowl_transform, (width, height), flags=cv2.INTER_NEAREST
                )
                target_contact = center + (source_contact - start_center) * scale
                arm_transform = similarity_affine(
                    np, arm_anchor, tuple(source_contact), tuple(target_contact)
                )
                arm_layer = cv2.warpAffine(
                    robot, arm_transform, (width, height), flags=cv2.INTER_LANCZOS4
                )
                moved_arm_mask = cv2.warpAffine(
                    arm_mask, arm_transform, (width, height), flags=cv2.INTER_NEAREST
                )
                candidate = _overlay(cv2, np, clean.copy(), bowl_layer, moved_bowl_mask)
                candidate = _overlay(cv2, np, candidate, arm_layer, moved_arm_mask)
                cv2.putText(
                    candidate,
                    f"CONTROL ONLY / {plan.label.upper()}",
                    (18, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (35, 245, 185),
                    1,
                    cv2.LINE_AA,
                )
                assert writer.stdin is not None
                writer.stdin.write(candidate.tobytes())
                trace.append(
                    {
                        "frame": frame_index,
                        "progress": frame_index / max(1, args.num_frames - 1),
                        "action_progress": progress,
                        "bowl_center_xy": center.tolist(),
                        "hand_contact_xy": target_contact.tolist(),
                        "coordinate_frame": CAMERA_PIXEL_FRAME,
                    }
                )
        finally:
            if writer.stdin is not None:
                writer.stdin.close()
            if writer.wait():
                raise RuntimeError(f"ffmpeg failed for {plan.label}")
        subprocess.run(
            [str(paths["ffmpeg"]), "-v", "error", "-i", str(output), "-f", "null", "-"],
            check=True,
        )
        subprocess.run(
            [
                str(paths["ffmpeg"]), "-y", "-v", "error", "-i", str(output),
                "-vf", "fps=1,scale=448:256,tile=5x1:padding=3:margin=3:color=black",
                "-frames:v", "1", str(output.parent / "contact-sheet.jpg"),
            ],
            check=True,
        )
        trace_path = output.parent / "trajectory.json"
        _write_json(trace_path, {"label": plan.label, "trace": trace})
        endpoints.append(target_center)
        records.append(
            {
                "label": plan.label,
                "instruction": actions[plan.label]["instruction"],
                "timeline": actions[plan.label]["timeline"],
                "plan": asdict(plan),
                "output": str(output),
                "output_sha256": _sha256(output),
                "trajectory": str(trace_path),
                "trajectory_sha256": _sha256(trace_path),
            }
        )
    endpoint_distances = []
    for left_index in range(len(plans)):
        for right_index in range(left_index + 1, len(plans)):
            distance = float(np.linalg.norm(endpoints[left_index] - endpoints[right_index]))
            endpoint_distances.append(
                {
                    "left": plans[left_index].label,
                    "right": plans[right_index].label,
                    "endpoint_distance_pixels": distance,
                }
            )
    endpoint_floor = min(item["endpoint_distance_pixels"] for item in endpoint_distances)
    packages = {}
    for name in ("numpy", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": "1.0.0",
        "method": "language_to_explicit_robot_contact_and_bowl_terminal_state_control_video",
        "status": "completed",
        "honest_status": "WORKING" if endpoint_floor >= 180.0 else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "gpu": {"used": False, "reason": "deterministic CPU control compilation"},
        "seed": args.seed,
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in paths.items()
        },
        "derived_inputs": {
            "aligned_real_source": {"path": str(aligned_source), "sha256": _sha256(aligned_source)},
            "robot_reference": {"path": str(robot_reference), "sha256": _sha256(robot_reference)},
            "scene_anchor": {"path": str(scene_anchor), "sha256": _sha256(scene_anchor)},
        },
        "coordinate_frames": {
            "plans": CAMERA_PIXEL_FRAME,
            "object_centers": CAMERA_PIXEL_FRAME,
            "hand_contact": CAMERA_PIXEL_FRAME,
            "output": CAMERA_PIXEL_FRAME,
        },
        "start_bowl_center_xy": start_center.tolist(),
        "variants": records,
        "endpoint_separation": endpoint_distances,
        "acceptance": {
            "all_outputs_decoded": True,
            "all_object_trajectories_explicit": True,
            "minimum_pairwise_endpoint_distance_pixels": endpoint_floor,
            "endpoint_separation_passed": endpoint_floor >= 180.0,
        },
        "limitations": [
            "These are intermediate controls, not MiniMax-H3 outputs or real-robot execution.",
            "The arm is a 2-D similarity warp and the bowl is a rigid image-plane layer.",
            "Contact physics and 3-D height are acceptance targets for H3, not established by this compiler.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"experiment": str(experiment), "acceptance": manifest["acceptance"]}, indent=2))
    return 0 if manifest["honest_status"] == "WORKING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
