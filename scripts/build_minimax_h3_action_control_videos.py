#!/usr/bin/env python3
"""Compile language action variants into explicit real-scene arm-control videos.

These videos are intermediate controls, not claimed model outputs.  They use a
reviewed robot anchor as texture and an arm-removed real-scene plate as the
background.  Every waypoint is named in the source-camera pixel frame so H3
receives visibly different motion instead of three weakly differentiated text
prompts.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_rigid_part_robot_replacement import (  # noqa: E402
    POSE_INDICES,
    _overlay,
    _piece_masks,
    _similarity,
    _warp_piece,
)


CAMERA_PIXEL_FRAME = "camera:source_anchor_pixels"


@dataclass(frozen=True)
class ArmWaypoint:
    progress: float
    wrist_x: float
    wrist_y: float
    hand_rotation_degrees: float = 0.0
    grasp: bool = False
    frame: str = CAMERA_PIXEL_FRAME

    def validate(self) -> None:
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("waypoint progress must be in [0, 1]")
        if not all(math.isfinite(value) for value in (self.wrist_x, self.wrist_y)):
            raise ValueError("waypoint pixel coordinates must be finite")
        if self.frame != CAMERA_PIXEL_FRAME:
            raise ValueError(f"waypoint frame must be {CAMERA_PIXEL_FRAME}")


@dataclass(frozen=True)
class ActionControlPlan:
    label: str
    left: tuple[ArmWaypoint, ...]
    right: tuple[ArmWaypoint, ...]
    object_transfer_progress: float | None = None

    def validate(self) -> None:
        for side in (self.left, self.right):
            if len(side) < 2:
                raise ValueError("each arm needs at least two waypoints")
            for waypoint in side:
                waypoint.validate()
            progresses = [waypoint.progress for waypoint in side]
            if progresses != sorted(progresses) or len(progresses) != len(set(progresses)):
                raise ValueError("arm waypoint progress must be unique and increasing")
            if progresses[0] != 0.0 or progresses[-1] != 1.0:
                raise ValueError("each arm plan must span progress 0 to 1")


def default_action_control_plans() -> tuple[ActionControlPlan, ...]:
    """Return three deliberately separated trajectories for the flower scene."""

    plans = (
        ActionControlPlan(
            "insert-flower",
            left=(
                ArmWaypoint(0.0, 719, 345),
                ArmWaypoint(0.22, 700, 375),
                ArmWaypoint(0.72, 690, 390, grasp=True),
                ArmWaypoint(1.0, 720, 370),
            ),
            right=(
                ArmWaypoint(0.0, 775, 416),
                ArmWaypoint(0.18, 720, 405),
                ArmWaypoint(0.34, 705, 330, grasp=True),
                ArmWaypoint(0.58, 660, 350, -15, True),
                ArmWaypoint(0.78, 675, 430, -35, True),
                ArmWaypoint(0.88, 675, 430, -35, False),
                ArmWaypoint(1.0, 790, 430),
            ),
        ),
        ActionControlPlan(
            "handover-flower",
            left=(
                ArmWaypoint(0.0, 719, 345),
                ArmWaypoint(0.28, 760, 365),
                ArmWaypoint(0.50, 815, 375, 10),
                ArmWaypoint(0.62, 805, 375, 10, True),
                ArmWaypoint(0.80, 710, 315, -15, True),
                ArmWaypoint(1.0, 700, 305, -15, True),
            ),
            right=(
                ArmWaypoint(0.0, 775, 416),
                ArmWaypoint(0.18, 720, 400, grasp=True),
                ArmWaypoint(0.46, 815, 382, 15, True),
                ArmWaypoint(0.60, 815, 382, 15, False),
                ArmWaypoint(0.78, 925, 455, 20),
                ArmWaypoint(1.0, 950, 470, 20),
            ),
            object_transfer_progress=0.60,
        ),
        ActionControlPlan(
            "inspect-flower",
            left=(
                ArmWaypoint(0.0, 719, 345),
                ArmWaypoint(0.30, 700, 425),
                ArmWaypoint(1.0, 700, 445),
            ),
            right=(
                ArmWaypoint(0.0, 775, 416),
                ArmWaypoint(0.18, 720, 395, grasp=True),
                ArmWaypoint(0.42, 790, 230, 0, True),
                ArmWaypoint(0.62, 805, 225, 45, True),
                ArmWaypoint(0.80, 805, 225, 45, True),
                ArmWaypoint(1.0, 785, 305, 45, True),
            ),
        ),
    )
    for plan in plans:
        plan.validate()
    return plans


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


def _timeline_from_manifest(action: dict[str, Any]) -> str:
    timeline = str(action.get("timeline", "")).strip()
    if timeline:
        return timeline
    phases = action.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("action requires either a timeline string or a phases list")
    return "; ".join(
        f"{float(phase['start_s']):.3f}-{float(phase['end_s']):.3f} s: "
        f"{str(phase['description']).strip()}"
        for phase in phases
    )


def _smooth(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def interpolate_waypoints(waypoints: tuple[ArmWaypoint, ...], progress: float) -> ArmWaypoint:
    if progress <= waypoints[0].progress:
        return waypoints[0]
    if progress >= waypoints[-1].progress:
        return waypoints[-1]
    for first, second in zip(waypoints, waypoints[1:]):
        if first.progress <= progress <= second.progress:
            local = _smooth((progress - first.progress) / (second.progress - first.progress))
            return ArmWaypoint(
                progress=progress,
                wrist_x=first.wrist_x + (second.wrist_x - first.wrist_x) * local,
                wrist_y=first.wrist_y + (second.wrist_y - first.wrist_y) * local,
                hand_rotation_degrees=(
                    first.hand_rotation_degrees
                    + (second.hand_rotation_degrees - first.hand_rotation_degrees) * local
                ),
                grasp=first.grasp if local < 0.5 else second.grasp,
            )
    raise AssertionError("unreachable waypoint interpolation")


def solve_elbow(np: Any, shoulder: Any, wrist: Any, upper: float, lower: float, bend: float) -> Any:
    """Solve one two-link elbow in the named 2D camera-pixel frame."""

    vector = wrist - shoulder
    distance = float(np.linalg.norm(vector))
    maximum = upper + lower - 1e-3
    minimum = abs(upper - lower) + 1e-3
    if distance < 1e-6:
        vector = np.asarray((1.0, 0.0), dtype=np.float64)
        distance = 1.0
    clamped = min(maximum, max(minimum, distance))
    direction = vector / distance
    wrist = shoulder + direction * clamped
    along = (upper * upper - lower * lower + clamped * clamped) / (2.0 * clamped)
    height = math.sqrt(max(0.0, upper * upper - along * along))
    perpendicular = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    return shoulder + direction * along + perpendicular * height * bend


def _reference_pose(np: Any) -> Any:
    points = np.zeros((33, 2), dtype=np.float64)
    # Extracted from the reviewed anchor-rig overlay.  These are camera pixels,
    # not robot-base coordinates.
    points[11], points[13], points[15], points[19] = (
        (840, 269), (743, 347), (719, 345), (690, 350)
    )
    points[12], points[14], points[16], points[20] = (
        (990, 326), (931, 488), (775, 416), (756, 397)
    )
    return points


def _rotation(np: Any, vector: Any, degrees: float) -> Any:
    angle = math.radians(degrees)
    matrix = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
        dtype=np.float64,
    )
    return matrix @ vector


def target_pose(np: Any, reference: Any, plan: ActionControlPlan, progress: float) -> Any:
    current = reference.copy()
    for side, waypoints, bend in (("left", plan.left, -1.0), ("right", plan.right, 1.0)):
        shoulder_index, elbow_index, wrist_index, hand_index = POSE_INDICES[side]
        waypoint = interpolate_waypoints(waypoints, progress)
        shoulder = reference[shoulder_index]
        wrist = np.asarray((waypoint.wrist_x, waypoint.wrist_y), dtype=np.float64)
        upper = float(np.linalg.norm(reference[elbow_index] - shoulder))
        lower = float(np.linalg.norm(reference[wrist_index] - reference[elbow_index]))
        elbow = solve_elbow(np, shoulder, wrist, upper, lower, bend)
        hand_vector = reference[hand_index] - reference[wrist_index]
        hand = wrist + _rotation(np, hand_vector, waypoint.hand_rotation_degrees)
        current[shoulder_index] = shoulder
        current[elbow_index] = elbow
        current[wrist_index] = wrist
        current[hand_index] = hand
    return current


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-anchor", type=Path, required=True)
    parser.add_argument("--anchor-mask", type=Path, required=True)
    parser.add_argument("--clean-base", type=Path, required=True)
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
    experiment.mkdir(parents=True, exist_ok=True)
    paths = {
        "robot_anchor": args.robot_anchor.expanduser().resolve(),
        "anchor_mask": args.anchor_mask.expanduser().resolve(),
        "clean_base": args.clean_base.expanduser().resolve(),
        "action_manifest": args.action_manifest.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required input does not exist or is empty: {path}")
    requested = json.loads(paths["action_manifest"].read_text())
    instructions = {item["label"]: item for item in requested["actions"]}
    plans = default_action_control_plans()
    if set(instructions) != {plan.label for plan in plans}:
        raise ValueError("action manifest labels do not match the compiled control plans")
    import cv2
    import numpy as np

    np.random.seed(args.seed)
    robot = cv2.imread(str(paths["robot_anchor"]), cv2.IMREAD_COLOR)
    anchor_mask = cv2.imread(str(paths["anchor_mask"]), cv2.IMREAD_GRAYSCALE)
    clean_base = cv2.imread(str(paths["clean_base"]), cv2.IMREAD_COLOR)
    if robot is None or anchor_mask is None or clean_base is None:
        raise RuntimeError("cannot decode control-video image assets")
    # The authored waypoints and reviewed pose overlay are both in the original
    # 1280x720 source-camera frame.  Normalize every image input to that named
    # frame instead of inheriting the odd-sized ImageGen clean plate.
    width, height = 1280, 720
    clean_base = cv2.resize(clean_base, (width, height), interpolation=cv2.INTER_LANCZOS4)
    robot = cv2.resize(robot, (width, height), interpolation=cv2.INTER_LANCZOS4)
    anchor_mask = cv2.resize(anchor_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    reference = _reference_pose(np)
    pieces = _piece_masks(cv2, np, anchor_mask, reference)
    records = []
    wrist_traces: dict[str, Any] = {}
    for plan in plans:
        output = experiment / "variants" / plan.label / "action-control.mp4"
        writer = _writer(paths["ffmpeg"], output, width, height, args.fps)
        trace = []
        try:
            for frame_index in range(args.num_frames):
                progress = frame_index / max(1, args.num_frames - 1)
                current = target_pose(np, reference, plan, progress)
                candidate = clean_base.copy()
                for side, (shoulder, elbow, wrist, hand) in POSE_INDICES.items():
                    for name, first, second in (
                        (f"{side}_upper", shoulder, elbow),
                        (f"{side}_lower", elbow, wrist),
                        (f"{side}_hand", wrist, hand),
                    ):
                        transform = _similarity(
                            np, reference[first], reference[second], current[first], current[second]
                        )
                        warped_image, warped_mask = _warp_piece(
                            cv2, robot, pieces[name], transform
                        )
                        candidate = _overlay(cv2, np, candidate, warped_image, warped_mask)
                cv2.putText(
                    candidate,
                    f"CONTROL ONLY / {plan.label.upper()}",
                    (18, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (35, 245, 185),
                    1,
                    cv2.LINE_AA,
                )
                assert writer.stdin is not None
                writer.stdin.write(candidate.tobytes())
                trace.append(
                    {
                        "frame": frame_index,
                        "progress": progress,
                        "left_wrist_xy": current[15].tolist(),
                        "right_wrist_xy": current[16].tolist(),
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
                "-vf", "fps=1,scale=416:234,tile=5x1:padding=3:margin=3:color=black",
                "-frames:v", "1", str(output.parent / "contact-sheet.jpg"),
            ],
            check=True,
        )
        wrist_traces[plan.label] = np.asarray(
            [item["left_wrist_xy"] + item["right_wrist_xy"] for item in trace],
            dtype=np.float64,
        )
        trace_path = output.parent / "trajectory.json"
        _write_json(trace_path, {"label": plan.label, "trace": trace})
        records.append(
            {
                "label": plan.label,
                "instruction": instructions[plan.label]["instruction"],
                "timeline": _timeline_from_manifest(instructions[plan.label]),
                "plan": asdict(plan),
                "output": str(output),
                "output_sha256": _sha256(output),
                "trajectory": str(trace_path),
                "trajectory_sha256": _sha256(trace_path),
            }
        )
    separations = []
    labels = [plan.label for plan in plans]
    for left_index in range(len(labels)):
        for right_index in range(left_index + 1, len(labels)):
            difference = wrist_traces[labels[left_index]] - wrist_traces[labels[right_index]]
            rms = float(np.sqrt(np.mean(np.square(difference))))
            separations.append(
                {"left": labels[left_index], "right": labels[right_index], "wrist_rms_pixels": rms}
            )
    separation_floor = min(item["wrist_rms_pixels"] for item in separations)
    manifest = {
        "schema_version": "1.0.0",
        "method": "language_action_to_explicit_camera_pixel_arm_trajectory_to_rigid_control_video",
        "status": "completed",
        "honest_status": "WORKING" if separation_floor >= 40.0 else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "opencv-python")
            if importlib.util.find_spec(name.split("-")[0]) is not None
        },
        "gpu": {"used": False, "reason": "deterministic CPU control-video compilation"},
        "seed": args.seed,
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in paths.items()
        },
        "coordinate_frames": {
            "waypoints": CAMERA_PIXEL_FRAME,
            "robot_pieces": "camera:robot_anchor_pixels",
            "output": "camera:source_anchor_pixels",
        },
        "variants": records,
        "trajectory_separation": separations,
        "acceptance": {
            "all_outputs_decoded": True,
            "all_trajectories_explicit": True,
            "minimum_pairwise_wrist_rms_pixels": separation_floor,
            "trajectory_separation_passed": separation_floor >= 40.0,
        },
        "limitations": [
            "These are deterministic intermediate motion controls, not final H3 outputs.",
            "The rigid 2D arm pieces provide visible pose timing but not contact physics or 3D kinematics.",
            "The small CONTROL ONLY label prevents accidental presentation as a generated result.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"experiment": str(experiment), "acceptance": manifest["acceptance"]}, indent=2))
    return 0 if manifest["honest_status"] == "WORKING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
