#!/usr/bin/env python3
"""Compile existing bowl contact paths into OSCAR skeleton conditions.

The input trajectories remain in their declared camera-pixel frame.  This
compiler resamples and draws a 2-D kinematic chain; it does not infer metric
depth, robot-base poses, joint angles, force, or contact physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.schema import ACWMActionCondition, ActionRepresentation  # noqa: E402

SOURCE_CAMERA_FRAME = "camera:hand2dex_2_reference_pixels"
OSCAR_CAMERA_FRAME = "camera:oscar_640x480_pixels"
CHANNELS = (
    "base_x_px",
    "base_y_px",
    "shoulder_x_px",
    "shoulder_y_px",
    "elbow_x_px",
    "elbow_y_px",
    "wrist_x_px",
    "wrist_y_px",
    "finger_left_x_px",
    "finger_left_y_px",
    "finger_right_x_px",
    "finger_right_y_px",
    "object_center_x_px",
    "object_center_y_px",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def resample_trace(trace: list[dict[str, Any]], frame_count: int) -> list[dict[str, Any]]:
    if len(trace) < 2 or frame_count < 2:
        raise ValueError("trajectory and output require at least two frames")
    return [
        trace[round(index * (len(trace) - 1) / (frame_count - 1))]
        for index in range(frame_count)
    ]


def skeleton_row(
    wrist_source_xy: tuple[float, float],
    *,
    source_size: tuple[int, int] = (896, 512),
    output_size: tuple[int, int] = (640, 480),
) -> tuple[float, ...]:
    """Map a screen-space contact point to a stable four-link 2-D arm chain."""

    source_width, source_height = source_size
    width, height = output_size
    sx, sy = width / source_width, height / source_height
    wrist = (wrist_source_xy[0] * sx, wrist_source_xy[1] * sy)
    base = (min(width - 5.0, 875.0 * sx), min(height - 5.0, 505.0 * sy))
    shoulder = (min(width - 15.0, 825.0 * sx), min(height - 35.0, 445.0 * sy))
    vector = (wrist[0] - shoulder[0], wrist[1] - shoulder[1])
    norm = max(1e-6, math.hypot(*vector))
    normal = (-vector[1] / norm, vector[0] / norm)
    bend = 0.11 * norm
    elbow = (
        shoulder[0] + 0.50 * vector[0] + bend * normal[0],
        shoulder[1] + 0.50 * vector[1] + bend * normal[1],
    )
    tangent = (vector[0] / norm, vector[1] / norm)
    finger_normal = (-tangent[1], tangent[0])
    finger_center = (wrist[0] + 17.0 * tangent[0], wrist[1] + 17.0 * tangent[1])
    finger_left = (
        finger_center[0] + 7.0 * finger_normal[0],
        finger_center[1] + 7.0 * finger_normal[1],
    )
    finger_right = (
        finger_center[0] - 7.0 * finger_normal[0],
        finger_center[1] - 7.0 * finger_normal[1],
    )
    return (*base, *shoulder, *elbow, *wrist, *finger_left, *finger_right)


def repaired_wrist_source_xy(
    samples: list[dict[str, Any]],
    index: int,
    terminal_output_xy: tuple[float, float] | None,
    *,
    source_size: tuple[int, int] = (896, 512),
    output_size: tuple[int, int] = (640, 480),
) -> tuple[float, float]:
    """Apply a smooth, frame-explicit terminal wrist repair in camera pixels."""

    original = tuple(float(value) for value in samples[index]["hand_contact_xy"])
    if terminal_output_xy is None:
        return original
    motion_start = round(0.25 * (len(samples) - 1))
    motion_end = round(0.72 * (len(samples) - 1))
    if index < motion_start:
        return original
    source_width, source_height = source_size
    output_width, output_height = output_size
    terminal_source = (
        terminal_output_xy[0] * source_width / output_width,
        terminal_output_xy[1] * source_height / output_height,
    )
    start = tuple(
        float(value) for value in samples[motion_start]["hand_contact_xy"]
    )
    linear = min(1.0, (index - motion_start) / max(1, motion_end - motion_start))
    alpha = linear * linear * (3.0 - 2.0 * linear)
    return (
        start[0] + alpha * (terminal_source[0] - start[0]),
        start[1] + alpha * (terminal_source[1] - start[1]),
    )


def _terminal_overrides(values: list[str]) -> dict[str, tuple[float, float]]:
    overrides: dict[str, tuple[float, float]] = {}
    for value in values:
        try:
            label, coordinates = value.split("=", 1)
            x_value, y_value = coordinates.split(",", 1)
            point = float(x_value), float(y_value)
        except ValueError as exc:
            raise ValueError(
                "terminal wrist overrides must use LABEL=X,Y in OSCAR camera pixels"
            ) from exc
        if label in overrides or not all(math.isfinite(item) for item in point):
            raise ValueError(f"invalid duplicate terminal wrist override: {value}")
        overrides[label] = point
    return overrides


def _label_mapping(values: list[str], *, option: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        try:
            label, template = value.split("=", 1)
        except ValueError as exc:
            raise ValueError(f"{option} values must use LABEL=TEMPLATE") from exc
        if not label or not template or label in mapping:
            raise ValueError(f"invalid duplicate {option} mapping: {value}")
        mapping[label] = template
    return mapping


def vertical_template_xy(
    primary_xy: tuple[float, float] | list[float],
    template_xy: tuple[float, float] | list[float],
) -> tuple[float, float]:
    """Keep the requested horizontal action and borrow a proven vertical prior."""

    values = (float(primary_xy[0]), float(template_xy[1]))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("vertical template coordinates must be finite")
    return values


def _writer(ffmpeg: Path, output: Path, width: int, height: int, fps: float) -> Any:
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            str(ffmpeg),
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
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _draw(cv2: Any, canvas: Any, row: tuple[float, ...]) -> None:
    skeleton_values = row[:12]
    points = [
        (round(skeleton_values[index]), round(skeleton_values[index + 1]))
        for index in range(0, len(skeleton_values), 2)
    ]
    base, shoulder, elbow, wrist, finger_left, finger_right = points
    for start, end in ((base, shoulder), (shoulder, elbow), (elbow, wrist)):
        cv2.line(canvas, start, end, (0, 255, 255), 4, cv2.LINE_AA)
    for point in (base, shoulder, elbow, wrist):
        cv2.circle(canvas, point, 5, (255, 0, 0), -1, cv2.LINE_AA)
    for finger in (finger_left, finger_right):
        cv2.arrowedLine(canvas, wrist, finger, (0, 0, 255), 5, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(canvas, wrist, (wrist[0] + 22, wrist[1]), (255, 0, 0), 3, cv2.LINE_AA)
    cv2.arrowedLine(canvas, wrist, (wrist[0], wrist[1] - 22), (0, 255, 0), 3, cv2.LINE_AA)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--action-manifest", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frame-count", type=int, default=81)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--only-action", action="append", default=[])
    parser.add_argument(
        "--terminal-wrist-override",
        action="append",
        default=[],
        metavar="LABEL=X,Y",
    )
    parser.add_argument(
        "--vertical-motion-template",
        action="append",
        default=[],
        metavar="LABEL=TEMPLATE",
        help=(
            "Use TEMPLATE's reviewed wrist/object y trajectory while retaining "
            "LABEL's x trajectory. This creates a lift-then-carry action prior."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    control = args.control_run.expanduser().resolve()
    action_manifest = args.action_manifest.expanduser().resolve()
    experiment = args.experiment_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if experiment.exists():
        raise FileExistsError(f"OSCAR condition experiment already exists: {experiment}")
    if not control.is_dir() or not action_manifest.is_file() or not ffmpeg.is_file():
        raise ValueError("control run, action manifest, and ffmpeg must exist")
    if args.frame_count < 2 or args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("output dimensions, frame count, and FPS must be positive")

    import cv2
    import numpy as np

    robot_reference = control / "input" / "robot-reference.png"
    source_video = control / "input" / "real-scene-source-124f.mp4"
    for path in (robot_reference, source_video):
        if not path.is_file():
            raise ValueError(f"missing control-run input: {path}")
    requested = json.loads(action_manifest.read_text())
    requested_actions = {item["label"]: item for item in requested["actions"]}
    only_actions = set(args.only_action or requested_actions)
    unknown_actions = only_actions - requested_actions.keys()
    terminal_overrides = _terminal_overrides(args.terminal_wrist_override)
    vertical_templates = _label_mapping(
        args.vertical_motion_template,
        option="vertical motion template",
    )
    unknown_overrides = terminal_overrides.keys() - requested_actions.keys()
    unknown_template_targets = vertical_templates.keys() - requested_actions.keys()
    missing_template_sources = {
        source
        for source in vertical_templates.values()
        if not (control / "variants" / source / "trajectory.json").is_file()
    }
    if unknown_actions or unknown_overrides or unknown_template_targets or missing_template_sources:
        raise ValueError(
            "unknown requested actions or templates: "
            f"{sorted(unknown_actions | unknown_overrides | unknown_template_targets | missing_template_sources)}"
        )
    experiment.mkdir(parents=True)
    input_dir = experiment / "input"
    input_dir.mkdir()
    first_frame = input_dir / "first-frame.png"
    source_copy = input_dir / "real-scene-source.mp4"
    image = cv2.imread(str(robot_reference), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode robot reference: {robot_reference}")
    image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(first_frame), image)
    shutil.copy2(source_video, source_copy)

    variants: list[dict[str, Any]] = []
    for label, requested_action in requested_actions.items():
        if label not in only_actions:
            continue
        trajectory_path = control / "variants" / label / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text())
        trace = trajectory["trace"]
        if any(item["coordinate_frame"] != SOURCE_CAMERA_FRAME for item in trace):
            raise ValueError(f"{label} trajectory has an unexpected coordinate frame")
        samples = resample_trace(trace, args.frame_count)
        template_label = vertical_templates.get(label)
        template_samples = None
        if template_label is not None:
            template_trajectory = json.loads(
                (control / "variants" / template_label / "trajectory.json").read_text()
            )
            template_trace = template_trajectory["trace"]
            if any(item["coordinate_frame"] != SOURCE_CAMERA_FRAME for item in template_trace):
                raise ValueError(
                    f"{template_label} template has an unexpected coordinate frame"
                )
            template_samples = resample_trace(template_trace, args.frame_count)
        terminal_override = terminal_overrides.get(label)
        rows_list = []
        for index, item in enumerate(samples):
            wrist_source = repaired_wrist_source_xy(
                samples,
                index,
                terminal_override,
                source_size=(896, 512),
                output_size=(args.width, args.height),
            )
            object_source = tuple(float(value) for value in item["bowl_center_xy"])
            if template_samples is not None:
                wrist_source = vertical_template_xy(
                    wrist_source,
                    template_samples[index]["hand_contact_xy"],
                )
                object_source = vertical_template_xy(
                    object_source,
                    template_samples[index]["bowl_center_xy"],
                )
            rows_list.append(
                (
                    *skeleton_row(
                        wrist_source,
                        source_size=(896, 512),
                        output_size=(args.width, args.height),
                    ),
                    object_source[0] * args.width / 896,
                    object_source[1] * args.height / 512,
                )
            )
        rows = tuple(rows_list)
        variant_dir = experiment / "variants" / label
        skeleton_video = variant_dir / "skeleton.mp4"
        overlay_video = variant_dir / "skeleton-overlay.mp4"
        skeleton_writer = _writer(ffmpeg, skeleton_video, args.width, args.height, args.fps)
        overlay_writer = _writer(ffmpeg, overlay_video, args.width, args.height, args.fps)
        try:
            for row in rows:
                skeleton = np.zeros_like(image)
                overlay = image.copy()
                _draw(cv2, skeleton, row)
                _draw(cv2, overlay, row)
                assert skeleton_writer.stdin is not None
                assert overlay_writer.stdin is not None
                skeleton_writer.stdin.write(skeleton.tobytes())
                overlay_writer.stdin.write(overlay.tobytes())
        finally:
            if skeleton_writer.stdin is not None:
                skeleton_writer.stdin.close()
            if overlay_writer.stdin is not None:
                overlay_writer.stdin.close()
            skeleton_code = skeleton_writer.wait()
            overlay_code = overlay_writer.wait()
        if skeleton_code or overlay_code:
            raise RuntimeError(f"ffmpeg failed for {label}: {skeleton_code}, {overlay_code}")
        action = ACWMActionCondition(
            label=label,
            instruction=str(requested_action["instruction"]),
            timeline=str(requested_action["timeline"]),
            representation=ActionRepresentation.KINEMATIC_SKELETON_2D,
            coordinate_frame=OSCAR_CAMERA_FRAME,
            timestamps_s=tuple(index / args.fps for index in range(args.frame_count)),
            channels=CHANNELS,
            values=rows,
            visual_condition=skeleton_video,
        )
        condition_path = variant_dir / "action-condition.json"
        action.to_json(condition_path)
        variants.append(
            {
                "label": label,
                "instruction": action.instruction,
                "timeline": action.timeline,
                "prompt": requested_action.get("prompt"),
                "condition": str(condition_path.relative_to(experiment)),
                "condition_sha256": _sha256(condition_path),
                "skeleton_video": str(skeleton_video.relative_to(experiment)),
                "skeleton_sha256": _sha256(skeleton_video),
                "overlay_video": str(overlay_video.relative_to(experiment)),
                "overlay_sha256": _sha256(overlay_video),
                "terminal_wrist_xy": list(rows[-1][6:8]),
                "action_condition_repair": (
                    {
                        "type": "smooth_terminal_wrist_override",
                        "coordinate_frame": OSCAR_CAMERA_FRAME,
                        "terminal_xy": list(terminal_override),
                        "motion_window_frame_fraction": [0.25, 0.72],
                    }
                    if terminal_override is not None
                    else None
                ),
                "vertical_motion_template": template_label,
            }
        )

    manifest = {
        "schema_version": "1.0.0",
        "status": "condition_compiled",
        "honest_status": "WORKING",
        "method": "camera_pixel_trajectory_to_oscar_2d_skeleton",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "command": sys.argv,
        "gpu": {"used": False, "reason": "deterministic CPU skeleton compilation"},
        "source_coordinate_frame": SOURCE_CAMERA_FRAME,
        "output_coordinate_frame": OSCAR_CAMERA_FRAME,
        "first_frame": str(first_frame.relative_to(experiment)),
        "first_frame_sha256": _sha256(first_frame),
        "source_video": str(source_copy.relative_to(experiment)),
        "source_video_sha256": _sha256(source_copy),
        "frame_count": args.frame_count,
        "fps": args.fps,
        "resolution": [args.width, args.height],
        "selected_actions": sorted(only_actions),
        "terminal_wrist_overrides": {
            label: list(point) for label, point in sorted(terminal_overrides.items())
        },
        "vertical_motion_templates": dict(sorted(vertical_templates.items())),
        "variants": variants,
        "limitations": [
            "The skeleton is a camera-pixel action condition, not a metric robot-base trajectory.",
            "No depth, joint angles, force, contact physics, or real-robot execution is inferred.",
            "Generated videos require independent automatic gates and mandatory human review.",
        ],
    }
    _write_json(experiment / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
