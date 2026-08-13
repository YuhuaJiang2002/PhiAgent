#!/usr/bin/env python3
"""Generate a topology-positive H3 curriculum over real-scene texture plates.

The source media is never treated as topology supervision.  A reviewed region
without a visible person is sampled into a background plate; a deterministic
articulated robot is then rendered with analytic shoulder/head/limb truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.training.h3_identity_rsi import (  # noqa: E402
    DomainCurriculumContract,
    IdentityDatasetPlan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTION_TAGS = (
    "left-raised-near-head",
    "right-raised-near-head",
    "cross-body",
    "bilateral-reach",
)
COMMON_TAGS = (
    "real-background",
    "full-body",
    "unique-shoulder-origins",
    "head-neck-clearance",
)


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


def _git_state() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0.0":
        raise ValueError("curriculum config must use schema_version 1.0.0")
    if payload.get("actions") != list(ACTION_TAGS):
        raise ValueError(f"actions must be exactly {list(ACTION_TAGS)}")
    if int(payload.get("fps", 0)) != 24:
        raise ValueError("curriculum must use 24 FPS")
    if (int(payload.get("num_frames", 0)) - 5) % 17:
        raise ValueError("num_frames must satisfy 17n+5")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or len(scenes) < 6:
        raise ValueError("curriculum requires at least six scene entries")
    scene_ids: set[str] = set()
    source_ids: set[str] = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            raise ValueError("scene entries must be JSON objects")
        for key in ("scene_id", "source_id", "subject_id", "split", "emblem"):
            if not str(scene.get(key, "")).strip():
                raise ValueError(f"scene entry is missing {key}")
        if scene["scene_id"] in scene_ids or scene["source_id"] in source_ids:
            raise ValueError("scene_id and source_id must each be unique")
        scene_ids.add(scene["scene_id"])
        source_ids.add(scene["source_id"])
        roi = scene.get("human_free_roi_normalized")
        if (
            not isinstance(roi, list)
            or len(roi) != 4
            or any(not isinstance(value, (int, float)) for value in roi)
        ):
            raise ValueError(f"{scene['scene_id']} has an invalid ROI")
        x0, y0, x1, y1 = (float(value) for value in roi)
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"{scene['scene_id']} ROI lies outside the frame")
        palette = scene.get("palette_bgr")
        if (
            not isinstance(palette, list)
            or len(palette) != 3
            or any(not isinstance(color, list) or len(color) != 3 for color in palette)
        ):
            raise ValueError(f"{scene['scene_id']} palette must contain three BGR colors")
    return payload


def _source_records(manifest_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        raise ValueError("source preparation manifest is not completed")
    records: dict[str, dict[str, Any]] = {}
    for raw in manifest.get("sources", []):
        source_id = str(raw["source_id"])
        source_path = Path(str(raw["path"])).expanduser().resolve()
        if not source_path.is_file() or _sha256(source_path) != raw["sha256"]:
            raise ValueError(f"source is missing or changed: {source_id}")
        records[source_id] = raw
    return records, manifest


def _capture_frame(cv2: Any, source: Path, seconds: float) -> Any:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open source: {source}")
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not decode {source} at {seconds:.3f}s")
    return frame


def _background_plate(
    cv2: Any,
    np: Any,
    frame: Any,
    roi: list[float],
    width: int,
    height: int,
) -> tuple[Any, list[int]]:
    source_height, source_width = frame.shape[:2]
    x0 = max(0, min(source_width - 2, round(float(roi[0]) * source_width)))
    y0 = max(0, min(source_height - 2, round(float(roi[1]) * source_height)))
    x1 = max(x0 + 2, min(source_width, round(float(roi[2]) * source_width)))
    y1 = max(y0 + 2, min(source_height, round(float(roi[3]) * source_height)))
    crop = frame[y0:y1, x0:x1]
    scale = max(width / crop.shape[1], height / crop.shape[0])
    resized = cv2.resize(
        crop,
        (math.ceil(crop.shape[1] * scale), math.ceil(crop.shape[0] * scale)),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    offset_x = (resized.shape[1] - width) // 2
    offset_y = (resized.shape[0] - height) // 2
    plate = resized[offset_y : offset_y + height, offset_x : offset_x + width].copy()
    plate = cv2.GaussianBlur(plate, (7, 7), 0)
    yy, xx = np.ogrid[:height, :width]
    radius = ((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2
    shading = np.clip(1.0 - 0.18 * radius, 0.72, 1.0)[..., None]
    plate = np.clip(plate.astype(np.float32) * shading * 0.82, 0, 255).astype(np.uint8)
    return plate, [x0, y0, x1 - x0, y1 - y0]


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _interpolate_point(start: tuple[int, int], target: tuple[int, int], amount: float) -> tuple[int, int]:
    return tuple(round(a + (b - a) * amount) for a, b in zip(start, target))  # type: ignore[return-value]


def _pose(action: str, frame_index: int, num_frames: int) -> dict[str, tuple[int, int]]:
    neutral = {
        "left_shoulder": (179, 82),
        "right_shoulder": (269, 82),
        "left_elbow": (153, 125),
        "right_elbow": (295, 125),
        "left_wrist": (164, 162),
        "right_wrist": (284, 162),
    }
    targets = {
        "left-raised-near-head": {
            "left_elbow": (145, 59), "left_wrist": (166, 27),
            "right_elbow": (299, 124), "right_wrist": (287, 163),
        },
        "right-raised-near-head": {
            "left_elbow": (149, 124), "left_wrist": (161, 163),
            "right_elbow": (303, 59), "right_wrist": (282, 27),
        },
        "cross-body": {
            "left_elbow": (158, 113), "left_wrist": (244, 129),
            "right_elbow": (290, 113), "right_wrist": (204, 143),
        },
        "bilateral-reach": {
            "left_elbow": (149, 105), "left_wrist": (105, 128),
            "right_elbow": (299, 105), "right_wrist": (343, 128),
        },
    }
    if action not in targets:
        raise ValueError(f"unknown action: {action}")
    phase = frame_index / max(1, num_frames - 1)
    amount = _smoothstep(min(1.0, phase / 0.72))
    breathing = round(2.0 * math.sin(phase * math.tau))
    result = dict(neutral)
    for joint, target in targets[action].items():
        result[joint] = _interpolate_point(neutral[joint], target, amount)
    for joint in result:
        x, y = result[joint]
        result[joint] = (x, y + breathing)
    return result


def _render_robot(
    cv2: Any,
    np: Any,
    frame: Any,
    pose: dict[str, tuple[int, int]],
    palette: list[list[int]],
    emblem: str,
) -> tuple[Any, dict[str, object]]:
    primary, secondary, accent = (tuple(int(channel) for channel in color) for color in palette)
    canvas = frame.copy()
    outline = (20, 23, 28)
    head_center = (224, 43)
    head_radius = 18
    neck_box = (214, 61, 234, 72)
    torso_polygon = [(184, 73), (264, 73), (274, 166), (174, 166)]
    left_hip, right_hip = (203, 166), (245, 166)
    left_knee, right_knee = (194, 211), (254, 211)
    left_foot, right_foot = (184, 245), (266, 245)

    for side in ("left", "right"):
        shoulder = pose[f"{side}_shoulder"]
        elbow = pose[f"{side}_elbow"]
        wrist = pose[f"{side}_wrist"]
        cv2.line(canvas, shoulder, elbow, outline, 22, cv2.LINE_AA)
        cv2.line(canvas, elbow, wrist, outline, 19, cv2.LINE_AA)
        cv2.line(canvas, shoulder, elbow, primary, 14, cv2.LINE_AA)
        cv2.line(canvas, elbow, wrist, secondary, 11, cv2.LINE_AA)
        for point in (shoulder, elbow, wrist):
            cv2.circle(canvas, point, 9, outline, -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 6, accent, -1, cv2.LINE_AA)

    for hip, knee, foot in ((left_hip, left_knee, left_foot), (right_hip, right_knee, right_foot)):
        cv2.line(canvas, hip, knee, outline, 24, cv2.LINE_AA)
        cv2.line(canvas, knee, foot, outline, 22, cv2.LINE_AA)
        cv2.line(canvas, hip, knee, secondary, 15, cv2.LINE_AA)
        cv2.line(canvas, knee, foot, primary, 13, cv2.LINE_AA)
        cv2.ellipse(canvas, (foot[0], foot[1] - 1), (18, 7), 0, 0, 360, outline, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (foot[0], foot[1] - 2), (14, 4), 0, 0, 360, accent, -1, cv2.LINE_AA)

    cv2.fillConvexPoly(canvas, np.array(torso_polygon), outline, cv2.LINE_AA)
    inner = [(190, 79), (258, 79), (266, 159), (182, 159)]
    cv2.fillConvexPoly(canvas, np.array(inner), primary, cv2.LINE_AA)
    cv2.rectangle(canvas, (neck_box[0], neck_box[1]), (neck_box[2], neck_box[3]), outline, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (218, 61), (230, 72), accent, -1, cv2.LINE_AA)
    cv2.circle(canvas, head_center, head_radius + 3, outline, -1, cv2.LINE_AA)
    cv2.circle(canvas, head_center, head_radius, secondary, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (211, 36), (237, 49), (12, 17, 23), -1, cv2.LINE_AA)
    cv2.circle(canvas, (218, 42), 3, accent, -1, cv2.LINE_AA)
    cv2.circle(canvas, (230, 42), 3, accent, -1, cv2.LINE_AA)
    cv2.circle(canvas, (224, 119), 19, outline, -1, cv2.LINE_AA)
    cv2.circle(canvas, (224, 119), 15, secondary, -1, cv2.LINE_AA)
    cv2.putText(canvas, emblem[:1], (216, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.52, accent, 2, cv2.LINE_AA)

    root_clearance = {
        side: not (
            203 <= pose[f"{side}_shoulder"][0] <= 245
            and 20 <= pose[f"{side}_shoulder"][1] <= 75
        )
        for side in ("left", "right")
    }
    if pose["left_shoulder"] == pose["right_shoulder"] or not all(root_clearance.values()):
        raise RuntimeError("analytic topology invariant failed")
    truth = {
        "head": {"center": list(head_center), "radius": head_radius},
        "neck_box_xyxy": list(neck_box),
        "torso_polygon": [list(point) for point in torso_polygon],
        "joints": {key: list(value) for key, value in pose.items()},
        "legs": {
            "left": [list(left_hip), list(left_knee), list(left_foot)],
            "right": [list(right_hip), list(right_knee), list(right_foot)],
        },
        "gates": {
            "single_robot_subject": True,
            "single_head_torso_chain": True,
            "exactly_two_arms": True,
            "unique_left_shoulder_origin": True,
            "unique_right_shoulder_origin": True,
            "arm_roots_clear_of_head_and_neck": all(root_clearance.values()),
            "continuous_arm_segments": True,
            "stable_robot_proportions": True,
            "no_extra_or_missing_limbs": True,
            "no_human_residual": True,
        },
        "gate_evidence": {
            "no_human_residual": "reviewed source ROI and generated storyboard",
            "all_other_gates": "analytic renderer construction",
        },
    }
    return canvas, truth


def _encode_clip(
    ffmpeg: Path,
    output: Path,
    frames: list[Any],
    fps: int,
    width: int,
    height: int,
    log: Path,
) -> list[str]:
    command = [
        str(ffmpeg), "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(command) + "  # raw BGR frames on stdin\n")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(stderr)
    if return_code:
        raise subprocess.CalledProcessError(return_code, command, stderr=stderr)
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/h3-identity-domain-curricula"))
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or ""))
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    source_manifest_path = (PROJECT_ROOT / str(config["source_manifest"])).resolve()
    source_manifest_path.relative_to(PROJECT_ROOT)
    source_records, source_manifest = _source_records(source_manifest_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = (
        args.experiment_dir.expanduser().resolve()
        if args.experiment_dir
        else args.output_root.expanduser().resolve() / f"{stamp}-{uuid4().hex[:8]}"
    )
    experiment.mkdir(parents=True, exist_ok=False)
    manifest_path = experiment / "manifest.json"
    log_path = experiment / "commands.log"
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "real_scene_texture_topology_positive_domain_curriculum",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": _git_state(),
        "seed": config["seed"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_config_sha256": source_manifest.get("config_sha256"),
        "background_policy": config["background_policy"],
        "records": [],
    }
    _write_json(manifest_path, manifest)
    try:
        if not args.ffmpeg.is_file():
            raise ValueError(f"ffmpeg is missing: {args.ffmpeg}")
        import cv2
        import numpy as np

        width = int(config["width"])
        height = int(config["height"])
        fps = int(config["fps"])
        num_frames = int(config["num_frames"])
        video_dir = experiment / "videos"
        truth_dir = experiment / "topology-truth"
        review_dir = experiment / "review"
        for directory in (video_dir, truth_dir, review_dir):
            directory.mkdir()
        plan_clips: list[dict[str, object]] = []
        records: list[dict[str, object]] = []
        review_frames: list[Any] = []
        for scene in config["scenes"]:
            record = source_records.get(str(scene["source_id"]))
            if record is None or record["split"] != scene["split"]:
                raise ValueError(f"source/split mismatch for {scene['scene_id']}")
            source = Path(str(record["path"])).resolve()
            source_frame = _capture_frame(cv2, source, float(scene["source_time_seconds"]))
            plate, roi_pixels = _background_plate(
                cv2, np, source_frame, scene["human_free_roi_normalized"], width, height
            )
            plate_path = review_dir / f"{scene['scene_id']}-background-plate.jpg"
            if not cv2.imwrite(str(plate_path), plate):
                raise RuntimeError(f"could not write {plate_path}")
            for action in ACTION_TAGS:
                clip_id = f"{scene['subject_id']}-{scene['scene_id']}-{action}"
                frames: list[Any] = []
                truths: list[dict[str, object]] = []
                for frame_index in range(num_frames):
                    pose = _pose(action, frame_index, num_frames)
                    rendered, truth = _render_robot(
                        cv2,
                        np,
                        plate,
                        pose,
                        scene["palette_bgr"],
                        str(scene["emblem"]),
                    )
                    truth["frame_index"] = frame_index
                    frames.append(rendered)
                    truths.append(truth)
                output = video_dir / f"{clip_id}.mp4"
                command = _encode_clip(args.ffmpeg, output, frames, fps, width, height, log_path)
                truth_path = truth_dir / f"{clip_id}.json"
                _write_json(
                    truth_path,
                    {
                        "schema_version": "1.0.0",
                        "clip_id": clip_id,
                        "coordinate_frame": "video-pixels-origin-top-left",
                        "width": width,
                        "height": height,
                        "frames": truths,
                    },
                )
                relative_output = output.relative_to(PROJECT_ROOT).as_posix()
                prompt_action = action.replace("-", " ")
                plan_clips.append(
                    {
                        "clip_id": clip_id,
                        "subject_id": scene["subject_id"],
                        "scene_id": scene["scene_id"],
                        "split": scene["split"],
                        "source_video": relative_output,
                        "prompt": (
                            "One full-body articulated humanoid robot with exactly two arms, "
                            "fixed unique shoulder origins and a clear head-neck gap performs "
                            f"{prompt_action} in a real-scene-derived workcell background."
                        ),
                        "source_start_seconds": 0.0,
                        "reference_frame": 0,
                        "license_id": record["license_id"],
                        "source_uri": record["landing_url"],
                        "review_status": "accepted",
                        "curriculum_tags": [*COMMON_TAGS, action],
                    }
                )
                records.append(
                    {
                        "clip_id": clip_id,
                        "split": scene["split"],
                        "subject_id": scene["subject_id"],
                        "scene_id": scene["scene_id"],
                        "action": action,
                        "source_id": scene["source_id"],
                        "source_sha256": record["sha256"],
                        "source_time_seconds": scene["source_time_seconds"],
                        "reviewed_roi_normalized": scene["human_free_roi_normalized"],
                        "reviewed_roi_pixels_xywh": roi_pixels,
                        "background_plate": str(plate_path),
                        "background_plate_sha256": _sha256(plate_path),
                        "video": str(output),
                        "video_sha256": _sha256(output),
                        "topology_truth": str(truth_path),
                        "topology_truth_sha256": _sha256(truth_path),
                        "encode_command": command,
                    }
                )
                if action == "left-raised-near-head":
                    review_frames.append(frames[-1])
        plan_path = experiment / "plan.json"
        _write_json(
            plan_path,
            {
                "schema_version": "1.0.0",
                "name": config["name"],
                "fps": fps,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "clips": plan_clips,
            },
        )
        plan = IdentityDatasetPlan.load(plan_path)
        assessment = DomainCurriculumContract().assess(plan)
        _write_json(experiment / "domain-curriculum-assessment.json", assessment.to_dict())
        if not assessment.passed:
            raise RuntimeError(f"domain curriculum failed: {assessment.failed_gates()}")
        tiles = [cv2.resize(frame, (224, 128), interpolation=cv2.INTER_AREA) for frame in review_frames]
        storyboard = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
        storyboard_path = review_dir / "held-same-action-six-domains.jpg"
        if not cv2.imwrite(str(storyboard_path), storyboard):
            raise RuntimeError(f"could not write {storyboard_path}")
        manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
                "plan": str(plan_path),
                "plan_sha256": _sha256(plan_path),
                "domain_curriculum_assessment": assessment.to_dict(),
                "storyboard": str(storyboard_path),
                "storyboard_sha256": _sha256(storyboard_path),
                "records": records,
                "limitations": [
                    "The backgrounds are deterministic crops of reviewed real-scene regions, not full source scenes.",
                    "The robot and topology truth are deterministic 2D supervision, not evidence of real-robot physics.",
                    "Human-free ROI review is visual and must be repeated if source time or crop coordinates change.",
                    "Adapter distribution remains blocked on combined upstream-model and training-data legal review.",
                ],
            }
        )
        shutil.copy2(config_path, experiment / "curriculum-config.json")
        _write_json(manifest_path, manifest)
        print(json.dumps({"experiment": str(experiment), "clips": len(records), "contract_passed": True}))
        return 0
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
