#!/usr/bin/env python3
"""Stitch two evaluated H3 windows and package a matched 10-second demo."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.h3_long_video import (  # noqa: E402
    merge_at_masked_seam,
    overlap_continuity_metrics,
)


DEFAULT_ACTION_LABELS = ("insert-flower", "handover-flower", "inspect-flower")


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


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    info: dict[str, float | int] = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    info["frames"] = len(frames)
    return frames, info


def _write_video(ffmpeg: Path, output: Path, frames: list[Any], fps: float) -> None:
    height, width = frames[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "-", "-an",
            "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt",
            "yuv420p", "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed to encode {output}")


def _resize_frames(cv2: Any, frames: list[Any], width: int, height: int) -> list[Any]:
    if frames[0].shape[:2] == (height, width):
        return frames
    return [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]


def build_subject_masks(
    cv2: Any,
    np: Any,
    source: list[Any],
    previous: list[Any],
    following: list[Any],
    *,
    following_start: int,
    threshold: float = 12.0,
) -> list[Any]:
    """Build conservative per-frame change support in one camera frame."""

    height, width = source[0].shape[:2]
    masks = [np.zeros((height, width), dtype=np.uint8) for _ in source]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    for absolute in range(len(source)):
        active = np.zeros((height, width), dtype=bool)
        if absolute < len(previous):
            difference = np.abs(
                previous[absolute].astype(np.float32) - source[absolute].astype(np.float32)
            )
            active |= np.max(difference, axis=2) >= threshold
        local = absolute - following_start
        if 0 <= local < len(following):
            difference = np.abs(
                following[local].astype(np.float32) - source[absolute].astype(np.float32)
            )
            active |= np.max(difference, axis=2) >= threshold
        mask = active.astype(np.uint8) * 255
        masks[absolute] = cv2.dilate(mask, kernel)
    return masks


def pairwise_distinctness(np: Any, first: list[Any], second: list[Any]) -> dict[str, float]:
    if len(first) != len(second):
        raise ValueError("variant frame counts differ")
    full, active_values = [], []
    for left, right in zip(first, second):
        difference = np.abs(left.astype(np.float32) - right.astype(np.float32))
        full.append(float(difference.mean()))
        active = difference.max(axis=2) >= 12.0
        active_values.append(float(difference[active].mean()) if active.any() else 0.0)
    return {
        "full_frame_mean_absolute_difference": float(np.mean(full)),
        "full_frame_peak_absolute_difference": float(np.max(full)),
        "active_pixel_mean_absolute_difference": float(np.mean(active_values)),
        "fraction_of_frames_above_2_mad": float(np.mean(np.asarray(full) >= 2.0)),
    }


def _header(cv2: Any, np: Any, frame: Any, title: str, subtitle: str) -> Any:
    width = 416
    video = cv2.resize(frame, (width, 240), interpolation=cv2.INTER_AREA)
    header = np.full((52, width, 3), 18, dtype=np.uint8)
    cv2.putText(header, title, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (92, 238, 170), 1, cv2.LINE_AA)
    cv2.putText(header, subtitle, (12, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 224, 228), 1, cv2.LINE_AA)
    return np.vstack((header, video))


def _comparison_frames(
    cv2: Any,
    np: Any,
    source: list[Any],
    variants: dict[str, list[Any]],
    *,
    action_labels: tuple[str, str, str] = DEFAULT_ACTION_LABELS,
    display_labels: dict[str, tuple[str, str]] | None = None,
    source_title: str = "REAL SOURCE",
    source_subtitle: str = "same 10.0 s scene interval",
) -> list[Any]:
    result = []
    labels = display_labels or {
        "insert-flower": ("INSERT", "right hand -> vase"),
        "handover-flower": ("HANDOVER", "right hand -> left hand"),
        "inspect-flower": ("INSPECT", "right hand -> eye line -> lower"),
    }
    for index, frame in enumerate(source):
        tiles = [_header(cv2, np, frame, source_title, source_subtitle)]
        for label in action_labels:
            tiles.append(_header(cv2, np, variants[label][index], *labels[label]))
        result.append(np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:]))))
    return result


def load_action_display(
    manifest_path: Path | None,
) -> tuple[tuple[str, str, str], dict[str, tuple[str, str]], str | None]:
    """Load three action labels without baking a task domain into the stitcher."""

    if manifest_path is None:
        return (
            DEFAULT_ACTION_LABELS,
            {
                "insert-flower": ("INSERT", "right hand -> vase"),
                "handover-flower": ("HANDOVER", "right hand -> left hand"),
                "inspect-flower": ("INSPECT", "right hand -> eye line -> lower"),
            },
            None,
        )
    payload = json.loads(manifest_path.read_text())
    actions = payload.get("actions")
    if not isinstance(actions, list) or len(actions) != 3:
        raise ValueError("action manifest must define exactly three actions")
    labels = tuple(str(action["label"]) for action in actions)
    if len(set(labels)) != 3:
        raise ValueError("action manifest labels must be unique")
    object_name = str(payload.get("object_name", "object"))
    suffix = f"-{object_name}"
    display: dict[str, tuple[str, str]] = {}
    for action in actions:
        label = str(action["label"])
        short = label.removesuffix(suffix).replace("-", " ").upper()
        instruction = " ".join(str(action.get("instruction", label)).split())
        if len(instruction) > 58:
            instruction = instruction[:55].rstrip() + "..."
        display[label] = (short, instruction)
    coordinate_frame = payload.get("coordinate_frame")
    return labels, display, str(coordinate_frame) if coordinate_frame else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--window-experiment", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-manifest", type=Path)
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--source-title", default="REAL SOURCE")
    parser.add_argument("--source-subtitle", default="same 10.0 s scene interval")
    parser.add_argument("--following-start-frame", type=int, default=116)
    parser.add_argument("--human-review", choices=("pending", "passed", "failed"), default="pending")
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    if len(args.window_experiment) != 2:
        raise ValueError("exactly two --window-experiment values are required")
    prepared = args.prepared_run.expanduser().resolve()
    experiments = [path.expanduser().resolve() for path in args.window_experiment]
    output_dir = args.output_dir.expanduser().resolve()
    action_manifest = (
        args.action_manifest.expanduser().resolve() if args.action_manifest else None
    )
    if action_manifest is not None and not action_manifest.is_file():
        raise ValueError(f"action manifest is missing: {action_manifest}")
    action_labels, display_labels, coordinate_frame = load_action_display(action_manifest)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"comparison already exists: {manifest_path}")
    ffmpeg = args.ffmpeg.expanduser().resolve()
    prepared_manifest = prepared / "manifest.json"
    source_path = (
        args.source_video.expanduser().resolve()
        if args.source_video
        else prepared / "input" / "real-source-240.mp4"
    )
    for path in (prepared_manifest, source_path, ffmpeg):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required input is missing or empty: {path}")

    import cv2
    import numpy as np

    source, source_info = _decode(cv2, source_path)
    if len(source) != 240 or abs(float(source_info["fps"]) - 24.0) > 1e-6:
        raise ValueError("comparison source must be exactly 240 frames at 24 FPS")
    stitched: dict[str, list[Any]] = {}
    action_records = []
    for label in action_labels:
        clips, evaluations = [], []
        for experiment in experiments:
            evaluation = experiment / "variants" / label / "agent-evaluation" / "evolution.json"
            candidate = experiment / "variants" / label / "agent-evaluation" / "final-background-locked.mp4"
            for path in (evaluation, candidate):
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"evaluated action window is missing: {path}")
            frames, info = _decode(cv2, candidate)
            if len(frames) != 124 or abs(float(info["fps"]) - 24.0) > 1e-6:
                raise ValueError(f"{candidate} is not a legal 124-frame H3 window")
            clips.append(frames)
            evaluations.append(json.loads(evaluation.read_text()))
        height, width = clips[0][0].shape[:2]
        if clips[1][0].shape[:2] != (height, width):
            raise RuntimeError("matched action windows have different resolutions")
        aligned_source = _resize_frames(cv2, source, width, height)
        masks = build_subject_masks(
            cv2,
            np,
            aligned_source,
            clips[0],
            clips[1],
            following_start=args.following_start_frame,
        )
        overlap_union = np.zeros((height, width), dtype=np.uint8)
        for mask in masks[args.following_start_frame:124]:
            overlap_union = cv2.bitwise_or(overlap_union, mask)
        if not np.any(overlap_union):
            overlap_union[:, :] = 255
        continuity = overlap_continuity_metrics(
            np,
            previous=clips[0],
            previous_start=0,
            following=clips[1],
            following_start=args.following_start_frame,
            subject_mask=overlap_union,
        )
        merged, seam = merge_at_masked_seam(
            np,
            current=clips[0],
            current_start=0,
            following=clips[1],
            following_start=args.following_start_frame,
            source=aligned_source,
            subject_masks=masks,
        )
        if len(merged) != 240:
            raise RuntimeError(f"stitched {label} has {len(merged)} frames instead of 240")
        output = output_dir / "variants" / label / f"{label}-10s.mp4"
        _write_video(ffmpeg, output, merged, 24.0)
        stitched[label] = merged
        scorecards = [item["best_scorecard"] for item in evaluations]
        aggregate = {
            key: min(float(scorecard[key]) for scorecard in scorecards)
            for key in (
                "background_lock",
                "object_lock",
                "subject_replacement",
                "robot_identity",
                "motion_preservation",
                "temporal_consistency",
                "epl_minimum",
            )
        }
        action_records.append(
            {
                "label": label,
                "window_evaluations": [
                    {
                        "path": str(experiment / "variants" / label / "agent-evaluation" / "evolution.json"),
                        "sha256": _sha256(experiment / "variants" / label / "agent-evaluation" / "evolution.json"),
                        "status": evaluation["status"],
                        "scorecard": evaluation["best_scorecard"],
                        "learned_repair_policy": evaluation.get("learned_repair_policy"),
                    }
                    for experiment, evaluation in zip(experiments, evaluations)
                ],
                "conservative_window_scorecard": aggregate,
                "continuity": continuity,
                "seam": seam,
                "output": str(output),
                "output_sha256": _sha256(output),
            }
        )

    pairwise = []
    for left_index, left in enumerate(action_labels):
        for right in action_labels[left_index + 1:]:
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    **pairwise_distinctness(np, stitched[left], stitched[right]),
                }
            )
    source_aligned = _resize_frames(
        cv2,
        source,
        stitched[action_labels[0]][0].shape[1],
        stitched[action_labels[0]][0].shape[0],
    )
    comparison = output_dir / "real-source-vs-three-actions-10s.mp4"
    _write_video(
        ffmpeg,
        comparison,
        _comparison_frames(
            cv2,
            np,
            source_aligned,
            stitched,
            action_labels=action_labels,
            display_labels=display_labels,
            source_title=args.source_title,
            source_subtitle=args.source_subtitle,
        ),
        24.0,
    )
    poster = output_dir / "poster.jpg"
    subprocess.run(
        [
            str(ffmpeg), "-y", "-v", "error", "-ss", "6.5", "-i", str(comparison),
            "-frames:v", "1", "-q:v", "2", str(poster),
        ],
        check=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    portable = output_dir / "portable"
    portable.mkdir(parents=True, exist_ok=True)
    for source_file in [comparison, poster, *[Path(item["output"]) for item in action_records]]:
        shutil.copy2(source_file, portable / source_file.name)

    minimum_distinctness = min(
        item["full_frame_mean_absolute_difference"] for item in pairwise
    )
    all_windows_accepted = all(
        window["status"] == "accepted"
        for action in action_records
        for window in action["window_evaluations"]
    )
    review_passed = args.human_review == "passed"
    strict_accepted = all_windows_accepted and minimum_distinctness >= 2.0 and review_passed
    packages = {}
    for name in ("numpy", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": "1.0.0",
        "method": "stateful_two_window_h3_action_control_comparison",
        "status": "accepted" if strict_accepted else "completed_partial",
        "honest_status": "WORKING" if strict_accepted else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "gpu": {
            "used": False,
            "reason": "CPU stitching of separately recorded GPU H3 experiments",
        },
        "coordinate_frame": coordinate_frame or "camera:H3_output_pixels aligned to real source frame index",
        "duration": {"frames": 240, "fps": 24, "seconds": 10.0},
        "prepared_run": {
            "path": str(prepared),
            "manifest_sha256": _sha256(prepared_manifest),
        },
        "action_manifest": (
            {"path": str(action_manifest), "sha256": _sha256(action_manifest)}
            if action_manifest
            else None
        ),
        "window_experiments": [str(path) for path in experiments],
        "actions": action_records,
        "pairwise_distinctness": pairwise,
        "acceptance": {
            "all_outputs_decode_to_240_frames": True,
            "all_outputs_exactly_10s": True,
            "minimum_pairwise_full_frame_mad": minimum_distinctness,
            "action_variants_visibly_distinct_proxy": minimum_distinctness >= 2.0,
            "all_windows_passed_strict_task_gates": all_windows_accepted,
            "human_review": args.human_review,
            "strict_accepted": strict_accepted,
        },
        "outputs": {
            "comparison": str(comparison),
            "comparison_sha256": _sha256(comparison),
            "poster": str(poster),
            "poster_sha256": _sha256(poster),
            "portable_dir": str(portable),
        },
        "limitations": [
            "MiniMax-H3 uses third-party NF4 weights and overlapping windows, not one native 10-second diffusion trajectory.",
            "Continuation is conditioned by one RGB frame; diffusion and metric robot state are not carried across windows.",
            "The real-world scene is a recorded observation; this is not real-robot execution or calibrated contact physics.",
            "Pixel and motion proxies do not replace dense human review of morphology, grasp causality, and seam quality.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"output": str(output_dir), "acceptance": manifest["acceptance"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
