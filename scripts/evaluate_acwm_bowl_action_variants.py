#!/usr/bin/env python3
"""Evaluate counterfactual bowl terminal states in matched MiniMax-H3 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAMERA_OUTPUT_FRAME = "camera:MiniMax-H3_output_pixels"


def classify_terminal_state(
    label: str,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    width: int,
    height: int,
) -> bool:
    """Apply mutually exclusive image-plane endpoint gates."""

    delta_x = end_xy[0] - start_xy[0]
    delta_y = end_xy[1] - start_xy[1]
    if label == "slide-left":
        return bool(delta_x <= -0.12 * width and abs(delta_y) <= 0.20 * height)
    if label == "slide-right":
        return bool(delta_x >= 0.12 * width and abs(delta_y) <= 0.20 * height)
    if label == "lift-up":
        return bool(delta_y <= -0.12 * height and abs(delta_x) <= 0.25 * width)
    raise ValueError(f"unsupported action label: {label}")


def pairwise_endpoint_floor(endpoints: dict[str, tuple[float, float]]) -> float:
    labels = sorted(endpoints)
    distances = []
    for left_index in range(len(labels)):
        for right_index in range(left_index + 1, len(labels)):
            left, right = endpoints[labels[left_index]], endpoints[labels[right_index]]
            distances.append(math.hypot(left[0] - right[0], left[1] - right[1]))
    if not distances:
        raise ValueError("at least two endpoints are required")
    return min(distances)


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
        raise RuntimeError(f"cannot decode {path}")
    info = {
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
    if len(frames) < 3:
        raise RuntimeError(f"decoded too few frames from {path}")
    info["frames"] = len(frames)
    return frames, info


def _yellow_bowl_observation(cv2: Any, np: Any, frame: Any) -> dict[str, object] | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # The saturation floor deliberately excludes the warm gray tabletop shadow,
    # which shares the bowl's hue but is not a second physical object.
    mask = cv2.inRange(hsv, (10, 100, 80), (43, 255, 255))
    mask[: round(frame.shape[0] * 0.10)] = 0
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = [
        index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= 300
    ]
    if not components:
        return None
    component = max(components, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    selected = np.where(labels == component, 255, 0).astype(np.uint8)
    moments = cv2.moments(selected)
    center = (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )
    pixels = hsv[:, :, 0][selected > 0]
    return {
        "center_xy": center,
        "area": int(stats[component, cv2.CC_STAT_AREA]),
        "mean_hue": float(pixels.mean()),
        "large_component_count": len(components),
    }


def _write_video(ffmpeg: Path, frames: list[Any], output: Path, fps: float) -> None:
    height, width = frames[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-", "-an",
            "-c:v", "libx264", "-crf", "16", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed to encode {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    control_root = args.control_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else experiment / "acwm-evaluation"
    )
    manifest_path = output_dir / "evaluation.json"
    if manifest_path.exists():
        raise FileExistsError(f"evaluation already exists: {manifest_path}")
    metadata_path = experiment / "metadata.json"
    control_manifest_path = control_root / "manifest.json"
    real_source_path = control_root / "input/real-scene-source-124f.mp4"
    ffmpeg = args.ffmpeg.expanduser().resolve()
    for path in (metadata_path, control_manifest_path, real_source_path, ffmpeg):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required input does not exist or is empty: {path}")
    metadata = json.loads(metadata_path.read_text())
    control_manifest = json.loads(control_manifest_path.read_text())
    if metadata.get("status") != "completed":
        raise ValueError("H3 experiment is not completed")
    import cv2
    import numpy as np

    variants: dict[str, list[Any]] = {}
    infos = {}
    observations = {}
    variant_records = []
    expected_by_label = {
        item["label"]: item for item in control_manifest["variants"]
    }
    for action in metadata["actions"]:
        label = action["label"]
        path = experiment / "variants" / label / "raw-h3-nf4.mp4"
        frames, info = _decode(cv2, path)
        if variants and info != next(iter(infos.values())):
            raise RuntimeError("H3 variants are not frame aligned")
        detected = [_yellow_bowl_observation(cv2, np, frame) for frame in frames]
        valid = [item for item in detected if item is not None]
        if not valid:
            raise RuntimeError(f"yellow bowl was never detected in {label}")
        variants[label] = frames
        infos[label] = info
        observations[label] = detected
        width, height = int(info["width"]), int(info["height"])
        control_trace = json.loads(
            (control_root / "variants" / label / "trajectory.json").read_text()
        )["trace"]
        scale_x = width / 896.0
        scale_y = height / 512.0
        expected = [
            (
                float(item["bowl_center_xy"][0]) * scale_x,
                float(item["bowl_center_xy"][1]) * scale_y,
            )
            for item in control_trace
        ]
        errors = []
        for observation, expected_xy in zip(detected, expected):
            if observation is None:
                continue
            center = observation["center_xy"]
            errors.append(math.hypot(center[0] - expected_xy[0], center[1] - expected_xy[1]))
        terminal_valid = [item for item in detected[-24:] if item is not None]
        start_valid = [item for item in detected[:24] if item is not None]
        start_xy = tuple(
            float(value)
            for value in np.median([item["center_xy"] for item in start_valid], axis=0)
        )
        end_xy = tuple(
            float(value)
            for value in np.median([item["center_xy"] for item in terminal_valid], axis=0)
        )
        expected_end = expected[-1]
        endpoint_error = math.hypot(end_xy[0] - expected_end[0], end_xy[1] - expected_end[1])
        areas = np.asarray([item["area"] for item in valid], dtype=np.float64)
        start_area = float(np.median([item["area"] for item in start_valid]))
        duplicate_fraction = float(
            np.mean([item is not None and item["large_component_count"] > 1 for item in detected])
        )
        direction_passed = classify_terminal_state(label, start_xy, end_xy, width, height)
        endpoint_passed = endpoint_error <= 0.10 * math.hypot(width, height)
        variant_records.append(
            {
                "label": label,
                "instruction": action["instruction"],
                "video": str(path),
                "video_sha256": _sha256(path),
                "coordinate_frame": CAMERA_OUTPUT_FRAME,
                "detected_frame_fraction": len(valid) / len(detected),
                "start_center_xy": start_xy,
                "terminal_center_xy": end_xy,
                "expected_terminal_center_xy": expected_end,
                "mean_control_path_error_pixels": float(np.mean(errors)),
                "terminal_error_pixels": endpoint_error,
                "terminal_direction_passed": direction_passed,
                "terminal_target_radius_passed": endpoint_passed,
                "median_area_ratio_to_start": float(np.median(areas) / start_area),
                "duplicate_component_fraction": duplicate_fraction,
                "single_bowl_identity_proxy_passed": duplicate_fraction <= 0.10,
            }
        )

    info = next(iter(infos.values()))
    width, height = int(info["width"]), int(info["height"])
    endpoints = {
        record["label"]: tuple(record["terminal_center_xy"]) for record in variant_records
    }
    endpoint_floor = pairwise_endpoint_floor(endpoints)
    pairwise_background = []
    labels = [item["label"] for item in variant_records]
    upper_end = round(height * 0.28)
    for left_index in range(len(labels)):
        for right_index in range(left_index + 1, len(labels)):
            left, right = variants[labels[left_index]], variants[labels[right_index]]
            values = [
                float(np.abs(a[:upper_end].astype(np.float32) - b[:upper_end].astype(np.float32)).mean())
                for a, b in zip(left, right)
            ]
            pairwise_background.append(
                {
                    "left": labels[left_index],
                    "right": labels[right_index],
                    "upper_background_pairwise_mad": float(np.mean(values)),
                }
            )
    background_ceiling = max(item["upper_background_pairwise_mad"] for item in pairwise_background)
    all_terminal = all(
        item["terminal_direction_passed"] and item["terminal_target_radius_passed"]
        for item in variant_records
    )
    all_detected = all(item["detected_frame_fraction"] >= 0.90 for item in variant_records)
    all_single = all(item["single_bowl_identity_proxy_passed"] for item in variant_records)
    separated = endpoint_floor >= 0.18 * min(width, height)
    background_passed = background_ceiling <= 18.0
    accepted = all_terminal and all_detected and all_single and separated and background_passed

    real_source_frames, real_source_info = _decode(cv2, real_source_path)
    if len(real_source_frames) != int(info["frames"]):
        raise RuntimeError("real source and H3 outputs are not frame aligned")
    tile_width, tile_height, header_height = 416, 240, 52
    colors = {"slide-left": (80, 230, 180), "slide-right": (80, 190, 245), "lift-up": (235, 190, 80)}
    comparison_frames = []
    frame_count = int(info["frames"])
    for frame_index in range(frame_count):
        source_tile = cv2.resize(
            real_source_frames[frame_index],
            (tile_width, tile_height),
            interpolation=cv2.INTER_AREA,
        )
        source_header = np.full((header_height, tile_width, 3), 16, dtype=np.uint8)
        cv2.putText(source_header, "REAL SOURCE", (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 220, 225), 1, cv2.LINE_AA)
        cv2.putText(source_header, "Hand2Dex-2 / captured lab video", (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (165, 174, 180), 1, cv2.LINE_AA)
        tiles = [np.vstack((source_header, source_tile))]
        for record in variant_records:
            label = record["label"]
            tile = cv2.resize(variants[label][frame_index], (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            observation = observations[label][frame_index]
            if observation is not None:
                center = observation["center_xy"]
                cv2.circle(
                    tile,
                    (round(center[0] * tile_width / width), round(center[1] * tile_height / height)),
                    7,
                    colors[label],
                    2,
                    cv2.LINE_AA,
                )
            header = np.full((header_height, tile_width, 3), 16, dtype=np.uint8)
            cv2.putText(header, label.upper(), (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[label], 1, cv2.LINE_AA)
            cv2.putText(
                header,
                f"terminal: {record['terminal_center_xy'][0]:.0f}, {record['terminal_center_xy'][1]:.0f}",
                (12, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (220, 224, 228),
                1,
                cv2.LINE_AA,
            )
            tiles.append(np.vstack((header, tile)))
        comparison_frames.append(
            np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
        )
    comparison_video = output_dir / "three-action-acwm-comparison.mp4"
    _write_video(ffmpeg, comparison_frames, comparison_video, float(info["fps"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    poster = output_dir / "poster.jpg"
    cv2.imwrite(str(poster), comparison_frames[-1])
    storyboard = output_dir / "storyboard.jpg"
    sample_indices = [0, round((frame_count - 1) * 0.5), frame_count - 1]
    cv2.imwrite(str(storyboard), np.vstack([comparison_frames[index] for index in sample_indices]))
    manifest = {
        "schema_version": "1.0.0",
        "method": "matched_action_conditioned_world_model_bowl_terminal_state_evaluation",
        "status": "accepted" if accepted else "rejected",
        "honest_status": "WORKING" if accepted else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "coordinate_frames": {
            "H3_outputs": CAMERA_OUTPUT_FRAME,
            "control_to_output": {
                "from": "camera:hand2dex_2_reference_pixels",
                "to": CAMERA_OUTPUT_FRAME,
                "operation": "independent_axis_scale",
                "scale_x": width / 896.0,
                "scale_y": height / 512.0,
            },
        },
        "inputs": {
            "experiment_metadata": {"path": str(metadata_path), "sha256": _sha256(metadata_path)},
            "control_manifest": {"path": str(control_manifest_path), "sha256": _sha256(control_manifest_path)},
            "real_source": {
                "path": str(real_source_path),
                "sha256": _sha256(real_source_path),
                "video_info": real_source_info,
            },
        },
        "video_info": info,
        "variants": variant_records,
        "pairwise_terminal_endpoint_floor_pixels": endpoint_floor,
        "pairwise_upper_background": pairwise_background,
        "acceptance": {
            "all_three_terminal_directions_passed": all_terminal,
            "yellow_bowl_detected_in_at_least_90pct_frames": all_detected,
            "single_bowl_identity_proxy_passed": all_single,
            "pairwise_terminal_separation_passed": separated,
            "upper_background_pairwise_mad_passed": background_passed,
            "upper_background_pairwise_mad_ceiling": background_ceiling,
        },
        "outputs": {
            "comparison_video": str(comparison_video),
            "comparison_video_sha256": _sha256(comparison_video),
            "poster": str(poster),
            "poster_sha256": _sha256(poster),
            "storyboard": str(storyboard),
            "storyboard_sha256": _sha256(storyboard),
        },
        "limitations": [
            "Yellow-HSV detection is a deterministic proxy, not a learned object tracker.",
            "Image-plane upward motion is evidence of a different terminal state but not metric 3-D height.",
            "The videos are world-model generations, not real-robot executions or physics validation.",
            "MiniMax-H3 uses third-party NF4 weights rather than official BF16 weights.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"evaluation": str(output_dir), "status": manifest["status"], "acceptance": manifest["acceptance"]}, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
