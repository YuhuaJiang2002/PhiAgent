#!/usr/bin/env python3
"""Repair a small set of verified temporal spikes with motion-aware bridges."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def group_transition_frames(frames: list[int] | tuple[int, ...]) -> list[tuple[int, int]]:
    """Collapse consecutive transition-frame numbers into inclusive groups."""

    normalized = sorted(set(frames))
    if not normalized:
        raise ValueError("at least one transition frame is required")
    if normalized[0] < 1:
        raise ValueError("transition frames must be positive")
    groups: list[tuple[int, int]] = []
    start = previous = normalized[0]
    for frame in normalized[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        groups.append((start, previous))
        start = previous = frame
    groups.append((start, previous))
    return groups


def intervals_overlap(
    first_start: int,
    first_end_exclusive: int,
    second_start: int,
    second_end_exclusive: int,
) -> bool:
    """Return whether two half-open frame intervals overlap."""

    return max(first_start, second_start) < min(
        first_end_exclusive, second_end_exclusive
    )


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    info: dict[str, float | int] = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "reported_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    info["decoded_frames"] = len(frames)
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    return frames, info


def motion_bridge(
    cv2: Any,
    np: Any,
    first: Any,
    second: Any,
    count: int,
    *,
    mode: str = "flow",
) -> list[Any]:
    """Create ``count`` intermediate frames without changing the endpoints."""

    if count < 1:
        raise ValueError("bridge count must be positive")
    if first.shape != second.shape:
        raise ValueError("bridge endpoints must have the same shape")
    flow_forward = flow_backward = None
    grid_x = grid_y = None
    if mode == "flow":
        first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
        estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        estimator.setUseSpatialPropagation(True)
        flow_forward = estimator.calc(first_gray, second_gray, None)
        estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        estimator.setUseSpatialPropagation(True)
        flow_backward = estimator.calc(second_gray, first_gray, None)
        height, width = first.shape[:2]
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
    elif mode != "crossfade":
        raise ValueError(f"unsupported bridge mode: {mode}")

    result = []
    for index in range(1, count + 1):
        progress = index / (count + 1)
        alpha = 0.5 - 0.5 * math.cos(math.pi * progress)
        if mode == "flow":
            assert flow_forward is not None and flow_backward is not None
            assert grid_x is not None and grid_y is not None
            first_warped = cv2.remap(
                first,
                grid_x - progress * flow_forward[..., 0],
                grid_y - progress * flow_forward[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            second_warped = cv2.remap(
                second,
                grid_x - (1.0 - progress) * flow_backward[..., 0],
                grid_y - (1.0 - progress) * flow_backward[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
        else:
            first_warped = first
            second_warped = second
        result.append(
            np.clip(
                np.rint(
                    first_warped.astype(np.float32) * (1.0 - alpha)
                    + second_warped.astype(np.float32) * alpha
                ),
                0,
                255,
            ).astype(np.uint8)
        )
    return result


def _transition_energy(cv2: Any, np: Any, frames: list[Any]) -> list[float]:
    gray = [
        cv2.cvtColor(
            cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        for frame in frames
    ]
    return [
        float(np.mean(cv2.absdiff(gray[index], gray[index - 1])))
        for index in range(1, len(gray))
    ]


def _energy_summary(np: Any, energy: list[float], selected: list[int]) -> dict[str, Any]:
    median = float(np.median(energy))
    maximum = max(energy)
    return {
        "median_transition": median,
        "maximum_transition": maximum,
        "maximum_transition_ratio": maximum / max(median, 1e-6),
        "selected_transition_energy": {
            str(frame): energy[frame - 1] for frame in selected
        },
    }


def _writer(ffmpeg: Path, output: Path, width: int, height: int, fps: float) -> Any:
    return subprocess.Popen(
        [
            str(ffmpeg),
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
            "12",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transition-frame", type=int, action="append", required=True)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--mode", choices=("flow", "crossfade"), default="flow")
    parser.add_argument("--protected-start-frame", type=int)
    parser.add_argument("--protected-end-frame-exclusive", type=int)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--human-review", choices=("pending", "passed", "failed"), default="pending")
    return parser


def main() -> int:
    args = _parser().parse_args()
    candidate_path = args.candidate.expanduser().resolve()
    source_path = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"repair output already exists: {manifest_path}")
    if args.radius < 2:
        raise ValueError("repair radius must be at least two frames")
    protected = None
    if (args.protected_start_frame is None) != (
        args.protected_end_frame_exclusive is None
    ):
        raise ValueError("both protected frame bounds must be provided together")
    if args.protected_start_frame is not None:
        if not (
            0 <= args.protected_start_frame < args.protected_end_frame_exclusive
        ):
            raise ValueError("protected frame range is invalid")
        protected = (
            args.protected_start_frame,
            args.protected_end_frame_exclusive,
        )
    ffmpeg = args.ffmpeg.expanduser().resolve()
    for label, path in (
        ("candidate", candidate_path),
        ("source", source_path),
        ("FFmpeg", ffmpeg),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    candidate, candidate_info = _decode(cv2, candidate_path)
    source, source_info = _decode(cv2, source_path)
    if len(candidate) != len(source):
        raise RuntimeError("candidate and source frame counts differ")
    groups = group_transition_frames(args.transition_frame)
    repaired = [frame.copy() for frame in candidate]
    repairs = []
    for transition_start, transition_end in groups:
        left_index = transition_start - args.radius
        right_index = transition_end + args.radius
        if left_index < 0 or right_index >= len(repaired):
            raise ValueError("repair interval falls outside the video")
        if protected is not None and intervals_overlap(
            left_index + 1,
            right_index,
            protected[0],
            protected[1],
        ):
            raise ValueError(
                "repair interval overlaps protected frames: "
                f"[{left_index + 1}, {right_index}) vs "
                f"[{protected[0]}, {protected[1]})"
            )
        intermediate = motion_bridge(
            cv2,
            np,
            candidate[left_index],
            candidate[right_index],
            right_index - left_index - 1,
            mode=args.mode,
        )
        repaired[left_index + 1 : right_index] = intermediate
        repairs.append(
            {
                "transition_start_frame": transition_start,
                "transition_end_frame": transition_end,
                "left_endpoint_frame": left_index,
                "right_endpoint_frame": right_index,
                "replaced_start_frame": left_index + 1,
                "replaced_end_frame_exclusive": right_index,
            }
        )

    before = _transition_energy(cv2, np, candidate)
    after = _transition_energy(cv2, np, repaired)
    selected = sorted(set(args.transition_frame))
    metrics = {
        "decoded_frames": len(repaired),
        "median_transition_before": float(np.median(before)),
        "median_transition_after": float(np.median(after)),
        "maximum_transition_before": max(before),
        "maximum_transition_after": max(after),
        "maximum_transition_ratio_before": max(before) / max(float(np.median(before)), 1e-6),
        "maximum_transition_ratio_after": max(after) / max(float(np.median(after)), 1e-6),
        "selected_transition_energy_before": {str(frame): before[frame - 1] for frame in selected},
        "selected_transition_energy_after": {str(frame): after[frame - 1] for frame in selected},
    }
    output = output_dir / "wan-animate2-full-27s-spike-repaired.mp4"
    writer = _writer(
        ffmpeg,
        output,
        int(candidate_info["width"]),
        int(candidate_info["height"]),
        float(candidate_info["fps"]),
    )
    try:
        assert writer.stdin is not None
        for frame in repaired:
            writer.stdin.write(frame.tobytes())
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
        if writer.wait():
            raise RuntimeError("FFmpeg failed to encode repaired video")
    subprocess.run(
        [str(ffmpeg), "-v", "error", "-i", str(output), "-f", "null", "-"],
        check=True,
    )
    encoded, encoded_info = _decode(cv2, output)
    if len(encoded) != len(repaired):
        raise RuntimeError("encoded repair changed the video frame count")
    encoded_energy = _transition_energy(cv2, np, encoded)
    metrics["encoded_output"] = _energy_summary(np, encoded_energy, selected)
    review_passed = args.human_review == "passed"
    packages = {}
    for package in ("numpy", "opencv-python", "opencv-python-headless"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {
        "schema_version": "1.0.0",
        "method": (
            "bidirectional_dis_flow_local_transition_bridge_v1"
            if args.mode == "flow"
            else "local_cosine_crossfade_transition_bridge_v1"
        ),
        "status": "accepted" if review_passed else "rejected" if args.human_review == "failed" else "review_required",
        "honest_status": "WORKING" if review_passed else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "packages": packages,
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
            "info": candidate_info,
        },
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "info": source_info,
        },
        "mode": args.mode,
        "radius": args.radius,
        "repairs": repairs,
        "protected_frame_range": list(protected) if protected else None,
        "protected_range_unchanged_preencode": (
            all(
                np.array_equal(candidate[index], repaired[index])
                for index in range(protected[0], protected[1])
            )
            if protected
            else None
        ),
        "metrics": metrics,
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
            "info": encoded_info,
        },
        "human_review": args.human_review,
        "limitations": [
            "Only explicitly listed transition neighborhoods are changed; this is not a global temporal model.",
            (
                "Optical-flow interpolation can distort thin stems or occlusion boundaries and requires consecutive-frame human review."
                if args.mode == "flow"
                else "Crossfade interpolation can ghost fast motion and requires consecutive-frame human review."
            ),
            "Transition repair does not establish exact contact, kinematics, physics, or robot execution.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"output": str(output), "metrics": metrics, "status": manifest["status"]}, indent=2))
    return 0 if review_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
