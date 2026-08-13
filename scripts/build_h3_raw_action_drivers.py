#!/usr/bin/env python3
"""Build sharp, un-repaired ten-second H3 action drivers for a second-stage model.

These videos are intermediate motion/geometry controls.  They intentionally bypass
the rejected background-lock repair, which can reintroduce source-human pixels.
"""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(cv2: Any, path: Path) -> tuple[list[Any], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"decoded no frames: {path}")
    return frames, fps


def _write_video(ffmpeg: Path, output: Path, frames: list[Any], fps: float) -> None:
    height, width = frames[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
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
            f"{fps:.8f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed to encode {output}")


def _git_state(root: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--short"], cwd=root, check=False,
        capture_output=True, text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False,
        capture_output=True, text=True,
    )
    return {
        "available": status.returncode == 0,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
        "error": status.stderr.strip() if status.returncode else None,
    }


def _subject_masks(
    cv2: Any,
    np: Any,
    source: list[Any],
    previous: list[Any],
    following: list[Any],
    following_start: int,
) -> list[Any]:
    height, width = previous[0].shape[:2]
    aligned_source = [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        for frame in source
    ]
    masks = [np.zeros((height, width), dtype=np.uint8) for _ in source]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    for absolute in range(len(source)):
        active = np.zeros((height, width), dtype=bool)
        if absolute < len(previous):
            difference = np.abs(
                previous[absolute].astype(np.float32)
                - aligned_source[absolute].astype(np.float32)
            )
            active |= np.max(difference, axis=2) >= 12.0
        local = absolute - following_start
        if 0 <= local < len(following):
            difference = np.abs(
                following[local].astype(np.float32)
                - aligned_source[absolute].astype(np.float32)
            )
            active |= np.max(difference, axis=2) >= 12.0
        masks[absolute] = cv2.dilate(active.astype(np.uint8) * 255, kernel)
    return masks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--window-zero", type=Path, required=True)
    parser.add_argument("--window-one", type=Path, required=True)
    parser.add_argument("--action-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--following-start-frame", type=int, default=116)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"driver manifest already exists: {manifest_path}")
    action_manifest = args.action_manifest.expanduser().resolve()
    action_payload = json.loads(action_manifest.read_text())
    labels = [str(item["label"]) for item in action_payload.get("actions", [])]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("action manifest must contain unique action labels")

    import cv2
    import numpy as np

    source, source_fps = _decode(cv2, args.source_video.expanduser().resolve())
    if len(source) != 240 or abs(source_fps - 24.0) > 1e-6:
        raise ValueError("source must contain exactly 240 frames at 24 FPS")
    records = []
    for label in labels:
        paths = [
            root.expanduser().resolve() / "variants" / label / "raw-h3-nf4.mp4"
            for root in (args.window_zero, args.window_one)
        ]
        clips = []
        for path in paths:
            frames, fps = _decode(cv2, path)
            if len(frames) != 124 or abs(fps - 24.0) > 1e-6:
                raise ValueError(f"raw H3 window is not 124 frames at 24 FPS: {path}")
            clips.append(frames)
        masks = _subject_masks(
            cv2,
            np,
            source,
            clips[0],
            clips[1],
            args.following_start_frame,
        )
        overlap_mask = np.zeros_like(masks[0])
        for mask in masks[args.following_start_frame : len(clips[0])]:
            overlap_mask = cv2.bitwise_or(overlap_mask, mask)
        if not np.any(overlap_mask):
            overlap_mask[:, :] = 255
        continuity = overlap_continuity_metrics(
            np,
            previous=clips[0],
            previous_start=0,
            following=clips[1],
            following_start=args.following_start_frame,
            subject_mask=overlap_mask,
        )
        aligned_source = [
            cv2.resize(
                frame,
                (clips[0][0].shape[1], clips[0][0].shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            for frame in source
        ]
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
            raise RuntimeError(f"stitched {label} driver has {len(merged)} frames")
        output = output_dir / "variants" / label / f"{label}-raw-driver-10s.mp4"
        _write_video(args.ffmpeg.expanduser().resolve(), output, merged, 24.0)
        records.append(
            {
                "label": label,
                "source_windows": [str(path) for path in paths],
                "source_window_sha256": [_sha256(path) for path in paths],
                "seam": seam,
                "continuity": continuity,
                "output": str(output),
                "output_sha256": _sha256(output),
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "status": "intermediate_control_only",
                "method": "raw_h3_hard_seam_no_background_lock_repair",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "command": [sys.executable, *sys.argv],
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "opencv-python")
                },
                "git": _git_state(PROJECT_ROOT),
                "gpu": {"used": False, "reason": "deterministic H3 window stitching"},
                "coordinate_frame": action_payload.get("coordinate_frame"),
                "action_manifest": str(action_manifest),
                "action_manifest_sha256": _sha256(action_manifest),
                "source_video": str(args.source_video.expanduser().resolve()),
                "source_video_sha256": _sha256(args.source_video.expanduser().resolve()),
                "actions": records,
                "limitations": [
                    "These are second-stage motion drivers, not accepted deliverables.",
                    "A hard seam remains and must be regularized by the stateful refinement stage.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
