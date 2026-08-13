#!/usr/bin/env python3
"""Join accepted flower windows with an explicit bounded overlap transition."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-prefix", type=Path, required=True)
    parser.add_argument("--left-window", type=Path, required=True)
    parser.add_argument("--right-window", type=Path, required=True)
    parser.add_argument("--prefix-global-start", type=int, required=True)
    parser.add_argument("--left-global-start", type=int, required=True)
    parser.add_argument("--right-global-start", type=int, required=True)
    parser.add_argument("--left-right-cut-global", type=int, required=True)
    parser.add_argument("--fade-weights", type=float, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compose_frames(
    np: Any,
    prefix: list[Any],
    left: list[Any],
    right: list[Any],
    *,
    prefix_global_start: int,
    left_global_start: int,
    right_global_start: int,
    left_right_cut_global: int,
    fade_weights: list[float],
) -> tuple[list[Any], dict[str, Any]]:
    prefix_end = prefix_global_start + len(prefix) - 1
    left_end = left_global_start + len(left) - 1
    right_end = right_global_start + len(right) - 1
    fade_start = prefix_end + 1
    fade_end = fade_start + len(fade_weights) - 1
    if not left_global_start <= fade_start <= fade_end <= left_right_cut_global <= left_end:
        raise ValueError("left window does not cover the fade and cut range")
    if not right_global_start <= left_right_cut_global + 1 <= right_end:
        raise ValueError("right window does not cover the post-cut range")
    if any(weight < 0.0 or weight > 1.0 for weight in fade_weights):
        raise ValueError("fade weights must lie in [0, 1]")
    if any(first < second for first, second in zip(fade_weights, fade_weights[1:])):
        raise ValueError("fade weights must be monotonically non-increasing")

    result = [frame.copy() for frame in prefix]
    anchor = prefix[-1].astype(np.float32)
    for global_frame, weight in zip(range(fade_start, fade_end + 1), fade_weights):
        incoming = left[global_frame - left_global_start].astype(np.float32)
        blended = anchor * weight + incoming * (1.0 - weight)
        result.append(np.clip(np.rint(blended), 0, 255).astype(np.uint8))
    for global_frame in range(fade_end + 1, left_right_cut_global + 1):
        result.append(left[global_frame - left_global_start].copy())
    for global_frame in range(left_right_cut_global + 1, right_end + 1):
        result.append(right[global_frame - right_global_start].copy())
    expected = right_end - prefix_global_start + 1
    if len(result) != expected:
        raise AssertionError(f"composed {len(result)} frames, expected {expected}")
    return result, {
        "global_range_inclusive": [prefix_global_start, right_end],
        "prefix_end_global": prefix_end,
        "fade_global_range_inclusive": [fade_start, fade_end],
        "left_right_cut_after_global": left_right_cut_global,
    }


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, Any]]:
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
        raise RuntimeError(f"video has no frames: {path}")
    height, width = frames[0].shape[:2]
    return frames, {"frames": len(frames), "fps": fps, "width": width, "height": height}


def _encode(ffmpeg: Path, frames: list[Any], output: Path, fps: float, *, lossless: bool) -> None:
    height, width = frames[0].shape[:2]
    codec = (
        ["-c:v", "libx264rgb", "-crf", "0", "-pix_fmt", "rgb24"]
        if lossless
        else ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"]
    )
    process = subprocess.Popen(
        [str(ffmpeg), "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "-", "-an",
         *codec, "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait():
        raise RuntimeError(f"FFmpeg failed to encode {output}")


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    inputs = {
        "accepted_prefix": args.accepted_prefix.expanduser().resolve(),
        "left_window": args.left_window.expanduser().resolve(),
        "right_window": args.right_window.expanduser().resolve(),
    }
    for path in inputs.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    output.mkdir(parents=True)
    (output / "provenance" / "execution-sources").mkdir(parents=True)

    import cv2
    import numpy as np

    decoded = {name: _decode(cv2, path) for name, path in inputs.items()}
    infos = {name: value[1] for name, value in decoded.items()}
    geometry = {(row["fps"], row["width"], row["height"]) for row in infos.values()}
    if len(geometry) != 1:
        raise ValueError("all input videos must share FPS and frame geometry")
    frames, composition = compose_frames(
        np,
        decoded["accepted_prefix"][0], decoded["left_window"][0], decoded["right_window"][0],
        prefix_global_start=args.prefix_global_start,
        left_global_start=args.left_global_start,
        right_global_start=args.right_global_start,
        left_right_cut_global=args.left_right_cut_global,
        fade_weights=list(args.fade_weights),
    )
    transitions = np.asarray(
        [float(np.abs(second.astype(np.float32) - first.astype(np.float32)).mean())
         for first, second in zip(frames, frames[1:])],
        dtype=np.float64,
    )
    ffmpeg = args.ffmpeg.expanduser().resolve() if args.ffmpeg else Path(shutil.which("ffmpeg") or "")
    if not ffmpeg.is_file():
        raise FileNotFoundError("FFmpeg is unavailable")
    lossless = output / f"candidate-{composition['global_range_inclusive'][0]:04d}-{composition['global_range_inclusive'][1]:04d}-{len(frames)}f-lossless.mp4"
    compatibility = output / f"candidate-{composition['global_range_inclusive'][0]:04d}-{composition['global_range_inclusive'][1]:04d}-{len(frames)}f.mp4"
    fps = infos["accepted_prefix"]["fps"]
    _encode(ffmpeg, frames, lossless, fps, lossless=True)
    _encode(ffmpeg, frames, compatibility, fps, lossless=False)
    script_copy = output / "provenance" / "execution-sources" / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), script_copy)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "decision": "RETRACK_AND_RUN_UNCHANGED_STRICT_GATES",
        "method": "bounded_full_frame_overlap_crossfade_then_hard_cut",
        "coordinate_frame": "camera:source_video_pixels",
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "seed": args.seed,
        "gpu": {"used": False, "reason": "CPU deterministic stitching of GPU-accepted inputs"},
        "packages": {"numpy": np.__version__, "opencv": cv2.__version__, "ffmpeg": str(ffmpeg)},
        "inputs": {name: {"path": str(path), "sha256": _sha256(path), **infos[name]} for name, path in inputs.items()},
        "composition": {**composition, "fade_weights_toward_prefix_terminal_frame": list(args.fade_weights)},
        "transition_metrics": {
            "mean": float(np.mean(transitions)),
            "p90": float(np.percentile(transitions, 90)),
            "maximum": float(np.max(transitions)),
            "prefix_to_first_fade": float(transitions[composition["prefix_end_global"] - args.prefix_global_start]),
            "left_to_right_cut": float(transitions[args.left_right_cut_global - args.prefix_global_start]),
        },
        "outputs": {
            "lossless": {"path": str(lossless), "sha256": _sha256(lossless)},
            "compatibility": {"path": str(compatibility), "sha256": _sha256(compatibility)},
        },
        "execution_source": {"path": str(script_copy), "sha256": _sha256(script_copy)},
        "limitations": [
            "Fade frames modify generated pixels and are not accepted until independent re-tracking and strict gates pass.",
            "The transition is image-space compositing, not 3D trajectory interpolation.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output), "lossless": str(lossless), "transition_metrics": manifest["transition_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
