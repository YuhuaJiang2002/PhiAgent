#!/usr/bin/env python3
"""Package three instruction-conditioned AC-WM tasks in one Ego scene."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
        return {"available": True, "head": head, "status": status}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": repr(exc)}


def _labeled_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or label in result:
            raise ValueError("--action requires unique LABEL=PATH pairs")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"action video is missing: {path}")
        result[label] = path
    if len(result) != 3:
        raise ValueError("--action must be supplied exactly three times")
    return result


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {path}")
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
    info["frames"] = len(frames)
    if len(frames) != 240 or abs(float(info["fps"]) - 24.0) > 1e-6:
        raise ValueError(f"video is not exactly 240 frames at 24 FPS: {path} {info}")
    return frames, info


def _tile(cv2: Any, np: Any, frame: Any, title: str, subtitle: str) -> Any:
    width = 416
    image = cv2.resize(frame, (width, 240), interpolation=cv2.INTER_AREA)
    header = np.full((58, width, 3), 18, dtype=np.uint8)
    cv2.putText(
        header, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (92, 238, 170), 1, cv2.LINE_AA,
    )
    cv2.putText(
        header, subtitle, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
        (220, 224, 228), 1, cv2.LINE_AA,
    )
    return np.vstack((header, image))


def _write_video(ffmpeg: Path, output: Path, frames: list[Any]) -> None:
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(
        [
            str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", "24",
            "-i", "-", "-an", "-c:v", "libx264", "-crf", "14",
            "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags",
            "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed to write {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--action", action="append", default=[])
    parser.add_argument("--action-manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    args = parser.parse_args()

    import cv2
    import numpy as np

    project_root = Path(__file__).resolve().parents[1]
    source_path = args.source.expanduser().resolve()
    action_manifest_path = args.action_manifest.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    for path in (source_path, action_manifest_path, evaluation_path, ffmpeg):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required input is missing: {path}")
    action_paths = _labeled_paths(args.action)
    requested = json.loads(action_manifest_path.read_text())
    action_items = requested["actions"]
    labels = tuple(str(item["label"]) for item in action_items)
    if set(labels) != set(action_paths) or len(labels) != 3:
        raise ValueError("video labels must exactly match the three-task action manifest")
    evaluation = json.loads(evaluation_path.read_text())
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"demo directory already exists: {output}")
    output.mkdir(parents=True)

    source, source_info = _decode(cv2, source_path)
    action_frames: dict[str, list[Any]] = {}
    records: dict[str, Any] = {}
    instructions = {str(item["label"]): str(item["instruction"]) for item in action_items}
    for label in labels:
        action_frames[label], info = _decode(cv2, action_paths[label])
        records[label] = {
            "instruction": instructions[label],
            "path": str(action_paths[label]),
            "sha256": _sha256(action_paths[label]),
            **info,
        }

    display = {
        "pour-bottle": ("POUR BOTTLE", "lift, tilt, return upright"),
        "shake-bottle": ("SHAKE BOTTLE", "secure, oscillate, settle"),
        "place-bottle-rack": ("PLACE IN RACK", "carry, lower, support, release"),
        "handover-bottle": (
            "HAND OVER BOTTLE",
            "screen-right -> dual contact -> screen-left",
        ),
        "unscrew-bottle-cap": ("UNSCREW CAP", "stabilize, twist, lift cap"),
        "rinse-bottle": ("RINSE BOTTLE", "faucet reach + bottle rotation"),
    }
    comparison_frames = []
    generated_only_frames = []
    for index in range(240):
        action_tiles = [
            _tile(cv2, np, action_frames[label][index], *display[label])
            for label in labels
        ]
        tiles = [
            _tile(cv2, np, source[index], "SAME REAL EGO SCENE", "EPIC-KITCHENS P03_28"),
            *action_tiles,
        ]
        comparison_frames.append(
            np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
        )
        generated_only_frames.append(np.hstack(action_tiles))

    video = output / "ego-same-scene-three-instruction-tasks-10s.mp4"
    poster = output / "ego-same-scene-three-instruction-tasks-poster.jpg"
    generated_video = output / "ego-three-instruction-tasks-generated-only-10s.mp4"
    generated_poster = output / "ego-three-instruction-tasks-generated-only-poster.jpg"
    _write_video(ffmpeg, video, comparison_frames)
    _write_video(ffmpeg, generated_video, generated_only_frames)
    if not cv2.imwrite(str(poster), comparison_frames[120]):
        raise RuntimeError("could not write comparison poster")
    if not cv2.imwrite(str(generated_poster), generated_only_frames[120]):
        raise RuntimeError("could not write generated-only poster")
    manifest = {
        "schema_version": "1.0.0",
        "status": evaluation.get("status", "PARTIAL"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "opencv-python")
        },
        "git": _git_state(project_root),
        "seed": 20260811,
        "claim_boundary": "Generated camera-frame action visualization; not physical-robot execution.",
        "source": {"path": str(source_path), "sha256": _sha256(source_path), **source_info},
        "action_manifest": {
            "path": str(action_manifest_path),
            "sha256": _sha256(action_manifest_path),
        },
        "evaluation": {
            "path": str(evaluation_path),
            "sha256": _sha256(evaluation_path),
            "accepted": evaluation.get("accepted", False),
        },
        "actions": records,
        "outputs": {
            path.name: {"sha256": _sha256(path)}
            for path in (video, poster, generated_video, generated_poster)
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
