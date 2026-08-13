#!/usr/bin/env python3
"""Package source/action and rejected/recovered AC-WM comparison videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _labeled_paths(values: list[str], option: str) -> dict[str, Path]:
    result = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or label in result:
            raise ValueError(f"{option} requires unique LABEL=PATH pairs")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing {option} video: {path}")
        result[label] = path
    if len(result) != 3:
        raise ValueError(f"{option} must be supplied exactly three times")
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
    header = np.full((52, width, 3), 18, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (92, 238, 170),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        subtitle,
        (12, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (220, 224, 228),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, image))


def _write_video(ffmpeg: Path, output: Path, frames: list[Any]) -> None:
    height, width = frames[0].shape[:2]
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
            "24",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "14",
            "-preset",
            "slow",
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
        raise RuntimeError(f"ffmpeg failed to write {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--old-action", action="append", default=[])
    parser.add_argument("--new-action", action="append", default=[])
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    args = parser.parse_args()

    import cv2
    import numpy as np

    source_path = args.source.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    if not evaluation_path.is_file() or evaluation_path.stat().st_size == 0:
        raise ValueError(f"evaluation is missing: {evaluation_path}")
    evaluation = json.loads(evaluation_path.read_text())
    old_paths = _labeled_paths(args.old_action, "--old-action")
    new_paths = _labeled_paths(args.new_action, "--new-action")
    if set(old_paths) != set(new_paths):
        raise ValueError("old and new action labels must match")
    labels = tuple(sorted(new_paths))
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"demo directory already exists: {output}")
    output.mkdir(parents=True)
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if not ffmpeg.is_file():
        raise ValueError(f"ffmpeg is missing: {ffmpeg}")

    source, source_info = _decode(cv2, source_path)
    old, new, records = {}, {}, {}
    for label in labels:
        old[label], old_info = _decode(cv2, old_paths[label])
        new[label], new_info = _decode(cv2, new_paths[label])
        records[label] = {
            "old": {"path": str(old_paths[label]), "sha256": _sha256(old_paths[label]), **old_info},
            "new": {"path": str(new_paths[label]), "sha256": _sha256(new_paths[label]), **new_info},
        }

    display = {
        "handover-bottle": ("HANDOVER", "robot-to-robot transfer"),
        "pour-bottle": ("POUR", "tilt bottle over sink"),
        "shake-bottle": ("SHAKE", "oscillatory bottle motion"),
    }
    source_action_frames = []
    quality_frames = []
    for index in range(240):
        source_tile = _tile(cv2, np, source[index], "REAL EGO SOURCE", "EPIC-KITCHENS P03_28")
        action_tiles = [
            _tile(cv2, np, new[label][index], *display.get(label, (label.upper(), "recovered")))
            for label in labels
        ]
        source_action_frames.append(
            np.vstack((np.hstack((source_tile, action_tiles[0])), np.hstack(action_tiles[1:])))
        )
        old_tiles = [
            _tile(cv2, np, old[label][index], f"OLD / {display[label][0]}", "rejected: hand ghosts + blur")
            for label in labels
        ]
        new_tiles = [
            _tile(cv2, np, new[label][index], f"NEW / {display[label][0]}", "robot-factored, no post blur")
            for label in labels
        ]
        quality_frames.append(np.vstack((np.hstack(old_tiles), np.hstack(new_tiles))))

    action_output = output / "ego-source-vs-three-recovered-actions-10s.mp4"
    quality_output = output / "old-vs-robot-factored-recovery-10s.mp4"
    _write_video(ffmpeg, action_output, source_action_frames)
    _write_video(ffmpeg, quality_output, quality_frames)
    action_poster = output / "ego-source-vs-three-recovered-actions-poster.jpg"
    quality_poster = output / "old-vs-robot-factored-recovery-poster.jpg"
    if not cv2.imwrite(str(action_poster), source_action_frames[120]):
        raise RuntimeError("could not write source/action poster")
    if not cv2.imwrite(str(quality_poster), quality_frames[120]):
        raise RuntimeError("could not write old/new poster")
    manifest = {
        "schema_version": "1.0.0",
        "status": evaluation.get("status", "PARTIAL"),
        "claim_boundary": "Generated image-space action visualization; not real-robot execution.",
        "method": {
            "robot_factored_visible_geometry": "H3 action driver plus joint Wan replacement",
            "human_residual_prevention": "driver mask plus fail-closed lower-frame guard",
            "history_context": "five generated frames condition every following Wan segment",
            "degradation_aware_route": "raw Wan candidate retained; no source alpha repair or temporal blur",
        },
        "paper_method_inspirations_not_training_claims": [
            "https://arxiv.org/abs/2607.22535",
            "https://arxiv.org/abs/2606.04463",
            "https://arxiv.org/abs/2508.03694",
            "https://arxiv.org/abs/2512.13604",
        ],
        "evaluation": {
            "path": str(evaluation_path),
            "sha256": _sha256(evaluation_path),
            "accepted": evaluation.get("accepted", False),
        },
        "source": {"path": str(source_path), "sha256": _sha256(source_path), **source_info},
        "actions": records,
        "outputs": {
            path.name: {"sha256": _sha256(path)}
            for path in (action_output, quality_output, action_poster, quality_poster)
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
