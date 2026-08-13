#!/usr/bin/env python3
"""Run bounded repairs and build a matched MiniMax-H3 action comparison."""

from __future__ import annotations

import argparse
import hashlib
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
        raise RuntimeError(f"decoded too few frames from {path}: {len(frames)}")
    info["frames"] = len(frames)
    return frames, info


def _pairwise_distinctness(np: Any, first: list[Any], second: list[Any]) -> dict[str, float]:
    if len(first) != len(second):
        raise ValueError("variant frame counts differ")
    per_frame = []
    foreground = []
    for left, right in zip(first, second):
        difference = np.abs(left.astype(np.float32) - right.astype(np.float32))
        per_frame.append(float(difference.mean()))
        active = difference.max(axis=2) >= 12.0
        foreground.append(float(difference[active].mean()) if active.any() else 0.0)
    array = np.asarray(per_frame, dtype=np.float64)
    return {
        "full_frame_mean_absolute_difference": float(array.mean()),
        "full_frame_peak_absolute_difference": float(array.max()),
        "active_pixel_mean_absolute_difference": float(np.mean(foreground)),
        "fraction_of_frames_above_2_mad": float(np.mean(array >= 2.0)),
    }


def _fit_text(cv2: Any, text: str, width: int, max_chars: int = 54) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:2]


def _compose_frames(
    cv2: Any,
    np: Any,
    sources: list[list[Any]],
    labels: list[tuple[str, str]],
    tile_width: int = 416,
    tile_height: int = 240,
    header_height: int = 54,
) -> list[Any]:
    frame_count = min(len(frames) for frames in sources)
    output = []
    for frame_index in range(frame_count):
        tiles = []
        for frames, (title, instruction) in zip(sources, labels):
            video = cv2.resize(
                frames[frame_index], (tile_width, tile_height), interpolation=cv2.INTER_AREA
            )
            header = np.full((header_height, tile_width, 3), 18, dtype=np.uint8)
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
            detail = _fit_text(cv2, instruction, tile_width)[0] if instruction else ""
            cv2.putText(
                header,
                detail,
                (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (220, 224, 228),
                1,
                cv2.LINE_AA,
            )
            tiles.append(np.vstack([header, video]))
        output.append(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]))
    return output


def _write_video(ffmpeg: Path, frames: list[Any], output: Path, fps: float) -> None:
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
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "16",
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--anchor-mask", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, default=60)
    parser.add_argument("--source-start-frame", type=int, default=216)
    parser.add_argument("--full-source-frames", type=int, default=660)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    experiment = args.experiment_dir.expanduser().resolve()
    metadata_path = experiment / "metadata.json"
    anchor_mask = args.anchor_mask.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    for path in (metadata_path, anchor_mask, ffmpeg):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required input does not exist or is empty: {path}")
    comparison_dir = experiment / "comparison"
    manifest_path = comparison_dir / "action-comparison.json"
    if manifest_path.exists():
        raise FileExistsError(f"comparison already exists: {manifest_path}")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "completed":
        raise ValueError("H3 action experiment is not completed")
    if len(metadata.get("actions", [])) != 3:
        raise ValueError("the 2x2 comparison requires exactly three action variants")
    source = experiment / "input" / "source.mp4"
    robot_reference = next((experiment / "input").glob("robot-reference.*"), None)
    if not source.is_file() or robot_reference is None:
        raise ValueError("copied experiment source or robot reference is missing")

    evaluations = []
    final_paths = []
    for action in metadata["actions"]:
        label = action["label"]
        raw = experiment / "variants" / label / "raw-h3-nf4.mp4"
        evaluation_dir = experiment / "variants" / label / "agent-evaluation"
        command = [
            sys.executable,
            str(project_root / "scripts" / "evaluate_minimax_h3_flower_validation.py"),
            "--source",
            str(source),
            "--raw-h3",
            str(raw),
            "--robot-reference",
            str(robot_reference),
            "--anchor-mask",
            str(anchor_mask),
            "--backend-metadata",
            str(metadata_path),
            "--candidate-label",
            f"minimax_h3_nf4_action_{label}",
            "--output-dir",
            str(evaluation_dir),
            "--anchor-frame",
            str(args.anchor_frame),
            "--source-start-frame",
            str(args.source_start_frame),
            "--full-source-frames",
            str(args.full_source_frames),
            "--ffmpeg",
            str(ffmpeg),
            "--action-override",
        ]
        action_control = experiment / "input" / "action-controls" / f"{label}.mp4"
        if action_control.is_file():
            command.extend(("--motion-reference", str(action_control)))
        completed = subprocess.run(command, check=False)
        if completed.returncode not in (0, 2):
            raise RuntimeError(f"agent evaluation failed for {label}: exit {completed.returncode}")
        evolution_path = evaluation_dir / "evolution.json"
        evolution = json.loads(evolution_path.read_text())
        final = evaluation_dir / "final-background-locked.mp4"
        final_paths.append(final)
        evaluations.append(
            {
                "label": label,
                "instruction": action["instruction"],
                "status": evolution["status"],
                "honest_status": evolution["honest_status"],
                "best_round": evolution["best_round"],
                "best_repair": evolution["best_repair"],
                "best_scorecard": evolution["best_scorecard"],
                "best_source_face_replacement": evolution["best_source_face_replacement"],
                "evolution": str(evolution_path),
                "final": str(final),
                "final_sha256": _sha256(final),
            }
        )

    import cv2
    import numpy as np

    source_frames, source_info = _decode(cv2, source)
    variants = []
    variant_info = []
    for path in final_paths:
        frames, info = _decode(cv2, path)
        if info != source_info:
            raise RuntimeError(f"comparison alignment mismatch: {source_info} vs {info}")
        variants.append(frames)
        variant_info.append(info)
    pairwise = []
    for left_index in range(len(variants)):
        for right_index in range(left_index + 1, len(variants)):
            pairwise.append(
                {
                    "left": evaluations[left_index]["label"],
                    "right": evaluations[right_index]["label"],
                    **_pairwise_distinctness(
                        np, variants[left_index], variants[right_index]
                    ),
                }
            )

    comparison_frames = _compose_frames(
        cv2,
        np,
        [source_frames, *variants],
        [
            ("REAL SOURCE / SCENE REFERENCE", "Original camera, workspace and objects"),
            *[
                (f"ACTION {index + 1} / {evaluation['label'].upper()}", evaluation["instruction"])
                for index, evaluation in enumerate(evaluations)
            ],
        ],
    )
    comparison_video = comparison_dir / "real-source-vs-three-language-actions.mp4"
    _write_video(ffmpeg, comparison_frames, comparison_video, float(source_info["fps"]))
    poster = comparison_dir / "poster.jpg"
    cv2.imwrite(str(poster), comparison_frames[len(comparison_frames) // 2])
    storyboard = comparison_dir / "storyboard.jpg"
    samples = [
        comparison_frames[round(index * (len(comparison_frames) - 1) / 3)]
        for index in range(4)
    ]
    cv2.imwrite(str(storyboard), np.vstack(samples))

    distinctness_floor = min(
        item["full_frame_mean_absolute_difference"] for item in pairwise
    )
    all_outputs_distinct = distinctness_floor >= 2.0
    visual_repairs_safe = all(
        item["best_scorecard"]["background_lock"] >= 0.99
        and item["best_scorecard"]["subject_replacement"] >= 0.80
        and item["best_source_face_replacement"] >= 0.95
        for item in evaluations
    )
    manifest = {
        "schema_version": "1.0.0",
        "method": "matched_minimax_h3_language_actions_plus_five_round_agent_repairs",
        "status": "completed",
        "honest_status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "inputs": {
            "experiment_metadata": {"path": str(metadata_path), "sha256": _sha256(metadata_path)},
            "source": {"path": str(source), "sha256": _sha256(source)},
            "anchor_mask": {"path": str(anchor_mask), "sha256": _sha256(anchor_mask)},
        },
        "coordinate_frames": {
            "source": "camera:source_clip_pixels",
            "variants": "camera:MiniMax-H3_output_pixels aligned by frame index",
            "comparison": "display:2x2_grid; each tile is an aligned camera frame",
        },
        "source_info": source_info,
        "evaluations": evaluations,
        "pairwise_action_distinctness": pairwise,
        "acceptance": {
            "three_h3_outputs_present": len(variants) == 3,
            "outputs_frame_aligned": True,
            "all_outputs_numerically_distinct": all_outputs_distinct,
            "pairwise_full_frame_mad_floor": distinctness_floor,
            "background_and_human_replacement_safety_proxies_pass": visual_repairs_safe,
            "semantic_action_adherence_human_reviewed": False,
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
            "PARTIAL until a human verifies that each video semantically follows its written action.",
            "Pairwise pixel difference proves outputs differ, not that a commanded action is correct.",
            "The five-round agent workflow optimizes scene lock, object safety, robot identity and temporal consistency; its legacy source-motion metric is diagnostic only because these actions intentionally override source motion.",
            "Tracked-mask repair can suppress commanded motion that leaves the original source-person support.",
            "H3 uses third-party NF4 weights rather than official BF16 weights.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "comparison": str(comparison_video),
                "honest_status": manifest["honest_status"],
                "distinctness_floor": distinctness_floor,
                "visual_repairs_safe": visual_repairs_safe,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
