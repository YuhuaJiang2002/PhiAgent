#!/usr/bin/env python3
"""Build a labelled real-source/raw/trained-policy flower comparison video."""

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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
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
    info["frames"] = len(frames)
    if len(frames) < 3 or min(info["width"], info["height"], info["fps"]) <= 0:
        raise RuntimeError(f"invalid video stream: {path}: {info}")
    return frames, info


def _writer(ffmpeg: Path, output: Path, width: int, height: int, fps: float) -> Any:
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
            f"{fps:.8f}",
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evolution", type=Path, required=True)
    parser.add_argument(
        "--regressed-evolution",
        type=Path,
        help="Optional prior aggregate-utility run shown beside the non-regressing result.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    evolution_path = args.evolution.expanduser().resolve()
    regressed_evolution_path = (
        None
        if args.regressed_evolution is None
        else args.regressed_evolution.expanduser().resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"demo output already exists: {output_dir}")
    required = [evolution_path, ffmpeg]
    if regressed_evolution_path is not None:
        required.append(regressed_evolution_path)
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required input does not exist or is empty: {path}")
    evolution = json.loads(evolution_path.read_text())
    if not isinstance(evolution, dict) or len(evolution.get("rounds", [])) != 2:
        raise ValueError("demo requires a raw-plus-learned-policy two-round evaluation")
    policy = evolution.get("learned_repair_policy")
    if not isinstance(policy, dict):
        raise ValueError("evaluation does not contain learned repair-policy evidence")
    raw_record, selected_record = evolution["rounds"]
    inputs = evolution.get("inputs", {})
    source = Path(inputs["source"]["path"])
    raw = Path(raw_record["output"])
    selected = Path(selected_record["output"])
    regressed_record = None
    regressed = None
    if regressed_evolution_path is not None:
        regressed_evolution = json.loads(regressed_evolution_path.read_text())
        if not isinstance(regressed_evolution, dict):
            raise ValueError("regressed evaluation must contain one JSON object")
        regressed_rounds = regressed_evolution.get("rounds", [])
        if not isinstance(regressed_rounds, list) or len(regressed_rounds) < 2:
            raise ValueError("regressed evaluation requires a selected repair round")
        regressed_record = regressed_rounds[-1]
        regressed = Path(regressed_record["output"])
    video_inputs = [source, raw]
    if regressed is not None:
        video_inputs.append(regressed)
    video_inputs.append(selected)
    for path in video_inputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"demo video input is missing: {path}")

    import cv2
    import numpy as np

    decoded = [_decode(cv2, path) for path in video_inputs]
    infos = [item[1] for item in decoded]
    if not all(info == infos[0] for info in infos[1:]):
        raise RuntimeError(f"comparison streams are not aligned: {infos}")
    frames = [item[0] for item in decoded]
    output_dir.mkdir(parents=True)
    tile_width = 320 if regressed is not None else 416
    tile_height, header_height = round(tile_width * 480 / 832), 76
    if tile_height % 2:
        tile_height += 1
    titles = ["REAL SOURCE / REFERENCE", "RAW WORLD MODEL"]
    raw_score = raw_record["scorecard"]
    selected_score = selected_record["scorecard"]
    details = [
        "Scene reference; not robot execution",
        f"utility {raw_score['mean_score']:.3f} | bg {raw_score['background_lock']:.3f} | flower {raw_score['object_lock']:.3f}",
    ]
    if regressed_record is not None:
        regressed_score = regressed_record["scorecard"]
        titles.append("OLD / AGGREGATE WINNER")
        details.append(
            f"motion {regressed_score['motion_preservation']:.3f} | EPL {regressed_score['epl_minimum']:.3f}"
        )
        titles.append("NEW / NON-REGRESSION GUARD")
        details.append(
            f"motion {selected_score['motion_preservation']:.3f} | EPL {selected_score['epl_minimum']:.3f}"
        )
    else:
        titles.append("AFTER / TRAINED FLOWER POLICY")
        details.append(
            f"utility {selected_score['mean_score']:.3f} | bg {selected_score['background_lock']:.3f} | flower {selected_score['object_lock']:.3f}"
        )
    output = output_dir / (
        "heldout-inspect-nonregression-guard.mp4"
        if regressed is not None
        else "heldout-inspect-before-after.mp4"
    )
    process = _writer(
        ffmpeg,
        output,
        tile_width * len(frames),
        tile_height + header_height,
        float(infos[0]["fps"]),
    )
    assert process.stdin is not None
    composed = []
    for index in range(int(infos[0]["frames"])):
        tiles = []
        for stream, title, detail in zip(frames, titles, details):
            image = cv2.resize(
                stream[index], (tile_width, tile_height), interpolation=cv2.INTER_AREA
            )
            header = np.full((header_height, tile_width, 3), 20, dtype=np.uint8)
            cv2.putText(
                header,
                title,
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (93, 238, 170),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                header,
                detail,
                (12, 53),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (220, 224, 228),
                1,
                cv2.LINE_AA,
            )
            tiles.append(np.vstack((header, image)))
        frame = np.hstack(tiles)
        process.stdin.write(frame.tobytes())
        composed.append(frame)
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed to encode {output}")
    poster = output_dir / "poster.jpg"
    cv2.imwrite(str(poster), composed[len(composed) // 2])
    storyboard = output_dir / "storyboard.jpg"
    sample_indices = [round(index * (len(composed) - 1) / 5) for index in range(6)]
    cv2.imwrite(str(storyboard), np.vstack([composed[index] for index in sample_indices]))
    if regressed_record is not None:
        regressed_score = regressed_record["scorecard"]
        non_regression = selected_record.get("non_regression", {})
        improved = (
            bool(non_regression.get("passed"))
            and float(selected_score["motion_preservation"])
            > float(regressed_score["motion_preservation"])
            and float(selected_score["epl_minimum"]) > float(regressed_score["epl_minimum"])
            and float(selected_score["background_lock"]) >= 0.999
        )
    else:
        improved = (
            float(selected_score["mean_score"]) > float(raw_score["mean_score"])
            and float(selected_score["background_lock"]) >= 0.999
            and float(selected_score["object_lock"]) >= 0.98
        )
    manifest = {
        "schema_version": "1.0.0",
        "method": "held_action_raw_vs_trained_flower_repair_policy_demo",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if improved else "rejected",
        "honest_status": "PARTIAL",
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "input_evaluation": {
            "path": str(evolution_path),
            "sha256": _sha256(evolution_path),
        },
        "policy": policy,
        "coordinate_frame": "display-only stack of aligned camera:H3_output_pixels videos",
        "before": {
            "path": str(raw),
            "sha256": _sha256(raw),
            "scorecard": raw_score,
        },
        "regressed_prior": (
            None
            if regressed is None or regressed_record is None
            else {
                "evaluation": str(regressed_evolution_path),
                "path": str(regressed),
                "sha256": _sha256(regressed),
                "scorecard": regressed_record["scorecard"],
            }
        ),
        "after": {
            "path": str(selected),
            "sha256": _sha256(selected),
            "scorecard": selected_score,
        },
        "improvement": {
            "mean_score": float(selected_score["mean_score"]) - float(raw_score["mean_score"]),
            "background_lock": float(selected_score["background_lock"])
            - float(raw_score["background_lock"]),
            "object_lock": float(selected_score["object_lock"]) - float(raw_score["object_lock"]),
            "candidate_evaluations_saved": policy["candidate_evaluations_saved"],
            "motion_recovery_over_regressed_prior": (
                None
                if regressed_record is None
                else float(selected_score["motion_preservation"])
                - float(regressed_record["scorecard"]["motion_preservation"])
            ),
            "epl_recovery_over_regressed_prior": (
                None
                if regressed_record is None
                else float(selected_score["epl_minimum"])
                - float(regressed_record["scorecard"]["epl_minimum"])
            ),
        },
        "outputs": {
            "video": str(output),
            "video_sha256": _sha256(output),
            "poster": str(poster),
            "poster_sha256": _sha256(poster),
            "storyboard": str(storyboard),
            "storyboard_sha256": _sha256(storyboard),
        },
        "limitations": [
            "The held-out action shares its real source scene and repair recipes with training campaigns.",
            "The after video is a learned selection among deterministic post-processing recipes, not a fine-tuned video-model generation.",
            "The non-regression guard can reject flower restoration when that repair suppresses commanded motion; preservation and action gates must both pass before task acceptance.",
            "Proxy gains do not establish hand-stem contact, 3-D control, or real-robot execution.",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"demo": str(output), "improved": improved}, indent=2))
    return 0 if improved else 2


if __name__ == "__main__":
    raise SystemExit(main())
