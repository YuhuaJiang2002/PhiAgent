#!/usr/bin/env python3
"""Package labeled DROID real-condition/generated/real-target comparisons."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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


WIDTH = 1280
HEIGHT = 720
FPS = 8
FRAMES = 17
PANEL_WIDTH = 292
PANEL_HEIGHT = 167


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--inference-metadata", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _decode(cv2: Any, np: Any, path: Path) -> Any:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != FRAMES:
        raise ValueError(f"expected {FRAMES} frames in {path}, got {len(frames)}")
    return np.stack(frames)


def _text(
    cv2: Any,
    canvas: Any,
    text: str,
    origin: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def _panel(
    cv2: Any,
    np: Any,
    frame: Any,
    title: str,
    subtitle: str,
    accent: tuple[int, int, int],
) -> Any:
    panel = np.full((235, 300, 3), (12, 17, 15), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (299, 234), accent, 2)
    cv2.rectangle(panel, (2, 2), (297, 62), (19, 26, 22), -1)
    _text(cv2, panel, title, (12, 25), 0.45, accent, 1)
    _text(cv2, panel, subtitle, (12, 49), 0.31, (190, 202, 195), 1)
    resized = cv2.resize(frame, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_CUBIC)
    panel[66 : 66 + PANEL_HEIGHT, 4 : 4 + PANEL_WIDTH] = resized
    return panel


def _compose(
    cv2: Any,
    np: Any,
    condition: Any,
    anchor: Any,
    generated: Any,
    target: Any,
    *,
    task: str,
    episode: int,
    frame_index: int,
    metrics: dict[str, float],
    accepted: bool,
) -> Any:
    canvas = np.full((HEIGHT, WIDTH, 3), (6, 10, 8), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (WIDTH - 1, 112), (15, 22, 18), -1)
    accent = (78, 234, 181) if accepted else (75, 178, 255)
    _text(
        cv2,
        canvas,
        "REAL CONDITIONS  >  OUR GENERATED VIDEO  >  REAL HELD-OUT TARGET",
        (24, 42),
        0.72,
        accent,
        2,
    )
    _text(cv2, canvas, task.upper(), (24, 77), 0.53, (230, 235, 232), 1)
    status = "ACCEPTED HELD-OUT GENERATION" if accepted else "CANDIDATE / NOT ACCEPTED"
    _text(
        cv2,
        canvas,
        f"EP {episode:02d}  |  {frame_index / FPS:04.2f}s  |  {status}",
        (24, 101),
        0.37,
        accent,
        1,
    )
    generated_title = "OUR GENERATED VIDEO" if accepted else "OUR GENERATED CANDIDATE"
    panels = [
        _panel(
            cv2,
            np,
            condition,
            "REAL CONDITION A",
            "FIRST-PERSON WRIST VIDEO",
            (105, 224, 255),
        ),
        _panel(
            cv2,
            np,
            anchor,
            "REAL CONDITION B",
            "ONE TARGET-VIEW ANCHOR FRAME",
            (105, 224, 255),
        ),
        _panel(
            cv2,
            np,
            generated,
            generated_title,
            "PHIAGENT VIEW LORA OUTPUT",
            accent,
        ),
        _panel(
            cv2,
            np,
            target,
            "REAL HELD-OUT TARGET",
            "EVALUATION ONLY / NOT MODEL INPUT",
            (207, 155, 255),
        ),
    ]
    for x, panel in zip((20, 330, 640, 950), panels):
        canvas[150:385, x : x + 300] = panel
    cv2.rectangle(canvas, (20, 420), (1250, 680), (35, 51, 43), -1)
    _text(
        cv2,
        canvas,
        "DISCLOSURE: MODEL INPUT = WRIST VIDEO + ONE REAL TARGET-VIEW ANCHOR + TASK TEXT",
        (42, 465),
        0.53,
        (222, 232, 226),
        1,
    )
    _text(
        cv2,
        canvas,
        "THE SYNCHRONIZED REAL TARGET VIDEO IS WITHHELD UNTIL POST-GENERATION EVALUATION",
        (42, 504),
        0.48,
        (207, 155, 255),
        1,
    )
    _text(
        cv2,
        canvas,
        (
            f"SSIM {metrics['mean_full_frame_ssim']:.3f}  |  SUBJECT ROI "
            f"{metrics['mean_subject_roi_ssim']:.3f}  |  EDGE F1 "
            f"{metrics['mean_subject_edge_f1']:.3f}"
        ),
        (42, 562),
        0.57,
        accent,
        2,
    )
    _text(
        cv2,
        canvas,
        (
            f"MOTION CORR {metrics['motion_correlation']:.3f}  |  STATIC-ANCHOR GAIN "
            f"{metrics['static_anchor_ssim_gain']:+.3f}"
        ),
        (42, 608),
        0.53,
        (218, 228, 222),
        1,
    )
    _text(
        cv2,
        canvas,
        "PHIAGENT DROID VIEW LORA  |  TARGET-A SPECIALIST  |  EPISODE-LEVEL HELD-OUT",
        (42, 650),
        0.42,
        (165, 180, 171),
        1,
    )
    return canvas


def _encode(ffmpeg: Path, np: Any, frames: Any, path: Path) -> None:
    process = subprocess.Popen(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.write(np.ascontiguousarray(frames).tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed to encode {path}")


def main() -> int:
    args = _parser().parse_args()
    contract_path = args.dataset_contract.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    evaluation = json.loads(evaluation_path.read_text())
    contract = json.loads(contract_path.read_text())
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite demo: {output}")
    output.mkdir(parents=True)

    import cv2
    import numpy as np

    evaluation_by_key = {
        (int(item["episode_index"]), str(item["view"])): item
        for item in evaluation["examples"]
    }
    rendered = []
    reel_frames = []
    for raw_metadata in args.inference_metadata:
        metadata_path = raw_metadata.expanduser().resolve()
        metadata = json.loads(metadata_path.read_text())
        episode = int(metadata["episode_index"])
        view = str(metadata["view"])
        item = evaluation_by_key[(episode, view)]
        record = next(
            row for row in contract["holdout_records"] if row["episode_index"] == episode
        )
        target_spec = record["targets"][view]
        condition_path = (contract_path.parent / record["condition"]["path"]).resolve()
        anchor_path = (contract_path.parent / target_spec["anchor"]).resolve()
        target_path = (contract_path.parent / target_spec["target"]).resolve()
        generated_path = metadata_path.parent / "our-generated-video.mp4"
        for path in (condition_path, anchor_path, target_path, generated_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"demo input is missing: {path}")
        condition = _decode(cv2, np, condition_path)
        generated = _decode(cv2, np, generated_path)
        target = _decode(cv2, np, target_path)
        anchor = cv2.imread(str(anchor_path))
        if anchor is None:
            raise ValueError(f"could not decode anchor: {anchor_path}")
        accepted = bool(item["accepted"])
        frames = np.stack(
            [
                _compose(
                    cv2,
                    np,
                    condition[index],
                    anchor,
                    generated[index],
                    target[index],
                    task=record["task"],
                    episode=episode,
                    frame_index=index,
                    metrics=item["metrics"],
                    accepted=accepted,
                )
                for index in range(FRAMES)
            ]
        )
        video = output / f"ep{episode:03d}-{view}-generated-comparison.mp4"
        poster = output / f"ep{episode:03d}-{view}-generated-comparison-poster.jpg"
        _encode(ffmpeg, np, frames, video)
        if not cv2.imwrite(str(poster), frames[FRAMES // 2]):
            raise RuntimeError(f"could not write {poster}")
        reel_frames.extend(frames)
        rendered.append(
            {
                "episode_index": episode,
                "view": view,
                "accepted": accepted,
                "video": str(video),
                "video_sha256": _sha256(video),
                "poster": str(poster),
                "poster_sha256": _sha256(poster),
            }
        )
    reel = output / "heldout-generated-comparison-reel.mp4"
    _encode(ffmpeg, np, np.stack(reel_frames), reel)
    manifest = {
        "schema_version": "1.0.0",
        "method": "phiagent_droid_view_generated_labeled_demo",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": evaluation["status"],
        "accepted": evaluation["accepted"],
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "annotation": {
            "real_condition_a": "measured wrist video",
            "real_condition_b": "one measured target-view anchor frame",
            "our_generated_video": "PhiAgent DROID View LoRA output",
            "real_heldout_target": "measured synchronized exterior video used only after generation",
        },
        "rendered": rendered,
        "reel": {"path": str(reel), "sha256": _sha256(reel)},
        "dataset_contract": str(contract_path),
        "dataset_contract_sha256": _sha256(contract_path),
        "evaluation": str(evaluation_path),
        "evaluation_sha256": _sha256(evaluation_path),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "numpy": importlib.metadata.version("numpy"),
            "opencv-python": importlib.metadata.version("opencv-python"),
        },
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "accepted": manifest["accepted"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
