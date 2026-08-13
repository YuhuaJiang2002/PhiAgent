#!/usr/bin/env python3
"""Build acceptance-gated, explicitly labeled Cosmos3 DROID I2V demos."""

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
FPS = 16
PANEL_WIDTH = 400
PANEL_HEIGHT = 225


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require_accepted_evaluation(evaluation: dict[str, Any]) -> None:
    if evaluation.get("method") != "phiagent_cosmos3_droid_multiview_i2v_strict_validation":
        raise ValueError("evaluation was not produced by the strict Cosmos3 validator")
    if evaluation.get("status") != "WORKING" or evaluation.get("accepted") is not True:
        raise ValueError("refusing to build a demo from an unaccepted Cosmos3 candidate")
    if evaluation.get("allowed_split") != "validation":
        raise ValueError("demo promotion is restricted to the validation split")


def generation_title(frame_index: int) -> tuple[str, str]:
    if frame_index == 0:
        return "REAL CONDITION (FRAME 1)", "THE MODEL INPUT IS REPEATED HERE"
    return "OUR GENERATED VIDEO", "COSMOS3 CONTINUATION / NO REAL FUTURE INPUT"


def _decode(cv2: Any, np: Any, path: Path) -> tuple[Any, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not decode video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    return np.stack(frames), fps


def _text(
    cv2: Any,
    canvas: Any,
    value: str,
    origin: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        value,
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
    panel = np.full((300, 408, 3), (10, 15, 13), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (407, 299), accent, 2)
    cv2.rectangle(panel, (2, 2), (405, 67), (18, 26, 22), -1)
    _text(cv2, panel, title, (12, 27), 0.48, accent, 1)
    _text(cv2, panel, subtitle, (12, 52), 0.29, (199, 210, 204), 1)
    resized = cv2.resize(frame, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    panel[71:296, 4:404] = resized
    return panel


def _compose(
    cv2: Any,
    np: Any,
    condition: Any,
    generated: Any,
    target: Any,
    *,
    frame_index: int,
    sample_id: str,
    task: str,
    aggregate_metrics: dict[str, float],
) -> Any:
    canvas = np.full((HEIGHT, WIDTH, 3), (5, 9, 7), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (WIDTH - 1, 122), (15, 23, 19), -1)
    _text(
        cv2,
        canvas,
        "REAL CONDITION  >  OUR GENERATED VIDEO  >  WITHHELD REAL TARGET",
        (22, 42),
        0.68,
        (89, 232, 185),
        2,
    )
    _text(cv2, canvas, task.upper()[:118], (22, 78), 0.46, (231, 237, 233), 1)
    _text(
        cv2,
        canvas,
        f"{sample_id}  |  {frame_index / FPS:05.2f}s  |  ACCEPTED ON VALIDATION",
        (22, 106),
        0.38,
        (89, 232, 185),
        1,
    )
    generated_title, generated_subtitle = generation_title(frame_index)
    panels = [
        _panel(
            cv2,
            np,
            condition,
            "REAL CONDITION",
            "2 EXTERIOR ANCHORS + WRIST VIEW + TASK TEXT",
            (100, 220, 255),
        ),
        _panel(
            cv2,
            np,
            generated,
            generated_title,
            generated_subtitle,
            (89, 232, 185),
        ),
        _panel(
            cv2,
            np,
            target,
            "WITHHELD REAL TARGET",
            "POST-GENERATION EVALUATION ONLY",
            (210, 158, 255),
        ),
    ]
    for x, panel in zip((12, 436, 860), panels):
        canvas[142:442, x : x + 408] = panel
    cv2.rectangle(canvas, (12, 468), (1267, 697), (28, 43, 35), -1)
    _text(
        cv2,
        canvas,
        "DISCLOSURE: THIS IS ANCHOR-CONDITIONED EGO-TO-THIRD-PERSON GENERATION, NOT WRIST-ONLY GENERATION.",
        (30, 514),
        0.43,
        (229, 236, 232),
        1,
    )
    _text(
        cv2,
        canvas,
        "ONLY FRAME 1 + TASK TEXT ENTER THE MODEL. REAL FRAMES 2+ ARE NEVER MODEL INPUTS.",
        (30, 552),
        0.47,
        (210, 158, 255),
        1,
    )
    _text(
        cv2,
        canvas,
        (
            f"VALIDATION: FULL SSIM {aggregate_metrics['mean_full_frame_ssim']:.3f}  |  "
            f"SUBJECT ROI {aggregate_metrics['mean_subject_roi_ssim']:.3f}  |  "
            f"EDGE F1 {aggregate_metrics['mean_subject_edge_f1']:.3f}"
        ),
        (30, 610),
        0.51,
        (89, 232, 185),
        2,
    )
    _text(
        cv2,
        canvas,
        (
            f"MOTION CORR {aggregate_metrics['motion_correlation']:.3f}  |  "
            f"MOTION RATIO {aggregate_metrics['motion_magnitude_ratio']:.3f}  |  "
            "PHIAGENT COSMOS3 DROID I2V"
        ),
        (30, 657),
        0.46,
        (202, 215, 207),
        1,
    )
    return canvas


def _encode(ffmpeg: Path, np: Any, frames: Any, output: Path) -> None:
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
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.write(np.ascontiguousarray(frames).tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed to encode {output}")


def main() -> int:
    args = _parser().parse_args()
    contract_path = args.dataset_contract.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    require_accepted_evaluation(evaluation)
    if evaluation.get("dataset_contract_sha256") != _sha256(contract_path):
        raise ValueError("evaluation does not match the supplied dataset contract")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite demo directory: {output}")
    output.mkdir(parents=True)

    import cv2
    import numpy as np

    records = {str(row["sample_id"]): row for row in contract["records"]}
    rendered = []
    reel_frames = []
    for example in evaluation["examples"]:
        sample_id = str(example["sample_id"])
        record = records[sample_id]
        if record.get("split") != "validation" or record.get("training_use") is not False:
            raise ValueError(f"demo sample is not validation-only: {sample_id}")
        condition_path = (
            contract_path.parent / record["real_composite_first_frame_condition"]
        ).resolve()
        generated_path = Path(example["generated"]).expanduser().resolve()
        target_path = Path(example["withheld_real_target"]).expanduser().resolve()
        if _sha256(condition_path) != record["real_composite_first_frame_condition_sha256"]:
            raise ValueError(f"real condition hash changed: {condition_path}")
        if _sha256(generated_path) != example["generated_sha256"]:
            raise ValueError(f"generated video hash changed: {generated_path}")
        if _sha256(target_path) != example["withheld_real_target_sha256"]:
            raise ValueError(f"withheld target hash changed: {target_path}")
        condition = cv2.imread(str(condition_path))
        if condition is None:
            raise ValueError(f"could not decode real condition: {condition_path}")
        generated, generated_fps = _decode(cv2, np, generated_path)
        target, target_fps = _decode(cv2, np, target_path)
        if abs(generated_fps - FPS) > 1e-3 or abs(target_fps - FPS) > 1e-3:
            raise ValueError("demo inputs must be 16 FPS")
        frame_count = min(len(generated), len(target))
        if frame_count < 3:
            raise ValueError("demo requires a condition plus generated continuation")
        frames = np.stack(
            [
                _compose(
                    cv2,
                    np,
                    condition,
                    generated[index],
                    target[index],
                    frame_index=index,
                    sample_id=sample_id,
                    task=str(record["raw_task_text"]),
                    aggregate_metrics=evaluation["aggregate_metrics"],
                )
                for index in range(frame_count)
            ]
        )
        video = output / f"{sample_id}-accepted-labeled-comparison.mp4"
        poster = output / f"{sample_id}-accepted-labeled-comparison-poster.jpg"
        _encode(args.ffmpeg.expanduser().resolve(), np, frames, video)
        if not cv2.imwrite(str(poster), frames[frame_count // 2]):
            raise RuntimeError(f"could not write poster: {poster}")
        reel_frames.extend(frames)
        rendered.append(
            {
                "sample_id": sample_id,
                "episode_index": int(record["episode_index"]),
                "video": str(video),
                "video_sha256": _sha256(video),
                "poster": str(poster),
                "poster_sha256": _sha256(poster),
                "frames": frame_count,
            }
        )
    reel = output / "accepted-validation-labeled-reel.mp4"
    _encode(args.ffmpeg.expanduser().resolve(), np, np.stack(reel_frames), reel)
    manifest = {
        "schema_version": "1.0.0",
        "method": "phiagent_cosmos3_droid_i2v_acceptance_gated_labeled_demo",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING",
        "accepted": True,
        "labels": {
            "real_condition": "2x2 real frame 1 plus real dataset task annotation",
            "our_generated_video": "Cosmos3 continuation frames 2 onward",
            "withheld_real_target": "post-generation validation only; never model input",
        },
        "claim": "anchor-conditioned ego-to-third-person generation; not wrist-only generation",
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "dataset_contract": str(contract_path),
        "dataset_contract_sha256": _sha256(contract_path),
        "evaluation": str(evaluation_path),
        "evaluation_sha256": _sha256(evaluation_path),
        "rendered": rendered,
        "reel": {"path": str(reel), "sha256": _sha256(reel)},
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "numpy": importlib.metadata.version("numpy"),
            "opencv-python": importlib.metadata.version("opencv-python"),
        },
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "reel": str(reel), "accepted": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
