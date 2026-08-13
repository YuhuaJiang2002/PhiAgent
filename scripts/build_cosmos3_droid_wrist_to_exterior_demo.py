#!/usr/bin/env python3
"""Build strictly accepted and explicitly labeled wrist-only Cosmos3 demos."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shlex
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_cosmos3_droid_i2v_demo import (  # noqa: E402
    FPS,
    HEIGHT,
    WIDTH,
    _decode,
    _encode,
    _panel,
    _text,
)


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
    if evaluation.get("method") != "phiagent_cosmos3_droid_wrist_only_to_exterior_strict_validation":
        raise ValueError("evaluation is not the strict wrist-only Cosmos3 validator")
    if evaluation.get("status") != "WORKING" or evaluation.get("accepted") is not True:
        raise ValueError("refusing to build a demo from an unaccepted wrist-only candidate")
    disclosure = evaluation.get("disclosure", {})
    if disclosure.get("pure_wrist_only_claim") is not True:
        raise ValueError("evaluation does not authorize the wrist-only claim")
    if disclosure.get("condition_contains_third_person_pixels") is not False:
        raise ValueError("evaluation condition contains third-person pixels")


def _compose(
    cv2: Any,
    np: Any,
    condition: Any,
    generated: Any,
    target: Any,
    *,
    sample_id: str,
    target_view: str,
    future_index: int,
    task: str,
    metrics: dict[str, float],
) -> Any:
    canvas = np.full((HEIGHT, WIDTH, 3), (5, 9, 7), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (WIDTH - 1, 122), (15, 23, 19), -1)
    _text(
        cv2,
        canvas,
        "REAL WRIST CONDITION  >  OUR GENERATED THIRD-PERSON  >  WITHHELD REAL TARGET",
        (20, 42),
        0.61,
        (89, 232, 185),
        2,
    )
    _text(cv2, canvas, task.upper()[:118], (20, 78), 0.46, (231, 237, 233), 1)
    _text(
        cv2,
        canvas,
        f"{sample_id}  |  {target_view}  |  FUTURE {future_index / FPS:05.2f}s  |  STRICT VALIDATION ACCEPTED",
        (20, 106),
        0.36,
        (89, 232, 185),
        1,
    )
    panels = [
        _panel(
            cv2,
            np,
            condition,
            "REAL CONDITION / FIRST-PERSON",
            "WRIST CAMERA FRAME 1 ONLY / MODEL INPUT",
            (100, 220, 255),
        ),
        _panel(
            cv2,
            np,
            generated,
            "OUR GENERATED VIDEO",
            f"{target_view.upper()} THIRD-PERSON / FRAMES 2+",
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
        "DISCLOSURE: THE MODEL RECEIVES ZERO THIRD-PERSON PIXELS; ONLY ONE REAL WRIST FRAME + TASK TEXT.",
        (30, 515),
        0.45,
        (229, 236, 232),
        1,
    )
    _text(
        cv2,
        canvas,
        "REAL EXTERIOR FUTURE FRAMES ARE WITHHELD UNTIL AFTER GENERATION.",
        (30, 553),
        0.50,
        (210, 158, 255),
        1,
    )
    _text(
        cv2,
        canvas,
        (
            f"VALIDATION: FULL SSIM {metrics['mean_full_frame_ssim']:.3f}  |  "
            f"SUBJECT ROI {metrics['mean_subject_roi_ssim']:.3f}  |  "
            f"EDGE F1 {metrics['mean_subject_edge_f1']:.3f}"
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
            f"MOTION CORR {metrics['motion_correlation']:.3f}  |  "
            f"MOTION RATIO {metrics['motion_magnitude_ratio']:.3f}  |  "
            "PHIAGENT COSMOS3 WRIST-ONLY I2V"
        ),
        (30, 657),
        0.46,
        (202, 215, 207),
        1,
    )
    return canvas


def main() -> int:
    args = _parser().parse_args()
    contract_path = args.dataset_contract.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    require_accepted_evaluation(evaluation)
    if evaluation.get("dataset_contract_sha256") != _sha256(contract_path):
        raise ValueError("evaluation does not match the supplied wrist-only dataset")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite demo directory: {output}")
    output.mkdir(parents=True)

    import cv2
    import numpy as np

    records = {str(row["sample_id"]): row for row in contract["records"]}
    rendered: list[dict[str, Any]] = []
    reel_frames = []
    for example in evaluation["examples"]:
        sample_id = str(example["sample_id"])
        record = records[sample_id]
        if record.get("split") != "validation" or record.get("training_use") is not False:
            raise ValueError(f"demo sample is not validation-only: {sample_id}")
        condition_path = (contract_path.parent / record["condition"]).resolve()
        generated_path = Path(example["generated"]).expanduser().resolve()
        target_path = Path(example["withheld_real_target"]).expanduser().resolve()
        if _sha256(condition_path) != record["condition_sha256"]:
            raise ValueError(f"real wrist condition hash changed: {condition_path}")
        if _sha256(generated_path) != example["generated_sha256"]:
            raise ValueError(f"generated video hash changed: {generated_path}")
        if _sha256(target_path) != example["withheld_real_target_sha256"]:
            raise ValueError(f"withheld real target hash changed: {target_path}")
        condition = cv2.imread(str(condition_path))
        if condition is None:
            raise ValueError(f"could not decode wrist condition: {condition_path}")
        generated, generated_fps = _decode(cv2, np, generated_path)
        target, target_fps = _decode(cv2, np, target_path)
        if abs(generated_fps - FPS) > 1e-3 or abs(target_fps - FPS) > 1e-3:
            raise ValueError("demo inputs must be 16 FPS")
        frame_count = min(len(generated), len(target))
        if frame_count < 4:
            raise ValueError("demo requires at least three third-person future frames")
        frames = np.stack(
            [
                _compose(
                    cv2,
                    np,
                    condition,
                    generated[index],
                    target[index],
                    sample_id=sample_id,
                    target_view=str(record["target_view"]),
                    future_index=index,
                    task=str(record.get("raw_task_text", contract.get("claim_scope", "robot manipulation"))),
                    metrics=evaluation["aggregate_metrics"],
                )
                for index in range(1, frame_count)
            ]
        )
        video = output / f"{sample_id}-accepted-wrist-only-comparison.mp4"
        poster = output / f"{sample_id}-accepted-wrist-only-comparison-poster.jpg"
        _encode(args.ffmpeg.expanduser().resolve(), np, frames, video)
        if not cv2.imwrite(str(poster), frames[len(frames) // 2]):
            raise RuntimeError(f"could not write poster: {poster}")
        reel_frames.extend(frames)
        rendered.append(
            {
                "sample_id": sample_id,
                "episode_index": int(record["episode_index"]),
                "target_view": record["target_view"],
                "video": str(video),
                "video_sha256": _sha256(video),
                "poster": str(poster),
                "poster_sha256": _sha256(poster),
                "generated_future_frames": len(frames),
            }
        )
    reel = output / "accepted-wrist-only-to-third-person-labeled-reel.mp4"
    _encode(args.ffmpeg.expanduser().resolve(), np, np.stack(reel_frames), reel)
    manifest = {
        "schema_version": "1.0.0",
        "method": "phiagent_cosmos3_droid_wrist_only_acceptance_gated_labeled_demo",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING",
        "accepted": True,
        "labels": {
            "real_condition": "one real first-person wrist-camera frame only",
            "our_generated_video": "Cosmos3 third-person continuation frames 2 onward",
            "withheld_real_target": "synchronized real exterior future; post-generation evaluation only",
        },
        "claim": "true wrist-only first-person to named third-person video generation",
        "condition_contains_third_person_pixels": False,
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
