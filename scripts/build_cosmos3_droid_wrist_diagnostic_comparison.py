#!/usr/bin/env python3
"""Build labeled diagnostic comparisons for accepted or rejected wrist I2V runs.

Unlike the public demo builder, this diagnostic artifact may render a rejected
evaluation.  Rejected outputs receive a permanent red NOT ACCEPTED banner and
are never described as a demo or promoted result.
"""

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
    parser.add_argument(
        "--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg")
    )
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


def validate_diagnostic_evaluation(evaluation: dict[str, Any]) -> tuple[str, bool]:
    if (
        evaluation.get("method")
        != "phiagent_cosmos3_droid_wrist_only_to_exterior_strict_validation"
    ):
        raise ValueError("evaluation is not the strict wrist-only Cosmos3 validator")
    accepted = evaluation.get("accepted") is True
    expected_status = "WORKING" if accepted else "PARTIAL"
    if evaluation.get("status") != expected_status:
        raise ValueError("evaluation status and accepted flag disagree")
    disclosure = evaluation.get("disclosure", {})
    if disclosure.get("pure_wrist_only_claim") is not True:
        raise ValueError("evaluation does not prove a pure wrist-only condition")
    if disclosure.get("condition_contains_third_person_pixels") is not False:
        raise ValueError("evaluation condition contains third-person pixels")
    return expected_status, accepted


def _compose(
    cv2: Any,
    np: Any,
    condition: Any,
    generated: Any,
    target: Any,
    *,
    example: dict[str, Any],
    frame_index: int,
    accepted: bool,
) -> Any:
    accent = (89, 232, 185) if accepted else (72, 72, 245)
    status = "ACCEPTED" if accepted else "NOT ACCEPTED / DIAGNOSTIC ONLY"
    canvas = np.full((HEIGHT, WIDTH, 3), (5, 9, 7), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (WIDTH - 1, 125), (15, 23, 19), -1)
    _text(
        cv2,
        canvas,
        "REAL CONDITION  >  OUR GENERATED VIDEO  >  WITHHELD REAL TARGET",
        (20, 38),
        0.59,
        (231, 237, 233),
        2,
    )
    _text(cv2, canvas, f"STRICT VALIDATION: {status}", (20, 76), 0.62, accent, 2)
    _text(
        cv2,
        canvas,
        f"{example['sample_id']} | {example['target_view']} | {frame_index / FPS:05.2f}s",
        (20, 107),
        0.39,
        (202, 215, 207),
        1,
    )
    panels = [
        _panel(
            cv2,
            np,
            condition,
            "REAL CONDITION / FIRST-PERSON",
            "ONE WRIST FRAME / MODEL INPUT",
            (100, 220, 255),
        ),
        _panel(
            cv2,
            np,
            generated,
            "OUR GENERATED VIDEO",
            f"{str(example['target_view']).upper()} / FRAMES 2+",
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
    metrics = example["metrics"]
    failed = [name for name, passed in example["gates"].items() if not passed]
    cv2.rectangle(canvas, (12, 468), (1267, 697), (28, 43, 35), -1)
    _text(
        cv2,
        canvas,
        "DISCLOSURE: MODEL INPUT = ONE REAL WRIST FRAME + TASK TEXT; ZERO THIRD-PERSON PIXELS.",
        (30, 510),
        0.43,
        (229, 236, 232),
        1,
    )
    _text(
        cv2,
        canvas,
        (
            f"FULL SSIM {metrics['mean_full_frame_ssim']:.3f} | "
            f"SUBJECT ROI {metrics['mean_subject_roi_ssim']:.3f} | "
            f"EDGE F1 {metrics['mean_subject_edge_f1']:.3f} | "
            f"MOTION CORR {metrics['motion_correlation']:.3f}"
        ),
        (30, 558),
        0.47,
        accent,
        2,
    )
    _text(
        cv2,
        canvas,
        (
            f"COND FRAME SSIM {example['condition_first_frame_ssim']:.3f} | "
            f"MOTION RATIO {metrics['motion_magnitude_ratio']:.3f} | "
            f"STATIC GAIN {metrics['static_anchor_ssim_gain']:.3f}"
        ),
        (30, 606),
        0.44,
        (202, 215, 207),
        1,
    )
    failed_text = ", ".join(failed[:5])
    if len(failed) > 5:
        failed_text += f" +{len(failed) - 5} MORE"
    _text(
        cv2,
        canvas,
        f"FAILED GATES: {failed_text.upper() if failed_text else 'NONE'}",
        (30, 653),
        0.38,
        accent,
        1,
    )
    return canvas


def main() -> int:
    args = _parser().parse_args()
    contract_path = args.dataset_contract.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    status, accepted = validate_diagnostic_evaluation(evaluation)
    if evaluation.get("dataset_contract_sha256") != _sha256(contract_path):
        raise ValueError("evaluation does not match the supplied wrist-only dataset")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic directory: {output}")
    output.mkdir(parents=True)

    import cv2
    import numpy as np

    records = {str(row["sample_id"]): row for row in contract["records"]}
    rendered: list[dict[str, Any]] = []
    reel_frames: list[Any] = []
    for example in evaluation["examples"]:
        sample_id = str(example["sample_id"])
        record = records[sample_id]
        if record.get("split") != "validation" or record.get("training_use") is not False:
            raise ValueError(f"diagnostic sample is not validation-only: {sample_id}")
        condition_path = Path(example["condition"]).expanduser().resolve()
        generated_path = Path(example["generated"]).expanduser().resolve()
        target_path = Path(example["withheld_real_target"]).expanduser().resolve()
        for path, expected, label in (
            (condition_path, example["condition_sha256"], "condition"),
            (generated_path, example["generated_sha256"], "generated"),
            (target_path, example["withheld_real_target_sha256"], "target"),
        ):
            if _sha256(path) != expected:
                raise ValueError(f"{label} hash changed: {path}")
        condition = cv2.imread(str(condition_path))
        if condition is None:
            raise ValueError(f"could not decode condition: {condition_path}")
        generated, generated_fps = _decode(cv2, np, generated_path)
        target, target_fps = _decode(cv2, np, target_path)
        if abs(generated_fps - FPS) > 1e-3 or abs(target_fps - FPS) > 1e-3:
            raise ValueError("diagnostic inputs must be 16 FPS")
        frame_count = min(len(generated), len(target))
        frames = np.stack(
            [
                _compose(
                    cv2,
                    np,
                    condition,
                    generated[index],
                    target[index],
                    example=example,
                    frame_index=index,
                    accepted=accepted,
                )
                for index in range(1, frame_count)
            ]
        )
        video = output / f"{sample_id}-diagnostic-comparison.mp4"
        poster = output / f"{sample_id}-diagnostic-comparison-poster.jpg"
        _encode(args.ffmpeg.expanduser().resolve(), np, frames, video)
        if not cv2.imwrite(str(poster), frames[len(frames) // 2]):
            raise RuntimeError(f"could not write poster: {poster}")
        reel_frames.extend(frames)
        rendered.append(
            {
                "sample_id": sample_id,
                "video": str(video),
                "video_sha256": _sha256(video),
                "poster": str(poster),
                "poster_sha256": _sha256(poster),
                "frames": len(frames),
                "accepted": bool(example["accepted"]),
            }
        )
    reel = output / "wrist-to-third-person-strict-diagnostic-reel.mp4"
    _encode(args.ffmpeg.expanduser().resolve(), np, np.stack(reel_frames), reel)
    manifest = {
        "schema_version": "1.0.0",
        "method": "phiagent_cosmos3_wrist_to_third_person_labeled_diagnostic_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "accepted": accepted,
        "artifact_class": "diagnostic_comparison_not_public_demo",
        "labels": {
            "real_condition": "REAL CONDITION / FIRST-PERSON",
            "generated": "OUR GENERATED VIDEO",
            "target": "WITHHELD REAL TARGET / EVALUATION ONLY",
        },
        "dataset_contract": str(contract_path),
        "dataset_contract_sha256": _sha256(contract_path),
        "evaluation": str(evaluation_path),
        "evaluation_sha256": _sha256(evaluation_path),
        "rendered": rendered,
        "reel": {"path": str(reel), "sha256": _sha256(reel)},
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "numpy": importlib.metadata.version("numpy"),
            "opencv-python": importlib.metadata.version("opencv-python"),
        },
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "reel": str(reel), "status": status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
