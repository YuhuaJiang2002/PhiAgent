#!/usr/bin/env python3
"""Strict per-view evaluation for generated Cosmos DROID composites."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_droid_view_lora import (  # noqa: E402
    GATES,
    _decode,
    _gate,
    _ssim_map,
    evaluate_arrays,
)


CONDITION_FIRST_FRAME_SSIM_MIN = 0.90
VIEWS = {
    "exterior_1": (0, 0, 384, 216),
    "exterior_2": (384, 0, 384, 216),
    "wrist": (0, 216, 384, 216),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--allowed-split",
        choices=("legacy_dev", "validation", "final_holdout"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _record(contract: dict[str, Any], sample_id: str, split: str) -> dict[str, Any]:
    rows = [row for row in contract.get("records", []) if row.get("sample_id") == sample_id]
    if len(rows) != 1:
        raise ValueError(f"sample is not unique in contract: {sample_id}")
    row = rows[0]
    if row.get("split") != split or row.get("training_use") is not False:
        raise ValueError(f"sample {sample_id} is not a non-training {split} record")
    return row


def _crop(frames: Any, box: tuple[int, int, int, int]) -> Any:
    x, y, width, height = box
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("video must have TxHxWx3 shape")
    if frames.shape[1] < y + height or frames.shape[2] < x + width:
        raise ValueError(f"video is too small for crop {box}: {frames.shape}")
    return frames[:, y : y + height, x : x + width]


def _condition_ssim(cv2: Any, np: Any, generated: Any, target: Any) -> float:
    return float(np.mean(_ssim_map(cv2, np, generated[0], target[0])))


def main() -> int:
    args = _parser().parse_args()
    contract_path = args.dataset_contract.expanduser().resolve()
    if not contract_path.is_file():
        raise ValueError(f"dataset contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text())
    if contract.get("leakage_checks", {}).get("final_holdout_used_for_training") is not False:
        raise ValueError("dataset contract does not attest final-holdout isolation")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    examples = []
    for inference_dir_raw in args.inference_dir:
        inference_dir = inference_dir_raw.expanduser().resolve()
        result_path = inference_dir / "result.json"
        config_path = inference_dir / "experiment-config.json"
        result = json.loads(result_path.read_text())
        config = json.loads(config_path.read_text())
        if result.get("status") != "WORKING":
            raise ValueError(f"inference is not WORKING: {inference_dir}")
        condition = config.get("conditioning", {})
        if condition.get("real_future_frames_passed_to_model") is not False:
            raise ValueError("inference failed the real-future-frame leakage guard")
        sample_id = str(condition["sample_id"])
        row = _record(contract, sample_id, args.allowed_split)
        generated_path = inference_dir / "generated.mp4"
        if _sha256(generated_path) != result.get("generated_video_sha256"):
            raise ValueError(f"generated video hash changed: {generated_path}")
        target_path = (contract_path.parent / row["real_multiview_target_video"]).resolve()
        if _sha256(target_path) != row["real_multiview_target_video_sha256"]:
            raise ValueError(f"real target hash changed: {target_path}")
        generated = _decode(cv2, np, generated_path)
        target = _decode(cv2, np, target_path)
        frame_count = min(len(generated), len(target))
        if frame_count < 3:
            raise ValueError("evaluation requires at least three aligned frames")
        generated = generated[:frame_count]
        target = target[:frame_count]

        for view, box in VIEWS.items():
            generated_view = _crop(generated, box)
            target_view = _crop(target, box)
            condition_ssim = _condition_ssim(cv2, np, generated_view, target_view)
            # The first frame is a real condition.  Quality gates cover generated
            # continuation frames only, preventing condition leakage from inflating scores.
            metrics = evaluate_arrays(
                cv2,
                np,
                generated_view[1:],
                target_view[1:],
                anchor=target_view[0],
            )
            gates = _gate(metrics)
            gates["condition_first_frame_ssim"] = (
                condition_ssim >= CONDITION_FIRST_FRAME_SSIM_MIN
            )
            examples.append(
                {
                    "sample_id": sample_id,
                    "episode_index": row["episode_index"],
                    "split": row["split"],
                    "view": view,
                    "generated": str(generated_path),
                    "real_target": str(target_path),
                    "evaluated_frames": frame_count - 1,
                    "condition_first_frame_ssim": condition_ssim,
                    "metrics": metrics,
                    "gates": gates,
                    "accepted": all(gates.values()),
                    "inference_result_sha256": _sha256(result_path),
                    "inference_config_sha256": _sha256(config_path),
                }
            )

    if not examples:
        raise ValueError("no inference examples were provided")
    aggregate_metrics = {
        key: float(sum(row["metrics"][key] for row in examples) / len(examples))
        for key in examples[0]["metrics"]
    }
    aggregate_condition_ssim = float(
        sum(row["condition_first_frame_ssim"] for row in examples) / len(examples)
    )
    aggregate_gates = _gate(aggregate_metrics)
    aggregate_gates["condition_first_frame_ssim"] = (
        aggregate_condition_ssim >= CONDITION_FIRST_FRAME_SSIM_MIN
    )
    accepted = all(row["accepted"] for row in examples) and all(
        aggregate_gates.values()
    )
    if any(not math.isfinite(value) for value in aggregate_metrics.values()):
        raise ValueError("evaluation produced a non-finite metric")
    payload = {
        "schema_version": "1.0.0",
        "method": "phiagent_cosmos_predict2_droid_multiview_strict_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if accepted else "PARTIAL",
        "accepted": accepted,
        "allowed_split": args.allowed_split,
        "gates": {**GATES, "condition_first_frame_ssim_min": CONDITION_FIRST_FRAME_SSIM_MIN},
        "aggregate_metrics": aggregate_metrics,
        "aggregate_condition_first_frame_ssim": aggregate_condition_ssim,
        "aggregate_gates": aggregate_gates,
        "examples": examples,
        "dataset_contract": str(contract_path),
        "dataset_contract_sha256": _sha256(contract_path),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "numpy": importlib.metadata.version("numpy"),
            "opencv-python": importlib.metadata.version("opencv-python"),
        },
        "disclosure": {
            "real_condition": "composite frame 1 and disclosed task text",
            "our_generated_video": "all evaluated continuation frames 2 onward",
            "real_target_usage": "evaluation only; never model input",
        },
    }
    _write_json(output, payload)
    print(json.dumps({"output": str(output), "accepted": accepted, "status": payload["status"]}))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
