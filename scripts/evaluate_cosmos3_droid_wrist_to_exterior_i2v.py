#!/usr/bin/env python3
"""Strict validation for true wrist-only to third-person Cosmos3 I2V."""

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
    _decode,
    _ssim_map,
    evaluate_arrays,
)


CANONICAL_WIDTH = 768
CANONICAL_HEIGHT = 432
CONDITION_FIRST_FRAME_SSIM_MIN = 0.90
FIRST_FUTURE_TARGET_SSIM_MIN = 0.35
VIEW_SWITCH_MARGIN_MIN = 0.10
WRIST_ONLY_GATES = {
    "mean_full_frame_ssim_min": 0.65,
    "minimum_full_frame_ssim_min": 0.35,
    "mean_subject_roi_ssim_min": 0.55,
    "minimum_subject_roi_ssim_min": 0.25,
    "mean_subject_edge_f1_min": 0.45,
    "motion_correlation_min": 0.35,
    "motion_magnitude_ratio_min": 0.40,
    "motion_magnitude_ratio_max": 2.50,
    "static_anchor_ssim_gain_min": 0.02,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--inference-metadata", type=Path, action="append", required=True)
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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validation_record(contract: dict[str, Any], sample_id: str) -> dict[str, Any]:
    if contract.get("method") != "cosmos3_nano_droid_wrist_only_to_exterior_i2v_sft_dataset":
        raise ValueError("dataset is not the wrist-only to exterior contract")
    leakage = contract.get("leakage_checks", {})
    required = {
        "final_holdout_used_for_training": False,
        "final_holdout_used_for_checkpoint_selection": False,
        "validation_future_frames_are_model_inputs": False,
        "condition_contains_exterior_pixels": False,
        "condition_contains_real_wrist_pixels_only": True,
    }
    if any(leakage.get(key) is not value for key, value in required.items()):
        raise ValueError("dataset does not pass the wrist-only leakage gates")
    rows = [
        row
        for row in contract.get("records", [])
        if str(row.get("sample_id")) == sample_id
    ]
    if len(rows) != 1:
        raise ValueError(f"sample is not unique in wrist-only contract: {sample_id}")
    row = rows[0]
    if row.get("split") != "validation" or row.get("training_use") is not False:
        raise ValueError(f"sample is not validation-only: {sample_id}")
    return row


def canonicalize_generated(cv2: Any, np: Any, frames: Any) -> Any:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("generated video must have TxHxWx3 shape")
    if frames.shape[1:3] == (CANONICAL_HEIGHT, CANONICAL_WIDTH):
        return frames.copy()
    return np.stack(
        [
            cv2.resize(
                frame,
                (CANONICAL_WIDTH, CANONICAL_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            for frame in frames
        ]
    )


def strict_gate(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "mean_full_frame_ssim": metrics["mean_full_frame_ssim"]
        >= WRIST_ONLY_GATES["mean_full_frame_ssim_min"],
        "minimum_full_frame_ssim": metrics["minimum_full_frame_ssim"]
        >= WRIST_ONLY_GATES["minimum_full_frame_ssim_min"],
        "mean_subject_roi_ssim": metrics["mean_subject_roi_ssim"]
        >= WRIST_ONLY_GATES["mean_subject_roi_ssim_min"],
        "minimum_subject_roi_ssim": metrics["minimum_subject_roi_ssim"]
        >= WRIST_ONLY_GATES["minimum_subject_roi_ssim_min"],
        "mean_subject_edge_f1": metrics["mean_subject_edge_f1"]
        >= WRIST_ONLY_GATES["mean_subject_edge_f1_min"],
        "motion_correlation": metrics["motion_correlation"]
        >= WRIST_ONLY_GATES["motion_correlation_min"],
        "motion_magnitude_ratio": WRIST_ONLY_GATES["motion_magnitude_ratio_min"]
        <= metrics["motion_magnitude_ratio"]
        <= WRIST_ONLY_GATES["motion_magnitude_ratio_max"],
        "static_anchor_ssim_gain": metrics["static_anchor_ssim_gain"]
        >= WRIST_ONLY_GATES["static_anchor_ssim_gain_min"],
    }


def view_switch_metrics(
    cv2: Any, np: Any, first_generated_future: Any, first_target_future: Any, wrist: Any
) -> dict[str, float]:
    target_ssim = float(
        np.mean(_ssim_map(cv2, np, first_generated_future, first_target_future))
    )
    wrist_ssim = float(np.mean(_ssim_map(cv2, np, first_generated_future, wrist)))
    return {
        "first_future_target_ssim": target_ssim,
        "first_future_wrist_ssim": wrist_ssim,
        "view_switch_margin": target_ssim - wrist_ssim,
    }


def _generated_path(metadata_path: Path, metadata: dict[str, Any]) -> Path:
    declared = Path(str(metadata.get("output", ""))).expanduser()
    if declared.is_file():
        return declared.resolve()
    sample_id = str(metadata.get("sampling", {}).get("name", ""))
    fallback = metadata_path.parent / "cosmos_output" / sample_id / "vision.mp4"
    if fallback.is_file():
        return fallback.resolve()
    raise ValueError(f"generated video is missing for {metadata_path}")


def main() -> int:
    args = _parser().parse_args()
    contract_path = args.dataset_contract.expanduser().resolve()
    if not contract_path.is_file():
        raise ValueError(f"dataset contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    examples: list[dict[str, Any]] = []
    for raw_metadata in args.inference_metadata:
        metadata_path = raw_metadata.expanduser().resolve()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "succeeded":
            raise ValueError(f"Cosmos3 inference is not succeeded: {metadata_path}")
        leakage = metadata.get("leakage_guard", {})
        if leakage.get("real_future_frames_passed_to_model") is not False:
            raise ValueError("inference failed the real-future-frame leakage guard")
        labels = metadata.get("labels", {})
        if labels.get("condition_image") != "REAL CONDITION":
            raise ValueError("inference metadata does not disclose the real condition")
        if labels.get("output_continuation") != "OUR GENERATED VIDEO":
            raise ValueError("inference metadata does not disclose generated continuation")
        sample_id = str(metadata.get("sampling", {}).get("name", ""))
        row = validation_record(contract, sample_id)
        generated_path = _generated_path(metadata_path, metadata)
        if _sha256(generated_path) != metadata.get("output_sha256"):
            raise ValueError(f"generated video hash changed: {generated_path}")
        condition_path = (contract_path.parent / row["condition"]).resolve()
        target_path = (contract_path.parent / row["target"]).resolve()
        if _sha256(condition_path) != row["condition_sha256"]:
            raise ValueError(f"real wrist condition hash changed: {condition_path}")
        if _sha256(target_path) != row["target_sha256"]:
            raise ValueError(f"withheld real target hash changed: {target_path}")

        generated_raw = _decode(cv2, np, generated_path)
        target = _decode(cv2, np, target_path)
        generated = canonicalize_generated(cv2, np, generated_raw)
        frame_count = min(len(generated), len(target))
        if frame_count < 4:
            raise ValueError("wrist-to-third evaluation requires at least three future frames")
        generated = generated[:frame_count]
        target = target[:frame_count]
        wrist = target[0]
        condition_first_ssim = float(np.mean(_ssim_map(cv2, np, generated[0], wrist)))
        transition = view_switch_metrics(cv2, np, generated[1], target[1], wrist)
        metrics = evaluate_arrays(
            cv2,
            np,
            generated[1:],
            target[1:],
            anchor=target[1],
        )
        gates = strict_gate(metrics)
        gates.update(
            {
                "condition_first_frame_ssim": condition_first_ssim
                >= CONDITION_FIRST_FRAME_SSIM_MIN,
                "first_future_target_ssim": transition["first_future_target_ssim"]
                >= FIRST_FUTURE_TARGET_SSIM_MIN,
                "view_switch_margin": transition["view_switch_margin"]
                >= VIEW_SWITCH_MARGIN_MIN,
            }
        )
        examples.append(
            {
                "sample_id": sample_id,
                "source_sample_id": row["source_sample_id"],
                "episode_index": int(row["episode_index"]),
                "target_view": row["target_view"],
                "condition": str(condition_path),
                "condition_sha256": row["condition_sha256"],
                "generated": str(generated_path),
                "generated_sha256": _sha256(generated_path),
                "withheld_real_target": str(target_path),
                "withheld_real_target_sha256": row["target_sha256"],
                "evaluated_continuation_frames": frame_count - 1,
                "condition_first_frame_ssim": condition_first_ssim,
                "view_switch_metrics": transition,
                "metrics": metrics,
                "gates": gates,
                "accepted": all(gates.values()),
                "pixel_frame_transform": {
                    "name": "cosmos_model_pixel_frame_to_resized_named_droid_camera_pixel_frame",
                    "generated_source_wh": [int(generated_raw.shape[2]), int(generated_raw.shape[1])],
                    "canonical_wh": [CANONICAL_WIDTH, CANONICAL_HEIGHT],
                    "operation": "independent full-frame area resize; no camera-frame mixing",
                },
                "inference_metadata": str(metadata_path),
                "inference_metadata_sha256": _sha256(metadata_path),
                "model": metadata.get("model"),
                "sampling": metadata.get("sampling"),
            }
        )

    if not examples:
        raise ValueError("no wrist-only validation examples were supplied")
    aggregate_metrics = {
        key: float(sum(row["metrics"][key] for row in examples) / len(examples))
        for key in examples[0]["metrics"]
    }
    aggregate_transition = {
        key: float(
            sum(row["view_switch_metrics"][key] for row in examples) / len(examples)
        )
        for key in examples[0]["view_switch_metrics"]
    }
    aggregate_condition_ssim = float(
        sum(row["condition_first_frame_ssim"] for row in examples) / len(examples)
    )
    aggregate_gates = strict_gate(aggregate_metrics)
    aggregate_gates.update(
        {
            "condition_first_frame_ssim": aggregate_condition_ssim
            >= CONDITION_FIRST_FRAME_SSIM_MIN,
            "first_future_target_ssim": aggregate_transition["first_future_target_ssim"]
            >= FIRST_FUTURE_TARGET_SSIM_MIN,
            "view_switch_margin": aggregate_transition["view_switch_margin"]
            >= VIEW_SWITCH_MARGIN_MIN,
        }
    )
    values = [*aggregate_metrics.values(), *aggregate_transition.values(), aggregate_condition_ssim]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("evaluation produced a non-finite metric")
    accepted = all(row["accepted"] for row in examples) and all(aggregate_gates.values())
    payload = {
        "schema_version": "1.0.0",
        "method": "phiagent_cosmos3_droid_wrist_only_to_exterior_strict_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if accepted else "PARTIAL",
        "accepted": accepted,
        "gates": {
            **WRIST_ONLY_GATES,
            "condition_first_frame_ssim_min": CONDITION_FIRST_FRAME_SSIM_MIN,
            "first_future_target_ssim_min": FIRST_FUTURE_TARGET_SSIM_MIN,
            "view_switch_margin_min": VIEW_SWITCH_MARGIN_MIN,
        },
        "aggregate_metrics": aggregate_metrics,
        "aggregate_view_switch_metrics": aggregate_transition,
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
            "real_condition": "one real first-person wrist-camera frame plus real task text",
            "condition_contains_third_person_pixels": False,
            "our_generated_video": "all third-person continuation frames after frame 1",
            "withheld_real_target": "synchronized exterior-camera future used only after generation",
            "pure_wrist_only_claim": True,
        },
        "limitations": [
            "Motion-derived subject ROIs are deterministic proxies rather than semantic instance masks.",
            "Acceptance requires every target view, every validation sample, and aggregate gates to pass.",
            "This validates held-out captured video, not a physical robot execution.",
        ],
    }
    _write_json(output, payload)
    print(json.dumps({"output": str(output), "accepted": accepted, "status": payload["status"]}))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
