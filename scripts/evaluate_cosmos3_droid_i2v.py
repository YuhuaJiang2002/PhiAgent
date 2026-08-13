#!/usr/bin/env python3
"""Strict validation-only evaluation for Cosmos3 DROID 2x2 I2V outputs.

Cosmos3 generates a model-native 16:9 raster (normally 832x480), while the
captured DROID composite is 768x432.  Each quadrant is therefore mapped from
its named model pixel frame into the canonical DROID tile pixel frame before
comparison.  This avoids silently treating two different camera/pixel frames
as if they were identical.
"""

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


CANONICAL_TILE_WIDTH = 384
CANONICAL_TILE_HEIGHT = 216
CONDITION_FIRST_FRAME_SSIM_MIN = 0.90
INACTIVE_BLACK_MEAN_MAX = 0.08
INACTIVE_BLACK_P99_MAX = 0.15
COSMOS3_GATES = {
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
ACTIVE_VIEWS = {
    "exterior_1": "top_left",
    "exterior_2": "top_right",
    "wrist": "bottom_left",
}
ALL_TILES = {
    "top_left": (0, 0),
    "top_right": (1, 0),
    "bottom_left": (0, 1),
    "bottom_right": (1, 1),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--inference-metadata", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowed-split", choices=("validation",), default="validation")
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


def _record(contract: dict[str, Any], sample_id: str, split: str) -> dict[str, Any]:
    rows = [row for row in contract.get("records", []) if row.get("sample_id") == sample_id]
    if len(rows) != 1:
        raise ValueError(f"sample is not unique in source contract: {sample_id}")
    row = rows[0]
    if row.get("split") != split or row.get("training_use") is not False:
        raise ValueError(f"sample {sample_id} is not a non-training {split} record")
    return row


def canonicalize_tile(cv2: Any, frames: Any, tile: str) -> Any:
    """Map one named 2x2 tile into canonical DROID tile pixel coordinates."""

    if tile not in ALL_TILES:
        raise ValueError(f"unknown 2x2 tile: {tile}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("video must have TxHxWx3 shape")
    height, width = (int(frames.shape[1]), int(frames.shape[2]))
    if height < 2 or width < 2 or height % 2 or width % 2:
        raise ValueError(f"2x2 video dimensions must be positive and even: {width}x{height}")
    column, row = ALL_TILES[tile]
    tile_width = width // 2
    tile_height = height // 2
    cropped = frames[
        :,
        row * tile_height : (row + 1) * tile_height,
        column * tile_width : (column + 1) * tile_width,
    ]
    if (tile_width, tile_height) == (CANONICAL_TILE_WIDTH, CANONICAL_TILE_HEIGHT):
        return cropped.copy()
    interpolation = (
        cv2.INTER_AREA
        if tile_width >= CANONICAL_TILE_WIDTH and tile_height >= CANONICAL_TILE_HEIGHT
        else cv2.INTER_CUBIC
    )
    return __import__("numpy").stack(
        [
            cv2.resize(
                frame,
                (CANONICAL_TILE_WIDTH, CANONICAL_TILE_HEIGHT),
                interpolation=interpolation,
            )
            for frame in cropped
        ]
    )


def condition_ssim(cv2: Any, np: Any, generated: Any, target: Any) -> float:
    return float(np.mean(_ssim_map(cv2, np, generated[0], target[0])))


def inactive_black_metrics(np: Any, tile: Any) -> dict[str, float]:
    continuation = tile[1:]
    if not len(continuation):
        raise ValueError("inactive-tile evaluation requires continuation frames")
    luminance = np.mean(continuation, axis=3)
    return {
        "mean_luminance": float(np.mean(luminance)),
        "p99_luminance": float(np.quantile(luminance, 0.99)),
    }


def strict_gate(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "mean_full_frame_ssim": metrics["mean_full_frame_ssim"]
        >= COSMOS3_GATES["mean_full_frame_ssim_min"],
        "minimum_full_frame_ssim": metrics["minimum_full_frame_ssim"]
        >= COSMOS3_GATES["minimum_full_frame_ssim_min"],
        "mean_subject_roi_ssim": metrics["mean_subject_roi_ssim"]
        >= COSMOS3_GATES["mean_subject_roi_ssim_min"],
        "minimum_subject_roi_ssim": metrics["minimum_subject_roi_ssim"]
        >= COSMOS3_GATES["minimum_subject_roi_ssim_min"],
        "mean_subject_edge_f1": metrics["mean_subject_edge_f1"]
        >= COSMOS3_GATES["mean_subject_edge_f1_min"],
        "motion_correlation": metrics["motion_correlation"]
        >= COSMOS3_GATES["motion_correlation_min"],
        "motion_magnitude_ratio": COSMOS3_GATES["motion_magnitude_ratio_min"]
        <= metrics["motion_magnitude_ratio"]
        <= COSMOS3_GATES["motion_magnitude_ratio_max"],
        "static_anchor_ssim_gain": metrics["static_anchor_ssim_gain"]
        >= COSMOS3_GATES["static_anchor_ssim_gain_min"],
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
    leakage = contract.get("leakage_checks", {})
    if leakage.get("final_holdout_used_for_training") is not False:
        raise ValueError("dataset contract does not isolate final holdout")
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
        guard = metadata.get("leakage_guard", {})
        if guard.get("real_future_frames_passed_to_model") is not False:
            raise ValueError("inference failed the real-future-frame leakage guard")
        labels = metadata.get("labels", {})
        if labels.get("condition_image") != "REAL CONDITION":
            raise ValueError("inference metadata does not disclose the real condition")
        if labels.get("output_continuation") != "OUR GENERATED VIDEO":
            raise ValueError("inference metadata does not disclose generated continuation")

        sample_id = str(metadata.get("sampling", {}).get("name", ""))
        row = _record(contract, sample_id, args.allowed_split)
        generated_path = _generated_path(metadata_path, metadata)
        if _sha256(generated_path) != metadata.get("output_sha256"):
            raise ValueError(f"generated video hash changed: {generated_path}")
        target_path = (contract_path.parent / row["real_multiview_target_video"]).resolve()
        if _sha256(target_path) != row["real_multiview_target_video_sha256"]:
            raise ValueError(f"withheld real target hash changed: {target_path}")

        generated = _decode(cv2, np, generated_path)
        target = _decode(cv2, np, target_path)
        frame_count = min(len(generated), len(target))
        if frame_count < 3:
            raise ValueError("evaluation requires a real condition plus two generated frames")
        generated = generated[:frame_count]
        target = target[:frame_count]
        generated_shape = [int(value) for value in generated.shape[1:3][::-1]]
        target_shape = [int(value) for value in target.shape[1:3][::-1]]

        view_rows = []
        for view, tile in ACTIVE_VIEWS.items():
            generated_view = canonicalize_tile(cv2, generated, tile)
            target_view = canonicalize_tile(cv2, target, tile)
            first_frame_ssim = condition_ssim(cv2, np, generated_view, target_view)
            metrics = evaluate_arrays(
                cv2,
                np,
                generated_view[1:],
                target_view[1:],
                anchor=target_view[0],
            )
            gates = strict_gate(metrics)
            gates["condition_first_frame_ssim"] = (
                first_frame_ssim >= CONDITION_FIRST_FRAME_SSIM_MIN
            )
            view_rows.append(
                {
                    "view": view,
                    "source_tile": tile,
                    "condition_first_frame_ssim": first_frame_ssim,
                    "metrics": metrics,
                    "gates": gates,
                    "accepted": all(gates.values()),
                }
            )

        inactive = inactive_black_metrics(
            np, canonicalize_tile(cv2, generated, "bottom_right")
        )
        inactive_gates = {
            "mean_luminance": inactive["mean_luminance"] <= INACTIVE_BLACK_MEAN_MAX,
            "p99_luminance": inactive["p99_luminance"] <= INACTIVE_BLACK_P99_MAX,
        }
        examples.append(
            {
                "sample_id": sample_id,
                "episode_index": int(row["episode_index"]),
                "split": row["split"],
                "generated": str(generated_path),
                "generated_sha256": _sha256(generated_path),
                "withheld_real_target": str(target_path),
                "withheld_real_target_sha256": _sha256(target_path),
                "evaluated_continuation_frames": frame_count - 1,
                "pixel_frame_transform": {
                    "name": "model_2x2_tile_pixel_frame_to_canonical_droid_tile_pixel_frame",
                    "generated_composite_wh": generated_shape,
                    "real_target_composite_wh": target_shape,
                    "canonical_tile_wh": [CANONICAL_TILE_WIDTH, CANONICAL_TILE_HEIGHT],
                    "operation": "split each composite at exact half-width/half-height, then independently resize each named tile",
                },
                "views": view_rows,
                "inactive_black_tile": {
                    "metrics": inactive,
                    "gates": inactive_gates,
                    "accepted": all(inactive_gates.values()),
                },
                "accepted": all(view["accepted"] for view in view_rows)
                and all(inactive_gates.values()),
                "inference_metadata": str(metadata_path),
                "inference_metadata_sha256": _sha256(metadata_path),
                "model": metadata.get("model"),
                "sampling": metadata.get("sampling"),
            }
        )

    if not examples:
        raise ValueError("no Cosmos3 validation examples were supplied")
    view_examples = [view for example in examples for view in example["views"]]
    aggregate_metrics = {
        key: float(sum(view["metrics"][key] for view in view_examples) / len(view_examples))
        for key in view_examples[0]["metrics"]
    }
    aggregate_condition_ssim = float(
        sum(view["condition_first_frame_ssim"] for view in view_examples)
        / len(view_examples)
    )
    aggregate_gates = strict_gate(aggregate_metrics)
    aggregate_gates["condition_first_frame_ssim"] = (
        aggregate_condition_ssim >= CONDITION_FIRST_FRAME_SSIM_MIN
    )
    accepted = all(example["accepted"] for example in examples) and all(
        aggregate_gates.values()
    )
    if any(not math.isfinite(value) for value in aggregate_metrics.values()):
        raise ValueError("evaluation produced a non-finite metric")

    payload = {
        "schema_version": "1.0.0",
        "method": "phiagent_cosmos3_droid_multiview_i2v_strict_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if accepted else "PARTIAL",
        "accepted": accepted,
        "allowed_split": args.allowed_split,
        "gates": {
            **COSMOS3_GATES,
            "condition_first_frame_ssim_min": CONDITION_FIRST_FRAME_SSIM_MIN,
            "inactive_black_mean_max": INACTIVE_BLACK_MEAN_MAX,
            "inactive_black_p99_max": INACTIVE_BLACK_P99_MAX,
        },
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
            "real_condition": (
                "one real 2x2 first frame containing two exterior identity anchors, "
                "one wrist/first-person view, and one black inactive tile; plus task text"
            ),
            "our_generated_video": "all model continuation frames after the disclosed first frame",
            "withheld_real_target": "post-generation validation only; never a model input",
            "pure_wrist_only_claim": False,
        },
        "limitations": [
            "The condition includes real third-person anchor pixels, so this is disclosed anchor-conditioned ego-to-third-person generalization, not wrist-only generation.",
            "Motion-derived subject ROIs are deterministic proxies rather than semantic instance masks.",
            "Acceptance requires every active view, every validation sample, the inactive tile, and aggregate gates to pass.",
        ],
    }
    _write_json(output, payload)
    print(json.dumps({"output": str(output), "accepted": accepted, "status": payload["status"]}))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
