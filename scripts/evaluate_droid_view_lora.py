#!/usr/bin/env python3
"""Evaluate held-out DROID novel-view generations against real synchronized targets."""

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


GATES = {
    "mean_full_frame_ssim_min": 0.60,
    "mean_subject_roi_ssim_min": 0.45,
    "mean_subject_edge_f1_min": 0.35,
    "motion_correlation_min": 0.30,
    "motion_magnitude_ratio_min": 0.25,
    "motion_magnitude_ratio_max": 4.0,
    "static_anchor_ssim_gain_min": 0.0,
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _heldout_target(
    contract: dict[str, Any],
    episode: int,
    view: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in contract.get("holdout_records", [])
        if int(row.get("episode_index", -1)) == episode
    ]
    if len(rows) != 1 or rows[0].get("training_use") is not False:
        raise ValueError(f"episode {episode} is not a unique held-out record")
    try:
        return rows[0]["targets"][view]
    except KeyError as exc:
        raise ValueError(f"held-out episode {episode} has no {view}") from exc


def _decode(cv2: Any, np: Any, path: Path) -> Any:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not decode video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    return np.stack(frames).astype(np.float32) / 255.0


def _ssim_map(cv2: Any, np: Any, first: Any, second: Any) -> Any:
    first_gray = cv2.cvtColor(first.astype(np.float32), cv2.COLOR_RGB2GRAY)
    second_gray = cv2.cvtColor(second.astype(np.float32), cv2.COLOR_RGB2GRAY)
    mu_first = cv2.GaussianBlur(first_gray, (11, 11), 1.5)
    mu_second = cv2.GaussianBlur(second_gray, (11, 11), 1.5)
    sigma_first = cv2.GaussianBlur(first_gray * first_gray, (11, 11), 1.5) - mu_first**2
    sigma_second = (
        cv2.GaussianBlur(second_gray * second_gray, (11, 11), 1.5) - mu_second**2
    )
    covariance = (
        cv2.GaussianBlur(first_gray * second_gray, (11, 11), 1.5)
        - mu_first * mu_second
    )
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_first * mu_second + c1) * (2 * covariance + c2)
    denominator = (mu_first**2 + mu_second**2 + c1) * (
        sigma_first + sigma_second + c2
    )
    return np.clip(numerator / np.maximum(denominator, 1e-12), -1.0, 1.0)


def _edge_f1(cv2: Any, np: Any, generated: Any, target: Any, mask: Any) -> float:
    generated_gray = cv2.cvtColor(
        np.rint(generated * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    )
    target_gray = cv2.cvtColor(np.rint(target * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    generated_edge = cv2.Canny(generated_gray, 70, 140) > 0
    target_edge = cv2.Canny(target_gray, 70, 140) > 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    generated_tolerant = cv2.dilate(generated_edge.astype(np.uint8), kernel) > 0
    target_tolerant = cv2.dilate(target_edge.astype(np.uint8), kernel) > 0
    generated_count = int(np.count_nonzero(generated_edge & mask))
    target_count = int(np.count_nonzero(target_edge & mask))
    if generated_count == 0 and target_count == 0:
        return 1.0
    if generated_count == 0 or target_count == 0:
        return 0.0
    precision = float(np.count_nonzero(generated_edge & target_tolerant & mask)) / generated_count
    recall = float(np.count_nonzero(target_edge & generated_tolerant & mask)) / target_count
    return 2 * precision * recall / max(precision + recall, 1e-12)


def evaluate_arrays(
    cv2: Any,
    np: Any,
    generated: Any,
    target: Any,
    *,
    anchor: Any | None = None,
) -> dict[str, float]:
    if generated.shape != target.shape or generated.ndim != 4 or generated.shape[-1] != 3:
        raise ValueError("generated and target videos must have equal TxHxWx3 shapes")
    if generated.shape[0] < 2:
        raise ValueError("evaluation requires at least two frames")
    anchor = target[0] if anchor is None else anchor
    if anchor.shape != target.shape[1:]:
        raise ValueError("anchor must have the same HxWx3 shape as one target frame")
    kernel = np.ones((7, 7), dtype=np.uint8)
    full_ssim = []
    subject_ssim = []
    edge_f1 = []
    static_ssim = []
    motion_pixels = []
    for generated_frame, target_frame in zip(generated, target):
        change = np.max(np.abs(target_frame - anchor), axis=2) > 0.045
        change = cv2.dilate(change.astype(np.uint8), kernel) > 0
        if np.count_nonzero(change) < 64:
            change = np.ones(change.shape, dtype=bool)
        current_map = _ssim_map(cv2, np, generated_frame, target_frame)
        static_map = _ssim_map(cv2, np, anchor, target_frame)
        full_ssim.append(float(np.mean(current_map)))
        subject_ssim.append(float(np.mean(current_map[change])))
        static_ssim.append(float(np.mean(static_map)))
        edge_f1.append(_edge_f1(cv2, np, generated_frame, target_frame, change))
        motion_pixels.append(float(np.mean(change)))
    generated_delta = np.mean(np.abs(np.diff(generated, axis=0)), axis=(1, 2, 3))
    target_delta = np.mean(np.abs(np.diff(target, axis=0)), axis=(1, 2, 3))
    if float(np.std(generated_delta)) < 1e-8 or float(np.std(target_delta)) < 1e-8:
        motion_correlation = 0.0
    else:
        motion_correlation = float(np.corrcoef(generated_delta, target_delta)[0, 1])
    motion_ratio = float(np.mean(generated_delta) / max(float(np.mean(target_delta)), 1e-8))
    mean_ssim = float(np.mean(full_ssim))
    mean_static = float(np.mean(static_ssim))
    return {
        "mean_full_frame_ssim": mean_ssim,
        "minimum_full_frame_ssim": float(np.min(full_ssim)),
        "mean_subject_roi_ssim": float(np.mean(subject_ssim)),
        "minimum_subject_roi_ssim": float(np.min(subject_ssim)),
        "mean_subject_edge_f1": float(np.mean(edge_f1)),
        "motion_correlation": motion_correlation,
        "motion_magnitude_ratio": motion_ratio,
        "mean_target_motion_fraction": float(np.mean(motion_pixels)),
        "static_anchor_mean_ssim": mean_static,
        "static_anchor_ssim_gain": mean_ssim - mean_static,
    }


def _gate(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "mean_full_frame_ssim": metrics["mean_full_frame_ssim"]
        >= GATES["mean_full_frame_ssim_min"],
        "mean_subject_roi_ssim": metrics["mean_subject_roi_ssim"]
        >= GATES["mean_subject_roi_ssim_min"],
        "mean_subject_edge_f1": metrics["mean_subject_edge_f1"]
        >= GATES["mean_subject_edge_f1_min"],
        "motion_correlation": metrics["motion_correlation"] >= GATES["motion_correlation_min"],
        "motion_magnitude_ratio": GATES["motion_magnitude_ratio_min"]
        <= metrics["motion_magnitude_ratio"]
        <= GATES["motion_magnitude_ratio_max"],
        "static_anchor_ssim_gain": metrics["static_anchor_ssim_gain"]
        >= GATES["static_anchor_ssim_gain_min"],
    }


def main() -> int:
    args = _parser().parse_args()
    contract_path = args.dataset_contract.expanduser().resolve()
    if not contract_path.is_file():
        raise ValueError(f"dataset contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text())
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    examples = []
    for raw_metadata in args.inference_metadata:
        metadata_path = raw_metadata.expanduser().resolve()
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") != "completed":
            raise ValueError(f"inference is not completed: {metadata_path}")
        guard = metadata.get("target_leakage_guard", {})
        if guard.get("real_target_passed_to_model") is not False:
            raise ValueError("inference did not pass the target-leakage guard")
        generated_path = metadata_path.parent / "our-generated-video.mp4"
        if _sha256(generated_path) != metadata.get("output_sha256"):
            raise ValueError(f"generated video hash changed: {generated_path}")
        episode = int(metadata["episode_index"])
        view = str(metadata["view"])
        target_spec = _heldout_target(contract, episode, view)
        target_path = (contract_path.parent / target_spec["target"]).resolve()
        if _sha256(target_path) != target_spec["target_sha256"]:
            raise ValueError(f"real target hash changed: {target_path}")
        generated = _decode(cv2, np, generated_path)
        target = _decode(cv2, np, target_path)
        metrics = evaluate_arrays(cv2, np, generated, target)
        gates = _gate(metrics)
        examples.append(
            {
                "episode_index": episode,
                "view": view,
                "generated": str(generated_path),
                "generated_sha256": _sha256(generated_path),
                "real_target": str(target_path),
                "real_target_sha256": _sha256(target_path),
                "metrics": metrics,
                "gates": gates,
                "accepted": all(gates.values()),
                "inference_metadata": str(metadata_path),
                "inference_metadata_sha256": _sha256(metadata_path),
            }
        )
    aggregate_metrics = {
        key: float(sum(item["metrics"][key] for item in examples) / len(examples))
        for key in examples[0]["metrics"]
    }
    aggregate_gates = _gate(aggregate_metrics)
    accepted = all(item["accepted"] for item in examples) and all(aggregate_gates.values())
    payload = {
        "schema_version": "1.0.0",
        "method": "phiagent_droid_view_lora_heldout_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if accepted else "PARTIAL",
        "accepted": accepted,
        "gates": GATES,
        "aggregate_metrics": aggregate_metrics,
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
        "limitations": [
            "Motion-derived subject ROIs are deterministic proxies, not semantic segmentations.",
            "Acceptance requires every held-out example and the aggregate to pass all gates.",
            "This evaluates captured held-out video, not physical robot execution.",
        ],
    }
    if any(not math.isfinite(value) for value in aggregate_metrics.values()):
        raise ValueError("evaluation produced a non-finite metric")
    _write_json(output, payload)
    print(json.dumps({"output": str(output), "accepted": accepted, "status": payload["status"]}))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
