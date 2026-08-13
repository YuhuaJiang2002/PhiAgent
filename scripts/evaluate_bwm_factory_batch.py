#!/usr/bin/env python3
"""Evaluate a BWM factory batch against frozen task-disjoint references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_bwm_heldout_pair import _decode, _flow_metrics, _ssim


LOWER_IS_BETTER = (
    "background_mad_0_1",
    "temporal_gradient_mae_0_1",
    "motion_amplitude_relative_error",
    "flow_endpoint_error_px_at_224x168",
    "endpoint_mad_0_1",
)
HIGHER_IS_BETTER = (
    "future_psnr_db",
    "future_ssim",
    "flow_direction_cosine",
    "endpoint_ssim",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measure(
    cv2: Any,
    np: Any,
    prediction: list[Any],
    reference: list[Any],
    static: Any,
    history_frames: int,
) -> dict[str, float]:
    future = range(history_frames, len(reference))
    squared = [
        float(np.mean((prediction[index].astype(np.float32) - reference[index]) ** 2))
        for index in future
    ]
    psnr = [10.0 * math.log10(255.0**2 / max(value, 1e-12)) for value in squared]
    ssims = [_ssim(cv2, np, prediction[index], reference[index]) for index in future]
    background = [
        float(
            np.mean(
                np.abs(prediction[index].astype(np.float32) - reference[index])[static]
            )
        )
        / 255.0
        for index in future
    ]
    temporal = [
        float(
            np.mean(
                np.abs(
                    (
                        prediction[index].astype(np.float32)
                        - prediction[index - 1].astype(np.float32)
                    )
                    - (
                        reference[index].astype(np.float32)
                        - reference[index - 1].astype(np.float32)
                    )
                )
            )
        )
        / 255.0
        for index in future
    ]
    prediction_future = prediction[history_frames - 1 :]
    reference_future = reference[history_frames - 1 :]
    prediction_motion = np.mean(
        [
            float(
                np.mean(
                    np.abs(
                        prediction_future[index].astype(np.float32)
                        - prediction_future[index - 1].astype(np.float32)
                    )
                )
            )
            for index in range(1, len(prediction_future))
        ]
    )
    reference_motion = np.mean(
        [
            float(
                np.mean(
                    np.abs(
                        reference_future[index].astype(np.float32)
                        - reference_future[index - 1].astype(np.float32)
                    )
                )
            )
            for index in range(1, len(reference_future))
        ]
    )
    flow_epe, flow_cosine = _flow_metrics(
        cv2, np, prediction_future, reference_future
    )
    return {
        "future_psnr_db": float(np.mean(psnr)),
        "future_ssim": float(np.mean(ssims)),
        "background_mad_0_1": float(np.mean(background)),
        "temporal_gradient_mae_0_1": float(np.mean(temporal)),
        "motion_amplitude_relative_error": float(
            abs(prediction_motion - reference_motion) / max(reference_motion, 1e-6)
        ),
        "flow_endpoint_error_px_at_224x168": flow_epe,
        "flow_direction_cosine": flow_cosine,
        "endpoint_ssim": _ssim(cv2, np, prediction[-1], reference[-1]),
        "endpoint_mad_0_1": float(
            np.mean(np.abs(prediction[-1].astype(np.float32) - reference[-1]))
            / 255.0
        ),
        "first_frame_mad_0_1": float(
            np.mean(np.abs(prediction[0].astype(np.float32) - reference[0]))
            / 255.0
        ),
    }


def _aggregate(np: Any, samples: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    fields = tuple(samples[0]["metrics"].keys())  # type: ignore[union-attr]
    result = {}
    for field in fields:
        values = np.asarray(
            [float(sample["metrics"][field]) for sample in samples],  # type: ignore[index]
            dtype=np.float64,
        )
        result[field] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return result


def _evaluate_directory(
    cv2: Any,
    np: Any,
    rows: list[dict[str, object]],
    dataset_root: Path,
    prediction_root: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    samples = []
    for row in rows:
        episode_index = int(row["episode_index"])
        video = row["video"]
        if not isinstance(video, dict):
            raise ValueError("metadata video field must be an object")
        reference_path = dataset_root / str(video["data"])
        prediction_path = prediction_root / f"episode{episode_index}.mp4"
        if not reference_path.is_file() or not prediction_path.is_file():
            raise ValueError(
                f"missing reference or prediction for episode {episode_index}"
            )
        start = int(video["start_frame"])
        end = int(video["end_frame"])
        reference_all = _decode(cv2, reference_path)
        reference = reference_all[start : end + 1]
        prediction = _decode(cv2, prediction_path)
        if len(reference) != int(row["length"]) or len(prediction) != len(reference):
            raise ValueError(f"frame-count mismatch for episode {episode_index}")
        size = reference[0].shape[1], reference[0].shape[0]
        prediction = [
            cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            if frame.shape[1::-1] != size
            else frame
            for frame in prediction
        ]
        gray = np.stack(
            [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in reference]
        )
        variation = np.std(gray.astype(np.float32), axis=0)
        threshold = float(np.percentile(variation, 60))
        static2d = variation <= threshold
        static = np.repeat(static2d[:, :, None], 3, axis=2)
        samples.append(
            {
                "episode_index": episode_index,
                "task": row.get("task"),
                "source_episode": row.get("source_episode"),
                "history_frames": int(row["history_frames"]),
                "reference": str(reference_path),
                "reference_sha256": _sha256(reference_path),
                "prediction": str(prediction_path),
                "prediction_sha256": _sha256(prediction_path),
                "static_background_threshold": threshold,
                "static_background_fraction": float(np.mean(static2d)),
                "metrics": _measure(
                    cv2,
                    np,
                    prediction,
                    reference,
                    static,
                    int(row["history_frames"]),
                ),
            }
        )
    return samples, _aggregate(np, samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--candidate-videos", type=Path, required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--baseline-videos", type=Path)
    parser.add_argument("--baseline-label", default="official-bwm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    import numpy as np

    metadata = args.metadata.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    candidate_root = args.candidate_videos.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    if not metadata.is_file() or not dataset_root.is_dir() or not candidate_root.is_dir():
        raise ValueError("metadata, dataset root, or candidate directory is missing")
    rows = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("metadata contains no samples")
    candidate_samples, candidate_aggregate = _evaluate_directory(
        cv2, np, rows, dataset_root, candidate_root
    )
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "metadata": str(metadata),
        "metadata_sha256": _sha256(metadata),
        "dataset_root": str(dataset_root),
        "candidate": {
            "label": args.candidate_label,
            "videos": str(candidate_root),
            "samples": candidate_samples,
            "aggregate": candidate_aggregate,
        },
        "limitations": [
            "Pixel and optical-flow metrics do not establish 3-D contact physics or task success.",
            (
                "References are the exact metadata-selected windows decoded from their "
                "declared source videos; source-codec loss remains part of the reference."
            ),
            "No physical robot was executed in this evaluation.",
        ],
    }
    if args.baseline_videos is not None:
        baseline_root = args.baseline_videos.expanduser().resolve()
        baseline_samples, baseline_aggregate = _evaluate_directory(
            cv2, np, rows, dataset_root, baseline_root
        )
        wins = {}
        for field in LOWER_IS_BETTER:
            wins[field] = (
                candidate_aggregate[field]["mean"]
                < baseline_aggregate[field]["mean"]
            )
        for field in HIGHER_IS_BETTER:
            wins[field] = (
                candidate_aggregate[field]["mean"]
                > baseline_aggregate[field]["mean"]
            )
        sample_ssim_wins = sum(
            float(candidate["metrics"]["future_ssim"])  # type: ignore[index]
            > float(baseline["metrics"]["future_ssim"])  # type: ignore[index]
            for candidate, baseline in zip(candidate_samples, baseline_samples)
        )
        ssim_win_fraction = sample_ssim_wins / len(candidate_samples)
        gates = {
            "mean_future_ssim_improves_by_0_002": (
                candidate_aggregate["future_ssim"]["mean"]
                >= baseline_aggregate["future_ssim"]["mean"] + 0.002
            ),
            "endpoint_ssim_non_regression": (
                candidate_aggregate["endpoint_ssim"]["mean"]
                >= baseline_aggregate["endpoint_ssim"]["mean"] - 0.005
            ),
            "background_non_regression": (
                candidate_aggregate["background_mad_0_1"]["mean"]
                <= 1.05 * baseline_aggregate["background_mad_0_1"]["mean"]
            ),
            "temporal_non_regression": (
                candidate_aggregate["temporal_gradient_mae_0_1"]["mean"]
                <= 1.05
                * baseline_aggregate["temporal_gradient_mae_0_1"]["mean"]
            ),
            "flow_endpoint_non_regression": (
                candidate_aggregate["flow_endpoint_error_px_at_224x168"]["mean"]
                <= 1.05
                * baseline_aggregate["flow_endpoint_error_px_at_224x168"]["mean"]
            ),
            "per_sample_ssim_win_fraction_at_least_half": ssim_win_fraction >= 0.5,
        }
        accepted = all(gates.values())
        payload.update(
            {
                "status": "WORKING" if accepted else "PARTIAL",
                "baseline": {
                    "label": args.baseline_label,
                    "videos": str(baseline_root),
                    "samples": baseline_samples,
                    "aggregate": baseline_aggregate,
                },
                "candidate_metric_wins": wins,
                "candidate_sample_future_ssim_win_fraction": ssim_win_fraction,
                "gates": gates,
                "accepted": accepted,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "WORKING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
