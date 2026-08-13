#!/usr/bin/env python3
"""Evaluate paired BWM rollouts with reference and counterfactual actions."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(cv2: Any, path: Path) -> list[Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video contains no frames: {path}")
    return frames


def _ssim(cv2: Any, np: Any, left: Any, right: Any) -> float:
    x = left.astype(np.float64)
    y = right.astype(np.float64)
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def _flow_metrics(cv2: Any, np: Any, prediction: list[Any], reference: list[Any]) -> tuple[float, float]:
    epes = []
    cosines = []
    size = (224, 168)
    for index in range(1, len(reference)):
        ref0 = cv2.resize(cv2.cvtColor(reference[index - 1], cv2.COLOR_BGR2GRAY), size)
        ref1 = cv2.resize(cv2.cvtColor(reference[index], cv2.COLOR_BGR2GRAY), size)
        pred0 = cv2.resize(cv2.cvtColor(prediction[index - 1], cv2.COLOR_BGR2GRAY), size)
        pred1 = cv2.resize(cv2.cvtColor(prediction[index], cv2.COLOR_BGR2GRAY), size)
        ref_flow = cv2.calcOpticalFlowFarneback(ref0, ref1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        pred_flow = cv2.calcOpticalFlowFarneback(pred0, pred1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        ref_mag = np.linalg.norm(ref_flow, axis=2)
        active = ref_mag >= max(0.25, float(np.percentile(ref_mag, 75)))
        if not np.any(active):
            continue
        delta = np.linalg.norm(pred_flow - ref_flow, axis=2)
        epes.append(float(np.mean(delta[active])))
        dot = np.sum(pred_flow * ref_flow, axis=2)
        pred_mag = np.linalg.norm(pred_flow, axis=2)
        cosine = dot / np.maximum(pred_mag * ref_mag, 1e-6)
        cosines.append(float(np.mean(np.clip(cosine[active], -1.0, 1.0))))
    return float(np.mean(epes)), float(np.mean(cosines))


def _measure(cv2: Any, np: Any, prediction: list[Any], reference: list[Any], static: Any) -> dict[str, float]:
    future = range(1, len(reference))
    squared = [float(np.mean((prediction[i].astype(np.float32) - reference[i]) ** 2)) for i in future]
    psnr = [10.0 * math.log10(255.0**2 / max(value, 1e-12)) for value in squared]
    ssims = [_ssim(cv2, np, prediction[i], reference[i]) for i in future]
    background = [
        float(np.mean(np.abs(prediction[i].astype(np.float32) - reference[i])[static])) / 255.0
        for i in future
    ]
    temporal = [
        float(
            np.mean(
                np.abs(
                    (prediction[i].astype(np.float32) - prediction[i - 1].astype(np.float32))
                    - (reference[i].astype(np.float32) - reference[i - 1].astype(np.float32))
                )
            )
        )
        / 255.0
        for i in future
    ]
    pred_motion = np.mean(
        [float(np.mean(np.abs(prediction[i].astype(np.float32) - prediction[i - 1]))) for i in future]
    )
    ref_motion = np.mean(
        [float(np.mean(np.abs(reference[i].astype(np.float32) - reference[i - 1]))) for i in future]
    )
    flow_epe, flow_cosine = _flow_metrics(cv2, np, prediction, reference)
    return {
        "future_psnr_db": float(np.mean(psnr)),
        "future_ssim": float(np.mean(ssims)),
        "background_mad_0_1": float(np.mean(background)),
        "temporal_gradient_mae_0_1": float(np.mean(temporal)),
        "motion_amplitude_relative_error": float(abs(pred_motion - ref_motion) / max(ref_motion, 1e-6)),
        "flow_endpoint_error_px_at_224x168": flow_epe,
        "flow_direction_cosine": flow_cosine,
        "endpoint_ssim": _ssim(cv2, np, prediction[-1], reference[-1]),
        "endpoint_mad_0_1": float(
            np.mean(np.abs(prediction[-1].astype(np.float32) - reference[-1])) / 255.0
        ),
        "first_frame_mad_0_1": float(
            np.mean(np.abs(prediction[0].astype(np.float32) - reference[0])) / 255.0
        ),
    }


def _compose(cv2: Any, np: Any, reference: list[Any], baseline: list[Any], candidate: list[Any], output: Path) -> None:
    width, height = 448, 336
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (width * 3, height + 44))
    if not writer.isOpened():
        raise RuntimeError("cannot open comparison video writer")
    labels = ("HELD-OUT REFERENCE", "OFFICIAL BWM", "TRAINED ADAPTER")
    for index in range(len(reference)):
        canvas = np.zeros((height + 44, width * 3, 3), dtype=np.uint8)
        for column, (label, frames) in enumerate(zip(labels, (reference, baseline, candidate))):
            cell = cv2.resize(frames[index], (width, height), interpolation=cv2.INTER_AREA)
            canvas[44:, column * width : (column + 1) * width] = cell
            cv2.putText(canvas, label, (column * width + 12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {index:02d}", (width * 3 - 126, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190, 220, 255), 2, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-counterfactual", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-counterfactual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-video", type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np

    inputs = {
        name: getattr(args, name).expanduser().resolve()
        for name in ("reference", "baseline", "baseline_counterfactual", "candidate", "candidate_counterfactual")
    }
    for path in inputs.values():
        if not path.is_file():
            raise ValueError(f"evaluation input is missing: {path}")
    reference_all = _decode(cv2, inputs["reference"])
    reference = reference_all[args.start_frame : args.end_frame + 1]
    if len(reference) != args.end_frame - args.start_frame + 1:
        raise ValueError("reference window is incomplete")
    size = reference[0].shape[1], reference[0].shape[0]
    videos = {}
    for name in ("baseline", "baseline_counterfactual", "candidate", "candidate_counterfactual"):
        frames = _decode(cv2, inputs[name])
        if len(frames) != len(reference):
            raise ValueError(f"{name} frame count does not match the reference")
        videos[name] = [
            cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR) if frame.shape[1::-1] != size else frame
            for frame in frames
        ]
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in reference])
    variation = np.std(gray.astype(np.float32), axis=0)
    threshold = float(np.percentile(variation, 60))
    static2d = variation <= threshold
    static = np.repeat(static2d[:, :, None], 3, axis=2)
    reports = {
        "baseline": _measure(cv2, np, videos["baseline"], reference, static),
        "candidate": _measure(cv2, np, videos["candidate"], reference, static),
    }
    for name, counterfactual_name in (("baseline", "baseline_counterfactual"), ("candidate", "candidate_counterfactual")):
        correct = videos[name]
        counterfactual = videos[counterfactual_name]
        correct_mad = float(np.mean([np.mean(np.abs(correct[i].astype(np.float32) - reference[i])) for i in range(1, len(reference))]) / 255.0)
        counterfactual_mad = float(np.mean([np.mean(np.abs(counterfactual[i].astype(np.float32) - reference[i])) for i in range(1, len(reference))]) / 255.0)
        separation = float(np.mean([np.mean(np.abs(correct[i].astype(np.float32) - counterfactual[i])) for i in range(1, len(reference))]) / 255.0)
        reports[name].update(
            {
                "correct_action_mad_0_1": correct_mad,
                "counterfactual_action_mad_0_1": counterfactual_mad,
                "action_causal_margin_0_1": counterfactual_mad - correct_mad,
                "action_counterfactual_separation_0_1": separation,
            }
        )
    lower = (
        "background_mad_0_1",
        "temporal_gradient_mae_0_1",
        "motion_amplitude_relative_error",
        "flow_endpoint_error_px_at_224x168",
        "endpoint_mad_0_1",
    )
    higher = ("future_psnr_db", "future_ssim", "flow_direction_cosine", "endpoint_ssim", "action_causal_margin_0_1")
    wins = {field: reports["candidate"][field] < reports["baseline"][field] for field in lower}
    wins.update({field: reports["candidate"][field] > reports["baseline"][field] for field in higher})
    gates = {
        "reference_fidelity_improves": wins["future_ssim"] and wins["endpoint_ssim"],
        "flow_improves": wins["flow_endpoint_error_px_at_224x168"] and wins["flow_direction_cosine"],
        "background_non_regression": reports["candidate"]["background_mad_0_1"] <= 1.05 * reports["baseline"]["background_mad_0_1"],
        "temporal_non_regression": reports["candidate"]["temporal_gradient_mae_0_1"] <= 1.05 * reports["baseline"]["temporal_gradient_mae_0_1"],
        "positive_action_causality": reports["candidate"]["action_causal_margin_0_1"] > 0 and wins["action_causal_margin_0_1"],
        "counterfactual_is_visible": reports["candidate"]["action_counterfactual_separation_0_1"] >= 0.002,
    }
    accepted = all(gates.values())
    comparison = args.comparison_video.expanduser().resolve()
    comparison.parent.mkdir(parents=True, exist_ok=True)
    _compose(cv2, np, reference, videos["baseline"], videos["candidate"], comparison)
    payload = {
        "schema_version": "1.0.0",
        "status": "WORKING" if accepted else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "reference_window": [args.start_frame, args.end_frame],
        "static_background": {"definition": "lowest 60% reference temporal-luminance standard deviation", "threshold": threshold, "pixel_fraction": float(np.mean(static2d))},
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in inputs.items()},
        "metrics": reports,
        "candidate_metric_wins": wins,
        "gates": gates,
        "accepted": accepted,
        "comparison_video": {"path": str(comparison), "sha256": _sha256(comparison)},
        "limitations": [
            "This is one short released-demo window and may overlap the original BWM training corpus.",
            "Optical flow and pixel metrics do not establish 3-D contact physics or task success.",
            "No physical robot was executed in this evaluation.",
        ],
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
