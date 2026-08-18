#!/usr/bin/env python3
"""Compute deterministic frame/temporal evidence for a dense/Sol H3 pair.

This is deliberately a quality *input* to the final assessor.  It cannot
assert action or physical correctness from generic T2VA pixels, so those gates
remain false until a task-specific evaluator and review supply evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acceleration.sol_engine import validate_matched_h3_benchmarks  # noqa: E402


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _frames(path: Path, *, width: int, height: int, max_frames: int):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - GPU runner has optional video deps
        raise RuntimeError("opencv-python and numpy are required for video comparison") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode video: {path}")
    frames = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    capture.release()
    if len(frames) < 3:
        raise ValueError(f"too few decoded frames from {path}: {len(frames)}")
    return frames


def _ssim(left, right) -> float:
    c1, c2 = 6.5025, 58.5225
    mean_left, mean_right = float(left.mean()), float(right.mean())
    variance_left, variance_right = float(left.var()), float(right.var())
    covariance = float(((left - mean_left) * (right - mean_right)).mean())
    return ((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / (
        (mean_left**2 + mean_right**2 + c1) * (variance_left + variance_right + c2)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-benchmark", type=Path, required=True)
    parser.add_argument("--sol-benchmark", type=Path, required=True)
    parser.add_argument("--dense-video", type=Path, required=True)
    parser.add_argument("--sol-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--minimum-psnr", type=float, default=25.0)
    parser.add_argument("--minimum-ssim", type=float, default=0.85)
    parser.add_argument("--maximum-temporal-relative-error", type=float, default=0.20)
    args = parser.parse_args()
    dense_benchmark, sol_benchmark = _json(args.dense_benchmark), _json(args.sol_benchmark)
    matched_inputs, mismatch_reasons = validate_matched_h3_benchmarks(
        dense_benchmark, sol_benchmark
    )
    dense, sol = _frames(args.dense_video, width=args.width, height=args.height, max_frames=args.max_frames), _frames(
        args.sol_video, width=args.width, height=args.height, max_frames=args.max_frames
    )
    count = min(len(dense), len(sol))
    dense, sol = dense[:count], sol[:count]
    import numpy as np

    mse = float(np.mean([(left - right) ** 2 for left, right in zip(dense, sol)]))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0**2) / mse)
    mean_ssim = float(np.mean([_ssim(left, right) for left, right in zip(dense, sol)]))
    dense_motion = np.asarray([np.mean(np.abs(right - left)) for left, right in zip(dense, dense[1:])])
    sol_motion = np.asarray([np.mean(np.abs(right - left)) for left, right in zip(sol, sol[1:])])
    temporal_relative_error = float(np.mean(np.abs(dense_motion - sol_motion)) / max(1e-6, float(np.mean(dense_motion))))
    automated_quality = psnr >= args.minimum_psnr and mean_ssim >= args.minimum_ssim
    temporal = temporal_relative_error <= args.maximum_temporal_relative_error
    evidence = {
        "source": "phiagent-sol-engine-h3-frame-temporal-comparator-v1",
        "matched_inputs": matched_inputs,
        "automated_quality_passed": automated_quality,
        "temporal_consistency_passed": temporal,
        "action_consistency_passed": False,
        "physical_gate_passed": False,
        "human_review_passed": False,
        "details": {
            "mismatch_reasons": list(mismatch_reasons),
            "frames_compared": count,
            "psnr": psnr,
            "ssim": mean_ssim,
            "temporal_relative_error": temporal_relative_error,
            "thresholds": {
                "minimum_psnr": args.minimum_psnr,
                "minimum_ssim": args.minimum_ssim,
                "maximum_temporal_relative_error": args.maximum_temporal_relative_error,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
