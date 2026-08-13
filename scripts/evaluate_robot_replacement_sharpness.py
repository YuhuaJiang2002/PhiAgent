#!/usr/bin/env python3
"""Compare two full robot-replacement clips on one camera-frame mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _measure(cv2, np, path: Path, mask):
    capture = cv2.VideoCapture(str(path))
    sharpness = []
    transitions = []
    previous = None
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
            sharpness.append(float(np.mean(laplacian[mask])))
            small = cv2.resize(gray, (256, 144), interpolation=cv2.INTER_AREA)
            if previous is not None:
                transitions.append(float(np.mean(cv2.absdiff(small, previous))))
            previous = small
            frames += 1
    finally:
        capture.release()
    median_transition = float(np.median(transitions))
    maximum_index = int(np.argmax(transitions)) + 1
    return {
        "frames": frames,
        "mean_masked_absolute_laplacian": float(np.mean(sharpness)),
        "p10_masked_absolute_laplacian": float(np.percentile(sharpness, 10)),
        "minimum_masked_absolute_laplacian": float(np.min(sharpness)),
        "median_full_frame_transition_energy": median_transition,
        "maximum_full_frame_transition_energy": float(np.max(transitions)),
        "maximum_full_frame_transition_ratio": float(
            np.max(transitions) / max(median_transition, 1e-6)
        ),
        "maximum_transition_frame": maximum_index,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np

    for path in (args.baseline, args.candidate, args.mask):
        if not path.expanduser().resolve().is_file():
            raise ValueError(f"input does not exist: {path}")
    candidate_capture = cv2.VideoCapture(str(args.candidate))
    width = int(candidate_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(candidate_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    candidate_capture.release()
    raw_mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if raw_mask is None:
        raise RuntimeError("cannot decode mask")
    raw_mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    raw_mask = cv2.dilate(
        (raw_mask >= 127).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    mask = raw_mask > 0
    baseline = _measure(cv2, np, args.baseline.expanduser().resolve(), mask)
    candidate = _measure(cv2, np, args.candidate.expanduser().resolve(), mask)
    if baseline["frames"] != candidate["frames"]:
        raise RuntimeError("comparison clips have different decoded frame counts")
    result = {
        "schema_version": "1.0.0",
        "baseline": {"path": str(args.baseline.expanduser().resolve()), **baseline},
        "candidate": {"path": str(args.candidate.expanduser().resolve()), **candidate},
        "comparison": {
            "mean_sharpness_ratio": candidate["mean_masked_absolute_laplacian"]
            / max(baseline["mean_masked_absolute_laplacian"], 1e-6),
            "p10_sharpness_ratio": candidate["p10_masked_absolute_laplacian"]
            / max(baseline["p10_masked_absolute_laplacian"], 1e-6),
            "candidate_transition_ratio_passed": candidate[
                "maximum_full_frame_transition_ratio"
            ]
            <= 4.0,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
