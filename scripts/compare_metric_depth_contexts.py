#!/usr/bin/env python3
"""Estimate context sensitivity of learned metric depth on shared video frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    result = {}
    for label, command in (
        ("head", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--short"]),
    ):
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
        )
        result[label] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-context-scale-variation", type=float, default=0.02)
    parser.add_argument("--maximum-relative-depth-residual-p95", type=float, default=0.05)
    args = parser.parse_args()
    if args.maximum_context_scale_variation <= 0 or args.maximum_relative_depth_residual_p95 <= 0:
        raise ValueError("context comparison limits must be positive")
    first_path = args.first.expanduser().resolve()
    second_path = args.second.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not first_path.is_file() or not second_path.is_file():
        raise ValueError("both learned metric-depth artifacts are required")
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite experiment directory: {output_dir}")
    output_dir.mkdir(parents=True)

    import numpy as np

    first = np.load(first_path, allow_pickle=False)
    second = np.load(second_path, allow_pickle=False)
    first_indices = first["source_frame_indices"].astype(int)
    second_lookup = {
        int(frame): index for index, frame in enumerate(second["source_frame_indices"])
    }
    shared = [int(frame) for frame in first_indices if int(frame) in second_lookup]
    if len(shared) < 3:
        raise ValueError("at least three shared source frames are required")
    first_lookup = {int(frame): index for index, frame in enumerate(first_indices)}
    rows = []
    for frame in shared:
        depth_a = first["depth_m"][first_lookup[frame]].astype(np.float64)
        depth_b = second["depth_m"][second_lookup[frame]].astype(np.float64)
        if depth_a.shape != depth_b.shape:
            raise ValueError("learned depth contexts have different image shapes")
        valid = np.isfinite(depth_a) & np.isfinite(depth_b) & (depth_a > 0) & (depth_b > 0)
        if float(np.mean(valid)) < 0.9:
            raise ValueError("shared depth frame has insufficient positive finite support")
        scale_ratio = float(np.median(depth_a[valid] / depth_b[valid]))
        relative = np.abs(depth_a[valid] - scale_ratio * depth_b[valid]) / depth_a[valid]
        rows.append(
            {
                "source_frame": frame,
                "scale_ratio_first_over_second": scale_ratio,
                "relative_depth_residual_median": float(np.median(relative)),
                "relative_depth_residual_p95": float(np.quantile(relative, 0.95)),
            }
        )
    ratios = np.asarray([row["scale_ratio_first_over_second"] for row in rows])
    median_ratio = float(np.median(ratios))
    variation = np.abs(ratios / median_ratio - 1.0)
    context_scale_variation_p95 = float(np.quantile(variation, 0.95))
    relative_residual_p95_max = max(
        row["relative_depth_residual_p95"] for row in rows
    )
    gates = {
        "shared_frames_sufficient": len(shared) >= 3,
        "context_scale_variation_bounded": (
            context_scale_variation_p95 <= args.maximum_context_scale_variation
        ),
        "relative_depth_residual_bounded": (
            relative_residual_p95_max <= args.maximum_relative_depth_residual_p95
        ),
    }
    passed = all(gates.values())
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING" if passed else "PARTIAL",
        "scope": "learned metric-depth context sensitivity only; not absolute calibration accuracy",
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "python": platform.python_version(),
        "git": _git_state(),
        "inputs": {
            "first": {"path": str(first_path), "sha256": _sha256(first_path)},
            "second": {"path": str(second_path), "sha256": _sha256(second_path)},
        },
        "shared_source_frames": shared,
        "per_frame": rows,
        "summary": {
            "median_scale_ratio_first_over_second": median_ratio,
            "context_scale_variation_fraction_p95": context_scale_variation_p95,
            "relative_depth_residual_p95_max": relative_residual_p95_max,
            "maximum_context_scale_variation": args.maximum_context_scale_variation,
            "maximum_relative_depth_residual_p95": args.maximum_relative_depth_residual_p95,
        },
        "gates": gates,
        "passed": passed,
        "limitations": [
            "Agreement between two contexts of the same model bounds context sensitivity, not common-mode absolute scale bias.",
            "An RGB-D sensor, fiducial, known robot link, or known-length scene object is still required for calibrated scale.",
        ],
    }
    path = output_dir / "context-scale-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
