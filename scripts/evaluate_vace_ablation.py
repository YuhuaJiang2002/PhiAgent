#!/usr/bin/env python3
"""Evaluate matched VACE outputs on one held-out target video."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(ffmpeg: Path, reference: Path, candidate: Path, filter_graph: str) -> str:
    completed = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "info",
            "-i",
            str(reference),
            "-i",
            str(candidate),
            "-lavfi",
            filter_graph,
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stderr


def _metric(text: str, pattern: str, label: str) -> float:
    matches = re.findall(pattern, text)
    if not matches:
        raise ValueError(f"ffmpeg did not report {label}")
    value = float(matches[-1])
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "VIDEO"))
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.candidate:
        raise ValueError("at least one named candidate is required")
    target = args.target.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if not target.is_file() or not ffmpeg.is_file():
        raise ValueError("target and ffmpeg must exist")
    results = {}
    for name, raw_path in args.candidate:
        candidate = Path(raw_path).expanduser().resolve()
        if not candidate.is_file():
            raise ValueError(f"candidate does not exist: {candidate}")
        standard = _run(ffmpeg, target, candidate, "ssim")
        psnr = _run(ffmpeg, target, candidate, "psnr")
        edge = _run(
            ffmpeg,
            target,
            candidate,
            "[0:v]format=gray,edgedetect[e0];"
            "[1:v]format=gray,edgedetect[e1];[e0][e1]ssim",
        )
        results[name] = {
            "path": str(candidate),
            "sha256": _sha256(candidate),
            "ssim": _metric(standard, r"SSIM .* All:([0-9.]+)", "SSIM"),
            "psnr_db": _metric(psnr, r"average:([0-9.]+)", "PSNR"),
            "edge_ssim": _metric(edge, r"SSIM .* All:([0-9.]+)", "edge SSIM"),
        }
    payload = {
        "schema_version": "1.0.0",
        "scope": "authorized_synthetic_held_out_development_only",
        "target": str(target),
        "target_sha256": _sha256(target),
        "results": results,
        "limitations": [
            "One held-out procedural clip is not evidence of real-video quality.",
            "Pixel metrics measure reconstruction, not physical grasp correctness.",
        ],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
