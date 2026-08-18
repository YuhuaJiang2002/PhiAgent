#!/usr/bin/env python3
"""Run the fail-closed H3 dense/Sol video evaluation after both jobs finish.

The program is intentionally an orchestrator rather than a hidden success path:
the image/temporal comparator supplies only its own evidence, and the final
assessor still requires independent action, physical, and human-review gates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--minimum-speedup", type=float, default=1.15)
    return parser


def _run(command: list[str]) -> int:
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main() -> int:
    args = _parser().parse_args()
    root = args.run_root.expanduser().resolve()
    dense, sol = root / "dense", root / "sol_fullopt_exact"
    comparator = Path(__file__).with_name("evaluate_sol_engine_h3_video_pair.py")
    assessor = Path(__file__).with_name("assess_sol_engine_h3_ab.py")
    evidence = root / "quality_evidence.auto.json"
    acceptance = root / "acceptance.auto.json"
    compare_status = _run(
        [
            sys.executable,
            str(comparator),
            "--dense-benchmark",
            str(dense / "benchmark.json"),
            "--sol-benchmark",
            str(sol / "benchmark.json"),
            "--dense-video",
            str(dense / "out.mp4"),
            "--sol-video",
            str(sol / "out.mp4"),
            "--output",
            str(evidence),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--max-frames",
            str(args.max_frames),
        ]
    )
    if compare_status != 0:
        return compare_status
    assessment_status = _run(
        [
            sys.executable,
            str(assessor),
            "--dense-benchmark",
            str(dense / "benchmark.json"),
            "--sol-benchmark",
            str(sol / "benchmark.json"),
            "--quality-evidence",
            str(evidence),
            "--output",
            str(acceptance),
            "--minimum-speedup",
            str(args.minimum_speedup),
        ]
    )
    print(
        json.dumps(
            {
                "quality_evidence": str(evidence),
                "acceptance": str(acceptance),
                "assessment_exit": assessment_status,
            },
            sort_keys=True,
        )
    )
    return assessment_status


if __name__ == "__main__":
    raise SystemExit(main())
