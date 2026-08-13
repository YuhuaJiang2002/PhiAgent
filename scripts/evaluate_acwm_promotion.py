#!/usr/bin/env python3
"""Evaluate an AC-WM candidate against every declared paired baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.promotion import evaluate_promotion  # noqa: E402

DEFAULT_SUITES = {
    "worldarena_test": (
        "action_following",
        "physics_adherence",
        "visual_consistency",
    ),
    "cross_embodiment_test": ("action_following", "embodiment_consistency"),
    "real_robot_test": ("task_success", "safety_violation_free"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-trials", type=int, default=20)
    parser.add_argument("--minimum-gain", type=float, default=0.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.expanduser().resolve().read_text())
    baselines = [
        json.loads(path.expanduser().resolve().read_text()) for path in args.baseline
    ]
    result = evaluate_promotion(
        candidate,
        baselines,
        required_suites=DEFAULT_SUITES,
        minimum_trials=args.minimum_trials,
        minimum_gain=args.minimum_gain,
        bootstrap_iterations=args.bootstrap_iterations,
        confidence=args.confidence,
        seed=args.seed,
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"promotion evidence already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
