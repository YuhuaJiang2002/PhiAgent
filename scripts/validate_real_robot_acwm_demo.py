#!/usr/bin/env python3
"""Validate and hash genuine pre-prediction/physical-execution AC-WM trials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.real_robot import (  # noqa: E402
    RealRobotTrialEvidence,
    compile_real_robot_demo,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-trials", type=int, default=3)
    args = parser.parse_args()
    trials = []
    for path in args.trial:
        resolved = path.expanduser().resolve()
        payload = json.loads(resolved.read_text())
        trials.append(RealRobotTrialEvidence.from_dict(payload, root=resolved.parent))
    result = compile_real_robot_demo(tuple(trials), minimum_trials=args.minimum_trials)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"real-robot demo evidence already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
