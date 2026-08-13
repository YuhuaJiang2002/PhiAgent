#!/usr/bin/env python3
"""Compare two exact behavior-eval manifests under a no-regression contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.agent.foundation_contact_skill_eval import compare_behavior_manifests  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"comparison already exists: {args.output}")
    incumbent = json.loads(args.incumbent.read_text())
    challenger = json.loads(args.challenger.read_text())
    decision = compare_behavior_manifests(incumbent, challenger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["promote_behavior_skill"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
