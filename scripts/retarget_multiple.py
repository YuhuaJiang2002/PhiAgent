#!/usr/bin/env python3
"""Map one EPL sequence to multiple robot embodiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.physical_language.schema import EPLSequence  # noqa: E402
from phiagent.retargeting.base import LinearRetargetingConfig  # noqa: E402
from phiagent.retargeting.multi import retarget_multiple  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epl", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.config) < 2:
        parser.error("provide at least two --config files")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epl = EPLSequence.from_json(args.epl)
    configs = tuple(
        LinearRetargetingConfig.from_dict(json.loads(path.read_text()))
        for path in args.config
    )
    result = retarget_multiple(epl, configs)
    summary = {
        "schema_version": "0.1.0",
        "source_epl": str(args.epl.resolve()),
        "canonical_dimension": result.canonical_dimension,
        "embodiments": {},
    }
    for name, retargeted in result.results.items():
        trajectory_path = args.output_dir / f"{name}.trajectory.json"
        retargeted.trajectory.to_json(trajectory_path)
        summary["embodiments"][name] = {
            "trajectory": str(trajectory_path),
            "reachability_failures": list(retargeted.reachability_failures),
            "canonical_actions": [
                action.to_dict() for action in result.canonical_actions[name]
            ],
        }
    summary_path = args.output_dir / "multi_embodiment.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(not item.reachability_failures for item in result.results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
