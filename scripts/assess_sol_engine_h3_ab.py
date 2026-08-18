#!/usr/bin/env python3
"""Fail-closed assessment for a matched MiniMax-H3 dense/Sol-Engine run."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acceleration.sol_engine import assess_h3_ab_result  # noqa: E402


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-benchmark", type=Path, required=True)
    parser.add_argument("--sol-benchmark", type=Path, required=True)
    parser.add_argument("--quality-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-speedup", type=float, default=1.15)
    args = parser.parse_args()
    result = assess_h3_ab_result(
        _json(args.dense_benchmark),
        _json(args.sol_benchmark),
        _json(args.quality_evidence),
        minimum_speedup=args.minimum_speedup,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    print(json.dumps(asdict(result), sort_keys=True))
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
