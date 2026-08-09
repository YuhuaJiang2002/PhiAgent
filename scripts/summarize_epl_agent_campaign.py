#!/usr/bin/env python3
"""Validate and summarize matched EPL-conditioned policy runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.training.campaign import summarize_campaign  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-mean-gain", type=float, default=0.05)
    args = parser.parse_args()
    root = args.campaign_root.expanduser().resolve()
    summary = summarize_campaign(root, args.minimum_mean_gain)
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "summary.json"
    )
    if output.exists():
        raise FileExistsError(f"campaign summary already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), **summary}, indent=2, sort_keys=True))
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
