#!/usr/bin/env python3
"""Freeze and validate one lightweight Sharpa adaptation data manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.data.adaptation import load_adaptation_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new manifest path; existing files and directories are never overwritten",
    )
    args = parser.parse_args()
    manifest = load_adaptation_spec(args.spec)
    output = args.output.expanduser().resolve()
    manifest.write_json(output)
    print(f"SHARPA_ADAPTATION_MANIFEST={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
