#!/usr/bin/env python3
"""Download and verify the official PhiZero hand-transfer reference videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.phizero_reference import prepare_reference_assets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("external/PhiZero-reference"),
        help="ignored directory for the pinned public reference assets",
    )
    args = parser.parse_args()
    manifest = prepare_reference_assets(args.output.expanduser().resolve())
    print(f"PHIZERO_REFERENCE_MANIFEST={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
