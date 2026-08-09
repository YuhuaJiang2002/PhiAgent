#!/usr/bin/env python3
"""Apply a graphite material to a SAM2-tracked robot hand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.evaluation.object_instance import NormalizedROI  # noqa: E402
from phiagent.rendering.hand_style import (  # noqa: E402
    GraphiteHandConfig,
    apply_graphite_hand_style,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--hand-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-roi", type=float, nargs=4, required=True)
    args = parser.parse_args()
    metadata = apply_graphite_hand_style(
        candidate=args.candidate,
        hand_mask=args.hand_mask,
        output=args.output,
        object_roi=NormalizedROI(*args.object_roi),
        config=GraphiteHandConfig(),
    )
    print(f"METADATA={metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
