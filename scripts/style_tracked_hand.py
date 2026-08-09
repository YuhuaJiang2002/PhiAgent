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
    SudoHandConfig,
    SudoRobotConfig,
    apply_graphite_hand_style,
    apply_sudo_hand_style,
    apply_sudo_robot_style,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--hand-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-roi", type=float, nargs=4, required=True)
    parser.add_argument(
        "--style", choices=("graphite", "sudo", "sudo-full"), default="graphite"
    )
    args = parser.parse_args()
    common = {
        "candidate": args.candidate,
        "hand_mask": args.hand_mask,
        "output": args.output,
        "object_roi": NormalizedROI(*args.object_roi),
    }
    if args.style == "graphite":
        metadata = apply_graphite_hand_style(**common, config=GraphiteHandConfig())
    elif args.style == "sudo":
        metadata = apply_sudo_hand_style(**common, config=SudoHandConfig())
    else:
        metadata = apply_sudo_robot_style(
            candidate=args.candidate,
            robot_mask=args.hand_mask,
            output=args.output,
            object_roi=common["object_roi"],
            config=SudoRobotConfig(),
        )
    print(f"METADATA={metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
