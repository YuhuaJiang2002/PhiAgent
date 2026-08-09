#!/usr/bin/env python3
"""Replay a canonical robot trajectory in MuJoCo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.data.schema import RobotTrajectory  # noqa: E402
from phiagent.simulation.base import ObjectPositionGoal, SimulationRequest  # noqa: E402
from phiagent.simulation.mujoco_backend import MujocoBackend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--object-body", action="append", default=[])
    parser.add_argument(
        "--required-contact",
        action="append",
        default=[],
        metavar="GEOM_A,GEOM_B",
    )
    parser.add_argument(
        "--forbidden-contact",
        action="append",
        default=[],
        metavar="GEOM_A,GEOM_B",
    )
    parser.add_argument(
        "--object-goal",
        action="append",
        default=[],
        metavar="BODY,X,Y,Z,TOLERANCE_M",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--camera")
    args = parser.parse_args()

    def contact_pairs(values: list[str]) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        for value in values:
            parts = tuple(part.strip() for part in value.split(","))
            if len(parts) != 2:
                parser.error(f"invalid contact pair {value!r}; expected GEOM_A,GEOM_B")
            pairs.append(parts)  # type: ignore[arg-type]
        return tuple(pairs)

    goals: list[ObjectPositionGoal] = []
    for value in args.object_goal:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 5:
            parser.error(
                f"invalid object goal {value!r}; expected BODY,X,Y,Z,TOLERANCE_M"
            )
        try:
            goals.append(
                ObjectPositionGoal(
                    parts[0],
                    (float(parts[1]), float(parts[2]), float(parts[3])),
                    float(parts[4]),
                )
            )
        except ValueError as exc:
            parser.error(str(exc))
    trajectory = RobotTrajectory.from_json(args.trajectory)
    result = MujocoBackend(camera=args.camera).simulate(
        SimulationRequest(
            model_xml=args.model,
            trajectory=trajectory,
            object_body_names=tuple(args.object_body),
            required_contact_pairs=contact_pairs(args.required_contact),
            forbidden_contact_pairs=contact_pairs(args.forbidden_contact),
            object_position_goals=tuple(goals),
            render_output=args.video,
        )
    )
    result.to_json(args.output_json)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    accepted = result.physically_valid and result.task_success is not False
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
