#!/usr/bin/env python3
"""Replay and verify a trajectory into a Cosmos-ready MuJoCo control bundle."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.data.schema import RobotTrajectory  # noqa: E402
from phiagent.rendering.control import produce_mujoco_control_bundle  # noqa: E402
from phiagent.simulation.base import ObjectPositionGoal, SimulationRequest  # noqa: E402


def _contact_pairs(
    parser: argparse.ArgumentParser,
    values: list[str],
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        parts = tuple(part.strip() for part in value.split(","))
        if len(parts) != 2:
            parser.error(f"invalid contact pair {value!r}; expected GEOM_A,GEOM_B")
        pairs.append((parts[0], parts[1]))
    return tuple(pairs)


def _goals(
    parser: argparse.ArgumentParser,
    values: list[str],
) -> tuple[ObjectPositionGoal, ...]:
    goals: list[ObjectPositionGoal] = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 5:
            parser.error(f"invalid object goal {value!r}; expected BODY,X,Y,Z,TOLERANCE_M")
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
    return tuple(goals)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--object-body", action="append", required=True)
    parser.add_argument("--required-contact", action="append", default=[])
    parser.add_argument("--forbidden-contact", action="append", default=[])
    parser.add_argument("--object-goal", action="append", default=[])
    parser.add_argument("--camera")
    parser.add_argument("--robot-base-name", required=True)
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/control"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, choices=(10, 16, 24, 30), default=30)
    args = parser.parse_args()

    bundle = produce_mujoco_control_bundle(
        SimulationRequest(
            model_xml=args.model,
            trajectory=RobotTrajectory.from_json(args.trajectory),
            object_body_names=tuple(args.object_body),
            required_contact_pairs=_contact_pairs(parser, args.required_contact),
            forbidden_contact_pairs=_contact_pairs(parser, args.forbidden_contact),
            object_position_goals=_goals(parser, args.object_goal),
            render_width=args.width,
            render_height=args.height,
            render_fps=args.fps,
        ),
        camera=args.camera,
        experiment_root=args.experiment_root,
        robot_base_name=args.robot_base_name,
    )
    print(json.dumps({key: str(value) for key, value in asdict(bundle).items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
