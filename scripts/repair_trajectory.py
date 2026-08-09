#!/usr/bin/env python3
"""Run the deterministic simulate/verify/replan loop on one trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.agent.repair import TrajectoryRepairController  # noqa: E402
from phiagent.data.schema import RobotTrajectory  # noqa: E402
from phiagent.simulation.base import SimulationRequest  # noqa: E402
from phiagent.simulation.mujoco_backend import MujocoBackend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-trajectory", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--object-body", action="append", default=[])
    parser.add_argument("--maximum-iterations", type=int, default=3)
    args = parser.parse_args()
    trajectory = RobotTrajectory.from_json(args.trajectory)
    backend = MujocoBackend()
    controller = TrajectoryRepairController(backend, args.maximum_iterations)

    def request_factory(candidate: RobotTrajectory) -> SimulationRequest:
        return SimulationRequest(
            model_xml=args.model,
            trajectory=candidate,
            object_body_names=tuple(args.object_body),
        )

    outcome = controller.run(trajectory, request_factory)
    outcome.final_trajectory.to_json(args.output_trajectory)
    outcome.to_json(args.trace)
    print(
        json.dumps(
            {
                "accepted": outcome.accepted,
                "trace_steps": len(outcome.trace),
                "output_trajectory": str(args.output_trajectory),
                "trace": str(args.trace),
            },
            indent=2,
        )
    )
    return 0 if outcome.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
