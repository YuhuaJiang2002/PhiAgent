#!/usr/bin/env python3
"""Generate versioned trajectory negatives and measure them in MuJoCo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.data.corruptions import (  # noqa: E402
    CorruptionConfig,
    CorruptionType,
    corrupt_trajectory,
)
from phiagent.data.repair_dataset import RepairExample, write_repair_jsonl  # noqa: E402
from phiagent.data.schema import RobotTrajectory  # noqa: E402
from phiagent.simulation.base import SimulationRequest  # noqa: E402
from phiagent.simulation.mujoco_backend import MujocoBackend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--good-trajectory", type=Path, required=True)
    parser.add_argument("--epl-uri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--corruption",
        action="append",
        choices=[item.value for item in CorruptionType],
        default=None,
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    good = RobotTrajectory.from_json(args.good_trajectory)
    backend = MujocoBackend()
    good_result = backend.simulate(
        SimulationRequest(model_xml=args.model, trajectory=good)
    )
    if not good_result.physically_valid or good_result.task_success is False:
        raise SystemExit("refusing negative generation: source trajectory did not validate")
    requested = args.corruption or [CorruptionType.JOINT_LIMIT.value]
    examples: list[RepairExample] = []
    for index, value in enumerate(requested):
        kind = CorruptionType(value)
        bad = corrupt_trajectory(
            good,
            kind,
            CorruptionConfig(seed=args.seed + index),
        )
        feedback = backend.simulate(
            SimulationRequest(model_xml=args.model, trajectory=bad)
        )
        examples.append(
            RepairExample(
                schema_version="0.1.0",
                example_id=f"{kind.value}-{args.seed + index:06d}",
                epl_uri=args.epl_uri,
                corruption_type=kind.value,
                bad_trajectory=bad,
                simulator_feedback=feedback.to_dict(),
                good_trajectory=good,
            )
        )
    count = write_repair_jsonl(examples, args.output)
    print(json.dumps({"examples": count, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
