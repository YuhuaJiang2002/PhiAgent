"""Versioned repair-training examples generated from simulation feedback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from phiagent.data.schema import RobotTrajectory


@dataclass(frozen=True)
class RepairExample:
    schema_version: str
    example_id: str
    epl_uri: str
    corruption_type: str
    bad_trajectory: RobotTrajectory
    simulator_feedback: dict[str, Any]
    good_trajectory: RobotTrajectory

    def __post_init__(self) -> None:
        if self.schema_version != "0.1.0":
            raise ValueError("unsupported repair-example schema")
        if not self.example_id or not self.epl_uri or not self.corruption_type:
            raise ValueError("repair example identifiers must be non-empty")
        if self.bad_trajectory.embodiment != self.good_trajectory.embodiment:
            raise ValueError("bad and good trajectories must use the same embodiment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "epl_uri": self.epl_uri,
            "corruption_type": self.corruption_type,
            "bad_trajectory": self.bad_trajectory.to_dict(),
            "simulator_feedback": self.simulator_feedback,
            "good_trajectory": self.good_trajectory.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RepairExample:
        return cls(
            schema_version=str(payload["schema_version"]),
            example_id=str(payload["example_id"]),
            epl_uri=str(payload["epl_uri"]),
            corruption_type=str(payload["corruption_type"]),
            bad_trajectory=RobotTrajectory.from_dict(payload["bad_trajectory"]),
            simulator_feedback=dict(payload["simulator_feedback"]),
            good_trajectory=RobotTrajectory.from_dict(payload["good_trajectory"]),
        )


def write_repair_jsonl(examples: Iterable[RepairExample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def read_repair_jsonl(path: Path) -> tuple[RepairExample, ...]:
    examples: list[RepairExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                examples.append(RepairExample.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid repair example at line {line_number}") from exc
    return tuple(examples)

