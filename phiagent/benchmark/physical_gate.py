"""Normalize per-step simulator traces into fail-closed L4 evidence.

The input contract is deliberately backend-neutral. Isaac Lab, MuJoCo, and
other simulators may produce the trace, but every required safety signal must
be present for every sampled step before ``physical_gate_complete`` can be
true. Intended task contacts are reported separately from forbidden collisions.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from phiagent.benchmark.schema import SimulationEvidence


TRACE_SCHEMA_VERSION = "0.2.0"
SAMPLE_FLAGS = (
    "ik_success",
    "joint_limit_violation",
    "velocity_violation",
    "forbidden_collision",
    "singularity",
)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simulation_evidence_from_trace(
    payload: Mapping[str, Any], *, trace_sha256: str
) -> SimulationEvidence:
    """Validate one trace and derive every rate instead of trusting summaries."""

    if payload.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError("physical gate trace schema_version must be 0.2.0")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("physical gate trace requires at least two samples")
    timestamps: list[float] = []
    flags: dict[str, list[bool]] = {name: [] for name in SAMPLE_FLAGS}
    for index, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, Mapping):
            raise ValueError(f"samples[{index}] must be an object")
        if set(SAMPLE_FLAGS) - set(raw_sample):
            missing = sorted(set(SAMPLE_FLAGS) - set(raw_sample))
            raise ValueError(f"samples[{index}] lacks required flags: {missing}")
        timestamp = float(raw_sample["timestamp_s"])
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(f"samples[{index}].timestamp_s must be finite and non-negative")
        timestamps.append(timestamp)
        for name in SAMPLE_FLAGS:
            flags[name].append(_boolean(raw_sample[name], f"samples[{index}].{name}"))
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("physical gate timestamps must be strictly increasing")

    stage_results = payload.get("stage_results")
    contact_results = payload.get("contact_results")
    if not isinstance(stage_results, list) or not stage_results:
        raise ValueError("physical gate trace requires non-empty stage_results")
    if not isinstance(contact_results, list) or not contact_results:
        raise ValueError("physical gate trace requires non-empty contact_results")
    stages = [_boolean(value, f"stage_results[{index}]") for index, value in enumerate(stage_results)]
    contacts = [
        _boolean(value, f"contact_results[{index}]")
        for index, value in enumerate(contact_results)
    ]
    task_success = _boolean(payload.get("task_success"), "task_success")

    ik_success_rate = _rate(flags["ik_success"])
    joint_rate = _rate(flags["joint_limit_violation"])
    velocity_rate = _rate(flags["velocity_violation"])
    collision_rate = _rate(flags["forbidden_collision"])
    singularity_rate = _rate(flags["singularity"])
    physically_valid = (
        ik_success_rate == 1.0
        and joint_rate == 0.0
        and velocity_rate == 0.0
        and collision_rate == 0.0
        and singularity_rate == 0.0
    )
    return SimulationEvidence.from_dict(
        {
            "backend": payload["backend"],
            "attempted": True,
            "physical_gate_complete": True,
            "physically_valid": physically_valid,
            "task_success": task_success,
            "stage_success_rate": _rate(stages),
            "contact_success_rate": _rate(contacts),
            "ik_success_rate": ik_success_rate,
            "joint_limit_violation_rate": joint_rate,
            "velocity_violation_rate": velocity_rate,
            "collision_rate": collision_rate,
            "singularity_rate": singularity_rate,
            "source_revision": payload["source_revision"],
            "episode_id": payload["episode_id"],
            "initial_state_id": payload["initial_state_id"],
            "seed": payload["seed"],
            "artifact_hashes": {"physical_gate_trace": trace_sha256},
        }
    )


def simulation_evidence_from_trace_file(path: Path) -> SimulationEvidence:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("physical gate trace must be a JSON object")
    return simulation_evidence_from_trace(payload, trace_sha256=_sha256(source))
