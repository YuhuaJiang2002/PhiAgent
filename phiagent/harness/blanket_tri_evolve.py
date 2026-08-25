"""Dependency-light Tri-Evolve contracts for collaborative blanket folding.

The module separates three concerns: a language decomposition, an explicit
camera-frame difficulty ladder, and fail-closed candidate promotion. Generated
RGB remains proposal evidence and cannot satisfy metric, force, safety, or
recorded-execution gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
PASS = "PASS"
FAIL = "FAIL"
UNAVAILABLE = "UNAVAILABLE"
ALLOWED_GATE_STATES = {PASS, FAIL, UNAVAILABLE}
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BlanketDifficulty:
    """Observable camera-frame task difficulty; never metric robot geometry."""

    environment_id: str
    level: str
    coordinate_frame: str
    obstacle_count: int
    quilt_rotation_degrees: float
    off_table_overhang_fraction: float
    self_occlusion_fraction: float
    required_action_stages: int
    required_regrasps: int
    terminal_transport_fraction: float

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.environment_id):
            raise ValueError("environment_id must be a safe lowercase identifier")
        if self.level not in {"E0", "E1", "E2"}:
            raise ValueError("blanket difficulty level must be E0, E1, or E2")
        if not self.coordinate_frame.startswith("camera:"):
            raise ValueError("blanket difficulty requires a named camera frame")
        if self.obstacle_count < 0 or self.required_action_stages < 1:
            raise ValueError("obstacle and action-stage counts must be non-negative")
        if self.required_regrasps < 0:
            raise ValueError("required_regrasps must be non-negative")
        if not math.isfinite(self.quilt_rotation_degrees) or not (
            0.0 <= self.quilt_rotation_degrees <= 90.0
        ):
            raise ValueError("quilt_rotation_degrees must be in [0, 90]")
        for name in (
            "off_table_overhang_fraction",
            "self_occlusion_fraction",
            "terminal_transport_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and normalized to [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        payload = {"schema_version": SCHEMA_VERSION, **asdict(self)}
        return {**payload, "difficulty_sha256": canonical_sha256(payload)}


_DIFFICULTY_FIELDS = (
    "obstacle_count",
    "quilt_rotation_degrees",
    "off_table_overhang_fraction",
    "self_occlusion_fraction",
    "required_action_stages",
    "required_regrasps",
    "terminal_transport_fraction",
)


def compare_difficulty(
    incumbent: BlanketDifficulty,
    challenger: BlanketDifficulty,
) -> dict[str, Any]:
    """Require a monotone challenge increase without silently easing a dimension."""

    if incumbent.coordinate_frame != challenger.coordinate_frame:
        raise ValueError("difficulty comparison requires the same named camera frame")
    deltas = {
        name: float(getattr(challenger, name) - getattr(incumbent, name))
        for name in _DIFFICULTY_FIELDS
    }
    regressed = sorted(name for name, value in deltas.items() if value < 0.0)
    increased = sorted(name for name, value in deltas.items() if value > 0.0)
    passed = not regressed and len(increased) >= 4
    return {
        "passed": passed,
        "reason": "all_contract_conditions_pass" if passed else "quality_regression",
        "increased_dimensions": increased,
        "regressed_dimensions": regressed,
        "deltas": deltas,
        "incumbent_sha256": incumbent.to_dict()["difficulty_sha256"],
        "challenger_sha256": challenger.to_dict()["difficulty_sha256"],
    }


def evaluate_hard_gates(
    required_gate_ids: Sequence[str],
    observations: Mapping[str, str],
) -> dict[str, Any]:
    """Fail closed: missing, FAIL, or UNAVAILABLE evidence rejects a candidate."""

    required = tuple(str(item).strip() for item in required_gate_ids)
    if not required or any(not item for item in required):
        raise ValueError("required hard-gate ids must be non-empty")
    if len(set(required)) != len(required):
        raise ValueError("required hard-gate ids must be unique")
    unknown_states = sorted(
        gate_id
        for gate_id, state in observations.items()
        if state not in ALLOWED_GATE_STATES
    )
    if unknown_states:
        raise ValueError(f"unsupported gate states for: {', '.join(unknown_states)}")
    states = {gate_id: observations.get(gate_id, UNAVAILABLE) for gate_id in required}
    failed = [gate_id for gate_id, state in states.items() if state != PASS]
    return {
        "passed": not failed,
        "reason": "all_contract_conditions_pass" if not failed else "hard_gate_failed",
        "failed_or_unavailable_gate_ids": failed,
        "gate_states": states,
        "aggregate_override_allowed": False,
    }


def build_failure_repair_directive(failed_gate_ids: Sequence[str]) -> str:
    """Turn exact failure IDs into a repair directive without changing thresholds."""

    failed = tuple(dict.fromkeys(str(item).strip() for item in failed_gate_ids))
    if not failed or any(not item for item in failed):
        raise ValueError("failure-aware scaling requires explicit failed gate ids")
    return (
        "Failure-aware Tri-Evolve repair. Preserve every frozen threshold and task "
        "phase. Repair only these failed gates: "
        + ", ".join(failed)
        + ". Missing evidence still rejects; no aggregate score may override a gate."
    )


def physical_promotion_decision(
    *,
    proposal_ready: bool,
    physical_gate_states: Mapping[str, bool],
    independent_source_groups: int,
) -> dict[str, Any]:
    """Apply the project promotion contract to a generated proposal."""

    if proposal_ready and not physical_gate_states.get("absolute_scale_verified", False):
        return {"promote": False, "reason": "proposal_not_physical_calibration"}
    if not physical_gate_states or not all(physical_gate_states.values()):
        return {"promote": False, "reason": "hard_gate_failed"}
    if independent_source_groups < 2:
        return {"promote": False, "reason": "insufficient_independent_groups"}
    return {"promote": True, "reason": "all_contract_conditions_pass"}
