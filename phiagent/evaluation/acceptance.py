"""Strict, dependency-free acceptance and experiment statistics.

This module deliberately keeps ranking scores separate from acceptance: a
candidate is accepted only when every required gate passes (and, when
configured, human review is explicitly approved).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping

_WILSON_Z_95 = 1.959963984540054
_GROUPING_KEYS = frozenset({"scene", "action", "object", "embodiment", "seed", "stable_id"})


def _finite_unit_interval(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0, 1]")


@dataclass(frozen=True)
class GateRequirement:
    """One named hard acceptance gate, with an optional ranking weight."""

    name: str
    threshold: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("gate name must be non-empty")
        _finite_unit_interval(self.threshold, f"{self.name} threshold")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (float, int))
            or not math.isfinite(self.weight)
            or self.weight <= 0.0
        ):
            raise ValueError(f"{self.name} weight must be finite and positive")


@dataclass(frozen=True)
class AcceptanceContract:
    """Configurable all-gates acceptance contract."""

    required_gates: tuple[GateRequirement, ...]
    human_review_required: bool = True

    def __post_init__(self) -> None:
        gates = tuple(self.required_gates)
        if not gates:
            raise ValueError("acceptance contract requires at least one named gate")
        names = tuple(gate.name for gate in gates)
        if len(names) != len(set(names)):
            raise ValueError("acceptance contract contains duplicate gate names")
        if not isinstance(self.human_review_required, bool):
            raise ValueError("human_review_required must be a boolean")
        object.__setattr__(self, "required_gates", gates)

    @classmethod
    def from_thresholds(
        cls,
        thresholds: Mapping[str, float],
        *,
        weights: Mapping[str, float] | None = None,
        human_review_required: bool = True,
    ) -> "AcceptanceContract":
        """Create a contract from named thresholds with optional ranking weights."""

        weights = {} if weights is None else weights
        unknown_weights = set(weights) - set(thresholds)
        if unknown_weights:
            raise ValueError(f"weights name unknown gates: {sorted(unknown_weights)}")
        return cls(
            tuple(
                GateRequirement(name, threshold, weights.get(name, 1.0))
                for name, threshold in sorted(thresholds.items())
            ),
            human_review_required=human_review_required,
        )


@dataclass(frozen=True)
class IndependentEvaluationUnit:
    """The independent unit of analysis, never an individual frame or view."""

    scene: str | None = None
    action: str | None = None
    object: str | None = None
    embodiment: str | None = None
    seed: int | str | None = None
    stable_id: str | None = None
    stable_unit_id: str | None = None

    def __post_init__(self) -> None:
        stable_values = tuple(
            value for value in (self.stable_id, self.stable_unit_id) if value is not None
        )
        if len(stable_values) > 1:
            raise ValueError("provide only one of stable_id or stable_unit_id")
        if stable_values:
            if not isinstance(stable_values[0], str) or not stable_values[0].strip():
                raise ValueError("stable unit id must be a non-empty string")
            return
        components = (self.scene, self.action, self.object, self.embodiment, self.seed)
        if any(value is None for value in components):
            raise ValueError(
                "an independent unit requires scene, action, object, embodiment, and seed "
                "or a stable unit id"
            )
        for name, value in zip(("scene", "action", "object", "embodiment"), components[:4]):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, str)):
            raise ValueError("seed must be an integer or string")
        if isinstance(self.seed, str) and not self.seed.strip():
            raise ValueError("seed must be non-empty")

    @property
    def unit_id(self) -> str:
        """A deterministic identity used to reject repeated experimental units."""

        stable_id = self.stable_id if self.stable_id is not None else self.stable_unit_id
        if stable_id is not None:
            return f"stable:{stable_id}"
        return "components:" + json.dumps(
            [self.scene, self.action, self.object, self.embodiment, self.seed],
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def group_value(self, grouping_key: str) -> str:
        """Return a group label, rejecting unknown or unavailable keys precisely."""

        if grouping_key not in _GROUPING_KEYS:
            raise ValueError(f"unknown grouping key: {grouping_key!r}")
        if grouping_key == "stable_id":
            value = self.stable_id if self.stable_id is not None else self.stable_unit_id
        else:
            value = getattr(self, grouping_key)
        if value is None:
            raise ValueError(
                f"grouping key {grouping_key!r} is unavailable for unit {self.unit_id!r}"
            )
        return str(value)


EvaluationUnit = IndependentEvaluationUnit


@dataclass(frozen=True)
class EvaluationRecord:
    """Scores and review status for exactly one independent evaluation unit."""

    unit: IndependentEvaluationUnit
    gate_scores: Mapping[str, float]
    human_review: bool | None = None

    def __post_init__(self) -> None:
        scores = dict(self.gate_scores)
        for name, value in scores.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("gate score names must be non-empty strings")
            _finite_unit_interval(value, f"{name} score")
        if self.human_review is not None and not isinstance(self.human_review, bool):
            raise ValueError("human_review must be True, False, or None")
        object.__setattr__(self, "gate_scores", MappingProxyType(dict(sorted(scores.items()))))


@dataclass(frozen=True)
class GateResult:
    """The pass/fail result of a single contract gate."""

    name: str
    threshold: float
    score: float | None
    passed: bool
    failure_reason: str | None = None


@dataclass(frozen=True)
class AcceptanceDecision:
    """A fail-closed decision and separate ranking diagnostics for one unit."""

    unit: IndependentEvaluationUnit
    accepted: bool
    gate_results: tuple[GateResult, ...]
    human_review: bool | None
    mean_score: float | None
    weighted_score: float | None
    validation_errors: tuple[str, ...] = ()

    @property
    def gate_failure_names(self) -> tuple[str, ...]:
        return tuple(result.name for result in self.gate_results if not result.passed)


def evaluate_acceptance(
    contract: AcceptanceContract, record: EvaluationRecord
) -> AcceptanceDecision:
    """Evaluate all hard gates without allowing a mean score to compensate."""

    results: list[GateResult] = []
    validation_errors: list[str] = []
    valid_scores: list[float] = []
    weighted_sum = 0.0
    weight_total = 0.0
    for gate in contract.required_gates:
        score = record.gate_scores.get(gate.name)
        if score is None:
            validation_errors.append(f"missing required gate: {gate.name}")
            results.append(GateResult(gate.name, gate.threshold, None, False, "missing"))
            continue
        passed = score >= gate.threshold
        results.append(
            GateResult(
                gate.name,
                gate.threshold,
                score,
                passed,
                None if passed else "below_threshold",
            )
        )
        valid_scores.append(score)
        weighted_sum += score * gate.weight
        weight_total += gate.weight

    all_scores_valid = len(valid_scores) == len(contract.required_gates)
    mean_score = sum(valid_scores) / len(valid_scores) if all_scores_valid else None
    weighted_score = weighted_sum / weight_total if all_scores_valid else None
    gates_passed = all(result.passed for result in results)
    human_passed = not contract.human_review_required or record.human_review is True
    if contract.human_review_required:
        if record.human_review is None:
            validation_errors.append("required human review is pending")
        elif record.human_review is False:
            validation_errors.append("required human review rejected")
    return AcceptanceDecision(
        unit=record.unit,
        accepted=gates_passed and human_passed,
        gate_results=tuple(results),
        human_review=record.human_review,
        mean_score=mean_score,
        weighted_score=weighted_score,
        validation_errors=tuple(validation_errors),
    )


@dataclass(frozen=True)
class ValidTransferRate:
    """Exact acceptance counts, proportion, and Wilson 95% interval."""

    passed: int
    total: int
    rate: float = field(init=False)
    wilson_95: tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.passed, bool) or isinstance(self.total, bool):
            raise ValueError("passed and total must be integers")
        if not isinstance(self.passed, int) or not isinstance(self.total, int):
            raise ValueError("passed and total must be integers")
        if self.total <= 0:
            raise ValueError("total must be positive")
        if not 0 <= self.passed <= self.total:
            raise ValueError("passed must be between zero and total")
        rate = self.passed / self.total
        z_squared = _WILSON_Z_95**2
        denominator = 1.0 + z_squared / self.total
        center = (rate + z_squared / (2.0 * self.total)) / denominator
        radius = (
            _WILSON_Z_95
            * math.sqrt(rate * (1.0 - rate) / self.total + z_squared / (4.0 * self.total**2))
            / denominator
        )
        object.__setattr__(self, "rate", rate)
        object.__setattr__(
            self, "wilson_95", (max(0.0, center - radius), min(1.0, center + radius))
        )

    @property
    def valid_transfer_rate(self) -> float:
        return self.rate


def wilson_95_confidence_interval(passed: int, total: int) -> tuple[float, float]:
    """Compute a two-sided Wilson 95% confidence interval."""

    return ValidTransferRate(passed, total).wilson_95


@dataclass(frozen=True)
class GroupRate:
    group: str
    valid_transfer_rate: ValidTransferRate


@dataclass(frozen=True)
class ExperimentStatistics:
    """Experiment-level all-gates statistics over unique independent units."""

    valid_transfer_rate: ValidTransferRate
    grouping_key: str
    per_group: tuple[GroupRate, ...]
    worst_group: GroupRate
    gate_failure_counts: Mapping[str, int]
    human_pending_count: int
    human_rejected_count: int

    def __post_init__(self) -> None:
        if self.grouping_key not in _GROUPING_KEYS:
            raise ValueError(f"unknown grouping key: {self.grouping_key!r}")
        if not self.per_group:
            raise ValueError("experiment statistics require at least one group")
        if self.worst_group not in self.per_group:
            raise ValueError("worst_group must be one of per_group")
        counts = dict(self.gate_failure_counts)
        if any(not isinstance(name, str) or count < 0 for name, count in counts.items()):
            raise ValueError("gate failure counts must be named non-negative integers")
        object.__setattr__(
            self, "gate_failure_counts", MappingProxyType(dict(sorted(counts.items())))
        )

    @property
    def passed(self) -> int:
        return self.valid_transfer_rate.passed

    @property
    def total(self) -> int:
        return self.valid_transfer_rate.total

    def to_dict(self) -> dict[str, object]:
        """Return deterministic, JSON-ready data suitable for experiment evidence."""

        def rate_payload(rate: ValidTransferRate) -> dict[str, object]:
            return {
                "passed": rate.passed,
                "rate": rate.rate,
                "total": rate.total,
                "wilson_95": list(rate.wilson_95),
            }

        return {
            "gate_failure_counts": dict(self.gate_failure_counts),
            "grouping_key": self.grouping_key,
            "human_pending_count": self.human_pending_count,
            "human_rejected_count": self.human_rejected_count,
            "per_group": [
                {
                    "group": group.group,
                    "valid_transfer_rate": rate_payload(group.valid_transfer_rate),
                }
                for group in self.per_group
            ],
            "valid_transfer_rate": rate_payload(self.valid_transfer_rate),
            "worst_group": {
                "group": self.worst_group.group,
                "valid_transfer_rate": rate_payload(self.worst_group.valid_transfer_rate),
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def summarize_experiment(
    contract: AcceptanceContract,
    records: Iterable[EvaluationRecord],
    *,
    grouping_key: str,
) -> ExperimentStatistics:
    """Summarize unique independent units and reject frame/view-count inflation."""

    if grouping_key not in _GROUPING_KEYS:
        raise ValueError(f"unknown grouping key: {grouping_key!r}")
    records = tuple(records)
    if not records:
        raise ValueError("cannot summarize an empty experiment")
    seen_units: set[str] = set()
    decisions: list[AcceptanceDecision] = []
    groups: dict[str, list[AcceptanceDecision]] = {}
    for record in records:
        unit_id = record.unit.unit_id
        if unit_id in seen_units:
            raise ValueError(f"duplicate independent evaluation unit: {unit_id}")
        seen_units.add(unit_id)
        decision = evaluate_acceptance(contract, record)
        decisions.append(decision)
        groups.setdefault(record.unit.group_value(grouping_key), []).append(decision)

    failure_counts = {gate.name: 0 for gate in contract.required_gates}
    for decision in decisions:
        for result in decision.gate_results:
            if not result.passed:
                failure_counts[result.name] += 1
    group_rates = tuple(
        GroupRate(
            group,
            ValidTransferRate(
                sum(decision.accepted for decision in group_decisions),
                len(group_decisions),
            ),
        )
        for group, group_decisions in sorted(groups.items())
    )
    worst_group = min(
        group_rates,
        key=lambda group: (group.valid_transfer_rate.rate, group.group),
    )
    return ExperimentStatistics(
        valid_transfer_rate=ValidTransferRate(
            sum(decision.accepted for decision in decisions), len(decisions)
        ),
        grouping_key=grouping_key,
        per_group=group_rates,
        worst_group=worst_group,
        gate_failure_counts=failure_counts,
        human_pending_count=sum(record.human_review is None for record in records),
        human_rejected_count=sum(record.human_review is False for record in records),
    )
