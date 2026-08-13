"""Lightweight learned ranking policy for flower-video repair recipes.

The policy is intentionally inference-only and standard-library-only. Training
is performed by an optional script so importing :mod:`phiagent` never requires
NumPy, PyTorch, a video model, or a GPU.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


RAW_SCORE_FIELDS = (
    "background_lock",
    "object_lock",
    "subject_replacement",
    "robot_identity",
    "motion_preservation",
    "temporal_consistency",
    "epl_minimum",
)

REPAIR_FEATURE_FIELDS = (
    "hard_background_lock",
    "restore_source_flowers",
    "exclude_source_face_from_flower_restore",
    "mask_dilation_pixels",
    "flower_dilation_pixels",
    "face_box_margin_pixels",
)

_REPAIR_SCALES = {
    "hard_background_lock": 1.0,
    "restore_source_flowers": 1.0,
    "exclude_source_face_from_flower_restore": 1.0,
    "mask_dilation_pixels": 3.0,
    "flower_dilation_pixels": 2.0,
    "face_box_margin_pixels": 12.0,
}


@dataclass(frozen=True)
class NonRegressionAssessment:
    """Measured capability deltas against the unmodified candidate."""

    passed: bool
    regressions: tuple[tuple[str, float], ...]
    tolerances: tuple[tuple[str, float], ...]
    excess_regressions: tuple[tuple[str, float], ...]
    minimum_margin: float

    @property
    def total_excess(self) -> float:
        return sum(value for _, value in self.excess_regressions)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "regressions": dict(self.regressions),
            "tolerances": dict(self.tolerances),
            "excess_regressions": dict(self.excess_regressions),
            "minimum_margin": self.minimum_margin,
            "total_excess": self.total_excess,
        }


@dataclass(frozen=True)
class NonRegressionContract:
    """Hard limits preventing one proxy gain from hiding capability collapse."""

    motion_preservation: float = 0.01
    epl_minimum: float = 0.01
    temporal_consistency: float = 0.01
    robot_identity: float = 0.01
    subject_replacement: float = 0.02

    def __post_init__(self) -> None:
        for field, tolerance in self.to_dict().items():
            if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
                raise ValueError(f"non-regression tolerance {field} must be in [0, 1]")

    def to_dict(self) -> dict[str, float]:
        return {
            "motion_preservation": self.motion_preservation,
            "epl_minimum": self.epl_minimum,
            "temporal_consistency": self.temporal_consistency,
            "robot_identity": self.robot_identity,
            "subject_replacement": self.subject_replacement,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "NonRegressionContract":
        try:
            return cls(**{field: float(payload[field]) for field in cls().to_dict()})
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed non-regression contract") from error

    def assess(
        self,
        baseline: Mapping[str, object],
        candidate: Mapping[str, object],
    ) -> NonRegressionAssessment:
        tolerances = self.to_dict()
        regressions = []
        excess = []
        margins = []
        for field, tolerance in tolerances.items():
            baseline_value = _finite_score(baseline, field)
            candidate_value = _finite_score(candidate, field)
            regression = max(0.0, baseline_value - candidate_value)
            regression_excess = max(0.0, regression - tolerance)
            regressions.append((field, regression))
            excess.append((field, regression_excess))
            margins.append(candidate_value - (baseline_value - tolerance))
        return NonRegressionAssessment(
            passed=all(value <= 1e-12 for _, value in excess),
            regressions=tuple(regressions),
            tolerances=tuple(tolerances.items()),
            excess_regressions=tuple(excess),
            minimum_margin=min(margins),
        )


def feature_names() -> tuple[str, ...]:
    """Return the stable ordered feature contract used by checkpoints."""

    repair_names = tuple(f"repair:{name}" for name in REPAIR_FEATURE_FIELDS)
    interactions = tuple(
        f"interaction:{score}*{repair}"
        for score in RAW_SCORE_FIELDS
        for repair in REPAIR_FEATURE_FIELDS
    )
    return (*RAW_SCORE_FIELDS, *repair_names, *interactions)


def _finite_score(scorecard: Mapping[str, object], field: str) -> float:
    try:
        value = float(scorecard[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"raw scorecard requires numeric {field}") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"raw scorecard {field} must be finite and in [0, 1]")
    return value


def _repair_value(repair: Mapping[str, object], field: str) -> float:
    try:
        value = float(repair[field]) / _REPAIR_SCALES[field]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"repair recipe requires numeric {field}") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"normalized repair feature {field} must be in [0, 1]")
    return value


def encode_features(
    raw_scorecard: Mapping[str, object], repair: Mapping[str, object]
) -> tuple[float, ...]:
    """Encode raw diagnostics, a candidate recipe, and their interactions."""

    raw = tuple(_finite_score(raw_scorecard, field) for field in RAW_SCORE_FIELDS)
    recipe = tuple(_repair_value(repair, field) for field in REPAIR_FEATURE_FIELDS)
    interactions = tuple(score * parameter for score in raw for parameter in recipe)
    return (*raw, *recipe, *interactions)


@dataclass(frozen=True)
class FlowerRepairPolicy:
    """A standardized ridge-regression utility ranker."""

    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    alpha: float
    training_actions: tuple[str, ...]
    held_out_action: str | None = None
    objective: str = "non_regression_constrained_utility"
    non_regression_contract: NonRegressionContract = NonRegressionContract()
    regression_penalty: float = 2.0
    schema_version: str = "1.1.0"

    def __post_init__(self) -> None:
        expected = len(feature_names())
        if self.schema_version not in {"1.0.0", "1.1.0"}:
            raise ValueError(f"unsupported flower repair policy: {self.schema_version}")
        if not (
            len(self.feature_mean) == len(self.feature_scale) == len(self.coefficients) == expected
        ):
            raise ValueError(f"flower repair policy requires {expected} features")
        values = (*self.feature_mean, *self.feature_scale, self.intercept, *self.coefficients)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("flower repair policy contains non-finite values")
        if any(value <= 0 for value in self.feature_scale):
            raise ValueError("flower repair feature scales must be positive")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("flower repair ridge alpha must be finite and positive")
        if self.objective not in {"mean_utility", "non_regression_constrained_utility"}:
            raise ValueError("unsupported flower repair training objective")
        if not math.isfinite(self.regression_penalty) or self.regression_penalty <= 0:
            raise ValueError("flower repair regression penalty must be finite and positive")
        if not self.training_actions or any(not action.strip() for action in self.training_actions):
            raise ValueError("flower repair policy requires named training actions")
        if len(set(self.training_actions)) != len(self.training_actions):
            raise ValueError("flower repair training actions must be unique")
        if self.held_out_action is not None and self.held_out_action in self.training_actions:
            raise ValueError("held-out action cannot also be a training action")

    def predict(self, raw_scorecard: Mapping[str, object], repair: Mapping[str, object]) -> float:
        encoded = encode_features(raw_scorecard, repair)
        standardized = (
            (value - mean) / scale
            for value, mean, scale in zip(encoded, self.feature_mean, self.feature_scale)
        )
        return self.intercept + sum(
            coefficient * value for coefficient, value in zip(self.coefficients, standardized)
        )

    def rank(
        self,
        raw_scorecard: Mapping[str, object],
        repairs: Sequence[Mapping[str, object]],
    ) -> tuple[tuple[Mapping[str, object], float], ...]:
        if not repairs:
            raise ValueError("at least one flower repair recipe is required")
        scored = tuple((repair, self.predict(raw_scorecard, repair)) for repair in repairs)
        return tuple(
            sorted(
                scored,
                key=lambda item: (item[1], str(item[0].get("name", ""))),
                reverse=True,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method": "flower_repair_ridge_utility_ranker",
            "feature_names": list(feature_names()),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "alpha": self.alpha,
            "training_actions": list(self.training_actions),
            "held_out_action": self.held_out_action,
            "objective": self.objective,
            "non_regression_contract": self.non_regression_contract.to_dict(),
            "regression_penalty": self.regression_penalty,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FlowerRepairPolicy":
        if payload.get("method") != "flower_repair_ridge_utility_ranker":
            raise ValueError("not a flower repair utility-ranker checkpoint")
        names = payload.get("feature_names")
        if names != list(feature_names()):
            raise ValueError("flower repair checkpoint feature contract does not match")
        try:
            means = tuple(float(value) for value in payload["feature_mean"])  # type: ignore[index]
            scales = tuple(float(value) for value in payload["feature_scale"])  # type: ignore[index]
            coefficients = tuple(float(value) for value in payload["coefficients"])  # type: ignore[index]
            actions = tuple(str(value) for value in payload["training_actions"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed flower repair policy checkpoint") from error
        held_out = payload.get("held_out_action")
        objective = str(payload.get("objective", "mean_utility"))
        contract_payload = payload.get("non_regression_contract")
        contract = (
            NonRegressionContract()
            if contract_payload is None
            else NonRegressionContract.from_dict(contract_payload)  # type: ignore[arg-type]
        )
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            feature_mean=means,
            feature_scale=scales,
            intercept=float(payload["intercept"]),
            coefficients=coefficients,
            alpha=float(payload["alpha"]),
            training_actions=actions,
            held_out_action=None if held_out is None else str(held_out),
            objective=objective,
            non_regression_contract=contract,
            regression_penalty=float(payload.get("regression_penalty", 2.0)),
        )

    @classmethod
    def load(cls, path: Path) -> "FlowerRepairPolicy":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("flower repair policy checkpoint must contain one JSON object")
        return cls.from_dict(payload)
