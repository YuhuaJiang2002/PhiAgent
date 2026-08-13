"""Lightweight learned repair routing for public-Ego AC-WM videos.

Inference uses only the Python standard library.  NumPy is required only by the
separate training entry point, keeping :mod:`phiagent` importable without a
heavy video or GPU stack.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


SCORE_FIELDS = (
    "background_lock",
    "object_lock",
    "subject_replacement",
    "robot_identity",
    "motion_preservation",
    "temporal_consistency",
    "epl_minimum",
)

REPAIR_FIELDS = ("support_dilation_pixels", "alpha_blur_sigma")
_REPAIR_SCALES = {"support_dilation_pixels": 24.0, "alpha_blur_sigma": 12.0}


def _score(scorecard: Mapping[str, object], field: str) -> float:
    try:
        value = float(scorecard[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"scorecard requires numeric {field}") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"scorecard {field} must be finite and in [0, 1]")
    return value


def _repair_value(repair: Mapping[str, object], field: str) -> float:
    try:
        value = float(repair[field]) / _REPAIR_SCALES[field]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"repair requires numeric {field}") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"normalized repair field {field} must be in [0, 1]")
    return value


def feature_names() -> tuple[str, ...]:
    repairs = tuple(f"repair:{field}" for field in REPAIR_FIELDS)
    interactions = tuple(
        f"interaction:{score}*{repair}"
        for score in SCORE_FIELDS
        for repair in REPAIR_FIELDS
    )
    return (*SCORE_FIELDS, *repairs, *interactions)


def encode_features(
    raw_scorecard: Mapping[str, object], repair: Mapping[str, object]
) -> tuple[float, ...]:
    """Encode raw diagnostics, bounded repair parameters, and interactions."""

    raw = tuple(_score(raw_scorecard, field) for field in SCORE_FIELDS)
    recipe = tuple(_repair_value(repair, field) for field in REPAIR_FIELDS)
    return (*raw, *recipe, *(left * right for left in raw for right in recipe))


@dataclass(frozen=True)
class EgoNonRegressionContract:
    """Hard capability limits a post-processing recipe may not hide."""

    motion_preservation: float = 0.02
    epl_minimum: float = 0.02
    temporal_consistency: float = 0.02
    robot_identity: float = 0.02
    subject_replacement: float = 0.03

    def __post_init__(self) -> None:
        for field, value in self.to_dict().items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid non-regression tolerance {field}")

    def to_dict(self) -> dict[str, float]:
        return {
            "motion_preservation": self.motion_preservation,
            "epl_minimum": self.epl_minimum,
            "temporal_consistency": self.temporal_consistency,
            "robot_identity": self.robot_identity,
            "subject_replacement": self.subject_replacement,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EgoNonRegressionContract":
        return cls(
            **{
                field: float(payload[field])
                for field in cls().to_dict()
            }
        )

    def assess(
        self,
        baseline: Mapping[str, object],
        candidate: Mapping[str, object],
    ) -> dict[str, object]:
        regressions = {
            field: max(0.0, _score(baseline, field) - _score(candidate, field))
            for field in self.to_dict()
        }
        excess = {
            field: max(0.0, regressions[field] - tolerance)
            for field, tolerance in self.to_dict().items()
        }
        return {
            "passed": all(value <= 1e-12 for value in excess.values()),
            "regressions": regressions,
            "tolerances": self.to_dict(),
            "excess_regressions": excess,
            "total_excess": sum(excess.values()),
        }


@dataclass(frozen=True)
class EgoRepairPolicy:
    """Ridge utility ranker trained only on domain-matched Ego evaluations."""

    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    alpha: float
    training_actions: tuple[str, ...]
    held_out_action: str | None = None
    non_regression_contract: EgoNonRegressionContract = EgoNonRegressionContract()
    regression_penalty: float = 2.0
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        expected = len(feature_names())
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported Ego repair policy schema")
        if not (
            len(self.feature_mean)
            == len(self.feature_scale)
            == len(self.coefficients)
            == expected
        ):
            raise ValueError(f"Ego repair policy requires {expected} features")
        numeric = (
            *self.feature_mean,
            *self.feature_scale,
            self.intercept,
            *self.coefficients,
            self.alpha,
            self.regression_penalty,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Ego repair policy contains non-finite values")
        if any(value <= 0 for value in self.feature_scale):
            raise ValueError("Ego repair feature scales must be positive")
        if self.alpha <= 0 or self.regression_penalty <= 0:
            raise ValueError("Ego repair regularization values must be positive")
        if not self.training_actions or len(set(self.training_actions)) != len(
            self.training_actions
        ):
            raise ValueError("Ego repair policy requires unique training actions")
        if self.held_out_action in self.training_actions:
            raise ValueError("held-out action cannot be a training action")

    def predict(
        self, raw_scorecard: Mapping[str, object], repair: Mapping[str, object]
    ) -> float:
        encoded = encode_features(raw_scorecard, repair)
        standardized = (
            (value - mean) / scale
            for value, mean, scale in zip(
                encoded, self.feature_mean, self.feature_scale
            )
        )
        return self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized)
        )

    def rank(
        self,
        raw_scorecard: Mapping[str, object],
        repairs: Sequence[Mapping[str, object]],
    ) -> tuple[tuple[Mapping[str, object], float], ...]:
        if not repairs:
            raise ValueError("at least one Ego repair recipe is required")
        return tuple(
            sorted(
                ((repair, self.predict(raw_scorecard, repair)) for repair in repairs),
                key=lambda item: (item[1], str(item[0].get("name", ""))),
                reverse=True,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method": "ego_bottle_repair_ridge_utility_ranker",
            "feature_names": list(feature_names()),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "alpha": self.alpha,
            "training_actions": list(self.training_actions),
            "held_out_action": self.held_out_action,
            "non_regression_contract": self.non_regression_contract.to_dict(),
            "regression_penalty": self.regression_penalty,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EgoRepairPolicy":
        if payload.get("method") != "ego_bottle_repair_ridge_utility_ranker":
            raise ValueError("not an Ego bottle repair policy checkpoint")
        if payload.get("feature_names") != list(feature_names()):
            raise ValueError("Ego repair feature contract does not match")
        try:
            return cls(
                schema_version=str(payload["schema_version"]),
                feature_mean=tuple(float(value) for value in payload["feature_mean"]),  # type: ignore[index]
                feature_scale=tuple(float(value) for value in payload["feature_scale"]),  # type: ignore[index]
                intercept=float(payload["intercept"]),
                coefficients=tuple(float(value) for value in payload["coefficients"]),  # type: ignore[index]
                alpha=float(payload["alpha"]),
                training_actions=tuple(str(value) for value in payload["training_actions"]),  # type: ignore[index]
                held_out_action=(
                    None
                    if payload.get("held_out_action") is None
                    else str(payload["held_out_action"])
                ),
                non_regression_contract=EgoNonRegressionContract.from_dict(
                    payload["non_regression_contract"]  # type: ignore[arg-type]
                ),
                regression_penalty=float(payload["regression_penalty"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed Ego repair policy checkpoint") from error

    @classmethod
    def load(cls, path: Path) -> "EgoRepairPolicy":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Ego repair checkpoint must contain one JSON object")
        return cls.from_dict(payload)
