"""Lightweight, fail-closed routing for reproducible demo-video production.

The module contains no video, CUDA, NumPy, or model dependencies. Heavy generators
remain behind command adapters; this layer learns which bounded recipe to try next
from measured baseline diagnostics and refuses promotion on held-group regressions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")


def _safe_id(value: object, name: str) -> str:
    identifier = str(value)
    if not _SAFE_ID.fullmatch(identifier):
        raise ValueError(f"{name} is not filesystem safe: {identifier!r}")
    return identifier


def _finite(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _unit(value: object, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _named_unit_map(payload: object, name: str) -> tuple[tuple[str, float], ...]:
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{name} must be a non-empty JSON object")
    rows = tuple(
        sorted(
            (_safe_id(key, f"{name} key"), _unit(value, f"{name}.{key}"))
            for key, value in payload.items()
        )
    )
    if len({key for key, _ in rows}) != len(rows):
        raise ValueError(f"{name} contains duplicate keys")
    return rows


def _named_nonnegative_map(
    payload: object, name: str
) -> tuple[tuple[str, float], ...]:
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{name} must be a non-empty JSON object")
    rows = tuple(
        sorted(
            (_safe_id(key, f"{name} key"), _finite(value, f"{name}.{key}"))
            for key, value in payload.items()
        )
    )
    if any(value < 0 for _, value in rows):
        raise ValueError(f"{name} values must be non-negative")
    return rows


def _mapping(rows: Sequence[tuple[str, float]]) -> dict[str, float]:
    return dict(rows)


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FactoryContract:
    """One domain-specific quality, cost, and non-regression contract."""

    domain: str
    baseline_recipe_id: str
    recipe_order: tuple[str, ...]
    context_fields: tuple[str, ...]
    metric_weights: tuple[tuple[str, float], ...]
    hard_thresholds: tuple[tuple[str, float], ...]
    non_regression_tolerances: tuple[tuple[str, float], ...]
    cost_budget_units: float = 1.0
    cost_weight: float = 0.05
    rejection_penalty: float = 2.0
    human_review_required: bool = True

    def __post_init__(self) -> None:
        _safe_id(self.domain, "domain")
        _safe_id(self.baseline_recipe_id, "baseline_recipe_id")
        if not self.recipe_order or self.recipe_order[0] != self.baseline_recipe_id:
            raise ValueError("recipe_order must start with baseline_recipe_id")
        for name, values in (
            ("recipe_order", self.recipe_order),
            ("context_fields", self.context_fields),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique values")
            for value in values:
                _safe_id(value, name)
        if not self.context_fields:
            raise ValueError("context_fields must not be empty")
        weights = _mapping(self.metric_weights)
        thresholds = _mapping(self.hard_thresholds)
        tolerances = _mapping(self.non_regression_tolerances)
        if not weights or sum(weights.values()) <= 0:
            raise ValueError("metric_weights must have positive total weight")
        if any(value < 0 for value in weights.values()):
            raise ValueError("metric weights must be non-negative")
        unknown = (set(thresholds) | set(tolerances)) - set(weights)
        if unknown:
            raise ValueError(f"quality contract refers to unweighted metrics: {sorted(unknown)}")
        if not set(self.context_fields).issubset(weights):
            raise ValueError("context_fields must be measured metric names")
        if self.cost_budget_units <= 0 or self.cost_weight < 0 or self.rejection_penalty <= 0:
            raise ValueError("cost budget/weight and rejection penalty are invalid")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FactoryContract":
        try:
            order = payload["recipe_order"]
            fields = payload["context_fields"]
            if not isinstance(order, list) or not isinstance(fields, list):
                raise ValueError("recipe_order and context_fields must be JSON arrays")
            human = payload.get("human_review_required", True)
            if not isinstance(human, bool):
                raise ValueError("human_review_required must be boolean")
            return cls(
                domain=str(payload["domain"]),
                baseline_recipe_id=str(payload["baseline_recipe_id"]),
                recipe_order=tuple(str(value) for value in order),
                context_fields=tuple(str(value) for value in fields),
                metric_weights=_named_nonnegative_map(
                    payload["metric_weights"], "metric_weights"
                ),
                hard_thresholds=_named_unit_map(payload["hard_thresholds"], "hard_thresholds"),
                non_regression_tolerances=_named_unit_map(
                    payload["non_regression_tolerances"],
                    "non_regression_tolerances",
                ),
                cost_budget_units=_finite(payload.get("cost_budget_units", 1.0), "cost_budget_units"),
                cost_weight=_finite(payload.get("cost_weight", 0.05), "cost_weight"),
                rejection_penalty=_finite(
                    payload.get("rejection_penalty", 2.0), "rejection_penalty"
                ),
                human_review_required=human,
            )
        except KeyError as error:
            raise ValueError(f"factory contract is missing {error.args[0]}") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "baseline_recipe_id": self.baseline_recipe_id,
            "recipe_order": list(self.recipe_order),
            "context_fields": list(self.context_fields),
            "metric_weights": _mapping(self.metric_weights),
            "hard_thresholds": _mapping(self.hard_thresholds),
            "non_regression_tolerances": _mapping(self.non_regression_tolerances),
            "cost_budget_units": self.cost_budget_units,
            "cost_weight": self.cost_weight,
            "rejection_penalty": self.rejection_penalty,
            "human_review_required": self.human_review_required,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class FactoryRecord:
    """One measured baseline or bounded-recipe video attempt."""

    episode_id: str
    group_id: str
    domain: str
    recipe_id: str
    recipe_parameters: Mapping[str, object]
    context: Mapping[str, float]
    metrics: Mapping[str, float]
    cost_units: float
    human_review_passed: bool | None
    video: str
    video_sha256: str
    diagnoses: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported factory record schema {self.schema_version!r}")
        for name in ("episode_id", "group_id", "domain", "recipe_id"):
            _safe_id(getattr(self, name), name)
        if not self.context or not self.metrics:
            raise ValueError("factory records require context and metrics")
        for name, values in (("context", self.context), ("metrics", self.metrics)):
            for key, value in values.items():
                _safe_id(key, f"{name} key")
                _unit(value, f"{name}.{key}")
        if self.cost_units < 0 or not math.isfinite(self.cost_units):
            raise ValueError("cost_units must be finite and non-negative")
        if self.human_review_passed not in {True, False, None}:
            raise ValueError("human_review_passed must be true, false, or null")
        if not self.video.strip():
            raise ValueError("video path must not be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.video_sha256):
            raise ValueError("video_sha256 must be a lowercase SHA-256 digest")
        if any(not diagnosis.strip() for diagnosis in self.diagnoses):
            raise ValueError("diagnoses must not contain empty strings")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FactoryRecord":
        try:
            recipe = payload["recipe"]
            artifact = payload["artifact"]
            if not isinstance(recipe, Mapping) or not isinstance(artifact, Mapping):
                raise ValueError("recipe and artifact must be JSON objects")
            parameters = recipe.get("parameters", {})
            context = payload["context"]
            metrics = payload["metrics"]
            diagnoses = payload.get("diagnoses", [])
            if not isinstance(parameters, Mapping):
                raise ValueError("recipe.parameters must be a JSON object")
            if not isinstance(context, Mapping) or not isinstance(metrics, Mapping):
                raise ValueError("context and metrics must be JSON objects")
            if not isinstance(diagnoses, list) or any(
                not isinstance(value, str) for value in diagnoses
            ):
                raise ValueError("diagnoses must be a JSON string list")
            return cls(
                schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
                episode_id=str(payload["episode_id"]),
                group_id=str(payload["group_id"]),
                domain=str(payload["domain"]),
                recipe_id=str(recipe["recipe_id"]),
                recipe_parameters=dict(parameters),
                context={str(key): float(value) for key, value in context.items()},
                metrics={str(key): float(value) for key, value in metrics.items()},
                cost_units=_finite(payload["cost_units"], "cost_units"),
                human_review_passed=payload.get("human_review_passed"),  # type: ignore[arg-type]
                video=str(artifact["video"]),
                video_sha256=str(artifact["sha256"]),
                diagnoses=tuple(diagnoses),
            )
        except KeyError as error:
            raise ValueError(f"factory record is missing {error.args[0]}") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "group_id": self.group_id,
            "domain": self.domain,
            "recipe": {
                "recipe_id": self.recipe_id,
                "parameters": dict(self.recipe_parameters),
            },
            "context": dict(self.context),
            "metrics": dict(self.metrics),
            "cost_units": self.cost_units,
            "human_review_passed": self.human_review_passed,
            "artifact": {"video": self.video, "sha256": self.video_sha256},
            "diagnoses": list(self.diagnoses),
        }


@dataclass(frozen=True)
class FactoryAssessment:
    accepted: bool
    automatic_gates_passed: bool
    human_gate_passed: bool
    non_regression_passed: bool
    hard_shortfalls: tuple[tuple[str, float], ...]
    non_regression_excess: tuple[tuple[str, float], ...]
    utility: float
    training_target: float

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "automatic_gates_passed": self.automatic_gates_passed,
            "human_gate_passed": self.human_gate_passed,
            "non_regression_passed": self.non_regression_passed,
            "hard_shortfalls": dict(self.hard_shortfalls),
            "non_regression_excess": dict(self.non_regression_excess),
            "utility": self.utility,
            "training_target": self.training_target,
        }


def assess_record(
    contract: FactoryContract,
    baseline_metrics: Mapping[str, float],
    record: FactoryRecord,
) -> FactoryAssessment:
    """Apply hard task gates before computing a cost-aware learning target."""

    if record.domain != contract.domain:
        raise ValueError(f"record domain {record.domain!r} does not match {contract.domain!r}")
    weights = _mapping(contract.metric_weights)
    missing = set(weights) - set(record.metrics)
    baseline_missing = set(_mapping(contract.non_regression_tolerances)) - set(
        baseline_metrics
    )
    if missing or baseline_missing:
        raise ValueError(
            f"record/baseline metric contract mismatch: record={sorted(missing)}, "
            f"baseline={sorted(baseline_missing)}"
        )
    hard_shortfalls = tuple(
        (field, max(0.0, threshold - _unit(record.metrics[field], field)))
        for field, threshold in contract.hard_thresholds
    )
    excess = []
    for field, tolerance in contract.non_regression_tolerances:
        regression = max(
            0.0,
            _unit(baseline_metrics[field], f"baseline.{field}")
            - _unit(record.metrics[field], field),
        )
        excess.append((field, max(0.0, regression - tolerance)))
    automatic = all(value <= 1e-12 for _, value in hard_shortfalls)
    non_regression = all(value <= 1e-12 for _, value in excess)
    human = record.human_review_passed is not False and (
        not contract.human_review_required or record.human_review_passed is True
    )
    weighted = sum(record.metrics[field] * weight for field, weight in weights.items())
    utility = weighted / sum(weights.values()) - contract.cost_weight * (
        record.cost_units / contract.cost_budget_units
    )
    accepted = automatic and non_regression and human
    failure = sum(value for _, value in hard_shortfalls) + sum(value for _, value in excess)
    if not human:
        failure += 1.0
    target = utility if accepted else utility - contract.rejection_penalty * (1.0 + failure)
    return FactoryAssessment(
        accepted=accepted,
        automatic_gates_passed=automatic,
        human_gate_passed=human,
        non_regression_passed=non_regression,
        hard_shortfalls=hard_shortfalls,
        non_regression_excess=tuple(excess),
        utility=utility,
        training_target=target,
    )


def policy_feature_names(
    context_fields: Sequence[str], recipe_ids: Sequence[str]
) -> tuple[str, ...]:
    return (
        *tuple(f"context:{field}" for field in context_fields),
        *tuple(f"recipe:{recipe_id}" for recipe_id in recipe_ids),
        *tuple(
            f"interaction:{field}*{recipe_id}"
            for field in context_fields
            for recipe_id in recipe_ids
        ),
    )


def encode_policy_features(
    context: Mapping[str, float],
    recipe_id: str,
    context_fields: Sequence[str],
    recipe_ids: Sequence[str],
) -> tuple[float, ...]:
    if recipe_id not in recipe_ids:
        raise ValueError(f"policy does not know recipe {recipe_id!r}")
    values = tuple(_unit(context[field], f"context.{field}") for field in context_fields)
    one_hot = tuple(float(recipe_id == candidate) for candidate in recipe_ids)
    return (*values, *one_hot, *(value * flag for value in values for flag in one_hot))


@dataclass(frozen=True)
class DemoFactoryPolicy:
    """Standard-library ridge router promoted only by grouped replay."""

    domain: str
    contract_sha256: str
    context_fields: tuple[str, ...]
    recipe_ids: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    alpha: float
    training_groups: tuple[str, ...]
    promoted: bool
    promotion_gates: Mapping[str, bool]
    held_group_metrics: Mapping[str, float]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported demo factory policy schema")
        _safe_id(self.domain, "domain")
        if not re.fullmatch(r"[0-9a-f]{64}", self.contract_sha256):
            raise ValueError("contract_sha256 must be a SHA-256 digest")
        names = policy_feature_names(self.context_fields, self.recipe_ids)
        if not (
            len(names)
            == len(self.feature_mean)
            == len(self.feature_scale)
            == len(self.coefficients)
        ):
            raise ValueError(f"policy requires exactly {len(names)} features")
        numbers = (*self.feature_mean, *self.feature_scale, self.intercept, *self.coefficients)
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("policy contains non-finite values")
        if any(value <= 0 for value in self.feature_scale) or self.alpha <= 0:
            raise ValueError("policy scales and alpha must be positive")
        if len(set(self.recipe_ids)) != len(self.recipe_ids) or not self.recipe_ids:
            raise ValueError("policy recipe IDs must be unique and non-empty")

    def predict(self, context: Mapping[str, float], recipe_id: str) -> float:
        encoded = encode_policy_features(
            context, recipe_id, self.context_fields, self.recipe_ids
        )
        return self.intercept + sum(
            coefficient * ((value - mean) / scale)
            for value, mean, scale, coefficient in zip(
                encoded, self.feature_mean, self.feature_scale, self.coefficients
            )
        )

    def rank(
        self, context: Mapping[str, float], available_recipes: Sequence[str]
    ) -> tuple[tuple[str, float], ...]:
        if not available_recipes:
            raise ValueError("at least one available recipe is required")
        unknown = set(available_recipes) - set(self.recipe_ids)
        if unknown:
            raise ValueError(f"policy does not know recipes: {sorted(unknown)}")
        return tuple(
            sorted(
                ((recipe, self.predict(context, recipe)) for recipe in available_recipes),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method": "held_group_cost_aware_demo_recipe_ridge_router",
            "domain": self.domain,
            "contract_sha256": self.contract_sha256,
            "context_fields": list(self.context_fields),
            "recipe_ids": list(self.recipe_ids),
            "feature_names": list(policy_feature_names(self.context_fields, self.recipe_ids)),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "alpha": self.alpha,
            "training_groups": list(self.training_groups),
            "promoted": self.promoted,
            "promotion_gates": dict(self.promotion_gates),
            "held_group_metrics": dict(self.held_group_metrics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DemoFactoryPolicy":
        if payload.get("method") != "held_group_cost_aware_demo_recipe_ridge_router":
            raise ValueError("not a demo factory router checkpoint")
        try:
            policy = cls(
                schema_version=str(payload["schema_version"]),
                domain=str(payload["domain"]),
                contract_sha256=str(payload["contract_sha256"]),
                context_fields=tuple(str(value) for value in payload["context_fields"]),  # type: ignore[index]
                recipe_ids=tuple(str(value) for value in payload["recipe_ids"]),  # type: ignore[index]
                feature_mean=tuple(float(value) for value in payload["feature_mean"]),  # type: ignore[index]
                feature_scale=tuple(float(value) for value in payload["feature_scale"]),  # type: ignore[index]
                intercept=float(payload["intercept"]),
                coefficients=tuple(float(value) for value in payload["coefficients"]),  # type: ignore[index]
                alpha=float(payload["alpha"]),
                training_groups=tuple(str(value) for value in payload["training_groups"]),  # type: ignore[index]
                promoted=bool(payload["promoted"]),
                promotion_gates={
                    str(key): bool(value)
                    for key, value in payload["promotion_gates"].items()  # type: ignore[union-attr]
                },
                held_group_metrics={
                    str(key): float(value)
                    for key, value in payload["held_group_metrics"].items()  # type: ignore[union-attr]
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed demo factory router checkpoint") from error
        expected = list(policy_feature_names(policy.context_fields, policy.recipe_ids))
        if payload.get("feature_names") != expected:
            raise ValueError("demo factory policy feature contract does not match")
        return policy

    @classmethod
    def load(cls, path: Path) -> "DemoFactoryPolicy":
        payload = json.loads(path.read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("policy checkpoint must contain one JSON object")
        return cls.from_dict(payload)


def load_records(paths: Sequence[Path]) -> tuple[FactoryRecord, ...]:
    records: list[FactoryRecord] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                if not isinstance(payload, Mapping):
                    raise ValueError("record must be a JSON object")
                record = FactoryRecord.from_dict(payload)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid factory record at {path}:{line_number}: {error}") from error
            key = (record.episode_id, record.recipe_id)
            if key in seen:
                raise ValueError(f"duplicate episode/recipe record: {key}")
            seen.add(key)
            records.append(record)
    if not records:
        raise ValueError("factory training data is empty")
    return tuple(records)


def _solve_linear_system(matrix: list[list[float]], values: list[float]) -> list[float]:
    """Solve a small dense system with pivoted Gauss-Jordan elimination."""

    size = len(values)
    augmented = [matrix[row][:] + [values[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("ridge design is singular; add groups or increase alpha")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if abs(factor) <= 1e-18:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _fit_policy(
    records: Sequence[FactoryRecord],
    contract: FactoryContract,
    recipe_ids: tuple[str, ...],
    alpha: float,
) -> tuple[tuple[float, ...], tuple[float, ...], float, tuple[float, ...]]:
    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    by_episode = _episodes(records, contract)
    features = []
    targets = []
    for record in records:
        baseline = by_episode[record.episode_id][contract.baseline_recipe_id]
        features.append(
            encode_policy_features(
                record.context, record.recipe_id, contract.context_fields, recipe_ids
            )
        )
        targets.append(assess_record(contract, baseline.metrics, record).training_target)
    width = len(features[0])
    mean = tuple(sum(row[index] for row in features) / len(features) for index in range(width))
    scale = []
    for index in range(width):
        variance = sum((row[index] - mean[index]) ** 2 for row in features) / len(features)
        scale.append(max(math.sqrt(variance), 1e-8))
    design = [
        [1.0, *((value - mean[index]) / scale[index] for index, value in enumerate(row))]
        for row in features
    ]
    size = width + 1
    gram = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size
    for row, target in zip(design, targets):
        for left in range(size):
            rhs[left] += row[left] * target
            for right in range(size):
                gram[left][right] += row[left] * row[right]
    for index in range(1, size):
        gram[index][index] += alpha
    gram[0][0] += 1e-12
    weights = _solve_linear_system(gram, rhs)
    return mean, tuple(scale), weights[0], tuple(weights[1:])


def _episodes(
    records: Sequence[FactoryRecord], contract: FactoryContract
) -> dict[str, dict[str, FactoryRecord]]:
    episodes: dict[str, dict[str, FactoryRecord]] = {}
    for record in records:
        if record.domain != contract.domain:
            raise ValueError("one router cannot mix video domains")
        missing_context = set(contract.context_fields) - set(record.context)
        if missing_context:
            raise ValueError(f"record context is missing {sorted(missing_context)}")
        episode = episodes.setdefault(record.episode_id, {})
        if record.recipe_id in episode:
            raise ValueError(f"episode {record.episode_id} repeats recipe {record.recipe_id}")
        episode[record.recipe_id] = record
    for episode_id, recipes in episodes.items():
        if contract.baseline_recipe_id not in recipes:
            raise ValueError(f"episode {episode_id} has no declared baseline")
        baseline = recipes[contract.baseline_recipe_id]
        expected_context = {
            field: baseline.metrics[field] for field in contract.context_fields
        }
        group = baseline.group_id
        for record in recipes.values():
            if record.group_id != group:
                raise ValueError(f"episode {episode_id} mixes held-out groups")
            if any(
                abs(record.context[field] - expected_context[field]) > 1e-9
                for field in contract.context_fields
            ):
                raise ValueError(f"episode {episode_id} context is not baseline-bound")
            assess_record(contract, baseline.metrics, record)
    return episodes


def _simulate_routes(
    policy: DemoFactoryPolicy,
    records: Sequence[FactoryRecord],
    contract: FactoryContract,
) -> dict[str, object]:
    episodes = _episodes(records, contract)
    learned_rows = []
    default_rows = []
    for episode_id, recipes in sorted(episodes.items()):
        baseline = recipes[contract.baseline_recipe_id]
        candidates = [recipe for recipe in recipes if recipe != contract.baseline_recipe_id]
        learned_order = [
            recipe for recipe, _ in policy.rank(baseline.context, candidates)
        ] if candidates else []
        default_order = [
            recipe for recipe in contract.recipe_order if recipe in recipes and recipe != contract.baseline_recipe_id
        ]
        for label, order, destination in (
            ("learned", learned_order, learned_rows),
            ("default", default_order, default_rows),
        ):
            attempted = [baseline]
            selected = baseline
            baseline_assessment = assess_record(contract, baseline.metrics, baseline)
            accepted = baseline_assessment.accepted
            if not accepted:
                for recipe_id in order:
                    candidate = recipes[recipe_id]
                    attempted.append(candidate)
                    assessment = assess_record(contract, baseline.metrics, candidate)
                    if assessment.accepted:
                        selected = candidate
                        accepted = True
                        break
            assessment = assess_record(contract, baseline.metrics, selected)
            destination.append(
                {
                    "route": label,
                    "episode_id": episode_id,
                    "group_id": baseline.group_id,
                    "attempted_recipes": [item.recipe_id for item in attempted],
                    "selected_recipe": selected.recipe_id,
                    "accepted": accepted,
                    "selected_non_regression": assessment.non_regression_passed,
                    "selected_utility": assessment.utility,
                    "attempts": len(attempted),
                    "cost_units": sum(item.cost_units for item in attempted),
                }
            )

    def aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
        count = len(rows)
        return {
            "acceptance_rate": sum(bool(row["accepted"]) for row in rows) / count,
            "selected_non_regression_rate": sum(
                bool(row["selected_non_regression"]) for row in rows
            ) / count,
            "mean_selected_utility": sum(float(row["selected_utility"]) for row in rows) / count,
            "mean_attempts": sum(int(row["attempts"]) for row in rows) / count,
            "mean_cost_units": sum(float(row["cost_units"]) for row in rows) / count,
        }

    return {
        "learned": aggregate(learned_rows),
        "default": aggregate(default_rows),
        "selections": learned_rows,
        "default_selections": default_rows,
    }


@dataclass(frozen=True)
class FactoryTrainingResult:
    policy: DemoFactoryPolicy
    evaluation: Mapping[str, object]
    preferences: tuple[Mapping[str, object], ...]


def train_grouped_router(
    records: Sequence[FactoryRecord],
    contract: FactoryContract,
    *,
    alpha: float = 0.01,
    minimum_acceptance_rate: float = 0.5,
    utility_regression_tolerance: float = 0.0,
    cost_regression_fraction: float = 0.0,
) -> FactoryTrainingResult:
    """Fit a router and promote it only on leave-one-group-out replay."""

    if not 0.0 <= minimum_acceptance_rate <= 1.0:
        raise ValueError("minimum_acceptance_rate must be in [0, 1]")
    if utility_regression_tolerance < 0 or cost_regression_fraction < 0:
        raise ValueError("promotion regression tolerances must be non-negative")
    episodes = _episodes(records, contract)
    groups = tuple(sorted({next(iter(rows.values())).group_id for rows in episodes.values()}))
    if len(groups) < 2:
        raise ValueError("held-group training requires at least two groups")
    recipe_ids = tuple(recipe for recipe in contract.recipe_order if any(
        recipe in episode for episode in episodes.values()
    ))
    unknown_recipes = {record.recipe_id for record in records} - set(recipe_ids)
    if unknown_recipes:
        raise ValueError(f"records contain recipes absent from recipe_order: {sorted(unknown_recipes)}")
    incomplete_episodes = {
        episode_id: sorted(set(recipe_ids) - set(measured))
        for episode_id, measured in episodes.items()
        if set(measured) != set(recipe_ids)
    }
    if incomplete_episodes:
        raise ValueError(
            f"every episode must measure every recipe: {incomplete_episodes}"
        )
    coverage = {
        recipe: {record.group_id for record in records if record.recipe_id == recipe}
        for recipe in recipe_ids
    }
    incomplete = {recipe: sorted(set(groups) - recipe_groups) for recipe, recipe_groups in coverage.items() if recipe_groups != set(groups)}
    if incomplete:
        raise ValueError(f"every held group must measure every recipe: {incomplete}")

    held_records: list[FactoryRecord] = []
    fold_reports = []
    fold_selections = []
    for held_group in groups:
        train_rows = [record for record in records if record.group_id != held_group]
        test_rows = [record for record in records if record.group_id == held_group]
        mean, scale, intercept, coefficients = _fit_policy(
            train_rows, contract, recipe_ids, alpha
        )
        fold_policy = DemoFactoryPolicy(
            domain=contract.domain,
            contract_sha256=contract.fingerprint,
            context_fields=contract.context_fields,
            recipe_ids=recipe_ids,
            feature_mean=mean,
            feature_scale=scale,
            intercept=intercept,
            coefficients=coefficients,
            alpha=alpha,
            training_groups=tuple(group for group in groups if group != held_group),
            promoted=False,
            promotion_gates={},
            held_group_metrics={},
        )
        replay = _simulate_routes(fold_policy, test_rows, contract)
        fold_reports.append(
            {
                "held_group": held_group,
                "training_rows": len(train_rows),
                "test_rows": len(test_rows),
                "learned": replay["learned"],
                "default": replay["default"],
            }
        )
        fold_selections.extend(replay["selections"])  # type: ignore[arg-type]
        held_records.extend(test_rows)

    # Every record appears once in held-group replay. Reconstruct aggregate metrics
    # from the fold policies rather than evaluating the final policy on training data.
    learned_metrics = {
        "acceptance_rate": sum(bool(row["accepted"]) for row in fold_selections) / len(fold_selections),
        "selected_non_regression_rate": sum(bool(row["selected_non_regression"]) for row in fold_selections) / len(fold_selections),
        "mean_selected_utility": sum(float(row["selected_utility"]) for row in fold_selections) / len(fold_selections),
        "mean_attempts": sum(float(row["attempts"]) for row in fold_selections) / len(fold_selections),
        "mean_cost_units": sum(float(row["cost_units"]) for row in fold_selections) / len(fold_selections),
    }
    default_selections = []
    for report in fold_reports:
        held_group = str(report["held_group"])
        held = [record for record in held_records if record.group_id == held_group]
        # A zero policy preserves the declared recipe order only for obtaining the
        # independently guarded default replay; predictions are ignored below.
        width = len(policy_feature_names(contract.context_fields, recipe_ids))
        neutral = DemoFactoryPolicy(
            domain=contract.domain,
            contract_sha256=contract.fingerprint,
            context_fields=contract.context_fields,
            recipe_ids=recipe_ids,
            feature_mean=(0.0,) * width,
            feature_scale=(1.0,) * width,
            intercept=0.0,
            coefficients=(0.0,) * width,
            alpha=alpha,
            training_groups=(),
            promoted=False,
            promotion_gates={},
            held_group_metrics={},
        )
        default_selections.extend(_simulate_routes(neutral, held, contract)["default_selections"])  # type: ignore[arg-type]
    default_metrics = {
        "acceptance_rate": sum(bool(row["accepted"]) for row in default_selections) / len(default_selections),
        "selected_non_regression_rate": sum(bool(row["selected_non_regression"]) for row in default_selections) / len(default_selections),
        "mean_selected_utility": sum(float(row["selected_utility"]) for row in default_selections) / len(default_selections),
        "mean_attempts": sum(float(row["attempts"]) for row in default_selections) / len(default_selections),
        "mean_cost_units": sum(float(row["cost_units"]) for row in default_selections) / len(default_selections),
    }
    gates = {
        "minimum_acceptance_rate": learned_metrics["acceptance_rate"] >= minimum_acceptance_rate,
        "acceptance_non_regression": learned_metrics["acceptance_rate"] >= default_metrics["acceptance_rate"],
        "capability_non_regression": learned_metrics["selected_non_regression_rate"] == 1.0,
        "utility_non_regression": learned_metrics["mean_selected_utility"] + utility_regression_tolerance >= default_metrics["mean_selected_utility"],
        "attempt_non_regression": learned_metrics["mean_attempts"]
        <= default_metrics["mean_attempts"],
        "cost_non_regression": learned_metrics["mean_cost_units"] <= default_metrics["mean_cost_units"] * (1.0 + cost_regression_fraction),
    }
    promoted = all(gates.values())
    mean, scale, intercept, coefficients = _fit_policy(records, contract, recipe_ids, alpha)
    held_metrics = {
        **{f"learned_{key}": value for key, value in learned_metrics.items()},
        **{f"default_{key}": value for key, value in default_metrics.items()},
    }
    policy = DemoFactoryPolicy(
        domain=contract.domain,
        contract_sha256=contract.fingerprint,
        context_fields=contract.context_fields,
        recipe_ids=recipe_ids,
        feature_mean=mean,
        feature_scale=scale,
        intercept=intercept,
        coefficients=coefficients,
        alpha=alpha,
        training_groups=groups,
        promoted=promoted,
        promotion_gates=gates,
        held_group_metrics=held_metrics,
    )

    preferences = []
    for episode_id, recipes in sorted(episodes.items()):
        baseline = recipes[contract.baseline_recipe_id]
        assessed = [
            (record, assess_record(contract, baseline.metrics, record))
            for record in recipes.values()
        ]
        accepted = [(record, assessment) for record, assessment in assessed if assessment.accepted]
        if not accepted:
            continue
        chosen, chosen_assessment = max(accepted, key=lambda item: item[1].utility)
        rejected = [
            {
                "recipe_id": record.recipe_id,
                "parameters": dict(record.recipe_parameters),
                "assessment": assessment.to_dict(),
            }
            for record, assessment in assessed
            if record.recipe_id != chosen.recipe_id
        ]
        if rejected:
            preferences.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "group_id": baseline.group_id,
                    "domain": baseline.domain,
                    "context": dict(baseline.context),
                    "chosen": {
                        "recipe_id": chosen.recipe_id,
                        "parameters": dict(chosen.recipe_parameters),
                        "assessment": chosen_assessment.to_dict(),
                    },
                    "rejected": rejected,
                }
            )
    evaluation = {
        "promoted": promoted,
        "gates": gates,
        "held_groups": list(groups),
        "learned": learned_metrics,
        "default": default_metrics,
        "folds": fold_reports,
    }
    return FactoryTrainingResult(policy, evaluation, tuple(preferences))
