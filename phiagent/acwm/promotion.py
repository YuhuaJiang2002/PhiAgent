"""Paired, frozen-test promotion gate for AC-WM candidates."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from typing import Any, Mapping, Sequence


def _metric_samples(
    model: Mapping[str, Any], suite: str, metric: str
) -> tuple[str, dict[str, float]]:
    suites = model.get("suites")
    if not isinstance(suites, Mapping) or suite not in suites:
        raise ValueError(f"model {model.get('model_id')!r} lacks suite {suite!r}")
    payload = suites[suite]
    if not isinstance(payload, Mapping) or payload.get("split_role") != "test":
        raise ValueError(f"suite {suite!r} must be a frozen test split")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or metric not in metrics:
        raise ValueError(f"suite {suite!r} lacks metric {metric!r}")
    item = metrics[metric]
    if not isinstance(item, Mapping):
        raise ValueError(f"metric {suite}/{metric} must be an object")
    direction = str(item.get("direction"))
    if direction not in {"higher", "lower"}:
        raise ValueError(f"metric {suite}/{metric} direction must be higher or lower")
    raw = item.get("samples")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"metric {suite}/{metric} requires per-trial samples")
    samples = {str(key): float(value) for key, value in raw.items()}
    if any(not math.isfinite(value) for value in samples.values()):
        raise ValueError(f"metric {suite}/{metric} contains non-finite samples")
    return direction, samples


def paired_bootstrap_lower_bound(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    higher_is_better: bool,
    seed: int,
    iterations: int = 5000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Compute a deterministic one-sided bootstrap lower confidence bound."""

    if set(candidate) != set(baseline):
        raise ValueError("candidate and baseline trial identifiers do not match")
    if iterations < 100 or not 0.5 < confidence < 1:
        raise ValueError("bootstrap requires >=100 iterations and 0.5 < confidence < 1")
    trial_ids = sorted(candidate)
    sign = 1.0 if higher_is_better else -1.0
    gains = [sign * (candidate[key] - baseline[key]) for key in trial_ids]
    generator = random.Random(seed)
    bootstrapped = sorted(
        statistics.fmean(gains[generator.randrange(len(gains))] for _ in gains)
        for _ in range(iterations)
    )
    lower_index = max(0, math.floor((1.0 - confidence) * iterations) - 1)
    return {
        "trials": len(gains),
        "mean_oriented_gain": statistics.fmean(gains),
        "lower_confidence_bound": bootstrapped[lower_index],
        "confidence": confidence,
        "bootstrap_iterations": iterations,
    }


def evaluate_promotion(
    candidate: Mapping[str, Any],
    baselines: Sequence[Mapping[str, Any]],
    *,
    required_suites: Mapping[str, Sequence[str]],
    minimum_trials: int = 20,
    minimum_gain: float = 0.0,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260811,
) -> dict[str, Any]:
    """Require significant paired gains over every baseline on every metric."""

    if not baselines:
        raise ValueError("promotion requires at least one baseline")
    if minimum_trials < 2 or not math.isfinite(minimum_gain):
        raise ValueError("invalid promotion thresholds")
    comparisons: list[dict[str, Any]] = []
    accepted = True
    for baseline in baselines:
        baseline_id = str(baseline.get("model_id", "unnamed-baseline"))
        for suite, metrics in required_suites.items():
            for metric in metrics:
                candidate_direction, candidate_samples = _metric_samples(
                    candidate, suite, metric
                )
                baseline_direction, baseline_samples = _metric_samples(
                    baseline, suite, metric
                )
                if candidate_direction != baseline_direction:
                    raise ValueError(
                        f"direction mismatch for {baseline_id}/{suite}/{metric}"
                    )
                identity = f"{seed}:{baseline_id}:{suite}:{metric}".encode()
                comparison_seed = int.from_bytes(
                    hashlib.sha256(identity).digest()[:8], "big"
                )
                evidence = paired_bootstrap_lower_bound(
                    candidate_samples,
                    baseline_samples,
                    higher_is_better=candidate_direction == "higher",
                    seed=comparison_seed,
                    iterations=bootstrap_iterations,
                    confidence=confidence,
                )
                passed = (
                    int(evidence["trials"]) >= minimum_trials
                    and float(evidence["mean_oriented_gain"]) > minimum_gain
                    and float(evidence["lower_confidence_bound"]) > minimum_gain
                )
                accepted = accepted and passed
                comparisons.append(
                    {
                        "baseline": baseline_id,
                        "suite": suite,
                        "metric": metric,
                        "direction": candidate_direction,
                        "passed": passed,
                        **evidence,
                    }
                )
    return {
        "schema_version": "1.0.0",
        "accepted": accepted,
        "status": "WORKING" if accepted else "PARTIAL",
        "candidate": str(candidate.get("model_id", "unnamed-candidate")),
        "baseline_count": len(baselines),
        "required_suites": {
            suite: list(metrics) for suite, metrics in required_suites.items()
        },
        "minimum_trials": minimum_trials,
        "minimum_gain": minimum_gain,
        "comparisons": comparisons,
        "claim_boundary": (
            "Acceptance establishes dominance only over the named baselines, frozen suites, "
            "metrics, and trials in this artifact; it is not an unrestricted SOTA claim."
        ),
    }
