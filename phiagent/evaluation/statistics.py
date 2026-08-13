"""Cluster-aware statistical gates for reproducible PhiAgent experiments."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping


def paired_cluster_bootstrap_lower_bound(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    cluster_by_trial: Mapping[str, str],
    *,
    higher_is_better: bool,
    seed: int,
    iterations: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap paired gains at the declared independent-cluster level."""

    if set(candidate) != set(baseline) or set(candidate) != set(cluster_by_trial):
        raise ValueError("candidate, baseline, and cluster trial identifiers must match")
    if not candidate:
        raise ValueError("cluster bootstrap requires at least one paired trial")
    if iterations < 100 or not 0.5 < confidence < 1.0:
        raise ValueError("cluster bootstrap requires >=100 iterations and valid confidence")
    sign = 1.0 if higher_is_better else -1.0
    grouped: dict[str, list[float]] = {}
    for trial_id in sorted(candidate):
        candidate_value = float(candidate[trial_id])
        baseline_value = float(baseline[trial_id])
        cluster = str(cluster_by_trial[trial_id]).strip()
        if (
            not cluster
            or not math.isfinite(candidate_value)
            or not math.isfinite(baseline_value)
        ):
            raise ValueError("cluster IDs and paired values must be non-empty and finite")
        grouped.setdefault(cluster, []).append(sign * (candidate_value - baseline_value))
    if len(grouped) < 2:
        raise ValueError("cluster bootstrap requires at least two independent clusters")
    cluster_gains = [
        statistics.fmean(values) for _, values in sorted(grouped.items())
    ]
    generator = random.Random(seed)
    bootstrapped = sorted(
        statistics.fmean(
            cluster_gains[generator.randrange(len(cluster_gains))]
            for _ in cluster_gains
        )
        for _ in range(iterations)
    )
    lower_index = max(0, math.floor((1.0 - confidence) * iterations) - 1)
    non_positive = sum(value <= 0.0 for value in bootstrapped)
    return {
        "trials": len(candidate),
        "independent_clusters": len(cluster_gains),
        "mean_oriented_gain": statistics.fmean(cluster_gains),
        "lower_confidence_bound": bootstrapped[lower_index],
        "one_sided_p_value": (non_positive + 1) / (iterations + 1),
        "confidence": confidence,
        "bootstrap_iterations": iterations,
    }


def holm_bonferroni(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Apply Holm's step-down family-wise error correction."""

    if not p_values:
        raise ValueError("Holm correction requires at least one hypothesis")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    values = {str(name): float(value) for name, value in p_values.items()}
    if any(
        not name or not math.isfinite(value) or not 0.0 <= value <= 1.0
        for name, value in values.items()
    ):
        raise ValueError("hypothesis names and p-values are invalid")
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    rejected: dict[str, bool] = {}
    rows = []
    continue_rejecting = True
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered, start=1):
        threshold = alpha / (total - rank + 1)
        reject = continue_rejecting and p_value <= threshold
        if not reject:
            continue_rejecting = False
        rejected[name] = reject
        rows.append(
            {
                "hypothesis": name,
                "p_value": p_value,
                "rank": rank,
                "threshold": threshold,
                "rejected": reject,
            }
        )
    return {
        "alpha": alpha,
        "family_size": total,
        "all_rejected": all(rejected.values()),
        "rejected": rejected,
        "ordered_tests": rows,
    }

