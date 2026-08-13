from __future__ import annotations

import pytest

from phiagent.evaluation.statistics import (
    holm_bonferroni,
    paired_cluster_bootstrap_lower_bound,
)


def test_cluster_bootstrap_counts_episodes_instead_of_clips() -> None:
    candidate = {
        "episode-a-clip-0": 0.9,
        "episode-a-clip-1": 0.8,
        "episode-b-clip-0": 0.7,
    }
    baseline = {trial: value - 0.1 for trial, value in candidate.items()}
    clusters = {
        "episode-a-clip-0": "episode-a",
        "episode-a-clip-1": "episode-a",
        "episode-b-clip-0": "episode-b",
    }

    result = paired_cluster_bootstrap_lower_bound(
        candidate,
        baseline,
        clusters,
        higher_is_better=True,
        seed=42,
        iterations=100,
    )

    assert result["trials"] == 3
    assert result["independent_clusters"] == 2
    assert result["mean_oriented_gain"] == pytest.approx(0.1)
    assert result["lower_confidence_bound"] == pytest.approx(0.1)


def test_cluster_bootstrap_rejects_one_pseudo_independent_episode() -> None:
    with pytest.raises(ValueError, match="two independent"):
        paired_cluster_bootstrap_lower_bound(
            {"clip-a": 1.0, "clip-b": 1.0},
            {"clip-a": 0.0, "clip-b": 0.0},
            {"clip-a": "episode", "clip-b": "episode"},
            higher_is_better=True,
            seed=42,
            iterations=100,
        )


def test_holm_bonferroni_stops_after_first_non_rejection() -> None:
    result = holm_bonferroni(
        {"strong": 0.001, "borderline": 0.03, "weak": 0.04},
        alpha=0.05,
    )

    assert result["rejected"] == {
        "strong": True,
        "borderline": False,
        "weak": False,
    }
    assert result["all_rejected"] is False
