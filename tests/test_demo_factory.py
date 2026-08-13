from __future__ import annotations

import json

import pytest

from phiagent.training.demo_factory import (
    DemoFactoryPolicy,
    FactoryContract,
    FactoryRecord,
    assess_record,
    train_grouped_router,
)


def _contract() -> FactoryContract:
    return FactoryContract.from_dict(
        {
            "domain": "robot-demo",
            "baseline_recipe_id": "raw",
            "recipe_order": ["raw", "generic", "targeted"],
            "context_fields": ["motion", "object"],
            "metric_weights": {"motion": 1.0, "object": 1.0},
            "hard_thresholds": {"motion": 0.75, "object": 0.75},
            "non_regression_tolerances": {"motion": 0.01, "object": 0.01},
            "cost_budget_units": 3.0,
            "cost_weight": 0.01,
            "rejection_penalty": 2.0,
            "human_review_required": True,
        }
    )


def _record(
    episode: str,
    group: str,
    recipe: str,
    score: float,
    *,
    baseline: float = 0.5,
) -> FactoryRecord:
    return FactoryRecord(
        episode_id=episode,
        group_id=group,
        domain="robot-demo",
        recipe_id=recipe,
        recipe_parameters={"strength": {"raw": 0, "generic": 1, "targeted": 2}[recipe]},
        context={"motion": baseline, "object": baseline},
        metrics={"motion": score, "object": score},
        cost_units=1.0,
        human_review_passed=True,
        video=f"/immutable/{episode}-{recipe}.mp4",
        video_sha256=("a" if recipe == "raw" else "b" if recipe == "generic" else "c")
        * 64,
    )


def _records() -> tuple[FactoryRecord, ...]:
    rows = []
    for group in ("scene-a", "scene-b"):
        for suffix in ("one", "two"):
            episode = f"{group}-{suffix}"
            rows.extend(
                (
                    _record(episode, group, "raw", 0.5),
                    _record(episode, group, "generic", 0.6),
                    _record(episode, group, "targeted", 0.9),
                )
            )
    return tuple(rows)


def test_assessment_rejects_hidden_capability_regression() -> None:
    contract = _contract()
    baseline = _record("episode", "scene-a", "raw", 0.9, baseline=0.9)
    candidate = FactoryRecord(
        **{
            **_record("episode", "scene-a", "targeted", 0.9, baseline=0.9).__dict__,
            "metrics": {"motion": 0.70, "object": 1.0},
        }
    )

    assessment = assess_record(contract, baseline.metrics, candidate)

    assert not assessment.accepted
    assert assessment.automatic_gates_passed is False
    assert assessment.non_regression_passed is False
    assert dict(assessment.non_regression_excess)["motion"] == pytest.approx(0.19)


def test_grouped_router_promotes_cost_reducing_recipe_order(tmp_path) -> None:
    result = train_grouped_router(_records(), _contract(), minimum_acceptance_rate=1.0)

    assert result.policy.promoted
    assert result.evaluation["learned"]["acceptance_rate"] == 1.0
    assert result.evaluation["learned"]["mean_attempts"] == 2.0
    assert result.evaluation["default"]["mean_attempts"] == 3.0
    assert result.policy.rank({"motion": 0.5, "object": 0.5}, ("generic", "targeted"))[0][0] == "targeted"
    assert len(result.preferences) == 4

    checkpoint = tmp_path / "policy.json"
    checkpoint.write_text(json.dumps(result.policy.to_dict()))
    loaded = DemoFactoryPolicy.load(checkpoint)
    assert loaded.to_dict() == result.policy.to_dict()


def test_grouped_router_rejects_scene_leakage() -> None:
    one_group = tuple(record for record in _records() if record.group_id == "scene-a")

    with pytest.raises(ValueError, match="at least two groups"):
        train_grouped_router(one_group, _contract())


def test_record_context_must_be_bound_to_baseline() -> None:
    rows = list(_records())
    rows[1] = FactoryRecord(**{**rows[1].__dict__, "context": {"motion": 0.4, "object": 0.5}})

    with pytest.raises(ValueError, match="baseline-bound"):
        train_grouped_router(rows, _contract())
