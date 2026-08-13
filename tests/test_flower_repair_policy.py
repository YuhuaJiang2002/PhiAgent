from __future__ import annotations

import json

import pytest

from phiagent.training.flower_repair_policy import (
    FlowerRepairPolicy,
    NonRegressionContract,
    encode_features,
    feature_names,
)


RAW = {
    "background_lock": 0.1,
    "object_lock": 0.2,
    "subject_replacement": 0.9,
    "robot_identity": 0.7,
    "motion_preservation": 0.6,
    "temporal_consistency": 0.8,
    "epl_minimum": 0.5,
}


def _repair(name: str, restore: bool) -> dict[str, object]:
    return {
        "name": name,
        "hard_background_lock": True,
        "restore_source_flowers": restore,
        "exclude_source_face_from_flower_restore": False,
        "mask_dilation_pixels": 3,
        "flower_dilation_pixels": 2 if restore else 0,
        "face_box_margin_pixels": 0,
    }


def test_feature_contract_contains_context_recipe_and_interactions() -> None:
    encoded = encode_features(RAW, _repair("restore", True))
    assert len(encoded) == len(feature_names()) == 55
    assert encoded[:7] == tuple(RAW.values())
    assert encoded[7:13] == (1.0, 1.0, 0.0, 1.0, 1.0, 0.0)


def test_policy_ranks_recipes_and_round_trips(tmp_path) -> None:
    coefficients = [0.0] * len(feature_names())
    coefficients[8] = 1.0  # repair:restore_source_flowers
    policy = FlowerRepairPolicy(
        feature_mean=(0.0,) * len(feature_names()),
        feature_scale=(1.0,) * len(feature_names()),
        intercept=0.0,
        coefficients=tuple(coefficients),
        alpha=0.01,
        training_actions=("insert-flower", "handover-flower"),
        held_out_action="inspect-flower",
    )
    ranked = policy.rank(RAW, (_repair("background", False), _repair("restore", True)))
    assert ranked[0][0]["name"] == "restore"
    checkpoint = tmp_path / "policy.json"
    checkpoint.write_text(json.dumps(policy.to_dict()))
    loaded = FlowerRepairPolicy.load(checkpoint)
    assert loaded.to_dict() == policy.to_dict()


def test_policy_rejects_action_leakage() -> None:
    with pytest.raises(ValueError, match="held-out"):
        FlowerRepairPolicy(
            feature_mean=(0.0,) * len(feature_names()),
            feature_scale=(1.0,) * len(feature_names()),
            intercept=0.0,
            coefficients=(0.0,) * len(feature_names()),
            alpha=0.01,
            training_actions=("inspect-flower",),
            held_out_action="inspect-flower",
        )


def test_non_regression_contract_rejects_hidden_motion_collapse() -> None:
    contract = NonRegressionContract()
    aggregate_winner = {
        **RAW,
        "background_lock": 1.0,
        "object_lock": 1.0,
        "motion_preservation": 0.48,
        "epl_minimum": 0.30,
        "subject_replacement": 0.80,
    }
    assessment = contract.assess(RAW, aggregate_winner)
    assert not assessment.passed
    assert dict(assessment.excess_regressions)["motion_preservation"] == pytest.approx(0.11)
    assert assessment.total_excess > 0.2


def test_non_regression_contract_allows_bounded_motion_noise() -> None:
    contract = NonRegressionContract()
    candidate = {
        **RAW,
        "background_lock": 1.0,
        "motion_preservation": 0.595,
        "epl_minimum": 0.495,
    }
    assessment = contract.assess(RAW, candidate)
    assert assessment.passed
    assert assessment.minimum_margin == pytest.approx(0.005)
