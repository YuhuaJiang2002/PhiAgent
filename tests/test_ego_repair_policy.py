from __future__ import annotations

import pytest

from phiagent.training.ego_repair_policy import (
    EgoNonRegressionContract,
    EgoRepairPolicy,
    encode_features,
    feature_names,
)


def _score(value: float = 0.8) -> dict[str, float]:
    return {
        "background_lock": value,
        "object_lock": value,
        "subject_replacement": value,
        "robot_identity": value,
        "motion_preservation": value,
        "temporal_consistency": value,
        "epl_minimum": value,
    }


def _repair(name: str, dilation: float, blur: float) -> dict[str, object]:
    return {
        "name": name,
        "support_dilation_pixels": dilation,
        "alpha_blur_sigma": blur,
    }


def test_ego_repair_feature_contract_is_bounded_and_stable() -> None:
    encoded = encode_features(_score(), _repair("tight", 12.0, 6.0))

    assert len(encoded) == len(feature_names())
    assert all(0.0 <= value <= 1.0 for value in encoded)


def test_ego_non_regression_rejects_motion_collapse() -> None:
    baseline = _score()
    candidate = _score()
    candidate["motion_preservation"] = 0.2

    assessment = EgoNonRegressionContract().assess(baseline, candidate)

    assert assessment["passed"] is False
    assert assessment["excess_regressions"]["motion_preservation"] > 0.5


def test_ego_repair_policy_roundtrip_and_rank(tmp_path) -> None:
    names = feature_names()
    coefficients = [0.0] * len(names)
    coefficients[names.index("repair:support_dilation_pixels")] = 1.0
    policy = EgoRepairPolicy(
        feature_mean=(0.0,) * len(names),
        feature_scale=(1.0,) * len(names),
        intercept=0.0,
        coefficients=tuple(coefficients),
        alpha=0.01,
        training_actions=("pour-bottle", "shake-bottle"),
        held_out_action="handover-bottle",
    )
    checkpoint = tmp_path / "policy.json"
    checkpoint.write_text(__import__("json").dumps(policy.to_dict()))

    loaded = EgoRepairPolicy.load(checkpoint)
    ranked = loaded.rank(
        _score(),
        [_repair("tight", 0.0, 3.0), _repair("wide", 20.0, 9.0)],
    )

    assert ranked[0][0]["name"] == "wide"
    assert loaded.to_dict() == policy.to_dict()


def test_ego_repair_policy_rejects_flower_checkpoint() -> None:
    with pytest.raises(ValueError, match="not an Ego bottle"):
        EgoRepairPolicy.from_dict({"method": "flower_repair_ridge_utility_ranker"})
