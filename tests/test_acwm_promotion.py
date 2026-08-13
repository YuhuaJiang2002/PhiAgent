import pytest

from phiagent.acwm.promotion import evaluate_promotion


REQUIRED = {
    "worldarena_test": ("action_following", "physics_adherence"),
    "cross_embodiment_test": ("action_following",),
    "real_robot_test": ("task_success", "safety_violation_free"),
}


def _model(model_id: str, value: float) -> dict:
    trials = {f"trial-{index:02d}": value for index in range(20)}
    return {
        "model_id": model_id,
        "suites": {
            suite: {
                "split_role": "test",
                "metrics": {
                    metric: {
                        "direction": "higher",
                        "samples": dict(trials),
                    }
                    for metric in metrics
                },
            }
            for suite, metrics in REQUIRED.items()
        },
    }


def test_promotion_requires_significant_gain_over_every_baseline() -> None:
    result = evaluate_promotion(
        _model("candidate", 0.9),
        [_model("bwm", 0.7), _model("ctrl-world", 0.8)],
        required_suites=REQUIRED,
        bootstrap_iterations=100,
    )

    assert result["accepted"] is True
    assert all(comparison["passed"] for comparison in result["comparisons"])


def test_one_regression_blocks_the_global_claim() -> None:
    candidate = _model("candidate", 0.9)
    candidate["suites"]["real_robot_test"]["metrics"]["task_success"]["samples"] = {
        f"trial-{index:02d}": 0.0 for index in range(20)
    }

    result = evaluate_promotion(
        candidate,
        [_model("bwm", 0.7)],
        required_suites=REQUIRED,
        bootstrap_iterations=100,
    )

    assert result["accepted"] is False
    assert any(not comparison["passed"] for comparison in result["comparisons"])


def test_missing_real_robot_suite_cannot_be_promoted() -> None:
    candidate = _model("candidate", 0.9)
    del candidate["suites"]["real_robot_test"]

    with pytest.raises(ValueError, match="real_robot_test"):
        evaluate_promotion(
            candidate,
            [_model("bwm", 0.7)],
            required_suites=REQUIRED,
            bootstrap_iterations=100,
        )
