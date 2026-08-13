from __future__ import annotations

from scripts.audit_cosmos3_droid_wrist_dataset import lineage_gates


def test_lineage_gates_require_wrist_and_future_exterior_similarity() -> None:
    good = {
        "condition_to_wrist_ssim": 0.99,
        "derived_frame0_to_wrist_ssim": 0.99,
        "minimum_future_to_exterior_ssim": 0.98,
        "wrist_over_exterior_margin": 0.20,
    }
    assert all(lineage_gates(good).values())
    bad = {**good, "minimum_future_to_exterior_ssim": 0.8}
    assert lineage_gates(bad)["minimum_future_to_exterior_ssim"] is False


def test_lineage_gates_reject_exterior_like_condition() -> None:
    metrics = {
        "condition_to_wrist_ssim": 0.99,
        "derived_frame0_to_wrist_ssim": 0.99,
        "minimum_future_to_exterior_ssim": 0.98,
        "wrist_over_exterior_margin": 0.01,
    }
    assert lineage_gates(metrics)["wrist_over_exterior_margin"] is False
