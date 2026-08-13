from __future__ import annotations

import pytest

from scripts.evaluate_cosmos3_droid_wrist_to_exterior_i2v import (
    strict_gate,
    validation_record,
)


def _contract() -> dict[str, object]:
    return {
        "method": "cosmos3_nano_droid_wrist_only_to_exterior_i2v_sft_dataset",
        "leakage_checks": {
            "final_holdout_used_for_training": False,
            "final_holdout_used_for_checkpoint_selection": False,
            "validation_future_frames_are_model_inputs": False,
            "condition_contains_exterior_pixels": False,
            "condition_contains_real_wrist_pixels_only": True,
        },
        "records": [
            {
                "sample_id": "ep012-wrist-to-exterior-1",
                "split": "validation",
                "training_use": False,
            }
        ],
    }


def test_validation_record_requires_true_wrist_only_leakage_contract() -> None:
    row = validation_record(_contract(), "ep012-wrist-to-exterior-1")
    assert row["training_use"] is False
    bad = _contract()
    bad["leakage_checks"]["condition_contains_exterior_pixels"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="leakage gates"):
        validation_record(bad, "ep012-wrist-to-exterior-1")


def test_strict_gate_rejects_subject_or_view_motion_failure() -> None:
    good = {
        "mean_full_frame_ssim": 0.70,
        "minimum_full_frame_ssim": 0.40,
        "mean_subject_roi_ssim": 0.60,
        "minimum_subject_roi_ssim": 0.30,
        "mean_subject_edge_f1": 0.50,
        "motion_correlation": 0.45,
        "motion_magnitude_ratio": 1.0,
        "static_anchor_ssim_gain": 0.04,
    }
    assert all(strict_gate(good).values())
    bad = {**good, "mean_subject_roi_ssim": 0.2, "motion_correlation": -0.1}
    gates = strict_gate(bad)
    assert gates["mean_subject_roi_ssim"] is False
    assert gates["motion_correlation"] is False
