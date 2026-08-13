from __future__ import annotations

import pytest
import numpy as np

from scripts.evaluate_cosmos_droid_multiview import _record
from scripts.evaluate_droid_view_lora import evaluate_arrays


def test_record_requires_requested_nontraining_split() -> None:
    contract = {
        "records": [
            {
                "sample_id": "ep012-clip00",
                "split": "validation",
                "training_use": False,
            }
        ]
    }
    assert _record(contract, "ep012-clip00", "validation")["split"] == "validation"
    with pytest.raises(ValueError, match="non-training final_holdout"):
        _record(contract, "ep012-clip00", "final_holdout")


def test_record_rejects_training_sample() -> None:
    contract = {
        "records": [
            {"sample_id": "ep000-clip00", "split": "train", "training_use": True}
        ]
    }
    with pytest.raises(ValueError, match="non-training"):
        _record(contract, "ep000-clip00", "train")


def test_continuation_evaluation_accepts_explicit_real_condition_anchor() -> None:
    cv2 = pytest.importorskip("cv2")
    target = np.zeros((2, 16, 16, 3), dtype=np.float32)
    generated = target.copy()
    anchor = np.ones((16, 16, 3), dtype=np.float32)
    metrics = evaluate_arrays(cv2, np, generated, target, anchor=anchor)
    assert metrics["mean_full_frame_ssim"] == pytest.approx(1.0)
    assert metrics["static_anchor_ssim_gain"] > 0.9
