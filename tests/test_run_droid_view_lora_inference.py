from __future__ import annotations

import pytest

from scripts.run_droid_view_lora_inference import select_heldout_example


def test_selects_only_explicit_heldout_record() -> None:
    contract = {
        "holdout_records": [
            {
                "episode_index": 21,
                "training_use": False,
                "targets": {"target_a": {"prompt": "demo"}},
            }
        ]
    }
    record, target = select_heldout_example(contract, 21, "target_a")
    assert record["training_use"] is False
    assert target["prompt"] == "demo"


def test_rejects_record_marked_for_training() -> None:
    contract = {
        "holdout_records": [
            {
                "episode_index": 21,
                "training_use": True,
                "targets": {"target_a": {}},
            }
        ]
    }
    with pytest.raises(ValueError, match="not held out"):
        select_heldout_example(contract, 21, "target_a")
