from __future__ import annotations

import pytest

from scripts.prepare_cosmos3_droid_sft_dataset import (
    sft_record,
    structured_caption,
    validate_contract,
)


def _contract() -> dict[str, object]:
    return {
        "video_contract": {"width": 768, "height": 432, "fps": 16, "frames": 97},
        "leakage_checks": {"final_holdout_used_for_training": False},
        "records": [
            {"episode_index": 1, "split": "train"},
            {"episode_index": 2, "split": "validation"},
            {"episode_index": 3, "split": "final_holdout"},
        ],
    }


def _record() -> dict[str, object]:
    return {
        "sample_id": "ep012-clip00",
        "raw_task_text": "Pick up the lid and put it on the pot",
        "prompt": "A multiview DROID robot picks up the lid and puts it on the pot.",
    }


def test_contract_requires_episode_disjoint_holdout() -> None:
    validate_contract(_contract())
    bad = _contract()
    bad["records"] = [
        {"episode_index": 1, "split": "train"},
        {"episode_index": 1, "split": "validation"},
    ]
    with pytest.raises(ValueError, match="episode leakage"):
        validate_contract(bad)


def test_structured_caption_names_layout_and_identity_constraints() -> None:
    caption = structured_caption(_record())
    assert "top-left" in caption["cinematography"]["framing"]
    assert "wrist" in caption["cinematography"]["framing"]
    assert "identity" in caption["subjects"][0]["state_changes"]
    assert caption["resolution"] == {"W": 768, "H": 432}
    assert caption["fps"] == 16


def test_sft_record_uses_complete_inclusive_frame_window() -> None:
    row = sft_record(_record(), "videos/ep012-clip00.mp4")
    assert row["vision_path"] == "videos/ep012-clip00.mp4"
    assert row["t2w_windows"][0]["start_frame"] == 0
    assert row["t2w_windows"][0]["end_frame"] == 96
    assert row["t2w_windows"][0]["caption_json"]["aspect_ratio"] == "16,9"
