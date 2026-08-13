from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare_cosmos3_droid_wrist_to_exterior_sft_dataset import (
    build_video_command,
    sft_record,
    structured_caption,
    validate_probe,
    validate_source_contract,
    view_switch_filter,
)


def _contract() -> dict[str, object]:
    return {
        "video_contract": {"width": 768, "height": 432, "fps": 16, "frames": 97},
        "layout": {
            "top_left": "exterior_1",
            "top_right": "exterior_2",
            "bottom_left": "wrist",
            "bottom_right": "inactive_black",
        },
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
    }


def test_source_contract_requires_named_layout_and_episode_isolation() -> None:
    validate_source_contract(_contract())
    bad = _contract()
    bad["layout"] = {"bottom_left": "wrist"}
    with pytest.raises(ValueError, match="unnamed composite layout"):
        validate_source_contract(bad)


def test_filter_uses_only_wrist_frame_zero_then_named_exterior_future() -> None:
    first = view_switch_filter("exterior_1")
    second = view_switch_filter("exterior_2")
    assert "crop=384:216:0:216" in first
    assert "select='eq(n\\,0)'" in first
    assert "crop=384:216:0:0" in first
    assert "crop=384:216:384:0" in second
    assert "select='gte(n\\,1)'" in second


def test_video_command_is_fixed_97_frame_real_view_switch(tmp_path: Path) -> None:
    command = build_video_command(
        Path("/usr/bin/ffmpeg"),
        tmp_path / "source.mp4",
        tmp_path / "derived.mp4",
        "exterior_2",
        18,
    )
    assert command[command.index("-frames:v") + 1] == "97"
    assert command[command.index("-r") + 1] == "16"
    assert command[command.index("-crf") + 1] == "18"
    assert "crop=384:216:384:0" in command[command.index("-filter_complex") + 1]


def test_caption_and_sft_row_disclose_true_wrist_to_third_transition() -> None:
    caption = structured_caption(_record(), "exterior_1")
    assert "wrist" in caption["cinematography"]["view_transition"]
    assert "every future frame" in caption["cinematography"]["view_transition"]
    row = sft_record(
        _record(),
        "ep012-clip00-wrist-to-exterior-1",
        "videos/sample.mp4",
        "exterior_1",
    )
    assert row["t2w_windows"][0]["end_frame"] == 96
    assert row["t2w_windows"][0]["caption_json"]["cinematography"]["camera_angle"] == "exterior_1"


def test_probe_contract_is_exact() -> None:
    validate_probe(
        {"width": 768, "height": 432, "avg_frame_rate": "16/1", "nb_read_frames": "97"}
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_probe(
            {"width": 768, "height": 432, "avg_frame_rate": "8/1", "nb_read_frames": "97"}
        )
