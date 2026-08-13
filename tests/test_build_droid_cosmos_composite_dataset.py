from __future__ import annotations

import pytest

from scripts.build_droid_cosmos_composite_dataset import (
    FRAMES,
    FPS,
    HEIGHT,
    TRAINING_WINDOW_FRAMES,
    VIEW_LAYOUT,
    WIDTH,
    droid_multiview_prompt,
    normalize_task,
    task_condition_kind,
    validate_source_video_contract,
)


def test_native_droid_layout_and_video_contract() -> None:
    assert VIEW_LAYOUT == {
        "top_left": "exterior_1",
        "top_right": "exterior_2",
        "bottom_left": "wrist",
        "bottom_right": "inactive_black",
    }
    assert (WIDTH, HEIGHT, FPS, FRAMES) == (768, 432, 16, 97)
    assert TRAINING_WINDOW_FRAMES == 93
    assert (FRAMES - 1) % 4 == 0


def test_prompt_describes_each_view_and_task_twice() -> None:
    prompt = droid_multiview_prompt("Close the drawer.")
    assert prompt.count("close the drawer") == 2
    assert "top-left" in prompt
    assert "top-right" in prompt
    assert "first-person perspective" in prompt
    assert "black screen" in prompt


def test_missing_task_uses_disclosed_neutral_fallback() -> None:
    assert normalize_task("  .  ") == "perform the manipulation task"
    assert task_condition_kind("  .  ") == "FIXED NEUTRAL FALLBACK"
    assert task_condition_kind("Close the drawer") == "REAL DATASET ANNOTATION"


def test_source_contract_must_be_native_16_fps() -> None:
    validate_source_video_contract({"video_contract": {"fps": 16, "frames": 97}})
    with pytest.raises(ValueError, match="sampled directly at 16 fps"):
        validate_source_video_contract({"video_contract": {"fps": 8, "frames": 97}})
