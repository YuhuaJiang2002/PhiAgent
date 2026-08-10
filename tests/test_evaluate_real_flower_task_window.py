from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_real_flower_task_window import (
    _masked_change,
    _masked_similarity,
    _motion_alignment,
    _validate_review,
)


def test_masked_similarity_and_change_use_only_selected_pixels() -> None:
    source = np.zeros((3, 3, 3), dtype=np.uint8)
    candidate = source.copy()
    candidate[1, 1] = 255
    center = np.zeros((3, 3), dtype=bool)
    center[1, 1] = True
    outside = ~center

    assert _masked_similarity(np, source, candidate, outside) == 1.0
    assert _masked_change(np, source, candidate, center) == 1.0


def test_motion_alignment_rewards_matching_temporal_motion() -> None:
    control = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    control[1][1, 1] = 100
    control[2][1, 2] = 100
    matching = [frame.copy() for frame in control]
    mismatched = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    masks = [np.ones((4, 4), dtype=bool) for _ in range(3)]

    assert _motion_alignment(np, control, matching, masks) == pytest.approx(1.0)
    assert _motion_alignment(np, control, matching, masks) > _motion_alignment(
        np, control, mismatched, masks
    )


def test_human_review_requires_strict_booleans() -> None:
    valid = {
        "reviewer": "frame audit",
        "candidates": {
            name: {
                "human_residue_absent": False,
                "two_robot_hands_visible": False,
                "causal_stem_contact_visible": False,
                "flowers_identity_preserved": True,
            }
            for name in ("zero_shot", "adapted")
        },
    }
    assert _validate_review(valid) == valid
    valid["candidates"]["adapted"]["human_residue_absent"] = 0
    with pytest.raises(ValueError, match="JSON boolean"):
        _validate_review(valid)
