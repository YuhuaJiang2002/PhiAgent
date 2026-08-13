from __future__ import annotations

import pytest

from scripts.evaluate_minimax_h3_ego_bottle_window import (
    action_support_mask,
    motion_adherence_score,
)


def test_action_support_mask_stays_in_named_camera_pixels() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = np.zeros((80, 120, 3), dtype=np.uint8)
    control = source.copy()
    control[35:50, 55:70] = 180

    mask = action_support_mask(cv2, np, source, control)

    assert mask.shape == (80, 120)
    assert mask[42, 62] == 255
    assert mask[0, 0] == 0
    assert float(np.mean(mask > 0)) < 0.3


def test_motion_adherence_is_one_for_identical_video() -> None:
    np = pytest.importorskip("numpy")
    frames = [
        np.full((20, 30, 3), index * 12, dtype=np.uint8)
        for index in range(5)
    ]
    masks = [np.full((20, 30), 255, dtype=np.uint8) for _ in frames]

    score, detail = motion_adherence_score(np, frames, frames, masks)

    assert score == pytest.approx(1.0)
    assert detail["mean_absolute_motion_error"] == pytest.approx(0.0)
