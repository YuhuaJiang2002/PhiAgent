from __future__ import annotations

import numpy as np

from scripts.evaluate_flower_task_adapter import _motion_similarity, _similarity


def test_similarity_is_one_for_equal_pixels_and_decreases_with_error() -> None:
    target = np.zeros((8, 8, 3), dtype=np.uint8)
    close = np.full_like(target, 8)
    far = np.full_like(target, 64)

    assert _similarity(np, target, target) == 1.0
    assert _similarity(np, target, close) > _similarity(np, target, far)


def test_motion_similarity_rewards_matching_contact_motion() -> None:
    target = [np.zeros((5, 5, 3), dtype=np.uint8) for _ in range(3)]
    target[1][2, 2] = 100
    target[2][2, 3] = 100
    matching = [frame.copy() for frame in target]
    static = [np.zeros((5, 5, 3), dtype=np.uint8) for _ in range(3)]
    masks = [np.ones((5, 5), dtype=bool) for _ in range(3)]

    assert _motion_similarity(np, target, matching, masks) == 1.0
    assert _motion_similarity(np, target, matching, masks) > _motion_similarity(
        np, target, static, masks
    )
