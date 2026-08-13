from __future__ import annotations

import cv2
import numpy as np

from scripts.compose_confidence_routed_flower_relighting import (
    build_safe_relight_mask,
    illumination_field,
    select_source_frame_positions,
)


def test_safe_mask_excludes_flowers_stem_and_hands() -> None:
    robot = np.ones((64, 64), dtype=bool)
    flowers = np.zeros_like(robot)
    stem = np.zeros_like(robot)
    hands = np.zeros_like(robot)
    flowers[20:25, 20:25] = True
    stem[30:40, 30:32] = True
    hands[40:50, 40:50] = True
    safe, protected = build_safe_relight_mask(
        cv2, np, robot, flowers, stem, hands
    )
    assert not bool(np.any(safe & protected))
    assert not bool(np.any(safe & flowers))
    assert not bool(np.any(safe & stem))
    assert not bool(np.any(safe & hands))
    assert bool(np.all(protected[flowers]))
    assert bool(np.all(protected[stem]))
    assert bool(np.all(protected[hands]))


def test_illumination_field_is_bounded_and_follows_proposal_direction() -> None:
    geometry = np.full((64, 64), 100, dtype=np.float32)
    proposal = np.full((64, 64), 140, dtype=np.float32)
    safe = np.ones((64, 64), dtype=bool)
    field = illumination_field(cv2, np, geometry, proposal, safe, 12.0)
    assert float(field.max()) <= 12.0
    assert float(field.min()) > 0.0


def test_select_source_frame_positions_returns_contiguous_subset() -> None:
    frames, positions = select_source_frame_positions(
        [270, 272, 273, 274, 280], [272, 275]
    )

    assert frames == [272, 273, 274]
    assert positions == [1, 2, 3]
