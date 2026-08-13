from __future__ import annotations

import numpy as np

from scripts.build_real_flower_contact_supervision import (
    closest_mask_points,
    compose_exact_flower_instance,
    select_source_frame_positions,
)


class _CV2:
    DIST_L2 = 2
    DIST_LABEL_PIXEL = 1

    @staticmethod
    def dilate(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        # The composition test needs only a one-pixel flower, so a conservative
        # identity dilation is sufficient for this dependency-free fake.
        del kernel
        return mask

    @staticmethod
    def erode(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        del kernel
        return mask


def test_closest_mask_points_reports_overlap_without_distance_transform() -> None:
    first = np.zeros((5, 5), dtype=bool)
    second = np.zeros((5, 5), dtype=bool)
    first[2, 2] = True
    second[2, 2] = True

    result = closest_mask_points(_CV2, np, first, second)

    assert result["distance_pixels"] == 0.0
    assert result["overlap_pixels"] == 1
    assert result["first_xy"] == [2.0, 2.0]


def test_composition_restores_exact_flower_but_keeps_foreground_hand() -> None:
    source = np.full((4, 4, 3), 200, dtype=np.uint8)
    candidate = np.full((4, 4, 3), 20, dtype=np.uint8)
    flower = np.zeros((4, 4), dtype=bool)
    flower[1, 1] = True
    flower[2, 2] = True
    hand = np.zeros((4, 4), dtype=bool)
    hand[2, 2] = True

    output, restored, exact = compose_exact_flower_instance(
        _CV2, np, source, candidate, flower, hand
    )

    assert restored[1, 1]
    assert not restored[2, 2]
    assert np.all(output[1, 1] == 200)
    assert np.all(output[2, 2] == 20)
    assert exact[1, 1]
    assert not exact[2, 2]


def test_select_source_frame_positions_requires_complete_contiguous_subset() -> None:
    frames, positions = select_source_frame_positions(
        [270, 272, 273, 274, 280], [272, 275]
    )

    assert frames == [272, 273, 274]
    assert positions == [1, 2, 3]


def test_select_source_frame_positions_rejects_missing_frame() -> None:
    try:
        select_source_frame_positions([272, 274], [272, 275])
    except ValueError as error:
        assert "missing [273]" in str(error)
    else:
        raise AssertionError("missing source frame must fail")
