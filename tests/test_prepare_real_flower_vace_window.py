from __future__ import annotations

import numpy as np
import pytest

from scripts.prepare_real_flower_vace_window import localized_edit_mask, selected_indices


def test_selected_indices_are_absolute_and_stride_preserving() -> None:
    assert selected_indices(272, 17, 3, 660) == tuple(range(272, 321, 3))


def test_selected_indices_reject_window_past_end() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        selected_indices(650, 17, 3, 660)


def test_localized_edit_mask_protects_flower_pixels() -> None:
    person = np.zeros((4, 5), dtype=np.uint8)
    robot = np.zeros_like(person)
    flower = np.zeros_like(person)
    person[1:3, 1:4] = 255
    robot[0:2, 3:5] = 255
    flower[1, 3] = 255

    edit = localized_edit_mask(np, person, robot, flower)

    assert edit[1, 3] == 0
    assert edit[2, 2]
    assert edit[0, 4]
    assert not edit[3, 0]
