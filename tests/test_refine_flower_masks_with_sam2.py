from __future__ import annotations

import pytest

from scripts.refine_flower_masks_with_sam2 import compose_refined_track


def test_sam2_refinement_requires_independent_evidence_inside_person() -> None:
    np = pytest.importorskip("numpy")
    tracked = np.ones((2, 4), dtype=bool)
    person = np.zeros_like(tracked)
    person[:, 1:4] = True
    hands = np.zeros_like(tracked)
    hands[1, 2] = True
    sam2_object = np.zeros_like(tracked)
    sam2_object[0, 1] = True
    appearance = np.zeros_like(tracked)
    appearance[0, 2] = True

    refined = compose_refined_track(
        np,
        tracked=tracked,
        person=person,
        hands=hands,
        sam2_object=sam2_object,
        appearance_core=appearance,
    )

    assert np.all(refined[:, 0])
    assert refined[0, 1]
    assert refined[0, 2]
    assert not refined[0, 3]
    assert not refined[1, 2]
