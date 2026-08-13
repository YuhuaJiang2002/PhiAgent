from __future__ import annotations

import numpy as np
import pytest

from scripts.track_real_flower_stem_instances import (
    merge_directional_track_mask,
    select_anchor_candidate,
    validate_seed_spec,
)


def _spec() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "coordinate_frame": "camera:source_video_pixels",
        "source_frame_indices": [10, 13, 16],
        "instances": [
            {
                "instance_id": "stem-1",
                "object_id": 1,
                "anchor_source_frame": 13,
                "box_xyxy": [1, 1, 6, 6],
                "positive_points_xy": [[3, 3]],
                "negative_points_xy": [[7, 7]],
                "minimum_area_pixels": 2,
                "maximum_area_pixels": 20,
            }
        ],
    }


def test_seed_spec_requires_named_frame_and_unique_instance() -> None:
    assert validate_seed_spec(_spec())["source_frame_indices"] == [10, 13, 16]
    invalid = _spec()
    invalid["instances"][0]["anchor_source_frame"] = 12  # type: ignore[index]
    with pytest.raises(ValueError, match="anchor"):
        validate_seed_spec(invalid)


def test_seed_spec_expands_compact_frame_range() -> None:
    spec = _spec()
    spec.pop("source_frame_indices")
    spec["source_frame_range"] = [10, 17, 3]
    validated = validate_seed_spec(spec)
    assert validated["source_frame_indices"] == [10, 13, 16]


def test_seed_spec_accepts_unique_correction_prompt() -> None:
    spec = _spec()
    spec["instances"][0]["correction_prompts"] = [  # type: ignore[index]
        {
            "anchor_source_frame": 16,
            "box_xyxy": [2, 2, 7, 7],
            "positive_points_xy": [[4, 4]],
            "negative_points_xy": [[8, 8]],
        }
    ]
    validated = validate_seed_spec(spec)
    assert validated["instances"][0]["correction_prompts"][0]["anchor_source_frame"] == 16


def test_seed_spec_rejects_duplicate_correction_frame() -> None:
    spec = _spec()
    spec["instances"][0]["correction_prompts"] = [  # type: ignore[index]
        {
            "anchor_source_frame": 13,
            "box_xyxy": [2, 2, 7, 7],
            "positive_points_xy": [[4, 4]],
            "negative_points_xy": [],
        }
    ]
    with pytest.raises(ValueError, match="prompt frames must be unique"):
        validate_seed_spec(spec)


def test_candidate_selection_rejects_hand_negative_and_large_expansion() -> None:
    masks = np.zeros((3, 10, 10), dtype=bool)
    masks[0, 2:5, 2:5] = True
    masks[1, 2:8, 2:8] = True
    masks[2, :, :] = True
    positive = np.asarray([[3, 3]], dtype=np.float32)
    negative = np.asarray([[7, 7]], dtype=np.float32)

    selected, rows = select_anchor_candidate(
        np,
        masks,
        np.asarray([0.7, 0.9, 0.99]),
        positive_points=positive,
        negative_points=negative,
        box=np.asarray([1, 1, 6, 6]),
        minimum_area=2,
        maximum_area=40,
    )

    assert selected == 0
    assert rows[0]["plausible"] is True
    assert rows[1]["plausible"] is False
    assert rows[2]["plausible"] is False


def test_directional_merge_preserves_forward_and_fills_empty_holes() -> None:
    forward = np.zeros((4, 4), dtype=bool)
    forward[1, 1] = True
    reverse = np.zeros((4, 4), dtype=bool)
    reverse[2, 2] = True

    assert np.array_equal(merge_directional_track_mask(np, forward, reverse), forward)
    assert np.array_equal(
        merge_directional_track_mask(np, np.zeros((4, 4), dtype=bool), reverse),
        reverse,
    )
