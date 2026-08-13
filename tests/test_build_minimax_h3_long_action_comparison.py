from __future__ import annotations

import pytest

from scripts.build_minimax_h3_long_action_comparison import (
    build_subject_masks,
    load_action_display,
    pairwise_distinctness,
)


def test_long_action_subject_mask_tracks_both_windows() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(12)]
    previous = [frame.copy() for frame in source[:8]]
    following = [frame.copy() for frame in source[4:]]
    previous[5][8:12, 10:14] = 100
    following[1][16:20, 28:32] = 100

    masks = build_subject_masks(
        cv2,
        np,
        source,
        previous,
        following,
        following_start=4,
    )

    assert masks[5][9, 11] == 255
    assert masks[5][17, 29] == 255
    assert masks[0].shape == (32, 48)


def test_pairwise_distinctness_reports_separated_actions() -> None:
    np = pytest.importorskip("numpy")
    first = [np.zeros((8, 10, 3), dtype=np.uint8) for _ in range(5)]
    second = [np.full((8, 10, 3), 20, dtype=np.uint8) for _ in range(5)]

    result = pairwise_distinctness(np, first, second)

    assert result["full_frame_mean_absolute_difference"] == pytest.approx(20.0)
    assert result["active_pixel_mean_absolute_difference"] == pytest.approx(20.0)
    assert result["fraction_of_frames_above_2_mad"] == 1.0


def test_load_action_display_is_domain_neutral(tmp_path) -> None:
    manifest = tmp_path / "actions.json"
    manifest.write_text(
        """{
          "coordinate_frame": "camera:ego_pixels",
          "object_name": "bottle",
          "actions": [
            {"label": "pour-bottle", "instruction": "Pour into the cup."},
            {"label": "shake-bottle", "instruction": "Shake four times."},
            {"label": "handover-bottle", "instruction": "Transfer to the left hand."}
          ]
        }"""
    )

    labels, display, coordinate_frame = load_action_display(manifest)

    assert labels == ("pour-bottle", "shake-bottle", "handover-bottle")
    assert display["pour-bottle"] == ("POUR", "Pour into the cup.")
    assert coordinate_frame == "camera:ego_pixels"
