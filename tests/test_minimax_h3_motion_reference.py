from __future__ import annotations

import cv2
import numpy as np
import pytest

from scripts.evaluate_minimax_h3_flower_validation import _align_motion_reference
from scripts.run_minimax_h3_action_variants import _compose_action_references


def test_motion_reference_resize_has_explicit_camera_frame_transform() -> None:
    frames = [np.zeros((6, 8, 3), dtype=np.uint8) for _ in range(3)]
    source_info = {
        "width": 8,
        "height": 6,
        "fps": 24.0,
        "decoded_frames": 3,
    }
    target_info = {
        "width": 4,
        "height": 3,
        "fps": 24.0,
        "decoded_frames": 3,
    }

    aligned, transform = _align_motion_reference(
        cv2, frames, source_info, target_info
    )

    assert aligned[0].shape == (3, 4, 3)
    assert transform["from"] == "camera:source_anchor_pixels"
    assert transform["to"] == "camera:H3_output_pixels"
    assert transform["operation"] == "independent_axis_scale"
    assert transform["scale_x"] == pytest.approx(0.5)
    assert transform["scale_y"] == pytest.approx(0.5)
    assert transform["normalized_positions_preserved"] is True


def test_motion_reference_rejects_temporal_frame_mismatch() -> None:
    frames = [np.zeros((6, 8, 3), dtype=np.uint8) for _ in range(3)]
    source_info = {"width": 8, "height": 6, "fps": 24.0, "decoded_frames": 3}
    target_info = {"width": 4, "height": 3, "fps": 24.0, "decoded_frames": 4}

    with pytest.raises(RuntimeError, match="frame-count mismatch"):
        _align_motion_reference(cv2, frames, source_info, target_info)


def test_control_video_does_not_replace_recursive_continuation_reference() -> None:
    robot = object()
    scene = object()
    continuation = object()
    control = object()

    references = _compose_action_references(
        [
            {"type": "image", "image": robot},
            {"type": "image", "image": scene},
        ],
        continuation=continuation,
        control_frames=control,
    )

    assert [item["type"] for item in references] == ["image", "image", "image", "video"]
    assert references[2]["image"] is continuation
    assert references[3]["video"] is control
