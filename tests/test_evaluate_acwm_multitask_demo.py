from __future__ import annotations

import cv2
import numpy as np
import pytest

from scripts.evaluate_acwm_multitask_demo import (
    _align_support,
    _measure_handover_transfer,
    _merge_manifest_items,
)


def test_align_support_preserves_binary_camera_frame_regions() -> None:
    mask = np.asarray([[False, True], [False, True]])
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    aligned = _align_support(cv2, [mask], [frame])

    assert aligned[0].dtype == np.bool_
    assert aligned[0].shape == (4, 6)
    assert not aligned[0][:, :3].any()
    assert aligned[0][:, 3:].all()


def test_align_support_rejects_empty_target_video() -> None:
    with pytest.raises(ValueError, match="empty video"):
        _align_support(cv2, [np.ones((2, 2), dtype=bool)], [])


def test_merge_manifest_items_rejects_cross_manifest_label_collision() -> None:
    payloads = [
        {"variants": [{"label": "handover-bottle"}]},
        {"variants": [{"label": "handover-bottle"}]},
    ]

    with pytest.raises(ValueError, match="duplicate variants label"):
        _merge_manifest_items(payloads, "variants")


def _handover_frame(x_fraction: float) -> np.ndarray:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    center = (int(round(x_fraction * frame.shape[1])), 78)
    # HSV (105, 220, 180) converts to an unambiguous saturated blue in BGR.
    blue = cv2.cvtColor(
        np.asarray([[[105, 220, 180]]], dtype=np.uint8),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    cv2.circle(frame, center, 10, tuple(int(value) for value in blue), -1)
    return frame


def test_measure_handover_transfer_accepts_screen_left_terminal_holder() -> None:
    frames = [
        _handover_frame(float(value))
        for value in np.linspace(0.56, 0.39, 240)
    ]

    result = _measure_handover_transfer(
        cv2,
        np,
        frames,
        final_x_max=0.47,
        final_p90_x_max=0.52,
        leftward_shift_min=0.04,
        valid_fraction_min=0.95,
    )

    assert result["passed"] is True
    assert result["checks"]["net_leftward_transfer"] is True
    assert result["final_window_median_x"] < 0.47


def test_measure_handover_transfer_rejects_bottle_retained_screen_right() -> None:
    frames = [
        _handover_frame(float(value))
        for value in np.linspace(0.51, 0.61, 240)
    ]

    result = _measure_handover_transfer(
        cv2,
        np,
        frames,
        final_x_max=0.47,
        final_p90_x_max=0.52,
        leftward_shift_min=0.04,
        valid_fraction_min=0.95,
    )

    assert result["passed"] is False
    assert result["checks"]["final_median_is_screen_left"] is False
    assert result["checks"]["net_leftward_transfer"] is False
