from __future__ import annotations

import numpy as np

from scripts.evaluate_droid_view_lora import evaluate_arrays


def test_identical_video_has_perfect_similarity() -> None:
    import cv2

    frames = np.zeros((3, 32, 48, 3), dtype=np.float32)
    frames[1, 8:20, 10:24] = 0.5
    frames[2, 10:22, 12:26] = 0.8
    metrics = evaluate_arrays(cv2, np, frames, frames.copy())
    assert metrics["mean_full_frame_ssim"] > 0.999
    assert metrics["mean_subject_roi_ssim"] > 0.999
    assert metrics["mean_subject_edge_f1"] > 0.99
    assert 0.99 < metrics["motion_magnitude_ratio"] < 1.01


def test_static_prediction_does_not_beat_static_anchor() -> None:
    import cv2

    target = np.zeros((3, 32, 48, 3), dtype=np.float32)
    target[1, 8:20, 10:24] = 0.5
    target[2, 10:22, 12:26] = 0.8
    generated = np.repeat(target[:1], len(target), axis=0)
    metrics = evaluate_arrays(cv2, np, generated, target)
    assert abs(metrics["static_anchor_ssim_gain"]) < 1e-8
    assert metrics["motion_magnitude_ratio"] == 0.0
