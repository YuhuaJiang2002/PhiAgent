from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_cosmos3_droid_i2v import (
    CANONICAL_TILE_HEIGHT,
    CANONICAL_TILE_WIDTH,
    _record,
    canonicalize_tile,
    inactive_black_metrics,
    strict_gate,
)


def test_record_rejects_training_or_final_holdout() -> None:
    contract = {
        "records": [
            {
                "sample_id": "ep012-clip00",
                "episode_index": 12,
                "split": "validation",
                "training_use": False,
            }
        ]
    }
    assert _record(contract, "ep012-clip00", "validation")["episode_index"] == 12
    with pytest.raises(ValueError, match="non-training final_holdout"):
        _record(contract, "ep012-clip00", "final_holdout")


def test_model_native_tiles_are_independently_mapped_to_canonical_pixels() -> None:
    cv2 = pytest.importorskip("cv2")
    frames = np.zeros((2, 480, 832, 3), dtype=np.float32)
    frames[:, :240, :416] = 0.1
    frames[:, :240, 416:] = 0.3
    frames[:, 240:, :416] = 0.6
    frames[:, 240:, 416:] = 0.9
    expected = {
        "top_left": 0.1,
        "top_right": 0.3,
        "bottom_left": 0.6,
        "bottom_right": 0.9,
    }
    for tile, value in expected.items():
        mapped = canonicalize_tile(cv2, frames, tile)
        assert mapped.shape == (2, CANONICAL_TILE_HEIGHT, CANONICAL_TILE_WIDTH, 3)
        assert float(mapped.mean()) == pytest.approx(value, abs=1e-5)


def test_tile_mapping_rejects_ambiguous_odd_composite_dimensions() -> None:
    cv2 = pytest.importorskip("cv2")
    with pytest.raises(ValueError, match="positive and even"):
        canonicalize_tile(cv2, np.zeros((2, 479, 832, 3)), "top_left")


def test_inactive_black_metrics_exclude_real_condition_frame() -> None:
    frames = np.zeros((3, 8, 8, 3), dtype=np.float32)
    frames[0] = 1.0
    frames[2] = 0.1
    metrics = inactive_black_metrics(np, frames)
    assert metrics["mean_luminance"] == pytest.approx(0.05)
    assert metrics["p99_luminance"] == pytest.approx(0.1)


def test_strict_gate_requires_mean_and_worst_subject_consistency() -> None:
    perfect = {
        "mean_full_frame_ssim": 1.0,
        "minimum_full_frame_ssim": 1.0,
        "mean_subject_roi_ssim": 1.0,
        "minimum_subject_roi_ssim": 1.0,
        "mean_subject_edge_f1": 1.0,
        "motion_correlation": 1.0,
        "motion_magnitude_ratio": 1.0,
        "static_anchor_ssim_gain": 0.5,
    }
    assert all(strict_gate(perfect).values())
    weak_worst_frame = {**perfect, "minimum_subject_roi_ssim": 0.24}
    gates = strict_gate(weak_worst_frame)
    assert gates["minimum_subject_roi_ssim"] is False
