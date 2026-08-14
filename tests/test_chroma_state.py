from __future__ import annotations

import cv2
import numpy as np
import pytest

from phiagent.rendering.chroma_state import (
    normalized_mask_blur,
    project_masked_chroma_state,
    project_masked_multiscale_chroma_state,
    restore_masked_luma_carrier,
)


def test_normalized_blur_does_not_import_outside_colour() -> None:
    values = np.zeros((9, 9), dtype=np.float32)
    values[:, 5:] = 255.0
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:5] = True
    blurred = normalized_mask_blur(cv2, np, values, mask, kernel_size=5)
    assert np.allclose(blurred[mask], 0.0)


def test_chroma_projection_preserves_outside_and_reduces_chroma_variation() -> None:
    frame = np.zeros((20, 24, 3), dtype=np.uint8)
    frame[:] = (90, 90, 90)
    mask = np.zeros((20, 24), dtype=bool)
    mask[3:17, 4:20] = True
    for x in range(4, 20):
        frame[3:17, x] = (25, 35, 220) if x % 2 else (105, 115, 220)
    before = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    result, metrics = project_masked_chroma_state(
        cv2,
        np,
        frame,
        mask,
        kernel_size=7,
        strength=1.0,
        maximum_chroma_delta=80.0,
    )
    after = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    assert np.array_equal(result[~mask], frame[~mask])
    assert np.abs(np.diff(after[8, 5:19])).mean() < np.abs(
        np.diff(before[8, 5:19])
    ).mean()
    assert metrics["mean_abs_luma_delta"] <= 1.0


def test_multiscale_projection_uses_one_state_and_reports_each_pass() -> None:
    frame = np.zeros((20, 24, 3), dtype=np.uint8)
    frame[:] = (55, 65, 210)
    frame[:, ::2] = (115, 125, 210)
    mask = np.zeros((20, 24), dtype=bool)
    mask[2:18, 3:21] = True
    result, metrics = project_masked_multiscale_chroma_state(
        cv2,
        np,
        frame,
        mask,
        kernel_sizes=(5, 9, 13),
        strength=1.0,
        maximum_chroma_delta=80.0,
    )
    assert [row["kernel_size"] for row in metrics["passes"]] == [5, 9, 13, 0]
    assert metrics["passes"][-1]["projection"] == "canonical_achromatic_material_prior"
    assert metrics["chroma_tv_after"] < metrics["chroma_tv_before"]
    assert metrics["mean_abs_luma_delta"] <= 1.0
    assert np.array_equal(result[~mask], frame[~mask])


def test_achromatic_prior_is_bounded_and_only_changes_editable_region() -> None:
    frame = np.full((12, 14, 3), (20, 45, 220), dtype=np.uint8)
    mask = np.zeros((12, 14), dtype=bool)
    mask[2:10, 3:11] = True
    before = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1]
    result, metrics = project_masked_multiscale_chroma_state(
        cv2,
        np,
        frame,
        mask,
        kernel_sizes=(5,),
        strength=1.0,
        maximum_chroma_delta=40.0,
        saturation_scale=0.0,
    )
    after = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)[..., 1]
    assert np.array_equal(result[~mask], frame[~mask])
    assert float((before.astype(float) - after.astype(float))[mask].max()) <= 42.0
    assert metrics["passes"][-1]["maximum_abs_chroma_delta"] == 40.0


def test_luma_carrier_projection_removes_integer_roundtrip_error() -> None:
    target = np.full((8, 9, 3), (33, 117, 204), dtype=np.uint8)
    changed = np.full_like(target, (80, 140, 175))
    mask = np.zeros((8, 9), dtype=bool)
    mask[1:7, 2:8] = True

    result, metrics = restore_masked_luma_carrier(
        cv2,
        np,
        changed,
        target,
        mask,
    )

    target_y = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)[..., 0]
    result_y = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)[..., 0]
    assert np.array_equal(result_y[mask], target_y[mask])
    assert np.array_equal(result[~mask], changed[~mask])
    assert metrics["nonzero_luma_pixels"] == 0.0


@pytest.mark.parametrize("kernel", [0, 2, 4])
def test_chroma_projection_rejects_invalid_kernel(kernel: int) -> None:
    with pytest.raises(ValueError):
        normalized_mask_blur(
            cv2,
            np,
            np.zeros((4, 4), dtype=np.float32),
            np.ones((4, 4), dtype=bool),
            kernel_size=kernel,
        )
