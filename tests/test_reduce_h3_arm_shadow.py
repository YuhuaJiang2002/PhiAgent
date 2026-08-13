from __future__ import annotations

import cv2
import numpy as np

from scripts.reduce_h3_arm_shadow import (
    _align_alpha_to_motion,
    _apply_plate_alpha_gain,
    _build_wide_person_coverage_alpha,
    _build_wide_person_graphite_material,
    _mask_centroids,
    _smooth_centroids,
    _temporal_union_masks,
)


def test_mask_centroids_keeps_empty_frames_explicit() -> None:
    masks = np.zeros((2, 5, 7), dtype=bool)
    masks[1, 2, 3] = True

    assert _mask_centroids(np, masks) == [None, (3.0, 2.0)]


def test_motion_alignment_translates_neighbor_alpha_to_current_arm() -> None:
    alpha = np.zeros((7, 9), dtype=np.float32)
    alpha[3, 1] = 0.20

    aligned, applied, shift = _align_alpha_to_motion(
        cv2,
        np,
        alpha,
        source_centroid=(1.0, 3.0),
        target_centroid=(4.0, 3.0),
        threshold_pixels=2.0,
        maximum_shift_pixels=8.0,
        blend_ramp_pixels=0.5,
        protect_threshold=0.70,
    )

    assert applied is True
    assert shift == 3.0
    assert int(np.argmax(aligned)) == 3 * 9 + 4


def test_motion_alignment_caps_large_displacements() -> None:
    alpha = np.zeros((7, 9), dtype=np.float32)
    alpha[3, 1] = 0.20

    aligned, applied, shift = _align_alpha_to_motion(
        cv2,
        np,
        alpha,
        source_centroid=(1.0, 3.0),
        target_centroid=(7.0, 3.0),
        threshold_pixels=2.0,
        maximum_shift_pixels=2.0,
        blend_ramp_pixels=0.5,
        protect_threshold=0.70,
    )

    assert applied is True
    assert shift == 2.0
    assert int(np.argmax(aligned)) == 3 * 9 + 3


def test_motion_alignment_is_inert_below_threshold() -> None:
    alpha = np.eye(5, dtype=np.float32)

    aligned, applied, shift = _align_alpha_to_motion(
        cv2,
        np,
        alpha,
        source_centroid=(2.0, 2.0),
        target_centroid=(3.0, 2.0),
        threshold_pixels=2.0,
        maximum_shift_pixels=8.0,
        blend_ramp_pixels=6.0,
        protect_threshold=0.70,
    )

    assert applied is False
    assert shift == 1.0
    assert np.array_equal(aligned, alpha)


def test_centroid_smoothing_rejects_one_frame_segmentation_spike() -> None:
    centroids = [(1.0, 2.0), (2.0, 2.0), (20.0, 20.0), (4.0, 2.0), (5.0, 2.0)]

    smoothed = _smooth_centroids(np, centroids, radius=2)

    assert smoothed[2] == (4.0, 2.0)


def test_motion_alignment_blends_continuously_and_preserves_hard_alpha() -> None:
    alpha = np.zeros((5, 9), dtype=np.float32)
    alpha[2, 1] = 0.20
    alpha[3, 1] = 0.80

    aligned, applied, shift = _align_alpha_to_motion(
        cv2,
        np,
        alpha,
        source_centroid=(1.0, 2.0),
        target_centroid=(4.0, 2.0),
        threshold_pixels=2.0,
        maximum_shift_pixels=8.0,
        blend_ramp_pixels=2.0,
        protect_threshold=0.70,
    )

    assert applied is True
    assert shift == 3.0
    assert np.isclose(aligned[2, 1], 0.10)
    assert np.isclose(aligned[2, 4], 0.10)
    assert np.isclose(aligned[3, 1], 0.80)
    assert np.isclose(aligned[3, 4], 0.0)


def test_plate_gain_lightens_only_low_alpha_shadow_band() -> None:
    alpha = np.asarray([0.0, 0.10, 0.30, 0.69, 0.70, 1.0], dtype=np.float32)

    result = _apply_plate_alpha_gain(
        np,
        alpha,
        gain=1.2,
        cap=0.32,
        protect_threshold=0.70,
    )

    assert np.allclose(result, [0.0, 0.12, 0.32, 0.32, 0.70, 1.0])


def test_temporal_union_masks_closes_neighbor_frame_gaps() -> None:
    masks = np.zeros((3, 5, 7), dtype=bool)
    masks[1, 2, 3] = True

    result = _temporal_union_masks(np, masks, radius=1)

    assert result[:, 2, 3].tolist() == [True, True, True]


def test_wide_person_coverage_expands_but_preserves_protected_pixels() -> None:
    person = np.zeros((9, 11), dtype=bool)
    person[4, 4] = True
    protected = np.zeros_like(person)
    protected[4, 5] = True
    safety = np.zeros_like(person)
    safety[1:8, 1:9] = True

    alpha = _build_wide_person_coverage_alpha(
        cv2,
        np,
        person_mask=person,
        protected_mask=protected,
        edit_safety=safety,
        dilation=2,
        feather_sigma=1.0,
        strength=0.8,
    )

    assert np.isclose(alpha[4, 4], 0.8)
    assert alpha[4, 3] > 0.0
    assert alpha[4, 5] == 0.0
    assert np.all(alpha[np.logical_not(safety)] == 0.0)


def test_wide_person_coverage_respects_lower_work_region() -> None:
    person = np.ones((11, 13), dtype=bool)
    protected = np.zeros_like(person)
    safety = np.ones_like(person)
    region = np.zeros_like(person)
    region[5:10, 3:11] = True

    alpha = _build_wide_person_coverage_alpha(
        cv2,
        np,
        person_mask=person,
        protected_mask=protected,
        edit_safety=safety,
        coverage_region=region,
        dilation=2,
        feather_sigma=0.0,
        strength=0.9,
    )

    assert np.all(alpha[5:10, 3:11] == 0.9)
    assert np.all(alpha[np.logical_not(region)] == 0.0)


def test_wide_person_graphite_material_neutralizes_coloured_clothing() -> None:
    frame = np.asarray([[[180, 120, 220], [30, 220, 60]]], dtype=np.uint8)

    material = _build_wide_person_graphite_material(cv2, np, frame)

    assert material.shape == frame.shape
    assert np.max(material, axis=2).item(0) - np.min(material, axis=2).item(0) < 20
    assert np.max(material, axis=2).item(1) - np.min(material, axis=2).item(1) < 20
    source_luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output_luminance = cv2.cvtColor(material, cv2.COLOR_BGR2GRAY)
    assert np.all(output_luminance >= source_luminance * 0.75)
