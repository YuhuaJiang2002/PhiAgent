from __future__ import annotations

import cv2
import numpy as np
import pytest

from phiagent.rendering.temporal_appearance import (
    bidirectional_flow_state,
    bidirectional_residual_consensus_update,
    flow_reference_frames,
    residual_state_update,
    weighted_residual_consensus,
)
from phiagent.rendering.temporal_masks import (
    apply_temporal_lock_envelope,
    build_torso_head_whitelist,
)


def test_temporal_whitelist_excludes_limb_flower_and_adjacent_locks() -> None:
    robot = np.zeros((24, 24), dtype=bool)
    robot[2:22, 2:22] = True
    limbs = np.zeros_like(robot)
    limbs[8:16, 8:16] = True
    flower = np.zeros_like(robot)
    flower[4:7, 17:20] = True

    editable, limb_lock, contact_lock = build_torso_head_whitelist(
        cv2,
        np,
        robot=robot,
        limbs=limbs,
        flower=flower,
        limb_dilation_pixels=3,
        torso_erosion_pixels=3,
        contact_dilation_pixels=5,
    )
    adjacent = np.zeros_like(robot)
    adjacent[18:20, 5:8] = True
    enveloped = apply_temporal_lock_envelope(
        np,
        editable=editable,
        adjacent_locked_masks=[adjacent],
    )

    assert not np.any(editable & limb_lock)
    assert not np.any(editable & contact_lock)
    assert not np.any(editable & flower)
    assert not np.any(enveloped & adjacent)


def test_bidirectional_consensus_suppresses_temporal_extremum_only() -> None:
    candidate = np.full((4, 5, 3), 130, dtype=np.uint8)
    current = np.full((4, 5, 3), 30.0, dtype=np.float32)
    previous = np.full((4, 5, 3), 10.0, dtype=np.float32)
    following = np.full((4, 5, 3), 12.0, dtype=np.float32)
    reliable = np.zeros((4, 5), dtype=bool)
    reliable[1:3, 1:4] = True

    repaired, metrics = bidirectional_residual_consensus_update(
        np,
        current_candidate=candidate,
        current_residual=current,
        warped_previous_residual=previous,
        warped_next_residual=following,
        previous_confidence=np.ones((4, 5), dtype=np.float32),
        next_confidence=np.ones((4, 5), dtype=np.float32),
        reliable=reliable,
        strength=0.5,
        maximum_residual_delta=20.0,
    )

    assert np.all(repaired[reliable] == 121)
    assert np.array_equal(repaired[~reliable], candidate[~reliable])
    assert metrics["mean_abs_applied_correction"] == pytest.approx(9.0)


def test_bidirectional_consensus_keeps_current_when_it_is_temporal_median() -> None:
    candidate = np.full((2, 2, 3), 100, dtype=np.uint8)
    current = np.full((2, 2, 3), 20.0, dtype=np.float32)
    previous = np.full((2, 2, 3), 10.0, dtype=np.float32)
    following = np.full((2, 2, 3), 30.0, dtype=np.float32)

    repaired, _ = bidirectional_residual_consensus_update(
        np,
        current_candidate=candidate,
        current_residual=current,
        warped_previous_residual=previous,
        warped_next_residual=following,
        previous_confidence=np.ones((2, 2), dtype=np.float32),
        next_confidence=np.ones((2, 2), dtype=np.float32),
        reliable=np.ones((2, 2), dtype=bool),
        strength=1.0,
        maximum_residual_delta=20.0,
    )

    assert np.array_equal(repaired, candidate)


def test_candidate_flow_reference_uses_generated_robot_observations() -> None:
    previous_candidate = object()
    candidate = object()
    previous_incumbent = object()
    incumbent = object()

    selected = flow_reference_frames(
        "candidate",
        previous_candidate=previous_candidate,
        candidate=candidate,
        previous_incumbent=previous_incumbent,
        incumbent=incumbent,
    )

    assert selected == (previous_candidate, candidate)


def test_incumbent_flow_reference_uses_source_observations() -> None:
    previous_candidate = object()
    candidate = object()
    previous_incumbent = object()
    incumbent = object()

    selected = flow_reference_frames(
        "incumbent",
        previous_candidate=previous_candidate,
        candidate=candidate,
        previous_incumbent=previous_incumbent,
        incumbent=incumbent,
    )

    assert selected == (previous_incumbent, incumbent)


def test_identity_flow_is_confident_on_static_texture() -> None:
    rng = np.random.default_rng(20260813)
    frame = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)

    flow = bidirectional_flow_state(
        cv2,
        np,
        frame,
        frame,
        scale=0.5,
    )

    assert flow.confidence.shape == frame.shape[:2]
    assert float(np.mean(flow.confidence)) > 0.98
    assert float(np.percentile(flow.cycle_error, 95)) < 0.05
    assert float(np.max(flow.photometric_error)) == 0.0


def test_residual_state_reduces_low_frequency_flicker_without_touching_geometry() -> None:
    incumbent = np.full((32, 40, 3), 100, dtype=np.uint8)
    previous_state = np.full((32, 40, 3), 110, dtype=np.uint8)
    current = np.full((32, 40, 3), 130, dtype=np.uint8)
    current[:, 20:] += 10
    reliable = np.zeros((32, 40), dtype=bool)
    reliable[4:28, 4:36] = True

    repaired, metrics = residual_state_update(
        cv2,
        np,
        current_incumbent=incumbent,
        current_candidate=current,
        warped_previous_incumbent=incumbent,
        warped_previous_state=previous_state,
        confidence=np.ones((32, 40), dtype=np.float32),
        reliable=reliable,
        strength=0.5,
        gaussian_sigma=1.5,
        maximum_residual_delta=12.0,
    )

    assert np.all(repaired[~reliable] == current[~reliable])
    assert float(np.mean(repaired[reliable])) < float(np.mean(current[reliable]))
    before_contrast = int(current[16, 24, 0]) - int(current[16, 16, 0])
    after_contrast = int(repaired[16, 24, 0]) - int(repaired[16, 16, 0])
    assert abs(after_contrast - before_contrast) <= 1
    assert metrics["active_fraction"] == pytest.approx(float(np.mean(reliable)))
    assert 0 < metrics["mean_abs_applied_correction"] <= 6.0
    assert metrics["maximum_abs_applied_correction"] <= 6.0


def test_residual_state_rejects_shape_or_strength_mismatch() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.ones((8, 8), dtype=bool)
    with pytest.raises(ValueError, match="must not exceed one"):
        residual_state_update(
            cv2,
            np,
            current_incumbent=frame,
            current_candidate=frame,
            warped_previous_incumbent=frame,
            warped_previous_state=frame,
            confidence=np.ones((8, 8), dtype=np.float32),
            reliable=mask,
            strength=1.1,
            gaussian_sigma=1.0,
            maximum_residual_delta=1.0,
        )


def test_weighted_consensus_rejects_one_outlier_without_blurring() -> None:
    residuals = np.asarray(
        [
            np.full((2, 3, 3), 10.0, dtype=np.float32),
            np.full((2, 3, 3), 11.0, dtype=np.float32),
            np.full((2, 3, 3), 100.0, dtype=np.float32),
        ]
    )
    weights = np.ones((3, 2, 3), dtype=np.float32)

    consensus = weighted_residual_consensus(
        np,
        residuals=residuals,
        weights=weights,
        minimum_observations=3,
        maximum_channel_mad=2.0,
    )

    assert np.all(consensus.value == 11.0)
    assert consensus.reliable.dtype == np.bool_
    assert np.all(consensus.reliable)
    assert np.all(consensus.support_count == 3)
    assert np.all(consensus.maximum_channel_mad == 1.0)


def test_weighted_consensus_requires_support_and_low_mad() -> None:
    residuals = np.zeros((3, 2, 2, 3), dtype=np.float32)
    residuals[1] = 20.0
    residuals[2] = 40.0
    weights = np.ones((3, 2, 2), dtype=np.float32)
    weights[2, 0, 0] = 0.0

    consensus = weighted_residual_consensus(
        np,
        residuals=residuals,
        weights=weights,
        minimum_observations=3,
        maximum_channel_mad=5.0,
    )

    assert not consensus.reliable[0, 0]
    assert not np.any(consensus.reliable)
