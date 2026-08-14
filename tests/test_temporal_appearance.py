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
from phiagent.rendering.temporal_occlusion import (
    evidence_ordered_flower_front,
    projected_contact_corridor,
    projected_contact_evidence_lock,
    propagate_robot_material_residual,
    reinforce_projected_contact_evidence,
    right_arm_flower_partition,
    source_motion_residual_median_update,
    source_owned_flower_restore_mask,
)


def test_evidence_ordered_flower_front_does_not_pull_track_through_robot() -> None:
    np = pytest.importorskip("numpy")
    source = np.zeros((3, 4, 3), dtype=np.uint8)
    candidate = source.copy()
    candidate[1, 0] = 80
    candidate[1, 1] = 80
    candidate[1, 2] = 80
    tracked = np.zeros((3, 4), dtype=bool)
    tracked[1, 0:3] = True
    resolved = np.zeros_like(tracked)
    resolved[1, 2] = True
    contested = np.zeros_like(tracked)
    contested[1, 1:3] = True

    front, source_retained = evidence_ordered_flower_front(
        np,
        candidate=candidate,
        source=source,
        resolved_flower=resolved,
        tracked_flower=tracked,
        contested_support=contested,
        replacement_threshold=12.0,
    )

    assert front[1, 0]
    assert not source_retained[1, 0]
    assert not front[1, 1]
    assert front[1, 2]
    assert not source_retained[1, 1]


def test_right_arm_flower_partition_has_explicit_disjoint_owners() -> None:
    arm = np.zeros((25, 25), dtype=bool)
    arm[6:20, 5:21] = True
    flower = np.zeros_like(arm)
    flower[9:14, 15:23] = True
    hand = np.zeros_like(arm)
    hand[14:20, 5:10] = True

    editable, flower_owner, protected_hand = right_arm_flower_partition(
        cv2,
        np,
        right_arm=arm,
        flower_visible=flower,
        hand_support=hand,
        corridor_dilation_pixels=3,
        hand_dilation_pixels=3,
    )

    assert np.any(flower_owner)
    assert not np.any(editable & flower_owner)
    assert not np.any(editable & protected_hand)
    assert np.all(flower_owner <= flower)


def test_flower_restore_owns_sampling_footprint_before_person_exclusion() -> None:
    owner = np.zeros((17, 17), dtype=bool)
    owner[8, 8] = True
    person = np.ones_like(owner)
    protected_hand = np.ones_like(owner)

    restored = source_owned_flower_restore_mask(
        cv2,
        np,
        flower_owner=owner,
        person=person,
        hand_core=np.zeros_like(owner),
        protected_hand=protected_hand,
        clean_plate_padding_pixels=9,
        sample_footprint_pixels=7,
    )

    assert np.all(restored[5:12, 5:12])
    assert not restored[4, 4]


def test_flower_restore_wider_padding_respects_protected_regions() -> None:
    owner = np.zeros((21, 21), dtype=bool)
    owner[10, 10] = True
    person = np.zeros_like(owner)
    person[:, 14:] = True
    protected_hand = np.zeros_like(owner)
    protected_hand[:7, :] = True

    restored = source_owned_flower_restore_mask(
        cv2,
        np,
        flower_owner=owner,
        person=person,
        hand_core=np.zeros_like(owner),
        protected_hand=protected_hand,
        clean_plate_padding_pixels=17,
        sample_footprint_pixels=3,
    )

    assert restored[10, 5]
    assert not restored[10, 15]
    assert not restored[5, 10]


def test_flower_sampling_footprint_preserves_hand_core_except_true_owner() -> None:
    owner = np.zeros((15, 15), dtype=bool)
    owner[7, 7] = True
    hand_core = np.zeros_like(owner)
    hand_core[7, 6:9] = True

    restored = source_owned_flower_restore_mask(
        cv2,
        np,
        flower_owner=owner,
        person=np.zeros_like(owner),
        hand_core=hand_core,
        protected_hand=hand_core,
        clean_plate_padding_pixels=7,
        sample_footprint_pixels=7,
    )

    assert restored[7, 7]
    assert not restored[7, 6]
    assert not restored[7, 8]


def test_projected_contact_lock_preserves_only_generated_corridor_evidence() -> None:
    source = np.full((17, 17, 3), 100, dtype=np.uint8)
    candidate = source.copy()
    candidate[8, 6] = 140
    candidate[8, 8] = 140
    candidate[2, 2] = 140
    hand = np.zeros((17, 17), dtype=bool)
    hand[8, 6] = True
    flower = np.zeros_like(hand)
    flower[8, 10] = True

    locked = projected_contact_evidence_lock(
        cv2,
        np,
        candidate=candidate,
        source=source,
        hand_core=hand,
        tracked_object=flower,
        replacement_threshold=12.0,
        contact_radius=3,
        maximum_source_occlusion_gap=6,
    )

    assert locked[8, 6]
    assert not locked[8, 8]
    assert not locked[2, 2]
    assert not locked[8, 10]


def test_projected_contact_corridor_matches_occluded_hand_object_gap() -> None:
    hand = np.zeros((17, 17), dtype=bool)
    flower = np.zeros_like(hand)
    hand[8, 4:7] = True
    flower[8, 12] = True

    corridor = projected_contact_corridor(
        np,
        hand_core=hand,
        tracked_object=flower,
        contact_radius=3,
        maximum_source_occlusion_gap=8,
    )

    assert np.all(corridor[8, 4:7])
    assert np.count_nonzero(corridor) == 3


def test_robot_material_residual_completes_contact_without_fixed_colour() -> None:
    source = np.full((5, 7, 3), 100, dtype=np.uint8)
    candidate = source.copy()
    candidate[2, 1] = (124, 92, 108)
    projected = source.copy()
    projected[2, 1] = candidate[2, 1]
    corridor = np.zeros((5, 7), dtype=bool)
    corridor[2, 1:5] = True
    seeds = np.zeros_like(corridor)
    seeds[2, 1] = True

    completed, metrics = propagate_robot_material_residual(
        np,
        projected=projected,
        candidate=candidate,
        source=source,
        corridor=corridor,
        seed_mask=seeds,
        replacement_threshold=12.0,
    )

    assert np.all(completed[2, 1:5] == np.asarray([124, 92, 108]))
    assert np.all(completed[0, 0] == source[0, 0])
    assert metrics["propagated_pixels"] == 3.0
    assert metrics["unresolved_source_like_pixels"] == 0.0


def test_contact_evidence_reinforcement_uses_incumbent_residual_direction() -> None:
    source = np.full((2, 2, 3), 100, dtype=np.uint8)
    candidate = source.copy()
    candidate[0, 0] = (118, 114, 110)
    projected = source.copy()
    evidence = np.zeros((2, 2), dtype=bool)
    evidence[0, 0] = True

    reinforced, metrics = reinforce_projected_contact_evidence(
        np,
        projected=projected,
        candidate=candidate,
        source=source,
        evidence_mask=evidence,
        replacement_threshold=12.0,
        codec_error_margin=8.0,
    )

    assert tuple(reinforced[0, 0]) == (126, 120, 114)
    assert np.all(reinforced[1, 1] == projected[1, 1])
    assert metrics["eligible_pixels"] == 1.0
    assert metrics["reinforced_pixels"] == 1.0


def test_source_motion_residual_median_removes_part_local_extremum() -> None:
    candidate = np.full((3, 4, 3), 140, dtype=np.uint8)
    current = np.full((3, 4, 3), 40.0, dtype=np.float32)
    previous = np.full((3, 4, 3), 10.0, dtype=np.float32)
    following = np.full((3, 4, 3), 12.0, dtype=np.float32)
    reliable = np.zeros((3, 4), dtype=bool)
    reliable[:, 1:3] = True

    repaired, metrics = source_motion_residual_median_update(
        np,
        current_candidate=candidate,
        current_residual=current,
        warped_previous_residual=previous,
        warped_next_residual=following,
        reliable=reliable,
        maximum_residual_delta=24.0,
    )

    assert np.all(repaired[reliable] == 116)
    assert np.array_equal(repaired[~reliable], candidate[~reliable])
    assert metrics["baseline_temporal_extremum_mae"] == pytest.approx(28.0)
    assert metrics["repaired_temporal_extremum_mae"] == pytest.approx(4.0)


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
