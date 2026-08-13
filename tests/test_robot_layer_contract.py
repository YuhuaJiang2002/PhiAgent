from __future__ import annotations

import pytest

from phiagent.rendering.robot_layer_contract import (
    RobotLayerContract,
    canonical_palette_histogram,
    frame_contract_metrics,
    make_state_control,
    mask_chebyshev_distance,
    merge_missing_replacement,
    occlusion_aware_grasp_metrics,
    project_missing_contact,
    robust_limit,
)


def _fixture(np):
    source = np.full((16, 16, 3), 20, dtype=np.uint8)
    candidate = source.copy()
    alpha = np.zeros((16, 16), dtype=bool)
    alpha[2:14, 4:12] = True
    arms = np.zeros_like(alpha)
    arms[6:10, 2:14] = True
    alpha |= arms
    hands = np.zeros_like(alpha)
    hands[7:10, 12:15] = True
    alpha |= hands
    flower = np.zeros_like(alpha)
    flower[6:11, 15] = True
    candidate[alpha] = (150, 152, 155)
    palette = canonical_palette_histogram(np, candidate, alpha & ~flower)
    return source, candidate, alpha, arms, hands, flower, palette


def test_contract_requires_named_coordinate_frames() -> None:
    with pytest.raises(ValueError, match="camera frame"):
        RobotLayerContract("", "absolute_frame:660", 16, 16).validate()


def test_state_control_has_stable_alpha_object_contact_channels() -> None:
    np = pytest.importorskip("numpy")
    _, _, alpha, _, hands, flower, _ = _fixture(np)

    control = make_state_control(
        np,
        robot_alpha=alpha,
        hand_mask=hands,
        object_mask=flower,
        contact_radius=1,
    )

    assert control.shape == (16, 16, 3)
    assert int(control[..., 0].max()) == 255
    assert int(control[..., 1].max()) == 255
    assert int(control[..., 2].max()) == 255


def test_adversarial_color_shift_increases_identity_and_chroma_scores() -> None:
    np = pytest.importorskip("numpy")
    source, candidate, alpha, arms, hands, flower, palette = _fixture(np)
    baseline = frame_contract_metrics(
        np,
        candidate_rgb=candidate,
        source_rgb=source,
        robot_alpha=alpha,
        arm_support=arms,
        hand_support=hands,
        object_mask=flower,
        palette=palette,
        contact_radius=1,
    )
    attacked = candidate.copy()
    attacked[alpha] = (225, 20, 180)
    shifted = frame_contract_metrics(
        np,
        candidate_rgb=attacked,
        source_rgb=source,
        robot_alpha=alpha,
        arm_support=arms,
        hand_support=hands,
        object_mask=flower,
        palette=palette,
        contact_radius=1,
    )

    assert shifted["palette_surprisal"] > baseline["palette_surprisal"]
    assert shifted["high_chroma_fraction"] > baseline["high_chroma_fraction"]


def test_adversarial_arm_erasure_reduces_topology_coverage() -> None:
    np = pytest.importorskip("numpy")
    source, candidate, alpha, arms, hands, flower, palette = _fixture(np)
    baseline = frame_contract_metrics(
        np,
        candidate_rgb=candidate,
        source_rgb=source,
        robot_alpha=alpha,
        arm_support=arms,
        hand_support=hands,
        object_mask=flower,
        palette=palette,
        contact_radius=1,
    )
    attacked = candidate.copy()
    attacked[arms] = source[arms]
    erased = frame_contract_metrics(
        np,
        candidate_rgb=attacked,
        source_rgb=source,
        robot_alpha=alpha,
        arm_support=arms,
        hand_support=hands,
        object_mask=flower,
        palette=palette,
        contact_radius=1,
    )

    assert erased["arm_replacement_coverage"] < baseline["arm_replacement_coverage"]
    assert erased["grid_topology_coverage"] < baseline["grid_topology_coverage"]


def test_adversarial_contact_gap_is_rejected() -> None:
    np = pytest.importorskip("numpy")
    source, candidate, alpha, arms, hands, flower, palette = _fixture(np)
    baseline = frame_contract_metrics(
        np,
        candidate_rgb=candidate,
        source_rgb=source,
        robot_alpha=alpha,
        arm_support=arms,
        hand_support=hands,
        object_mask=flower,
        palette=palette,
        contact_radius=1,
    )
    attacked = candidate.copy()
    attacked[hands] = source[hands]
    detached = frame_contract_metrics(
        np,
        candidate_rgb=attacked,
        source_rgb=source,
        robot_alpha=alpha,
        arm_support=arms,
        hand_support=hands,
        object_mask=flower,
        palette=palette,
        contact_radius=1,
    )

    assert baseline["contact_required"] is True
    assert baseline["contact_pass"] is True
    assert detached["contact_pass"] is False


def test_robust_limits_are_one_sided() -> None:
    np = pytest.importorskip("numpy")
    values = np.asarray([1.0, 1.0, 1.1, 0.9, 1.0])
    assert robust_limit(np, values, direction="upper") > 1.0
    assert robust_limit(np, values, direction="lower") < 1.0


def test_missing_replacement_union_never_overwrites_protected_object() -> None:
    np = pytest.importorskip("numpy")
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    base = source.copy()
    donor = source.copy()
    hand = np.zeros((8, 8), dtype=bool)
    hand[2:6, 2:6] = True
    protected = np.zeros_like(hand)
    protected[3, 3] = True
    base[2:6, 2:4] = 120
    donor[2:6, 2:6] = 150

    merged, copied, _ = merge_missing_replacement(
        np,
        base_rgb=base,
        donor_rgb=donor,
        source_rgb=source,
        hand_support=hand,
        protected_object=protected,
        replacement_threshold=12,
        expansion_radius=1,
    )

    assert int(copied.sum()) > 0
    assert bool(copied[3, 3]) is False
    assert np.array_equal(merged[3, 3], base[3, 3])
    assert np.all(merged[4, 5] > 0)


def test_contact_projection_bridges_only_tracked_hand_and_protects_object() -> None:
    np = pytest.importorskip("numpy")
    source = np.zeros((12, 16, 3), dtype=np.uint8)
    candidate = source.copy()
    hand = np.zeros((12, 16), dtype=bool)
    hand[4:8, 2:13] = True
    flower = np.zeros_like(hand)
    flower[4:8, 13:15] = True
    candidate[4:8, 2:7] = np.asarray([160, 170, 180], dtype=np.uint8)

    projected, added, steps, passed = project_missing_contact(
        np,
        candidate_rgb=candidate,
        source_rgb=source,
        hand_support=hand,
        protected_object=flower,
        replacement_threshold=12,
        contact_radius=1,
        maximum_bridge_steps=6,
    )

    assert passed is True
    assert 1 <= steps <= 6
    assert int(added.sum()) > 0
    assert int(added.sum()) <= int(hand.sum())
    assert np.array_equal(projected[flower], candidate[flower])
    assert not np.any(added & ~hand)
    assert not np.any(added & flower)


def test_contact_projection_is_noop_without_required_source_contact() -> None:
    np = pytest.importorskip("numpy")
    source = np.zeros((8, 10, 3), dtype=np.uint8)
    candidate = source.copy()
    candidate[2:5, 1:3] = 200
    hand = np.zeros((8, 10), dtype=bool)
    hand[2:5, 1:4] = True
    flower = np.zeros_like(hand)
    flower[2:5, 8:9] = True

    projected, added, steps, passed = project_missing_contact(
        np,
        candidate_rgb=candidate,
        source_rgb=source,
        hand_support=hand,
        protected_object=flower,
        replacement_threshold=12,
        contact_radius=1,
    )

    assert np.array_equal(projected, candidate)
    assert not np.any(added)
    assert steps == 0
    assert passed is False


def test_occlusion_aware_grasp_accepts_replaced_hand_corridor() -> None:
    np = pytest.importorskip("numpy")
    source = np.zeros((12, 20, 3), dtype=np.uint8)
    candidate = source.copy()
    hands = np.zeros((12, 20), dtype=bool)
    hands[4:8, 7:12] = True
    flower = np.zeros_like(hands)
    flower[4:8, 15:17] = True
    candidate[hands] = np.asarray([150, 160, 170], dtype=np.uint8)

    metrics = occlusion_aware_grasp_metrics(
        np,
        candidate_rgb=candidate,
        source_rgb=source,
        hand_support=hands,
        object_mask=flower,
        replacement_threshold=12,
        contact_radius=1,
        maximum_source_occlusion_gap=4,
        minimum_bridge_coverage=0.8,
    )

    assert mask_chebyshev_distance(np, hands, flower, maximum_radius=4) == 4
    assert metrics["robot_direct_contact"] is False
    assert metrics["occlusion_bridge_coverage"] == 1.0
    assert metrics["visual_grasp_pass"] is True


def test_occlusion_aware_grasp_rejects_floating_object_after_hand_erasure() -> None:
    np = pytest.importorskip("numpy")
    source = np.zeros((12, 20, 3), dtype=np.uint8)
    candidate = source.copy()
    hands = np.zeros((12, 20), dtype=bool)
    hands[4:8, 7:12] = True
    flower = np.zeros_like(hands)
    flower[4:8, 15:17] = True

    metrics = occlusion_aware_grasp_metrics(
        np,
        candidate_rgb=candidate,
        source_rgb=source,
        hand_support=hands,
        object_mask=flower,
        replacement_threshold=12,
        contact_radius=1,
        maximum_source_occlusion_gap=4,
        minimum_bridge_coverage=0.8,
    )

    assert metrics["source_hold_observable"] is True
    assert metrics["occlusion_bridge_coverage"] == 0.0
    assert metrics["visual_grasp_pass"] is False


def test_mask_distance_finds_exact_bounded_radius_with_binary_search() -> None:
    np = pytest.importorskip("numpy")
    first = np.zeros((20, 30), dtype=bool)
    second = np.zeros_like(first)
    first[10, 3] = True
    second[10, 20] = True

    assert mask_chebyshev_distance(np, first, second, maximum_radius=16) is None
    assert mask_chebyshev_distance(np, first, second, maximum_radius=24) == 17


def test_frame_metric_fractions_remain_bounded_on_large_masks() -> None:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    source = rng.integers(0, 255, (256, 384, 3), dtype=np.uint8)
    candidate = source.copy()
    candidate[40:220, 70:330] = 255 - candidate[40:220, 70:330]
    alpha = np.zeros((256, 384), dtype=bool)
    alpha[30:230, 60:340] = True
    flower = np.zeros_like(alpha)
    flower[100:180, 170:230] = True
    arms = np.zeros_like(alpha)
    arms[50:210, 80:320] = True
    hands = np.zeros_like(alpha)
    hands[100:190, 140:260] = True
    snapshots = [value.copy() for value in (alpha, flower, arms, hands)]
    palette_mask = np.logical_and(alpha, np.logical_not(flower))
    palette = canonical_palette_histogram(np, candidate, palette_mask)

    metrics = frame_contract_metrics(
        np,
        candidate_rgb=candidate,
        source_rgb=source,
        robot_alpha=alpha,
        arm_support=arms,
        hand_support=hands,
        object_mask=flower,
        palette=palette,
    )

    bounded = (
        "high_chroma_fraction",
        "skin_like_fraction",
        "replacement_coverage",
        "arm_replacement_coverage",
        "hand_replacement_coverage",
        "grid_topology_coverage",
    )
    assert all(0.0 <= float(metrics[name]) <= 1.0 for name in bounded)
    assert all(
        np.array_equal(value, snapshot)
        for value, snapshot in zip((alpha, flower, arms, hands), snapshots, strict=True)
    )
