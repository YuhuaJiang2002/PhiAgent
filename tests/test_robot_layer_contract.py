from __future__ import annotations

import pytest

from phiagent.rendering.robot_layer_contract import (
    RobotLayerContract,
    canonical_palette_histogram,
    frame_contract_metrics,
    make_state_control,
    merge_missing_replacement,
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
