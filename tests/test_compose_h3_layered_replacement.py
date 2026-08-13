from __future__ import annotations

import pytest

from scripts.compose_h3_layered_replacement import (
    _stabilize_area_outliers,
    apply_conservative_arm_shadow_cleanup,
    build_residual_arm_skin_support,
    build_layer_masks,
    build_conservative_arm_shadow_alpha,
    build_tracked_robot_arm_material,
    build_tracked_polygon_alpha,
    fill_selected_component_hulls,
    interpolate_polygon_keyframes,
    select_arm_skin_components,
    compose_layered_frame,
)


def test_polygon_keyframes_interpolate_geometry_and_strength() -> None:
    np = pytest.importorskip("numpy")
    keyframes = [
        {"frame": 2, "points": [[2, 2], [6, 2], [4, 6]], "strength": 0.0},
        {"frame": 6, "points": [[6, 4], [10, 4], [8, 8]], "strength": 1.0},
    ]

    polygon, strength = interpolate_polygon_keyframes(np, keyframes, 4)

    assert np.allclose(polygon, [[4, 3], [8, 3], [6, 7]])
    assert strength == pytest.approx(0.5)
    assert interpolate_polygon_keyframes(np, keyframes, 1) == (None, 0.0)


def test_tracked_polygon_alpha_has_full_core_and_smooth_fade() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    tracks = [
        {
            "keyframes": [
                {
                    "frame": 0,
                    "points": [[4, 4], [12, 4], [12, 12], [4, 12]],
                    "strength": 0.0,
                },
                {
                    "frame": 4,
                    "points": [[8, 4], [16, 4], [16, 12], [8, 12]],
                    "strength": 1.0,
                },
            ]
        }
    ]

    alpha = build_tracked_polygon_alpha(
        cv2,
        np,
        shape=(20, 20),
        tracks=tracks,
        frame_index=2,
        feather_sigma=1.0,
    )

    assert alpha[8, 10] == pytest.approx(0.5)
    assert 0.0 < alpha[8, 5] < 0.5
    assert not np.any(
        build_tracked_polygon_alpha(
            cv2,
            np,
            shape=(20, 20),
            tracks=tracks,
            frame_index=8,
            feather_sigma=1.0,
        )
    )


def test_tracked_robot_arm_material_replaces_skin_chroma_inside_only() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.full((24, 32, 3), (70, 105, 145), dtype=np.uint8)
    tracks = [
        {
            "keyframes": [
                {
                    "frame": 3,
                    "points": [[6, 7], [25, 7], [25, 17], [6, 17]],
                    "strength": 1.0,
                }
            ]
        }
    ]

    material = build_tracked_robot_arm_material(
        cv2,
        np,
        frame=frame,
        tracks=tracks,
        frame_index=3,
        style="silver",
    )

    assert np.array_equal(material[:5], frame[:5])
    center = material[12, 15].astype(int)
    assert center[0] >= center[1] >= center[2]
    assert int(center.max() - center.min()) < 45
    assert not np.array_equal(center, frame[12, 15])


def test_residual_arm_skin_support_keeps_large_component_not_wide_search_box() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.full((48, 80, 3), (80, 80, 80), dtype=np.uint8)
    frame[20:31, 20:57] = (82, 132, 205)
    frame[8:12, 65:70] = (82, 132, 205)
    search = np.zeros((48, 80), dtype=np.float32)
    search[5:42, 8:74] = 1.0

    support = build_residual_arm_skin_support(
        cv2,
        np,
        frame=frame,
        search_alpha=search,
        close_width=9,
        close_height=5,
        minimum_area=100,
        dilation=2,
    )

    assert np.all(support[21:30, 21:56])
    assert not np.any(support[8:12, 65:70])
    assert not np.any(support[:4])
    assert float(np.mean(support)) < 0.20


def test_layer_builder_preserves_inputs_and_separates_semantics() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    height, width = 40, 48
    source = np.full((height, width, 3), 20, dtype=np.uint8)
    source[28:33, 18:23] = (30, 180, 30)
    source[24:29, 26:31] = (90, 140, 200)
    body_values = np.zeros((height, width), dtype=np.uint8)
    body_values[10:28, 18:30] = 1
    packed = np.packbits(body_values.reshape(-1), bitorder="little")
    body = (
        np.unpackbits(packed, bitorder="little")[: height * width]
        .reshape(height, width)
        .view(bool)
    )
    wrist = body.copy()
    wrist[25:34, 14:21] = True
    wrist[2:6, 2:6] = True
    arms = np.zeros((height, width), dtype=bool)
    arms[8:36, 10:34] = True
    hands = np.zeros((height, width), dtype=bool)
    hands[23:34, 12:32] = True
    flower_instances = np.zeros((height, width), dtype=bool)
    flower_instances[27:35, 17:25] = True
    safety = np.zeros((height, width), dtype=bool)
    safety[5:38, 8:38] = True
    snapshots = [item.copy() for item in (body, wrist, arms, hands, flower_instances, safety)]

    robot, flowers, skin_negative, generated_flowers, metrics = build_layer_masks(
        cv2,
        np,
        source=source,
        generated=source,
        body_mask=body,
        wrist_mask=wrist,
        robot_limb_mask=None,
        generated_flower_instance_mask=np.zeros_like(body),
        source_person_semantic_mask=arms,
        source_arms=arms,
        source_hands=hands,
        flower_instance_mask=flower_instances,
        safety_mask=safety,
        limb_corridor_dilation=2,
        body_neighborhood_dilation=8,
    )

    assert np.all(robot[body])
    assert np.any(robot[28:34, 14:21])
    assert not np.any(robot[2:6, 2:6])
    all_flowers = np.logical_or(flowers, generated_flowers)
    assert np.all(all_flowers[29:33, 18:23])
    assert np.any(skin_negative[24:29, 26:31])
    assert not np.any(flowers[24:29, 26:31] & skin_negative[24:29, 26:31])
    assert metrics["skin_negative_fraction"] < 0.25
    for current, before in zip((body, wrist, arms, hands, flower_instances, safety), snapshots):
        assert np.array_equal(current, before)


def test_layered_compositor_keeps_robot_then_restores_flowers() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = np.full((32, 36, 3), 20, dtype=np.uint8)
    generated = np.full((32, 36, 3), 70, dtype=np.uint8)
    clean = np.full((32, 36, 3), 35, dtype=np.uint8)
    person = np.zeros((32, 36), dtype=bool)
    person[4:30, 5:32] = True
    robot = np.zeros((32, 36), dtype=bool)
    robot[10:26, 12:27] = True
    flowers = np.zeros((32, 36), dtype=bool)
    flowers[20:27, 19:25] = True
    robot_before = robot.copy()
    flowers_before = flowers.copy()

    result, _ = compose_layered_frame(
        cv2,
        np,
        source=source,
        generated=generated,
        clean_plate=clean,
        source_person_mask=person,
        robot_mask=robot,
        flower_mask=flowers,
        robot_dilation=1,
        person_feather_sigma=0.0,
    )

    assert np.all(result[12:18, 14:20] == 70)
    assert np.all(result[20:27, 19:25] == 20)
    assert np.all(result[6:9, 7:10] == 35)
    assert np.all(result[:3, :3] == 20)
    assert np.array_equal(robot, robot_before)
    assert np.array_equal(flowers, flowers_before)


def test_layer_builder_requires_consensus_for_limbs_and_generated_flowers() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    height, width = 48, 64
    source = np.full((height, width, 3), 20, dtype=np.uint8)
    generated = source.copy()
    generated[22:27, 19:24] = (20, 180, 20)
    generated[22:27, 49:54] = (20, 180, 20)
    body = np.zeros((height, width), dtype=bool)
    body[12:32, 27:38] = True
    wrist = body.copy()
    wrist[20:31, 16:28] = True
    raw_limb = np.zeros_like(body)
    raw_limb[20:31, 15:29] = True
    raw_limb[20:31, 47:57] = True
    generated_flower_instances = np.zeros_like(body)
    generated_flower_instances[21:28, 18:25] = True
    generated_flower_instances[21:28, 48:55] = True
    empty = np.zeros_like(body)
    safety = np.ones_like(body)

    robot, _, _, generated_flowers, metrics = build_layer_masks(
        cv2,
        np,
        source=source,
        generated=generated,
        body_mask=body,
        wrist_mask=wrist,
        robot_limb_mask=raw_limb,
        generated_flower_instance_mask=generated_flower_instances,
        source_person_semantic_mask=empty,
        source_arms=empty,
        source_hands=empty,
        flower_instance_mask=empty,
        safety_mask=safety,
        limb_corridor_dilation=0,
        body_neighborhood_dilation=0,
        limb_consensus_dilation=1,
        generated_flower_support_radius=2,
        generated_flower_limb_radius=4,
    )

    assert np.any(robot[20:31, 15:29])
    assert not np.any(robot[20:31, 47:57])
    assert np.any(generated_flowers[21:28, 18:25])
    assert not np.any(generated_flowers[21:28, 48:55])
    assert metrics["limb_extra_fraction"] < metrics["raw_limb_fraction"]


def test_generated_flower_area_spike_uses_temporal_majority() -> None:
    np = pytest.importorskip("numpy")
    masks = np.zeros((5, 12, 12), dtype=bool)
    masks[:, 4:8, 4:8] = True
    masks[2, 1:11, 1:11] = True

    corrected = _stabilize_area_outliers(np, masks, ratio=1.65)

    assert corrected == [2]
    assert np.array_equal(masks[2], masks[1])


def test_layered_compositor_rejects_unknown_background_method() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)

    with pytest.raises(ValueError, match="unknown background method"):
        compose_layered_frame(
            cv2,
            np,
            source=frame,
            generated=frame,
            clean_plate=frame,
            source_person_mask=mask,
            robot_mask=mask,
            flower_mask=mask,
            robot_dilation=0,
            person_feather_sigma=0.0,
            background_method="unknown",
        )


def test_conservative_shadow_cleanup_only_changes_unprotected_neutral_band() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    generated = np.full((48, 64, 3), (90, 92, 94), dtype=np.uint8)
    clean = np.full((48, 64, 3), (165, 170, 175), dtype=np.uint8)
    generated[20:28, 24:34] = (45, 80, 210)
    safety = np.zeros((48, 64), dtype=bool)
    safety[6:43, 8:57] = True
    protected = np.zeros_like(safety)
    protected[18:31, 22:37] = True
    arm = np.zeros_like(safety)
    arm[21:28, 25:46] = True

    alpha = build_conservative_arm_shadow_alpha(
        cv2,
        np,
        generated=generated,
        clean_plate=clean,
        safety_mask=safety,
        protected_mask=protected,
        arm_mask=arm,
        protect_radius=2,
        cleanup_radius=12,
        maximum_strength=0.6,
        neutral_chroma_limit=80.0,
        difference_threshold=10.0,
        feather_sigma=1.0,
    )
    result = apply_conservative_arm_shadow_cleanup(
        np,
        generated=generated,
        clean_plate=clean,
        alpha=alpha,
        protected_mask=protected,
    )

    assert float(alpha.max()) <= 0.600001
    assert not np.any(alpha[protected])
    assert not np.any(alpha[np.logical_not(safety)])
    assert np.any(alpha[12:18, 36:48] > 0.1)
    assert np.array_equal(result[protected], generated[protected])
    assert np.array_equal(result[:5], generated[:5])
    assert np.mean(result[12:18, 36:48]) > np.mean(generated[12:18, 36:48])


def test_conservative_shadow_alpha_rejects_invalid_geometry() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=bool)

    with pytest.raises(ValueError, match="cleanup radius"):
        build_conservative_arm_shadow_alpha(
            cv2,
            np,
            generated=frame,
            clean_plate=frame,
            safety_mask=mask,
            protected_mask=mask,
            arm_mask=mask,
            protect_radius=4,
            cleanup_radius=4,
            maximum_strength=0.5,
            neutral_chroma_limit=80.0,
            difference_threshold=10.0,
            feather_sigma=1.0,
        )


def test_conservative_shadow_alpha_can_target_skin_without_touching_flowers() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    generated = np.full((40, 56, 3), (80, 84, 88), dtype=np.uint8)
    clean = np.full((40, 56, 3), (170, 170, 170), dtype=np.uint8)
    generated[16:24, 24:39] = (82, 132, 205)
    generated[16:24, 14:22] = (82, 132, 205)
    safety = np.ones((40, 56), dtype=bool)
    arm = np.zeros_like(safety)
    arm[18:22, 18:42] = True
    protected = np.zeros_like(safety)
    protected[14:26, 12:24] = True
    skin_to_core = np.zeros_like(safety)
    skin_to_core[:, 24:40] = True

    alpha = build_conservative_arm_shadow_alpha(
        cv2,
        np,
        generated=generated,
        clean_plate=clean,
        safety_mask=safety,
        protected_mask=protected,
        arm_mask=arm,
        protect_radius=1,
        cleanup_radius=14,
        maximum_strength=0.25,
        neutral_chroma_limit=30.0,
        difference_threshold=8.0,
        feather_sigma=0.0,
        skin_strength=0.75,
        skin_to_core_mask=skin_to_core,
    )

    assert not np.any(alpha[protected])
    assert float(alpha[20, 24]) > 0.3
    assert float(np.mean(alpha[17:23, 32:38])) > 0.3


def test_skin_cleanup_can_use_a_wider_domain_than_neutral_shadow() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    generated = np.full((48, 80, 3), (70, 74, 78), dtype=np.uint8)
    clean = np.full((48, 80, 3), (165, 165, 165), dtype=np.uint8)
    generated[20:28, 58:66] = (82, 132, 205)
    neutral_safety = np.zeros((48, 80), dtype=bool)
    neutral_safety[:, :48] = True
    skin_safety = np.zeros_like(neutral_safety)
    skin_safety[:, :72] = True
    arm = np.zeros_like(neutral_safety)
    arm[22:26, 28:36] = True
    protected = np.zeros_like(neutral_safety)

    alpha = build_conservative_arm_shadow_alpha(
        cv2,
        np,
        generated=generated,
        clean_plate=clean,
        safety_mask=neutral_safety,
        protected_mask=protected,
        arm_mask=arm,
        protect_radius=2,
        cleanup_radius=14,
        maximum_strength=0.25,
        neutral_chroma_limit=30.0,
        difference_threshold=8.0,
        feather_sigma=0.0,
        skin_strength=0.8,
        skin_safety_mask=skin_safety,
        skin_cleanup_radius=36,
    )

    assert float(alpha[24, 60]) > 0.3
    assert not np.any(alpha[:, 72:])


def test_skin_component_selector_keeps_only_arm_contact_component_in_roi() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    skin = np.zeros((64, 96), dtype=bool)
    skin[20:36, 26:46] = True
    skin[42:58, 24:52] = True
    skin[18:38, 70:90] = True
    arm = np.zeros_like(skin)
    arm[24:32, 38:54] = True

    selected = select_arm_skin_components(
        cv2,
        np,
        skin_mask=skin,
        arm_mask=arm,
        x_min=12,
        x_max=60,
        y_min=10,
        y_max=40,
        arm_dilation=4,
        minimum_area=100,
        maximum_area=400,
        minimum_arm_overlap=0.10,
    )

    assert np.all(selected[20:36, 26:46])
    assert not np.any(selected[42:58, 24:52])
    assert not np.any(selected[18:38, 70:90])


def test_selected_component_hulls_fill_gaps_without_joining_components() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    mask = np.zeros((36, 64), dtype=bool)
    mask[8:20, 8:11] = True
    mask[8:11, 8:24] = True
    mask[17:20, 8:24] = True
    mask[8:20, 21:24] = True
    mask[24:30, 48:54] = True

    hulls = fill_selected_component_hulls(cv2, np, mask)

    assert np.all(hulls[9:19, 9:23])
    assert np.all(hulls[24:30, 48:54])
    assert not np.any(hulls[:, 30:44])
