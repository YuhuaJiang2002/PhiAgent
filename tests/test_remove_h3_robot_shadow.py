from __future__ import annotations

import pytest

from scripts.remove_h3_robot_shadow import (
    _fill_mask_holes,
    _load_packed_masks,
    compose_shadow_free_frame,
)


def test_shadow_compositor_keeps_robot_and_restores_clean_exterior() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = np.full((40, 48, 3), 20, dtype=np.uint8)
    generated = np.full((40, 48, 3), 70, dtype=np.uint8)
    clean = np.full((40, 48, 3), 35, dtype=np.uint8)
    robot = np.zeros((40, 48), dtype=bool)
    robot[14:27, 18:31] = True
    person = np.zeros((40, 48), dtype=bool)
    person[4:36, 5:43] = True
    flowers = np.zeros((40, 48), dtype=bool)
    flowers[30:34, 20:26] = True
    human_residual = np.zeros((40, 48), dtype=bool)
    human_residual[6:9, 8:12] = True
    robot_before = robot.copy()
    flowers_before = flowers.copy()

    result, metrics = compose_shadow_free_frame(
        cv2,
        np,
        source=source,
        generated=generated,
        clean_plate=clean,
        robot_mask=robot,
        source_person_mask=person,
        flower_mask=flowers,
        dilation_pixels=1,
        source_human_residual_mask=human_residual,
    )

    assert np.all(result[18:23, 21:27] == 70)
    assert np.all(result[6:9, 8:12] == 35)
    assert np.all(result[:3, :3] == 20)
    assert np.all(result[30:34, 20:26] == 20)
    assert metrics["robot_core_exact_fraction"] == 1.0
    assert metrics["flower_exact_fraction"] == 1.0
    assert metrics["halo_background_mae"] == 0.0
    assert metrics["baseline_halo_background_mae"] == 35.0
    assert metrics["halo_remaining_fraction"] == 0.0
    assert metrics["protected_exterior_exact_fraction"] == 1.0
    assert metrics["source_human_residual_retained_fraction"] == 0.0
    assert metrics["robot_mask_fraction"] == pytest.approx(float(np.mean(robot)))
    assert np.array_equal(robot, robot_before)
    assert np.array_equal(flowers, flowers_before)


def test_shadow_compositor_does_not_mutate_unpacked_boolean_views() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    packed = np.packbits(
        np.pad(np.ones((7, 9), dtype=np.uint8), ((5, 8), (6, 7))).reshape(-1),
        bitorder="little",
    )
    robot = np.unpackbits(packed, bitorder="little")[: 20 * 22].reshape(20, 22).view(bool)
    flowers = np.zeros((20, 22), dtype=np.uint8)
    flowers[8:13, 10:16] = 255
    robot_before = robot.copy()
    flowers_before = flowers.copy()

    _, metrics = compose_shadow_free_frame(
        cv2,
        np,
        source=np.full((20, 22, 3), 20, dtype=np.uint8),
        generated=np.full((20, 22, 3), 70, dtype=np.uint8),
        clean_plate=np.full((20, 22, 3), 35, dtype=np.uint8),
        robot_mask=robot,
        source_person_mask=np.ones((20, 22), dtype=bool),
        flower_mask=flowers,
        dilation_pixels=1,
    )

    assert metrics["robot_mask_fraction"] == pytest.approx(float(np.mean(robot_before)))
    assert np.array_equal(robot, robot_before)
    assert np.array_equal(flowers, flowers_before)


def test_shadow_compositor_feathers_only_outside_person_core() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = np.full((32, 32, 3), 20, dtype=np.uint8)
    clean = np.full((32, 32, 3), 100, dtype=np.uint8)
    person = np.zeros((32, 32), dtype=bool)
    person[10:22, 10:22] = True

    result, _ = compose_shadow_free_frame(
        cv2,
        np,
        source=source,
        generated=source,
        clean_plate=clean,
        robot_mask=np.zeros((32, 32), dtype=bool),
        source_person_mask=person,
        flower_mask=np.zeros((32, 32), dtype=bool),
        dilation_pixels=0,
        person_background_feather_sigma=2.0,
    )

    assert np.all(result[person] == 100)
    assert np.all(result[0, 0] == 20)
    assert np.all(result[9, 16] > 20)
    assert np.all(result[9, 16] < 100)


def test_load_packed_masks_round_trips(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    masks = np.zeros((2, 5, 7), dtype=np.uint8)
    masks[0, 1:4, 2:5] = 1
    masks[1, 0:2, 4:7] = 1
    path = tmp_path / "masks.npz"
    packed = np.stack(
        [np.packbits(mask.reshape(-1), bitorder="little") for mask in masks]
    )
    np.savez_compressed(path, packed=packed, height=5, width=7)

    loaded = _load_packed_masks(
        np,
        path,
        expected_frames=2,
        expected_height=5,
        expected_width=7,
    )

    assert np.array_equal(np.stack(loaded), masks.astype(bool))


def test_fill_mask_holes_keeps_exterior_shape() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    mask = np.zeros((20, 24), dtype=bool)
    mask[4:16, 5:19] = True
    mask[8:12, 10:14] = False

    filled = _fill_mask_holes(cv2, np, mask)

    assert np.all(filled[4:16, 5:19])
    assert not np.any(filled[:4])
    assert not np.any(filled[:, :5])
