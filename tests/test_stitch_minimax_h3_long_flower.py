from __future__ import annotations

import pytest

from scripts.stitch_minimax_h3_long_flower import (
    PackedMasks,
    hard_relock_in_place,
    low_motion_flow_deflicker,
    stabilize_subject_color,
)


def test_packed_masks_round_trip() -> None:
    np = pytest.importorskip("numpy")
    masks = []
    for index in range(3):
        mask = np.zeros((9, 13), dtype=np.uint8)
        mask[index : index + 4, 2:11] = 255
        masks.append(mask)

    packed = PackedMasks(np, masks)

    assert len(packed) == 3
    for expected, actual in zip(masks, packed):
        assert np.array_equal(expected, actual)


def test_color_stabilizer_keeps_background_objects_and_frozen_interval_exact() -> None:
    np = pytest.importorskip("numpy")
    source = [np.full((12, 16, 3), 20, dtype=np.uint8) for _ in range(8)]
    frames = []
    subjects, objects = [], []
    for index in range(8):
        frame = source[index].copy()
        frame[2:10, 4:14] = 100 + (8 if index % 2 else -8)
        frames.append(frame)
        subject = np.zeros((12, 16), dtype=np.uint8)
        subject[2:10, 4:14] = 255
        subjects.append(subject)
        protected = np.zeros((12, 16), dtype=np.uint8)
        protected[5:7, 8:10] = 255
        objects.append(protected)
    before_background = [frame[:, :4].copy() for frame in frames]
    before_objects = [frame[5:7, 8:10].copy() for frame in frames]
    before_frozen = [frames[index].copy() for index in range(3, 5)]

    record = stabilize_subject_color(
        np,
        frames,
        source,
        PackedMasks(np, subjects),
        PackedMasks(np, objects),
        frozen_interval=(3, 5),
    )

    assert record["frozen_contact_interval"] == [3, 5]
    assert all(np.array_equal(frame[:, :4], before) for frame, before in zip(frames, before_background))
    assert all(np.array_equal(frame[5:7, 8:10], before) for frame, before in zip(frames, before_objects))
    assert np.array_equal(frames[3], before_frozen[0])
    assert np.array_equal(frames[4], before_frozen[1])


def test_flow_deflicker_can_freeze_the_entire_sequence() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frames = [np.full((24, 32, 3), 80 + index, dtype=np.uint8) for index in range(4)]
    source = [np.full((24, 32, 3), 30 + index, dtype=np.uint8) for index in range(4)]
    masks = [np.full((24, 32), 255, dtype=np.uint8) for _ in range(4)]
    objects = [np.zeros((24, 32), dtype=np.uint8) for _ in range(4)]
    before = [frame.copy() for frame in frames]

    record = low_motion_flow_deflicker(
        cv2,
        np,
        frames,
        source,
        PackedMasks(np, masks),
        PackedMasks(np, objects),
        frozen_interval=(0, 4),
    )

    assert record["frozen_contact_interval"] == [0, 4]
    assert all(np.array_equal(frame, expected) for frame, expected in zip(frames, before))


def test_hard_relock_restores_background_and_objects_after_lossy_decode() -> None:
    np = pytest.importorskip("numpy")
    source = [np.full((8, 10, 3), 20, dtype=np.uint8) for _ in range(2)]
    frames = [np.full((8, 10, 3), 90, dtype=np.uint8) for _ in range(2)]
    allowed = []
    objects = []
    for _ in frames:
        allowed_mask = np.zeros((8, 10), dtype=np.uint8)
        allowed_mask[2:7, 3:9] = 255
        allowed.append(allowed_mask)
        object_mask = np.zeros((8, 10), dtype=np.uint8)
        object_mask[4:6, 5:7] = 255
        objects.append(object_mask)

    record = hard_relock_in_place(
        frames,
        source,
        PackedMasks(np, allowed),
        PackedMasks(np, objects),
    )

    assert record["background_pixels"] == 100
    assert record["object_pixels"] == 8
    assert all(np.all(frame[:2] == 20) for frame in frames)
    assert all(np.all(frame[4:6, 5:7] == 20) for frame in frames)
    assert all(np.all(frame[2:4, 3:5] == 90) for frame in frames)
