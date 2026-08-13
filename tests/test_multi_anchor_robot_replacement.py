from __future__ import annotations

import pytest

import scripts.build_multi_anchor_robot_replacement as replacement
from scripts.build_multi_anchor_robot_replacement import (
    Anchor,
    _bracket,
    _source_person_mask,
    _warp_anchor,
    _warp_anchor_layers,
)


def _anchor(frame: int) -> Anchor:
    return Anchor(frame, None, None, None, None)


def test_bracket_uses_smooth_temporal_weight() -> None:
    left, right, weight = _bracket((_anchor(0), _anchor(10)), 5)

    assert (left.frame, right.frame) == (0, 10)
    assert weight == pytest.approx(0.5)


def test_warp_anchor_keeps_full_strength_mask_nonempty() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    height, width = 48, 64
    source = np.zeros((height, width, 3), dtype=np.uint8)
    source[10:38, 22:48] = 180
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    robot = np.full_like(source, 90)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[9:39, 21:49] = 255
    anchor = Anchor(0, source, robot, mask, gray)

    _, warped_mask, _ = _warp_anchor(
        cv2,
        np,
        anchor,
        gray,
        width,
        height,
        flow_clip_pixels=16.0,
        flow_strength=0.3,
    )

    assert int(warped_mask.max()) == 255
    assert int(np.count_nonzero(warped_mask)) >= int(np.count_nonzero(mask) * 0.95)


def test_robot_and_robot_mask_use_the_same_attenuated_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    height, width = 40, 60
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    def fixed_flow(*_args: object, **_kwargs: object) -> tuple[object, object, float]:
        return grid_x + 10.0, grid_y, 10.0

    monkeypatch.setattr(replacement, "_flow_map", fixed_flow)
    source = np.zeros((height, width, 3), dtype=np.uint8)
    robot = np.zeros_like(source)
    robot[10:30, 20:40] = 255
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:30, 20:40] = 255
    anchor = Anchor(0, source, robot, mask, np.zeros((20, 30), dtype=np.uint8), mask)

    warped_robot, warped_mask, warped_person, _ = _warp_anchor_layers(
        cv2,
        np,
        anchor,
        np.zeros((20, 30), dtype=np.uint8),
        width,
        height,
        flow_clip_pixels=16.0,
        flow_strength=0.5,
    )

    robot_columns = np.where(warped_robot[..., 0] > 127)[1]
    mask_columns = np.where(warped_mask > 127)[1]
    person_columns = np.where(warped_person > 127)[1]
    assert float(robot_columns.mean()) == pytest.approx(float(mask_columns.mean()), abs=0.1)
    assert float(person_columns.mean()) < float(mask_columns.mean()) - 4.0


def test_source_person_mask_uses_current_frame_segmentation() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    class Result:
        segmentation_mask = np.pad(
            np.ones((24, 24), dtype=np.float32),
            ((0, 0), (20, 20)),
        )

    class Segmenter:
        def process(self, rgb: object) -> Result:
            assert getattr(rgb, "shape") == (24, 64, 3)
            return Result()

    mask = _source_person_mask(
        cv2,
        np,
        Segmenter(),
        np.zeros((24, 64, 3), dtype=np.uint8),
    )

    assert int(mask[:, :20].max()) == 0
    assert int(mask[:, 29:43].min()) == 255
