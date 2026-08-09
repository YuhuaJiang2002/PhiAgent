from __future__ import annotations

import pytest

from phiagent.evaluation.object_instance import ObjectColorModel, ObjectTrack, RGBFrames
from phiagent.rendering.hybrid_compositor import (
    ScreenSpaceOverlayConfig,
    composite_robot_layer,
)


def _rgb_frame(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    return bytes(color * (width * height))


def test_hybrid_compositor_preserves_source_object_exactly() -> None:
    width = height = 6
    source_frame = bytearray(_rgb_frame(width, height, (10, 20, 30)))
    object_index = 3 * width + 3
    source_frame[object_index * 3 : object_index * 3 + 3] = bytes((20, 180, 170))
    source = RGBFrames((bytes(source_frame),), width, height)

    robot_frame = bytearray(_rgb_frame(4, 4, (0, 0, 0)))
    for index in (5, 6, 9, 10):
        robot_frame[index * 3 : index * 3 + 3] = bytes((180, 180, 180))
    robot = RGBFrames((bytes(robot_frame),), 4, 4)

    mask = bytearray(width * height)
    mask[object_index] = 1
    track = ObjectTrack(
        masks=(bytes(mask),),
        boxes=((3, 3, 4, 4),),
        mean_colors=((20.0, 180.0, 170.0),),
        areas=(1,),
        model=ObjectColorModel(160.0, 150.0, 10.0, 123.3),
        width=width,
        height=height,
    )
    composited, metrics = composite_robot_layer(
        source,
        robot,
        track,
        ScreenSpaceOverlayConfig(
            target_width_fraction=0.5,
            anchor_offset_x_fraction=0,
            anchor_offset_y_fraction=0,
            black_level=5,
            edge_softness=1,
            quarter_turns_clockwise=1,
        ),
    )

    output = composited.frames[0]
    assert output[object_index * 3 : object_index * 3 + 3] == bytes((20, 180, 170))
    assert metrics.object_exact_fraction == 1.0
    assert metrics.restored_object_pixels == 1
    assert metrics.robot_pixels > 0
    assert metrics.source_unchanged_fraction < 1.0


def test_hybrid_compositor_rejects_empty_robot_layer() -> None:
    source = RGBFrames((_rgb_frame(2, 2, (1, 2, 3)),), 2, 2)
    robot = RGBFrames((_rgb_frame(2, 2, (0, 0, 0)),), 2, 2)
    mask = bytes((1, 0, 0, 0))
    track = ObjectTrack(
        masks=(mask,),
        boxes=((0, 0, 1, 1),),
        mean_colors=((1.0, 2.0, 3.0),),
        areas=(1,),
        model=ObjectColorModel(1.0, 2.0, -1.0, 2.0),
        width=2,
        height=2,
    )
    with pytest.raises(ValueError, match="no pixels above black_level"):
        composite_robot_layer(source, robot, track, ScreenSpaceOverlayConfig())


def test_overlay_config_names_screen_space_constraints() -> None:
    with pytest.raises(ValueError, match="target_width_fraction"):
        ScreenSpaceOverlayConfig(target_width_fraction=1.1)
    with pytest.raises(ValueError, match="quarter_turns_clockwise"):
        ScreenSpaceOverlayConfig(quarter_turns_clockwise=4)
