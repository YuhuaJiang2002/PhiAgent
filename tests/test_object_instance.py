from __future__ import annotations

from dataclasses import replace

import pytest

from phiagent.evaluation.object_instance import (
    NormalizedROI,
    ObjectTrackerConfig,
    RGBFrames,
    composite_source_object,
    encode_video,
    evaluate_character_mask_coverage,
    evaluate_object_instance,
    expand_object_restoration_masks,
    remove_duplicate_colored_objects,
    route_object_preservation,
    track_colored_object,
)


WIDTH = 32
HEIGHT = 16
TEAL = (48, 96, 96)
HAND = (180, 96, 72)


def _frame(
    *,
    object_x: int,
    object_width: int,
    object_y: int = 8,
    hand: bool = False,
    color: tuple[int, int, int] = TEAL,
) -> bytes:
    pixels = bytearray([16, 16, 16] * (WIDTH * HEIGHT))
    for y in range(object_y, object_y + 4):
        for x in range(object_x, object_x + object_width):
            index = (y * WIDTH + x) * 3
            pixels[index : index + 3] = bytes(color)
    if hand:
        for y in range(7, 11):
            for x in range(object_x - 3, object_x):
                index = (y * WIDTH + x) * 3
                pixels[index : index + 3] = bytes(HAND)
    return bytes(pixels)


def _frames(
    widths: tuple[int, ...],
    *,
    object_ys: tuple[int, ...] | None = None,
    hand: bool = False,
    color: tuple[int, int, int] = TEAL,
) -> RGBFrames:
    ys = object_ys or (8,) * len(widths)
    if len(ys) != len(widths):
        raise ValueError("object_ys must match widths")
    return RGBFrames(
        tuple(
            _frame(
                object_x=8 + index,
                object_width=width,
                object_y=object_y,
                hand=hand,
                color=color,
            )
            for index, (width, object_y) in enumerate(zip(widths, ys))
        ),
        WIDTH,
        HEIGHT,
    )


def _config() -> ObjectTrackerConfig:
    return ObjectTrackerConfig(
        initial_roi=NormalizedROI(0.20, 0.40, 0.60, 0.50),
        minimum_component_pixels=4,
    )


def test_identical_object_track_has_perfect_instance_scores() -> None:
    source = _frames((10, 10, 10, 10))

    metrics = evaluate_object_instance(source, source, _config())

    assert metrics.contour_similarity == pytest.approx(1.0)
    assert metrics.color_similarity == pytest.approx(1.0)
    assert metrics.temporal_deformation == pytest.approx(1.0)
    assert metrics.object_consistency == pytest.approx(1.0)


@pytest.mark.parametrize("color", (TEAL, (220, 190, 40), (180, 45, 35)))
def test_tracker_learns_arbitrary_chromatic_object_from_roi(
    color: tuple[int, int, int],
) -> None:
    source = _frames((10, 10, 10), color=color)

    track = track_colored_object(source, _config())

    assert all(box is not None for box in track.boxes)
    assert all(area == 40 for area in track.areas)


def test_length_and_temporal_shape_drift_fail_instance_gate() -> None:
    source = _frames((10, 10, 10, 10))
    deformed = _frames((10, 20, 4, 20))

    metrics = evaluate_object_instance(source, deformed, _config())

    assert metrics.contour_similarity < 0.75
    assert metrics.temporal_deformation < 0.75
    assert metrics.object_consistency < 0.75


def test_stationary_dropped_object_fails_trajectory_and_lift_gates() -> None:
    source = _frames((10, 10, 10, 10), object_ys=(8, 6, 4, 3))
    dropped = _frames((10, 10, 10, 10), object_ys=(8, 8, 8, 8))

    metrics = evaluate_object_instance(source, dropped, _config())

    assert metrics.contour_similarity == pytest.approx(1.0)
    assert metrics.color_similarity == pytest.approx(1.0)
    assert metrics.trajectory_similarity < 0.75
    assert metrics.lift_recall == 0.0
    assert metrics.object_consistency == 0.0


def test_confidence_route_preserves_complete_lifted_candidate() -> None:
    source = _frames((10, 10, 10, 10), object_ys=(8, 6, 4, 3))
    route = route_object_preservation(source, source, _config())
    assert route.decision == "preserve_raw_candidate_all_frames"
    assert not route.repair_applied
    assert route.candidate_track_all_frames
    assert route.candidate_lift_recall == 1.0


def test_confidence_route_repairs_candidate_that_does_not_lift() -> None:
    source = _frames((10, 10, 10, 10), object_ys=(8, 6, 4, 3))
    dropped = _frames((10, 10, 10, 10), object_ys=(8, 8, 8, 8))
    route = route_object_preservation(source, dropped, _config())
    assert route.decision == "repair_candidate_object"
    assert route.repair_applied
    assert route.candidate_lift_recall == 0.0


def test_confidence_route_does_not_repair_from_unreliable_source_track() -> None:
    source = _frames((4, 20, 4), object_ys=(8, 5, 3))
    candidate = _frames((4, 4, 4), object_ys=(8, 5, 3))
    route = route_object_preservation(
        source,
        candidate,
        replace(_config(), maximum_area_ratio=10, search_margin=0.5),
    )
    assert route.decision == "preserve_raw_candidate_source_track_unreliable"
    assert not route.repair_applied
    assert route.source_area_ratio > 3
    assert route.candidate_track_all_frames


def test_compositing_never_restores_adjacent_hand_pixels() -> None:
    source = _frames((10, 10, 10), hand=True)
    candidate_frame = bytes([220, 220, 220] * (WIDTH * HEIGHT))
    candidate = RGBFrames((candidate_frame,) * 3, WIDTH, HEIGHT)
    track = track_colored_object(source, _config())

    improved = composite_source_object(source, candidate, track)

    hand_pixel = (8 * WIDTH + 5) * 3
    object_pixel = (9 * WIDTH + 11) * 3
    assert improved.frames[0][hand_pixel : hand_pixel + 3] == bytes((220, 220, 220))
    assert improved.frames[0][object_pixel : object_pixel + 3] == bytes(TEAL)


@pytest.mark.parametrize("color", (TEAL, (220, 190, 40)))
def test_duplicate_cleanup_removes_dropped_copy_before_restoration(
    color: tuple[int, int, int],
) -> None:
    source = _frames((10, 10, 10), object_ys=(8, 5, 3), color=color)
    track = track_colored_object(source, _config())
    candidate_frames = []
    for frame in source.frames:
        output = bytearray(frame)
        for y in range(8, 12):
            for x in range(8, 18):
                pixel = (y * WIDTH + x) * 3
                output[pixel : pixel + 3] = bytes(color)
        candidate_frames.append(bytes(output))
    candidate = RGBFrames(tuple(candidate_frames), WIDTH, HEIGHT)

    cleaned, removed = remove_duplicate_colored_objects(
        source, candidate, track, _config()
    )
    restored = composite_source_object(source, cleaned, track)

    duplicate_pixel = (10 * WIDTH + 10) * 3
    assert removed[-1] > 0
    assert restored.frames[-1][duplicate_pixel : duplicate_pixel + 3] == source.frames[
        -1
    ][duplicate_pixel : duplicate_pixel + 3]


def test_restoration_mask_recovers_connected_low_chroma_edges() -> None:
    frame = bytearray(_frame(object_x=8, object_width=10))
    edge_color = bytes((60, 65, 65))
    for y in range(8, 12):
        for x in (7, 18):
            pixel = (y * WIDTH + x) * 3
            frame[pixel : pixel + 3] = edge_color
    decoded = RGBFrames((bytes(frame),), WIDTH, HEIGHT)
    track = track_colored_object(decoded, _config())

    restored = expand_object_restoration_masks(decoded, track, _config())[0]

    assert restored[9 * WIDTH + 7]
    assert restored[9 * WIDTH + 18]
    assert not restored[2 * WIDTH + 2]


def test_character_mask_overlap_reports_object_coverage() -> None:
    source = _frames((10, 10, 10))
    track = track_colored_object(source, _config())
    character_masks = []
    for object_mask in track.masks:
        character_masks.append(bytes(255 if selected else 0 for selected in object_mask))

    coverage = evaluate_character_mask_coverage(track, character_masks)

    assert coverage.covered
    assert coverage.mean_fraction == pytest.approx(1.0)


def test_tracker_rejects_component_that_swallows_same_color_background() -> None:
    first = _frame(object_x=8, object_width=6)
    contaminated = bytearray(first)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            index = (y * WIDTH + x) * 3
            contaminated[index : index + 3] = bytes(TEAL)
    decoded = RGBFrames((first, bytes(contaminated)), WIDTH, HEIGHT)

    track = track_colored_object(decoded, _config())

    assert track.boxes[0] is not None
    assert track.boxes[1] is None
    assert track.areas[1] == 0


def test_video_encoder_rejects_invalid_output_format(tmp_path) -> None:
    with pytest.raises(ValueError, match="output pixel format"):
        encode_video(
            (),
            tmp_path / "output.mp4",
            tmp_path / "ffmpeg",
            width=32,
            height=16,
            fps=30,
            pixel_format="rgb24",
            output_pixel_format="rgb24",
        )


def test_cyan_initialization_rejects_non_cyan_chromatic_region() -> None:
    frames = RGBFrames((bytes((180, 30, 30) * 4),), 2, 2)
    config = ObjectTrackerConfig(
        NormalizedROI(0, 0, 1, 1),
        initial_color_mode="cyan",
    )
    with pytest.raises(ValueError, match="no chromatic object component"):
        track_colored_object(frames, config)
