from __future__ import annotations

import pytest

from phiagent.evaluation.video_quality import (
    ImageFrame,
    VideoQualityConfig,
    VideoQualityInput,
    evaluate_video_quality,
)


def _frame(
    name: str, timestamp: float, pixels: bytes, width: int = 12, height: int = 12
) -> ImageFrame:
    return ImageFrame(name, "camera_gray", timestamp, width, height, pixels)


def _moving(count: int = 8) -> VideoQualityInput:
    frames = []
    for time in range(count):
        image = bytearray(12 * 12)
        for y in range(3, 7):
            for x in range(1 + time // 2, 5 + time // 2):
                image[y * 12 + x] = 220
        frames.append(_frame(f"frame-{time}", time * 0.1, bytes(image)))
    return VideoQualityInput(tuple(frames))


def _shift(frame: ImageFrame, dx: int, dy: int) -> bytes:
    output = bytearray(len(frame.pixels))
    for y in range(frame.height):
        for x in range(frame.width):
            source_x, source_y = x - dx, y - dy
            if 0 <= source_x < frame.width and 0 <= source_y < frame.height:
                output[y * frame.width + x] = frame.pixels[source_y * frame.width + source_x]
    return bytes(output)


def test_smooth_motion_passes_and_frozen_fails_required_activity() -> None:
    reference = _moving()
    scores = evaluate_video_quality(reference, reference, VideoQualityConfig(require_activity=True))
    assert scores.motion_requirement_score > 0.9
    frozen = VideoQualityInput(
        tuple(
            _frame(f"frozen-{index}", index * 0.1, reference.frames[0].pixels) for index in range(8)
        )
    )
    assert (
        evaluate_video_quality(
            frozen, reference, VideoQualityConfig(require_activity=True)
        ).motion_requirement_score
        == 0.0
    )
    textured = bytes((x * 23 + y * 17) % 256 for y in range(12) for x in range(12))
    rigid = VideoQualityInput(
        tuple(
            _frame(
                f"rigid-{index}", index * 0.1, _shift(_frame("source", 0.0, textured), index % 3, 0)
            )
            for index in range(8)
        )
    )
    assert (
        evaluate_video_quality(
            rigid, rigid, VideoQualityConfig(require_articulation=True)
        ).motion_requirement_score
        == 0.0
    )


def test_late_roi_flicker_is_exposed_by_late_and_worst_scores() -> None:
    reference = _moving()
    frames = list(reference.frames)
    for index in range(5, 8):
        pixels = bytearray(frames[index].pixels)
        for y in range(3, 7):
            for x in range(4, 8):
                pixels[y * 12 + x] = 255 if index % 2 else 0
        frames[index] = _frame(f"corrupt-{index}", index * 0.1, bytes(pixels))
    scores = evaluate_video_quality(
        VideoQualityInput(tuple(frames)), reference, VideoQualityConfig(roi=(0.3, 0.2, 0.5, 0.5))
    )
    assert scores.temporal_late_score < 0.5
    assert scores.temporal_worst_window_score < 0.5
    assert scores.temporal_roi_score < 0.5


def test_vector_kinematics_detect_constant_speed_direction_reversals() -> None:
    base = bytes(12 * 12)
    frames = tuple(_frame(f"zigzag-{index}", float(index), base) for index in range(5))
    zigzag = VideoQualityInput(
        frames,
        trajectory=((0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (-1.0, 0.0), (0.0, 0.0)),
    )

    diagnostics = evaluate_video_quality(
        zigzag, VideoQualityInput(frames), VideoQualityConfig(trajectory_scale=1.0)
    ).candidate_trajectory

    assert diagnostics.speeds == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert diagnostics.accelerations == pytest.approx((2.0, 0.0, 2.0))
    assert diagnostics.jerks == pytest.approx((2.0, 2.0))
    assert diagnostics.smoothness_score < 1.0


def test_timestamp_unit_declaration_preserves_equivalent_kinematics() -> None:
    base = bytes(12 * 12)
    points = ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (-1.0, 0.0), (0.0, 0.0))
    seconds = VideoQualityInput(
        tuple(_frame(f"seconds-{index}", float(index), base) for index in range(5)),
        trajectory=points,
    )
    milliseconds = VideoQualityInput(
        tuple(_frame(f"milliseconds-{index}", float(index * 1000), base) for index in range(5)),
        trajectory=points,
    )
    seconds_diagnostics = evaluate_video_quality(
        seconds, VideoQualityInput(seconds.frames), VideoQualityConfig(trajectory_scale=1.0)
    ).candidate_trajectory
    milliseconds_diagnostics = evaluate_video_quality(
        milliseconds,
        VideoQualityInput(milliseconds.frames),
        VideoQualityConfig(trajectory_scale=1.0, timestamp_unit_seconds=0.001),
    ).candidate_trajectory

    assert milliseconds_diagnostics.speeds == pytest.approx(seconds_diagnostics.speeds)
    assert milliseconds_diagnostics.accelerations == pytest.approx(
        seconds_diagnostics.accelerations
    )
    assert milliseconds_diagnostics.jerks == pytest.approx(seconds_diagnostics.jerks)
    assert milliseconds_diagnostics.smoothness_score == pytest.approx(
        seconds_diagnostics.smoothness_score
    )


def test_background_compensation_excludes_foreground_and_detects_corruption() -> None:
    base = bytes((x * 23 + y * 17) % 256 for y in range(12) for x in range(12))
    reference = VideoQualityInput(tuple(_frame(f"reference-{i}", i * 0.1, base) for i in range(4)))
    shifted = VideoQualityInput(
        tuple(_frame(f"shift-{i}", i * 0.1, _shift(reference.frames[i], 1, 0)) for i in range(4))
    )
    assert evaluate_video_quality(shifted, reference).background_preservation_score > 0.98
    foreground = bytearray(base)
    foreground[5 * 12 + 5] = 255
    mask = bytes(1 if index == 5 * 12 + 5 else 0 for index in range(144))
    excluded = VideoQualityInput(
        tuple(_frame(f"foreground-{i}", i * 0.1, bytes(foreground)) for i in range(4)), (mask,) * 4
    )
    assert evaluate_video_quality(excluded, reference).background_preservation_score > 0.99
    corrupted = bytearray(base)
    corrupted[0:30] = b"\xff" * 30
    bad = VideoQualityInput(tuple(_frame(f"bad-{i}", i * 0.1, bytes(corrupted)) for i in range(4)))
    assert evaluate_video_quality(bad, reference).background_preservation_score < 0.8
    fully_masked = VideoQualityInput(
        tuple(_frame(f"masked-{i}", i * 0.1, base) for i in range(4)),
        (bytes([1]) * 144,) * 4,
    )
    with pytest.raises(ValueError, match="no valid background pixels"):
        evaluate_video_quality(fully_masked, reference)
    tiny = VideoQualityInput(
        tuple(
            ImageFrame(f"tiny-{index}", "camera_gray", index * 0.1, 3, 3, bytes(9))
            for index in range(3)
        )
    )
    assert evaluate_video_quality(tiny, tiny).background_preservation_score == pytest.approx(1.0)


def test_blur_loses_sharpness_and_input_mismatches_raise() -> None:
    base = bytes(255 if (x + y) % 2 else 0 for y in range(12) for x in range(12))
    reference = VideoQualityInput(tuple(_frame(f"reference-{i}", i * 0.1, base) for i in range(3)))
    blur = bytes(127 for _ in base)
    candidate = VideoQualityInput(tuple(_frame(f"blur-{i}", i * 0.1, blur) for i in range(3)))
    assert evaluate_video_quality(candidate, reference).sharpness_score < 0.1
    with pytest.raises(ValueError, match="byte length"):
        ImageFrame("bad", "camera_gray", 0.0, 2, 2, b"\x00")
    with pytest.raises(ValueError, match="strictly increasing"):
        VideoQualityInput((reference.frames[0], reference.frames[0], reference.frames[2]))
    different_shape = VideoQualityInput(
        tuple(_frame(f"small-{index}", index * 0.1, bytes(8 * 8), 8, 8) for index in range(3))
    )
    with pytest.raises(ValueError, match="incompatible"):
        evaluate_video_quality(reference, different_shape)
    shifted_clock = VideoQualityInput(
        tuple(_frame(f"clock-{index}", index * 0.2, base) for index in range(3))
    )
    with pytest.raises(ValueError, match="timestamps must match"):
        evaluate_video_quality(reference, shifted_clock)
    other = VideoQualityInput(
        tuple(
            ImageFrame(f.name, "other", f.timestamp, f.width, f.height, f.pixels)
            for f in reference.frames
        )
    )
    with pytest.raises(ValueError, match="incompatible"):
        evaluate_video_quality(reference, other)
