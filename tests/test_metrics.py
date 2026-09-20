from __future__ import annotations

import pytest

from ego_video_qc.metrics import QCThresholds, evaluate_samples, parse_fraction


def test_parse_ffprobe_fraction() -> None:
    assert parse_fraction("30000/1001") == pytest.approx(29.97002997)
    assert parse_fraction("N/A") is None


def test_moving_samples_have_motion_and_no_bad_frames() -> None:
    frames = tuple(bytes((index * 25 + pixel) % 256 for pixel in range(64)) for index in range(4))
    metrics = evaluate_samples(frames)
    assert metrics.sample_count == 4
    assert metrics.bad_frame_fraction == 0
    assert metrics.motion_fraction == 1
    assert metrics.frozen_transition_fraction == 0


def test_frozen_samples_are_detected() -> None:
    frame = bytes([80] * 64)
    metrics = evaluate_samples((frame, frame, frame))
    assert metrics.frozen_transition_fraction == 1
    assert metrics.motion_fraction == 0


def test_black_samples_are_bad() -> None:
    metrics = evaluate_samples((bytes([0] * 64), bytes([255] * 64)))
    assert metrics.bad_frame_fraction == 1


def test_thresholds_fail_closed_for_invalid_values() -> None:
    with pytest.raises(ValueError):
        QCThresholds(min_width=0)
