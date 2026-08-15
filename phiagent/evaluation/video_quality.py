"""Dependency-free, deterministic diagnostics for decoded grayscale video.

These are pixel-space proxies.  They measure continuity, motion, background
agreement, and sharpness; they do not measure perceptual realism or semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


def _unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ImageFrame:
    """A named, timestamped grayscale image in a declared image coordinate frame."""

    name: str
    frame_name: str
    timestamp: float
    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("image frame name must be non-empty")
        if not isinstance(self.frame_name, str) or not self.frame_name.strip():
            raise ValueError("image coordinate frame name must be non-empty")
        if (
            not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or min(self.width, self.height) <= 0
        ):
            raise ValueError("image dimensions must be positive")
        if not isinstance(self.pixels, bytes):
            raise ValueError("grayscale image pixels must be immutable bytes")
        if len(self.pixels) != self.width * self.height:
            raise ValueError("grayscale image byte length does not match dimensions")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError("image timestamp must be finite")


@dataclass(frozen=True)
class VideoQualityInput:
    """Frames, optional foreground masks, and optional image-plane trajectory."""

    frames: tuple[ImageFrame, ...]
    foreground_masks: tuple[bytes, ...] | None = None
    trajectory: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frames, tuple) or not all(
            isinstance(frame, ImageFrame) for frame in self.frames
        ):
            raise ValueError("video frames must be an immutable tuple of image frames")
        if len(self.frames) < 3:
            raise ValueError("video quality evaluation requires at least three frames")
        first = self.frames[0]
        for previous, frame in zip(self.frames, self.frames[1:]):
            if (frame.width, frame.height, frame.frame_name) != (
                first.width,
                first.height,
                first.frame_name,
            ):
                raise ValueError("video frames must have compatible dimensions and frame names")
            if frame.timestamp <= previous.timestamp:
                raise ValueError("video frame timestamps must be strictly increasing")
        if self.foreground_masks is not None:
            if not isinstance(self.foreground_masks, tuple) or any(
                not isinstance(mask, bytes) for mask in self.foreground_masks
            ):
                raise ValueError("foreground masks must be an immutable tuple of bytes")
            if len(self.foreground_masks) != len(self.frames):
                raise ValueError("foreground mask count must match frame count")
            expected = first.width * first.height
            if any(len(mask) != expected for mask in self.foreground_masks):
                raise ValueError("foreground mask byte length does not match frame dimensions")
            if any(value not in (0, 1) for mask in self.foreground_masks for value in mask):
                raise ValueError("foreground masks must contain only 0 or 1")
        if self.trajectory is not None:
            if not isinstance(self.trajectory, tuple):
                raise ValueError("trajectory must be an immutable tuple")
            if len(self.trajectory) != len(self.frames):
                raise ValueError("trajectory count must match frame count")
            if any(
                not isinstance(point, tuple)
                or len(point) != 2
                or any(not math.isfinite(value) for value in point)
                for point in self.trajectory
            ):
                raise ValueError("trajectory points must contain two finite values")


@dataclass(frozen=True)
class VideoQualityConfig:
    """Thresholds use image scale and explicitly declared time and jerk units.

    ``timestamp_unit_seconds`` is the physical duration represented by one
    timestamp unit. ``jerk_normalization`` is a normalized-position-per-second-
    cubed reference that makes the smoothness exponent dimensionless.
    """

    roi: tuple[float, float, float, float] | None = None
    late_fraction: float = 1.0 / 3.0
    temporal_window: int = 3
    max_translation_pixels: int = 3
    activity_threshold: float = 0.01
    articulation_threshold: float = 0.015
    trajectory_scale: float | None = None
    timestamp_unit_seconds: float = 1.0
    jerk_normalization: float = 1.0
    require_activity: bool = False
    require_articulation: bool = False

    def __post_init__(self) -> None:
        if self.roi is not None:
            x, y, width, height = self.roi
            if not all(math.isfinite(value) for value in self.roi) or not (
                0.0 <= x < 1.0
                and 0.0 <= y < 1.0
                and 0.0 < width <= 1.0 - x
                and 0.0 < height <= 1.0 - y
            ):
                raise ValueError("ROI must be a finite normalized (x, y, width, height)")
        if not math.isfinite(self.late_fraction) or not 0.0 < self.late_fraction <= 1.0:
            raise ValueError("late_fraction must be in (0, 1]")
        if self.temporal_window < 3:
            raise ValueError("temporal_window must be at least three")
        if self.max_translation_pixels < 0:
            raise ValueError("max_translation_pixels must be non-negative")
        _finite_positive(self.activity_threshold, "activity_threshold")
        _finite_positive(self.articulation_threshold, "articulation_threshold")
        if self.trajectory_scale is not None:
            _finite_positive(self.trajectory_scale, "trajectory_scale")
        _finite_positive(self.timestamp_unit_seconds, "timestamp_unit_seconds")
        _finite_positive(self.jerk_normalization, "jerk_normalization")


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    """Image-plane derivative magnitudes per declared seconds and image scale."""

    scale: float
    speeds: tuple[float, ...]
    accelerations: tuple[float, ...]
    jerks: tuple[float, ...]
    smoothness_score: float
    timestamp_unit_seconds: float = 1.0
    jerk_normalization: float = 1.0

    def __post_init__(self) -> None:
        _finite_positive(self.scale, "trajectory scale")
        _finite_positive(self.timestamp_unit_seconds, "timestamp_unit_seconds")
        _finite_positive(self.jerk_normalization, "jerk_normalization")
        for value in (*self.speeds, *self.accelerations, *self.jerks, self.smoothness_score):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("trajectory diagnostics must be finite and non-negative")
        if self.smoothness_score > 1.0:
            raise ValueError("trajectory smoothness score must be in [0, 1]")


@dataclass(frozen=True)
class VideoQualityScorecard:
    """Raw diagnostics and bounded component scores for downstream integration."""

    temporal_global_score: float
    temporal_late_score: float
    temporal_roi_score: float
    temporal_worst_window_score: float
    temporal_score: float
    motion_activity_score: float
    articulation_score: float
    motion_smoothness_score: float
    motion_requirement_score: float
    background_preservation_score: float
    sharpness_full_score: float
    sharpness_roi_score: float
    sharpness_score: float
    candidate_trajectory: TrajectoryDiagnostics
    reference_trajectory: TrajectoryDiagnostics
    mean_background_error: float
    mean_activity: float
    mean_articulation: float
    estimated_translations: tuple[tuple[int, int], ...]
    candidate_temporal_errors: tuple[float, ...]
    reference_temporal_errors: tuple[float, ...]
    candidate_roi_temporal_errors: tuple[float, ...]
    reference_roi_temporal_errors: tuple[float, ...]
    background_errors: tuple[float, ...]
    limitations: Literal[
        "Deterministic grayscale pixel proxies; not a perceptual realism metric."
    ] = "Deterministic grayscale pixel proxies; not a perceptual realism metric."

    def __post_init__(self) -> None:
        for value in (
            self.temporal_global_score,
            self.temporal_late_score,
            self.temporal_roi_score,
            self.temporal_worst_window_score,
            self.temporal_score,
            self.motion_activity_score,
            self.articulation_score,
            self.motion_smoothness_score,
            self.motion_requirement_score,
            self.background_preservation_score,
            self.sharpness_full_score,
            self.sharpness_roi_score,
            self.sharpness_score,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("component scores must be finite and in [0, 1]")
        for value in (self.mean_background_error, self.mean_activity, self.mean_articulation):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("raw diagnostics must be finite and non-negative")
        for value in (
            *self.candidate_temporal_errors,
            *self.reference_temporal_errors,
            *self.candidate_roi_temporal_errors,
            *self.reference_roi_temporal_errors,
            *self.background_errors,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("raw temporal diagnostics must be finite and non-negative")

    def component_scores(self) -> dict[str, float]:
        return {
            "temporal": self.temporal_score,
            "temporal_global": self.temporal_global_score,
            "temporal_late": self.temporal_late_score,
            "temporal_roi": self.temporal_roi_score,
            "temporal_worst_window": self.temporal_worst_window_score,
            "motion_activity": self.motion_activity_score,
            "articulation": self.articulation_score,
            "motion_smoothness": self.motion_smoothness_score,
            "motion_requirement": self.motion_requirement_score,
            "background_preservation": self.background_preservation_score,
            "sharpness_full": self.sharpness_full_score,
            "sharpness_roi": self.sharpness_roi_score,
            "sharpness": self.sharpness_score,
        }


def _region_indices(
    width: int, height: int, roi: tuple[float, float, float, float] | None
) -> tuple[int, ...]:
    if roi is None:
        return tuple(range(width * height))
    x, y, region_width, region_height = roi
    left, top = int(x * width), int(y * height)
    right = max(left + 1, math.ceil((x + region_width) * width))
    bottom = max(top + 1, math.ceil((y + region_height) * height))
    return tuple(
        row * width + column for row in range(top, bottom) for column in range(left, right)
    )


def _mean_abs(left: bytes, right: bytes, indices: tuple[int, ...]) -> float:
    return sum(abs(left[index] - right[index]) for index in indices) / (255.0 * len(indices))


def _shift_error(
    reference: ImageFrame, candidate: ImageFrame, mask: bytes | None, dx: int, dy: int
) -> float:
    total = 0
    count = 0
    for y in range(reference.height):
        source_y = y - dy
        if not 0 <= source_y < reference.height:
            continue
        for x in range(reference.width):
            source_x = x - dx
            if not 0 <= source_x < reference.width:
                continue
            target = y * reference.width + x
            source = source_y * reference.width + source_x
            if mask is not None and (mask[target] or mask[source]):
                continue
            total += abs(reference.pixels[target] - candidate.pixels[source])
            count += 1
    if not count:
        raise ValueError("foreground mask leaves no valid background pixels for comparison")
    return total / (255.0 * count)


def _combined_mask(
    candidate: VideoQualityInput, reference: VideoQualityInput, index: int
) -> bytes | None:
    candidate_mask = candidate.foreground_masks[index] if candidate.foreground_masks else None
    reference_mask = reference.foreground_masks[index] if reference.foreground_masks else None
    if candidate_mask is None:
        return reference_mask
    if reference_mask is None:
        return candidate_mask
    return bytes(left or right for left, right in zip(candidate_mask, reference_mask))


def _translation(
    reference: ImageFrame, candidate: ImageFrame, mask: bytes | None, maximum: int
) -> tuple[int, int, float]:
    choices: list[tuple[int, int, float]] = []
    for dy in range(-maximum, maximum + 1):
        for dx in range(-maximum, maximum + 1):
            try:
                error = _shift_error(reference, candidate, mask, dx, dy)
            except ValueError:
                continue
            choices.append((dx, dy, error))
    if not choices:
        raise ValueError("foreground mask leaves no valid background pixels for comparison")
    return min(
        choices,
        key=lambda choice: (
            choice[2],
            abs(choice[0]) + abs(choice[1]),
            choice[:2],
        ),
    )


def _temporal_errors(video: VideoQualityInput, indices: tuple[int, ...]) -> tuple[float, ...]:
    return tuple(
        sum(
            abs(next_frame.pixels[index] - 2 * current.pixels[index] + previous.pixels[index])
            for index in indices
        )
        / (255.0 * len(indices))
        for previous, current, next_frame in zip(video.frames, video.frames[1:], video.frames[2:])
    )


def _temporal_score(candidate: tuple[float, ...], reference: tuple[float, ...]) -> float:
    return math.exp(
        -24.0
        * sum(max(0.0, value - baseline) for value, baseline in zip(candidate, reference))
        / len(candidate)
    )


def _trajectory(
    video: VideoQualityInput,
    translations: tuple[tuple[int, int], ...],
    scale: float,
    timestamp_unit_seconds: float,
    jerk_normalization: float,
) -> TrajectoryDiagnostics:
    points = (
        video.trajectory
        if video.trajectory is not None
        else tuple((float(x), float(y)) for x, y in translations)
    )
    intervals = tuple(
        (right.timestamp - left.timestamp) * timestamp_unit_seconds
        for left, right in zip(video.frames, video.frames[1:])
    )
    velocities = tuple(
        (
            (right[0] - left[0]) / interval / scale,
            (right[1] - left[1]) / interval / scale,
        )
        for left, right, interval in zip(points, points[1:], intervals)
    )
    speeds = tuple(math.hypot(*velocity) for velocity in velocities)
    acceleration_vectors = tuple(
        (
            (right[0] - left[0]) / ((intervals[index] + intervals[index + 1]) / 2.0),
            (right[1] - left[1]) / ((intervals[index] + intervals[index + 1]) / 2.0),
        )
        for index, (left, right) in enumerate(zip(velocities, velocities[1:]))
    )
    accelerations = tuple(math.hypot(*acceleration) for acceleration in acceleration_vectors)
    jerk_intervals = tuple(
        (intervals[index] + intervals[index + 1]) / 2.0 for index in range(len(accelerations))
    )
    jerk_vectors = tuple(
        (
            (right[0] - left[0]) / ((jerk_intervals[index] + jerk_intervals[index + 1]) / 2.0),
            (right[1] - left[1]) / ((jerk_intervals[index] + jerk_intervals[index + 1]) / 2.0),
        )
        for index, (left, right) in enumerate(zip(acceleration_vectors, acceleration_vectors[1:]))
    )
    jerks = tuple(math.hypot(*jerk) for jerk in jerk_vectors)
    smoothness = math.exp(-(sum(jerks) / len(jerks)) / jerk_normalization) if jerks else 1.0
    return TrajectoryDiagnostics(
        scale=scale,
        speeds=speeds,
        accelerations=accelerations,
        jerks=jerks,
        smoothness_score=smoothness,
        timestamp_unit_seconds=timestamp_unit_seconds,
        jerk_normalization=jerk_normalization,
    )


def _sharpness(frame: ImageFrame, indices: tuple[int, ...]) -> float:
    allowed = set(indices)
    total = count = 0
    for index in indices:
        y, x = divmod(index, frame.width)
        if 0 < x < frame.width - 1 and 0 < y < frame.height - 1:
            neighbors = (index - 1, index + 1, index - frame.width, index + frame.width)
            if all(neighbor in allowed for neighbor in neighbors):
                total += abs(
                    4 * frame.pixels[index] - sum(frame.pixels[neighbor] for neighbor in neighbors)
                )
                count += 1
    return total / (1020.0 * count) if count else 0.0


def _sharpness_score(
    candidate: VideoQualityInput, reference: VideoQualityInput, indices: tuple[int, ...]
) -> float:
    candidate_value = sum(_sharpness(frame, indices) for frame in candidate.frames) / len(
        candidate.frames
    )
    reference_value = sum(_sharpness(frame, indices) for frame in reference.frames) / len(
        reference.frames
    )
    return _unit(candidate_value / reference_value) if reference_value > 1e-12 else 1.0


def evaluate_video_quality(
    candidate: VideoQualityInput,
    reference: VideoQualityInput,
    config: VideoQualityConfig = VideoQualityConfig(),
) -> VideoQualityScorecard:
    """Evaluate equal-horizon candidate/reference videos without external dependencies."""
    if len(candidate.frames) != len(reference.frames):
        raise ValueError("candidate and reference must have the same horizon length")
    candidate_first, reference_first = candidate.frames[0], reference.frames[0]
    if (candidate_first.width, candidate_first.height, candidate_first.frame_name) != (
        reference_first.width,
        reference_first.height,
        reference_first.frame_name,
    ):
        raise ValueError("candidate and reference image frames are incompatible")
    if any(
        abs(left.timestamp - right.timestamp) > 1e-9
        for left, right in zip(candidate.frames, reference.frames)
    ):
        raise ValueError("candidate and reference timestamps must match")

    width, height = candidate_first.width, candidate_first.height
    full = _region_indices(width, height, None)
    roi = _region_indices(width, height, config.roi)
    candidate_full, reference_full = (
        _temporal_errors(candidate, full),
        _temporal_errors(reference, full),
    )
    candidate_roi, reference_roi = (
        _temporal_errors(candidate, roi),
        _temporal_errors(reference, roi),
    )
    global_score = _temporal_score(candidate_full, reference_full)
    late_count = max(1, math.ceil(len(candidate_full) * config.late_fraction))
    late_score = _temporal_score(candidate_full[-late_count:], reference_full[-late_count:])
    roi_score = _temporal_score(candidate_roi, reference_roi)
    window = min(config.temporal_window - 2, len(candidate_full))
    worst_score = min(
        _temporal_score(
            candidate_full[index : index + window], reference_full[index : index + window]
        )
        for index in range(len(candidate_full) - window + 1)
    )

    translations: list[tuple[int, int]] = []
    background_errors: list[float] = []
    for index, (candidate_frame, reference_frame) in enumerate(
        zip(candidate.frames, reference.frames)
    ):
        mask = _combined_mask(candidate, reference, index)
        dx, dy, error = _translation(
            reference_frame, candidate_frame, mask, config.max_translation_pixels
        )
        translations.append((dx, dy))
        background_errors.append(error)
    motion_steps: list[tuple[int, int]] = []
    residual: list[float] = []
    for index in range(1, len(candidate.frames)):
        mask = candidate.foreground_masks[index] if candidate.foreground_masks else None
        dx, dy, error = _translation(
            candidate.frames[index - 1],
            candidate.frames[index],
            mask,
            config.max_translation_pixels,
        )
        motion_steps.append((dx, dy))
        residual.append(error)
    activity = tuple(math.hypot(dx, dy) / math.hypot(width, height) for dx, dy in motion_steps)
    # Residual change is activity too: a moving local object should not be
    # mistaken for a frozen camera merely because global compensation is zero.
    mean_activity = sum(
        max(global_motion, local_motion) for global_motion, local_motion in zip(activity, residual)
    ) / len(activity)
    mean_articulation = sum(residual) / len(residual)
    activity_score = _unit(mean_activity / config.activity_threshold)
    articulation_score = _unit(mean_articulation / config.articulation_threshold)
    scale = config.trajectory_scale or math.hypot(width, height)
    candidate_positions = [(0, 0)]
    reference_positions = [(0, 0)]
    for index, (dx, dy) in enumerate(motion_steps, start=1):
        previous_x, previous_y = candidate_positions[-1]
        candidate_positions.append((previous_x + dx, previous_y + dy))
        reference_mask = reference.foreground_masks[index] if reference.foreground_masks else None
        reference_dx, reference_dy, _ = _translation(
            reference.frames[index - 1],
            reference.frames[index],
            reference_mask,
            config.max_translation_pixels,
        )
        reference_x, reference_y = reference_positions[-1]
        reference_positions.append((reference_x + reference_dx, reference_y + reference_dy))
    candidate_trajectory = _trajectory(
        candidate,
        tuple(candidate_positions),
        scale,
        config.timestamp_unit_seconds,
        config.jerk_normalization,
    )
    reference_trajectory = _trajectory(
        reference,
        tuple(reference_positions),
        scale,
        config.timestamp_unit_seconds,
        config.jerk_normalization,
    )
    requirement = 1.0
    if config.require_activity:
        requirement = min(requirement, activity_score)
    if config.require_articulation:
        requirement = min(requirement, articulation_score)
    full_sharpness = _sharpness_score(candidate, reference, full)
    roi_sharpness = _sharpness_score(candidate, reference, roi)
    return VideoQualityScorecard(
        global_score,
        late_score,
        roi_score,
        worst_score,
        min(global_score, late_score, roi_score, worst_score),
        activity_score,
        articulation_score,
        candidate_trajectory.smoothness_score,
        requirement,
        math.exp(-8.0 * (sum(background_errors) / len(background_errors))),
        full_sharpness,
        roi_sharpness,
        min(full_sharpness, roi_sharpness),
        candidate_trajectory,
        reference_trajectory,
        sum(background_errors) / len(background_errors),
        mean_activity,
        mean_articulation,
        tuple(translations),
        candidate_full,
        reference_full,
        candidate_roi,
        reference_roi,
        tuple(background_errors),
    )
