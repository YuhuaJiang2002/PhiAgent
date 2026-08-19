"""Fail-closed cloth-conservation gates for image-space video proposals.

The measurements in this module operate on tracked camera-frame material
landmarks.  They are useful as visual production gates, but they are not a
metric 3-D cloth state or proof of physical inextensibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, sqrt
from statistics import fmean
from typing import Mapping, Sequence


Point2D = tuple[float, float]


@dataclass(frozen=True)
class SleeveObservation:
    """One tracked sleeve seam in a named camera frame."""

    frame_index: int
    polyline_xy: tuple[Point2D, ...]
    confidence: float
    visible: bool = True


@dataclass(frozen=True)
class SleeveLengthThresholds:
    minimum_tracking_fraction: float = 0.98
    minimum_confidence: float = 0.55
    maximum_length_cv: float = 0.05
    maximum_relative_deviation: float = 0.10
    maximum_segment_relative_deviation: float = 0.18
    maximum_terminal_relative_deviation: float = 0.08

    def __post_init__(self) -> None:
        unit_interval = {
            "minimum_tracking_fraction": self.minimum_tracking_fraction,
            "minimum_confidence": self.minimum_confidence,
            "maximum_length_cv": self.maximum_length_cv,
            "maximum_relative_deviation": self.maximum_relative_deviation,
            "maximum_segment_relative_deviation": self.maximum_segment_relative_deviation,
            "maximum_terminal_relative_deviation": self.maximum_terminal_relative_deviation,
        }
        for name, value in unit_interval.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class SleeveLengthScore:
    sleeve_id: str
    baseline_length_pixels: float
    observed_fraction: float
    minimum_confidence: float
    length_cv: float
    maximum_relative_deviation: float
    maximum_segment_relative_deviation: float
    terminal_relative_deviation: float
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        def finite_or_none(value: float) -> float | None:
            return value if isfinite(value) else None

        return {
            "sleeve_id": self.sleeve_id,
            "baseline_length_pixels": self.baseline_length_pixels,
            "observed_fraction": self.observed_fraction,
            "minimum_confidence": self.minimum_confidence,
            "length_cv": finite_or_none(self.length_cv),
            "maximum_relative_deviation": finite_or_none(
                self.maximum_relative_deviation
            ),
            "maximum_segment_relative_deviation": finite_or_none(
                self.maximum_segment_relative_deviation
            ),
            "terminal_relative_deviation": finite_or_none(
                self.terminal_relative_deviation
            ),
            "passed": self.passed,
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class TshirtFoldCandidateScore:
    seed: int
    sleeve_scores: Mapping[str, SleeveLengthScore]
    frame_zero_score: float
    background_score: float
    action_completion_score: float
    temporal_score: float
    minimum_frame_zero_score: float
    minimum_background_score: float
    minimum_action_completion_score: float
    minimum_temporal_score: float
    human_review_passed: bool | None = None

    @property
    def automatic_failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        for sleeve_id, score in sorted(self.sleeve_scores.items()):
            if not score.passed:
                failures.append(f"sleeve_length:{sleeve_id}")
        if self.frame_zero_score < self.minimum_frame_zero_score:
            failures.append("frame_zero")
        if self.background_score < self.minimum_background_score:
            failures.append("background")
        if self.action_completion_score < self.minimum_action_completion_score:
            failures.append("action_completion")
        if self.temporal_score < self.minimum_temporal_score:
            failures.append("temporal")
        return tuple(failures)

    @property
    def automatic_passed(self) -> bool:
        return not self.automatic_failures

    @property
    def promoted(self) -> bool:
        return self.automatic_passed and self.human_review_passed is True

    @property
    def soft_score(self) -> float:
        values = (
            self.frame_zero_score,
            self.background_score,
            self.action_completion_score,
            self.temporal_score,
        )
        if any(value <= 0.0 for value in values):
            return 0.0
        return len(values) / sum(1.0 / value for value in values)

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "sleeve_scores": {
                key: value.to_dict() for key, value in self.sleeve_scores.items()
            },
            "frame_zero_score": self.frame_zero_score,
            "background_score": self.background_score,
            "action_completion_score": self.action_completion_score,
            "temporal_score": self.temporal_score,
            "soft_score": self.soft_score,
            "automatic_passed": self.automatic_passed,
            "automatic_failures": list(self.automatic_failures),
            "human_review_passed": self.human_review_passed,
            "promoted": self.promoted,
        }


def polyline_segment_lengths(polyline_xy: Sequence[Point2D]) -> tuple[float, ...]:
    if len(polyline_xy) < 2:
        raise ValueError("a sleeve polyline requires at least two material landmarks")
    segments = tuple(
        hypot(end[0] - start[0], end[1] - start[1])
        for start, end in zip(polyline_xy, polyline_xy[1:])
    )
    if any(length <= 0.0 for length in segments):
        raise ValueError("adjacent sleeve material landmarks must be distinct")
    return segments


def score_sleeve_length_conservation(
    sleeve_id: str,
    observations: Sequence[SleeveObservation],
    *,
    expected_frames: int,
    thresholds: SleeveLengthThresholds = SleeveLengthThresholds(),
) -> SleeveLengthScore:
    """Require visible material-landmark geometry before checking its length.

    Missing or low-confidence frames are failures rather than neutral samples.
    This prevents a candidate from hiding a length change behind an occlusion or
    tracker loss.
    """

    if expected_frames <= 1:
        raise ValueError("expected_frames must be greater than one")
    if not observations:
        raise ValueError("sleeve observations must not be empty")
    ordered = sorted(observations, key=lambda item: item.frame_index)
    if len({item.frame_index for item in ordered}) != len(ordered):
        raise ValueError("sleeve observations contain duplicate frame indices")
    if ordered[0].frame_index != 0:
        raise ValueError("the baseline sleeve observation must be frame zero")
    point_count = len(ordered[0].polyline_xy)
    if point_count < 2:
        raise ValueError("a sleeve polyline requires at least two points")
    for item in ordered:
        if not 0 <= item.frame_index < expected_frames:
            raise ValueError("a sleeve observation lies outside the video")
        if len(item.polyline_xy) != point_count:
            raise ValueError("all sleeve observations require corresponding landmarks")
        if not 0.0 <= item.confidence <= 1.0:
            raise ValueError("sleeve confidence must lie in [0, 1]")

    baseline_segments = polyline_segment_lengths(ordered[0].polyline_xy)
    baseline_length = sum(baseline_segments)
    valid = [
        item
        for item in ordered
        if item.visible and item.confidence >= thresholds.minimum_confidence
    ]
    observed_fraction = len(valid) / expected_frames
    valid_lengths = [sum(polyline_segment_lengths(item.polyline_xy)) for item in valid]
    minimum_confidence = min((item.confidence for item in valid), default=0.0)
    if valid_lengths:
        mean_length = fmean(valid_lengths)
        length_cv = (
            sqrt(fmean((length - mean_length) ** 2 for length in valid_lengths))
            / baseline_length
        )
        maximum_relative_deviation = max(
            abs(length / baseline_length - 1.0) for length in valid_lengths
        )
        maximum_segment_relative_deviation = max(
            abs(length / baseline - 1.0)
            for item in valid
            for length, baseline in zip(
                polyline_segment_lengths(item.polyline_xy), baseline_segments
            )
        )
    else:
        length_cv = float("inf")
        maximum_relative_deviation = float("inf")
        maximum_segment_relative_deviation = float("inf")
    terminal = next(
        (item for item in reversed(valid) if item.frame_index == expected_frames - 1),
        None,
    )
    terminal_relative_deviation = (
        abs(sum(polyline_segment_lengths(terminal.polyline_xy)) / baseline_length - 1.0)
        if terminal is not None
        else float("inf")
    )

    failures: list[str] = []
    if observed_fraction < thresholds.minimum_tracking_fraction:
        failures.append("tracking_fraction")
    if minimum_confidence < thresholds.minimum_confidence:
        failures.append("tracking_confidence")
    if length_cv > thresholds.maximum_length_cv:
        failures.append("length_cv")
    if maximum_relative_deviation > thresholds.maximum_relative_deviation:
        failures.append("length_relative_deviation")
    if maximum_segment_relative_deviation > thresholds.maximum_segment_relative_deviation:
        failures.append("segment_relative_deviation")
    if terminal_relative_deviation > thresholds.maximum_terminal_relative_deviation:
        failures.append("terminal_relative_deviation")
    return SleeveLengthScore(
        sleeve_id=sleeve_id,
        baseline_length_pixels=baseline_length,
        observed_fraction=observed_fraction,
        minimum_confidence=minimum_confidence,
        length_cv=length_cv,
        maximum_relative_deviation=maximum_relative_deviation,
        maximum_segment_relative_deviation=maximum_segment_relative_deviation,
        terminal_relative_deviation=terminal_relative_deviation,
        passed=not failures,
        failures=tuple(failures),
    )


def rank_tshirt_fold_candidates(
    candidates: Sequence[TshirtFoldCandidateScore],
    *,
    require_human_review: bool,
) -> tuple[TshirtFoldCandidateScore, ...]:
    """Rank only candidates that cleared every hard gate.

    Soft or mean quality can never override a failed sleeve gate.
    """

    eligible = [
        item
        for item in candidates
        if (item.promoted if require_human_review else item.automatic_passed)
    ]
    return tuple(sorted(eligible, key=lambda item: (-item.soft_score, item.seed)))
