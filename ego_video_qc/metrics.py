"""Pure-Python metrics used by the ego-video quality gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from typing import Iterable


def parse_fraction(value: str | int | float | None) -> float | None:
    """Parse ffprobe's rational values such as ``30000/1001``."""

    if value is None or value == "N/A":
        return None
    try:
        result = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int | None
    codec: str | None = None


@dataclass(frozen=True)
class QCThresholds:
    min_width: int = 320
    min_height: int = 240
    min_fps: float = 10.0
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 600.0
    max_bad_frame_fraction: float = 0.02
    max_frozen_transition_fraction: float = 0.20
    motion_delta: float = 1.0

    def __post_init__(self) -> None:
        if self.min_width <= 0 or self.min_height <= 0:
            raise ValueError("minimum dimensions must be positive")
        if self.min_fps <= 0 or self.min_duration_seconds <= 0:
            raise ValueError("minimum FPS and duration must be positive")
        if self.max_duration_seconds < self.min_duration_seconds:
            raise ValueError("maximum duration must not be below minimum duration")
        if not 0 <= self.max_bad_frame_fraction <= 1:
            raise ValueError("bad-frame fraction must be in [0, 1]")
        if not 0 <= self.max_frozen_transition_fraction <= 1:
            raise ValueError("frozen-transition fraction must be in [0, 1]")
        if self.motion_delta < 0:
            raise ValueError("motion delta must be non-negative")


@dataclass(frozen=True)
class FrameSampleMetrics:
    sample_count: int
    mean_luminance: float
    bad_frame_fraction: float
    frozen_transition_fraction: float
    motion_fraction: float
    max_frame_delta: float

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample count must be positive")
        for name in (
            "mean_luminance",
            "bad_frame_fraction",
            "frozen_transition_fraction",
            "motion_fraction",
            "max_frame_delta",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("bad_frame_fraction", "frozen_transition_fraction", "motion_fraction"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be at most 1")


def _mean_absolute_difference(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("sampled frames must have equal non-zero size")
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def evaluate_samples(
    frames: Iterable[bytes],
    *,
    motion_delta: float = 1.0,
    bad_pixel_limit: int = 4,
    bad_pixel_fraction: float = 0.98,
) -> FrameSampleMetrics:
    """Calculate conservative luminance, freeze, and motion metrics.

    A frame is marked bad only when at least ``bad_pixel_fraction`` of pixels
    are near black or near white. This deliberately catches decode failures,
    not ordinary bright or dark scenes.
    """

    if motion_delta < 0 or not 0 <= bad_pixel_fraction <= 1:
        raise ValueError("invalid frame metric thresholds")
    materialized = tuple(frames)
    if not materialized or any(not frame for frame in materialized):
        raise ValueError("at least one non-empty frame is required")
    size = len(materialized[0])
    if any(len(frame) != size for frame in materialized):
        raise ValueError("sampled frames must have equal size")

    luminance = sum(sum(frame) / size for frame in materialized) / len(materialized)
    bad_count = sum(
        sum(pixel <= bad_pixel_limit or pixel >= 255 - bad_pixel_limit for pixel in frame)
        / size
        >= bad_pixel_fraction
        for frame in materialized
    )
    deltas = tuple(
        _mean_absolute_difference(previous, current)
        for previous, current in zip(materialized, materialized[1:])
    )
    frozen = sum(delta <= motion_delta for delta in deltas)
    moving = sum(delta > motion_delta for delta in deltas)
    transitions = len(deltas)
    return FrameSampleMetrics(
        sample_count=len(materialized),
        mean_luminance=luminance,
        bad_frame_fraction=bad_count / len(materialized),
        frozen_transition_fraction=frozen / transitions if transitions else 1.0,
        motion_fraction=moving / transitions if transitions else 0.0,
        max_frame_delta=max(deltas, default=0.0),
    )


def build_checks(metadata: VideoMetadata, samples: FrameSampleMetrics, thresholds: QCThresholds) -> list[dict[str, object]]:
    """Return serializable, fail-closed automatic checks."""

    checks = (
        ("resolution", metadata.width >= thresholds.min_width and metadata.height >= thresholds.min_height, f"{metadata.width}x{metadata.height}"),
        ("frame_rate", metadata.fps >= thresholds.min_fps, f"{metadata.fps:.3f} fps"),
        ("duration", thresholds.min_duration_seconds <= metadata.duration_seconds <= thresholds.max_duration_seconds, f"{metadata.duration_seconds:.3f} s"),
        ("decoded_samples", samples.sample_count > 0, f"{samples.sample_count} frames"),
        ("bad_frame_fraction", samples.bad_frame_fraction <= thresholds.max_bad_frame_fraction, f"{samples.bad_frame_fraction:.3f}"),
        ("motion", samples.frozen_transition_fraction <= thresholds.max_frozen_transition_fraction, f"{samples.frozen_transition_fraction:.3f} frozen"),
    )
    return [{"name": name, "passed": passed, "observed": observed} for name, passed, observed in checks]


def report_dict(metadata: VideoMetadata, samples: FrameSampleMetrics, checks: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "ego-video-qc/v1",
        "passed": all(bool(check["passed"]) for check in checks),
        "metadata": asdict(metadata),
        "samples": asdict(samples),
        "checks": checks,
        "manual_review": [
            "first-person ego viewpoint",
            "hand/object identity and occlusion",
            "task completion and contact correctness",
        ],
    }
