"""Dependency-free proxy metrics over locally decoded grayscale video frames."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from phiagent.evaluation.object_instance import (
    NormalizedROI,
    ObjectInstanceMetrics,
    ObjectTrackerConfig,
    RGBFrames,
    decode_video,
    evaluate_object_instance,
)

TEMPORAL_JERK_SENSITIVITY = 32.0


@dataclass(frozen=True)
class DecodedFrames:
    frames: tuple[bytes, ...]
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("decoded frame dimensions must be positive")
        if not self.frames:
            raise ValueError("decoded frame sequence cannot be empty")
        expected = self.width * self.height
        if any(len(frame) != expected for frame in self.frames):
            raise ValueError("decoded grayscale frame has an unexpected byte count")


@dataclass(frozen=True)
class LocalVideoMetrics:
    motion_preservation: float
    target_identity: float
    object_consistency: float
    temporal_consistency: float
    motion_cosine: float
    motion_energy_ratio: float
    first_frame_anchor: float
    reference_structure: float
    object_contour_similarity: float
    object_color_similarity: float
    object_temporal_deformation: float
    object_tracking_coverage: float
    object_trajectory_similarity: float
    object_lift_recall: float
    candidate_temporal_jerk: float
    reference_temporal_jerk: float
    candidate_late_temporal_jerk: float
    reference_late_temporal_jerk: float
    global_temporal_consistency: float
    late_temporal_consistency: float
    regional_temporal_consistency: float

    def __post_init__(self) -> None:
        for field in (
            "motion_preservation",
            "target_identity",
            "object_consistency",
            "temporal_consistency",
            "motion_cosine",
            "motion_energy_ratio",
            "first_frame_anchor",
            "reference_structure",
            "object_contour_similarity",
            "object_color_similarity",
            "object_temporal_deformation",
            "object_tracking_coverage",
            "object_trajectory_similarity",
            "object_lift_recall",
        ):
            value = getattr(self, field)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be finite and in [0, 1]")
        for field in (
            "candidate_temporal_jerk",
            "reference_temporal_jerk",
            "candidate_late_temporal_jerk",
            "reference_late_temporal_jerk",
        ):
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and non-negative")
        for field in (
            "global_temporal_consistency",
            "late_temporal_consistency",
            "regional_temporal_consistency",
        ):
            value = getattr(self, field)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be finite and in [0, 1]")

    def scorecard(self) -> dict[str, float]:
        return {
            "motion_preservation": self.motion_preservation,
            "target_identity": self.target_identity,
            "object_consistency": self.object_consistency,
            "temporal_consistency": self.temporal_consistency,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_ffmpeg(explicit: Path | None = None) -> Path:
    if explicit is not None:
        resolved = explicit.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"ffmpeg executable does not exist: {resolved}")
        return resolved
    discovered = shutil.which("ffmpeg")
    if discovered is None:
        raise RuntimeError(
            "ffmpeg is required for local video evaluation; install it or pass --ffmpeg"
        )
    return Path(discovered).resolve()


def decode_grayscale(
    media: Path,
    ffmpeg: Path,
    *,
    width: int = 64,
    height: int = 64,
    sample_fps: float = 8.0,
    maximum_seconds: float = 4.0,
    image: bool = False,
) -> DecodedFrames:
    if not media.is_file():
        raise ValueError(f"evaluation media does not exist: {media}")
    if min(width, height) <= 0:
        raise ValueError("evaluation decode dimensions must be positive")
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("evaluation sample FPS must be finite and positive")
    if not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("evaluation duration must be finite and positive")

    filters = [f"scale={width}:{height}:flags=area", "format=gray"]
    if not image:
        filters.insert(0, f"fps={sample_fps}")
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-i",
        str(media),
        "-vf",
        ",".join(filters),
    ]
    if image:
        command.extend(["-frames:v", "1"])
    else:
        command.extend(["-t", str(maximum_seconds)])
    command.extend(["-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"])
    completed = subprocess.run(command, check=True, capture_output=True)
    frame_bytes = width * height
    if not completed.stdout or len(completed.stdout) % frame_bytes:
        raise ValueError(
            f"ffmpeg produced an invalid grayscale byte count for {media}: "
            f"{len(completed.stdout)}"
        )
    frames = tuple(
        completed.stdout[index : index + frame_bytes]
        for index in range(0, len(completed.stdout), frame_bytes)
    )
    return DecodedFrames(frames, width, height)


def _mean_absolute_difference(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("frame vectors must have equal non-zero length")
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def _correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("metric vectors must have equal non-zero length")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale < 1e-12 or right_scale < 1e-12:
        return 1.0 if left == right else 0.0
    return max(-1.0, min(1.0, covariance / (left_scale * right_scale)))


def _structure_similarity(left: bytes, right: bytes) -> float:
    correlation = (_correlation(tuple(left), tuple(right)) + 1.0) / 2.0
    mae_score = 1.0 - _mean_absolute_difference(left, right) / 255.0
    return max(0.0, min(1.0, 0.6 * correlation + 0.4 * mae_score))


def _resample(frames: tuple[bytes, ...], count: int) -> tuple[bytes, ...]:
    if count <= 0 or not frames:
        raise ValueError("resampling requires frames and a positive count")
    if count == 1:
        return (frames[0],)
    return tuple(
        frames[round(index * (len(frames) - 1) / (count - 1))]
        for index in range(count)
    )


def _motion_vector(previous: bytes, current: bytes, width: int, height: int) -> tuple[float, ...]:
    grid = 8
    if width % grid or height % grid:
        raise ValueError("metric frame dimensions must be divisible by 8")
    block_width = width // grid
    block_height = height // grid
    values: list[float] = []
    for grid_y in range(grid):
        for grid_x in range(grid):
            total = 0
            for y in range(grid_y * block_height, (grid_y + 1) * block_height):
                offset = y * width
                for x in range(grid_x * block_width, (grid_x + 1) * block_width):
                    index = offset + x
                    total += abs(current[index] - previous[index])
            values.append(total / (block_width * block_height))
    return tuple(values)


def _motion_signature(decoded: DecodedFrames, transition_count: int) -> tuple[float, ...]:
    frames = _resample(decoded.frames, transition_count + 1)
    return tuple(
        component
        for previous, current in zip(frames, frames[1:])
        for component in _motion_vector(
            previous, current, decoded.width, decoded.height
        )
    )


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm < 1e-12 or right_norm < 1e-12:
        return 1.0 if left_norm < 1e-12 and right_norm < 1e-12 else 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _energy_ratio(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_energy = sum(left) / len(left)
    right_energy = sum(right) / len(right)
    maximum = max(left_energy, right_energy)
    if maximum < 1e-12:
        return 1.0
    return min(left_energy, right_energy) / maximum


def _aligned_similarity(
    left: DecodedFrames,
    right: DecodedFrames,
    transform=lambda frame, width, height: frame,
) -> float:
    count = min(len(left.frames), len(right.frames))
    if count < 1:
        raise ValueError("aligned similarity requires non-empty sequences")
    left_frames = _resample(left.frames, count)
    right_frames = _resample(right.frames, count)
    return sum(
        _structure_similarity(
            transform(left_frame, left.width, left.height),
            transform(right_frame, right.width, right.height),
        )
        for left_frame, right_frame in zip(left_frames, right_frames)
    ) / count


def _temporal_jerk(decoded: DecodedFrames) -> float:
    if len(decoded.frames) < 3:
        raise ValueError("temporal consistency requires at least three frames")
    total = 0.0
    comparisons = 0
    for previous, current, following in zip(
        decoded.frames, decoded.frames[1:], decoded.frames[2:]
    ):
        total += sum(
            abs(following_value - 2 * current_value + previous_value)
            for previous_value, current_value, following_value in zip(
                previous, current, following
            )
        ) / len(current)
        comparisons += 1
    return total / comparisons / 255.0


def _late_frames(decoded: DecodedFrames) -> DecodedFrames:
    count = max(3, math.ceil(len(decoded.frames) / 3))
    return DecodedFrames(decoded.frames[-count:], decoded.width, decoded.height)


def _regional_temporal_jerk(decoded: DecodedFrames) -> tuple[float, ...]:
    if len(decoded.frames) < 3:
        raise ValueError("regional temporal consistency requires at least three frames")
    grid = 8
    block_width = decoded.width // grid
    block_height = decoded.height // grid
    if block_width == 0 or block_height == 0:
        raise ValueError("regional temporal consistency requires at least 8x8 frames")
    totals = [0.0] * (grid * grid)
    comparisons = 0
    for previous, current, following in zip(
        decoded.frames, decoded.frames[1:], decoded.frames[2:]
    ):
        for grid_y in range(grid):
            for grid_x in range(grid):
                total = 0
                for y in range(grid_y * block_height, (grid_y + 1) * block_height):
                    offset = y * decoded.width
                    for x in range(grid_x * block_width, (grid_x + 1) * block_width):
                        index = offset + x
                        total += abs(
                            following[index] - 2 * current[index] + previous[index]
                        )
                block_index = grid_y * grid + grid_x
                totals[block_index] += total / (block_width * block_height) / 255.0
        comparisons += 1
    return tuple(total / comparisons for total in totals)


def _excess_jerk_score(candidate: float, reference: float) -> float:
    return math.exp(
        -TEMPORAL_JERK_SENSITIVITY * max(0.0, candidate - reference)
    )


def evaluate_decoded_proxy(
    source: DecodedFrames,
    reference: DecodedFrames,
    target_image: DecodedFrames,
    candidate: DecodedFrames,
    object_metrics: ObjectInstanceMetrics,
) -> LocalVideoMetrics:
    dimensions = {(item.width, item.height) for item in (source, reference, target_image, candidate)}
    if len(dimensions) != 1:
        raise ValueError("all decoded proxy inputs must share dimensions")
    if len(target_image.frames) != 1:
        raise ValueError("target image decoding must produce exactly one frame")
    if min(len(source.frames), len(reference.frames), len(candidate.frames)) < 3:
        raise ValueError("source, reference, and candidate require at least three frames")

    transition_count = min(
        len(source.frames) - 1,
        len(candidate.frames) - 1,
        31,
    )
    source_motion = _motion_signature(source, transition_count)
    candidate_motion = _motion_signature(candidate, transition_count)
    motion_cosine = _cosine_similarity(source_motion, candidate_motion)
    motion_energy_ratio = _energy_ratio(source_motion, candidate_motion)
    motion_preservation = 0.75 * motion_cosine + 0.25 * motion_energy_ratio

    first_frame_anchor = _structure_similarity(
        target_image.frames[0], candidate.frames[0]
    )
    reference_structure = _aligned_similarity(reference, candidate)
    target_identity = 0.6 * first_frame_anchor + 0.4 * reference_structure

    candidate_jerk = _temporal_jerk(candidate)
    reference_jerk = _temporal_jerk(reference)
    global_temporal_consistency = _excess_jerk_score(
        candidate_jerk, reference_jerk
    )
    candidate_late_jerk = _temporal_jerk(_late_frames(candidate))
    reference_late_jerk = _temporal_jerk(_late_frames(reference))
    late_temporal_consistency = _excess_jerk_score(
        candidate_late_jerk, reference_late_jerk
    )
    candidate_regions = _regional_temporal_jerk(candidate)
    reference_regions = _regional_temporal_jerk(reference)
    regional_excess = sorted(
        max(0.0, candidate_value - reference_value)
        for candidate_value, reference_value in zip(
            candidate_regions, reference_regions
        )
    )
    tail_count = max(1, len(regional_excess) // 20)
    regional_temporal_consistency = math.exp(
        -TEMPORAL_JERK_SENSITIVITY
        * (sum(regional_excess[-tail_count:]) / tail_count)
    )
    temporal_consistency = min(
        global_temporal_consistency,
        late_temporal_consistency,
        regional_temporal_consistency,
    )

    return LocalVideoMetrics(
        motion_preservation=motion_preservation,
        target_identity=target_identity,
        object_consistency=object_metrics.object_consistency,
        temporal_consistency=temporal_consistency,
        motion_cosine=motion_cosine,
        motion_energy_ratio=motion_energy_ratio,
        first_frame_anchor=first_frame_anchor,
        reference_structure=reference_structure,
        object_contour_similarity=object_metrics.contour_similarity,
        object_color_similarity=object_metrics.color_similarity,
        object_temporal_deformation=object_metrics.temporal_deformation,
        object_tracking_coverage=object_metrics.tracking_coverage,
        object_trajectory_similarity=object_metrics.trajectory_similarity,
        object_lift_recall=object_metrics.lift_recall,
        candidate_temporal_jerk=candidate_jerk,
        reference_temporal_jerk=reference_jerk,
        candidate_late_temporal_jerk=candidate_late_jerk,
        reference_late_temporal_jerk=reference_late_jerk,
        global_temporal_consistency=global_temporal_consistency,
        late_temporal_consistency=late_temporal_consistency,
        regional_temporal_consistency=regional_temporal_consistency,
    )


def evaluate_local_videos(
    *,
    source: Path,
    reference: Path,
    target_image: Path,
    candidate: Path,
    ffmpeg: Path,
    width: int = 64,
    height: int = 64,
    sample_fps: float = 8.0,
    maximum_seconds: float = 4.0,
    object_roi: NormalizedROI,
    object_width: int = 224,
    object_height: int = 128,
) -> LocalVideoMetrics:
    decode_options = {
        "ffmpeg": ffmpeg,
        "width": width,
        "height": height,
        "sample_fps": sample_fps,
        "maximum_seconds": maximum_seconds,
    }
    object_source = RGBFrames(
        decode_video(
            source,
            ffmpeg,
            width=object_width,
            height=object_height,
            fps=max(1, round(sample_fps)),
            frame_num=max(3, round(sample_fps * maximum_seconds)),
            pixel_format="rgb24",
        ),
        object_width,
        object_height,
    )
    object_candidate = RGBFrames(
        decode_video(
            candidate,
            ffmpeg,
            width=object_width,
            height=object_height,
            fps=max(1, round(sample_fps)),
            frame_num=max(3, round(sample_fps * maximum_seconds)),
            pixel_format="rgb24",
        ),
        object_width,
        object_height,
    )
    object_metrics = evaluate_object_instance(
        object_source,
        object_candidate,
        ObjectTrackerConfig(initial_roi=object_roi),
    )
    return evaluate_decoded_proxy(
        decode_grayscale(source, **decode_options),
        decode_grayscale(reference, **decode_options),
        decode_grayscale(target_image, **decode_options, image=True),
        decode_grayscale(candidate, **decode_options),
        object_metrics,
    )


def evaluate_decoded_core_proxy(
    source: DecodedFrames,
    reference: DecodedFrames,
    target_image: DecodedFrames,
    candidate: DecodedFrames,
) -> dict[str, float]:
    """Evaluate non-object proxy dimensions without inventing object evidence."""

    unresolved = ObjectInstanceMetrics(
        contour_similarity=0.0,
        color_similarity=0.0,
        temporal_deformation=0.0,
        tracking_coverage=0.0,
        trajectory_similarity=0.0,
        lift_recall=0.0,
        object_consistency=0.0,
    )
    metrics = evaluate_decoded_proxy(
        source,
        reference,
        target_image,
        candidate,
        unresolved,
    )
    return {
        name: float(getattr(metrics, name))
        for name in (
            "motion_preservation",
            "target_identity",
            "temporal_consistency",
            "motion_cosine",
            "motion_energy_ratio",
            "first_frame_anchor",
            "reference_structure",
            "candidate_temporal_jerk",
            "reference_temporal_jerk",
            "candidate_late_temporal_jerk",
            "reference_late_temporal_jerk",
            "global_temporal_consistency",
            "late_temporal_consistency",
            "regional_temporal_consistency",
        )
    }


def evaluate_local_core_videos(
    *,
    source: Path,
    reference: Path,
    target_image: Path,
    candidate: Path,
    ffmpeg: Path,
    width: int = 64,
    height: int = 64,
    sample_fps: float = 8.0,
    maximum_seconds: float = 4.0,
) -> dict[str, float]:
    decode_options = {
        "ffmpeg": ffmpeg,
        "width": width,
        "height": height,
        "sample_fps": sample_fps,
        "maximum_seconds": maximum_seconds,
    }
    return evaluate_decoded_core_proxy(
        decode_grayscale(source, **decode_options),
        decode_grayscale(reference, **decode_options),
        decode_grayscale(target_image, **decode_options, image=True),
        decode_grayscale(candidate, **decode_options),
    )


def write_evaluation_evidence(
    path: Path,
    *,
    source: Path,
    reference: Path,
    target_image: Path,
    candidate: Path,
    backend_metadata: Path,
    ffmpeg: Path,
    metrics: LocalVideoMetrics,
    width: int,
    height: int,
    sample_fps: float,
    maximum_seconds: float,
) -> None:
    if path.exists():
        raise FileExistsError(f"evaluation evidence already exists: {path}")
    payload = {
        "schema_version": "3.0.0",
        "evaluator": "phiagent-local-video-evaluator-v4-object-trajectory",
        "inputs": {
            "source": str(source.resolve()),
            "source_sha256": file_sha256(source),
            "reference": str(reference.resolve()),
            "reference_sha256": file_sha256(reference),
            "target_image": str(target_image.resolve()),
            "target_image_sha256": file_sha256(target_image),
            "candidate": str(candidate.resolve()),
            "candidate_sha256": file_sha256(candidate),
            "backend_metadata": str(backend_metadata.resolve()),
            "backend_metadata_sha256": file_sha256(backend_metadata),
        },
        "decoder": {
            "ffmpeg": str(ffmpeg),
            "width": width,
            "height": height,
            "sample_fps": sample_fps,
            "maximum_seconds": maximum_seconds,
        },
        "metrics": asdict(metrics),
        "limitations": [
            "Scores are deterministic grayscale visual proxies, not PhiZero paper metrics.",
            (
                "object_consistency is the minimum of tracked-instance contour, color, "
                "temporal-deformation, tracking-coverage, relative-trajectory, and "
                "lift-recall scores."
            ),
            "target_identity combines first-frame anchoring and reference structure.",
            "temporal consistency uses human-calibrated excess-jerk sensitivity 32.0.",
            "temporal consistency is the minimum of global, late-window, and regional scores.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
