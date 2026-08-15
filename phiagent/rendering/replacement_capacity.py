"""Capacity planning for large JoyAI replacement-video campaigns."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from phiagent.rendering.joyai_video_edit import causal_padded_frame_count


@dataclass(frozen=True)
class JoyAIThroughputBenchmark:
    """Measured end-to-end rates used by the capacity model."""

    generated_frames: int = 665
    generation_wall_seconds: float = 105.692
    postprocessed_frames: int = 660
    postprocess_wall_seconds: float = 12.761
    hardware: str = "NVIDIA A800"
    source: str = "docs/JOYAI_SCISSORS_CONTACT.md"

    def validate(self) -> None:
        values = (
            self.generated_frames,
            self.generation_wall_seconds,
            self.postprocessed_frames,
            self.postprocess_wall_seconds,
        )
        if any(not math.isfinite(float(value)) or value <= 0 for value in values):
            raise ValueError("benchmark frame counts and durations must be finite and positive")

    @property
    def generation_fps(self) -> float:
        self.validate()
        return self.generated_frames / self.generation_wall_seconds

    @property
    def postprocess_fps(self) -> float:
        self.validate()
        return self.postprocessed_frames / self.postprocess_wall_seconds

    def to_manifest(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "generation_fps": self.generation_fps,
            "postprocess_fps": self.postprocess_fps,
        }


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _utilization(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1]")


def estimate_joyai_replacement_capacity(
    *,
    video_hours: float,
    fps: int,
    average_clip_frames: int,
    gpu_count: int,
    gpu_utilization: float,
    postprocess_workers: int,
    postprocess_utilization: float,
    session_overhead_seconds: float = 0.0,
    review_bitrate_mbps: float = 50.0,
    protocol_jpeg_kib: float = 200.0,
    chunk_frames: int = 8,
    benchmark: JoyAIThroughputBenchmark | None = None,
) -> dict[str, Any]:
    """Estimate compute, wall time, and storage for one replacement campaign.

    The model assumes independent clips can be distributed evenly across resident
    JoyAI services. It does not assume that one causal stream can be split without
    a quality cost.
    """

    benchmark = benchmark or JoyAIThroughputBenchmark()
    benchmark.validate()
    _positive_finite(video_hours, "video_hours")
    _positive_finite(float(fps), "fps")
    _positive_finite(float(average_clip_frames), "average_clip_frames")
    _positive_finite(float(gpu_count), "gpu_count")
    _positive_finite(float(postprocess_workers), "postprocess_workers")
    _utilization(gpu_utilization, "gpu_utilization")
    _utilization(postprocess_utilization, "postprocess_utilization")
    if not math.isfinite(session_overhead_seconds) or session_overhead_seconds < 0:
        raise ValueError("session_overhead_seconds must be finite and non-negative")
    _positive_finite(review_bitrate_mbps, "review_bitrate_mbps")
    _positive_finite(protocol_jpeg_kib, "protocol_jpeg_kib")
    _positive_finite(float(chunk_frames), "chunk_frames")

    source_frames = math.ceil(video_hours * 3600 * fps)
    full_clips, remainder_frames = divmod(source_frames, average_clip_frames)
    clip_count = full_clips + int(remainder_frames > 0)
    padded_full_clip_frames = causal_padded_frame_count(
        average_clip_frames, chunk_frames=chunk_frames
    )
    generated_frames = full_clips * padded_full_clip_frames
    if remainder_frames:
        generated_frames += causal_padded_frame_count(
            remainder_frames, chunk_frames=chunk_frames
        )

    generation_compute_seconds = (
        generated_frames / benchmark.generation_fps
        + clip_count * session_overhead_seconds
    )
    postprocess_compute_seconds = source_frames / benchmark.postprocess_fps
    generation_wall_seconds = generation_compute_seconds / (
        gpu_count * gpu_utilization
    )
    postprocess_wall_seconds = postprocess_compute_seconds / (
        postprocess_workers * postprocess_utilization
    )
    pipelined_wall_seconds = max(generation_wall_seconds, postprocess_wall_seconds)
    sequential_wall_seconds = generation_wall_seconds + postprocess_wall_seconds

    balanced_postprocess_workers = max(
        1,
        math.ceil(
            postprocess_compute_seconds
            * gpu_count
            * gpu_utilization
            / (generation_compute_seconds * postprocess_utilization)
        ),
    )
    active_spools = min(gpu_count, clip_count)
    spool_bytes_per_worker = padded_full_clip_frames * protocol_jpeg_kib * 1024
    review_bytes = video_hours * 3600 * review_bitrate_mbps * 1_000_000 / 8

    return {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "benchmark": benchmark.to_manifest(),
        "assumptions": {
            "video_hours": video_hours,
            "fps": fps,
            "average_clip_frames": average_clip_frames,
            "average_clip_seconds": average_clip_frames / fps,
            "chunk_frames": chunk_frames,
            "gpu_count": gpu_count,
            "gpu_utilization": gpu_utilization,
            "postprocess_workers": postprocess_workers,
            "postprocess_utilization": postprocess_utilization,
            "session_overhead_seconds": session_overhead_seconds,
            "review_bitrate_mbps": review_bitrate_mbps,
            "protocol_jpeg_kib": protocol_jpeg_kib,
        },
        "workload": {
            "source_frames": source_frames,
            "clip_count": clip_count,
            "full_clips": full_clips,
            "remainder_frames": remainder_frames,
            "generated_frames_with_padding": generated_frames,
            "tail_padding_frames": generated_frames - source_frames,
            "tail_padding_percent": (generated_frames / source_frames - 1) * 100,
            "legacy_protocol_small_files": generated_frames,
        },
        "compute": {
            "generation_gpu_hours": generation_compute_seconds / 3600,
            "postprocess_worker_hours": postprocess_compute_seconds / 3600,
            "session_overhead_gpu_hours": (
                clip_count * session_overhead_seconds / 3600
            ),
        },
        "calendar": {
            "generation_hours": generation_wall_seconds / 3600,
            "postprocess_hours": postprocess_wall_seconds / 3600,
            "pipelined_hours": pipelined_wall_seconds / 3600,
            "pipelined_days": pipelined_wall_seconds / 86400,
            "sequential_hours": sequential_wall_seconds / 3600,
            "realtime_factor": pipelined_wall_seconds / (video_hours * 3600),
            "output_video_hours_per_day": (
                video_hours * 86400 / pipelined_wall_seconds
            ),
        },
        "storage": {
            "estimated_review_bytes": round(review_bytes),
            "estimated_review_decimal_tb": review_bytes / 1_000_000_000_000,
            "estimated_protocol_write_bytes": round(
                generated_frames * protocol_jpeg_kib * 1024
            ),
            "bounded_spool_bytes_per_worker": round(spool_bytes_per_worker),
            "bounded_spool_fleet_peak_bytes": round(
                spool_bytes_per_worker * active_spools
            ),
        },
        "recommendation": {
            "balanced_postprocess_workers": balanced_postprocess_workers,
            "parallelism_unit": "independent_source_clip",
            "throughput_client_flags": [
                "--throughput-mode",
                "--no-profile-timings",
            ],
        },
        "limitations": [
            "Generation throughput is one measured A800 run, not a multi-run confidence interval.",
            "Session startup overhead defaults to zero because it has not been measured.",
            "Multi-GPU scaling assumes independent clips and resident one-GPU services.",
            "CRF output size is content-dependent; review storage uses an explicit bitrate assumption.",
            "The optimized client path still requires a real A800 acceptance benchmark.",
        ],
    }
