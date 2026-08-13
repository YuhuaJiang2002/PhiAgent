"""CPU-side helpers for protected MiniMax-H3 flow retake experiments.

This module deliberately has no Torch, CUDA, or DiffSynth imports.  GPU entry
points can use the planning and mask helpers without making ``phiagent`` depend
on heavyweight inference packages at import time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class H3FlowRetakeWindow:
    """One source-frame window and its H3-aligned padded length."""

    index: int
    start_frame: int
    source_frames: int
    model_frames: int

    @property
    def end_frame_exclusive(self) -> int:
        return self.start_frame + self.source_frames


def h3_model_frame_count(frame_count: int) -> int:
    """Return the smallest H3-compatible frame count covering ``frame_count``."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    aligned = max(5, int(frame_count))
    while (aligned - 5) % 17:
        aligned += 1
    return aligned


def h3_latent_frame_count(model_frames: int) -> int:
    """Return the MiniMax-H3 VAE temporal latent length."""

    if model_frames < 5 or (model_frames - 5) % 17:
        raise ValueError("model_frames must satisfy model_frames = 17n + 5")
    return ((model_frames - 5) // 17) * 5 + 2


def plan_h3_flow_retake_windows(
    total_frames: int,
    *,
    window_frames: int = 124,
    overlap_frames: int = 22,
) -> tuple[H3FlowRetakeWindow, ...]:
    """Plan an H3 timeline while allowing a short padded final window.

    Unlike the original long-video runner, the final window is not pinned back
    to a full 124-frame span.  A short H3-compatible padded tail avoids a very
    large last overlap and keeps every real source frame represented once.
    """

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if window_frames < 5 or (window_frames - 5) % 17:
        raise ValueError("window_frames must satisfy window_frames = 17n + 5")
    if not 1 <= overlap_frames < window_frames:
        raise ValueError("overlap_frames must be in [1, window_frames - 1]")
    stride = window_frames - overlap_frames
    windows: list[H3FlowRetakeWindow] = []
    start = 0
    while start < total_frames:
        source_frames = min(window_frames, total_frames - start)
        model_frames = h3_model_frame_count(source_frames)
        windows.append(
            H3FlowRetakeWindow(
                index=len(windows),
                start_frame=start,
                source_frames=source_frames,
                model_frames=model_frames,
            )
        )
        if start + source_frames >= total_frames:
            break
        start += stride
    return tuple(windows)


def denoising_strength_for_sigma(target_sigma: float, *, flow_shift: float = 12.0) -> float:
    """Invert DiffSynth's shifted MiniMax-H3 sigma parameterization."""

    if not 0.0 < target_sigma < 1.0:
        raise ValueError("target_sigma must be strictly between zero and one")
    if flow_shift <= 0:
        raise ValueError("flow_shift must be positive")
    denominator = flow_shift - target_sigma * (flow_shift - 1.0)
    if denominator <= 0:
        raise ValueError("target_sigma is invalid for the requested flow shift")
    return target_sigma / denominator


def shifted_sigma(denoising_strength: float, *, flow_shift: float = 12.0) -> float:
    """Evaluate DiffSynth's shifted MiniMax-H3 starting sigma."""

    if not 0.0 <= denoising_strength <= 1.0:
        raise ValueError("denoising_strength must be in [0, 1]")
    if flow_shift <= 0:
        raise ValueError("flow_shift must be positive")
    return flow_shift * denoising_strength / (
        1.0 + (flow_shift - 1.0) * denoising_strength
    )


def source_ranges_for_latents(
    source_frames: int,
    latent_frames: int,
    *,
    temporal_radius: int = 1,
) -> tuple[tuple[int, int], ...]:
    """Map each latent time index to a conservative source-frame interval."""

    if source_frames <= 0 or latent_frames <= 0:
        raise ValueError("source_frames and latent_frames must be positive")
    if temporal_radius < 0:
        raise ValueError("temporal_radius must be non-negative")
    edges = [round(i * source_frames / latent_frames) for i in range(latent_frames + 1)]
    ranges = []
    for index in range(latent_frames):
        start = max(0, edges[index] - temporal_radius)
        end = min(source_frames, max(edges[index + 1], edges[index] + 1) + temporal_radius)
        ranges.append((start, end))
    return tuple(ranges)


def project_binary_masks_to_h3_latents(
    masks: Sequence[Any],
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    cv2: Any,
    np: Any,
    temporal_radius: int = 1,
    patch_size: int = 2,
) -> Any:
    """Conservatively project camera-frame masks into H3 video latents.

    The result is binary and constant inside every DiT spatial patch.  This is
    required because H3 currently reads one scalar per patch when assigning
    conditioned versus denoised token timesteps.
    """

    if not masks:
        raise ValueError("masks must not be empty")
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError("latent dimensions must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    ranges = source_ranges_for_latents(
        len(masks), latent_frames, temporal_radius=temporal_radius
    )
    projected = np.zeros((latent_frames, latent_height, latent_width), dtype=np.float32)
    for latent_index, (start, end) in enumerate(ranges):
        combined = np.maximum.reduce([np.asarray(mask) > 0 for mask in masks[start:end]])
        resized = cv2.resize(
            combined.astype(np.float32),
            (latent_width, latent_height),
            interpolation=cv2.INTER_AREA,
        )
        binary = resized > 0
        if patch_size > 1:
            for y in range(0, latent_height, patch_size):
                for x in range(0, latent_width, patch_size):
                    if binary[y : y + patch_size, x : x + patch_size].any():
                        binary[y : y + patch_size, x : x + patch_size] = True
        projected[latent_index] = binary.astype(np.float32)
    return projected


def feathered_protected_composite(
    base_frame: Any,
    generated_frame: Any,
    edit_mask: Any,
    *,
    cv2: Any,
    np: Any,
    feather_sigma: float,
) -> Any:
    """Composite a model frame over the accepted base inside a bounded mask."""

    if base_frame.shape != generated_frame.shape:
        raise ValueError("base and generated frames must have identical shapes")
    if edit_mask.shape != base_frame.shape[:2]:
        raise ValueError("edit_mask must match the frame height and width")
    alpha = (np.asarray(edit_mask, dtype=np.float32) / 255.0).clip(0.0, 1.0)
    if feather_sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather_sigma).clip(0.0, 1.0)
    result = (
        generated_frame.astype(np.float32) * alpha[..., None]
        + base_frame.astype(np.float32) * (1.0 - alpha[..., None])
    )
    return np.rint(result).clip(0, 255).astype(np.uint8)


def overlap_ramp(length: int) -> tuple[float, ...]:
    """Return a raised-cosine 0..1 ramp for overlap stitching."""

    if length <= 0:
        raise ValueError("length must be positive")
    if length == 1:
        return (1.0,)
    return tuple(
        0.5 - 0.5 * math.cos(math.pi * index / (length - 1))
        for index in range(length)
    )


def window_temporal_weight(
    windows: Sequence[H3FlowRetakeWindow], window_index: int, local_frame: int
) -> float:
    """Return the raised-cosine stitch weight for one window-local frame."""

    if not 0 <= window_index < len(windows):
        raise IndexError(window_index)
    window = windows[window_index]
    if not 0 <= local_frame < window.source_frames:
        raise IndexError(local_frame)
    weight = 1.0
    if window_index > 0:
        previous = windows[window_index - 1]
        overlap = previous.end_frame_exclusive - window.start_frame
        if overlap > 0 and local_frame < overlap:
            weight *= overlap_ramp(overlap)[local_frame]
    if window_index + 1 < len(windows):
        following = windows[window_index + 1]
        overlap = window.end_frame_exclusive - following.start_frame
        overlap_start = window.source_frames - overlap
        if overlap > 0 and local_frame >= overlap_start:
            weight *= 1.0 - overlap_ramp(overlap)[local_frame - overlap_start]
    return weight
