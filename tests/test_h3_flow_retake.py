from __future__ import annotations

import numpy as np

from phiagent.rendering.h3_flow_retake import (
    denoising_strength_for_sigma,
    feathered_protected_composite,
    h3_latent_frame_count,
    overlap_ramp,
    plan_h3_flow_retake_windows,
    project_binary_masks_to_h3_latents,
    shifted_sigma,
    source_ranges_for_latents,
    window_temporal_weight,
)


def test_sigma_parameterization_round_trip() -> None:
    for sigma in (0.12, 0.20, 0.28):
        strength = denoising_strength_for_sigma(sigma)
        assert abs(shifted_sigma(strength) - sigma) < 1e-9


def test_flow_retake_plan_uses_a_short_aligned_tail() -> None:
    windows = plan_h3_flow_retake_windows(660)
    assert [window.start_frame for window in windows] == [0, 102, 204, 306, 408, 510, 612]
    assert windows[-1].source_frames == 48
    assert windows[-1].model_frames == 56
    assert all((window.model_frames - 5) % 17 == 0 for window in windows)


def test_h3_latent_frame_contract() -> None:
    assert h3_latent_frame_count(56) == 17
    assert h3_latent_frame_count(124) == 37


def test_source_ranges_cover_all_frames() -> None:
    ranges = source_ranges_for_latents(56, 17, temporal_radius=0)
    covered = set()
    for start, end in ranges:
        covered.update(range(start, end))
    assert covered == set(range(56))


def test_latent_projection_expands_to_dit_patch() -> None:
    import cv2

    masks = [np.zeros((16, 16), dtype=np.uint8) for _ in range(5)]
    masks[2][7, 7] = 255
    projected = project_binary_masks_to_h3_latents(
        masks,
        latent_frames=2,
        latent_height=4,
        latent_width=4,
        cv2=cv2,
        np=np,
        temporal_radius=1,
    )
    assert projected.shape == (2, 4, 4)
    assert projected[0].sum() in (0, 4)
    assert projected[1].sum() == 4


def test_protected_composite_is_exact_outside_mask() -> None:
    import cv2

    base = np.full((8, 8, 3), 10, dtype=np.uint8)
    generated = np.full((8, 8, 3), 210, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    result = feathered_protected_composite(
        base, generated, mask, cv2=cv2, np=np, feather_sigma=0
    )
    assert np.array_equal(result[0], base[0])
    assert np.array_equal(result[:, 0], base[:, 0])
    assert np.array_equal(result[3, 3], generated[3, 3])


def test_overlap_ramp_is_monotonic() -> None:
    ramp = overlap_ramp(9)
    assert ramp[0] == 0.0
    assert ramp[-1] == 1.0
    assert list(ramp) == sorted(ramp)


def test_adjacent_window_weights_sum_to_one() -> None:
    windows = plan_h3_flow_retake_windows(660)
    previous, current = windows[0], windows[1]
    overlap = previous.end_frame_exclusive - current.start_frame
    for offset in range(overlap):
        previous_local = previous.source_frames - overlap + offset
        assert abs(
            window_temporal_weight(windows, 0, previous_local)
            + window_temporal_weight(windows, 1, offset)
            - 1.0
        ) < 1e-9
