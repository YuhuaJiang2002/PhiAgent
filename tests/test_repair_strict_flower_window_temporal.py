from __future__ import annotations

import numpy as np

from scripts.repair_strict_flower_window_temporal import (
    _fit_mask_frames,
    apply_local_crossfade_pass,
    build_protection_masks,
    repair_eligibility_mode,
)


class FakeCV2:
    @staticmethod
    def dilate(mask: np.ndarray, _kernel: np.ndarray) -> np.ndarray:
        # The zero-dilation branch is used by the protection test.
        return mask


def test_fit_mask_frames_duplicates_only_missing_terminal_frame() -> None:
    masks = np.zeros((2, 3, 4), dtype=bool)
    masks[-1, 1, 2] = True
    fitted = _fit_mask_frames(np, masks, 3)
    assert fitted.shape == (3, 3, 4)
    assert np.array_equal(fitted[-1], fitted[-2])


def test_protection_includes_flowers_and_contact_intersection() -> None:
    flowers = np.zeros((2, 4, 5), dtype=bool)
    hands = np.zeros_like(flowers)
    flowers[:, 1, 1] = True
    hands[:, 1, 1:3] = True
    protected, contact = build_protection_masks(
        FakeCV2,
        np,
        flowers,
        hands,
        flower_dilation=0,
        contact_band=0,
    )
    assert np.array_equal(contact, flowers & hands)
    assert np.array_equal(protected, flowers)


def test_crossfade_keeps_protected_pixels_exact() -> None:
    frames = [
        np.full((2, 2, 3), value, dtype=np.uint8) for value in (0, 100, 200)
    ]
    alpha = np.ones((3, 2, 2), dtype=np.float32)
    protected = np.zeros((3, 2, 2), dtype=bool)
    protected[1, 0, 0] = True
    result = apply_local_crossfade_pass(
        np, frames, alpha, protected, strength=1.0
    )
    assert np.array_equal(result[1][0, 0], frames[1][0, 0])
    assert np.array_equal(result[0], frames[0])
    assert np.array_equal(result[-1], frames[-1])


def test_motion_recovery_requires_only_aligned_proxy_failures() -> None:
    report = {
        "geometry_all_gates_passed": False,
        "gates": {
            "flower:visible": True,
            "hand:all_frames": True,
            "semantic:dense_review": True,
            "aligned:identity": True,
            "aligned:motion": False,
            "aligned:temporal": False,
        },
    }
    assert repair_eligibility_mode(
        report,
        baseline_motion=0.79,
        allow_aligned_motion_recovery=True,
        minimum_recovery_baseline_motion=0.75,
    ) == "bounded-aligned-motion-and-temporal-recovery"


def test_motion_recovery_rejects_any_instance_gate_failure() -> None:
    report = {
        "geometry_all_gates_passed": False,
        "gates": {
            "hand:all_frames": False,
            "aligned:motion": False,
            "aligned:temporal": False,
        },
    }
    try:
        repair_eligibility_mode(
            report,
            baseline_motion=0.79,
            allow_aligned_motion_recovery=True,
            minimum_recovery_baseline_motion=0.75,
        )
    except RuntimeError as error:
        assert "every non-motion/non-temporal strict gate" in str(error)
    else:
        raise AssertionError("instance-gate failure must not enter motion recovery")
