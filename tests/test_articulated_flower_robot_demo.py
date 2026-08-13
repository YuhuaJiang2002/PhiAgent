from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_script():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "build_articulated_flower_robot_demo.py"
    spec = importlib.util.spec_from_file_location(
        "build_articulated_flower_robot_demo", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pose_targets_are_smoothed_and_frame_explicit() -> None:
    module = _load_script()
    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(33)]
    landmarks[15] = SimpleNamespace(x=0.70, y=0.60)
    landmarks[16] = SimpleNamespace(x=0.56, y=0.58)

    raw = module._pose_targets(np, landmarks, None)
    previous = {
        "left": np.asarray((0.25, 0.0, 0.8)),
        "right": np.asarray((0.25, 0.0, 0.8)),
    }
    smoothed = module._pose_targets(np, landmarks, previous)

    assert set(raw) == {"left", "right"}
    assert raw["left"].shape == (3,)
    assert np.linalg.norm(smoothed["left"] - previous["left"]) < np.linalg.norm(
        raw["left"] - previous["left"]
    )


def test_default_hand_geometry_has_valid_finger_segments() -> None:
    module = _load_script()

    points = module._default_hand_points(np)

    assert points.shape == (21, 3)
    assert np.linalg.norm(points[12] - points[9]) > 0


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_pose_targets_carry_only_invalid_observations(bad_value: float) -> None:
    module = _load_script()
    landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=1.0) for _ in range(33)]
    landmarks[15] = SimpleNamespace(x=bad_value, y=0.5, visibility=1.0)
    previous = {
        "left": np.asarray((0.30, 0.1, 0.9)),
        "right": np.asarray((0.30, -0.1, 0.9)),
    }

    targets = module._pose_targets(np, landmarks, previous)

    np.testing.assert_array_equal(targets["left"], previous["left"])
    assert np.all(np.isfinite(targets["right"]))


def test_pose_target_stress_clamps_outliers_and_remains_finite() -> None:
    module = _load_script()
    rng = np.random.default_rng(20260809)
    previous = None

    for _ in range(5_000):
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(33)]
        for index in (15, 16):
            landmarks[index] = SimpleNamespace(
                x=float(rng.uniform(-1e6, 1e6)),
                y=float(rng.uniform(-1e6, 1e6)),
                visibility=1.0,
            )
        targets = module._pose_targets(np, landmarks, previous)
        for target in targets.values():
            assert np.all(np.isfinite(target))
            assert target[0] == pytest.approx(0.30)
            assert -0.75 <= target[1] <= 0.50
            assert 0.57 <= target[2] <= 1.25
        previous = targets


def test_hand_point_validation_rejects_corrupt_and_degenerate_observations() -> None:
    module = _load_script()
    valid = module._default_hand_points(np)

    assert module._validated_hand_points(np, valid) is not None
    for corrupt in (
        valid[:20],
        np.full((21, 3), np.nan),
        np.full((21, 3), np.inf),
        np.zeros((21, 3)),
        valid * 1_000,
    ):
        assert module._validated_hand_points(np, corrupt) is None


def test_hand_point_stress_accepts_small_tracking_noise() -> None:
    module = _load_script()
    rng = np.random.default_rng(20260809)
    baseline = module._default_hand_points(np)

    for _ in range(5_000):
        noisy = baseline + rng.normal(0.0, 1e-4, baseline.shape)
        validated = module._validated_hand_points(np, noisy)
        assert validated is not None
        assert np.all(np.isfinite(validated))
