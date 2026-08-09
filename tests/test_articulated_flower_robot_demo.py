from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


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
