from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_contact_calibrated_3d_robot.py"


def _module():
    spec = importlib.util.spec_from_file_location("render_contact_calibrated_3d_robot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Wrist:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.visibility = 1.0


def test_pose_targets_are_symmetric_about_named_camera_center() -> None:
    module = _module()
    landmarks = [_Wrist(0.0, 0.0) for _ in range(17)]
    landmarks[15] = _Wrist(0.76, 0.5)
    landmarks[16] = _Wrist(0.56, 0.5)

    targets = module._pose_targets(
        np,
        landmarks,
        None,
        center_x=0.66,
        horizontal_gain=2.0,
        vertical_origin=1.45,
        vertical_gain=0.95,
        observation_weight=1.0,
    )

    assert targets["left"][1] == pytest.approx(0.2)
    assert targets["right"][1] == pytest.approx(-0.2)
    assert targets["left"][2] == targets["right"][2]


def test_pose_target_smoothing_retains_more_than_old_twelve_percent() -> None:
    module = _module()
    landmarks = [_Wrist(0.0, 0.0) for _ in range(17)]
    landmarks[15] = _Wrist(0.76, 0.5)
    landmarks[16] = _Wrist(0.56, 0.5)
    previous = {
        "left": np.asarray((0.3, 0.0, 0.9)),
        "right": np.asarray((0.3, 0.0, 0.9)),
    }

    targets = module._pose_targets(
        np,
        landmarks,
        previous,
        center_x=0.66,
        horizontal_gain=2.0,
        vertical_origin=1.45,
        vertical_gain=0.95,
        observation_weight=0.35,
    )

    assert targets["left"][1] == pytest.approx(0.07)
