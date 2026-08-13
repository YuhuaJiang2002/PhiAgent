from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_hybrid_3d_flower_replacement.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_hybrid_3d_flower_replacement", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_largest_components_removes_small_noise() -> None:
    import cv2

    module = _module()
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[8:16, 8:16] = 1

    cleaned = module._largest_components(cv2, np, mask, minimum_area=10)

    assert not cleaned[1, 1]
    assert cleaned[10, 10] == 255


def test_recovered_mask_uses_difference_from_pinned_scene() -> None:
    import cv2

    module = _module()
    scene = np.full((30, 30, 3), 100, dtype=np.uint8)
    frame = scene.copy()
    frame[8:22, 10:20] = 180

    mask = module._recover_robot_mask(cv2, np, frame, scene, threshold=18)

    assert mask[15, 15] == 255
    assert mask[0, 0] == 0
