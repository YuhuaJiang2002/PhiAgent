from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "extract_metric_stem_centerlines.py"
    spec = importlib.util.spec_from_file_location("extract_metric_stem_centerlines", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_centerline_skeleton_is_ordered_from_lower_root_to_tip() -> None:
    module = _module()
    mask = np.zeros((80, 60), dtype=np.uint8)
    points = np.asarray([[30, 72], [29, 55], [25, 38], [18, 20], [12, 8]], np.int32)
    cv2.polylines(mask, [points], False, 255, 5, cv2.LINE_8)
    centerline = module.extract_centerline_pixels(cv2, np, mask > 0, 9)
    assert centerline.shape == (9, 2)
    assert centerline[0, 1] > centerline[-1, 1]
    assert np.all(np.linalg.norm(np.diff(centerline, axis=0), axis=1) > 0)


def test_centerline_skeleton_rejects_insufficient_support() -> None:
    module = _module()
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    try:
        module.extract_centerline_pixels(cv2, np, mask, 4)
    except ValueError as exc:
        assert "fewer pixels" in str(exc)
    else:
        raise AssertionError("insufficient skeleton support was accepted")
