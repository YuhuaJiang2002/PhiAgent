from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "render_droid_depth_warp_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("render_droid_depth_warp_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nearest_timestamp_index_is_deterministic() -> None:
    assert _module().nearest_timestamp_index([100, 110, 130], 116) == 1
    assert _module().nearest_timestamp_index([100, 110, 130], 120) == 1


def test_nearest_timestamp_index_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        _module().nearest_timestamp_index([], 0)


def test_rotation_xyz_matches_official_scipy_convention() -> None:
    actual = _module()._rotation_xyz(np, [0.3, 0.2, 0.1])
    expected = np.asarray(
        [
            [0.9751703272, -0.0369570135, 0.2183506631],
            [0.0978433950, 0.9564250858, -0.2750958473],
            [-0.1986693308, 0.2896294776, 0.9362933636],
        ]
    )

    assert actual == pytest.approx(expected)
