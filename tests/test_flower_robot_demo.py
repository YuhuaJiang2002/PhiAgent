from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_flower_robot_demo.py"
    spec = importlib.util.spec_from_file_location("build_flower_robot_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finger_bends_are_finite_and_bounded() -> None:
    module = _load_script()
    points = np.zeros((21, 3), dtype=float)
    for finger, indices in enumerate(
        ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    ):
        x = (finger - 2) * 0.02
        for step, index in enumerate(indices, 1):
            points[index] = (x, step * 0.03, 0.0)

    bends = module._finger_bends(np, points)

    assert set(bends) == {"thumb", "index", "middle", "ring", "pinky"}
    assert all(
        np.isfinite(value) and 0.0 <= value <= np.pi
        for values in bends.values()
        for value in values
    )


def test_gpu_selection_requires_requested_free_memory() -> None:
    module = _load_script()
    inventory = [
        {
            "physical_index": 7,
            "name": "A800",
            "memory_total_mib": 81920,
            "memory_used_mib": 1024,
            "memory_free_mib": 80896,
            "utilization_percent": 0,
        }
    ]

    assert module._select_gpu(inventory, 7, 1024) == inventory[0]
    with pytest.raises(RuntimeError, match="only 80896 MiB free"):
        module._select_gpu(inventory, 7, 80897)
    with pytest.raises(RuntimeError, match="not present"):
        module._select_gpu(inventory, 6, 1024)
