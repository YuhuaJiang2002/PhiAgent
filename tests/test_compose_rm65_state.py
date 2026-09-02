from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compose_rm65_state", ROOT / "scripts" / "compose_rm65_state.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _state(path: Path, left_value: float, right_value: float, fps: float = 24.0) -> None:
    np.savez_compressed(
        path,
        left_q=np.full((5, 6), left_value, dtype=np.float64),
        right_q=np.full((5, 6), right_value, dtype=np.float64),
        left_target_xyz=np.full((5, 3), left_value, dtype=np.float64),
        right_target_xyz=np.full((5, 3), right_value, dtype=np.float64),
        fps=np.asarray(fps),
        gripper=np.arange(5),
    )


def test_compose_preserves_base_payload_and_selects_each_side(tmp_path: Path) -> None:
    base_path, left_path, right_path = (
        tmp_path / "base.npz",
        tmp_path / "left.npz",
        tmp_path / "right.npz",
    )
    _state(base_path, 1.0, 2.0)
    _state(left_path, 3.0, 4.0)
    _state(right_path, 5.0, 6.0)

    with np.load(base_path) as base, np.load(left_path) as left, np.load(right_path) as right:
        payload, manifest = MODULE.compose_states(base, left, right, 1, 1)

    np.testing.assert_allclose(payload["left_q"], 3.0)
    np.testing.assert_allclose(payload["right_q"], 6.0)
    np.testing.assert_array_equal(payload["gripper"], np.arange(5))
    assert manifest["frames"] == 5


def test_compose_rejects_fps_mismatch(tmp_path: Path) -> None:
    base_path, left_path, right_path = (
        tmp_path / "base.npz",
        tmp_path / "left.npz",
        tmp_path / "right.npz",
    )
    _state(base_path, 1.0, 2.0)
    _state(left_path, 3.0, 4.0, fps=30.0)
    _state(right_path, 5.0, 6.0)

    with np.load(base_path) as base, np.load(left_path) as left, np.load(right_path) as right:
        with pytest.raises(ValueError, match="FPS differs"):
            MODULE.compose_states(base, left, right, 1, 1)
