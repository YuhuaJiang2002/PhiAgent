from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "refine_rm65_intermediate_link",
    SCRIPTS / "refine_rm65_intermediate_link.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_angle_deg_supports_undirected_parallel_jaw_axis() -> None:
    first = np.asarray((1.0, 0.0))
    second = np.asarray((-1.0, 0.0))

    assert MODULE._angle_deg(first, second) == pytest.approx(180.0)
    assert MODULE._angle_deg(first, second, undirected=True) == pytest.approx(0.0)


def test_unit_rejects_degenerate_projected_axis() -> None:
    with pytest.raises(ValueError, match="near-zero"):
        MODULE._unit(np.zeros(2))
