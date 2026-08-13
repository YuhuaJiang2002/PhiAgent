from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_robotwin_reset_preflight.py"
    )
    spec = importlib.util.spec_from_file_location("run_robotwin_reset_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reset_max_difference_is_strict_and_shape_safe() -> None:
    module = _module()

    assert module._max_difference([1.0, 2.0], [1.0, 2.000001]) == pytest.approx(
        0.000001
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        module._max_difference([1.0], [1.0, 2.0])
