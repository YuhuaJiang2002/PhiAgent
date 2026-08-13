from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_hybrid_flower_contact.py"


def _module():
    spec = importlib.util.spec_from_file_location("evaluate_hybrid_flower_contact", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fraction_is_a_hard_threshold_fraction() -> None:
    assert _module()._fraction(np.asarray([1.0, 2.0, 4.0]), 2.0) == pytest.approx(2 / 3)


def test_fraction_rejects_missing_evidence() -> None:
    with pytest.raises(ValueError, match="empty"):
        _module()._fraction(np.asarray([]), 1.0)
