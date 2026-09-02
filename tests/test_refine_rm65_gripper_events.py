from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refine_rm65_gripper_events",
    ROOT / "scripts" / "refine_rm65_gripper_events.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_event_command_preserves_reviewed_keyframes_and_crossings() -> None:
    command = MODULE.event_command(
        12,
        np.asarray((0, 3, 7, 11)),
        np.asarray((0.0, 1.0, 1.0, 0.0)),
        0.8,
    )

    np.testing.assert_allclose(command[(0, 3, 7, 11),], (0.0, 1.0, 1.0, 0.0))
    assert (np.flatnonzero(np.diff(command >= 0.5)) + 1).tolist() == [2, 10]


def test_event_command_rejects_missing_final_frame() -> None:
    with pytest.raises(ValueError, match="first and final"):
        MODULE.event_command(12, np.asarray((0, 3, 7)), np.asarray((0, 1, 0)), 0.0)
