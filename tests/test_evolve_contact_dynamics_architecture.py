from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evolve_contact_dynamics_architecture.py"
SPEC = importlib.util.spec_from_file_location("evolve_contact_dynamics_architecture", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_visual_group_fails_on_frozen_contact_response() -> None:
    rows = [
        {
            "frame": frame,
            "measurement_valid": True,
            "projected_contact": True,
            "hand_motion_p90": 2.0,
            "flower_motion_p90": 0.0 if 2 <= frame <= 4 else 1.0,
        }
        for frame in range(6)
    ]
    result = MODULE.assess_visual_group(
        rows,
        hand_motion_floor=0.5,
        stem_motion_floor=0.25,
        maximum_response_lag_frames=0,
        maximum_frozen_run_frames=2,
    )
    assert result["passed"] is False
    assert result["maximum_frozen_run_frames"] == 3


def test_visual_group_passes_only_when_every_driven_frame_responds() -> None:
    rows = [
        {
            "frame": frame,
            "measurement_valid": True,
            "projected_contact": True,
            "hand_motion_p90": 2.0,
            "flower_motion_p90": 1.0,
        }
        for frame in range(6)
    ]
    result = MODULE.assess_visual_group(
        rows,
        hand_motion_floor=0.5,
        stem_motion_floor=0.25,
        maximum_response_lag_frames=2,
        maximum_frozen_run_frames=2,
    )
    assert result["passed"] is True
    assert result["frozen_driver_frames"] == 0
