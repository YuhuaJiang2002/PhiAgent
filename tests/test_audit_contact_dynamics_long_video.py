from __future__ import annotations

import numpy as np

from scripts.audit_contact_dynamics_long_video import _adversarial_audit


class Args:
    maximum_response_lag_frames = 1
    maximum_frozen_run_frames = 0
    target_frame_name = "camera:test"
    expected_frames = 6
    fps = 24.0


def test_adversarial_audit_detects_response_depth_and_topology_attacks() -> None:
    grasp = np.asarray([False, True, True, True, True, False])
    hand = np.asarray([0.0, 2.0, 2.0, 2.0, 2.0, 0.0])
    stem = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    result = _adversarial_audit(
        np,
        grasp,
        hand,
        stem,
        {"hand_motion": 1.0, "stem_motion": 0.5},
        Args(),
    )
    assert result["all_attacks_detected"] is True
    assert all(result["gates"].values())
