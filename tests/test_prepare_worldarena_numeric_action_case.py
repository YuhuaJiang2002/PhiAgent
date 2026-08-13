from __future__ import annotations

import pytest

from phiagent.acwm.robotwin import BWM_EEF_CHANNELS
from phiagent.acwm.worldarena import WORLD_ARENA_EEF_QUATERNION_CHANNELS
from scripts.prepare_worldarena_numeric_action_case import (
    correct_worldarena_action_stats,
    quaternion_norm_bounds,
)


def _rows() -> list[list[float]]:
    row = [0.0] * 14
    row[6] = 1.0
    row[13] = 1.0
    return [row, row.copy()]


def test_detects_dual_unit_quaternion_pose() -> None:
    assert quaternion_norm_bounds(_rows()) == {
        "left": (1.0, 1.0),
        "right": (1.0, 1.0),
    }


def test_corrects_legacy_worldarena_channel_labels_without_changing_stats() -> None:
    payload = {
        "state_pose": {
            "coordinate_frame": "robot_base:test",
            "channels": list(BWM_EEF_CHANNELS),
            "min": [-1.0] * 14,
            "max": [1.0] * 14,
        }
    }

    corrected = correct_worldarena_action_stats(
        payload,
        quaternion_bounds=quaternion_norm_bounds(_rows()),
    )

    assert corrected["state_pose"]["channels"] == list(
        WORLD_ARENA_EEF_QUATERNION_CHANNELS
    )
    assert corrected["state_pose"]["min"] == payload["state_pose"]["min"]
    assert corrected["semantic_correction"]["source_channels"] == list(BWM_EEF_CHANNELS)


def test_rejects_non_unit_worldarena_orientation() -> None:
    rows = _rows()
    rows[0][13] = 0.5

    with pytest.raises(ValueError, match="right.*not unit quaternion"):
        correct_worldarena_action_stats(
            {
                "state_pose": {
                    "channels": list(BWM_EEF_CHANNELS),
                    "min": [-1.0] * 14,
                    "max": [1.0] * 14,
                }
            },
            quaternion_bounds=quaternion_norm_bounds(rows),
        )
