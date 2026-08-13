from __future__ import annotations

import pytest

from phiagent.view_generation.readiness import (
    audit_droid_novel_view_readiness,
    extrinsic_variation,
)


def _info() -> dict:
    video = {"dtype": "video"}
    vector = {"dtype": "float32", "names": {"motors": ["eef_x", "eef_y"]}}
    return {
        "features": {
            "observation.images.wrist_image_left": video,
            "observation.images.exterior_image_1_left": video,
            "observation.state": vector,
            "action": vector,
            "timestamp": {"dtype": "float32"},
            "episode_index": {"dtype": "int64"},
        }
    }


def _contract() -> dict:
    return {
        "rights_reviewed": True,
        "timestamp_alignment_verified": True,
        "depth_lineage_verified": True,
        "cameras": {
            "observation.images.wrist_image_left": {
                "coordinate_frame": "camera:wrist",
                "intrinsics": {"fx": 1.0},
                "extrinsics": {"mode": "per_frame"},
            },
            "observation.images.exterior_image_1_left": {
                "coordinate_frame": "camera:exterior-a",
                "intrinsics": {"fx": 1.0},
                "extrinsics": {"mode": "static"},
            },
        },
    }


def test_readiness_requires_raw_calibration_and_rights() -> None:
    result = audit_droid_novel_view_readiness(_info(), None)

    assert result["ready"] is False
    assert result["status"] == "BLOCKED"
    assert "time-varying world_T_wrist_camera" in result["missing_requirements"]


def test_readiness_accepts_complete_frame_explicit_contract() -> None:
    result = audit_droid_novel_view_readiness(_info(), _contract())

    assert result["ready"] is True
    assert result["missing_requirements"] == []


def test_extrinsic_variation_separates_translation_and_rotation() -> None:
    result = extrinsic_variation(
        [
            [0.0, 0.1, 0.2, 0.0, 0.0, 0.0],
            [0.03, 0.1, 0.2, 0.0, 0.2, 0.0],
        ]
    )

    assert result["translation_max_range_m"] == pytest.approx(0.03)
    assert result["euler_max_range_rad"] == pytest.approx(0.2)
