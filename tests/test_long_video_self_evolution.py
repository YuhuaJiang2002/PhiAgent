from __future__ import annotations

import pytest

from phiagent.agent.long_video_self_evolution import (
    FrozenVisualContract,
    evaluate_specialty_contract,
    evaluate_visual_iteration,
)
from scripts.audit_robot_layer_long_video import _resolve_frame_masks


def test_tracked_flower_front_layer_excludes_hand_and_face_skin() -> None:
    np = pytest.importorskip("numpy")
    source = np.zeros((40, 40, 3), dtype=np.uint8)
    person = np.ones((40, 40), dtype=bool)
    tracked = np.zeros((40, 40), dtype=bool)
    tracked[25, 18:23] = True
    tracked[0, 0] = True
    tracked[18, 35] = True
    tracked[30, 35] = True
    source[25, 18] = (25, 160, 40)
    source[25, 22] = (25, 160, 40)
    source[18, 35] = (25, 160, 40)
    source[30, 35] = (25, 160, 40)
    hands = np.zeros((40, 40), dtype=bool)
    hands[25, 20] = True

    _, flower = _resolve_frame_masks(
        np,
        source_rgb=source,
        person=person,
        tracked_flower=tracked,
        hands=hands,
        person_dilation=0,
        skin_negative_dilation=0,
        person_core_negative_erosion=0,
        flower_mask_contract="tracked_front_layer_with_human_negatives",
    )

    assert flower[25, 18]
    assert not flower[25, 20]
    assert flower[25, 22]
    assert not flower[0, 0]
    assert flower[18, 35]


def test_tracked_flower_front_layer_rejects_neutral_person_track_blob() -> None:
    np = pytest.importorskip("numpy")
    source = np.full((20, 20, 3), 128, dtype=np.uint8)
    person = np.ones((20, 20), dtype=bool)
    tracked = np.zeros((20, 20), dtype=bool)
    tracked[8:13, 8:13] = True
    hands = np.zeros_like(tracked)

    _, flower = _resolve_frame_masks(
        np,
        source_rgb=source,
        person=person,
        tracked_flower=tracked,
        hands=hands,
        person_dilation=0,
        skin_negative_dilation=0,
        person_core_negative_erosion=0,
        flower_mask_contract="tracked_front_layer_with_human_negatives",
    )

    assert not np.any(flower)


def test_foundation_refined_front_layer_only_removes_source_hand() -> None:
    np = pytest.importorskip("numpy")
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    person = np.ones((8, 8), dtype=bool)
    tracked = np.zeros((8, 8), dtype=bool)
    tracked[6:8, 3:6] = True
    hands = np.zeros_like(tracked)
    hands[6, 4] = True

    _, flower = _resolve_frame_masks(
        np,
        source_rgb=source,
        person=person,
        tracked_flower=tracked,
        hands=hands,
        person_dilation=0,
        skin_negative_dilation=0,
        person_core_negative_erosion=0,
        flower_mask_contract="foundation_refined_front_layer",
    )

    assert flower[6, 3]
    assert not flower[6, 4]
    assert flower[7, 5]


def test_foundation_refined_front_layer_rejects_isolated_face_blob() -> None:
    np = pytest.importorskip("numpy")
    source = np.zeros((40, 40, 3), dtype=np.uint8)
    person = np.ones((40, 40), dtype=bool)
    tracked = np.zeros((40, 40), dtype=bool)
    tracked[2, 2] = True
    tracked[25, 20:24] = True
    hands = np.zeros_like(tracked)

    _, flower = _resolve_frame_masks(
        np,
        source_rgb=source,
        person=person,
        tracked_flower=tracked,
        hands=hands,
        person_dilation=0,
        skin_negative_dilation=0,
        person_core_negative_erosion=0,
        flower_mask_contract="foundation_refined_front_layer",
    )

    assert not flower[2, 2]
    assert flower[25, 20]


def _contract() -> FrozenVisualContract:
    return FrozenVisualContract(
        maximum_stage_wall_seconds=10.0,
        maximum_self_flow_mean=2.0,
        maximum_self_flow_p95=4.0,
        maximum_self_flow_high_count=3,
        maximum_source_flow_mean=3.0,
        maximum_source_flow_p95=6.0,
        maximum_source_flow_high_count=8,
        maximum_wrong_occlusion_mean=0.01,
        maximum_wrong_occlusion_p95=0.0,
        maximum_owner_flip_mean=0.02,
        maximum_owner_flip_p95=0.0,
    )


def _payloads() -> tuple[dict, dict, dict]:
    repair = {"gates": {"decode": True}, "metrics": {"wall_seconds": 5.0}}
    specialty = {
        "metrics": {
            "challenger": {
                "self_flow_arm_mae": {"mean": 1.5, "p95": 3.0},
                "source_flow_residual_mae": {"mean": 2.5, "p95": 5.0},
                "wrong_flower_occlusion_fraction": {"mean": 0.0, "p95": 0.0},
                "flower_owner_flip_fraction": {"mean": 0.0, "p95": 0.0},
            },
            "high_flicker_counts": {
                "challenger": {"self_flow_arm_mae": 2, "source_flow_residual_mae": 6}
            },
        }
    }
    full = {
        "candidates": [
            {
                "summary": {"gates": {"late_chroma": True, "contact": True}},
                "adversarial": {"gates": {"colour_attack": True, "detach": True}},
            }
        ]
    }
    return repair, specialty, full


def test_all_frozen_conditions_only_request_human_review() -> None:
    repair, specialty, full = _payloads()
    result = evaluate_visual_iteration(
        repair_manifest=repair,
        specialty_report=specialty,
        full_report=full,
        contract=_contract(),
    )
    assert result["automatic_pass"] is True
    assert result["status"] == "AWAITING_HIGH_RESOLUTION_REVIEW"
    assert result["physical_evidence"] is False


def test_one_failed_gate_cannot_be_hidden_by_other_metrics() -> None:
    repair, specialty, full = _payloads()
    full["candidates"][0]["summary"]["gates"]["contact"] = False
    result = evaluate_visual_iteration(
        repair_manifest=repair,
        specialty_report=specialty,
        full_report=full,
        contract=_contract(),
    )
    assert result["automatic_pass"] is False
    assert result["failed_checks"] == ["all_full_video_gates"]


def test_specialty_failure_can_reject_before_full_audit() -> None:
    repair, specialty, _ = _payloads()
    specialty["metrics"]["challenger"]["wrong_flower_occlusion_fraction"][
        "mean"
    ] = 0.5
    result = evaluate_specialty_contract(
        repair_manifest=repair,
        specialty_report=specialty,
        contract=_contract(),
    )

    assert result["automatic_pass"] is False
    assert result["failed_checks"] == ["wrong_occlusion_mean_non_regression"]
