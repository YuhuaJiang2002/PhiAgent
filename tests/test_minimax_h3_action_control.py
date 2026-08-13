from scripts.build_minimax_h3_action_control_videos import (
    CAMERA_PIXEL_FRAME,
    _timeline_from_manifest,
    default_action_control_plans,
    interpolate_waypoints,
)


def test_action_control_plans_are_explicit_and_visibly_separated() -> None:
    plans = {plan.label: plan for plan in default_action_control_plans()}

    assert set(plans) == {"insert-flower", "handover-flower", "inspect-flower"}
    insert_mid = interpolate_waypoints(plans["insert-flower"].right, 0.62)
    handover_mid = interpolate_waypoints(plans["handover-flower"].right, 0.62)
    inspect_mid = interpolate_waypoints(plans["inspect-flower"].right, 0.62)
    assert insert_mid.frame == CAMERA_PIXEL_FRAME
    assert handover_mid.wrist_x - insert_mid.wrist_x > 100
    assert insert_mid.wrist_y - inspect_mid.wrist_y > 100
    assert inspect_mid.hand_rotation_degrees == 45


def test_handover_switches_object_ownership() -> None:
    handover = next(
        plan for plan in default_action_control_plans() if plan.label == "handover-flower"
    )
    assert handover.object_transfer_progress == 0.60
    assert interpolate_waypoints(handover.right, 0.46).grasp
    assert interpolate_waypoints(handover.left, 0.72).grasp


def test_long_phase_manifest_compiles_to_control_timeline() -> None:
    timeline = _timeline_from_manifest(
        {
            "phases": [
                {
                    "start_s": 0.0,
                    "end_s": 1.5,
                    "description": "approach the flower",
                },
                {
                    "start_s": 1.5,
                    "end_s": 2.5,
                    "description": "grasp the stem",
                },
            ]
        }
    )

    assert timeline == (
        "0.000-1.500 s: approach the flower; 1.500-2.500 s: grasp the stem"
    )
