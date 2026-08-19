from __future__ import annotations

import math
from pathlib import Path

from phiagent.acwm.schema import ACWMActionCondition
from phiagent.world_model.joyai_sc3 import load_config
from scripts.build_joyai_sc3_intent_contrast import _intent_geometry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_contrast_pair_has_same_start_and_distinct_terminal_targets() -> None:
    carry = ACWMActionCondition.from_json(
        PROJECT_ROOT / "demo" / "showcase" / "oscar-acwm-carry-right-action.json"
    )
    lift = ACWMActionCondition.from_json(
        PROJECT_ROOT / "demo" / "showcase" / "oscar-acwm-lift-up-action.json"
    )
    x = carry.channels.index("object_center_x_px")
    y = carry.channels.index("object_center_y_px")
    carry_start = carry.values[0][x], carry.values[0][y]
    lift_start = lift.values[0][x], lift.values[0][y]
    carry_end = carry.values[-1][x], carry.values[-1][y]
    lift_end = lift.values[-1][x], lift.values[-1][y]

    assert carry.coordinate_frame == lift.coordinate_frame
    assert carry_start == lift_start
    assert math.dist(carry_end, lift_end) == 175.0
    assert carry_end[0] - carry_start[0] > 150
    assert abs(lift_end[0] - lift_start[0]) < 5


def test_contrast_configs_share_real_first_frame_but_not_action_carrier() -> None:
    carry = load_config(
        PROJECT_ROOT / "configs" / "joyai" / "sc3_oscar_carry_right_best_of_4_v1.json"
    )
    lift = load_config(
        PROJECT_ROOT / "configs" / "joyai" / "sc3_oscar_lift_up_best_of_4_v1.json"
    )

    assert carry.first_frame == lift.first_frame
    assert carry.source_video == lift.source_video
    assert carry.carrier.video != lift.carrier.video
    assert carry.action_condition != lift.action_condition
    assert carry.candidate_seeds == lift.candidate_seeds


def test_generated_endpoints_remain_distinct_in_selected_runs() -> None:
    carry = ACWMActionCondition.from_json(
        PROJECT_ROOT / "demo" / "showcase" / "oscar-acwm-carry-right-action.json"
    )
    lift = ACWMActionCondition.from_json(
        PROJECT_ROOT / "demo" / "showcase" / "oscar-acwm-lift-up-action.json"
    )
    carry_evaluation = {
        "metrics": {"observed_terminal_xy": [449.213767694396, 76.19790535298681]}
    }
    lift_evaluation = {
        "metrics": {"observed_terminal_xy": [256.624908300886, 92.62499294781382]}
    }

    geometry = _intent_geometry(lift, carry, lift_evaluation, carry_evaluation)

    assert geometry["expected_terminal_separation_px"] == 175.0
    assert geometry["observed_terminal_separation_px"] > 190
