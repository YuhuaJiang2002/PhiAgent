from __future__ import annotations

import numpy as np
import pytest

from scripts.build_acwm_bowl_action_controls import (
    CAMERA_PIXEL_FRAME,
    action_progress,
    default_bowl_action_plans,
    similarity_affine,
)


def test_bowl_action_endpoints_are_mutually_exclusive() -> None:
    plans = {plan.label: plan for plan in default_bowl_action_plans()}

    assert set(plans) == {"slide-left", "slide-right", "lift-up"}
    assert plans["slide-right"].target_center_x - plans["slide-left"].target_center_x >= 400
    assert plans["slide-left"].target_center_y - plans["lift-up"].target_center_y >= 180
    assert all(plan.coordinate_frame == CAMERA_PIXEL_FRAME for plan in plans.values())


def test_action_progress_has_common_start_and_terminal_hold() -> None:
    assert action_progress(0, 124) == 0.0
    assert action_progress(20, 124) == 0.0
    assert action_progress(100, 124) == 1.0
    assert action_progress(123, 124) == 1.0


def test_similarity_affine_keeps_arm_base_and_maps_contact() -> None:
    anchor = np.asarray((8.0, 5.0, 1.0))
    source_contact = np.asarray((4.0, 2.0, 1.0))
    target_contact = np.asarray((2.0, 4.0))
    transform = similarity_affine(
        np, tuple(anchor[:2]), tuple(source_contact[:2]), tuple(target_contact)
    )

    assert transform @ anchor == pytest.approx(anchor[:2])
    assert transform @ source_contact == pytest.approx(target_contact)
