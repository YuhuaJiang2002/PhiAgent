from __future__ import annotations

from scripts.build_sharp_single_anchor_robot_replacement import (
    _build_fixed_anchor_plans,
    _compose_step,
    _hard_composite,
    _identity_map,
)


def test_zero_step_flow_preserves_identity_coordinates() -> None:
    import cv2
    import numpy as np

    gray = np.zeros((32, 48), dtype=np.uint8)
    map_x, map_y = _identity_map(np, 48, 32)

    result_x, result_y, maximum = _compose_step(
        cv2, np, gray, gray, map_x, map_y, 16.0, 96
    )

    assert maximum == 0.0
    assert np.allclose(result_x, map_x)
    assert np.allclose(result_y, map_y)


def test_hard_composite_never_passes_source_inside_safety() -> None:
    import cv2
    import numpy as np

    source = np.full((40, 60, 3), 220, dtype=np.uint8)
    robot = np.full((40, 60, 3), 35, dtype=np.uint8)
    dynamic = np.zeros((40, 60), dtype=np.uint8)
    safety = np.zeros((40, 60), dtype=np.uint8)
    safety[8:34, 18:48] = 255

    candidate, _, _ = _hard_composite(
        cv2, np, source, robot, dynamic, safety
    )

    assert np.all(candidate[safety > 0] == robot[safety > 0])
    assert np.all(candidate[0, 0] == source[0, 0])


def test_fixed_anchor_plan_uses_one_identity_for_every_frame() -> None:
    import cv2
    import numpy as np

    from scripts.build_multi_anchor_robot_replacement import Anchor

    grays = [np.zeros((18, 24), dtype=np.uint8) for _ in range(5)]
    anchors = (
        Anchor(0, None, None, None, grays[0]),
        Anchor(2, None, None, None, grays[2]),
        Anchor(4, None, None, None, grays[4]),
    )

    plans, maximum = _build_fixed_anchor_plans(
        cv2, np, grays, anchors, 1, 8.0, 48
    )

    assert maximum == 0.0
    assert len(plans) == 5
    assert {plan.anchor_index for plan in plans} == {1}
