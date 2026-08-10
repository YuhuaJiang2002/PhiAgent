from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from scripts.repair_acwm_hand_structure import fixed_scale_transform


def _apply(matrix, point):
    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2],
        matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2],
    )


def test_fixed_scale_transform_maps_contact_without_scale_breathing() -> None:
    reference_contact = (80.0, 75.0)
    reference_elbow = (130.0, 110.0)
    target_contact = (170.0, 120.0)
    target_elbow = (190.0, 180.0)

    matrix = fixed_scale_transform(
        reference_contact,
        reference_elbow,
        target_contact,
        target_elbow,
        scale=1.0,
    )

    assert _apply(matrix, reference_contact) == pytest.approx(target_contact)
    unit_x = _apply(matrix, (reference_contact[0] + 1, reference_contact[1]))
    assert math.dist(unit_x, target_contact) == pytest.approx(1.0)


def test_rigid_hand_projection_preserves_one_component_and_area() -> None:
    canonical = np.zeros((240, 320), dtype=np.uint8)
    cv2.circle(canonical, (120, 110), 28, 255, cv2.FILLED)
    for offset in (-20, -10, 0, 10, 20):
        cv2.line(canonical, (105 + offset, 100), (85 + offset, 60), 255, 7)
    assert cv2.connectedComponents(canonical)[0] - 1 == 1
    source_area = int(np.count_nonzero(canonical))
    areas = []
    for target in ((140.0, 120.0), (180.0, 130.0), (220.0, 115.0)):
        matrix = np.asarray(
            fixed_scale_transform(
                (120.0, 110.0),
                (160.0, 160.0),
                target,
                (target[0] + 45.0, target[1] + 35.0),
                scale=1.0,
            ),
            dtype=np.float32,
        )
        projected = cv2.warpAffine(
            canonical,
            matrix,
            (320, 240),
            flags=cv2.INTER_NEAREST,
        )
        assert cv2.connectedComponents(projected)[0] - 1 == 1
        areas.append(int(np.count_nonzero(projected)))

    assert max(areas) / min(areas) <= 1.02
    assert np.mean(areas) == pytest.approx(source_area, rel=0.02)
