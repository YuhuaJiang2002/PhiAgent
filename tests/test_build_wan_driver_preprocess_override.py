from __future__ import annotations

import cv2
import numpy as np

from scripts.build_wan_driver_preprocess_override import _pose_capsule_mask


def test_pose_capsule_mask_tracks_robot_lines_and_blue_object_without_texture_noise() -> None:
    pose = np.zeros((64, 96, 3), dtype=np.uint8)
    cv2.line(pose, (18, 58), (48, 25), (0, 255, 0), 1)
    driver = np.zeros_like(pose)
    cv2.circle(driver, (54, 20), 6, (200, 80, 20), -1)
    cv2.rectangle(driver, (82, 2), (94, 12), (0, 0, 255), -1)

    mask = _pose_capsule_mask(
        cv2,
        np,
        pose,
        driver,
        pose_dilation_pixels=6,
        blue_dilation_pixels=3,
        minimum_component_area=20,
    )

    assert mask[45, 30] == 255
    assert mask[20, 54] == 255
    assert mask[6, 88] == 0
    assert 0.05 < float(np.mean(mask > 0)) < 0.45
