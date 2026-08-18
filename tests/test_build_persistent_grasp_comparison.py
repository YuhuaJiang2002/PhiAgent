from __future__ import annotations

import cv2

from scripts.build_persistent_grasp_comparison import _render_header, comparison_filter


def test_comparison_filter_keeps_named_left_right_and_header_layout() -> None:
    value = comparison_filter(panel_width=640, panel_height=360)

    assert "[0:v]scale=640:360" in value
    assert "[1:v]scale=640:360" in value
    assert "[left][right]hstack=inputs=2[body]" in value
    assert "[2:v][body]vstack=inputs=2[out]" in value


def test_render_header_without_mandatory_pillow(tmp_path) -> None:
    path = tmp_path / "header.png"

    _render_header(path, width=1280, height=48)

    image = cv2.imread(str(path))
    assert image is not None
    assert image.shape == (48, 1280, 3)
