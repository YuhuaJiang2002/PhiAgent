from __future__ import annotations

from scripts.build_persistent_grasp_comparison import comparison_filter


def test_comparison_filter_keeps_named_left_right_and_header_layout() -> None:
    value = comparison_filter(panel_width=640, panel_height=360)

    assert "[0:v]scale=640:360" in value
    assert "[1:v]scale=640:360" in value
    assert "[left][right]hstack=inputs=2[body]" in value
    assert "[2:v][body]vstack=inputs=2[out]" in value
