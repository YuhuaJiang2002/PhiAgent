from __future__ import annotations

import pytest

from scripts.retime_wan_raw_window import build_filter


def test_build_filter_preserves_raw_frames_and_clones_terminal_frame() -> None:
    assert build_filter(30.0, 24.0) == (
        "setpts=1.25*PTS,tpad=stop_mode=clone:stop_duration=0.0416666666667,fps=24"
    )


def test_build_filter_rejects_nonpositive_fps() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_filter(30.0, 0.0)
