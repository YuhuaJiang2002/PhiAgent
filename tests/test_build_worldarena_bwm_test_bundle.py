from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_worldarena_bwm_test_bundle.py"
    )
    spec = importlib.util.spec_from_file_location("build_worldarena_bwm_test_bundle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_rewrites_video_but_preserves_action_window() -> None:
    row = {
        "length": 57,
        "start_frame": 160,
        "end_frame": 216,
        "source_episode": "wipe_table/episode_0",
        "video": {"start_frame": 160, "end_frame": 216},
        "action": {"start_frame": 160, "end_frame": 216},
    }
    meta = {
        "source_episode": "wipe_table/episode_0",
        "source_start_frame": 160,
        "source_end_frame": 216,
    }

    result = _module().bundled_row(row, meta)

    assert result["video"]["start_frame"] == 0
    assert result["video"]["end_frame"] == 56
    assert result["action"] == row["action"]
    assert result["source_video_window"] == {
        "start_frame": 160,
        "end_frame": 216,
    }
