from __future__ import annotations

import numpy as np
import pytest

from scripts.prepare_worldarena2_bwm_factory_data import (
    _cached_task_episodes,
    _stats,
    motion_windows,
    terminal_video_frames_excluded,
)
from phiagent.acwm.worldarena import WORLD_ARENA_EEF_QUATERNION_CHANNELS


def test_motion_windows_selects_separated_high_motion_regions() -> None:
    values = np.zeros((200, 14), dtype=np.float64)
    values[20:60, 0] = np.arange(40)
    values[120:160, 7] = np.arange(40) * 2

    starts = motion_windows(values, num_frames=57, count=2, stride=1)

    assert len(starts) == 2
    assert starts[1] - starts[0] >= 57
    assert starts[0] <= 59
    assert starts[1] >= 80


def test_motion_windows_is_deterministic_on_ties() -> None:
    values = np.zeros((130, 14), dtype=np.float64)

    assert motion_windows(values, num_frames=57, count=2, stride=1) == (0, 57)


def test_motion_windows_rejects_implicit_wrong_action_width() -> None:
    with pytest.raises(ValueError, match="N x 14"):
        motion_windows(np.zeros((100, 7)), num_frames=57, count=1)


def test_cached_task_episodes_uses_numeric_order(tmp_path) -> None:
    task = tmp_path / "clean_table"
    for name in ("episode_10", "episode_2", "episode_1"):
        (task / name).mkdir(parents=True)

    assert _cached_task_episodes(tmp_path, "clean_table", 2) == (
        "episode_1",
        "episode_2",
    )


def test_worldarena_alignment_accepts_one_terminal_video_sentinel() -> None:
    assert terminal_video_frames_excluded(100, 100) == 0
    assert terminal_video_frames_excluded(101, 100) == 1
    with pytest.raises(ValueError, match="one extra terminal"):
        terminal_video_frames_excluded(102, 100)


def test_worldarena_statistics_name_quaternion_pose_channels() -> None:
    values = np.zeros((4, 14), dtype=np.float64)
    values[:, 6] = 1.0
    values[:, 13] = 1.0

    stats = _stats([values], np)["state_pose"]

    assert stats["channels"] == list(WORLD_ARENA_EEF_QUATERNION_CHANNELS)
