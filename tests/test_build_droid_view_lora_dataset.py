from __future__ import annotations

import pytest

from scripts.build_droid_view_lora_dataset import (
    HOLDOUT_EPISODES,
    VALIDATION_EPISODES,
    choose_evenly,
    plan_clip_starts,
    split_episode_ids,
)


def test_choose_evenly_is_deterministic_and_spread() -> None:
    assert choose_evenly(list(range(10)), 4) == (0, 3, 6, 9)
    assert choose_evenly([2, 4, 6], 3) == (2, 4, 6)


def test_episode_splits_exclude_all_heldout_and_validation_ids() -> None:
    episodes = [{"episode_index": index} for index in range(100)]
    splits = split_episode_ids(episodes, 12)
    train = set(splits["train"])
    validation = set(splits["validation"])
    holdout = set(splits["holdout"])
    assert holdout == set(HOLDOUT_EPISODES)
    assert validation == set(VALIDATION_EPISODES)
    assert not train & validation
    assert not train & holdout
    assert not validation & holdout


def test_clip_starts_remain_inside_episode_and_are_reproducible() -> None:
    first = plan_clip_starts(10.0, 20.0, 3, episode_index=4, seed=42)
    second = plan_clip_starts(10.0, 20.0, 3, episode_index=4, seed=42)
    assert first == second
    assert first[0] == 10.0
    assert first[-1] + 17 / 8 + 1 / 15 <= 20.0 + 1e-9


def test_clip_planner_rejects_short_episode() -> None:
    with pytest.raises(ValueError, match="too short"):
        plan_clip_starts(2.0, 3.0, 1, episode_index=9, seed=42)
