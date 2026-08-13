from __future__ import annotations

import pytest

from scripts.build_droid_cosmos_multiview_dataset import (
    FINAL_HOLDOUT_EPISODES,
    LEGACY_DEV_EPISODES,
    VALIDATION_EPISODES,
    plan_clip_starts,
    split_episode_ids,
)


def test_four_way_split_is_disjoint_and_final_holdout_is_clean() -> None:
    splits = split_episode_ids([{"episode_index": index} for index in range(100)])
    assert set(splits["legacy_dev"]) == set(LEGACY_DEV_EPISODES)
    assert set(splits["validation"]) == set(VALIDATION_EPISODES)
    assert set(splits["final_holdout"]) == set(FINAL_HOLDOUT_EPISODES)
    all_ids = [value for values in splits.values() for value in values]
    assert len(all_ids) == len(set(all_ids)) == 100
    assert not set(splits["train"]) & set(FINAL_HOLDOUT_EPISODES)


def test_split_rejects_overlapping_reserved_sets() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        split_episode_ids(
            [{"episode_index": index} for index in range(10)],
            legacy_dev=(1,),
            validation=(1,),
            final_holdout=(2,),
        )


def test_clip_starts_are_reproducible_and_inside_episode() -> None:
    first = plan_clip_starts(10.0, 20.0, 3, episode_index=4, seed=42)
    second = plan_clip_starts(10.0, 20.0, 3, episode_index=4, seed=42)
    assert first == second
    assert first[0] == 10.0
    assert first[-1] + 17 / 8 + 1 / 15 <= 20.0 + 1e-9


def test_clip_planner_rejects_short_episode() -> None:
    with pytest.raises(ValueError, match="too short"):
        plan_clip_starts(2.0, 3.0, 1, episode_index=9, seed=42)
