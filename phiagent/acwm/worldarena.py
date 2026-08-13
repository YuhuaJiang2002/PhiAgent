"""WorldArena metadata lineage helpers shared by evaluation campaigns."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

WORLD_ARENA_EEF_QUATERNION_CHANNELS = (
    "left_eef_pos_x_m",
    "left_eef_pos_y_m",
    "left_eef_pos_z_m",
    "left_eef_quaternion_x",
    "left_eef_quaternion_y",
    "left_eef_quaternion_z",
    "left_eef_quaternion_w",
    "right_eef_pos_x_m",
    "right_eef_pos_y_m",
    "right_eef_pos_z_m",
    "right_eef_quaternion_x",
    "right_eef_quaternion_y",
    "right_eef_quaternion_z",
    "right_eef_quaternion_w",
)


def worldarena_episode_lineage(
    dataset_manifest: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    """Index compiled clips by their physical source episode."""

    episodes = dataset_manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("dataset manifest requires an episodes array")
    lineage: dict[str, dict[str, object]] = {}
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("dataset manifest episode records must be objects")
        task = str(episode.get("task", "")).strip()
        name = str(episode.get("episode", "")).strip()
        meta = episode.get("meta")
        if not task or not name or not isinstance(meta, Mapping):
            raise ValueError("dataset manifest episode lacks task, episode, or meta")
        source_episode = str(meta.get("source_episode", "")).strip()
        episode_id = str(meta.get("episode_id", "")).strip()
        if not source_episode and not episode_id:
            raise ValueError("dataset manifest lacks physical episode lineage")
        key = f"{task}/{name}"
        if key in lineage:
            raise ValueError(f"duplicate compiled episode lineage {key!r}")
        lineage[key] = {
            "independent_group_id": source_episode
            or f"{task}/physical-episode-{episode_id}",
            "physical_episode_id": episode_id,
            "source_clip_index": meta.get("source_clip_index"),
            "device_id": meta.get("device_id"),
            "device_name": meta.get("device_name"),
        }
    return lineage


def attach_worldarena_lineage(
    rows: Sequence[Mapping[str, Any]],
    dataset_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Copy compiled rows and attach their physical-episode lineage."""

    lineage = worldarena_episode_lineage(dataset_manifest)
    enriched = []
    for row in rows:
        result = copy.deepcopy(dict(row))
        key = str(result.get("source_episode", ""))
        if key not in lineage:
            raise ValueError(f"source metadata row lacks manifest lineage: {key}")
        result.update(lineage[key])
        enriched.append(result)
    return tuple(enriched)
