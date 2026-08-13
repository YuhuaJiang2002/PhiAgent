import math

import pytest

from phiagent.acwm.robotwin import (
    BWM_EEF_CHANNELS,
    RoboTwinEpisode,
    bwm_clip_record,
    eef16_to_bwm14,
    grouped_split,
    overlapping_clip_starts,
    parse_robotwin_task,
)


def test_eef16_to_bwm14_preserves_positions_and_converts_xyzw() -> None:
    half_turn_z = math.sqrt(0.5)
    source = (
        0.1,
        0.2,
        0.3,
        0.0,
        0.0,
        half_turn_z,
        half_turn_z,
        1.0,
        -0.1,
        -0.2,
        0.4,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )

    converted = eef16_to_bwm14(source)

    assert len(converted) == len(BWM_EEF_CHANNELS) == 14
    assert converted[:3] == pytest.approx((0.1, 0.2, 0.3))
    assert converted[3:6] == pytest.approx((0.0, 0.0, math.pi / 2))
    assert converted[6] == 1.0
    assert converted[7:10] == pytest.approx((-0.1, -0.2, 0.4))
    assert converted[10:13] == pytest.approx((0.0, 0.0, 0.0))
    assert converted[13] == 0.0


def test_grouped_split_is_stable_for_every_paraphrase_in_group() -> None:
    parsed = parse_robotwin_task(
        "[franka] adjust_bottle: Using the right arm, lift the green bottle"
    )
    assert parsed == (
        "franka",
        "adjust_bottle",
        "Using the right arm, lift the green bottle",
    )
    first = grouped_split(parsed[0], parsed[1], seed=20260811)
    second = grouped_split("franka", "adjust_bottle", seed=20260811)
    assert first == second
    assert first in {"train", "validation", "test"}


def test_overlapping_clips_include_terminal_state() -> None:
    assert overlapping_clip_starts(57) == (0,)
    assert overlapping_clip_starts(116, num_frames=57, history=9) == (0, 48, 59)
    assert overlapping_clip_starts(56) == ()


def test_bwm_clip_keeps_video_and_action_offsets_separate() -> None:
    episode = RoboTwinEpisode(
        episode_index=7,
        embodiment="piper",
        task="stack_blocks_two",
        instruction="Stack the red block on the blue block",
        length=100,
        data_path="data/chunk-000/file-001.parquet",
        video_path="videos/observation.images.head/chunk-000/file-003.mp4",
        data_start_frame=12,
        video_start_frame=90,
        coordinate_frame="robot_base:robotwin2-piper",
    )

    row = bwm_clip_record(episode, clip_start=20)

    assert row["video"]["start_frame"] == 110
    assert row["video"]["end_frame"] == 166
    assert row["action"]["start_frame"] == 32
    assert row["action"]["end_frame"] == 88
    assert row["coordinate_frame"] == "robot_base:robotwin2-piper"


def test_episode_rejects_implicit_world_or_camera_frame() -> None:
    with pytest.raises(ValueError, match="robot-base"):
        RoboTwinEpisode(
            episode_index=0,
            embodiment="ur5",
            task="turn_switch",
            instruction="Turn the switch",
            length=80,
            data_path="actions.parquet",
            video_path="video.mp4",
            data_start_frame=0,
            video_start_frame=0,
            coordinate_frame="world:robotwin2",
        )
