from __future__ import annotations

import pytest

from phiagent.rendering.joyai_video_edit import (
    DEFAULT_SCISSORS_CONTRACT,
    HeldToolContract,
    causal_padded_frame_count,
    causal_tail_padding_frames,
    flower_full_stream_prompt,
)
from scripts.repair_robot_hand_layer import expanded_repair_frames


def test_full_stream_contract_preserves_scissors_and_flower_dynamics() -> None:
    prompt = flower_full_stream_prompt().lower()
    assert "never freezes" in prompt
    assert "floats" in prompt
    assert "black-handled" in prompt
    assert "fingers remain closed through its handles" in prompt
    assert "pivot remains attached" in prompt


def test_scissors_contract_requires_native_resolution_human_veto() -> None:
    manifest = DEFAULT_SCISSORS_CONTRACT.to_manifest()
    assert manifest["source_start_frame"] == 398
    assert manifest["source_end_frame"] == 447
    assert manifest["holder"] == "robot_right_hand"
    assert manifest["review_authority"] == "native_resolution_human_veto"
    assert manifest["automatic_promotion"] is False
    assert manifest["physical_evidence"] is False


def test_scissors_review_frames_must_be_inside_interval() -> None:
    with pytest.raises(ValueError, match="outside"):
        HeldToolContract(
            name="scissors",
            source_start_frame=10,
            source_end_frame=20,
            holder="robot_right_hand",
            topology=("one_rigid_tool",),
            required_review_frames=(9, 15),
        ).validate(total_frames=30)


def test_full_stream_uses_only_minimum_causal_tail_padding() -> None:
    assert causal_padded_frame_count(660) == 665
    assert causal_tail_padding_frames(660) == 5


def test_failed_contact_frames_expand_into_bounded_temporal_intervals() -> None:
    assert expanded_repair_frames(
        [626, 628, 638], total_frames=660, padding=2, maximum_gap=3
    ) == (*range(624, 631), *range(636, 641))
