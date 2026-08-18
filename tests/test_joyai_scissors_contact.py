from __future__ import annotations

import pytest

from phiagent.rendering.joyai_video_edit import (
    DEFAULT_SCISSORS_CONTRACT,
    HeldToolContract,
    JOYAI_DIT_BYTES,
    JOYAI_DIT_RELATIVE_PATH,
    JOYAI_MODEL_REVISION,
    JOYAI_REPOSITORY_REVISION,
    build_server_argv,
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
    assert "local shape popping" in prompt
    assert "crawling texture" in prompt
    assert "causal chunk boundary" in prompt


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


def test_rv2v_server_contract_pins_upgraded_0811_release(tmp_path) -> None:
    assert JOYAI_DIT_RELATIVE_PATH.name == "joyai_video_edit_dit_0811.pth"
    assert JOYAI_DIT_BYTES == 32_527_662_903
    assert JOYAI_MODEL_REVISION == "e14d9ac50d4ad8e9f91b655bfab270c02a43923b"
    assert JOYAI_REPOSITORY_REVISION == "3478e4b8c9a79fe935157d1d477cd3e57bb41f1f"
    argv = build_server_argv(
        python_executable=tmp_path / "python",
        repository=tmp_path / "repository",
        checkpoint_root=tmp_path / "checkpoints",
        record_dir=tmp_path / "records",
        host="127.0.0.1",
        port=18080,
    )
    assert "--fps" in argv
    assert "--max-inflight-chunks" in argv
    assert "--holder-idle-timeout-s" not in argv
    assert "--no-person-count-reedit" not in argv


def test_failed_contact_frames_expand_into_bounded_temporal_intervals() -> None:
    assert expanded_repair_frames(
        [626, 628, 638], total_frames=660, padding=2, maximum_gap=3
    ) == (*range(624, 631), *range(636, 641))
