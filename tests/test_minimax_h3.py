from pathlib import Path

import pytest

from phiagent.rendering.minimax_h3 import (
    H3ActionVariant,
    MiniMaxH3ValidationConfig,
    align_h3_frame_count,
    build_action_conditioned_flower_ref2va_prompt,
    build_action_conditioned_ego_bottle_ref2va_prompt,
    build_action_conditioned_tabletop_ref2va_prompt,
    build_flower_ref2va_prompt,
    build_flower_window_epl_constraint,
    flower_epl_phase,
    plan_h3_long_windows,
)


def test_tabletop_action_prompt_binds_object_terminal_state_to_control_video() -> None:
    action = H3ActionVariant(
        label="slide-left",
        instruction="Push the yellow bowl to the left target and hold it there.",
        timeline="0-1s approach; 1-4s push left; 4-5s hold.",
    )

    prompt = build_action_conditioned_tabletop_ref2va_prompt(5.167, action)

    assert "yellow handled bowl" in prompt
    assert "terminal bowl state" in prompt
    assert "<Video 1>" in prompt
    assert "human arm is absent" in prompt
    assert "contact the yellow bowl before it moves" in prompt


def test_ego_bottle_prompt_enforces_first_person_robot_hands_and_state() -> None:
    action = H3ActionVariant(
        label="pour-bottle",
        instruction="Pick up the bottle and pour into the cup.",
        timeline="0-2 s grasp; 2-7 s pour; 7-10 s place.",
    )

    prompt = build_action_conditioned_ego_bottle_ref2va_prompt(5.167, action)

    assert "EPIC-KITCHENS" in prompt
    assert "head-mounted first-person" in prompt
    assert "No robot head or torso is visible" in prompt
    assert "one persistent bottle" in prompt
    assert "holder transitions" in prompt
    assert action.instruction in prompt


def test_align_h3_frame_count() -> None:
    assert align_h3_frame_count(1) == 5
    assert align_h3_frame_count(5) == 5
    assert align_h3_frame_count(6) == 22
    assert align_h3_frame_count(124) == 124


def test_flower_prompt_has_required_ref2va_sections() -> None:
    prompt = build_flower_ref2va_prompt(5.167)
    for section in (
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ):
        assert section in prompt
    assert "<Picture 1>" in prompt
    assert "<Video 1>" in prompt
    assert "five-finger" in prompt


def test_action_conditioned_prompt_uses_language_instead_of_source_motion() -> None:
    action = H3ActionVariant(
        label="lift-and-inspect",
        instruction="Lift one pink flower and inspect it.",
        timeline="0-2 s approach; 2-4 s lift; 4-5.167 s hold.",
    )
    prompt = build_action_conditioned_flower_ref2va_prompt(5.167, action)

    assert action.instruction in prompt
    assert action.timeline in prompt
    assert "must not be copied" in prompt
    assert "language-conditioned action overrides" in prompt
    assert "same real room" in prompt


def test_action_conditioned_anchor_prompt_has_no_video_motion_reference() -> None:
    action = H3ActionVariant("inspect", "Inspect one flower.", "0-5 s inspect.")
    prompt = build_action_conditioned_flower_ref2va_prompt(
        5.167, action, scene_reference="anchor_image"
    )

    assert "<Picture 2>" in prompt
    assert "anchor frame extracted from the existing source video" in prompt
    assert "all temporal motion exclusively from <Subject 3>" in prompt
    assert "<Video 1>" not in prompt


def test_action_conditioned_control_prompt_splits_identity_scene_and_motion() -> None:
    action = H3ActionVariant("handover", "Transfer the flower.", "0-5 s transfer.")
    prompt = build_action_conditioned_flower_ref2va_prompt(
        5.167, action, scene_reference="control_video"
    )

    assert "identity from <Picture 1>" in prompt
    assert "real scene from <Picture 2>" in prompt
    assert "exact robot shoulder, elbow, wrist and hand trajectories" in prompt
    assert "ignore its CONTROL ONLY caption" in prompt


def test_action_variant_rejects_unsafe_label_and_empty_fields() -> None:
    with pytest.raises(ValueError, match="action label"):
        H3ActionVariant("../escape", "move", "now").validate()
    with pytest.raises(ValueError, match="instruction"):
        H3ActionVariant("valid", " ", "now").validate()


def test_h3_long_windows_cover_660_frames_with_overlap() -> None:
    windows = plan_h3_long_windows(660, window_frames=124, overlap_frames=28)

    assert [window.start_frame for window in windows] == [0, 96, 192, 288, 384, 480, 536]
    assert all(window.frame_count == 124 for window in windows)
    assert all(window.padded_frames == 0 for window in windows)
    covered = {
        frame
        for window in windows
        for frame in range(window.start_frame, window.end_frame_exclusive)
    }
    assert covered == set(range(660))


def test_h3_long_windows_reject_invalid_contract() -> None:
    with pytest.raises(ValueError, match="17n"):
        plan_h3_long_windows(660, window_frames=120)
    with pytest.raises(ValueError, match="overlap"):
        plan_h3_long_windows(660, overlap_frames=124)


def test_window_epl_constraint_marks_contact_and_absolute_phases() -> None:
    constraint = build_flower_window_epl_constraint(192, 124)

    assert "absolute source frames [192, 316)" in constraint
    assert "grasp, manipulate" in constraint
    assert "hard flower-contact interval" in constraint
    assert flower_epl_phase(0) == "approach"
    assert flower_epl_phase(659) == "retract"


def test_validation_config_rejects_invalid_h3_geometry(tmp_path: Path) -> None:
    paths = []
    for name in ("source.mp4", "robot.png", "prompt.txt"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths.append(path)
    config = MiniMaxH3ValidationConfig(
        source_video=paths[0],
        robot_reference=paths[1],
        prompt_file=paths[2],
        diffsynth_repo=tmp_path,
        model_base_path=tmp_path / "models",
        width=830,
    )
    with pytest.raises(ValueError, match="multiples of 32"):
        config.validate()
