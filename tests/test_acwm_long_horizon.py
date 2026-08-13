from __future__ import annotations

import json
from pathlib import Path

import pytest

from phiagent.acwm.long_horizon import (
    LongHorizonActionSet,
    window_action_manifest,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "minimax_h3_long_flower_actions_v1.json"
EGO_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "minimax_h3_long_epic_ego_bottle_actions_v1.json"
)


def test_ten_second_actions_compile_to_two_legal_h3_windows() -> None:
    action_set = LongHorizonActionSet.load(CONFIG)

    compiled = action_set.compile_matched_windows(
        total_frames=240,
        fps=24,
        window_frames=124,
        overlap_frames=8,
    )

    assert len(compiled) == 3
    assert all([(item.start_frame, item.frame_count) for item in windows] == [(0, 124), (116, 124)] for windows in compiled)
    assert all((item.frame_count - 5) % 17 == 0 for windows in compiled for item in windows)
    assert all(windows[1].start_frame < windows[0].end_frame for windows in compiled)


def test_window_contract_preserves_action_specific_object_state() -> None:
    action_set = LongHorizonActionSet.load(CONFIG)
    compiled = action_set.compile_matched_windows(
        total_frames=240,
        fps=24,
        window_frames=124,
        overlap_frames=8,
    )
    second = {action.label: windows[1] for action, windows in zip(action_set.actions, compiled)}

    assert second["insert-flower"].entry_object_holder == "right_hand"
    assert second["insert-flower"].exit_object_holder == "vase"
    assert second["handover-flower"].exit_object_holder == "left_hand"
    assert second["inspect-flower"].exit_object_holder == "right_hand"
    assert "Do not reset" in second["handover-flower"].timeline
    assert second["insert-flower"].coordinate_frame == action_set.coordinate_frame
    assert second["insert-flower"].object_name == "flower"


def test_public_ego_object_and_support_states_are_not_flower_specific(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text())
    payload["scene"] = "public EPIC-KITCHENS egocentric bottle task"
    payload["coordinate_frame"] = "camera:epic_kitchens_p03_28_pixels"
    payload["object_name"] = "bottle"
    for action in payload["actions"]:
        action["label"] = action["label"].replace("flower", "bottle")
        action["instruction"] = action["instruction"].replace("flower", "bottle")
        for phase in action["phases"]:
            phase["description"] = phase["description"].replace("flower", "bottle")
            if phase["entry_object_holder"] == "vase":
                phase["entry_object_holder"] = "counter"
            if phase["exit_object_holder"] == "vase":
                phase["exit_object_holder"] = "counter"
    path = tmp_path / "ego.json"
    path.write_text(json.dumps(payload))

    action_set = LongHorizonActionSet.load(path)
    windows = action_set.compile_matched_windows(
        total_frames=240, fps=24, window_frames=124, overlap_frames=8
    )

    assert windows[0][1].coordinate_frame == "camera:epic_kitchens_p03_28_pixels"
    assert windows[0][1].object_name == "bottle"
    assert "flower" not in windows[0][1].timeline.lower()
    assert "bottle" in windows[0][1].timeline.lower()


def test_window_manifest_is_accepted_by_existing_h3_action_loader(tmp_path: Path) -> None:
    from scripts.run_minimax_h3_action_variants import _load_actions

    action_set = LongHorizonActionSet.load(CONFIG)
    compiled = action_set.compile_matched_windows(
        total_frames=240,
        fps=24,
        window_frames=124,
        overlap_frames=8,
    )
    path = tmp_path / "window-actions.json"
    path.write_text(
        json.dumps(window_action_manifest(action_set.actions, [items[1] for items in compiled]))
    )

    loaded = _load_actions(path)

    assert [item.label for item in loaded] == [
        "insert-flower",
        "handover-flower",
        "inspect-flower",
    ]


def test_action_state_discontinuity_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text())
    payload["actions"][0]["phases"][1]["entry_object_holder"] = "left_hand"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="object-holder state"):
        LongHorizonActionSet.load(path)


def test_epic_ego_bottle_contract_has_distinct_terminal_states() -> None:
    action_set = LongHorizonActionSet.load(EGO_CONFIG)
    compiled = action_set.compile_matched_windows(
        total_frames=240, fps=24, window_frames=124, overlap_frames=8
    )
    second = {action.label: windows[1] for action, windows in zip(action_set.actions, compiled)}

    assert action_set.object_name == "bottle"
    assert second["pour-bottle"].exit_object_holder == "counter"
    assert second["shake-bottle"].entry_object_holder == "both_hands"
    assert second["handover-bottle"].exit_object_holder == "left_hand"
    assert all("flower" not in window.timeline.lower() for windows in compiled for window in windows)
