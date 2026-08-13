from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_hrdexdb_object_pilot.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_hrdexdb_object_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hrdexdb_pilot_split_is_object_disjoint() -> None:
    split = _module().pilot_split()

    assert len(set().union(*map(set, split.values()))) == 10
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])
    assert _module().FROZEN_PAIRS["banana"] == {
        "robot_scene": 3,
        "human_episode": 3,
        "robot_c2r": False,
    }


def test_hrdexdb_grasp_result_must_be_successful_and_paired() -> None:
    module = _module()

    module.validate_grasp_result(
        {"grasp_success": True, "human_paired_episode": 0},
        "apple",
    )
    with pytest.raises(ValueError, match="not a successful"):
        module.validate_grasp_result(
            {"grasp_success": False, "human_paired_episode": 0},
            "apple",
        )
    with pytest.raises(ValueError, match="not paired"):
        module.validate_grasp_result(
            {"grasp_success": True, "human_paired_episode": None},
            "apple",
        )
    with pytest.raises(ValueError, match="changed"):
        module.validate_grasp_result(
            {"grasp_success": True, "human_paired_episode": 2},
            "apple",
            expected_human_episode=1,
        )
