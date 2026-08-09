from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_script():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "build_full_robot_flower_demo.py"
    spec = importlib.util.spec_from_file_location("build_full_robot_flower_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hand(wrist_x: float) -> np.ndarray:
    points = np.zeros((21, 2), dtype=float)
    points[:, 0] = wrist_x
    return points


def test_hand_selection_tracks_available_hands_without_reusing_indices() -> None:
    module = _load_script()

    assert module._select_hands(np, [_hand(300), _hand(100)], None) == [1, 0]
    assert module._select_hands(
        np,
        [_hand(110), _hand(290)],
        [_hand(100), _hand(300)],
    ) == [0, 1]
    assert module._select_hands(np, [_hand(110)], [_hand(100), _hand(300)]) == [0]


def test_default_hand_pose_is_non_degenerate() -> None:
    module = _load_script()

    image_hands, world_hands = module._default_hand_pose(np, (960, 540))

    assert image_hands[0].shape == (21, 2)
    assert world_hands[0].shape == (21, 3)
    assert np.linalg.norm(world_hands[0][9] - world_hands[0][0]) > 0
