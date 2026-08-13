from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.build_droid_ego_to_third_person_demo import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    _compose_frame,
    _load_config,
    _stream_contract,
)


def _valid_config(tmp_path: Path) -> dict[str, object]:
    paths = {}
    for name in ("first_person", "third_person_a", "third_person_b"):
        path = tmp_path / f"{name}.mp4"
        path.write_bytes(b"video")
        paths[name] = path
    return {
        "fps": 15.0,
        "streams": {
            "first_person": {
                "path": str(paths["first_person"]),
                "coordinate_frame": "camera:wrist_image_left_pixels",
            },
            "third_person_a": {
                "path": str(paths["third_person_a"]),
                "coordinate_frame": "camera:exterior_image_1_left_pixels",
            },
            "third_person_b": {
                "path": str(paths["third_person_b"]),
                "coordinate_frame": "camera:exterior_image_2_left_pixels",
            },
        },
        "episodes": [
            {
                "episode_index": index,
                "label": f"task-{index}",
                "task": f"Task {index}",
                "start_frame": index * 10,
                "frame_count": 10,
                "timestamps": {
                    "first_person": {"from_seconds": float(index), "to_seconds": float(index + 1)},
                    "third_person_a": {"from_seconds": float(index), "to_seconds": float(index + 1)},
                    "third_person_b": {"from_seconds": float(index), "to_seconds": float(index + 1)},
                },
            }
            for index in range(3)
        ],
    }


def test_config_requires_explicit_synchronized_camera_frames(tmp_path: Path) -> None:
    payload = _valid_config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    loaded = _load_config(path)
    assert loaded["streams"]["first_person"]["coordinate_frame"] == "camera:wrist_image_left_pixels"

    payload["episodes"][0]["timestamps"]["third_person_b"]["to_seconds"] = 1.1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not synchronized"):
        _load_config(path)


def test_stream_contract_rejects_different_frame_counts() -> None:
    base = {"codec": "av1", "width": 320, "height": 180, "fps": 15.0, "frames": 100, "duration_seconds": 6.666667}
    probes = {
        "first_person": dict(base),
        "third_person_a": dict(base),
        "third_person_b": {**base, "frames": 99},
    }
    result = _stream_contract(probes)
    assert not result["passed"]
    assert not result["frame_counts_equal"]


def test_composed_frame_keeps_three_named_views() -> None:
    first = np.full((180, 320, 3), (10, 20, 30), dtype=np.uint8)
    third_a = np.full((180, 320, 3), (40, 50, 60), dtype=np.uint8)
    third_b = np.full((180, 320, 3), (70, 80, 90), dtype=np.uint8)
    composed = _compose_frame(
        cv2,
        np,
        first,
        third_a,
        third_b,
        episode_index=21,
        task="Put the duck in the pot",
        frame_index=4,
        frame_count=10,
    )
    assert composed.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
    assert composed[250, 200].mean() < composed[250, 620].mean() < composed[250, 1040].mean()
