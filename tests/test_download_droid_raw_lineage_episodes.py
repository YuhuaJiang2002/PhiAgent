from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "download_droid_raw_lineage_episodes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "download_droid_raw_lineage_episodes", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _object(prefix: str, relative: str) -> dict[str, str]:
    return {
        "name": f"robotics/droid_raw/{prefix}/{relative}",
        "size": "10",
        "md5Hash": "hash",
        "generation": "123",
        "mediaLink": "https://example.invalid/object",
    }


def test_selects_only_calibration_critical_raw_objects() -> None:
    prefix = "1.0.1/AUTOLab/success/date/episode"
    objects = [
        _object(prefix, "trajectory.h5"),
        _object(prefix, "trajectory_im128.h5"),
        _object(prefix, "metadata_episode.json"),
        _object(prefix, "recordings/SVO/1.svo"),
        _object(prefix, "recordings/SVO/2.svo"),
        _object(prefix, "recordings/SVO/3.svo"),
        _object(prefix, "recordings/MP4/1.mp4"),
    ]

    selected = _module().select_calibration_objects(objects, prefix)

    assert [row["relative_path"] for row in selected] == [
        "metadata_episode.json",
        "recordings/SVO/1.svo",
        "recordings/SVO/2.svo",
        "recordings/SVO/3.svo",
        "trajectory.h5",
    ]


def test_selection_rejects_incomplete_camera_set() -> None:
    prefix = "1.0.1/AUTOLab/success/date/episode"
    objects = [
        _object(prefix, "trajectory.h5"),
        _object(prefix, "metadata_episode.json"),
        _object(prefix, "recordings/SVO/1.svo"),
        _object(prefix, "recordings/SVO/2.svo"),
    ]

    with pytest.raises(ValueError, match="three SVO"):
        _module().select_calibration_objects(objects, prefix)


def test_partial_paths_do_not_collide_between_svos(tmp_path: Path) -> None:
    module = _module()

    assert module.partial_path(tmp_path / "1.svo") != module.partial_path(
        tmp_path / "2.svo"
    )
