from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_droid_sequence_lineage.py"
    )
    spec = importlib.util.spec_from_file_location("audit_droid_sequence_lineage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_derive_raw_paths_recovers_gcs_prefix() -> None:
    episode_root = (
        "/nfs/kun2/datasets/r2d2/r2d2-data-full/AUTOLab/success/"
        "2023-07-14/Fri_Jul_14_16:55:45_2023"
    )

    actual = _module().derive_raw_paths(
        f"{episode_root}/trajectory.h5",
        f"{episode_root}/recordings/MP4",
    )

    assert actual["raw_gcs_prefix"] == (
        "1.0.1/AUTOLab/success/2023-07-14/Fri_Jul_14_16:55:45_2023"
    )


def test_derive_raw_paths_rejects_mismatched_recording_root() -> None:
    with pytest.raises(ValueError, match="does not share"):
        _module().derive_raw_paths(
            "/nfs/kun2/datasets/r2d2/r2d2-data-full/A/trajectory.h5",
            "/nfs/kun2/datasets/r2d2/r2d2-data-full/B/recordings/MP4",
        )


def test_dhash_hamming_counts_changed_bits() -> None:
    assert _module().dhash_hamming(0b1010, 0b0011) == 2


def test_nearest_rank_percentile_is_deterministic() -> None:
    assert _module().percentile_nearest_rank([4.0, 1.0, 3.0, 2.0], 0.05) == 1.0
    assert _module().percentile_nearest_rank([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0


def test_explicit_git_state_supports_remote_experiment_copy() -> None:
    state = _module()._git_state("a" * 40, "main")

    assert state["commit"] == "a" * 40
    assert state["branch"] == "main"
    assert state["dirty"] is None
    assert len(state["audit_script_sha256"]) == 64


def test_exterior_assignment_detects_episode_local_camera_swap() -> None:
    first = "exterior_image_1_left"
    second = "exterior_image_2_left"
    metrics = {
        (first, first): {"psnr_db_median": 9.0},
        (first, second): {"psnr_db_median": 35.0},
        (second, first): {"psnr_db_median": 36.0},
        (second, second): {"psnr_db_median": 9.5},
    }

    selected, mapping, scores = _module().select_exterior_assignment(metrics)

    assert selected == "swapped"
    assert mapping == {first: second, second: first}
    assert scores["swapped"] == 71.0
