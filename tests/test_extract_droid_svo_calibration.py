from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "extract_droid_svo_calibration.py"
    )
    spec = importlib.util.spec_from_file_location("extract_droid_svo_calibration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_svo_sample_indices_cover_start_middle_and_end() -> None:
    assert _module().sample_indices(10) == (0, 5, 8)
    assert _module().sample_indices(1) == (0,)


def test_svo_sample_indices_reject_empty_video() -> None:
    with pytest.raises(ValueError, match="positive"):
        _module().sample_indices(0)


def test_resolves_lineage_mapped_episode_identity() -> None:
    actual = _module().resolve_episode_identity(
        {
            "episode_index": 21,
            "raw_gcs_prefix": "1.0.1/AUTOLab/success/example",
            "exterior_assignment": "swapped",
            "sequence_payload_sha256": "a" * 64,
        },
        {"uuid": "AUTOLab+user+timestamp"},
    )

    assert actual["episode_id"] == "AUTOLab+user+timestamp"
    assert actual["episode_index"] == 21
    assert actual["lineage_mapped_to_lerobot"] is True


def test_preserves_legacy_smoke_episode_id() -> None:
    actual = _module().resolve_episode_identity(
        {"episode_id": "IPRL+user+timestamp"},
        {"uuid": "metadata-uuid"},
    )

    assert actual == {
        "episode_id": "IPRL+user+timestamp",
        "lineage_mapped_to_lerobot": False,
    }
