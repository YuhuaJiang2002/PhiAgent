from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_droid_raw_sequence_alignment.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_droid_raw_sequence_alignment", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timestamp_alignment_accepts_one_terminal_row_and_constant_offset() -> None:
    metrics = _module().timestamp_alignment_metrics(
        [
            1_041_000_000,
            1_108_000_000,
            1_175_000_000,
        ],
        [1000, 1067, 1134, 1201],
    )

    assert metrics["svo_minus_hdf_offset_median_ms"] == pytest.approx(41.0)
    assert metrics["centered_absolute_residual_p95_ms"] == pytest.approx(0.0)
    assert metrics["terminal_hdf_row_after_last_svo_ms"] == pytest.approx(26.0)


def test_timestamp_alignment_rejects_missing_terminal_row() -> None:
    with pytest.raises(ValueError, match="exactly one terminal"):
        _module().timestamp_alignment_metrics(
            [1_041_000_000, 1_108_000_000],
            [1000, 1067],
        )


def test_camera_roles_keep_raw_names_explicit() -> None:
    assert _module().CAMERA_ROLES["exterior_1"] == (
        "ext1_cam_serial",
        "exterior_image_1_left",
    )


def test_selected_gpu_processes_match_uuid_not_memory_only() -> None:
    actual = _module().selected_gpu_processes(
        ["0, GPU-zero", "5, GPU-five"],
        [
            "GPU-zero, 100, python, 2000",
            "GPU-five, 200, python, 0",
        ],
        5,
    )

    assert actual == ["GPU-five, 200, python, 0"]
