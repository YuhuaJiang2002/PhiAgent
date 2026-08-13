from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "wait_for_droid_raw_sequence_alignment.py"
    )
    spec = importlib.util.spec_from_file_location(
        "wait_for_droid_raw_sequence_alignment", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_audit_command_keeps_cases_and_gpu_explicit(tmp_path: Path) -> None:
    command = _module().build_audit_command(
        python_executable=Path("/venv/bin/python"),
        cases=[
            (Path("/raw/21.json"), Path("/sequence/21.bin")),
            (Path("/raw/77.json"), Path("/sequence/77.bin")),
        ],
        lineage_manifest=Path("/lineage/manifest.json"),
        output_dir=tmp_path / "alignment",
        gpu=5,
        minimum_free_gpu_mib=81_000,
        seed=20260812,
        git_commit="a" * 40,
        git_branch="main",
    )

    assert command.count("--case") == 2
    assert command[command.index("--gpu") + 1] == "5"
    assert command[command.index("--minimum-free-gpu-mib") + 1] == "81000"
