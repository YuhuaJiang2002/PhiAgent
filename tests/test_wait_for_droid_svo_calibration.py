from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "wait_for_droid_svo_calibration.py"
    )
    spec = importlib.util.spec_from_file_location(
        "wait_for_droid_svo_calibration", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_free_classification_rejects_process_and_allocated_memory() -> None:
    gpus = _module().classify_gpu_lines(
        [
            "0, GPU-free, A800, 81920, 0",
            "1, GPU-process, A800, 81920, 0",
            "2, GPU-memory, A800, 81920, 1024",
        ],
        ["GPU-process, 123, python, 0"],
        maximum_used_mib=1023,
    )

    assert [row["classification"] for row in gpus] == [
        "free",
        "reserved_or_busy",
        "reserved_or_busy",
    ]


def test_selection_respects_requested_priority() -> None:
    gpus = [
        {"index": 0, "classification": "reserved_or_busy"},
        {"index": 2, "classification": "free"},
        {"index": 3, "classification": "free"},
    ]

    assert _module().select_strictly_free_gpu(gpus, [0, 3, 2]) == 3


def test_python_executable_path_preserves_venv_symlink(tmp_path: Path) -> None:
    target = tmp_path / "base-python"
    target.write_text("")
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    executable = venv / "python"
    executable.symlink_to(target)

    actual = _module().absolute_executable_path(executable)

    assert actual == executable
    assert actual.resolve() == target


def test_runtime_environment_prepends_explicit_pythonpath(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("PYTHONPATH", "/inherited")

    environment = _module().build_runtime_environment([first, second], "C")

    assert environment["PYTHONPATH"] == (
        f"{first}:{second}:/inherited"
    )
    assert environment["LC_ALL"] == "C"
    assert environment["LANG"] == "C"
