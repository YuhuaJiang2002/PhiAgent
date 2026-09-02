from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from phiagent.perception.cachevda import (
    CacheVDAConfig,
    CacheVDARequest,
    CacheVDARunner,
    validate_cachevda_timing,
)


def _timing_payload(frame_count: int = 120) -> dict[str, object]:
    return {
        "status": "completed",
        "video": {"frame_count": frame_count},
        "inference": {"window_count": 6, "peak_reserved_gib": 7.2},
    }


def test_timing_contract_accepts_completed_relative_depth_run() -> None:
    payload = _timing_payload()
    assert validate_cachevda_timing(payload, expected_max_frames=120) is payload


def test_timing_contract_rejects_wrong_frame_count() -> None:
    with pytest.raises(RuntimeError, match="produced 119 frames"):
        validate_cachevda_timing(_timing_payload(119), expected_max_frames=120)


def test_cachevda_command_uses_external_python_and_output(tmp_path: Path) -> None:
    repository = tmp_path / "cachevda"
    config = CacheVDAConfig(repository=repository, gpu_index=3)
    request = CacheVDARequest(
        input_video=tmp_path / "rgb.mp4",
        experiment_dir=tmp_path / "experiment",
        max_frames=120,
    )
    output = tmp_path / "experiment" / "outputs"
    command = CacheVDARunner(config).build_command(request, output)
    assert command[0] == str((repository / ".venv/bin/python").absolute())
    assert str(repository / "experiments/feature_cache/run_e2e_optimized.py") in command
    assert command[command.index("--max-frames") + 1] == "120"
    assert command[command.index("--output-dir") + 1] == str(output.resolve())


def test_invalid_cachevda_configuration_fails_without_model_import() -> None:
    with pytest.raises(ValueError, match="encoder"):
        CacheVDAConfig(repository=Path("external/cachevda"), encoder="invalid")
    with pytest.raises(ValueError, match="max_frames"):
        CacheVDARequest(Path("rgb.mp4"), Path("run"), max_frames=0)


def test_importing_cachevda_adapter_does_not_import_torch() -> None:
    probe = (
        "import json, sys; import phiagent.perception.cachevda; "
        "print(json.dumps({'torch': 'torch' in sys.modules, "
        "'numpy': 'numpy' in sys.modules, 'cv2': 'cv2' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"torch": False, "numpy": False, "cv2": False}
