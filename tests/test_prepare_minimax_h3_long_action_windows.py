from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_long_action_window_preparer_has_help() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_minimax_h3_long_action_windows.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--overlap-frames" in completed.stdout
