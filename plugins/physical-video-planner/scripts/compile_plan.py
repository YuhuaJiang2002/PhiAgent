#!/usr/bin/env python3
"""Forward to the active PhiAgent physical-video plan compiler."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    project_root = Path(completed.stdout.strip()).resolve()
    compiler = project_root / "scripts" / "compile_physical_video_plan.py"
    if not compiler.is_file():
        raise RuntimeError(
            "the active Git workspace does not provide scripts/compile_physical_video_plan.py"
        )
    return subprocess.run(
        [sys.executable, str(compiler), *sys.argv[1:]],
        cwd=project_root,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
