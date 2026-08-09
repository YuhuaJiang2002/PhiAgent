#!/usr/bin/env python3
"""Prepare the pinned Apache-2.0 DiffSynth Wan-Animate training source."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.training.diffsynth_animate import (
    DIFFSYNTH_COMMIT,
    DIFFSYNTH_REPOSITORY,
    verify_diffsynth_checkout,
)


def _run(command: list[str], cwd: Path | None = None) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, default=Path("external"))
    args = parser.parse_args()
    external_root = args.external_root.expanduser().resolve()
    external_root.mkdir(parents=True, exist_ok=True)
    repo = external_root / "DiffSynth-Studio"
    if not repo.exists():
        _run(["git", "clone", "--filter=blob:none", DIFFSYNTH_REPOSITORY, str(repo)])
    _run(["git", "fetch", "origin", DIFFSYNTH_COMMIT], cwd=repo)
    _run(["git", "checkout", "--detach", DIFFSYNTH_COMMIT], cwd=repo)
    verify_diffsynth_checkout(repo)
    print(f"DIFFSYNTH_REPO={repo}")
    print(f"DIFFSYNTH_COMMIT={DIFFSYNTH_COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
