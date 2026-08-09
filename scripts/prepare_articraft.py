#!/usr/bin/env python3
"""Prepare the pinned, optional mini-ArtiCraft checkout."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ARTICRAFT_COMMIT = "7d43e25b26e9459aabf53d77d1d9325805bc1ea3"
ARTICRAFT_REPOSITORY = "https://github.com/articraftresearch/Articraft.git"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, default=Path("external"))
    parser.add_argument(
        "--install",
        action="store_true",
        help="run the upstream uv sync command, including its optional simulation dependencies",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python used for the isolated venv when uv is unavailable",
    )
    args = parser.parse_args()

    if shutil.which("git") is None:
        raise SystemExit("git is required")
    external_root = args.external_root.expanduser().resolve()
    external_root.mkdir(parents=True, exist_ok=True)
    repo = external_root / "Articraft"
    if not repo.exists():
        run(["git", "clone", ARTICRAFT_REPOSITORY, str(repo)])
    run(["git", "fetch", "origin", ARTICRAFT_COMMIT], cwd=repo)
    run(["git", "checkout", "--detach", ARTICRAFT_COMMIT], cwd=repo)

    if args.install:
        uv = shutil.which("uv")
        if uv is not None:
            run([uv, "sync", "--group", "sim"], cwd=repo)
        else:
            python = args.python.expanduser().absolute()
            if not python.is_file():
                raise SystemExit(f"Python executable does not exist: {python}")
            venv = repo / ".venv"
            run([str(python), "-m", "venv", str(venv)])
            venv_python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            run([str(venv_python), "-m", "pip", "install", "-e", str(repo)])

    print(f"ARTICRAFT_REPO={repo}")
    print(f"ARTICRAFT_PYTHON={repo / '.venv/bin/python'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
