#!/usr/bin/env python3
"""Prepare pinned Apache-2.0 Sharpa real and simulated interaction media."""

from __future__ import annotations

import argparse
import hashlib
import shlex
import subprocess
from pathlib import Path

SHARPA_RL_COMMIT = "95ccda3d948801bb5da4cb7ffea766e03067a63b"
EXPECTED_MEDIA = {
    "resources/real.gif": "d53c8b62cc8a3de91ebe5f6015c08b1d87c81d52443649ec12bfca05409ab8b7",
    "resources/sim.gif": "8d8f7b99e8f087d574f4da8a7c3855381b79ff3483162ca175517f9a4d42957a",
}


def _run(command: list[str], cwd: Path | None = None) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, default=Path("external"))
    args = parser.parse_args()
    external_root = args.external_root.expanduser().resolve()
    external_root.mkdir(parents=True, exist_ok=True)
    repo = external_root / "sharpa-rl-lab"
    if not repo.exists():
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/sharpa-robotics/sharpa-rl-lab.git",
                str(repo),
            ]
        )
    _run(["git", "fetch", "origin", SHARPA_RL_COMMIT], cwd=repo)
    _run(["git", "checkout", "--detach", SHARPA_RL_COMMIT], cwd=repo)
    license_path = repo / "LICENSE"
    if not license_path.is_file() or "Apache License" not in license_path.read_text():
        raise SystemExit("pinned Sharpa RL source is missing its Apache-2.0 license")
    for relative, expected in EXPECTED_MEDIA.items():
        path = repo / relative
        if not path.is_file() or _sha256(path) != expected:
            raise SystemExit(f"Sharpa RL media verification failed: {path}")
    print(f"SHARPA_RL_REPO={repo}")
    print(f"SHARPA_RL_COMMIT={SHARPA_RL_COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
