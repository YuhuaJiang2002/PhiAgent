#!/usr/bin/env python3
"""Clone the pinned HaMeR source without bypassing MANO licensing."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.perception.hand.hamer import HAMER_COMMIT


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("external/hamer"))
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "https://github.com/geopavlakos/hamer.git", str(destination)])
    run(["git", "fetch", "origin", HAMER_COMMIT], destination)
    run(["git", "checkout", "--detach", HAMER_COMMIT], destination)
    mano = destination / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl"
    print(f"HAMER_REPOSITORY={destination}")
    print(f"MANO_REQUIRED_AT={mano}")
    if not mano.is_file():
        print("MANO_STATUS=missing_separately_licensed_asset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
