#!/usr/bin/env python3
"""Repository-local entry point for the Milestone 1 visual transfer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

