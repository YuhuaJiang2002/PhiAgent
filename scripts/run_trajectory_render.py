#!/usr/bin/env python3
"""Run the Cosmos 3 trajectory-conditioned renderer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.cosmos_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
