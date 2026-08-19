"""Small, dependency-free provenance capture for physical-video harness runs."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(args: Sequence[str], root: Path) -> dict[str, object]:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {"ok": True, "value": completed.stdout.strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": str(exc)}


def capture_provenance(root: Path, command: Sequence[str], seed: int) -> dict[str, object]:
    packages = sorted(
        {
            f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
            for distribution in metadata.distributions()
        }
    )
    return {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "command": list(command),
        "seed": seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": {
            "head": _git(("rev-parse", "--verify", "HEAD"), root),
            "status": _git(("status", "--short", "--untracked-files=no"), root),
        },
        "packages": packages,
    }


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
