#!/usr/bin/env python3
"""Derive fail-closed architecture experiments from a physical pipeline report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.agent.contact_dynamics_evolution import (  # noqa: E402
    derive_foundation_pipeline_experiments,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    state: dict[str, object] = {}
    for label, command in (
        ("head", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--short"]),
    ):
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        state[label] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    source = args.pipeline_report.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output_dir}")
    output_dir.mkdir(parents=True)

    report = json.loads(source.read_text())
    if not isinstance(report, dict):
        raise ValueError("pipeline report must contain a JSON object")
    result = derive_foundation_pipeline_experiments(report)
    result.update(
        {
            "schema_version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {
                "numpy": importlib.metadata.version("numpy"),
            },
            "seed": args.seed,
            "git": _git_state(),
            "input": {
                "path": str(source),
                "sha256": _sha256(source),
            },
            "skillhone": {
                "configured": (Path.home() / ".skillhone/settings.json").is_file(),
                "behavioral_run_started": False,
                "reason": (
                    "architecture planning is deterministic; the independent SkillHone "
                    "behavioral/adversarial result is recorded by its own immutable run"
                ),
            },
        }
    )
    output = output_dir / "evolution-plan.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if bool(result["promotable"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
