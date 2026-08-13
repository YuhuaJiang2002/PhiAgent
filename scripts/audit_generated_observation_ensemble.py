#!/usr/bin/env python3
"""Bind and audit multiple VLM physical-observation probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.perception.generated_observation_authority import (  # noqa: E402
    audit_vlm_ensemble,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output_dir}")
    paths = [path.expanduser().resolve() for path in args.report]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("all observation reports must exist")
    reports = [json.loads(path.read_text()) for path in paths]
    audit = audit_vlm_ensemble(reports)
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    payload = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "git_status": completed.stdout.splitlines(),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "audit": audit,
        "status": "PARTIAL",
        "status_reason": (
            "model ensemble produced useful triage labels but zero independent "
            "physical acquisition groups"
        ),
    }
    output_dir.mkdir(parents=True)
    report_path = output_dir / "ensemble-observation-audit.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
