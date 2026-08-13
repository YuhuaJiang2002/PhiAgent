#!/usr/bin/env python3
"""Continuously supervise foundation-contact reports and retain only improvements."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.agent.foundation_contact_supervisor import supervise_once  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return value


def _git_state() -> dict[str, object]:
    result = {}
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
        result[label] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-report", type=Path, required=True)
    parser.add_argument("--evolution-plan", type=Path, required=True)
    parser.add_argument("--da3-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--watch-root", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-cycles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def _discover_latest(root: Path, current: Path) -> Path:
    reports = sorted(root.glob("**/pipeline-report.json"), key=lambda path: path.stat().st_mtime)
    return reports[-1].resolve() if reports else current


def main() -> int:
    args = _parser().parse_args()
    if args.maximum_cycles < 1 or args.poll_seconds <= 0:
        raise ValueError("cycles and poll interval must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output_dir}")
    inputs = {
        "pipeline_report": args.pipeline_report.expanduser().resolve(),
        "evolution_plan": args.evolution_plan.expanduser().resolve(),
        "da3_manifest": args.da3_manifest.expanduser().resolve()
        if args.da3_manifest
        else None,
    }
    for path in inputs.values():
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    watch_root = args.watch_root.expanduser().resolve() if args.watch_root else None
    if watch_root is not None and not watch_root.is_dir():
        raise FileNotFoundError(watch_root)
    output_dir.mkdir(parents=True)
    cycles_dir = output_dir / "cycles"
    cycles_dir.mkdir()
    started = time.perf_counter()
    observations = []
    current_report = inputs["pipeline_report"]
    last_digest = None
    for cycle_index in range(args.maximum_cycles):
        if watch_root is not None:
            current_report = _discover_latest(watch_root, current_report)
        digest = _sha256(current_report)
        if digest != last_digest:
            result = supervise_once(
                pipeline_report=_load(current_report),
                evolution_plan=_load(inputs["evolution_plan"]),
                da3_manifest=_load(inputs["da3_manifest"])
                if inputs["da3_manifest"]
                else None,
            )
            result.update(
                {
                    "cycle": cycle_index,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "pipeline_report": str(current_report),
                    "pipeline_report_file_sha256": digest,
                }
            )
            cycle_dir = cycles_dir / f"cycle-{cycle_index:04d}"
            cycle_dir.mkdir()
            (cycle_dir / "supervision.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            observations.append(result)
            last_digest = digest
        if cycle_index + 1 < args.maximum_cycles:
            time.sleep(args.poll_seconds)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {"numpy": importlib.metadata.version("numpy")},
        "seed": args.seed,
        "git": _git_state(),
        "inputs": {
            name: ({"path": str(path), "sha256": _sha256(path)} if path else None)
            for name, path in inputs.items()
        },
        "watch_root": str(watch_root) if watch_root else None,
        "requested_cycles": args.maximum_cycles,
        "observed_unique_reports": len(observations),
        "wall_seconds": time.perf_counter() - started,
        "latest": observations[-1] if observations else None,
        "status": "WORKING" if observations else "PARTIAL",
        "honest_scope": (
            "WORKING means the supervisor audited and ranked evidence, not that a physical "
            "model was promoted. Promotion remains encoded in latest.promoted."
        ),
    }
    (output_dir / "supervisor-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if observations else 2


if __name__ == "__main__":
    raise SystemExit(main())
