#!/usr/bin/env python3
"""Audit physical authority across real and model-derived observation sources."""

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
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.perception.generated_observation_authority import (  # noqa: E402
    ObservationSource,
    ObservationSourceKind,
    audit_observation_sources,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("source manifest must contain a JSON object")
    return value


def main() -> int:
    args = _parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output_dir}")
    manifest = _load(manifest_path)
    rows = manifest.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest requires a non-empty sources list")
    sources = []
    artifacts = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source row must be an object")
        artifact = Path(str(row["artifact_path"])).expanduser().resolve()
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        digest = _sha256(artifact)
        if digest != row["source_sha256"]:
            raise ValueError(f"source digest mismatch: {artifact}")
        artifacts.append({"path": str(artifact), "sha256": digest})
        sources.append(
            ObservationSource(
                source_id=str(row["source_id"]),
                kind=ObservationSourceKind(str(row["kind"])),
                acquisition_group_id=(
                    str(row["acquisition_group_id"])
                    if row.get("acquisition_group_id") is not None
                    else None
                ),
                source_sha256=digest,
                timeline=str(row["timeline"]),
                coordinate_frame=str(row["coordinate_frame"]),
                synchronized=bool(row["synchronized"]),
                physically_captured=bool(row["physically_captured"]),
                generated_from_source_sha256=row.get("generated_from_source_sha256"),
                metric_calibration_passed=bool(
                    row.get("metric_calibration_passed", False)
                ),
                exact_asset_bound=bool(row.get("exact_asset_bound", False)),
                solver_inputs_physically_accepted=bool(
                    row.get("solver_inputs_physically_accepted", False)
                ),
            )
        )
    audit = audit_observation_sources(sources)
    git = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "artifacts": artifacts,
        "git_status": git.stdout.splitlines(),
        "audit": audit,
        "status": "PARTIAL",
        "status_reason": (
            "the original RGB capture is a physical group, but it is not a passed "
            "metric calibration and model-derived sources add no physical authority"
        ),
    }
    output_dir.mkdir(parents=True)
    path = output_dir / "observation-source-authority.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
