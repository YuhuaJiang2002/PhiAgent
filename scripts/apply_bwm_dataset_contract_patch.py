#!/usr/bin/env python3
"""Apply and record BWM's DiffSynth dataset-runner contract patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.adapters import BWM_REPOSITORY_COMMIT  # noqa: E402


def _patch(repository: Path, content: bytes, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["patch", "-p1", "--batch", "--force", *arguments],
        cwd=repository,
        input=content,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.expanduser().resolve()
    marker = repository / ".phiagent-source-revision"
    revision = marker.read_text().strip() if marker.is_file() else None
    if revision != BWM_REPOSITORY_COMMIT:
        raise ValueError(f"BWM source revision is {revision!r}, expected {BWM_REPOSITORY_COMMIT}")
    patch_path = (
        Path(__file__).resolve().parents[1]
        / "patches"
        / "bwm-dataset-runner-contract.patch"
    )
    content = patch_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    manifest_path = repository / ".phiagent-dataset-contract-patch-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        reverse = _patch(repository, content, "--dry-run", "--reverse")
        if manifest.get("patch_sha256") == digest and reverse.returncode == 0:
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        raise ValueError("existing BWM dataset-contract patch evidence is invalid")
    forward = _patch(repository, content, "--dry-run", "--forward")
    already_applied = False
    if forward.returncode != 0:
        reverse = _patch(repository, content, "--dry-run", "--reverse")
        if reverse.returncode != 0:
            raise RuntimeError("BWM dataset-contract patch cannot be applied or verified")
        already_applied = True
    if not already_applied:
        applied = _patch(repository, content, "--forward")
        if applied.returncode != 0:
            raise RuntimeError(applied.stdout.decode(errors="replace"))
    manifest = {
        "schema_version": "1.0.0",
        "source_revision": revision,
        "patch": str(patch_path),
        "patch_sha256": digest,
        "already_applied_on_entry": already_applied,
        "purpose": "declare that RoboTwin JSONL samples are not precomputed training cache",
        "applied_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
