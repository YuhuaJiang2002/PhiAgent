#!/usr/bin/env python3
"""Apply and record the reviewed RoboTwin packed-offset compatibility patch."""

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


def _source_revision(repository: Path) -> str:
    if (repository / ".git").is_dir():
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    marker = repository / ".phiagent-source-revision"
    if not marker.is_file():
        raise ValueError(f"BWM source has no revision evidence: {repository}")
    return marker.read_text().strip()


def _run_patch(repository: Path, patch: bytes, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["patch", "-p1", "--batch", "--force", *arguments],
        cwd=repository,
        input=patch,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument(
        "--patch",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "patches"
        / "bwm-robotwin-packed-offsets.patch",
    )
    args = parser.parse_args()
    repository = args.repository.expanduser().resolve()
    patch_path = args.patch.expanduser().resolve()
    revision = _source_revision(repository)
    if revision != BWM_REPOSITORY_COMMIT:
        raise ValueError(
            f"BWM source revision is {revision}, expected {BWM_REPOSITORY_COMMIT}"
        )
    patch = patch_path.read_bytes()
    patch_sha256 = hashlib.sha256(patch).hexdigest()
    manifest_path = repository / ".phiagent-patch-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("source_revision") == revision
            and manifest.get("patch_sha256") == patch_sha256
        ):
            reverse = _run_patch(repository, patch, "--dry-run", "--reverse")
            if reverse.returncode == 0:
                print(json.dumps(manifest, indent=2, sort_keys=True))
                return 0
        raise ValueError("existing BWM patch manifest does not match source contents")

    check = _run_patch(repository, patch, "--dry-run", "--forward")
    already_applied = False
    if check.returncode != 0:
        reverse = _run_patch(repository, patch, "--dry-run", "--reverse")
        if reverse.returncode != 0:
            raise RuntimeError(
                "BWM patch preflight failed in both directions:\n"
                + check.stdout.decode(errors="replace")
            )
        already_applied = True
    if not already_applied:
        applied = _run_patch(repository, patch, "--forward")
        if applied.returncode != 0:
            raise RuntimeError("BWM patch failed:\n" + applied.stdout.decode(errors="replace"))
    manifest = {
        "schema_version": "1.0.0",
        "source_revision": revision,
        "patch": str(patch_path),
        "patch_sha256": patch_sha256,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "preserve distinct packed video and action frame offsets",
        "already_applied_on_entry": already_applied,
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
