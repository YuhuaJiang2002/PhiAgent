"""Integrity checks for frozen, repository-local benchmark sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze_manifest(manifest_path: Path, *, repository_root: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.expanduser().resolve().read_text())
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "0.1.0":
        raise ValueError("unsupported freeze manifest")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("freeze manifest requires artifacts")
    results = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ValueError("freeze artifact must be an object")
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("freeze artifact paths must remain repository-relative")
        root = repository_root.resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("freeze artifact resolves outside the repository")
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        digest = _sha256(path) if exists else None
        expected_size = int(item["bytes"])
        expected_digest = str(item["sha256"])
        results.append(
            {
                "path": str(relative),
                "exists": exists,
                "size_match": size == expected_size,
                "sha256_match": digest == expected_digest,
                "actual_bytes": size,
                "actual_sha256": digest,
            }
        )
    return {
        "suite": str(payload["suite"]),
        "artifact_count": len(results),
        "valid": all(
            row["exists"] and row["size_match"] and row["sha256_match"]
            for row in results
        ),
        "artifacts": results,
    }
