#!/usr/bin/env python3
"""Compile BWM's three released demo episodes as a runtime-only smoke dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.adapters import BWM_REPOSITORY_COMMIT  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"smoke dataset already exists: {output}")
    marker = repository / ".phiagent-source-revision"
    if not marker.is_file() or marker.read_text().strip() != BWM_REPOSITORY_COMMIT:
        raise ValueError("BWM demo source does not match the pinned commit")
    source_metadata = repository / "demo" / "demo.jsonl"
    source_stats = repository / "demo" / "stat.json"
    rows = [json.loads(line) for line in source_metadata.read_text().splitlines() if line]
    if len(rows) != 3:
        raise ValueError(f"pinned BWM smoke dataset should contain three rows, found {len(rows)}")
    compiled = []
    for row in rows:
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        video = (repository / "demo" / str(row["video"])).resolve()
        action = (repository / "demo" / str(row["action"])).resolve()
        for path in (video, action):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"BWM smoke input is missing: {path}")
        compiled.append(
            {
                **row,
                "video": {"data": str(video), "start_frame": start, "end_frame": end},
                "action": {"data": str(action), "start_frame": start, "end_frame": end},
                "split": "runtime_smoke_only",
                "claim_boundary": "released BWM demo; never use for benchmark selection",
            }
        )
    output.mkdir(parents=True)
    (output / "train.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in compiled)
    )
    shutil.copy2(source_stats, output / "action-stat.json")
    manifest = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "purpose": "runtime_smoke_only",
        "source_revision": BWM_REPOSITORY_COMMIT,
        "episodes": len(compiled),
        "source_metadata_sha256": _sha256(source_metadata),
        "source_action_stats_sha256": _sha256(source_stats),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "The released demo may overlap BWM training data.",
            "A successful smoke train is not a benchmark or SOTA result.",
            "The videos are simulated RoboTwin scenes, not real-robot evidence."
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
