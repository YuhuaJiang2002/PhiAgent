#!/usr/bin/env python3
"""Build a compact, action-aligned WorldArena BWM test-only transfer bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import shlex
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def bundled_row(
    row: dict[str, Any],
    compact_meta: dict[str, Any],
) -> dict[str, Any]:
    """Rewrite only the video interval after deterministic clip materialization."""

    result = copy.deepcopy(row)
    length = int(result["length"])
    original_start = int(result["video"]["start_frame"])
    original_end = int(result["video"]["end_frame"])
    if original_end - original_start + 1 != length:
        raise ValueError("compiled video interval does not match row length")
    if (
        int(compact_meta.get("source_start_frame", -1)) != original_start
        or int(compact_meta.get("source_end_frame", -1)) != original_end
        or str(compact_meta.get("source_episode")) != str(result["source_episode"])
    ):
        raise ValueError("compact clip lineage does not match compiled test row")
    result["source_video_window"] = {
        "start_frame": original_start,
        "end_frame": original_end,
    }
    result["start_frame"] = 0
    result["end_frame"] = length - 1
    result["video"]["start_frame"] = 0
    result["video"]["end_frame"] = length - 1
    result["video"]["transfer_encoding"] = "h264-crf27-384x288"
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-root", type=Path, required=True)
    parser.add_argument("--compact-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    compiled = args.compiled_root.expanduser().resolve()
    compact = args.compact_cache.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite BWM test bundle: {output}")
    compiled_manifest_path = compiled / "manifest.json"
    compact_manifest_path = compact / "manifest.json"
    for path in (compiled_manifest_path, compact_manifest_path):
        if not path.is_file():
            raise ValueError(f"required source manifest is missing: {path}")
    compiled_manifest = json.loads(compiled_manifest_path.read_text())
    compact_manifest = json.loads(compact_manifest_path.read_text())
    if (
        compiled_manifest.get("status") != "completed"
        or compact_manifest.get("status") != "completed"
    ):
        raise ValueError("compiled dataset and compact cache must both be completed")
    rows = [
        json.loads(line)
        for line in (compiled / "test.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) < 20:
        raise ValueError("BWM SOTA test bundle requires at least 20 rows")
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    rewritten = []
    files = []
    for row in rows:
        task = str(row["task"])
        source_name = str(row["source_episode"]).split("/", maxsplit=1)[1]
        compact_root = compact / task / source_name
        meta_path = compact_root / "meta.json"
        video_path = compact_root / "cam_high.mp4"
        if not meta_path.is_file() or not video_path.is_file():
            raise ValueError(f"compact clip is missing for {row['source_episode']}")
        compact_meta = json.loads(meta_path.read_text())
        rewritten.append(bundled_row(row, compact_meta))
        video_destination = output / str(row["video"]["data"])
        action_source = compiled / str(row["action"]["data"])
        action_destination = output / str(row["action"]["data"])
        for source, destination, role in (
            (video_path, video_destination, "video"),
            (action_source, action_destination, "action"),
        ):
            if not source.is_file():
                raise ValueError(f"bundle source is missing: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            files.append(
                {
                    "role": role,
                    "path": str(destination.relative_to(output)),
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )
    action_stats = output / "action-stat.json"
    shutil.copy2(compiled / "action-stat.json", action_stats)
    test_metadata = output / "test.jsonl"
    test_metadata.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rewritten)
    )
    files.extend(
        {
            "role": role,
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for role, path in (
            ("action_stats", action_stats),
            ("test_metadata", test_metadata),
        )
    )
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "compiled_manifest": str(compiled_manifest_path),
        "compiled_manifest_sha256": _sha256(compiled_manifest_path),
        "compact_manifest": str(compact_manifest_path),
        "compact_manifest_sha256": _sha256(compact_manifest_path),
        "test_rows": len(rewritten),
        "independent_source_episodes": len(
            {str(row["source_episode"]) for row in rewritten}
        ),
        "files": sorted(files, key=lambda item: str(item["path"])),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "claim_boundary": (
            "The bundle changes only transfer encoding and video-relative indices. "
            "Action windows, physical episode IDs, splits, and reference lineage remain frozen."
        ),
    }
    _write_json(output / "manifest.json", result)
    (output / "bundle.log").write_text(
        f"rows={result['test_rows']} bytes={result['total_bytes']}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

