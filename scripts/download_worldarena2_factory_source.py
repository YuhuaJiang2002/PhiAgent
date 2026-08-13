#!/usr/bin/env python3
"""Download a pinned, hash-indexed WorldArena2.0 source subset."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from prepare_worldarena2_bwm_factory_data import (
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    HF_ROOT,
    _download,
    _sha256,
    _task_episodes,
    _write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--source-revision", default=DATASET_REVISION)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"source cache already exists: {output}")
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "downloading",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "pinned_worldarena2_factory_source_cache",
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "dataset_id": DATASET_ID,
        "source_revision": args.source_revision,
        "license": DATASET_LICENSE,
        "tasks": list(args.task),
        "episodes_per_task": args.episodes_per_task,
        "files": [],
    }
    _write_json(manifest_path, manifest)
    try:
        if args.source_revision != DATASET_REVISION:
            raise ValueError(f"source revision must equal pinned {DATASET_REVISION}")
        if args.episodes_per_task <= 0:
            raise ValueError("episodes-per-task must be positive")
        if len(args.task) != len(set(args.task)):
            raise ValueError("tasks must be unique")
        files = []
        for task in args.task:
            for episode_name in _task_episodes(task, args.episodes_per_task):
                for name in ("meta.json", "episode.hdf5", "cam_high.mp4"):
                    relative = Path(task) / episode_name / name
                    url = (
                        f"{HF_ROOT}/resolve/{args.source_revision}/{task}/"
                        f"{episode_name}/{name}"
                    )
                    destination = output / relative
                    _download(url, destination)
                    files.append(
                        {
                            "path": str(relative),
                            "bytes": destination.stat().st_size,
                            "sha256": _sha256(destination),
                            "source_uri": url,
                        }
                    )
        manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "files": files,
                "file_count": len(files),
                "total_bytes": sum(int(item["bytes"]) for item in files),
            }
        )
        _write_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
