#!/usr/bin/env python3
"""Build an experiment-owned RoboTwin source/assets runtime view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


ROBOTWIN_COMMIT = "266f3aadf505a4f7fe9af0faa41a20f5f47cd123"
ASSET_REVISION = "c15cc97be71e35244b6605d2d84c187f8565cc4d"


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    source_manifest_path = args.source_manifest.expanduser().resolve()
    asset_manifest_path = args.asset_manifest.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RoboTwin runtime: {output}")
    for path in (source_manifest_path, asset_manifest_path):
        if not path.is_file():
            raise ValueError(f"required RoboTwin manifest is missing: {path}")
    source_manifest = json.loads(source_manifest_path.read_text())
    asset_manifest = json.loads(asset_manifest_path.read_text())
    if (
        source_manifest.get("status") != "WORKING"
        or source_manifest.get("robotwin_commit") != ROBOTWIN_COMMIT
        or asset_manifest.get("status") != "WORKING"
        or asset_manifest.get("dataset_revision") != ASSET_REVISION
    ):
        raise ValueError("RoboTwin source or asset identity is not pinned and working")
    source = Path(str(source_manifest["source"])).expanduser().resolve()
    asset_root = asset_manifest_path.parent / "assets"
    objects = asset_root / "objects"
    embodiments = asset_root / "embodiments"
    for path in (source, objects, embodiments):
        if not path.is_dir():
            raise ValueError(f"RoboTwin runtime input is missing: {path}")
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    runtime_source = output / "source"
    shutil.copytree(source, runtime_source, symlinks=True)
    runtime_assets = runtime_source / "assets"
    runtime_assets.mkdir(exist_ok=True)
    for name, target in (("objects", objects), ("embodiments", embodiments)):
        destination = runtime_assets / name
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"RoboTwin runtime source already contains {name}")
        destination.symlink_to(target, target_is_directory=True)
    required = (
        runtime_source / "envs" / "adjust_bottle.py",
        runtime_source / "scripts" / "test_render.py",
        runtime_source / "scripts" / "collect_data.py",
        runtime_assets / "objects",
        runtime_assets / "embodiments",
    )
    if any(not path.exists() for path in required):
        raise ValueError("RoboTwin runtime view is incomplete")
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "PARTIAL",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "robotwin_commit": ROBOTWIN_COMMIT,
        "asset_revision": ASSET_REVISION,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "asset_manifest": str(asset_manifest_path),
        "asset_manifest_sha256": _sha256(asset_manifest_path),
        "runtime_source": str(runtime_source),
        "asset_links": {
            "objects": {
                "path": str(runtime_assets / "objects"),
                "target": os.readlink(runtime_assets / "objects"),
            },
            "embodiments": {
                "path": str(runtime_assets / "embodiments"),
                "target": os.readlink(runtime_assets / "embodiments"),
            },
        },
        "limitations": [
            "XPolicyLab is absent in the render-only source snapshot.",
            "Background textures are absent; only demo_clean tasks are eligible.",
            "Runtime reproducibility still requires a real render preflight.",
        ],
    }
    _write_json(output / "manifest.json", result)
    (output / "build.log").write_text("built pinned render-only RoboTwin runtime view\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

