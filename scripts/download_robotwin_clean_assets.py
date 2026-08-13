#!/usr/bin/env python3
"""Download pinned RoboTwin 2.0 embodiment/object assets for clean tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import socket
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


DATASET_ID = "TianxingChen/RoboTwin2.0"
DATASET_REVISION = "c15cc97be71e35244b6605d2d84c187f8565cc4d"
DATASET_LICENSE = "mit"
FILES = {
    "embodiments.zip": (
        219_859_313,
        "6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6",
    ),
    "objects.zip": (
        3_737_778_549,
        "6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27",
    ),
}


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


def _url(name: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{DATASET_REVISION}/{name}"
    )


def _download(root: Path, name: str) -> dict[str, object]:
    expected_bytes, expected_sha256 = FILES[name]
    destination = root / name
    partial = destination.with_suffix(".zip.partial")
    request = urllib.request.Request(_url(name), headers={"User-Agent": "PhiAgent/0"})
    try:
        with (
            urllib.request.urlopen(request, timeout=600) as response,
            partial.open("wb") as handle,
        ):
            while block := response.read(4 * 1024 * 1024):
                handle.write(block)
    except Exception as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download RoboTwin asset {name}: {error}") from error
    if partial.stat().st_size != expected_bytes:
        raise ValueError(
            f"RoboTwin asset byte mismatch for {name}: "
            f"{partial.stat().st_size} != {expected_bytes}"
        )
    partial.replace(destination)
    actual_sha256 = _sha256(destination)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"RoboTwin asset hash mismatch for {name}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return {
        "name": name,
        "path": str(destination),
        "bytes": expected_bytes,
        "sha256": actual_sha256,
        "source_url": _url(name),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extract", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RoboTwin assets: {output}")
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "status": "STARTED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
            "files": {
                name: {"bytes": size, "sha256": sha}
                for name, (size, sha) in FILES.items()
            },
            "extract": args.extract,
            "scope": "demo_clean tasks; randomized background textures intentionally omitted",
        },
    )
    archives = output / "archives"
    archives.mkdir()
    records = [_download(archives, name) for name in FILES]
    extracted = []
    if args.extract:
        assets = output / "assets"
        assets.mkdir()
        for record in records:
            archive = Path(str(record["path"]))
            with zipfile.ZipFile(archive) as handle:
                members = handle.namelist()
                handle.extractall(assets)
            extracted.append(
                {
                    "archive": archive.name,
                    "member_count": len(members),
                    "destination": str(assets),
                }
            )
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "PARTIAL",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
        "files": records,
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "extracted": extracted,
        "limitations": [
            "Background textures are omitted; only demo_clean tasks are eligible.",
            "Asset availability does not establish RoboTwin runtime reproducibility.",
        ],
    }
    _write_json(output / "manifest.json", result)
    (output / "download.log").write_text(
        f"downloaded {len(records)} files / {result['total_bytes']} bytes\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

