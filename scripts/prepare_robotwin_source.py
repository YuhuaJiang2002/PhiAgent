#!/usr/bin/env python3
"""Download pinned RoboTwin and XPolicyLab source archives with provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import socket
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROBOTWIN_COMMIT = "266f3aadf505a4f7fe9af0faa41a20f5f47cd123"
XPOLICYLAB_COMMIT = "c37109c500be67d0dea6b36bf7337bbd26e763cd"
ARCHIVES = {
    "robotwin": (
        f"https://codeload.github.com/RoboTwin-Platform/RoboTwin/tar.gz/{ROBOTWIN_COMMIT}"
    ),
    "xpolicylab": (
        f"https://codeload.github.com/XPolicyLab/XPolicyLab/tar.gz/{XPOLICYLAB_COMMIT}"
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


def _download(url: str, destination: Path) -> dict[str, object]:
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "PhiAgent/0"})
    try:
        with (
            urllib.request.urlopen(request, timeout=300) as response,
            partial.open("wb") as handle,
        ):
            while block := response.read(1024 * 1024):
                handle.write(block)
    except Exception as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download source archive {url}: {error}") from error
    if partial.stat().st_size == 0:
        raise ValueError(f"source archive is empty: {url}")
    partial.replace(destination)
    return {
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        roots = {Path(member.name).parts[0] for member in members if member.name}
        if len(roots) != 1:
            raise ValueError(f"archive does not contain one source root: {archive}")
        for member in members:
            resolved = (destination / member.name).resolve()
            if destination.resolve() not in resolved.parents and resolved != destination.resolve():
                raise ValueError(f"archive member escapes destination: {member.name}")
        handle.extractall(destination, members=members, filter="data")
    return destination / next(iter(roots))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--without-xpolicylab", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RoboTwin source: {output}")
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "status": "STARTED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "robotwin_commit": ROBOTWIN_COMMIT,
            "xpolicylab_commit": XPOLICYLAB_COMMIT,
            "archives": ARCHIVES,
            "include_xpolicylab": not args.without_xpolicylab,
        },
    )
    archives = output / "archives"
    archives.mkdir()
    selected_archives = {
        name: url
        for name, url in ARCHIVES.items()
        if name == "robotwin" or not args.without_xpolicylab
    }
    records = {
        name: _download(url, archives / f"{name}.tar.gz")
        for name, url in selected_archives.items()
    }
    staging = output / "staging"
    staging.mkdir()
    robotwin_root = _safe_extract(
        Path(str(records["robotwin"]["path"])), staging / "robotwin"
    )
    source = output / "source"
    robotwin_root.rename(source)
    if not args.without_xpolicylab:
        xpolicy_root = _safe_extract(
            Path(str(records["xpolicylab"]["path"])), staging / "xpolicylab"
        )
        xpolicy_destination = source / "XPolicyLab"
        if xpolicy_destination.exists():
            xpolicy_destination.rmdir()
        xpolicy_root.rename(xpolicy_destination)
    required = (
        source / "envs" / "_base_task.py",
        source / "envs" / "adjust_bottle.py",
        source / "scripts" / "collect_data.py",
        source / "scripts" / "test_render.py",
    )
    if not args.without_xpolicylab:
        required = (*required, source / "XPolicyLab" / "pyproject.toml")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"pinned RoboTwin source is incomplete: {missing}")
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "robotwin_commit": ROBOTWIN_COMMIT,
        "xpolicylab_commit": XPOLICYLAB_COMMIT,
        "xpolicylab_included": not args.without_xpolicylab,
        "archives": records,
        "source": str(source),
        "required_file_sha256": {
            str(path.relative_to(source)): _sha256(path) for path in required
        },
        "license_boundary": (
            "RoboTwin code is MIT. Downloaded datasets/assets retain their own manifest "
            "and are not inferred to share the code license."
        ),
        "limitations": (
            []
            if not args.without_xpolicylab
            else [
                "XPolicyLab is omitted; this source is eligible only for simulator "
                "render/replay preflight, not XPolicyLab dataset conversion."
            ]
        ),
    }
    _write_json(output / "manifest.json", result)
    (output / "prepare.log").write_text(
        f"prepared RoboTwin {ROBOTWIN_COMMIT} and XPolicyLab {XPOLICYLAB_COMMIT}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
