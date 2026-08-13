#!/usr/bin/env python3
"""Download calibration-critical raw files for lineage-verified DROID episodes."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib.metadata
import json
import platform
import shlex
import socket
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUCKET = "gresearch"
OBJECT_ROOT = "robotics/droid_raw"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--git-commit")
    parser.add_argument("--git-branch")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode("ascii")


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def partial_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".partial")


def _git_state(
    commit_override: str | None = None,
    branch_override: str | None = None,
) -> dict[str, object]:
    if (commit_override is None) != (branch_override is None):
        raise ValueError("git-commit and git-branch must be provided together")
    if commit_override is not None:
        if len(commit_override) != 40 or any(
            character not in "0123456789abcdef" for character in commit_override
        ):
            raise ValueError("git-commit must be a lowercase 40-character SHA-1")
        return {
            "commit": commit_override,
            "branch": branch_override,
            "dirty": None,
            "status_porcelain": None,
            "resolution": "explicit source-worktree snapshot",
            "download_script_sha256": _sha256(Path(__file__).resolve()),
        }

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = run("status", "--porcelain=v1")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "resolution": "local Git worktree",
        "download_script_sha256": _sha256(Path(__file__).resolve()),
    }


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in ("certifi", "setuptools"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def _object_prefix(raw_gcs_prefix: str) -> str:
    normalized = raw_gcs_prefix.strip("/")
    if not normalized or ".." in normalized.split("/"):
        raise ValueError("raw GCS prefix must be a non-empty relative path")
    return f"{OBJECT_ROOT}/{normalized}/"


def _listing_url(raw_gcs_prefix: str, page_token: str | None = None) -> str:
    query = {
        "prefix": _object_prefix(raw_gcs_prefix),
        "fields": "items(name,size,md5Hash,generation,mediaLink),nextPageToken",
    }
    if page_token:
        query["pageToken"] = page_token
    return (
        f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?"
        f"{urllib.parse.urlencode(query)}"
    )


def _list_objects(raw_gcs_prefix: str) -> tuple[list[dict[str, str]], list[str]]:
    objects: list[dict[str, str]] = []
    urls = []
    token = None
    while True:
        url = _listing_url(raw_gcs_prefix, token)
        urls.append(url)
        request = urllib.request.Request(url, headers={"User-Agent": "PhiAgent/0"})
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read())
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise ValueError("GCS listing response has no items list")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("GCS listing item is not an object")
            objects.append({str(key): str(value) for key, value in item.items()})
        token = payload.get("nextPageToken")
        if not token:
            break
        if not isinstance(token, str):
            raise ValueError("GCS listing page token is not a string")
    return objects, urls


def select_calibration_objects(
    objects: list[dict[str, str]],
    raw_gcs_prefix: str,
) -> list[dict[str, str]]:
    prefix = _object_prefix(raw_gcs_prefix)
    selected = []
    relatives = set()
    for item in objects:
        name = item.get("name")
        if not isinstance(name, str) or not name.startswith(prefix):
            raise ValueError("GCS object is outside the requested DROID prefix")
        relative = name.removeprefix(prefix)
        keep = (
            relative == "trajectory.h5"
            or (
                relative.startswith("metadata_")
                and relative.endswith(".json")
                and "/" not in relative
            )
            or (
                relative.startswith("recordings/SVO/")
                and relative.endswith(".svo")
                and relative.count("/") == 2
            )
        )
        if not keep:
            continue
        if relative in relatives:
            raise ValueError(f"duplicate GCS object relative path: {relative}")
        for field in ("size", "md5Hash", "generation", "mediaLink"):
            if not item.get(field):
                raise ValueError(f"GCS object {name} is missing {field}")
        record = dict(item)
        record["relative_path"] = relative
        relatives.add(relative)
        selected.append(record)
    trajectories = [row for row in selected if row["relative_path"] == "trajectory.h5"]
    metadata = [
        row for row in selected if row["relative_path"].startswith("metadata_")
    ]
    svos = [
        row
        for row in selected
        if row["relative_path"].startswith("recordings/SVO/")
    ]
    if len(trajectories) != 1 or len(metadata) != 1 or len(svos) != 3:
        raise ValueError(
            "expected exactly one trajectory, one metadata JSON, and three SVO files"
        )
    return sorted(selected, key=lambda row: row["relative_path"])


def _download(episode_root: Path, item: dict[str, str]) -> dict[str, object]:
    destination = episode_root / "data" / item["relative_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"download destination already exists: {destination}")
    partial = partial_path(destination)
    request = urllib.request.Request(
        item["mediaLink"],
        headers={"User-Agent": "PhiAgent/0"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=600) as response,
            partial.open("wb") as handle,
        ):
            while block := response.read(4 * 1024 * 1024):
                handle.write(block)
    except (
        OSError,
        TimeoutError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"failed to download raw DROID object {item['name']}: {error}"
        ) from error
    expected_bytes = int(item["size"])
    if partial.stat().st_size != expected_bytes:
        raise ValueError(
            f"raw DROID byte mismatch for {item['relative_path']}: "
            f"{partial.stat().st_size} != {expected_bytes}"
        )
    partial.replace(destination)
    actual_md5 = _md5_base64(destination)
    if actual_md5 != item["md5Hash"]:
        raise ValueError(
            f"raw DROID MD5 mismatch for {item['relative_path']}: "
            f"{actual_md5} != {item['md5Hash']}"
        )
    return {
        "path": item["relative_path"],
        "bytes": expected_bytes,
        "gcs_generation": item["generation"],
        "gcs_md5_base64": actual_md5,
        "sha256": _sha256(destination),
        "source_object": item["name"],
        "source_url": item["mediaLink"],
    }


def _validate_config(payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("DROID raw lineage config must contain an object")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("DROID raw lineage config must contain episodes")
    indices = set()
    prefixes = set()
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("each DROID episode config must be an object")
        index = episode.get("episode_index")
        prefix = episode.get("raw_gcs_prefix")
        if not isinstance(index, int) or not isinstance(prefix, str):
            raise ValueError("episode index/prefix types are invalid")
        _object_prefix(prefix)
        if index in indices or prefix in prefixes:
            raise ValueError("episode indices and raw prefixes must be unique")
        indices.add(index)
        prefixes.add(prefix)
    return episodes


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"DROID raw lineage config is missing: {config_path}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DROID raw lineage run: {output}")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    config = json.loads(config_path.read_text())
    episodes = _validate_config(config)

    output.mkdir(parents=True)
    (output / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n"
    )
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_config": str(config_path),
            "source_config_sha256": _sha256(config_path),
            "workers": args.workers,
            "seed": args.seed,
            "seed_use": "recorded for reproducibility; download is deterministic",
            "rights_status": config.get("rights_status"),
            "git_commit": args.git_commit,
            "git_branch": args.git_branch,
        },
    )
    _write_json(
        output / "git-state.json",
        _git_state(args.git_commit, args.git_branch),
    )
    _write_json(output / "package-versions.json", _package_versions())
    log_path = output / "download.log"
    log_path.write_text(
        f"{datetime.now(timezone.utc).isoformat()} starting DROID raw download\n"
    )

    episode_results = []
    for episode in episodes:
        index = episode["episode_index"]
        raw_prefix = episode["raw_gcs_prefix"]
        episode_root = output / "episodes" / f"episode-{index:03d}"
        episode_root.mkdir(parents=True)
        objects, listing_urls = _list_objects(raw_prefix)
        if not objects:
            _write_json(
                episode_root / "gcs-listing.json",
                {
                    "raw_gcs_prefix": raw_prefix,
                    "listing_urls": listing_urls,
                    "all_objects": [],
                    "selected_objects": [],
                },
            )
            episode_manifest = {
                "schema_version": "1.0.0",
                "status": "BLOCKED",
                "honest_status": "BLOCKED",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "episode_index": index,
                "task": episode["task"],
                "raw_gcs_prefix": raw_prefix,
                "sequence_payload_sha256": episode["sequence_payload_sha256"],
                "exterior_assignment": episode["exterior_assignment"],
                "blocker": "no_objects_under_lineage_verified_public_gcs_prefix",
                "files": [],
                "file_count": 0,
                "total_bytes": 0,
                "rights_boundary": (
                    "The official DROID site has no verified dataset license. "
                    "Public raw availability is a separate blocker."
                ),
            }
            manifest_path = episode_root / "manifest.json"
            _write_json(manifest_path, episode_manifest)
            episode_results.append(
                {
                    "episode_index": index,
                    "status": "BLOCKED",
                    "blocker": episode_manifest["blocker"],
                    "manifest": str(manifest_path),
                    "manifest_sha256": _sha256(manifest_path),
                    "file_count": 0,
                    "total_bytes": 0,
                }
            )
            with log_path.open("a") as handle:
                handle.write(
                    f"{datetime.now(timezone.utc).isoformat()} episode={index} "
                    "blocked=no_public_objects\n"
                )
            continue
        selected = select_calibration_objects(objects, raw_prefix)
        _write_json(
            episode_root / "gcs-listing.json",
            {
                "raw_gcs_prefix": raw_prefix,
                "listing_urls": listing_urls,
                "all_objects": objects,
                "selected_objects": selected,
            },
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, len(selected))
        ) as executor:
            files = list(
                executor.map(lambda item: _download(episode_root, item), selected)
            )
        files.sort(key=lambda row: str(row["path"]))
        metadata_records = [
            row for row in files if str(row["path"]).startswith("metadata_")
        ]
        metadata_path = episode_root / "data" / str(metadata_records[0]["path"])
        metadata = json.loads(metadata_path.read_text())
        episode_manifest = {
            "schema_version": "1.0.0",
            "status": "WORKING",
            "honest_status": "BLOCKED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "episode_index": index,
            "task": episode["task"],
            "raw_gcs_prefix": raw_prefix,
            "sequence_payload_sha256": episode["sequence_payload_sha256"],
            "exterior_assignment": episode["exterior_assignment"],
            "metadata": metadata,
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(int(row["bytes"]) for row in files),
            "rights_boundary": (
                "The official DROID site has no verified dataset license. These "
                "files are retained only for internal technical calibration audit "
                "and are not claim-eligible training or redistribution evidence."
            ),
        }
        manifest_path = episode_root / "manifest.json"
        _write_json(manifest_path, episode_manifest)
        episode_results.append(
            {
                "episode_index": index,
                "status": "WORKING",
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "file_count": len(files),
                "total_bytes": episode_manifest["total_bytes"],
            }
        )
        with log_path.open("a") as handle:
            handle.write(
                f"{datetime.now(timezone.utc).isoformat()} episode={index} "
                f"bytes={episode_manifest['total_bytes']}\n"
            )

    completed_at = datetime.now(timezone.utc).isoformat()
    all_available = all(row["status"] == "WORKING" for row in episode_results)
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING" if all_available else "PARTIAL",
        "honest_status": "BLOCKED",
        "completed_at": completed_at,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "seed": args.seed,
        "source_config": str(config_path),
        "source_config_sha256": _sha256(config_path),
        "episodes": episode_results,
        "episode_count": len(episode_results),
        "working_episode_count": sum(
            row["status"] == "WORKING" for row in episode_results
        ),
        "blocked_episode_count": sum(
            row["status"] == "BLOCKED" for row in episode_results
        ),
        "file_count": sum(int(row["file_count"]) for row in episode_results),
        "total_bytes": sum(int(row["total_bytes"]) for row in episode_results),
        "rights_boundary": (
            "Download/calibration lineage can be WORKING while raw-data rights "
            "remain BLOCKED for training and redistribution."
        ),
    }
    _write_json(output / "manifest.json", result)
    with log_path.open("a") as handle:
        handle.write(
            f"{completed_at} completed files={result['file_count']} "
            f"status={result['status']}\n"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all_available else 2


if __name__ == "__main__":
    raise SystemExit(main())
