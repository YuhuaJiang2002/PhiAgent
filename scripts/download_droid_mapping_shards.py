#!/usr/bin/env python3
"""Download exact DROID-100 shards needed to recover raw held-out lineage."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import platform
import re
import shlex
import socket
import struct
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


FILES = {
    "r2d2_faceblur-train.tfrecord-00007-of-00031": (
        41_133_699,
        "HHQPxysW9Yox9u2BLKMG+g==",
    ),
    "r2d2_faceblur-train.tfrecord-00019-of-00031": (
        14_490_316,
        "HxPlxzbPY2fK4OPqUVXpuA==",
    ),
    "r2d2_faceblur-train.tfrecord-00023-of-00031": (
        80_456_132,
        "VyxJuwp2XkKLQ85ancTOfw==",
    ),
}
TARGETS = {
    21: ("r2d2_faceblur-train.tfrecord-00007-of-00031", 2),
    60: ("r2d2_faceblur-train.tfrecord-00019-of-00031", 0),
    77: ("r2d2_faceblur-train.tfrecord-00023-of-00031", 5),
}


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


def _url(name: str) -> str:
    return f"https://storage.googleapis.com/gresearch/robotics/droid_100/1.0.0/{name}"


def partial_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".partial")


def _download(root: Path, name: str) -> dict[str, object]:
    expected_bytes, expected_md5 = FILES[name]
    destination = root / name
    partial = partial_path(destination)
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
        raise RuntimeError(f"failed to download DROID mapping shard {name}: {error}") from error
    if partial.stat().st_size != expected_bytes:
        raise ValueError(f"DROID mapping shard byte mismatch: {name}")
    partial.replace(destination)
    actual_md5 = _md5_base64(destination)
    if actual_md5 != expected_md5:
        raise ValueError(f"DROID mapping shard MD5 mismatch: {name}")
    return {
        "name": name,
        "path": str(destination),
        "bytes": expected_bytes,
        "gcs_md5_base64": actual_md5,
        "sha256": _sha256(destination),
        "source_url": _url(name),
    }


def tfrecord_at(path: Path, offset: int) -> bytes:
    if offset < 0:
        raise ValueError("TFRecord offset must be non-negative")
    with path.open("rb") as handle:
        for index in range(offset + 1):
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"TFRecord ended before record {offset}: {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            if len(handle.read(4)) != 4:
                raise ValueError("TFRecord length CRC is truncated")
            payload = handle.read(length)
            if len(payload) != length or len(handle.read(4)) != 4:
                raise ValueError("TFRecord payload or CRC is truncated")
            if index == offset:
                return payload
    raise AssertionError("unreachable")


def printable_metadata(payload: bytes) -> tuple[str, ...]:
    strings = {
        match.decode("utf-8", errors="ignore").strip()
        for match in re.findall(rb"[\x20-\x7e]{4,}", payload)
    }
    return tuple(
        sorted(
            value
            for value in strings
            if any(
                token in value.lower()
                for token in (
                    "/",
                    "record",
                    "task",
                    "scene",
                    "success",
                    "uuid",
                    "droid",
                    "iprl",
                    "grasp",
                    "drawer",
                    "bowl",
                    "pot",
                )
            )
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DROID mapping shards: {output}")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    shards = output / "shards"
    shards.mkdir()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(lambda name: _download(shards, name), FILES))
    mappings = {}
    for episode, (name, offset) in TARGETS.items():
        payload = tfrecord_at(shards / name, offset)
        payload_path = output / f"episode-{episode:03d}-sequence-example.bin"
        payload_path.write_bytes(payload)
        mappings[str(episode)] = {
            "shard": name,
            "record_offset": offset,
            "payload": str(payload_path),
            "payload_bytes": len(payload),
            "payload_sha256": _sha256(payload_path),
            "printable_metadata": printable_metadata(payload),
        }
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "PARTIAL",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "files": sorted(records, key=lambda item: str(item["name"])),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "mappings": mappings,
        "claim_boundary": (
            "Shard/record positions follow the official TFDS shardLengths metadata. "
            "Printable protobuf strings are diagnostic until recording_folderpath and "
            "file_path are decoded and cross-checked against LeRobot video hashes."
        ),
    }
    _write_json(output / "manifest.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
