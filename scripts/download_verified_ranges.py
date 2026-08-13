#!/usr/bin/env python3
"""Download one large public artifact with resumable verified byte ranges.

The destination is only promoted from ``.partial`` after its exact size and
SHA-256 match the caller-provided contract.  Completed range markers make an
interrupted download resumable without trusting the presence of sparse bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=12)
    return parser


def plan_ranges(size: int, workers: int) -> tuple[tuple[int, int], ...]:
    """Partition inclusive byte ranges without gaps or overlaps."""
    if size <= 0 or workers <= 0:
        raise ValueError("size and workers must be positive")
    workers = min(size, workers)
    base, remainder = divmod(size, workers)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for index in range(workers):
        length = base + (1 if index < remainder else 0)
        ranges.append((cursor, cursor + length - 1))
        cursor += length
    if cursor != size:
        raise AssertionError("range planner did not cover the requested size")
    return tuple(ranges)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _download_range(
    *,
    url: str,
    file_descriptor: int,
    start: int,
    end: int,
    marker: Path,
    retries: int,
) -> dict[str, int]:
    if marker.is_file():
        return {"start": start, "end": end, "bytes": end - start + 1, "resumed": 1}

    import requests

    cursor = start
    failures = 0
    while cursor <= end:
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={cursor}-{end}"},
                stream=True,
                allow_redirects=True,
                timeout=(30, 120),
            )
            if response.status_code != 206:
                raise RuntimeError(f"expected HTTP 206, got {response.status_code}")
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {cursor}-"):
                raise RuntimeError(f"unexpected Content-Range: {content_range}")
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not block:
                    continue
                remaining = end - cursor + 1
                if len(block) > remaining:
                    block = block[:remaining]
                os.pwrite(file_descriptor, block, cursor)
                cursor += len(block)
            if cursor <= end:
                raise RuntimeError(f"range ended early at byte {cursor}")
        except Exception:
            failures += 1
            if failures > retries:
                raise
            time.sleep(min(30, 2**min(failures, 4)))
    marker.write_text(f"{start} {end}\n")
    return {"start": start, "end": end, "bytes": end - start + 1, "resumed": 0}


def main() -> int:
    args = _parser().parse_args()
    expected_sha = args.sha256.lower()
    if len(expected_sha) != 64 or any(character not in "0123456789abcdef" for character in expected_sha):
        raise ValueError("sha256 must be a lowercase 64-character hex digest")
    if args.size <= 0 or args.workers <= 0 or args.retries < 0:
        raise ValueError("invalid download settings")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        if output.stat().st_size == args.size and _sha256(output) == expected_sha:
            print(json.dumps({"output": str(output), "status": "already_verified"}))
            return 0
        raise FileExistsError(f"refusing to overwrite unverified destination: {output}")

    partial = output.with_suffix(output.suffix + ".partial")
    state_dir = output.parent / f".{output.name}.download-state"
    state_dir.mkdir(exist_ok=True)
    ranges = plan_ranges(args.size, args.workers)
    _write_json(
        state_dir / "contract.json",
        {
            "schema_version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "url": args.url,
            "output": str(output),
            "size": args.size,
            "sha256": expected_sha,
            "workers": len(ranges),
            "ranges": ranges,
        },
    )
    descriptor = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        os.ftruncate(descriptor, args.size)
        with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = {
                executor.submit(
                    _download_range,
                    url=args.url,
                    file_descriptor=descriptor,
                    start=start,
                    end=end,
                    marker=state_dir / f"segment-{index:03d}.done",
                    retries=args.retries,
                ): index
                for index, (start, end) in enumerate(ranges)
            }
            for future in as_completed(futures):
                result = future.result()
                print(json.dumps({"segment": futures[future], **result}), flush=True)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    actual_size = partial.stat().st_size
    actual_sha = _sha256(partial)
    verification = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "expected_size": args.size,
        "actual_size": actual_size,
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "verified": actual_size == args.size and actual_sha == expected_sha,
    }
    _write_json(state_dir / "verification.json", verification)
    if not verification["verified"]:
        raise RuntimeError(f"download verification failed: {verification}")
    partial.replace(output)
    print(json.dumps({"output": str(output), "status": "verified", **verification}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
