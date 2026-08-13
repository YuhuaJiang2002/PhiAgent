#!/usr/bin/env python3
"""Download one immutable HTTP object with validated parallel byte ranges."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--connections", type=int, default=16)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument(
        "--resume-existing-prefix",
        action="store_true",
        help="Treat the existing output bytes as a complete prefix and fetch only its tail.",
    )
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--skip-remote-probe",
        action="store_true",
        help="Trust expected size for a pinned URL; the final hash remains mandatory.",
    )
    return parser


def _probe_range_object_once(url: str, timeout_seconds: float) -> dict[str, str | int]:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        size = int(response.headers.get("Content-Length", "0"))
        accept_ranges = response.headers.get("Accept-Ranges", "").lower()
        result: dict[str, str | int] = {
            "url": response.url,
            "size": size,
            "accept_ranges": accept_ranges,
        }
    if size > 0 and accept_ranges == "bytes":
        return result

    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or "/" not in content_range:
            raise ValueError(
                "remote object did not provide a usable range probe: "
                f"HTTP {response.status}, Content-Range={content_range!r}"
            )
        total = int(content_range.rsplit("/", maxsplit=1)[1])
        response.read(1)
        return {
            "url": response.url,
            "size": total,
            "accept_ranges": "bytes",
        }


def probe_range_object(
    url: str,
    timeout_seconds: float,
    retries: int = 3,
) -> dict[str, str | int]:
    if retries < 0:
        raise ValueError("probe retries cannot be negative")
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            return _probe_range_object_once(url, timeout_seconds)
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            http.client.HTTPException,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"range probe failed after {retries + 1} attempts: {last_error}"
    ) from last_error


def split_ranges(size: int, connections: int) -> list[tuple[int, int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if connections <= 0:
        raise ValueError("connections must be positive")
    count = min(size, connections)
    span = (size + count - 1) // count
    return [
        (start, min(start + span - 1, size - 1))
        for start in range(0, size, span)
    ]


def split_interval(start: int, end: int, connections: int) -> list[tuple[int, int]]:
    if start < 0 or end < start or connections <= 0:
        raise ValueError("range interval and connection count are invalid")
    relative = split_ranges(end - start + 1, connections)
    return [(start + left, start + right) for left, right in relative]


def _pwrite_all(file_descriptor: int, payload: bytes, offset: int) -> None:
    written = 0
    while written < len(payload):
        count = os.pwrite(file_descriptor, payload[written:], offset + written)
        if count <= 0:
            raise OSError("pwrite returned no progress")
        written += count


def _download_range(
    *,
    url: str,
    file_descriptor: int,
    start: int,
    end: int,
    retries: int,
    timeout_seconds: float,
    chunk_bytes: int,
) -> dict[str, int]:
    expected = end - start + 1
    offset = start
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                if response.status != 206:
                    raise RuntimeError(
                        f"range {start}-{end} returned HTTP {response.status}, expected 206"
                    )
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-{end}/"):
                    raise RuntimeError(
                        f"range {offset}-{end} returned unexpected Content-Range "
                        f"{content_range!r}"
                    )
                while offset <= end:
                    payload = response.read(min(chunk_bytes, end - offset + 1))
                    if not payload:
                        break
                    _pwrite_all(file_descriptor, payload, offset)
                    offset += len(payload)
                observed = offset - start
                if observed != expected:
                    raise RuntimeError(
                        f"range {start}-{end} ended after {observed}/{expected} bytes"
                    )
                return {
                    "start": start,
                    "end": end,
                    "bytes": observed,
                    "attempts": attempt,
                }
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            http.client.HTTPException,
            RuntimeError,
        ) as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"range {start}-{end} failed after {retries + 1} attempts: {last_error}"
    ) from last_error


def download(
    *,
    url: str,
    output: Path,
    expected_size: int,
    connections: int,
    retries: int,
    timeout_seconds: float,
    chunk_bytes: int,
    resume_existing_prefix: bool = False,
    expected_sha256: str | None = None,
    skip_remote_probe: bool = False,
    explicit_ranges: list[tuple[int, int]] | None = None,
) -> dict[str, object]:
    if expected_size <= 0 or retries < 0 or timeout_seconds <= 0 or chunk_bytes <= 0:
        raise ValueError("size, timeout, and chunk size must be positive; retries cannot be negative")
    destination = output.expanduser().resolve()
    if destination.exists() and not resume_existing_prefix and not explicit_ranges:
        raise FileExistsError(f"refusing to overwrite HTTP range output: {destination}")
    if explicit_ranges and not destination.is_file():
        raise FileNotFoundError("explicit range repair requires an existing output file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if skip_remote_probe and not expected_sha256:
        raise ValueError("skipping the remote probe requires an expected SHA-256")
    probe = (
        {"url": url, "size": expected_size, "accept_ranges": "bytes"}
        if skip_remote_probe
        else probe_range_object(url, timeout_seconds, retries=retries)
    )
    if probe["size"] != expected_size:
        raise ValueError(
            f"remote object size {probe['size']} does not match expected {expected_size}"
        )
    if probe["accept_ranges"] != "bytes":
        raise ValueError(f"remote object does not advertise byte ranges: {probe}")

    prefix_bytes = destination.stat().st_size if destination.exists() else 0
    if prefix_bytes > expected_size:
        raise ValueError(
            f"existing prefix has {prefix_bytes} bytes, expected at most {expected_size}"
        )
    if explicit_ranges:
        ranges = sorted(explicit_ranges)
        if any(start < 0 or end < start or end >= expected_size for start, end in ranges):
            raise ValueError("explicit ranges must lie inside the expected output")
        if any(left[1] >= right[0] for left, right in zip(ranges, ranges[1:])):
            raise ValueError("explicit ranges must not overlap")
    else:
        ranges = (
            split_interval(prefix_bytes, expected_size - 1, connections)
            if prefix_bytes < expected_size
            else []
        )
    scheduled_bytes = sum(end - start + 1 for start, end in ranges)
    started = time.monotonic()
    flags = os.O_WRONLY if destination.exists() else os.O_CREAT | os.O_EXCL | os.O_WRONLY
    file_descriptor = os.open(destination, flags, 0o644)
    try:
        if not explicit_ranges:
            os.ftruncate(file_descriptor, expected_size)
        results = []
        with ThreadPoolExecutor(max_workers=max(1, len(ranges))) as executor:
            futures = [
                executor.submit(
                    _download_range,
                    url=url,
                    file_descriptor=file_descriptor,
                    start=start,
                    end=end,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    chunk_bytes=chunk_bytes,
                )
                for start, end in ranges
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                try:
                    print(json.dumps({"completed_range": result}), flush=True)
                except BrokenPipeError:
                    pass
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)

    elapsed = time.monotonic() - started
    observed_size = destination.stat().st_size
    if observed_size != expected_size:
        raise RuntimeError(
            f"downloaded file size {observed_size} does not match expected {expected_size}"
        )
    observed_sha256 = None
    if expected_sha256:
        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            for payload in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(payload)
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"download SHA-256 {observed_sha256} != expected {expected_sha256}"
            )
    return {
        "url": url,
        "resolved_url": probe["url"],
        "remote_probe_skipped": skip_remote_probe,
        "output": str(destination),
        "bytes": observed_size,
        "connections": len(ranges),
        "resumed_prefix_bytes": prefix_bytes if not explicit_ranges else None,
        "explicit_ranges": [list(row) for row in ranges] if explicit_ranges else None,
        "effective_download_bytes": scheduled_bytes,
        "ranges": sorted(results, key=lambda row: row["start"]),
        "elapsed_seconds": elapsed,
        "bytes_per_second": scheduled_bytes / max(elapsed, 1e-9),
        "sha256": observed_sha256,
    }


def main() -> int:
    args = _parser().parse_args()
    result = download(
        url=args.url,
        output=args.output,
        expected_size=args.expected_size,
        connections=args.connections,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        chunk_bytes=args.chunk_bytes,
        resume_existing_prefix=args.resume_existing_prefix,
        expected_sha256=args.expected_sha256,
        skip_remote_probe=args.skip_remote_probe,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
