from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.download_http_ranges import (
    download,
    probe_range_object,
    split_interval,
    split_ranges,
)


class _RangeHandler(BaseHTTPRequestHandler):
    payload = bytes(range(251)) * 4096

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_HEAD(self) -> None:
        self.send_response(200)
        size = "0" if self.path == "/head-without-size.bin" else str(len(self.payload))
        self.send_header("Content-Length", size)
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:
        value = self.headers.get("Range", "")
        if not value.startswith("bytes="):
            self.send_error(400)
            return
        start_text, end_text = value.removeprefix("bytes=").split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
        chunk = self.payload[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.payload)}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(chunk)


@pytest.fixture
def range_server() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/object.bin"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_split_ranges_cover_every_byte_once() -> None:
    ranges = split_ranges(11, 4)

    assert ranges == [(0, 2), (3, 5), (6, 8), (9, 10)]
    assert split_interval(5, 15, 4) == [(5, 7), (8, 10), (11, 13), (14, 15)]


def test_range_probe_falls_back_when_head_omits_object_size(range_server: str) -> None:
    probe = probe_range_object(
        range_server.replace("/object.bin", "/head-without-size.bin"),
        timeout_seconds=10,
    )

    assert probe["size"] == len(_RangeHandler.payload)
    assert probe["accept_ranges"] == "bytes"


def test_parallel_range_download_reconstructs_exact_object(
    tmp_path: Path,
    range_server: str,
) -> None:
    output = tmp_path / "object.bin"

    result = download(
        url=range_server,
        output=output,
        expected_size=len(_RangeHandler.payload),
        connections=7,
        retries=1,
        timeout_seconds=10,
        chunk_bytes=8192,
    )

    assert result["bytes"] == len(_RangeHandler.payload)
    assert hashlib.sha256(output.read_bytes()).digest() == hashlib.sha256(
        _RangeHandler.payload
    ).digest()
    with pytest.raises(FileExistsError):
        download(
            url=range_server,
            output=output,
            expected_size=len(_RangeHandler.payload),
            connections=2,
            retries=0,
            timeout_seconds=10,
            chunk_bytes=8192,
        )


def test_parallel_range_download_resumes_an_existing_prefix_and_checks_hash(
    tmp_path: Path,
    range_server: str,
) -> None:
    output = tmp_path / "resumed.bin"
    prefix = 123_457
    output.write_bytes(_RangeHandler.payload[:prefix])
    expected_hash = hashlib.sha256(_RangeHandler.payload).hexdigest()

    result = download(
        url=range_server,
        output=output,
        expected_size=len(_RangeHandler.payload),
        connections=5,
        retries=1,
        timeout_seconds=10,
        chunk_bytes=8192,
        resume_existing_prefix=True,
        expected_sha256=expected_hash,
        skip_remote_probe=True,
    )

    assert result["resumed_prefix_bytes"] == prefix
    assert result["remote_probe_skipped"] is True
    assert result["sha256"] == expected_hash
    assert output.read_bytes() == _RangeHandler.payload

    with pytest.raises(ValueError, match="requires an expected SHA-256"):
        download(
            url=range_server,
            output=tmp_path / "unverified.bin",
            expected_size=len(_RangeHandler.payload),
            connections=1,
            retries=0,
            timeout_seconds=10,
            chunk_bytes=8192,
            skip_remote_probe=True,
        )


def test_explicit_range_repair_preserves_completed_bytes_and_restores_hash(
    tmp_path: Path,
    range_server: str,
) -> None:
    output = tmp_path / "repair.bin"
    damaged = bytearray(_RangeHandler.payload)
    repairs = [(111, 9999), (500_000, 600_000)]
    for start, end in repairs:
        damaged[start : end + 1] = b"\0" * (end - start + 1)
    output.write_bytes(damaged)
    expected_hash = hashlib.sha256(_RangeHandler.payload).hexdigest()

    result = download(
        url=range_server,
        output=output,
        expected_size=len(_RangeHandler.payload),
        connections=2,
        retries=1,
        timeout_seconds=10,
        chunk_bytes=8192,
        expected_sha256=expected_hash,
        skip_remote_probe=True,
        explicit_ranges=repairs,
    )

    assert result["explicit_ranges"] == [[111, 9999], [500_000, 600_000]]
    assert result["effective_download_bytes"] == sum(
        end - start + 1 for start, end in repairs
    )
    assert output.read_bytes() == _RangeHandler.payload
