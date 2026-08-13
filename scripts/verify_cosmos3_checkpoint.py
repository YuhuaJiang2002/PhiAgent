#!/usr/bin/env python3
"""Verify a pinned Cosmos3 checkpoint without importing model dependencies.

The verifier treats Hugging Face's safetensors index ``metadata.total_size`` as
the aggregate tensor-data byte contract.  Safetensors file sizes are larger by
their 8-byte prefix and JSON header, so comparing raw file sizes to that field
is incorrect.  The verifier validates every indexed file, its parsed header
offsets, aggregate tensor-data size, and small configuration-file hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import shlex
import socket
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SMALL_FILE_HASH_LIMIT = 64 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-missing-revision-marker", action="store_true")
    parser.add_argument(
        "--commit-revision-marker",
        action="store_true",
        help="atomically persist the expected revision only after verification passes",
    )
    parser.add_argument(
        "--completion-marker",
        type=Path,
        help="optional JSON completion marker written after report and revision commit",
    )
    parser.add_argument(
        "--require-file",
        action="append",
        default=["config.json"],
        help="Checkpoint-relative file that must exist; may be repeated.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unresolved"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_status": run("status", "--short", "--untracked-files=no"),
    }


def commit_verified_completion(
    checkpoint: Path,
    expected_revision: str,
    report_path: Path,
    completion_marker: Path | None,
) -> None:
    root = checkpoint.expanduser().resolve()
    report = report_path.expanduser().resolve()
    if not report.is_file():
        raise ValueError(f"verification report is missing: {report}")
    payload = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("status") != "WORKING" or payload.get("revision") != expected_revision:
        raise ValueError("verification report is not a matching WORKING result")
    revision_marker = root / ".phiagent-model-revision"
    _write_text_atomic(revision_marker, expected_revision + "\n")
    if revision_marker.read_text(encoding="utf-8").strip() != expected_revision:
        raise RuntimeError("revision marker did not persist the verified revision")
    if completion_marker is not None:
        _write_json(
            completion_marker.expanduser().resolve(),
            {
                "status": "WORKING",
                "checkpoint": str(root),
                "revision": expected_revision,
                "verification_report": str(report),
                "verification_report_sha256": _sha256(report),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )


def _read_index(path: Path) -> tuple[set[str], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    total_size = payload.get("metadata", {}).get("total_size")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"index has no non-empty weight_map: {path}")
    if not isinstance(total_size, int) or total_size <= 0:
        raise ValueError(f"index has no positive metadata.total_size: {path}")
    filenames = {str(filename) for filename in weight_map.values()}
    if any(Path(filename).is_absolute() or ".." in Path(filename).parts for filename in filenames):
        raise ValueError(f"index contains an unsafe weight path: {path}")
    return filenames, total_size


def _validate_safetensors_header(path: Path) -> dict[str, int]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError(f"safetensors file is shorter than its header: {path}")
    header_size = struct.unpack("<Q", raw)[0]
    if header_size <= 1 or 8 + header_size >= size:
        raise ValueError(
            f"invalid safetensors header boundary {header_size} for {size} bytes: {path}"
        )
    with path.open("rb") as handle:
        handle.seek(8)
        raw_header = handle.read(header_size)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    offsets: list[tuple[int, int]] = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict):
            raise ValueError(f"invalid safetensors tensor entry {name!r}: {path}")
        data_offsets = tensor.get("data_offsets")
        if (
            not isinstance(data_offsets, list)
            or len(data_offsets) != 2
            or any(not isinstance(value, int) for value in data_offsets)
        ):
            raise ValueError(f"invalid safetensors data offsets for {name!r}: {path}")
        start, end = data_offsets
        if start < 0 or end < start:
            raise ValueError(f"invalid safetensors data range for {name!r}: {path}")
        offsets.append((start, end))
    if not offsets:
        raise ValueError(f"safetensors header contains no tensors: {path}")
    tensor_data_size = size - 8 - header_size
    if max(end for _, end in offsets) != tensor_data_size:
        raise ValueError(
            f"safetensors data boundary mismatch for {path}: "
            f"header ends at {max(end for _, end in offsets)}, file has {tensor_data_size}"
        )
    return {
        "header_bytes": header_size,
        "tensor_data_bytes": tensor_data_size,
        "tensor_count": len(offsets),
    }


def verify_checkpoint(
    checkpoint: Path,
    expected_revision: str,
    required_files: list[str],
    *,
    allow_missing_revision_marker: bool = False,
) -> dict[str, Any]:
    root = checkpoint.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"checkpoint directory is missing: {root}")
    marker = root / ".phiagent-model-revision"
    if marker.is_file():
        actual_revision = marker.read_text(encoding="utf-8").strip()
        if actual_revision != expected_revision:
            raise ValueError(
                f"checkpoint revision mismatch: expected {expected_revision}, got {actual_revision}"
            )
        revision_source = "persisted marker"
    elif allow_missing_revision_marker:
        actual_revision = expected_revision
        revision_source = "expected revision argument; completion marker pending"
    else:
        raise ValueError(f"checkpoint revision marker is missing: {marker}")

    required_inventory: list[dict[str, Any]] = []
    for relative in sorted(set(required_files)):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe required file path: {relative}")
        path = root / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required checkpoint file is missing or empty: {path}")
        item: dict[str, Any] = {
            "path": relative_path.as_posix(),
            "size_bytes": path.stat().st_size,
        }
        if path.stat().st_size <= SMALL_FILE_HASH_LIMIT:
            item["sha256"] = _sha256(path)
        required_inventory.append(item)

    index_paths = sorted(
        path
        for path in root.rglob("*.safetensors.index.json")
        if ".cache" not in path.relative_to(root).parts
    )
    if not index_paths:
        raise ValueError(f"no safetensors index found below {root}")

    indexes: list[dict[str, Any]] = []
    for index_path in index_paths:
        filenames, expected_size = _read_index(index_path)
        weights: list[dict[str, Any]] = []
        actual_tensor_data_size = 0
        actual_file_size = 0
        for filename in sorted(filenames):
            weight = index_path.parent / filename
            if not weight.is_file() or weight.stat().st_size == 0:
                raise ValueError(f"indexed weight is missing or empty: {weight}")
            size = weight.stat().st_size
            header = _validate_safetensors_header(weight)
            actual_file_size += size
            actual_tensor_data_size += header["tensor_data_bytes"]
            weights.append(
                {
                    "path": weight.relative_to(root).as_posix(),
                    "size_bytes": size,
                    "safetensors_header_bytes": header["header_bytes"],
                    "tensor_data_bytes": header["tensor_data_bytes"],
                    "tensor_count": header["tensor_count"],
                }
            )
        if actual_tensor_data_size != expected_size:
            raise ValueError(
                f"indexed weight size mismatch for {index_path}: "
                f"expected {expected_size} tensor-data bytes, "
                f"got {actual_tensor_data_size}"
            )
        indexes.append(
            {
                "path": index_path.relative_to(root).as_posix(),
                "sha256": _sha256(index_path),
                "expected_total_size_bytes": expected_size,
                "actual_total_size_bytes": actual_tensor_data_size,
                "actual_total_file_size_bytes": actual_file_size,
                "weights": weights,
            }
        )

    return {
        "status": "WORKING",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(root),
        "revision": actual_revision,
        "revision_source": revision_source,
        "required_files": required_inventory,
        "indexes": indexes,
        "limitations": [
            "Large weight payloads are validated against the pinned index tensor-data byte contract and parsed safetensors header offsets, not re-hashed in full.",
            "Checkpoint integrity does not establish model inference or task-video quality.",
        ],
    }


def main() -> int:
    args = _parser().parse_args()
    if args.completion_marker and not args.commit_revision_marker:
        raise ValueError("--completion-marker requires --commit-revision-marker")
    report = verify_checkpoint(
        args.checkpoint,
        args.expected_revision,
        list(args.require_file),
        allow_missing_revision_marker=args.allow_missing_revision_marker,
    )
    command = [sys.executable, *sys.argv]
    report.update(
        {
            "seed": 0,
            "determinism": "no stochastic operations; seed recorded as zero",
            "command": command,
            "command_shell": shlex.join(command),
            "git": _git_state(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("huggingface-hub",)
                if importlib.util.find_spec(name.replace("-", "_")) is not None
            },
        }
    )
    report_path = args.report.expanduser().resolve()
    _write_json(report_path, report)
    if args.commit_revision_marker:
        commit_verified_completion(
            args.checkpoint,
            args.expected_revision,
            report_path,
            args.completion_marker,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
