#!/usr/bin/env python3
"""Resume the pinned JoyAI DiT from a contiguous prefix and verify its full hash."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.joyai_video_edit import (  # noqa: E402
    JOYAI_DIT_BYTES,
    JOYAI_DIT_RELATIVE_PATH,
    JOYAI_LARGE_FILE_CONTRACT,
    JOYAI_MODELSCOPE_MODEL_ID,
    JOYAI_MODELSCOPE_MODEL_REVISION,
    sha256_file,
    write_json,
)
from scripts.download_http_ranges import download  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--existing-prefix", type=Path)
    parser.add_argument(
        "--repair-range",
        nargs=2,
        type=int,
        action="append",
        metavar=("START", "END"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def _git_state() -> dict[str, Any]:
    state: dict[str, Any] = {}
    for label, command in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        state[label] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    return state


def main() -> int:
    args = _parser().parse_args()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    prefix = args.existing_prefix.expanduser().resolve() if args.existing_prefix else None
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"JoyAI DiT resume experiment exists: {output}")
    if args.repair_range:
        if prefix is not None:
            raise ValueError("explicit repair ranges and an existing prefix are mutually exclusive")
    elif prefix is None or not prefix.is_file() or not 0 < prefix.stat().st_size < JOYAI_DIT_BYTES:
        raise ValueError("existing prefix must be a non-empty proper prefix of the JoyAI DiT")
    output.mkdir(parents=True)
    destination = checkpoint_root / JOYAI_DIT_RELATIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = JOYAI_LARGE_FILE_CONTRACT[JOYAI_DIT_RELATIVE_PATH][1]
    url = (
        "https://modelscope.cn/models/"
        f"{JOYAI_MODELSCOPE_MODEL_ID}/resolve/{JOYAI_MODELSCOPE_MODEL_REVISION}/"
        "dit/joyai_video_edit_dit_0804.pth"
    )
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "stage": "dit_tail_resume_started",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": {"executable": sys.executable, "version": sys.version},
        "packages": {
            "huggingface-hub": _version("huggingface-hub"),
            "modelscope": _version("modelscope"),
        },
        "git": _git_state(),
        "seed": None,
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "source": {
            "model_id": JOYAI_MODELSCOPE_MODEL_ID,
            "revision": JOYAI_MODELSCOPE_MODEL_REVISION,
            "url": url,
        },
        "existing_prefix": (
            {"path": str(prefix), "bytes": prefix.stat().st_size}
            if prefix is not None
            else None
        ),
        "repair_ranges": args.repair_range,
        "destination": str(destination),
        "error": None,
    }
    write_json(manifest_path, manifest)
    try:
        if args.repair_range:
            if not destination.is_file() or destination.stat().st_size != JOYAI_DIT_BYTES:
                raise ValueError("range repair requires the preallocated full-size DiT destination")
        else:
            if destination.exists():
                raise FileExistsError(f"refusing to replace existing DiT destination: {destination}")
            assert prefix is not None
            os.link(prefix, destination)
        result = download(
            url=url,
            output=destination,
            expected_size=JOYAI_DIT_BYTES,
            connections=args.connections,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            chunk_bytes=8 * 1024 * 1024,
            resume_existing_prefix=True,
            expected_sha256=expected_hash,
            skip_remote_probe=True,
            explicit_ranges=(
                [(int(start), int(end)) for start, end in args.repair_range]
                if args.repair_range
                else None
            ),
        )
        marker = checkpoint_root / "JoyAI-Video-Edit/.phiagent-model-revision"
        marker.write_text(
            f"modelscope:{JOYAI_MODELSCOPE_MODEL_REVISION}\n", encoding="utf-8"
        )
        manifest.update(
            {
                "status": "WORKING",
                "stage": "dit_tail_resume_hash_validated",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result": result,
                "marker": {"path": str(marker), "value": marker.read_text().strip()},
            }
        )
        write_json(manifest_path, manifest)
        print(json.dumps({"experiment": str(output), "status": "WORKING", **result}, indent=2))
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "PARTIAL",
                "stage": "dit_tail_resume_failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
                "observed_destination": (
                    {
                        "bytes": destination.stat().st_size,
                        "sha256": sha256_file(destination)
                        if destination.is_file() and destination.stat().st_size == JOYAI_DIT_BYTES
                        else None,
                    }
                    if destination.exists()
                    else None
                ),
            }
        )
        write_json(manifest_path, manifest)
        print(json.dumps({"experiment": str(output), "status": "PARTIAL", "error": repr(exc)}, indent=2), file=sys.stderr)
        return 1


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
