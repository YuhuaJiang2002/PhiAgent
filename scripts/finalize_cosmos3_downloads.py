#!/usr/bin/env python3
"""Wait for pinned Cosmos3 downloads and produce immutable verification reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_cosmos3_checkpoint import verify_checkpoint  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nano-checkpoint", type=Path, required=True)
    parser.add_argument("--nano-download-marker", type=Path, required=True)
    parser.add_argument("--nano-revision", required=True)
    parser.add_argument(
        "--nano-require-file",
        action="append",
        default=["config.json"],
        help="Nano-relative small file to hash and bind; may be repeated",
    )
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-download-marker", type=Path, required=True)
    parser.add_argument("--vae-revision", required=True)
    parser.add_argument("--vae-file", default="Wan2.2_VAE.pth")
    parser.add_argument("--vae-size-bytes", type=int, required=True)
    parser.add_argument("--vae-sha256", required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=86_400.0)
    parser.add_argument("--project-source-revision", required=True)
    parser.add_argument("--project-source-branch", required=True)
    return parser


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_markers(
    markers: list[Path], poll_seconds: float, timeout_seconds: float
) -> float:
    if poll_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("poll and timeout seconds must be positive")
    started = time.monotonic()
    while True:
        if all(marker.is_file() for marker in markers):
            return time.monotonic() - started
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            missing = [str(marker) for marker in markers if not marker.is_file()]
            raise TimeoutError(f"download markers did not appear: {missing}")
        time.sleep(min(poll_seconds, timeout_seconds - elapsed))


def verify_vae(
    checkpoint: Path,
    expected_revision: str,
    relative_file: str,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, Any]:
    root = checkpoint.expanduser().resolve()
    relative = Path(relative_file)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe VAE file path: {relative_file}")
    marker = root / ".phiagent-model-revision"
    if not marker.is_file():
        raise ValueError(f"VAE revision marker is missing: {marker}")
    actual_revision = marker.read_text(encoding="utf-8").strip()
    if actual_revision != expected_revision:
        raise ValueError(
            f"VAE revision mismatch: expected {expected_revision}, got {actual_revision}"
        )
    path = root / relative
    if not path.is_file():
        raise ValueError(f"VAE file is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"VAE size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"VAE SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "status": "WORKING",
        "checkpoint": str(root),
        "revision": actual_revision,
        "file": relative.as_posix(),
        "size_bytes": actual_size,
        "sha256": actual_sha256,
        "limitations": [
            "File integrity does not establish VAE loading, inference, or task-video quality."
        ],
    }


def _package_inventory(python: Path) -> str:
    uv = shutil.which("uv")
    commands = []
    if uv:
        commands.append([uv, "pip", "freeze", "--python", str(python)])
    commands.append([str(python), "-m", "pip", "freeze"])
    for command in commands:
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0:
            return completed.stdout
    return "PACKAGE INVENTORY UNAVAILABLE\n"


def main() -> int:
    args = _parser().parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    if experiment.exists():
        raise ValueError(f"experiment directory already exists: {experiment}")
    experiment.mkdir(parents=True)
    command = [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *sys.argv[1:]]
    (experiment / "command.txt").write_text(shlex.join(command) + "\n")
    (experiment / "packages.txt").write_text(
        _package_inventory(Path(sys.executable).resolve()), encoding="utf-8"
    )
    config = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "seed": None,
        "project_source_revision": args.project_source_revision,
        "project_source_branch": args.project_source_branch,
        "nano_checkpoint": str(args.nano_checkpoint.expanduser().resolve()),
        "nano_revision": args.nano_revision,
        "nano_required_files": sorted(set(args.nano_require_file)),
        "vae_checkpoint": str(args.vae_checkpoint.expanduser().resolve()),
        "vae_revision": args.vae_revision,
        "timeout_seconds": args.timeout_seconds,
    }
    _write_json(experiment / "config.json", config)

    try:
        waited = wait_for_markers(
            [
                args.nano_download_marker.expanduser().resolve(),
                args.vae_download_marker.expanduser().resolve(),
            ],
            args.poll_seconds,
            args.timeout_seconds,
        )
        nano = verify_checkpoint(
            args.nano_checkpoint,
            args.nano_revision,
            list(args.nano_require_file),
        )
        vae = verify_vae(
            args.vae_checkpoint,
            args.vae_revision,
            args.vae_file,
            args.vae_size_bytes,
            args.vae_sha256,
        )
        result = {
            "status": "WORKING",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "waited_seconds": waited,
            "nano": nano,
            "vae": vae,
            "limitations": [
                "Checkpoint verification does not establish model loading, inference, training, subject consistency, or viewpoint generalization."
            ],
        }
        _write_json(experiment / "result.json", result)
        (experiment / "verification.completed").touch()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        failure = {
            "status": "PARTIAL",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_json(experiment / "failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
