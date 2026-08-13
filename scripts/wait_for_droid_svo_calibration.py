#!/usr/bin/env python3
"""Wait for a strictly free GPU, then calibrate real lineage-verified DROID SVOs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--pythonpath", type=Path, action="append", default=[])
    parser.add_argument("--locale", default="C")
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--reserved-gpu", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--maximum-wait-seconds", type=int, default=604_800)
    parser.add_argument("--maximum-used-gpu-mib", type=int, default=1023)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=1024)
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


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def absolute_executable_path(path: Path) -> Path:
    """Make a venv executable absolute without dereferencing its symlink."""

    return Path(os.path.abspath(path.expanduser()))


def build_runtime_environment(
    python_paths: Sequence[Path],
    locale: str,
) -> dict[str, str]:
    if not locale:
        raise ValueError("runtime locale must be non-empty")
    environment = os.environ.copy()
    environment["LC_ALL"] = locale
    environment["LANG"] = locale
    if python_paths:
        entries = [str(path) for path in python_paths]
        inherited = environment.get("PYTHONPATH")
        if inherited:
            entries.append(inherited)
        environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def classify_gpu_lines(
    inventory_lines: Sequence[str],
    process_lines: Sequence[str],
    maximum_used_mib: int,
) -> list[dict[str, Any]]:
    if maximum_used_mib < 0:
        raise ValueError("maximum used GPU memory must be non-negative")
    process_uuids = {
        line.split(",", maxsplit=1)[0].strip()
        for line in process_lines
        if line.strip()
    }
    result = []
    for line in inventory_lines:
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ValueError(f"unexpected GPU inventory line: {line}")
        index, uuid, name, total, used = fields
        used_mib = int(used)
        result.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "total_mib": int(total),
                "used_mib": used_mib,
                "classification": (
                    "free"
                    if used_mib <= maximum_used_mib and uuid not in process_uuids
                    else "reserved_or_busy"
                ),
            }
        )
    return result


def select_strictly_free_gpu(
    gpus: Sequence[dict[str, Any]],
    requested: Sequence[int],
) -> int | None:
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("requested GPU indices must be non-empty and unique")
    by_index = {int(gpu["index"]): gpu for gpu in gpus}
    return next(
        (
            index
            for index in requested
            if index in by_index and by_index[index]["classification"] == "free"
        ),
        None,
    )


def _query_gpus(maximum_used_mib: int) -> tuple[list[dict[str, Any]], str, str]:
    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    processes_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    processes = (
        processes_result.stdout
        if processes_result.returncode == 0
        else f"process query failed: {processes_result.stderr.strip()}"
    )
    classified = classify_gpu_lines(
        inventory.strip().splitlines(),
        processes.strip().splitlines(),
        maximum_used_mib,
    )
    return classified, inventory, processes


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
            "waiter_script_sha256": _sha256(Path(__file__).resolve()),
            "extractor_script_sha256": _sha256(
                PROJECT_ROOT / "scripts" / "extract_droid_svo_calibration.py"
            ),
        }
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "resolution": "local Git worktree",
        "waiter_script_sha256": _sha256(Path(__file__).resolve()),
        "extractor_script_sha256": _sha256(
            PROJECT_ROOT / "scripts" / "extract_droid_svo_calibration.py"
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    manifests = [path.expanduser().resolve() for path in args.episode_manifest]
    output = args.output_root.expanduser().resolve()
    python_executable = absolute_executable_path(args.python_executable)
    python_paths = [path.expanduser().resolve() for path in args.pythonpath]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DROID calibration queue: {output}")
    if not python_executable.is_file():
        raise ValueError(f"Python executable is missing: {python_executable}")
    missing_python_paths = [path for path in python_paths if not path.is_dir()]
    if missing_python_paths:
        raise ValueError(f"PYTHONPATH directories are missing: {missing_python_paths}")
    if len(set(manifests)) != len(manifests):
        raise ValueError("episode manifests must be unique")
    if len(set(args.gpu)) != len(args.gpu) or any(index < 0 for index in args.gpu):
        raise ValueError("GPU indices must be unique and non-negative")
    if (
        len(set(args.reserved_gpu)) != len(args.reserved_gpu)
        or any(index < 0 for index in args.reserved_gpu)
        or set(args.gpu) & set(args.reserved_gpu)
    ):
        raise ValueError("reserved GPUs must be unique, non-negative, and not requested")
    if (
        args.poll_seconds <= 0
        or args.maximum_wait_seconds <= 0
        or args.maximum_used_gpu_mib < 0
        or args.minimum_free_gpu_mib <= 0
        or not args.locale
    ):
        raise ValueError("wait and GPU thresholds are invalid")
    manifest_rows = []
    for manifest in manifests:
        if not manifest.is_file():
            raise ValueError(f"episode manifest is missing: {manifest}")
        payload = json.loads(manifest.read_text())
        if not isinstance(payload, dict) or payload.get("status") != "WORKING":
            raise ValueError(f"episode download is not WORKING: {manifest}")
        index = payload.get("episode_index")
        if not isinstance(index, int):
            raise ValueError(f"episode manifest has no integer index: {manifest}")
        manifest_rows.append(
            {
                "episode_index": index,
                "path": str(manifest),
                "sha256": _sha256(manifest),
            }
        )

    output.mkdir(parents=True)
    (output / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n"
    )
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "status": "WAITING",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": str(python_executable),
            "pythonpath": [str(path) for path in python_paths],
            "locale": args.locale,
            "episode_manifests": manifest_rows,
            "requested_physical_gpus": args.gpu,
            "reserved_physical_gpus": args.reserved_gpu,
            "maximum_used_gpu_mib": args.maximum_used_gpu_mib,
            "minimum_free_gpu_mib": args.minimum_free_gpu_mib,
            "poll_seconds": args.poll_seconds,
            "maximum_wait_seconds": args.maximum_wait_seconds,
            "seed": args.seed,
            "seed_use": "recorded for reproducibility; calibration is deterministic",
        },
    )
    _write_json(
        output / "git-state.json",
        _git_state(args.git_commit, args.git_branch),
    )
    runtime_environment = build_runtime_environment(python_paths, args.locale)
    packages = subprocess.run(
        [str(python_executable), "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
        env=runtime_environment,
    )
    (output / "packages.txt").write_text(packages.stdout)

    started = time.monotonic()
    selected = None
    while time.monotonic() - started < args.maximum_wait_seconds:
        gpus, inventory, processes = _query_gpus(args.maximum_used_gpu_mib)
        selected = select_strictly_free_gpu(gpus, args.gpu)
        _append_jsonl(
            output / "heartbeat.jsonl",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "selected_physical_gpu": selected,
                "requested_physical_gpus": args.gpu,
                "gpus": gpus,
                "gpu_inventory_raw": inventory,
                "gpu_processes_raw": processes,
            },
        )
        if selected is not None:
            break
        time.sleep(args.poll_seconds)
    if selected is None:
        result = {
            "status": "BLOCKED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "blocker": "timed_out_waiting_for_strictly_free_gpu",
        }
        _write_json(output / "result.json", result)
        return 2

    runs = []
    for row in manifest_rows:
        episode_index = row["episode_index"]
        episode_output = output / "calibration" / f"episode-{episode_index:03d}"
        command = [
            str(python_executable),
            str(PROJECT_ROOT / "scripts" / "extract_droid_svo_calibration.py"),
            "--download-manifest",
            row["path"],
            "--output-dir",
            str(episode_output),
            "--gpu",
            str(selected),
            "--minimum-free-gpu-mib",
            str(args.minimum_free_gpu_mib),
        ]
        with (output / f"episode-{episode_index:03d}.log").open("w") as log:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env=runtime_environment,
            )
        runs.append(
            {
                "episode_index": episode_index,
                "return_code": completed.returncode,
                "command": command,
                "output_dir": str(episode_output),
            }
        )
        if completed.returncode != 0:
            result = {
                "status": "BLOCKED",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "blocker": f"episode_{episode_index:03d}_calibration_failed",
                "selected_physical_gpu": selected,
                "runs": runs,
            }
            _write_json(output / "result.json", result)
            return 2

    result = {
        "status": "WORKING",
        "honest_status": "BLOCKED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "selected_physical_gpu": selected,
        "wall_seconds": time.monotonic() - started,
        "runs": runs,
        "rights_boundary": (
            "Calibration can be WORKING while raw DROID training and "
            "redistribution remain blocked on unresolved official rights."
        ),
    }
    _write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
