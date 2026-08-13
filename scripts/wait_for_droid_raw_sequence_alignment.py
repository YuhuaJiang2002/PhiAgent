#!/usr/bin/env python3
"""Wait for a strictly free GPU, then run the raw DROID alignment audit."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wait_for_droid_svo_calibration import (  # noqa: E402
    _git_state,
    _query_gpus,
    _sha256,
    _write_json,
    absolute_executable_path,
    build_runtime_environment,
    select_strictly_free_gpu,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        nargs=2,
        action="append",
        required=True,
        metavar=("RAW_MANIFEST", "SEQUENCE_PAYLOAD"),
    )
    parser.add_argument("--sequence-lineage-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--pythonpath", type=Path, action="append", default=[])
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--reserved-gpu", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--maximum-wait-seconds", type=int, default=604_800)
    parser.add_argument("--maximum-used-gpu-mib", type=int, default=1023)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=81_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--git-commit")
    parser.add_argument("--git-branch")
    return parser


def build_audit_command(
    *,
    python_executable: Path,
    cases: list[tuple[Path, Path]],
    lineage_manifest: Path,
    output_dir: Path,
    gpu: int,
    minimum_free_gpu_mib: int,
    seed: int,
    git_commit: str | None,
    git_branch: str | None,
) -> list[str]:
    command = [
        str(python_executable),
        str(PROJECT_ROOT / "scripts" / "audit_droid_raw_sequence_alignment.py"),
    ]
    for raw_manifest, sequence_payload in cases:
        command.extend(
            (
                "--case",
                str(raw_manifest),
                str(sequence_payload),
            )
        )
    command.extend(
        (
            "--sequence-lineage-manifest",
            str(lineage_manifest),
            "--output-dir",
            str(output_dir),
            "--gpu",
            str(gpu),
            "--minimum-free-gpu-mib",
            str(minimum_free_gpu_mib),
            "--minimum-p05-psnr-db",
            "25",
            "--maximum-p95-dhash-hamming",
            "8",
            "--minimum-aligned-adjacent-psnr-gap-db",
            "1",
            "--maximum-centered-timestamp-p95-ms",
            "5",
            "--maximum-terminal-row-gap-ms",
            "200",
            "--seed",
            str(seed),
        )
    )
    if git_commit is not None and git_branch is not None:
        command.extend(
            (
                "--git-commit",
                git_commit,
                "--git-branch",
                git_branch,
            )
        )
    return command


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.expanduser().resolve()
    python_executable = absolute_executable_path(args.python_executable)
    python_paths = [path.expanduser().resolve() for path in args.pythonpath]
    lineage_manifest = args.sequence_lineage_manifest.expanduser().resolve()
    cases = [
        (
            Path(raw).expanduser().resolve(),
            Path(sequence).expanduser().resolve(),
        )
        for raw, sequence in args.case
    ]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DROID alignment queue: {output}")
    if not python_executable.is_file():
        raise ValueError(f"Python executable is missing: {python_executable}")
    if any(not path.is_dir() for path in python_paths):
        raise ValueError("all PYTHONPATH entries must be directories")
    if not lineage_manifest.is_file():
        raise ValueError("sequence lineage manifest must exist")
    if len(set(cases)) != len(cases):
        raise ValueError("alignment cases must be unique")
    if any(
        not raw.is_file() or not sequence.is_file()
        for raw, sequence in cases
    ):
        raise ValueError("all raw manifests and sequence payloads must exist")
    if len(set(args.gpu)) != len(args.gpu) or any(gpu < 0 for gpu in args.gpu):
        raise ValueError("GPU indices must be unique and non-negative")
    if (
        len(set(args.reserved_gpu)) != len(args.reserved_gpu)
        or any(gpu < 0 for gpu in args.reserved_gpu)
        or set(args.gpu) & set(args.reserved_gpu)
    ):
        raise ValueError("reserved GPUs must not overlap requested GPUs")
    if (
        args.poll_seconds <= 0
        or args.maximum_wait_seconds <= 0
        or args.maximum_used_gpu_mib < 0
        or args.minimum_free_gpu_mib <= 0
    ):
        raise ValueError("queue timing and GPU thresholds are invalid")

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
            "cases": [
                {
                    "raw_manifest": str(raw),
                    "raw_manifest_sha256": _sha256(raw),
                    "sequence_payload": str(sequence),
                    "sequence_payload_sha256": _sha256(sequence),
                }
                for raw, sequence in cases
            ],
            "sequence_lineage_manifest": str(lineage_manifest),
            "sequence_lineage_manifest_sha256": _sha256(lineage_manifest),
            "requested_physical_gpus": args.gpu,
            "reserved_physical_gpus": args.reserved_gpu,
            "maximum_used_gpu_mib": args.maximum_used_gpu_mib,
            "minimum_free_gpu_mib": args.minimum_free_gpu_mib,
            "poll_seconds": args.poll_seconds,
            "maximum_wait_seconds": args.maximum_wait_seconds,
            "seed": args.seed,
        },
    )
    git_state = _git_state(args.git_commit, args.git_branch)
    git_state.update(
        {
            "alignment_waiter_script_sha256": _sha256(Path(__file__).resolve()),
            "alignment_audit_script_sha256": _sha256(
                PROJECT_ROOT / "scripts" / "audit_droid_raw_sequence_alignment.py"
            ),
        }
    )
    _write_json(output / "git-state.json", git_state)
    environment = build_runtime_environment(python_paths, "C")
    environment["TF_CPP_MIN_LOG_LEVEL"] = "3"
    packages = subprocess.run(
        [str(python_executable), "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    (output / "packages.txt").write_text(packages.stdout)

    heartbeat = output / "heartbeat.jsonl"
    started = time.monotonic()
    selected = None
    while time.monotonic() - started < args.maximum_wait_seconds:
        gpus, inventory, processes = _query_gpus(args.maximum_used_gpu_mib)
        selected = select_strictly_free_gpu(gpus, args.gpu)
        with heartbeat.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "selected_physical_gpu": selected,
                        "requested_physical_gpus": args.gpu,
                        "gpus": gpus,
                        "gpu_inventory_raw": inventory,
                        "gpu_processes_raw": processes,
                    },
                    sort_keys=True,
                )
                + "\n"
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

    audit_output = output / "alignment-v1"
    command = build_audit_command(
        python_executable=python_executable,
        cases=cases,
        lineage_manifest=lineage_manifest,
        output_dir=audit_output,
        gpu=selected,
        minimum_free_gpu_mib=args.minimum_free_gpu_mib,
        seed=args.seed,
        git_commit=args.git_commit,
        git_branch=args.git_branch,
    )
    with (output / "alignment.log").open("w") as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=environment,
        )
    alignment_path = audit_output / "alignment.json"
    alignment = (
        json.loads(alignment_path.read_text())
        if alignment_path.is_file()
        else None
    )
    accepted = (
        completed.returncode == 0
        and isinstance(alignment, dict)
        and alignment.get("accepted") is True
    )
    result = {
        "status": "WORKING" if accepted else "BLOCKED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "selected_physical_gpu": selected,
        "wall_seconds": time.monotonic() - started,
        "command": command,
        "return_code": completed.returncode,
        "alignment": str(alignment_path),
        "alignment_sha256": (
            _sha256(alignment_path) if alignment_path.is_file() else None
        ),
        "accepted": accepted,
    }
    if not accepted:
        result["blocker"] = "raw_sequence_alignment_audit_failed"
    _write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
