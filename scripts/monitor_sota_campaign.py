#!/usr/bin/env python3
"""Persistently monitor SOTA jobs, GPU capacity, terminal artifacts, and successors."""

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
from typing import Any, Sequence


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def classify_gpu_lines(
    inventory_lines: Sequence[str],
    process_lines: Sequence[str],
) -> list[dict[str, Any]]:
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
                    if used_mib < 1024 and uuid not in process_uuids
                    else "reserved_or_busy"
                ),
            }
        )
    return result


def _ssh(host: str, remote_args: Sequence[str], *, timeout: int = 45) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            host,
            *remote_args,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _capacity(host: str) -> dict[str, Any]:
    command = (
        "nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used "
        "--format=csv,noheader,nounits; echo __PROCESSES__; "
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory "
        "--format=csv,noheader,nounits"
    )
    completed = _ssh(host, [command])
    if completed.returncode != 0:
        return {
            "host": host,
            "reachable": False,
            "error": (completed.stderr or completed.stdout).strip(),
        }
    inventory, separator, processes = completed.stdout.partition("__PROCESSES__\n")
    if not separator:
        raise ValueError(f"GPU capacity output lacks separator for {host}")
    inventory_lines = inventory.strip().splitlines()
    process_lines = processes.strip().splitlines()
    return {
        "host": host,
        "reachable": True,
        "gpus": classify_gpu_lines(inventory_lines, process_lines),
        "processes_raw": process_lines,
    }


def _remote_json(host: str, path: str) -> dict[str, Any] | None:
    completed = _ssh(host, [f"test -f {shlex.quote(path)} && cat {shlex.quote(path)}"])
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"remote JSON must contain an object: {host}:{path}")
    return payload


def _remote_process_status(job: dict[str, Any]) -> dict[str, Any]:
    host = str(job["host"])
    success = _remote_json(host, str(job["success_json"]))
    if success is not None:
        status = str(success.get("status", "unknown")).upper()
        return {
            "state": "success" if status == str(job["success_status"]).upper() else "failed",
            "artifact": success,
        }
    pid = int(job["pid"])
    completed = _ssh(host, [f"test -d /proc/{pid}"])
    return {"state": "running" if completed.returncode == 0 else "failed"}


def _remote_file_status(job: dict[str, Any]) -> dict[str, Any]:
    host = str(job["host"])
    path = str(job["path"])
    completed = _ssh(
        host,
        [
            f"test -f {shlex.quote(path)} && "
            f"stat -c %s {shlex.quote(path)} && sha256sum {shlex.quote(path)}"
        ],
    )
    if completed.returncode != 0:
        return {"state": "running"}
    lines = completed.stdout.strip().splitlines()
    if len(lines) != 2:
        return {"state": "running"}
    size = int(lines[0])
    digest = lines[1].split()[0]
    if size < int(job["expected_bytes"]):
        return {"state": "running", "bytes": size}
    passed = size == int(job["expected_bytes"]) and digest == job["expected_sha256"]
    return {
        "state": "success" if passed else "failed",
        "bytes": size,
        "sha256": digest,
    }


def _job_status(job: dict[str, Any]) -> dict[str, Any]:
    job_type = job.get("type")
    if job_type == "remote_process":
        return _remote_process_status(job)
    if job_type == "remote_file":
        return _remote_file_status(job)
    raise ValueError(f"unsupported monitor job type: {job_type}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--maximum-runtime-seconds", type=int, default=604_800)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SOTA monitor: {output}")
    if not config_path.is_file():
        raise ValueError(f"SOTA monitor config is missing: {config_path}")
    if args.poll_seconds <= 0 or args.maximum_runtime_seconds <= 0:
        raise ValueError("monitor timing must be positive")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("jobs"), list):
        raise ValueError("SOTA monitor config requires a jobs array")
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "status": "RUNNING",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "source_config": str(config_path),
            "source_config_payload": config,
            "poll_seconds": args.poll_seconds,
            "maximum_runtime_seconds": args.maximum_runtime_seconds,
        },
    )
    states = {
        str(job["id"]): {"state": "unknown", "successor_started": False}
        for job in config["jobs"]
    }
    started = time.monotonic()
    while time.monotonic() - started < args.maximum_runtime_seconds:
        timestamp = datetime.now(timezone.utc).isoformat()
        capacity = [_capacity(str(host)) for host in config.get("hosts", [])]
        _append_jsonl(
            output / "capacity.jsonl",
            {"timestamp": timestamp, "hosts": capacity},
        )
        for job in config["jobs"]:
            job_id = str(job["id"])
            current = _job_status(job)
            previous = states[job_id]["state"]
            if current["state"] != previous:
                _append_jsonl(
                    output / "events.jsonl",
                    {
                        "timestamp": timestamp,
                        "job_id": job_id,
                        "previous": previous,
                        "current": current,
                    },
                )
                states[job_id]["state"] = current["state"]
            if (
                current["state"] == "success"
                and job.get("successor")
                and not states[job_id]["successor_started"]
            ):
                successor = [str(value) for value in job["successor"]]
                completed = subprocess.run(
                    successor,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                (output / f"{job_id}-successor.log").write_text(
                    completed.stdout + completed.stderr
                )
                states[job_id]["successor_started"] = True
                _append_jsonl(
                    output / "events.jsonl",
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "job_id": job_id,
                        "event": "successor_completed",
                        "command": successor,
                        "return_code": completed.returncode,
                    },
                )
        heartbeat = {
            "timestamp": timestamp,
            "elapsed_seconds": time.monotonic() - started,
            "states": states,
        }
        _write_json(output / "heartbeat.json", heartbeat)
        if all(
            state["state"] in {"success", "failed"} for state in states.values()
        ):
            break
        time.sleep(args.poll_seconds)
    final = {
        "schema_version": "1.0.0",
        "status": (
            "WORKING"
            if all(state["state"] == "success" for state in states.values())
            else "PARTIAL"
        ),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "states": states,
    }
    _write_json(output / "result.json", final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

