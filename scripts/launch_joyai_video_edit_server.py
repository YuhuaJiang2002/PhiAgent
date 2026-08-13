#!/usr/bin/env python3
"""Launch pinned JoyAI inference after two-physical-GPU and weight preflight."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.joyai_video_edit import (  # noqa: E402
    JoyAIPreflightError,
    build_server_argv,
    sha256_file,
    validate_checkpoint_layout,
    validate_upstream_checkout,
    write_json,
)
from phiagent.rendering.wan_animate import acquire_gpu_lease  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--physical-gpu", type=int, action="append", required=True)
    parser.add_argument("--minimum-primary-free-mib", type=int, default=60 * 1024)
    parser.add_argument("--minimum-vae-free-mib", type=int, default=20 * 1024)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--ready-timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--overlay",
        type=Path,
        default=PROJECT_ROOT / "third_party_overlays/joyai_video_edit/a800_streaming_timeout.patch",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def query_gpu_inventory() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise JoyAIPreflightError("nvidia-smi is required for JoyAI GPU preflight")
    command = [
        executable,
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=20)
    if result.returncode:
        raise JoyAIPreflightError(f"nvidia-smi failed: {result.stderr.strip()}")
    inventory = []
    for row in csv.reader(line for line in result.stdout.splitlines() if line.strip()):
        if len(row) != 7:
            raise JoyAIPreflightError(f"unexpected nvidia-smi row: {row!r}")
        inventory.append(
            {
                "physical_index": int(row[0].strip()),
                "uuid": row[1].strip(),
                "name": row[2].strip(),
                "total_mib": int(row[3].strip()),
                "used_mib": int(row[4].strip()),
                "free_mib": int(row[5].strip()),
                "compute_capability": row[6].strip(),
            }
        )
    return inventory


def select_gpu_pair(
    inventory: list[dict[str, Any]], requested: list[int], primary_min: int, vae_min: int
) -> list[dict[str, Any]]:
    if len(requested) != 2 or len(set(requested)) != 2:
        raise JoyAIPreflightError("JoyAI requires exactly two distinct physical GPU indices")
    selected = []
    for logical, index in enumerate(requested):
        gpu = next((row for row in inventory if row["physical_index"] == index), None)
        if gpu is None:
            raise JoyAIPreflightError(f"physical GPU {index} is not in nvidia-smi inventory")
        minimum = primary_min if logical == 0 else vae_min
        if gpu["free_mib"] < minimum:
            raise JoyAIPreflightError(
                f"physical GPU {index} has {gpu['free_mib']} MiB free; {minimum} MiB required"
            )
        if float(gpu["compute_capability"]) < 8.0:
            raise JoyAIPreflightError(
                f"physical GPU {index} compute capability {gpu['compute_capability']} < 8.0"
            )
        selected.append({**gpu, "logical_index": logical, "minimum_free_mib": minimum})
    return selected


def probe_runtime(python: Path, environment: dict[str, str]) -> dict[str, Any]:
    probe = r"""
import importlib.metadata as m
import json
import torch
import av
import cv2
import flash_attn.cute
import joyomni_ops
names = ['torch','transformers','diffusers','fastapi','uvicorn','av','opencv-python-headless','websockets']
versions = {}
for name in names:
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
result = {
  'packages': versions,
  'torch_cuda': torch.version.cuda,
  'cuda_available': torch.cuda.is_available(),
  'cuda_device_count': torch.cuda.device_count(),
  'logical_devices': [
    {'logical_index': i, 'name': torch.cuda.get_device_name(i), 'capability': list(torch.cuda.get_device_capability(i))}
    for i in range(torch.cuda.device_count())
  ],
  'joyomni_has_fp8': bool(joyomni_ops.has_fp8()),
  'cv2': cv2.__version__,
  'av': av.__version__,
  'cuda_smoke': [],
}
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        value = torch.ones(16, device=f'cuda:{index}', dtype=torch.float32).sum().item()
        torch.cuda.synchronize(index)
        result['cuda_smoke'].append({'logical_index': index, 'finite_sum': value})
print(json.dumps(result))
"""
    completed = subprocess.run(
        [str(python), "-c", probe],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if completed.returncode:
        raise JoyAIPreflightError(
            "JoyAI optional runtime import/CUDA probe failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    result = json.loads(completed.stdout.splitlines()[-1])
    if result.get("cuda_available") is not True or result.get("cuda_device_count") != 2:
        raise JoyAIPreflightError(f"JoyAI runtime did not expose exactly two CUDA devices: {result}")
    torch_version = str(result["packages"].get("torch") or "")
    if torch_version.split("+", maxsplit=1)[0] != "2.9.1" or result["torch_cuda"] != "12.8":
        raise JoyAIPreflightError(
            "JoyAI runtime must use the pinned torch 2.9.1 / CUDA 12.8 stack; "
            f"observed torch={torch_version}, CUDA={result['torch_cuda']}"
        )
    if any(row["finite_sum"] != 16.0 for row in result["cuda_smoke"]):
        raise JoyAIPreflightError(f"JoyAI CUDA smoke operation was not finite: {result}")
    return result


def stage_runtime_source(repository: Path, destination: Path, overlay: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(
        repository,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "deps", "recordings"),
    )
    command = ["patch", "-p1", "-i", str(overlay.resolve())]
    completed = subprocess.run(
        command, cwd=destination, capture_output=True, text=True, check=False
    )
    if completed.returncode:
        raise JoyAIPreflightError(
            "could not apply the pinned A800 timeout overlay: "
            + completed.stdout
            + completed.stderr
        )
    server = destination / "deploy/xvideo/serving/serve_joyomni_streaming.py"
    if "--holder-idle-timeout-s" not in server.read_text(encoding="utf-8"):
        raise JoyAIPreflightError("staged JoyAI server is missing the timeout overlay")
    return {
        "source_repository": str(repository),
        "staged_repository": str(destination),
        "overlay": str(overlay.resolve()),
        "overlay_sha256": sha256_file(overlay.resolve()),
        "command": command,
    }


def _git_state() -> dict[str, Any]:
    def capture(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    status = capture("status", "--short")
    return {
        "head": capture("rev-parse", "HEAD"),
        "branch": capture("branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": __import__("hashlib").sha256(status.encode()).hexdigest(),
    }


def health(url: str, timeout: float = 5.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)
    except Exception:
        return None


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"JoyAI server experiment already exists: {output}")
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "stage": "joyai_server_preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "git": _git_state(),
        "physical_evidence": False,
        "error": None,
    }
    write_json(manifest_path, manifest)
    process: subprocess.Popen[bytes] | None = None
    lease_handles = []
    try:
        repository = args.repository.expanduser().resolve()
        checkpoint_root = args.checkpoint_root.expanduser().resolve()
        # Keep the venv interpreter path itself. Path.resolve() would follow its
        # symlink to the shared base interpreter and bypass venv package lookup.
        python = Path(os.path.abspath(args.python.expanduser()))
        overlay = args.overlay.expanduser().resolve()
        if not python.is_file() or not overlay.is_file():
            raise FileNotFoundError("JoyAI Python executable or source overlay is missing")
        source = validate_upstream_checkout(repository)
        checkpoints = validate_checkpoint_layout(checkpoint_root, verify_large_hashes=True)
        inventory = query_gpu_inventory()
        selected = select_gpu_pair(
            inventory,
            args.physical_gpu,
            args.minimum_primary_free_mib,
            args.minimum_vae_free_mib,
        )
        for gpu in sorted(selected, key=lambda row: row["physical_index"]):
            lease_handles.append(acquire_gpu_lease(gpu["physical_index"]))
        inventory_after_lease = query_gpu_inventory()
        selected = select_gpu_pair(
            inventory_after_lease,
            args.physical_gpu,
            args.minimum_primary_free_mib,
            args.minimum_vae_free_mib,
        )

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in args.physical_gpu)
        environment["JOYOMNI_FP8_IMG"] = "0"
        environment["JOYOMNI_FP8_TXT"] = "0"
        environment["PYTHONUNBUFFERED"] = "1"
        cache = checkpoint_root.parent / "joyai-runtime-cache"
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "torchinductor")
        environment["TRITON_CACHE_DIR"] = str(cache / "triton")
        environment["CUDA_CACHE_PATH"] = str(cache / "nv_compute")
        environment["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        environment["CUDA_HOME"] = environment.get("CUDA_HOME", "/usr/local/cuda-12.8")
        environment["PATH"] = str(Path(environment["CUDA_HOME"]) / "bin") + os.pathsep + environment.get("PATH", "")
        for directory in (Path(environment["TORCHINDUCTOR_CACHE_DIR"]), Path(environment["TRITON_CACHE_DIR"]), Path(environment["CUDA_CACHE_PATH"])):
            directory.mkdir(parents=True, exist_ok=True)
        runtime = probe_runtime(python, environment)
        staged = stage_runtime_source(repository, output / "runtime-source", overlay)
        records = output / "recordings"
        records.mkdir()
        argv = build_server_argv(
            python_executable=python,
            repository=Path(staged["staged_repository"]),
            checkpoint_root=checkpoint_root,
            record_dir=records,
            host=args.host,
            port=args.port,
        )
        environment["PYTHONPATH"] = str(Path(staged["staged_repository"]) / "deploy")
        manifest.update(
            {
                "source": source,
                "checkpoints": checkpoints,
                "gpu": {
                    "inventory": inventory,
                    "inventory_after_lease": inventory_after_lease,
                    "selected": selected,
                    "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
                    "logical_placement": {
                        "dit_and_text_encoder": "cuda:0",
                        "vae_and_postprocess": "cuda:1",
                    },
                    "lease_paths": [str(path) for path, _ in lease_handles],
                },
                "runtime": runtime,
                "staged_source": staged,
                "server_argv": list(argv),
                "environment": {
                    key: environment[key]
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "JOYOMNI_FP8_IMG",
                        "JOYOMNI_FP8_TXT",
                        "CUDA_HOME",
                        "TORCHINDUCTOR_CACHE_DIR",
                        "TRITON_CACHE_DIR",
                        "CUDA_CACHE_PATH",
                        "PYTHONPATH",
                    )
                },
                "preflight_completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if args.preflight_only:
            manifest.update({"status": "WORKING", "stage": "joyai_server_preflight_only"})
            write_json(manifest_path, manifest)
            print(json.dumps({"experiment": str(output), "status": "WORKING", "preflight_only": True}, indent=2))
            return 0

        stdout = (output / "server.stdout.log").open("wb")
        stderr = (output / "server.stderr.log").open("wb")
        process = subprocess.Popen(
            list(argv),
            cwd=Path(staged["staged_repository"]) / "deploy",
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
        manifest.update(
            {
                "stage": "joyai_server_starting",
                "server_pid": process.pid,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(manifest_path, manifest)

        def terminate(_signum: int, _frame: Any) -> None:
            if process is not None and process.poll() is None:
                process.terminate()

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        health_url = f"http://{args.host}:{args.port}/health"
        deadline = time.monotonic() + args.ready_timeout_seconds
        health_result = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"JoyAI server exited during preload with code {process.returncode}")
            health_result = health(health_url)
            if health_result is not None:
                break
            time.sleep(5)
        if health_result is None:
            raise TimeoutError(f"JoyAI server did not become healthy within {args.ready_timeout_seconds}s")
        manifest.update(
            {
                "status": "WORKING",
                "stage": "joyai_server_ready",
                "ready_at": datetime.now(timezone.utc).isoformat(),
                "health_url": health_url,
                "health": health_result,
            }
        )
        write_json(manifest_path, manifest)
        print(json.dumps({"experiment": str(output), "status": "WORKING", "health": health_result}, indent=2))
        returncode = process.wait()
        manifest.update(
            {
                "status": "PARTIAL",
                "stage": "joyai_server_stopped",
                "server_returncode": returncode,
                "stopped_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(manifest_path, manifest)
        return returncode
    except Exception as exc:
        manifest.update(
            {
                "status": "PARTIAL",
                "stage": "joyai_server_preflight_or_start_failed",
                "error": repr(exc),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(manifest_path, manifest)
        if process is not None and process.poll() is None:
            process.terminate()
        print(json.dumps({"experiment": str(output), "status": "PARTIAL", "error": repr(exc)}, indent=2), file=sys.stderr)
        return 1
    finally:
        for _, lease in reversed(lease_handles):
            lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
