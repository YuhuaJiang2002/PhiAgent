#!/usr/bin/env python3
"""Run pinned RoboTwin render preflight on one validated physical GPU."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest_path = args.runtime_manifest.expanduser().resolve()
    python = args.python.expanduser().resolve()
    overlay = args.overlay.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RoboTwin render preflight: {output}")
    if not manifest_path.is_file() or not python.is_file() or not overlay.is_dir():
        raise ValueError("RoboTwin runtime manifest, Python, or overlay is missing")
    runtime = json.loads(manifest_path.read_text())
    if runtime.get("status") != "WORKING":
        raise ValueError("RoboTwin runtime view is not working")
    source = Path(str(runtime["runtime_source"])).expanduser().resolve()
    test_render = source / "scripts" / "test_render.py"
    if not test_render.is_file():
        raise ValueError(f"RoboTwin render entry point is missing: {test_render}")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    environment["PHIAGENT_PHYSICAL_GPU_INDEX"] = str(selected.physical_index)
    environment["ASSETS_PATH"] = str(source / "assets")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(overlay), str(source), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    command = [str(python), str(test_render)]
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "status": "STARTED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "runtime_manifest": str(manifest_path),
            "runtime_source": str(source),
            "runtime_overlay": str(overlay),
            "selected_physical_gpu": asdict(selected),
            "gpu_inventory_raw": inventory,
            "gpu_processes_raw": processes,
            "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
            "assets_path": environment["ASSETS_PATH"],
            "timeout_seconds": args.timeout_seconds,
            "command": command,
        },
    )
    completed = subprocess.run(
        command,
        cwd=source,
        env=environment,
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    log = completed.stdout + completed.stderr
    (output / "render.log").write_text(log)
    passed = completed.returncode == 0 and "Render Well" in log and "Render Error" not in log
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING" if passed else "BLOCKED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "return_code": completed.returncode,
        "render_well_marker": "Render Well" in log,
        "render_error_marker": "Render Error" in log,
        "selected_physical_gpu": asdict(selected),
        "claim_boundary": (
            "Render preflight validates SAPIEN/Vulkan startup only. It does not "
            "establish task reset reproducibility or executable counterfactuals."
        ),
    }
    _write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

