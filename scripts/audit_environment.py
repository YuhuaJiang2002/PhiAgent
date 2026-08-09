#!/usr/bin/env python3
"""Emit a non-mutating local or remote PhiAgent environment audit as JSON."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def capture(command: Sequence[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False}
    completed = subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "available": True,
        "executable": executable,
        "return_code": completed.returncode,
        "output": completed.stdout.strip(),
    }


def disk(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def audit() -> dict[str, Any]:
    home = Path.home()
    report: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "os_release": (
            platform.freedesktop_os_release() if sys.platform.startswith("linux") else {}
        ),
        "architecture": platform.machine(),
        "python_current": {"executable": sys.executable, "version": sys.version},
        "python_candidates": {},
        "disk": [disk(home), disk(Path.cwd())],
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "tools": {
            "conda": capture(["conda", "--version"]),
            "ffmpeg": capture(["ffmpeg", "-version"]),
            "git_lfs": capture(["git-lfs", "version"]),
            "nvcc": capture(["nvcc", "--version"]),
            "nvidia_smi": capture(["nvidia-smi"]),
            "gpu_inventory": capture(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version,memory.total,"
                    "memory.used,memory.free,compute_cap",
                    "--format=csv,noheader",
                ]
            ),
            "gpu_processes": capture(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                    "--format=csv,noheader",
                ]
            ),
        },
    }
    for candidate in ("python3.12", "python3.11", "python3.10", "python3", "python"):
        report["python_candidates"][candidate] = capture([candidate, "--version"])
    report["torch"] = capture(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.__version__); print(torch.version.cuda); "
            "print(torch.cuda.is_available()); print(torch.backends.cudnn.version())",
        ]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args()
    payload = json.dumps(audit(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

