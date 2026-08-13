#!/usr/bin/env python3
"""Install the pinned optional JoyAI runtime with a reproducible uv manifest."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.joyai_video_edit import (  # noqa: E402
    sha256_file,
    validate_upstream_checkout,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uv", type=Path, default=Path(shutil.which("uv") or "uv"))
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda-12.8"))
    parser.add_argument("--cuda-archs", default="80")
    parser.add_argument("--max-jobs", type=int, default=16)
    parser.add_argument("--default-index-url")
    parser.add_argument(
        "--index-strategy",
        choices=("first-index", "unsafe-first-match", "unsafe-best-match"),
        default="first-index",
    )
    parser.add_argument("--http-timeout-seconds", type=int, default=600)
    parser.add_argument("--http-retries", type=int, default=8)
    parser.add_argument(
        "--include-fp8-build-deps",
        action="store_true",
        help="Install CUTLASS/CUDA Python build dependencies (not needed by A800 BF16).",
    )
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


def _run(
    command: list[str], *, environment: dict[str, str], log: Path, cwd: Path
) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    log.write_text(
        "$ " + shlex.join(command) + "\n" + completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"command returned {completed.returncode}; inspect {log}")


def _runtime_versions(python: Path, environment: dict[str, str]) -> dict[str, Any]:
    code = r"""
import importlib.metadata as m
import json
import sys
names = ['torch','torchvision','torchaudio','transformers','diffusers','flash-attn-4','nvidia-cutlass-dsl','joyomni_ops']
versions = {}
for name in names:
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps({'python': sys.version, 'packages': versions}))
"""
    completed = subprocess.run(
        [str(python), "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return json.loads(completed.stdout.splitlines()[-1])


def write_effective_requirements(
    source: Path, destination: Path, *, include_fp8_build_deps: bool
) -> dict[str, Any]:
    """Remove only packages proven exclusive to the disabled FP8 extension."""

    fp8_only = {
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-base",
        "cuda-python",
        "cuda-bindings",
        "cuda-core",
        "cuda-pathfinder",
    }
    kept: list[str] = []
    removed: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        normalized = line.strip().split("==", maxsplit=1)[0].lower()
        if not include_fp8_build_deps and normalized in fp8_only:
            removed.append(line.strip())
        else:
            kept.append(line)
    destination.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return {
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "effective": {
            "path": str(destination),
            "sha256": sha256_file(destination),
        },
        "include_fp8_build_deps": include_fp8_build_deps,
        "removed_requirements": removed,
        "reason": (
            "none"
            if include_fp8_build_deps
            else "JOYOMNI_OPS_NO_FP8=1 excludes the only code importing these build packages"
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    repository = args.repository.expanduser().resolve()
    # Preserve the venv symlink path; resolving it selects the base environment.
    python = Path(os.path.abspath(args.python.expanduser()))
    output = args.output_dir.expanduser().resolve()
    uv = args.uv.expanduser().resolve()
    cuda_home = args.cuda_home.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"JoyAI runtime experiment already exists: {output}")
    if min(args.max_jobs, args.http_timeout_seconds, args.http_retries) < 1:
        raise ValueError("max jobs, HTTP timeout, and retries must be positive")
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "stage": "runtime_install_started",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "git": _git_state(),
        "seed": None,
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "error": None,
    }
    write_json(manifest_path, manifest)
    try:
        if not python.is_file() or not uv.is_file() or not cuda_home.is_dir():
            raise FileNotFoundError("Python, uv, or CUDA 12.8 path is missing")
        source = validate_upstream_checkout(repository)
        requirements = repository / "requirements.txt"
        joyomni = repository / "joyomni_ops"
        if not requirements.is_file() or not joyomni.is_dir():
            raise FileNotFoundError("JoyAI requirements or joyomni_ops source is missing")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_HOME": str(cuda_home),
                "PATH": str(cuda_home / "bin") + os.pathsep + environment.get("PATH", ""),
                "JOYOMNI_OPS_NO_FP8": "1",
                "JOYOMNI_OPS_CUDA_ARCHS": args.cuda_archs,
                "MAX_JOBS": str(args.max_jobs),
                "UV_HTTP_TIMEOUT": str(args.http_timeout_seconds),
                "UV_HTTP_RETRIES": str(args.http_retries),
            }
        )
        effective_requirements = output / "requirements-effective.txt"
        requirements_record = write_effective_requirements(
            requirements,
            effective_requirements,
            include_fp8_build_deps=args.include_fp8_build_deps,
        )
        uv_index_args = ["--index-strategy", args.index_strategy]
        if args.default_index_url:
            uv_index_args.extend(["--default-index", args.default_index_url])
        commands = [
            [
                str(uv),
                "pip",
                "install",
                "--python",
                str(python),
                *uv_index_args,
                "-r",
                str(effective_requirements),
            ],
            [
                str(uv),
                "pip",
                "install",
                "--python",
                str(python),
                *uv_index_args,
                "--no-build-isolation",
                str(joyomni),
            ],
        ]
        for index, command in enumerate(commands):
            _run(
                command,
                environment=environment,
                log=output / f"install-{index:02d}.log",
                cwd=PROJECT_ROOT,
            )
        freeze = subprocess.run(
            [str(python), "-m", "pip", "freeze"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        freeze_path = output / "packages.txt"
        freeze_path.write_text(freeze.stdout, encoding="utf-8")
        if freeze.returncode:
            raise RuntimeError(f"pip freeze failed: {freeze.stderr.strip()}")
        manifest.update(
            {
                "status": "WORKING",
                "stage": "runtime_install_completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "requirements": requirements_record,
                "environment": {
                    key: environment[key]
                    for key in (
                        "CUDA_HOME",
                        "JOYOMNI_OPS_NO_FP8",
                        "JOYOMNI_OPS_CUDA_ARCHS",
                        "MAX_JOBS",
                        "UV_HTTP_TIMEOUT",
                        "UV_HTTP_RETRIES",
                    )
                },
                "install_commands": commands,
                "runtime": _runtime_versions(python, environment),
                "packages": {
                    "path": str(freeze_path),
                    "sha256": sha256_file(freeze_path),
                },
            }
        )
        write_json(manifest_path, manifest)
        print(json.dumps({"experiment": str(output), "status": "WORKING"}, indent=2))
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "PARTIAL",
                "stage": "runtime_install_failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
            }
        )
        write_json(manifest_path, manifest)
        print(
            json.dumps(
                {"experiment": str(output), "status": "PARTIAL", "error": repr(exc)},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
