"""Auditable adapters that expose existing scripts as workflow nodes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping

from .core import NodeContext


class CommandPreflightError(RuntimeError):
    """Raised before a command starts when its declared contract is invalid."""


@dataclass(frozen=True)
class CommandSpec:
    """One shell-free command invocation.

    A physical GPU index opts into strict GPU preflight.  The adapter validates
    the physical device through ``nvidia-smi``, sets ``CUDA_VISIBLE_DEVICES``,
    and records the selection beside the command before starting it.
    """

    argv: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None
    expected_outputs: tuple[str, ...] = ()
    physical_gpu_index: int | None = None

    def validate(self) -> None:
        if not self.argv or any(not isinstance(item, str) or not item for item in self.argv):
            raise CommandPreflightError("argv must contain non-empty strings")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise CommandPreflightError("timeout_seconds must be positive")
        if self.physical_gpu_index is not None and self.physical_gpu_index < 0:
            raise CommandPreflightError("physical_gpu_index must be non-negative")
        if any(not key or not isinstance(value, str) for key, value in self.env.items()):
            raise CommandPreflightError("environment overrides must be non-empty strings")


CommandBuilder = Callable[[Mapping[str, Any], NodeContext], CommandSpec]


class SubprocessNode:
    """Run an existing CLI safely and return a compact execution record."""

    def __init__(self, builder: CommandBuilder, *, result_key: str = "last_command") -> None:
        self.builder = builder
        self.result_key = result_key

    def __call__(self, state: Mapping[str, Any], context: NodeContext) -> dict[str, Any]:
        spec = self.builder(state, context)
        if not isinstance(spec, CommandSpec):
            raise TypeError("command builder must return CommandSpec")
        spec.validate()
        node_dir = context.node_dir
        stdout_path = node_dir / "stdout.log"
        stderr_path = node_dir / "stderr.log"
        cwd = Path(spec.cwd).expanduser().resolve() if spec.cwd else Path.cwd().resolve()
        if not cwd.is_dir():
            raise CommandPreflightError(f"command working directory does not exist: {cwd}")

        environment = os.environ.copy()
        environment.update(spec.env)
        gpu = None
        if spec.physical_gpu_index is not None:
            gpu = _select_physical_gpu(spec.physical_gpu_index)
            environment["CUDA_VISIBLE_DEVICES"] = str(spec.physical_gpu_index)

        started_at = datetime.now(timezone.utc).isoformat()
        command_record = {
            "schema_version": "1.0.0",
            "argv": list(spec.argv),
            "cwd": str(cwd),
            "started_at": started_at,
            "environment_overrides": {
                key: _redact(key, value) for key, value in sorted(spec.env.items())
            },
            "gpu_selection": gpu,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "expected_outputs": list(spec.expected_outputs),
        }
        _write_json(node_dir / "command.json", command_record)
        start = monotonic()
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                list(spec.argv),
                cwd=cwd,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=spec.timeout_seconds,
                check=False,
            )
        wall_seconds = monotonic() - start
        missing = []
        for raw_path in spec.expected_outputs:
            path = Path(raw_path)
            resolved = path if path.is_absolute() else cwd / path
            if not resolved.exists():
                missing.append(str(resolved))
        result = {
            **command_record,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": wall_seconds,
            "returncode": completed.returncode,
            "missing_outputs": missing,
        }
        _write_json(node_dir / "result.json", result)
        if completed.returncode != 0:
            raise RuntimeError(
                f"command returned {completed.returncode}; inspect {stderr_path} and {stdout_path}"
            )
        if missing:
            raise RuntimeError(f"command did not create declared outputs: {missing}")
        return {self.result_key: result}


def _select_physical_gpu(index: int) -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise CommandPreflightError("nvidia-smi is required for a declared GPU command")
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if completed.returncode != 0:
        raise CommandPreflightError(f"nvidia-smi failed: {completed.stderr.strip()}")
    inventory = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) != 4:
            continue
        inventory.append(
            {
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
            }
        )
    selected = next((row for row in inventory if row["physical_index"] == index), None)
    if selected is None:
        raise CommandPreflightError(
            f"physical GPU {index} is unavailable; discovered {[row['physical_index'] for row in inventory]}"
        )
    return {**selected, "cuda_visible_devices": str(index), "inventory": inventory}


def _redact(key: str, value: str) -> str:
    normalized = key.upper()
    if any(marker in normalized for marker in ("TOKEN", "SECRET", "PASSWORD", "KEY")):
        return "<redacted>"
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
