"""Subprocess-isolated adapter for the supported mini-ArtiCraft implementation."""

from __future__ import annotations

import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from phiagent.assets.base import (
    AssetCompilationRequest,
    AssetGenerationRequest,
    AssetGenerationResult,
)

ARTICRAFT_COMMIT = "7d43e25b26e9459aabf53d77d1d9325805bc1ea3"

_RUNNER = """
import json
import sys
from pathlib import Path

import mini_articraft

request = json.loads(sys.argv[1])
kwargs = {
    "provider": request["provider"],
    "output_dir": request["output_dir"],
}
if request["model"] is not None:
    kwargs["model"] = request["model"]
if request["reference_image"] is not None:
    kwargs["image"] = request["reference_image"]
result = mini_articraft.generate(request["description"], **kwargs)
status = getattr(result.status, "value", str(result.status))
Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "status": status,
            "run_dir": str(result.run_dir),
            "artifact": str(result.artifact),
        },
        sort_keys=True,
    )
    + "\\n",
    encoding="utf-8",
)
"""

_COMPILER = """
import json
import shutil
import sys
from pathlib import Path

from mini_articraft.agent.workspace import LocalWorkspace

model_file = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
result_path = Path(sys.argv[3]).resolve()
workspace = LocalWorkspace(output_dir=output_dir)
run_dir = workspace.create_run("authored-model")
shutil.copy2(model_file, run_dir / "workspace" / "main.py")
result = workspace.compile_path(run_dir)
result_path.write_text(
    json.dumps(
        {
            "status": result["status"],
            "run_dir": str(run_dir),
            "artifact": str(result.get("usdz") or ""),
            "compile_report": result.get("compile_report"),
            "error": result.get("error"),
        },
        sort_keys=True,
    )
    + "\\n",
    encoding="utf-8",
)
"""


class AssetGenerationError(RuntimeError):
    """Raised when ArtiCraft cannot safely produce a usable asset."""


@dataclass(frozen=True)
class ArticraftConfig:
    """Pinned mini-ArtiCraft checkout and its isolated Python runtime."""

    repo: Path
    python_executable: Path


def _run_capture(command: Sequence[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _absolute_without_resolving(path: Path) -> Path:
    """Make a runtime path absolute without following virtualenv symlinks."""

    return Path(os.path.abspath(path.expanduser()))


class ArticraftAssetGenerator:
    """Generate USDZ assets with mini-ArtiCraft and retain complete run evidence."""

    def __init__(self, config: ArticraftConfig):
        self.config = config

    def preflight(self) -> dict[str, str]:
        repo = self.config.repo.expanduser().resolve()
        python = _absolute_without_resolving(self.config.python_executable)
        if not repo.is_dir():
            raise AssetGenerationError(f"ArtiCraft checkout does not exist: {repo}")
        if not python.is_file():
            raise AssetGenerationError(f"ArtiCraft Python executable does not exist: {python}")
        try:
            commit = _run_capture(["git", "rev-parse", "HEAD"], cwd=repo)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise AssetGenerationError(f"cannot inspect ArtiCraft checkout: {exc}") from exc
        if commit != ARTICRAFT_COMMIT:
            raise AssetGenerationError(
                f"ArtiCraft checkout is {commit!r}, expected pinned {ARTICRAFT_COMMIT}"
            )
        try:
            version = _run_capture(
                [
                    str(python),
                    "-c",
                    "import mini_articraft; print(mini_articraft.__version__)",
                ],
                cwd=repo,
            ).splitlines()[-1]
        except (FileNotFoundError, subprocess.CalledProcessError, IndexError) as exc:
            raise AssetGenerationError(
                "mini_articraft is not importable in the configured Python runtime"
            ) from exc
        return {
            "articraft_commit": commit,
            "mini_articraft_version": version,
            "python_executable": str(python),
        }

    @staticmethod
    def _new_experiment_dir(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment = root / f"{timestamp}-{uuid.uuid4().hex[:8]}"
        experiment.mkdir()
        return experiment

    def _project_provenance(self) -> dict[str, Any]:
        project_root = Path(__file__).resolve().parents[2]
        provenance: dict[str, Any] = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "argv": sys.argv,
        }
        for key, command in {
            "git_commit": ["git", "rev-parse", "HEAD"],
            "git_status": ["git", "status", "--porcelain"],
            "package_versions": [
                str(_absolute_without_resolving(self.config.python_executable)),
                "-m",
                "pip",
                "freeze",
            ],
        }.items():
            try:
                provenance[key] = _run_capture(command, cwd=project_root)
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                provenance[key] = f"unavailable: {exc}"
        return provenance

    @staticmethod
    def _execute(
        command: Sequence[str], cwd: Path, log_path: Path, env: Mapping[str, str]
    ) -> None:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=dict(env),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code:
            raise AssetGenerationError(
                f"ArtiCraft failed with exit code {return_code}; inspect {log_path}: "
                + shlex.join(command)
            )

    def generate(self, request: AssetGenerationRequest) -> AssetGenerationResult:
        reference_image: Path | None = None
        if request.reference_image is not None:
            reference_image = request.reference_image.expanduser().resolve()
            if not reference_image.is_file() or reference_image.stat().st_size == 0:
                raise AssetGenerationError(
                    f"reference image does not exist or is empty: {reference_image}"
                )
            if reference_image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                raise AssetGenerationError(
                    f"unsupported reference image extension: {reference_image.suffix!r}"
                )

        preflight = self.preflight()
        experiment = self._new_experiment_dir(request.experiment_root.expanduser().resolve())
        metadata_path = experiment / "metadata.json"
        result_path = experiment / "articraft-result.json"
        payload = {
            "description": request.description,
            "reference_image": str(reference_image) if reference_image else None,
            "provider": request.provider,
            "model": request.model,
            "output_dir": str(experiment / "runs"),
        }
        command = [
            str(_absolute_without_resolving(self.config.python_executable)),
            "-c",
            _RUNNER,
            json.dumps(payload, sort_keys=True),
            str(result_path),
        ]
        metadata: dict[str, Any] = {
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": asdict(request),
            "backend": "mini-articraft",
            "preflight": preflight,
            "command": command,
            "provenance": self._project_provenance(),
            "limitations": [
                "The generated USDZ is an asset candidate, not a verified handover scene.",
                "Generation is provider-dependent and is not claimed to be deterministic.",
                "The asset must pass conversion, collision, contact, and handover physics gates.",
            ],
        }
        _write_json(metadata_path, metadata)

        environment = os.environ.copy()
        try:
            self._execute(command, experiment, experiment / "articraft.log", environment)
            if not result_path.is_file():
                raise AssetGenerationError(f"ArtiCraft did not write its result: {result_path}")
            raw_result = json.loads(result_path.read_text(encoding="utf-8"))
            if str(raw_result.get("status", "")).lower() not in {"success", "succeeded"}:
                raise AssetGenerationError(
                    f"ArtiCraft reported unsuccessful status: {raw_result.get('status')!r}"
                )
            artifact = Path(raw_result["artifact"]).expanduser()
            run_dir = Path(raw_result["run_dir"]).expanduser()
            if not artifact.is_absolute():
                artifact = experiment / artifact
            if not run_dir.is_absolute():
                run_dir = experiment / run_dir
            artifact = artifact.resolve()
            run_dir = run_dir.resolve()
            if not _inside(artifact, experiment) or not _inside(run_dir, experiment):
                raise AssetGenerationError("ArtiCraft outputs escaped the isolated experiment directory")
            if (
                not artifact.is_file()
                or artifact.stat().st_size == 0
                or artifact.suffix.lower() != ".usdz"
            ):
                raise AssetGenerationError(f"ArtiCraft output is not a non-empty USDZ: {artifact}")
            if not run_dir.is_dir():
                raise AssetGenerationError(f"ArtiCraft run directory is missing: {run_dir}")
            metadata.update(
                {
                    "status": "succeeded",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "artifact": str(artifact),
                    "artifact_format": "usdz",
                    "upstream_run_dir": str(run_dir),
                }
            )
            _write_json(metadata_path, metadata)
        except Exception as exc:
            metadata.update(
                {
                    "status": "failed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error": repr(exc),
                }
            )
            _write_json(metadata_path, metadata)
            raise

        return AssetGenerationResult(
            artifact=artifact,
            artifact_format="usdz",
            upstream_run_dir=run_dir,
            experiment_dir=experiment,
            metadata=metadata_path,
        )

    def compile_model(self, request: AssetCompilationRequest) -> AssetGenerationResult:
        """Compile a trusted, authored SDK model without invoking an LLM provider."""

        model_file = request.model_file.expanduser().resolve()
        if not model_file.is_file() or model_file.stat().st_size == 0:
            raise AssetGenerationError(f"ArtiCraft model file does not exist or is empty: {model_file}")
        if model_file.suffix.lower() != ".py":
            raise AssetGenerationError(f"ArtiCraft model must be a Python file: {model_file}")

        preflight = self.preflight()
        experiment = self._new_experiment_dir(request.experiment_root.expanduser().resolve())
        metadata_path = experiment / "metadata.json"
        result_path = experiment / "articraft-result.json"
        command = [
            str(_absolute_without_resolving(self.config.python_executable)),
            "-c",
            _COMPILER,
            str(model_file),
            str(experiment / "runs"),
            str(result_path),
        ]
        metadata: dict[str, Any] = {
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": asdict(request),
            "backend": "mini-articraft-sdk",
            "preflight": preflight,
            "command": command,
            "provenance": self._project_provenance(),
            "limitations": [
                "Authored model.py code executes in the upstream isolated compile worker.",
                "The generated USDZ is an asset candidate, not a verified handover scene.",
                "The asset must pass conversion, collision, contact, and handover physics gates.",
            ],
        }
        _write_json(metadata_path, metadata)

        try:
            self._execute(command, experiment, experiment / "articraft-compile.log", os.environ.copy())
            if not result_path.is_file():
                raise AssetGenerationError(f"ArtiCraft did not write its result: {result_path}")
            raw_result = json.loads(result_path.read_text(encoding="utf-8"))
            if raw_result.get("status") != "success":
                raise AssetGenerationError(
                    "ArtiCraft compilation failed: "
                    f"{raw_result.get('error') or raw_result.get('compile_report')}"
                )
            artifact = Path(raw_result["artifact"]).expanduser()
            run_dir = Path(raw_result["run_dir"]).expanduser()
            if not artifact.is_absolute():
                artifact = experiment / artifact
            if not run_dir.is_absolute():
                run_dir = experiment / run_dir
            artifact = artifact.resolve()
            run_dir = run_dir.resolve()
            if not _inside(artifact, experiment) or not _inside(run_dir, experiment):
                raise AssetGenerationError("ArtiCraft outputs escaped the isolated experiment directory")
            if (
                not artifact.is_file()
                or artifact.stat().st_size == 0
                or artifact.suffix.lower() != ".usdz"
            ):
                raise AssetGenerationError(f"ArtiCraft output is not a non-empty USDZ: {artifact}")
            metadata.update(
                {
                    "status": "succeeded",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "artifact": str(artifact),
                    "artifact_format": "usdz",
                    "upstream_run_dir": str(run_dir),
                    "compile_report": raw_result.get("compile_report"),
                }
            )
            _write_json(metadata_path, metadata)
        except Exception as exc:
            metadata.update(
                {
                    "status": "failed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error": repr(exc),
                }
            )
            _write_json(metadata_path, metadata)
            raise

        return AssetGenerationResult(
            artifact=artifact,
            artifact_format="usdz",
            upstream_run_dir=run_dir,
            experiment_dir=experiment,
            metadata=metadata_path,
        )
