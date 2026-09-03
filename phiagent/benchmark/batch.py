"""Dependency-light, resumable batch orchestration for benchmark workers.

The controller executes explicit argv arrays and never invokes a shell. Heavy
models and simulators remain external worker commands that emit immutable
artifacts and optional ``record-patch.json`` evidence fragments.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from phiagent.benchmark.artifacts import ArtifactRecord, ArtifactStore, sha256_file
from phiagent.benchmark.schema import BenchmarkSuite, Submission, SubmissionRecord


BATCH_SCHEMA_VERSION = "0.2.0"
JOB_STATES = {"planned", "running", "succeeded", "failed", "blocked"}
PATCH_KEYS = {
    "generated_uri",
    "visual",
    "geometry",
    "action",
    "simulations",
    "real_trials",
    "runtime",
    "policy_utility",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _identifier(value: object, label: str) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", text):
        raise ValueError(f"invalid {label}: {value!r}")
    return text


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "job"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_state(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        diff = run("diff", "--binary", "HEAD")
        return {
            "revision": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "status_short": run("status", "--short").splitlines(),
            "working_tree_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}


def _package_inventory() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[str(name)] = distribution.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


@dataclass(frozen=True)
class StageResources:
    gpus: int = 0
    cpus: int = 1
    gpu_memory_gb: float | None = None

    @classmethod
    def from_dict(cls, payload: object) -> "StageResources":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("stage resources must be an object")
        resources = cls(
            gpus=int(payload.get("gpus", 0)),
            cpus=int(payload.get("cpus", 1)),
            gpu_memory_gb=(
                float(payload["gpu_memory_gb"])
                if payload.get("gpu_memory_gb") is not None
                else None
            ),
        )
        if resources.gpus < 0 or resources.cpus <= 0:
            raise ValueError("stage resources require gpus >= 0 and cpus > 0")
        if resources.gpu_memory_gb is not None and resources.gpu_memory_gb <= 0:
            raise ValueError("stage gpu_memory_gb must be positive")
        return resources

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "gpus": self.gpus,
                "cpus": self.cpus,
                "gpu_memory_gb": self.gpu_memory_gb,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class StageTemplate:
    name: str
    command: tuple[str, ...]
    depends_on: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    environment: dict[str, str]
    resources: StageResources

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageTemplate":
        name = _identifier(payload["name"], "stage name")
        command = payload.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise ValueError(f"stage {name} command must be a non-empty argv array")
        depends = tuple(_identifier(item, "stage dependency") for item in payload.get("depends_on", ()))
        outputs = tuple(str(item) for item in payload.get("expected_outputs", ()))
        for output in outputs:
            path = Path(output)
            if path.is_absolute() or ".." in path.parts or not output:
                raise ValueError(f"stage {name} expected output must be job-relative: {output}")
        raw_environment = payload.get("environment", {})
        if not isinstance(raw_environment, Mapping):
            raise ValueError(f"stage {name} environment must be an object")
        return cls(
            name=name,
            command=tuple(command),
            depends_on=depends,
            expected_outputs=outputs,
            environment={str(key): str(value) for key, value in raw_environment.items()},
            resources=StageResources.from_dict(payload.get("resources")),
        )


@dataclass(frozen=True)
class MethodManifest:
    method: str
    candidates_per_case: int
    seed: int
    working_directory: Path
    stages: tuple[StageTemplate, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, root: Path) -> "MethodManifest":
        if payload.get("schema_version") != BATCH_SCHEMA_VERSION:
            raise ValueError("method manifest schema_version must be 0.2.0")
        count = int(payload.get("candidates_per_case", 1))
        if count <= 0:
            raise ValueError("candidates_per_case must be positive")
        raw_workdir = Path(str(payload.get("working_directory", root)))
        workdir = raw_workdir if raw_workdir.is_absolute() else root / raw_workdir
        workdir = workdir.expanduser().resolve()
        if not workdir.is_dir():
            raise ValueError(f"method working directory does not exist: {workdir}")
        stages = tuple(StageTemplate.from_dict(item) for item in payload["stages"])
        names = [stage.name for stage in stages]
        if not stages or len(names) != len(set(names)):
            raise ValueError("method stages must be non-empty and uniquely named")
        known: set[str] = set()
        for stage in stages:
            if any(dependency not in known for dependency in stage.depends_on):
                raise ValueError(
                    f"stage {stage.name} depends on an unknown or later stage"
                )
            known.add(stage.name)
        return cls(
            method=_identifier(payload["method"], "method"),
            candidates_per_case=count,
            seed=int(payload.get("seed", 0)),
            working_directory=workdir,
            stages=stages,
        )


def _candidate_seed(base: int, case_id: str, candidate_index: int) -> int:
    digest = hashlib.sha256(f"{base}:{case_id}:{candidate_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def plan_batch_run(
    *, suite_path: Path, method_path: Path, output_dir: Path
) -> dict[str, Any]:
    suite_path = suite_path.expanduser().resolve()
    method_path = method_path.expanduser().resolve()
    run_dir = output_dir.expanduser().resolve()
    if run_dir.exists():
        raise ValueError(f"batch run directory already exists: {run_dir}")
    suite_payload = _read_json(suite_path)
    method_payload = _read_json(method_path)
    suite = BenchmarkSuite.from_dict(suite_payload)
    method = MethodManifest.from_dict(method_payload, root=method_path.parent)
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "suite.json", suite_payload)
    resolved_method_payload = dict(method_payload)
    resolved_method_payload["working_directory"] = str(method.working_directory)
    _write_json(run_dir / "method.json", resolved_method_payload)

    jobs: list[dict[str, Any]] = []
    for case in suite.cases:
        case_path = run_dir / "cases" / f"{_safe_component(case.case_id)}.json"
        _write_json(case_path, case.to_dict())
        for candidate_index in range(method.candidates_per_case):
            candidate_id = f"{case.case_id}:candidate-{candidate_index:03d}"
            seed = _candidate_seed(method.seed, case.case_id, candidate_index)
            stage_job_ids: dict[str, str] = {}
            for stage in method.stages:
                job_id = _safe_component(f"{case.case_id}--c{candidate_index:03d}--{stage.name}")
                job_dir = run_dir / "jobs" / job_id
                values = {
                    "run_dir": str(run_dir),
                    "job_dir": str(job_dir),
                    "case_json": str(case_path),
                    "case_id": case.case_id,
                    "candidate_id": candidate_id,
                    "candidate_index": str(candidate_index),
                    "generation_seed": str(seed),
                    "stage": stage.name,
                }
                try:
                    command = [item.format_map(values) for item in stage.command]
                    environment = {
                        key: value.format_map(values) for key, value in stage.environment.items()
                    }
                except KeyError as exc:
                    raise ValueError(
                        f"unknown placeholder in stage {stage.name}: {exc.args[0]}"
                    ) from exc
                dependencies = [stage_job_ids[name] for name in stage.depends_on]
                stage_job_ids[stage.name] = job_id
                job = {
                    "schema_version": BATCH_SCHEMA_VERSION,
                    "job_id": job_id,
                    "case_id": case.case_id,
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "generation_seed": seed,
                    "stage": stage.name,
                    "dependencies": dependencies,
                    "command": command,
                    "environment": environment,
                    "working_directory": str(method.working_directory),
                    "job_dir": str(job_dir),
                    "expected_outputs": list(stage.expected_outputs),
                    "resources": stage.resources.to_dict(),
                }
                _write_json(job_dir / "job.json", job)
                _write_json(
                    job_dir / "status.json",
                    {"state": "planned", "updated_at": _utc_now(), "attempts": 0},
                )
                jobs.append(job)

    manifest = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "suite": suite.name,
        "method": method.method,
        "created_at": _utc_now(),
        "suite_sha256": sha256_file(suite_path),
        "method_sha256": sha256_file(method_path),
        "case_count": len(suite.cases),
        "candidate_count": len(suite.cases) * method.candidates_per_case,
        "job_count": len(jobs),
        "jobs": [job["job_id"] for job in jobs],
        "provenance": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "argv": sys.argv,
            "git": _git_state(method.working_directory),
            "packages": _package_inventory(),
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    return manifest


class BatchController:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.expanduser().resolve()
        self.manifest = _read_json(self.run_dir / "manifest.json")
        self.store = ArtifactStore(self.run_dir / "artifacts" / "sha256")
        self.jobs = {
            job_id: _read_json(self.run_dir / "jobs" / job_id / "job.json")
            for job_id in self.manifest["jobs"]
        }

    def _status_path(self, job_id: str) -> Path:
        return self.run_dir / "jobs" / job_id / "status.json"

    def _status(self, job_id: str) -> dict[str, Any]:
        status = _read_json(self._status_path(job_id))
        if status.get("state") not in JOB_STATES:
            raise ValueError(f"unknown job state for {job_id}: {status.get('state')}")
        return status

    def _verify_succeeded(self, job_id: str) -> None:
        status = self._status(job_id)
        records = [ArtifactRecord(**item) for item in status.get("artifacts", ())]
        if not records and self.jobs[job_id]["expected_outputs"]:
            raise ValueError(f"succeeded job lacks artifact records: {job_id}")
        if not all(ArtifactStore.verify(record) for record in records):
            raise ValueError(f"succeeded job artifact changed or disappeared: {job_id}")
        for record in status.get("logs", ()):
            path = Path(record["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(record["bytes"])
                or sha256_file(path) != record["sha256"]
            ):
                raise ValueError(f"succeeded job log changed or disappeared: {job_id}")

    @staticmethod
    def _gpu_selection(
        job: Mapping[str, Any], assigned_gpus: tuple[str, ...]
    ) -> dict[str, Any]:
        requested = int(job.get("resources", {}).get("gpus", 0))
        configured = (
            ",".join(assigned_gpus)
            if assigned_gpus
            else str(
                job.get("environment", {}).get(
                    "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "")
                )
            ).strip()
        )
        if requested:
            selected = [item.strip() for item in configured.split(",") if item.strip()]
            if len(selected) != requested:
                raise ValueError(
                    f"GPU stage requests {requested} devices but CUDA_VISIBLE_DEVICES "
                    f"selects {len(selected)}"
                )
        executable = "nvidia-smi"
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        ) if requested else None
        if requested and (completed is None or completed.returncode != 0):
            raise ValueError("GPU stage cannot inspect physical devices with nvidia-smi")
        return {
            "requested_gpus": requested,
            "scheduler_assigned": bool(assigned_gpus),
            "cuda_visible_devices": configured or None,
            "physical_inventory": (
                [line.strip() for line in completed.stdout.splitlines() if line.strip()]
                if completed is not None
                else []
            ),
        }

    def _execute(self, job_id: str, assigned_gpus: tuple[str, ...] = ()) -> str:
        job = self.jobs[job_id]
        job_dir = Path(job["job_dir"])
        status = self._status(job_id)
        attempts = int(status.get("attempts", 0)) + 1
        _write_json(
            self._status_path(job_id),
            {
                "state": "running",
                "updated_at": _utc_now(),
                "attempts": attempts,
                "started_at": _utc_now(),
            },
        )
        environment = os.environ.copy()
        environment.update(job["environment"])
        if assigned_gpus:
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(assigned_gpus)
        try:
            gpu_selection = self._gpu_selection(job, assigned_gpus)
        except (OSError, ValueError) as exc:
            _write_json(
                self._status_path(job_id),
                {
                    "state": "failed",
                    "updated_at": _utc_now(),
                    "attempts": attempts,
                    "reason": f"resource_validation_failed: {exc}",
                },
            )
            return "failed"
        started = datetime.now(timezone.utc)
        completed = subprocess.run(
            job["command"],
            cwd=job["working_directory"],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        (job_dir / "stdout.log").write_text(completed.stdout)
        (job_dir / "stderr.log").write_text(completed.stderr)
        logs = [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (job_dir / "stdout.log", job_dir / "stderr.log")
        ]
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        if completed.returncode != 0:
            _write_json(
                self._status_path(job_id),
                {
                    "state": "failed",
                    "updated_at": _utc_now(),
                    "attempts": attempts,
                    "returncode": completed.returncode,
                    "elapsed_seconds": elapsed,
                    "gpu_selection": gpu_selection,
                    "logs": logs,
                },
            )
            return "failed"
        artifacts: list[dict[str, Any]] = []
        try:
            for relative in job["expected_outputs"]:
                output = (job_dir / relative).resolve()
                if job_dir.resolve() not in output.parents:
                    raise ValueError(f"job output escapes its directory: {relative}")
                artifacts.append(self.store.add(output).to_dict())
        except (OSError, ValueError) as exc:
            _write_json(
                self._status_path(job_id),
                {
                    "state": "failed",
                    "updated_at": _utc_now(),
                    "attempts": attempts,
                    "returncode": 0,
                    "elapsed_seconds": elapsed,
                    "reason": f"artifact_validation_failed: {exc}",
                    "gpu_selection": gpu_selection,
                    "logs": logs,
                },
            )
            return "failed"
        _write_json(
            self._status_path(job_id),
            {
                "state": "succeeded",
                "updated_at": _utc_now(),
                "attempts": attempts,
                "returncode": 0,
                "elapsed_seconds": elapsed,
                "artifacts": artifacts,
                "gpu_selection": gpu_selection,
                "logs": logs,
            },
        )
        return "succeeded"

    def run(
        self,
        *,
        max_workers: int = 1,
        retry_failed: bool = False,
        gpu_devices: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if any(not device.strip() for device in gpu_devices) or len(set(gpu_devices)) != len(
            gpu_devices
        ):
            raise ValueError("gpu_devices must contain unique non-empty physical identifiers")
        if gpu_devices:
            oversized = [
                job_id
                for job_id, job in self.jobs.items()
                if int(job.get("resources", {}).get("gpus", 0)) > len(gpu_devices)
            ]
            if oversized:
                raise ValueError(f"jobs request more GPUs than the supplied pool: {oversized}")
        for job_id in self.jobs:
            state = self._status(job_id)["state"]
            if state == "running":
                previous = self._status(job_id)
                _write_json(
                    self._status_path(job_id),
                    {
                        "state": "failed",
                        "updated_at": _utc_now(),
                        "attempts": int(previous.get("attempts", 0)),
                        "reason": "interrupted_before_resume",
                    },
                )
            elif state == "succeeded":
                self._verify_succeeded(job_id)
            elif state in {"blocked", "failed"} and retry_failed:
                previous = self._status(job_id)
                _write_json(
                    self._status_path(job_id),
                    {
                        "state": "planned",
                        "updated_at": _utc_now(),
                        "attempts": int(previous.get("attempts", 0)),
                        "reason": "retry_requested",
                    },
                )
        while True:
            states = {job_id: self._status(job_id)["state"] for job_id in self.jobs}
            waiting = {
                job_id
                for job_id, state in states.items()
                if state == "planned"
            }
            ready = [
                job_id
                for job_id in sorted(waiting)
                if all(states[dependency] == "succeeded" for dependency in self.jobs[job_id]["dependencies"])
            ]
            if ready:
                assignments: dict[str, tuple[str, ...]] = {}
                available = list(gpu_devices)
                selected_ready: list[str] = []
                for job_id in ready:
                    requested = int(
                        self.jobs[job_id].get("resources", {}).get("gpus", 0)
                    )
                    if gpu_devices and requested > len(available):
                        continue
                    assigned = tuple(available[:requested]) if gpu_devices else ()
                    if assigned:
                        del available[:requested]
                    selected_ready.append(job_id)
                    assignments[job_id] = assigned
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(self._execute, job_id, assignments[job_id]): job_id
                        for job_id in selected_ready
                    }
                    for future in as_completed(futures):
                        future.result()
                continue
            for job_id in waiting:
                dependencies = self.jobs[job_id]["dependencies"]
                if any(states[dependency] in {"failed", "blocked"} for dependency in dependencies):
                    previous = self._status(job_id)
                    _write_json(
                        self._status_path(job_id),
                        {
                            "state": "blocked",
                            "updated_at": _utc_now(),
                            "attempts": int(previous.get("attempts", 0)),
                            "reason": "dependency_failed",
                        },
                    )
            break
        return self.status()

    def status(self) -> dict[str, Any]:
        states = {job_id: self._status(job_id)["state"] for job_id in self.jobs}
        counts = {state: sum(value == state for value in states.values()) for state in sorted(JOB_STATES)}
        return {
            "schema_version": BATCH_SCHEMA_VERSION,
            "run_id": self.manifest["run_id"],
            "job_count": len(states),
            "counts": counts,
            "complete": counts["succeeded"] == len(states),
            "jobs": states,
        }


def compile_submission(
    *, run_dir: Path, output: Path, selection_path: Path | None = None
) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    target = output.expanduser().resolve()
    if target.exists():
        raise ValueError(f"compiled submission already exists: {target}")
    manifest = _read_json(root / "manifest.json")
    suite = BenchmarkSuite.from_json(root / "suite.json")
    method = MethodManifest.from_dict(_read_json(root / "method.json"), root=root)
    selection = _read_json(selection_path.expanduser().resolve()) if selection_path else {}
    selected = selection.get("selected_candidates", {})
    if not isinstance(selected, Mapping):
        raise ValueError("selected_candidates must be an object")
    records: list[SubmissionRecord] = []
    controller = BatchController(root)
    for case in suite.cases:
        candidate_index = int(selected.get(case.case_id, 0))
        if not 0 <= candidate_index < method.candidates_per_case:
            raise ValueError(f"invalid selected candidate for {case.case_id}")
        candidate_id = f"{case.case_id}:candidate-{candidate_index:03d}"
        jobs = [
            job
            for job in controller.jobs.values()
            if job["case_id"] == case.case_id and job["candidate_id"] == candidate_id
        ]
        if not jobs or any(controller._status(job["job_id"])["state"] != "succeeded" for job in jobs):
            raise ValueError(f"selected candidate jobs are incomplete: {candidate_id}")
        merged: dict[str, Any] = {
            "case_id": case.case_id,
            "candidate_id": candidate_id,
            "candidate_index": candidate_index,
            "generation_seed": jobs[0]["generation_seed"],
        }
        stage_order = {stage.name: index for index, stage in enumerate(method.stages)}
        for job in sorted(jobs, key=lambda item: stage_order[item["stage"]]):
            patch_path = Path(job["job_dir"]) / "record-patch.json"
            if not patch_path.is_file():
                continue
            if "record-patch.json" not in job["expected_outputs"]:
                raise ValueError(
                    f"record patch was not declared as an immutable output: {patch_path}"
                )
            patch = _read_json(patch_path)
            if patch.get("case_id", case.case_id) != case.case_id:
                raise ValueError(f"record patch case mismatch: {patch_path}")
            unknown = set(patch) - PATCH_KEYS - {"case_id"}
            if unknown:
                raise ValueError(f"record patch contains unsupported keys: {sorted(unknown)}")
            for key, value in patch.items():
                if key == "case_id":
                    continue
                if key in {"simulations", "real_trials"}:
                    if not isinstance(value, list):
                        raise ValueError(f"{key} patch must be an array")
                    merged.setdefault(key, []).extend(value)
                elif key in merged:
                    raise ValueError(f"record patch attempts to overwrite {key}: {patch_path}")
                else:
                    merged[key] = value
        if not str(merged.get("generated_uri", "")).strip():
            raise ValueError(f"selected candidate lacks generated_uri: {candidate_id}")
        records.append(SubmissionRecord.from_dict(merged))
    submission = Submission(
        method=method.method,
        suite_name=suite.name,
        records=tuple(records),
        metadata={
            "batch_run_id": manifest["run_id"],
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "selection": dict(selected),
        },
        schema_version=BATCH_SCHEMA_VERSION,
    )
    payload = submission.to_dict()
    _write_json(target, payload)
    return payload
