#!/usr/bin/env python3
"""Run a guarded batch of demo-video generator/evaluator command adapters."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TextIO
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.training.demo_factory import (  # noqa: E402
    DemoFactoryPolicy,
    FactoryContract,
    FactoryRecord,
    assess_record,
)


_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
_ALLOWED_PLACEHOLDERS = {"case_manifest", "recipe_manifest", "attempt_dir", "seed"}


class GPUPreflightError(RuntimeError):
    """Raised before any worker runs when physical GPU selection is unsafe."""


@dataclass(frozen=True)
class GPUInfo:
    physical_index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int


def _parse_gpu_inventory(output: str) -> tuple[GPUInfo, ...]:
    rows = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) != 5:
            raise GPUPreflightError(f"unexpected nvidia-smi row: {row!r}")
        try:
            rows.append(
                GPUInfo(
                    physical_index=int(row[0].strip()),
                    name=row[1].strip(),
                    total_mib=int(row[2].strip()),
                    used_mib=int(row[3].strip()),
                    free_mib=int(row[4].strip()),
                )
            )
        except ValueError as error:
            raise GPUPreflightError(f"could not parse nvidia-smi row: {row!r}") from error
    if not rows:
        raise GPUPreflightError("nvidia-smi reported no physical GPUs")
    return tuple(rows)


def query_gpus() -> tuple[tuple[GPUInfo, ...], str, str]:
    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if inventory.returncode != 0:
        raise GPUPreflightError(
            "nvidia-smi inventory failed: " + (inventory.stderr.strip() or "not available")
        )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    process_text = processes.stdout if processes.returncode == 0 else processes.stderr
    return _parse_gpu_inventory(inventory.stdout), inventory.stdout, process_text


def select_gpu(
    gpus: Sequence[GPUInfo], requested_index: int | None, minimum_free_mib: int
) -> GPUInfo:
    if requested_index is not None:
        selected = next(
            (gpu for gpu in gpus if gpu.physical_index == requested_index), None
        )
        if selected is None:
            raise GPUPreflightError(f"requested physical GPU {requested_index} was not reported")
        if selected.free_mib < minimum_free_mib:
            raise GPUPreflightError(
                f"physical GPU {requested_index} has {selected.free_mib} MiB free; "
                f"{minimum_free_mib} MiB is required"
            )
        return selected
    eligible = [gpu for gpu in gpus if gpu.free_mib >= minimum_free_mib]
    if not eligible:
        raise GPUPreflightError("no physical GPU satisfies minimum_free_gpu_mib")
    return max(eligible, key=lambda gpu: gpu.free_mib)


def acquire_gpu_lease(physical_index: int) -> tuple[Path, TextIO]:
    lease_path = Path("/tmp") / f"phiagent-gpu-{physical_index}.lock"
    lease = lease_path.open("a+", encoding="utf-8")
    fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
    return lease_path, lease


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_id(value: object, name: str) -> str:
    identifier = str(value)
    if not _SAFE_ID.fullmatch(identifier):
        raise ValueError(f"{name} is not filesystem safe: {identifier!r}")
    return identifier


def _new_experiment(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = root / f"{stamp}-{uuid4().hex[:8]}"
    experiment.mkdir()
    return experiment


def _git_state() -> dict[str, object]:
    state: dict[str, object] = {}
    for name, command in {
        "head": ["git", "rev-parse", "--verify", "HEAD"],
        "tracked_changes": ["git", "status", "--short", "--untracked-files=no"],
    }.items():
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            state[name] = (
                completed.stdout.strip()
                if name == "head"
                else completed.stdout.splitlines()
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            state[name] = f"unavailable: {error}"
    return state


def _load_campaign(path: Path) -> tuple[
    dict[str, object],
    FactoryContract,
    dict[str, dict[str, object]],
    tuple[dict[str, object], ...],
    dict[str, object],
]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("campaign must contain one JSON object")
    if payload.get("schema_version") != "1.0.0":
        raise ValueError("unsupported demo factory campaign schema")
    _safe_id(payload.get("campaign_id"), "campaign_id")
    contract_payload = payload.get("contract")
    if not isinstance(contract_payload, dict):
        raise ValueError("campaign.contract must be a JSON object")
    contract = FactoryContract.from_dict(contract_payload)

    recipes_payload = payload.get("recipes")
    if not isinstance(recipes_payload, list) or not recipes_payload:
        raise ValueError("campaign.recipes must be a non-empty JSON array")
    recipes: dict[str, dict[str, object]] = {}
    for raw in recipes_payload:
        if not isinstance(raw, dict):
            raise ValueError("every recipe must be a JSON object")
        recipe_id = _safe_id(raw.get("recipe_id"), "recipe_id")
        command = raw.get("command")
        parameters = raw.get("parameters", {})
        if not isinstance(command, list) or not command or any(
            not isinstance(token, str) or not token for token in command
        ):
            raise ValueError(f"recipe {recipe_id} command must be a string array")
        if not isinstance(parameters, dict):
            raise ValueError(f"recipe {recipe_id} parameters must be a JSON object")
        placeholders = {
            match.group(1) for token in command for match in _PLACEHOLDER.finditer(token)
        }
        unknown = placeholders - _ALLOWED_PLACEHOLDERS
        if unknown:
            raise ValueError(f"recipe {recipe_id} has unknown placeholders: {sorted(unknown)}")
        if "attempt_dir" not in placeholders:
            raise ValueError(f"recipe {recipe_id} command must use {{attempt_dir}}")
        try:
            cost = float(raw.get("estimated_cost_units", 1.0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"recipe {recipe_id} cost must be numeric") from error
        if cost < 0:
            raise ValueError(f"recipe {recipe_id} cost must be non-negative")
        if recipe_id in recipes:
            raise ValueError(f"duplicate recipe {recipe_id}")
        recipes[recipe_id] = {
            "recipe_id": recipe_id,
            "command": command,
            "parameters": parameters,
            "estimated_cost_units": cost,
        }
    if tuple(recipes) != contract.recipe_order:
        raise ValueError("campaign recipe order must exactly match contract.recipe_order")

    cases_payload = payload.get("cases")
    if not isinstance(cases_payload, list) or not cases_payload:
        raise ValueError("campaign.cases must be a non-empty JSON array")
    cases = []
    episode_ids: set[str] = set()
    for raw in cases_payload:
        if not isinstance(raw, dict):
            raise ValueError("every case must be a JSON object")
        episode_id = _safe_id(raw.get("episode_id"), "episode_id")
        group_id = _safe_id(raw.get("group_id"), "group_id")
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode {episode_id}")
        episode_ids.add(episode_id)
        domain = str(raw.get("domain", contract.domain))
        if domain != contract.domain:
            raise ValueError(f"case {episode_id} has the wrong domain")
        manifest_value = raw.get("manifest")
        if not isinstance(manifest_value, str):
            raise ValueError(f"case {episode_id} requires a manifest path")
        manifest = Path(manifest_value).expanduser()
        if not manifest.is_absolute():
            manifest = (path.parent / manifest).resolve()
        if not manifest.is_file() or manifest.stat().st_size == 0:
            raise ValueError(f"case manifest does not exist or is empty: {manifest}")
        seed = raw.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"case {episode_id} seed must be a non-negative integer")
        cases.append(
            {
                "episode_id": episode_id,
                "group_id": group_id,
                "domain": domain,
                "manifest": manifest,
                "seed": seed,
            }
        )

    execution = payload.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("campaign.execution must be a JSON object")
    device = execution.get("device", "cpu")
    if device not in {"cpu", "gpu"}:
        raise ValueError("execution.device must be cpu or gpu")
    maximum_attempts = execution.get("maximum_attempts_per_episode", len(recipes))
    if not isinstance(maximum_attempts, int) or not 1 <= maximum_attempts <= len(recipes):
        raise ValueError("maximum_attempts_per_episode is outside the recipe count")
    collect_all = execution.get("collect_all_recipes", False)
    if not isinstance(collect_all, bool):
        raise ValueError("collect_all_recipes must be boolean")
    if collect_all and maximum_attempts != len(recipes):
        raise ValueError("collect_all_recipes requires maximum_attempts_per_episode=recipe count")
    execution = {
        **execution,
        "device": device,
        "maximum_attempts_per_episode": maximum_attempts,
        "collect_all_recipes": collect_all,
    }
    return payload, contract, recipes, tuple(cases), execution


def _render_command(command: list[str], values: Mapping[str, str]) -> list[str]:
    rendered = []
    for token in command:
        value = token
        for name, replacement in values.items():
            value = value.replace("{" + name + "}", replacement)
        rendered.append(value)
    return rendered


def _last_json_object(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("worker stdout did not end with a JSON object")


def _run_attempt(
    *,
    case: Mapping[str, object],
    recipe: Mapping[str, object],
    attempt_dir: Path,
    environment: Mapping[str, str],
    contract: FactoryContract,
    baseline_metrics: Mapping[str, float] | None,
) -> tuple[FactoryRecord, dict[str, object]]:
    attempt_dir.mkdir(parents=True)
    case_snapshot = attempt_dir / "case.json"
    recipe_snapshot = attempt_dir / "recipe.json"
    original_manifest = Path(case["manifest"])  # type: ignore[arg-type]
    manifest_snapshot = attempt_dir / f"input-manifest{original_manifest.suffix}"
    shutil.copy2(original_manifest, manifest_snapshot)
    _write_json(
        case_snapshot,
        {
            **{key: value for key, value in case.items() if key != "manifest"},
            "manifest": str(original_manifest),
            "manifest_sha256": _sha256(original_manifest),
            "manifest_snapshot": str(manifest_snapshot),
        },
    )
    _write_json(recipe_snapshot, recipe)
    command = _render_command(
        recipe["command"],  # type: ignore[arg-type]
        {
            "case_manifest": str(manifest_snapshot),
            "recipe_manifest": str(recipe_snapshot),
            "attempt_dir": str(attempt_dir),
            "seed": str(case["seed"]),
        },
    )
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )
    wall_seconds = time.monotonic() - started
    (attempt_dir / "stdout.log").write_text(completed.stdout)
    (attempt_dir / "stderr.log").write_text(completed.stderr)
    command_payload = {
        "command": command,
        "returncode": completed.returncode,
        "wall_seconds": wall_seconds,
    }
    _write_json(attempt_dir / "command.json", command_payload)
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed with exit {completed.returncode}; see {attempt_dir / 'stderr.log'}"
        )
    payload = _last_json_object(completed.stdout)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("worker result requires a metrics object")
    required = set(dict(contract.metric_weights))
    if not required.issubset(metrics):
        raise ValueError(f"worker metrics are missing {sorted(required - set(metrics))}")
    numeric_metrics = {str(key): float(value) for key, value in metrics.items()}
    context = (
        {field: numeric_metrics[field] for field in contract.context_fields}
        if baseline_metrics is None
        else {field: float(baseline_metrics[field]) for field in contract.context_fields}
    )
    video_value = payload.get("video")
    if not isinstance(video_value, str):
        raise ValueError("worker result requires a video path")
    video = Path(video_value).expanduser()
    if not video.is_absolute():
        video = (attempt_dir / video).resolve()
    if not video.is_relative_to(attempt_dir.resolve()):
        raise ValueError("worker video must be created inside its immutable attempt directory")
    if video.suffix.lower() != ".mp4" or not video.is_file() or video.stat().st_size == 0:
        raise ValueError(f"worker video is missing, empty, or not MP4: {video}")
    human = payload.get("human_review_passed")
    if human not in {True, False, None}:
        raise ValueError("worker human_review_passed must be true, false, or null")
    diagnoses = payload.get("diagnoses", [])
    if not isinstance(diagnoses, list) or any(not isinstance(value, str) for value in diagnoses):
        raise ValueError("worker diagnoses must be a JSON string array")
    cost_units = float(payload.get("cost_units", recipe["estimated_cost_units"]))
    record = FactoryRecord(
        episode_id=str(case["episode_id"]),
        group_id=str(case["group_id"]),
        domain=str(case["domain"]),
        recipe_id=str(recipe["recipe_id"]),
        recipe_parameters=recipe["parameters"],  # type: ignore[arg-type]
        context=context,
        metrics=numeric_metrics,
        cost_units=cost_units,
        human_review_passed=human,  # type: ignore[arg-type]
        video=str(video),
        video_sha256=_sha256(video),
        diagnoses=tuple(diagnoses),
    )
    return record, {**command_payload, "worker_result": payload}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--experiment-root", type=Path, default=Path("outputs/demo-video-factory")
    )
    parser.add_argument("--allow-unpromoted-policy", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    campaign_path = args.campaign.expanduser().resolve()
    if not campaign_path.is_file():
        raise ValueError(f"campaign does not exist: {campaign_path}")
    campaign, contract, recipes, cases, execution = _load_campaign(campaign_path)
    policy = None
    policy_path = None
    if args.policy is not None:
        policy_path = args.policy.expanduser().resolve()
        policy = DemoFactoryPolicy.load(policy_path)
        if policy.domain != contract.domain or policy.contract_sha256 != contract.fingerprint:
            raise ValueError("policy domain or quality contract does not match the campaign")
        if set(policy.recipe_ids) != set(recipes):
            raise ValueError("policy recipe set does not match the campaign")
        if not policy.promoted and not args.allow_unpromoted_policy:
            raise ValueError("refusing an unpromoted router; use --allow-unpromoted-policy only for research")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "campaign_id": campaign["campaign_id"],
                    "cases": len(cases),
                    "recipes": list(recipes),
                    "device": execution["device"],
                    "policy": str(policy_path) if policy_path else None,
                    "valid": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    experiment = _new_experiment(args.experiment_root.expanduser().resolve())
    manifest_path = experiment / "manifest.json"
    dataset_path = experiment / "episodes.jsonl"
    accepted_index_path = experiment / "accepted-video-index.json"
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "guarded_agentic_demo_video_batch_factory",
        "campaign_id": campaign["campaign_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "preflight",
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": _git_state(),
        "gpu": {"status": "pending"},
        "campaign": {
            "path": str(campaign_path),
            "sha256": _sha256(campaign_path),
            "contract_sha256": contract.fingerprint,
        },
        "policy": None
        if policy_path is None
        else {
            "path": str(policy_path),
            "sha256": _sha256(policy_path),
            "promoted": policy.promoted,  # type: ignore[union-attr]
        },
        "episodes": [],
    }
    _write_json(manifest_path, manifest)
    environment = os.environ.copy()
    gpu_manifest: dict[str, object]
    lease = None
    try:
        if execution["device"] == "gpu":
            gpus, inventory, processes = query_gpus()
            requested = execution.get("physical_gpu_index")
            if requested is not None and (
                not isinstance(requested, int) or isinstance(requested, bool)
            ):
                raise ValueError("physical_gpu_index must be an integer or null")
            minimum = execution.get("minimum_free_gpu_mib")
            if not isinstance(minimum, int) or minimum <= 0:
                raise ValueError("GPU campaigns require positive minimum_free_gpu_mib")
            selected = select_gpu(gpus, requested, minimum)
            lease_path, lease = acquire_gpu_lease(selected.physical_index)
            environment["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
            environment["PHIAGENT_PHYSICAL_GPU"] = str(selected.physical_index)
            gpu_manifest = {
                "used": True,
                "physical_selection": selected.__dict__,
                "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
                "inventory": inventory.splitlines(),
                "processes": processes.splitlines(),
                "lease": str(lease_path),
            }
        else:
            gpu_manifest = {"used": False, "reason": "campaign execution.device=cpu"}
    except Exception as error:
        manifest.update(
            {
                "status": "preflight_failed",
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "gpu": {
                    "used": False,
                    "error": f"{type(error).__name__}: {error}",
                },
            }
        )
        _write_json(manifest_path, manifest)
        print(json.dumps({"experiment": str(experiment), "status": manifest["status"]}, indent=2))
        return 2
    manifest.update({"status": "running", "gpu": gpu_manifest})
    _write_json(manifest_path, manifest)
    records: list[FactoryRecord] = []
    accepted_index = []
    maximum_attempts = int(execution["maximum_attempts_per_episode"])
    collect_all_recipes = bool(execution["collect_all_recipes"])
    try:
        for case_index, case in enumerate(cases):
            episode_dir = experiment / "episodes" / f"{case_index:05d}-{case['episode_id']}"
            episode_dir.mkdir(parents=True)
            attempt_reports = []
            baseline = None
            selected = None
            baseline_recipe = recipes[contract.baseline_recipe_id]
            try:
                baseline, execution_report = _run_attempt(
                    case=case,
                    recipe=baseline_recipe,
                    attempt_dir=episode_dir / f"attempt-00-{contract.baseline_recipe_id}",
                    environment=environment,
                    contract=contract,
                    baseline_metrics=None,
                )
                records.append(baseline)
                assessment = assess_record(contract, baseline.metrics, baseline)
                attempt_reports.append(
                    {
                        "record": baseline.to_dict(),
                        "assessment": assessment.to_dict(),
                        "execution": execution_report,
                    }
                )
                if assessment.accepted:
                    selected = baseline
                candidates = [
                    recipe for recipe in contract.recipe_order if recipe != contract.baseline_recipe_id
                ]
                if policy is not None:
                    candidates = [recipe for recipe, _ in policy.rank(baseline.context, candidates)]
                for attempt_index, recipe_id in enumerate(candidates, start=1):
                    if (
                        (selected is not None and not collect_all_recipes)
                        or attempt_index >= maximum_attempts
                    ):
                        break
                    try:
                        record, execution_report = _run_attempt(
                            case=case,
                            recipe=recipes[recipe_id],
                            attempt_dir=episode_dir / f"attempt-{attempt_index:02d}-{recipe_id}",
                            environment=environment,
                            contract=contract,
                            baseline_metrics=baseline.metrics,
                        )
                        records.append(record)
                        assessment = assess_record(contract, baseline.metrics, record)
                        attempt_reports.append(
                            {
                                "record": record.to_dict(),
                                "assessment": assessment.to_dict(),
                                "execution": execution_report,
                            }
                        )
                        if assessment.accepted and selected is None:
                            selected = record
                    except Exception as error:  # preserve a failed adapter attempt and continue
                        attempt_reports.append(
                            {
                                "recipe_id": recipe_id,
                                "status": "worker_failure",
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
            except Exception as error:
                attempt_reports.append(
                    {
                        "recipe_id": contract.baseline_recipe_id,
                        "status": "baseline_worker_failure",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            episode_report = {
                "episode_id": case["episode_id"],
                "group_id": case["group_id"],
                "status": "accepted" if selected is not None else "rejected",
                "selected_recipe": selected.recipe_id if selected is not None else None,
                "selected_video": selected.video if selected is not None else None,
                "attempts": attempt_reports,
            }
            manifest["episodes"].append(episode_report)  # type: ignore[union-attr]
            if selected is not None:
                accepted_index.append(
                    {
                        "episode_id": selected.episode_id,
                        "group_id": selected.group_id,
                        "recipe_id": selected.recipe_id,
                        "video": selected.video,
                        "sha256": selected.video_sha256,
                        "metrics": dict(selected.metrics),
                    }
                )
            _write_json(manifest_path, manifest)
        with dataset_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        _write_json(accepted_index_path, {"videos": accepted_index})
        accepted = len(accepted_index)
        status = "accepted" if accepted == len(cases) else "partial"
        manifest.update(
            {
                "status": status,
                "honest_status": "WORKING" if status == "accepted" else "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "episodes": len(cases),
                    "accepted": accepted,
                    "acceptance_rate": accepted / len(cases),
                    "measured_attempts": len(records),
                },
                "artifacts": {
                    "training_records": {
                        "path": str(dataset_path),
                        "sha256": _sha256(dataset_path),
                    },
                    "accepted_video_index": {
                        "path": str(accepted_index_path),
                        "sha256": _sha256(accepted_index_path),
                    },
                },
                "claim_boundary": (
                    "Accepted rows passed this campaign's declared proxy and human gates. "
                    "They are generated data, not physical robot executions; downstream "
                    "model training requires a separately frozen evaluation split."
                ),
            }
        )
        _write_json(manifest_path, manifest)
    finally:
        if lease is not None:
            lease.close()
    print(
        json.dumps(
            {
                "experiment": str(experiment),
                "status": manifest["status"],
                "summary": manifest.get("summary"),
                "training_records": str(dataset_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
