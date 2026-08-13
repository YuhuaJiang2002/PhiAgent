#!/usr/bin/env python3
"""Adapt one native AC-WM attempt to the demo-factory worker protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORE_FIELDS = (
    "action_adherence",
    "embodiment_consistency",
    "object_interaction",
    "temporal_consistency",
    "background_consistency",
)
_FORBIDDEN_RUNNER_FLAGS = {
    "--case",
    "--prompt-suffix",
    "--experiment-root",
    "--maximum-rounds",
    "--seed",
    "--gpu",
}
_HEX_SHA256 = frozenset("0123456789abcdef")


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


def _load_object(path: Path, name: str) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return payload


def _parse_runner_stdout(stdout: str) -> dict[str, object]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start < 0 or end < start:
            raise ValueError("AC-WM runner stdout contains no JSON object") from None
        payload = json.loads(stdout[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("AC-WM runner result must be a JSON object")
    return payload


def _best_candidate(trace: Mapping[str, object], case_id: str) -> Mapping[str, object]:
    candidates = trace.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("AC-WM trace contains no candidates")
    indices = trace.get("best_candidate_indices")
    if not isinstance(indices, list):
        legacy = trace.get("best_candidate_index")
        indices = [legacy] if isinstance(legacy, int) else []
    indexed = {
        int(candidate["candidate_index"]): candidate
        for candidate in candidates
        if isinstance(candidate, dict) and isinstance(candidate.get("candidate_index"), int)
    }
    for index in indices:
        if not isinstance(index, int) or index not in indexed:
            continue
        candidate = indexed[index]
        proposal = candidate.get("proposal")
        if isinstance(proposal, dict) and proposal.get("case_id") == case_id:
            return candidate
    matching = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and isinstance(candidate.get("proposal"), dict)
        and candidate["proposal"].get("case_id") == case_id
    ]
    if len(matching) != 1:
        raise ValueError(f"AC-WM trace does not identify one best candidate for {case_id}")
    return matching[0]


def _runner_value(command: list[str], flag: str) -> str:
    if command.count(flag) != 1:
        raise ValueError(f"runner_command must contain exactly one {flag}")
    index = command.index(flag)
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise ValueError(f"runner_command {flag} requires one value")
    return command[index + 1]


def _validate_revision(payload: Mapping[str, object], field: str) -> None:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"AC-WM factory case requires {field} metadata")
    for key in ("id", "revision"):
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"AC-WM factory case requires {field}.{key}")


def _validate_case(payload: Mapping[str, object]) -> tuple[str, list[str]]:
    if payload.get("schema_version") != "1.0.0":
        raise ValueError("unsupported AC-WM factory case schema")
    case_id = payload.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("AC-WM factory case requires case_id")
    for field in (
        "episode_id",
        "group_id",
        "license_id",
        "source_uri",
        "action_coordinate_frame",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"AC-WM factory case requires explicit {field}")
    seed = payload.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("AC-WM factory case requires a non-negative integer seed")
    source_sha256 = payload.get("source_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or set(source_sha256) - _HEX_SHA256
    ):
        raise ValueError("AC-WM factory case requires lowercase source_sha256")
    _validate_revision(payload, "generator")
    _validate_revision(payload, "evaluator")
    command = payload.get("runner_command")
    if not isinstance(command, list) or not command or any(
        not isinstance(token, str) or not token for token in command
    ):
        raise ValueError("runner_command must be a non-empty JSON string array")
    collisions = _FORBIDDEN_RUNNER_FLAGS & set(command)
    if collisions:
        raise ValueError(f"runner_command contains worker-owned flags: {sorted(collisions)}")
    condition_value = _runner_value(list(command), "--condition-manifest")
    condition = Path(condition_value).expanduser()
    if not condition.is_absolute():
        condition = (PROJECT_ROOT / condition).resolve()
    if not condition.is_file():
        raise ValueError(f"AC-WM condition manifest does not exist: {condition}")
    condition_sha256 = payload.get("condition_manifest_sha256")
    if condition_sha256 != _sha256(condition):
        raise ValueError("AC-WM condition manifest hash does not match the case")
    review_value = _runner_value(list(command), "--human-review-dir")
    review_dir = Path(review_value).expanduser()
    if not review_dir.is_absolute():
        review_dir = (PROJECT_ROOT / review_dir).resolve()
    if not review_dir.is_dir():
        raise ValueError(f"AC-WM human review directory does not exist: {review_dir}")
    return case_id, list(command)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--recipe-manifest", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    case_path = args.case_manifest.expanduser().resolve()
    recipe_path = args.recipe_manifest.expanduser().resolve()
    attempt_dir = args.attempt_dir.expanduser().resolve()
    if not case_path.is_file() or not recipe_path.is_file():
        raise ValueError("case and recipe manifests must exist")
    if not attempt_dir.is_dir():
        raise ValueError("attempt directory must be created by the factory")
    case = _load_object(case_path, "case manifest")
    recipe = _load_object(recipe_path, "recipe manifest")
    case_id, runner_command = _validate_case(case)
    if args.seed != case["seed"]:
        raise ValueError("factory campaign seed does not match the immutable AC-WM case")
    parameters = recipe.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("recipe parameters must be a JSON object")
    allowed_parameters = {"prompt_suffix", "seed_offset", "cost_units"}
    unknown = set(parameters) - allowed_parameters
    if unknown:
        raise ValueError(f"unsupported AC-WM recipe parameters: {sorted(unknown)}")
    prompt_suffix = parameters.get("prompt_suffix", "")
    seed_offset = parameters.get("seed_offset", 0)
    if not isinstance(prompt_suffix, str):
        raise ValueError("prompt_suffix must be a string")
    if not isinstance(seed_offset, int) or isinstance(seed_offset, bool):
        raise ValueError("seed_offset must be an integer")
    seed = args.seed + seed_offset
    if seed < 0:
        raise ValueError("effective seed must be non-negative")
    physical_gpu = os.environ.get("PHIAGENT_PHYSICAL_GPU")
    if physical_gpu is None or not physical_gpu.isdigit():
        raise ValueError(
            "AC-WM factory workers require the outer batch runner's "
            "PHIAGENT_PHYSICAL_GPU selection"
        )
    acwm_root = attempt_dir / "acwm-runs"
    command = [
        *runner_command,
        "--case",
        case_id,
        "--experiment-root",
        str(acwm_root),
        "--maximum-rounds",
        "1",
        "--seed",
        str(seed),
        "--gpu",
        physical_gpu,
    ]
    if prompt_suffix.strip():
        command.extend(("--prompt-suffix", prompt_suffix.strip()))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
    )
    (attempt_dir / "acwm-runner.stdout.log").write_text(completed.stdout)
    (attempt_dir / "acwm-runner.stderr.log").write_text(completed.stderr)
    if completed.returncode not in {0, 2, 3}:
        raise RuntimeError(
            f"AC-WM runner failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    runner_result = _parse_runner_stdout(completed.stdout)
    trace_value = runner_result.get("trace")
    if not isinstance(trace_value, str):
        raise ValueError("AC-WM runner did not report a trace path")
    trace_path = Path(trace_value).expanduser().resolve()
    if not trace_path.is_file() or not trace_path.is_relative_to(attempt_dir):
        raise ValueError("AC-WM trace must exist inside the factory attempt")
    trace = _load_object(trace_path, "AC-WM trace")
    candidate = _best_candidate(trace, case_id)
    scorecard = candidate.get("scorecard")
    result = candidate.get("result")
    if not isinstance(scorecard, dict) or not isinstance(result, dict):
        raise ValueError("AC-WM best candidate lacks result or scorecard")
    missing = set(SCORE_FIELDS) - set(scorecard)
    if missing:
        raise ValueError(f"AC-WM scorecard is missing {sorted(missing)}")
    video_value = result.get("output")
    if not isinstance(video_value, str):
        raise ValueError("AC-WM best candidate has no video output")
    video = Path(video_value).expanduser().resolve()
    if not video.is_file() or video.suffix.lower() != ".mp4":
        raise ValueError(f"AC-WM candidate video is invalid: {video}")
    if not video.is_relative_to(attempt_dir):
        raise ValueError("AC-WM candidate video must remain inside the factory attempt")
    human = scorecard.get("human_review_passed")
    if human not in {True, False, None}:
        raise ValueError("AC-WM human review must be true, false, or null")
    diagnoses = scorecard.get("diagnoses", [])
    if not isinstance(diagnoses, list) or any(
        not isinstance(value, str) for value in diagnoses
    ):
        raise ValueError("AC-WM diagnoses must be a string array")
    cost_units = parameters.get("cost_units")
    if cost_units is not None:
        cost_units = float(cost_units)
        if cost_units < 0:
            raise ValueError("cost_units must be non-negative")
    adapter_report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "native_acwm_to_demo_factory_worker",
        "case_manifest": {"path": str(case_path), "sha256": _sha256(case_path)},
        "recipe_manifest": {"path": str(recipe_path), "sha256": _sha256(recipe_path)},
        "license_id": case["license_id"],
        "source_uri": case["source_uri"],
        "source_sha256": case["source_sha256"],
        "action_coordinate_frame": case["action_coordinate_frame"],
        "generator": case["generator"],
        "evaluator": case["evaluator"],
        "physical_gpu": int(physical_gpu),
        "command": command,
        "returncode": completed.returncode,
        "trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
        "video": {"path": str(video), "sha256": _sha256(video)},
        "trace_status": trace.get("status"),
    }
    _write_json(attempt_dir / "acwm-factory-adapter.json", adapter_report)
    worker_result: dict[str, object] = {
        "video": str(video),
        "metrics": {field: float(scorecard[field]) for field in SCORE_FIELDS},
        "human_review_passed": human,
        "diagnoses": diagnoses,
        "adapter_evidence": str(attempt_dir / "acwm-factory-adapter.json"),
    }
    if cost_units is not None:
        worker_result["cost_units"] = cost_units
    print(json.dumps(worker_result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
