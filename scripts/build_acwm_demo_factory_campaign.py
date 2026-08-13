#!/usr/bin/env python3
"""Build a provenance-complete multi-scene AC-WM factory campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.training.demo_factory import FactoryContract  # noqa: E402


DOMAIN = "agentic-robot-demo-video"
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECIPE_PROMPTS = {
    "raw-generator": "",
    "identity-safe-repair": (
        "Keep exactly one robot with stable embodiment geometry, articulation, "
        "appearance, and topology throughout the video."
    ),
    "object-safe-repair": (
        "Preserve the instructed object, causal grasp and contact, object identity, "
        "and terminal state without duplication or disappearance."
    ),
    "temporal-safe-repair": (
        "Preserve the fixed real background and continuous robot-object motion; "
        "avoid flicker, jumps, smearing, and unsupported scene changes."
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _safe_id(value: object, field: str) -> str:
    identifier = str(value)
    if not _SAFE_ID.fullmatch(identifier):
        raise ValueError(f"{field} is not filesystem safe: {identifier!r}")
    return identifier


def _required_string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"case requires explicit {field}")
    return value


def _validate_revision(payload: Mapping[str, object], field: str) -> dict[str, str]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"case requires {field} metadata")
    return {
        key: _required_string(value, key)
        for key in ("id", "revision")
    }


def _command_value(command: list[str], flag: str) -> str:
    if command.count(flag) != 1:
        raise ValueError(f"runner_command must contain exactly one {flag}")
    index = command.index(flag)
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise ValueError(f"runner_command {flag} requires one value")
    return command[index + 1]


def _resolve_runner_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_case(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0.0":
        raise ValueError(f"unsupported AC-WM factory case schema: {path}")
    episode_id = _safe_id(payload.get("episode_id"), "episode_id")
    group_id = _safe_id(payload.get("group_id"), "group_id")
    case_id = _safe_id(payload.get("case_id"), "case_id")
    domain = str(payload.get("domain", DOMAIN))
    if domain != DOMAIN:
        raise ValueError(f"case {episode_id} has the wrong domain")
    seed = payload.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"case {episode_id} requires a non-negative integer seed")
    source_sha256 = _required_string(payload, "source_sha256")
    if not _SHA256.fullmatch(source_sha256):
        raise ValueError(f"case {episode_id} source_sha256 must be lowercase SHA-256")
    command = payload.get("runner_command")
    if not isinstance(command, list) or not command or any(
        not isinstance(token, str) or not token for token in command
    ):
        raise ValueError(f"case {episode_id} runner_command must be a string array")
    command = list(command)
    condition = _resolve_runner_path(_command_value(command, "--condition-manifest"))
    if not condition.is_file():
        raise ValueError(f"case {episode_id} condition manifest does not exist: {condition}")
    if payload.get("condition_manifest_sha256") != _sha256(condition):
        raise ValueError(f"case {episode_id} condition manifest hash does not match")
    review_dir = _resolve_runner_path(_command_value(command, "--human-review-dir"))
    if not review_dir.is_dir():
        raise ValueError(f"case {episode_id} human review directory does not exist: {review_dir}")
    return {
        **payload,
        "episode_id": episode_id,
        "group_id": group_id,
        "case_id": case_id,
        "domain": domain,
        "seed": seed,
        "license_id": _required_string(payload, "license_id"),
        "source_uri": _required_string(payload, "source_uri"),
        "source_sha256": source_sha256,
        "action_coordinate_frame": _required_string(payload, "action_coordinate_frame"),
        "generator": _validate_revision(payload, "generator"),
        "evaluator": _validate_revision(payload, "evaluator"),
        "runner_command": command,
        "manifest_path": str(path),
        "manifest_sha256": _sha256(path),
    }


def _load_contract(path: Path) -> FactoryContract:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("contract file must contain one JSON object")
    raw = payload.get("contract", payload)
    if not isinstance(raw, dict):
        raise ValueError("contract payload must be a JSON object")
    contract = FactoryContract.from_dict(raw)
    if contract.domain != DOMAIN or tuple(_RECIPE_PROMPTS) != contract.recipe_order:
        raise ValueError("AC-WM campaign builder requires the v1 agentic video contract")
    return contract


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--case-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/demo_factory/agentic_video_demo_contract_v1.json"),
    )
    parser.add_argument("--mode", choices=("bootstrap", "production"), default="bootstrap")
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=60000)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser


def main() -> int:
    args = _parser().parse_args()
    campaign_id = _safe_id(args.campaign_id, "campaign_id")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite campaign: {output}")
    if args.minimum_free_gpu_mib < 0:
        raise ValueError("minimum_free_gpu_mib must be non-negative")
    if args.physical_gpu_index is not None and args.physical_gpu_index < 0:
        raise ValueError("physical_gpu_index must be non-negative")
    contract_path = args.contract.expanduser().resolve()
    contract = _load_contract(contract_path)
    case_paths = [path.expanduser().resolve() for path in args.case_manifest]
    if any(not path.is_file() for path in case_paths):
        raise ValueError("every case manifest must exist")
    cases = [_load_case(path) for path in case_paths]
    episode_ids = [str(case["episode_id"]) for case in cases]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("case manifests contain duplicate episode_id values")
    groups = {str(case["group_id"]) for case in cases}
    sources = {str(case["source_uri"]) for case in cases}
    if len(groups) < 2 or len(sources) < 2:
        raise ValueError(
            "AC-WM campaigns require at least two independent group_id and source_uri values"
        )
    worker = PROJECT_ROOT / "scripts" / "run_acwm_demo_factory_worker.py"
    worker_command = [
        str(Path(args.python_executable).expanduser()),
        str(worker),
        "--case-manifest",
        "{case_manifest}",
        "--recipe-manifest",
        "{recipe_manifest}",
        "--attempt-dir",
        "{attempt_dir}",
        "--seed",
        "{seed}",
    ]
    recipes = [
        {
            "recipe_id": recipe_id,
            "command": worker_command,
            "parameters": {
                "prompt_suffix": prompt,
                "seed_offset": 0,
                "cost_units": 1.0,
            },
            "estimated_cost_units": 1.0,
        }
        for recipe_id, prompt in _RECIPE_PROMPTS.items()
    ]
    execution: dict[str, object] = {
        "device": "gpu",
        "minimum_free_gpu_mib": args.minimum_free_gpu_mib,
        "maximum_attempts_per_episode": len(recipes),
        "collect_all_recipes": args.mode == "bootstrap",
    }
    if args.physical_gpu_index is not None:
        execution["physical_gpu_index"] = args.physical_gpu_index
    campaign = {
        "schema_version": "1.0.0",
        "campaign_id": campaign_id,
        "contract": contract.to_dict(),
        "recipes": recipes,
        "cases": [
            {
                "episode_id": case["episode_id"],
                "group_id": case["group_id"],
                "domain": case["domain"],
                "manifest": case["manifest_path"],
                "seed": case["seed"],
            }
            for case in cases
        ],
        "execution": execution,
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "builder": str(Path(__file__).resolve()),
            "mode": args.mode,
            "contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
            "case_manifests": [
                {
                    "path": case["manifest_path"],
                    "sha256": case["manifest_sha256"],
                    "license_id": case["license_id"],
                    "source_uri": case["source_uri"],
                    "source_sha256": case["source_sha256"],
                    "action_coordinate_frame": case["action_coordinate_frame"],
                    "generator": case["generator"],
                    "evaluator": case["evaluator"],
                }
                for case in cases
            ],
        },
    }
    _write_json(output, campaign)
    print(
        json.dumps(
            {
                "campaign": str(output),
                "campaign_id": campaign_id,
                "cases": len(cases),
                "groups": sorted(groups),
                "mode": args.mode,
                "recipes": list(_RECIPE_PROMPTS),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
