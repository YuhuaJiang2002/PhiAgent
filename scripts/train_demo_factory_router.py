#!/usr/bin/env python3
"""Train and held-group-promote a lightweight demo-video recipe router."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.training.demo_factory import (  # noqa: E402
    FactoryContract,
    load_records,
    train_grouped_router,
)


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


def _new_experiment(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = root / f"{stamp}-{uuid4().hex[:8]}"
    experiment.mkdir()
    return experiment


def _git_state() -> dict[str, object]:
    result: dict[str, object] = {}
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
            result[name] = (
                completed.stdout.strip()
                if name == "head"
                else completed.stdout.splitlines()
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            result[name] = f"unavailable: {error}"
    return result


def _load_contract(path: Path) -> FactoryContract:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("contract file must contain one JSON object")
    contract = payload.get("contract", payload)
    if not isinstance(contract, dict):
        raise ValueError("contract must be a JSON object")
    return FactoryContract.from_dict(contract)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/demo-factory-router"),
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--minimum-acceptance-rate", type=float, default=0.5)
    parser.add_argument("--utility-regression-tolerance", type=float, default=0.0)
    parser.add_argument("--cost-regression-fraction", type=float, default=0.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    datasets = tuple(path.expanduser().resolve() for path in args.dataset)
    if len(datasets) != len(set(datasets)):
        raise ValueError("dataset paths must be unique")
    for path in datasets:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"training dataset does not exist or is empty: {path}")
    contract_path = args.contract.expanduser().resolve()
    if not contract_path.is_file():
        raise ValueError(f"contract does not exist: {contract_path}")
    contract = _load_contract(contract_path)
    records = load_records(datasets)
    experiment = _new_experiment(args.experiment_root.expanduser().resolve())
    manifest_path = experiment / "manifest.json"
    initial_manifest = {
        "schema_version": "1.0.0",
        "method": "lightweight_demo_video_factory_router_training",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "training",
        "command": [sys.executable, *sys.argv],
        "config": {
            "alpha": args.alpha,
            "minimum_acceptance_rate": args.minimum_acceptance_rate,
            "utility_regression_tolerance": args.utility_regression_tolerance,
            "cost_regression_fraction": args.cost_regression_fraction,
            "contract": contract.to_dict(),
            "contract_sha256": contract.fingerprint,
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": {
            "used": False,
            "reason": "small deterministic standard-library ridge router",
        },
        "git": _git_state(),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (*datasets, contract_path)
        ],
        "records": len(records),
    }
    _write_json(manifest_path, initial_manifest)
    try:
        result = train_grouped_router(
            records,
            contract,
            alpha=args.alpha,
            minimum_acceptance_rate=args.minimum_acceptance_rate,
            utility_regression_tolerance=args.utility_regression_tolerance,
            cost_regression_fraction=args.cost_regression_fraction,
        )
    except Exception as error:
        initial_manifest.update(
            {
                "status": "training_failed",
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write_json(manifest_path, initial_manifest)
        raise
    snapshot = experiment / "training-records.jsonl"
    with snapshot.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    preferences = experiment / "distillation-preferences.jsonl"
    with preferences.open("w", encoding="utf-8") as handle:
        for row in result.preferences:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    checkpoint = experiment / "policy.json"
    evaluation = experiment / "held-group-evaluation.json"
    _write_json(checkpoint, result.policy.to_dict())
    _write_json(evaluation, result.evaluation)
    manifest = {
        "schema_version": "1.0.0",
        "method": "lightweight_demo_video_factory_router_training",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "promoted" if result.policy.promoted else "rejected",
        "honest_status": "WORKING" if result.policy.promoted else "PARTIAL",
        "command": [sys.executable, *sys.argv],
        "config": {
            "alpha": args.alpha,
            "minimum_acceptance_rate": args.minimum_acceptance_rate,
            "utility_regression_tolerance": args.utility_regression_tolerance,
            "cost_regression_fraction": args.cost_regression_fraction,
            "contract": contract.to_dict(),
            "contract_sha256": contract.fingerprint,
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": {
            "used": False,
            "reason": "small deterministic standard-library ridge router",
        },
        "git": _git_state(),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (*datasets, contract_path)
        ],
        "records": len(records),
        "groups": list(result.policy.training_groups),
        "evaluation": result.evaluation,
        "artifacts": {
            "policy": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
            "training_records": {"path": str(snapshot), "sha256": _sha256(snapshot)},
            "held_group_evaluation": {
                "path": str(evaluation),
                "sha256": _sha256(evaluation),
            },
            "distillation_preferences": {
                "path": str(preferences),
                "sha256": _sha256(preferences),
                "rows": len(result.preferences),
            },
        },
        "claim_boundary": (
            "Promotion validates recipe routing on held groups only. It does not prove "
            "a better foundation model, new physical capability, or generalization to "
            "unmeasured scenes, objects, embodiments, or generators."
        ),
    }
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "experiment": str(experiment),
                "policy": str(checkpoint),
                "promoted": result.policy.promoted,
                "gates": result.policy.promotion_gates,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.policy.promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
