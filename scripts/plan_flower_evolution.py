#!/usr/bin/env python3
"""Persist one evidence-backed flower-transfer evolution decision."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.agent.flower_evolution import (  # noqa: E402
    FlowerAcceptanceContract,
    FlowerCandidateEvaluation,
    FlowerEvolutionAgent,
    FlowerPipelineConfig,
)


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.expanduser().resolve().read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _git_state(root: Path) -> dict[str, object]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=root, check=True, text=True, capture_output=True
        ).stdout.splitlines()
        return {"available": True, "head": head, "status": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-counts", type=Path)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite experiment directory: {output_dir}")
    output_dir.mkdir(parents=True)

    config = FlowerPipelineConfig.from_dict(_json(args.config))
    evaluation = FlowerCandidateEvaluation.from_dict(_json(args.evaluation))
    failure_counts = _json(args.failure_counts) if args.failure_counts else {}
    if not all(isinstance(value, int) and value >= 0 for value in failure_counts.values()):
        raise ValueError("failure counts must be non-negative integers")
    contract = FlowerAcceptanceContract.strict()
    decision = FlowerEvolutionAgent().propose(
        config, evaluation, contract, failure_counts=failure_counts
    )

    _write(output_dir / "config.json", config.to_dict())
    _write(output_dir / "evaluation.json", evaluation.to_dict())
    _write(output_dir / "acceptance-contract.json", contract.to_dict())
    _write(output_dir / "decision.json", decision.to_dict())
    packages = {}
    for package in ("phiagent",):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "honest_status": decision.status,
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "seed": args.seed,
        "gpu": {"used": False, "reason": "dependency-free planning and acceptance only"},
        "git": _git_state(root),
        "inputs": {
            "config": str(args.config.expanduser().resolve()),
            "evaluation": str(args.evaluation.expanduser().resolve()),
        },
        "outputs": {
            "contract": str(output_dir / "acceptance-contract.json"),
            "decision": str(output_dir / "decision.json"),
        },
    }
    _write(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "decision": decision.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
