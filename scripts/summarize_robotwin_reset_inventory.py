#!/usr/bin/env python3
"""Summarize stable RoboTwin reset seeds into a frozen replay inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stable_seed_records(records: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    stable = {}
    for record in records:
        if (
            record.get("status") != "WORKING"
            or record.get("deterministic_reset") is not True
            or record.get("different_seed_changes_scene") is not True
        ):
            continue
        seed = int(record["same_seed"])
        stable.setdefault(seed, record)
    return stable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--extra-result", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-seeds", type=int, default=20)
    return parser


def main() -> int:
    args = _parser().parse_args()
    inventory = args.inventory_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reset inventory summary: {output}")
    if not inventory.is_dir() or args.minimum_seeds <= 0:
        raise ValueError("reset inventory root and minimum seed count are invalid")
    result_paths = sorted(inventory.glob("**/result.json"))
    result_paths.extend(path.expanduser().resolve() for path in args.extra_result)
    if not result_paths:
        raise ValueError("reset inventory contains no results")
    records = []
    evidence = []
    for path in result_paths:
        if not path.is_file():
            raise ValueError(f"reset result is missing: {path}")
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"reset result must contain an object: {path}")
        records.append(payload)
        evidence.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "status": payload.get("status"),
                "same_seed": payload.get("same_seed"),
                "different_seed": payload.get("different_seed"),
                "failed_run": payload.get("failed_run"),
            }
        )
    stable = stable_seed_records(records)
    stable_seeds = sorted(stable)
    passed = len(stable_seeds) >= args.minimum_seeds
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING" if passed else "PARTIAL",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "inventory_root": str(inventory),
        "minimum_stable_seeds": args.minimum_seeds,
        "stable_seed_count": len(stable_seeds),
        "stable_seeds": stable_seeds,
        "gate_passed": passed,
        "result_count": len(records),
        "blocked_result_count": sum(
            record.get("status") != "WORKING" for record in records
        ),
        "evidence": evidence,
        "claim_boundary": (
            "Stable reset inventory validates deterministic initial states only. "
            "Expert A+ collection and paired A_swap execution remain required."
        ),
    }
    _write_json(output / "manifest.json", result)
    (output / "summary.log").write_text(
        f"stable={len(stable_seeds)} minimum={args.minimum_seeds} passed={passed}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

