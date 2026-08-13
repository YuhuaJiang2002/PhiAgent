#!/usr/bin/env python3
"""Persist a fail-closed calibration audit for DROID novel-view generation."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.view_generation.readiness import (  # noqa: E402
    audit_droid_novel_view_readiness,
)


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--raw-contract", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def main() -> int:
    args = _parser().parse_args()
    info_path = args.dataset_info.expanduser().resolve()
    contract_path = args.raw_contract.expanduser().resolve() if args.raw_contract else None
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite novel-view audit: {output}")
    if not info_path.is_file() or info_path.stat().st_size == 0:
        raise ValueError(f"dataset info is missing or empty: {info_path}")
    if contract_path is not None and (
        not contract_path.is_file() or contract_path.stat().st_size == 0
    ):
        raise ValueError(f"raw calibration contract is missing or empty: {contract_path}")
    info = json.loads(info_path.read_text())
    contract = json.loads(contract_path.read_text()) if contract_path is not None else None
    if not isinstance(info, dict) or (contract is not None and not isinstance(contract, dict)):
        raise ValueError("readiness inputs must contain JSON objects")
    result = audit_droid_novel_view_readiness(info, contract)
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "dataset_info": str(info_path),
            "dataset_info_sha256": _sha256(info_path),
            "raw_contract": str(contract_path) if contract_path is not None else None,
            "raw_contract_sha256": (
                _sha256(contract_path) if contract_path is not None else None
            ),
        },
    )
    evidence = {
        **result,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    _write_json(output / "readiness.json", evidence)
    (output / "audit.log").write_text(
        "\n".join(result["missing_requirements"]) + "\n"
        if result["missing_requirements"]
        else "all readiness requirements passed\n"
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

