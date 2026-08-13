#!/usr/bin/env python3
"""Summarize frozen Hand2Dex raw-baseline smoke results and stop decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import socket
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCORES = (
    "motion_preservation",
    "target_identity",
    "object_consistency",
    "temporal_consistency",
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


def summarize_smoke(
    config: dict[str, Any],
    case_results: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless every configured case and score is present."""

    configured_cases = {int(row["case"]) for row in config.get("cases", [])}
    if configured_cases != set(case_results):
        raise ValueError("case results do not match the frozen configured cases")
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(SCORES):
        raise ValueError("frozen thresholds must cover all four proxy scores")
    rows = []
    for case in sorted(case_results):
        result = case_results[case]
        scores = {name: float(result[name]) for name in SCORES}
        gates = {
            name: scores[name] >= float(thresholds[name]) for name in SCORES
        }
        rows.append(
            {
                "case": case,
                "scores": scores,
                "gates": gates,
                "all_gates_pass": all(gates.values()),
                "diagnoses": list(result.get("diagnoses", [])),
            }
        )
    passes = sum(row["all_gates_pass"] for row in rows)
    required = int(config.get("seed_expansion_requires_case_passes", 1))
    if required <= 0:
        raise ValueError("seed expansion requires a positive case-pass count")
    return {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "PARTIAL",
        "cases": rows,
        "case_passes": passes,
        "case_total": len(rows),
        "all_gates_pass_rate": passes / len(rows),
        "mean_scores": {
            name: statistics.fmean(row["scores"][name] for row in rows)
            for name in SCORES
        },
        "seed_expansion_requires_case_passes": required,
        "expand_to_remaining_seeds": passes >= required,
        "decision": (
            "expand_remaining_seeds"
            if passes >= required
            else "stop_raw_wan_animate2_seed_expansion"
        ),
        "claim_boundary": (
            "Three public Hand2Dex cases are a fail-fast harness smoke, not a SOTA "
            "benchmark or evidence of physical contact."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--case-result",
        nargs=2,
        action="append",
        required=True,
        metavar=("CASE", "RESULT_JSON"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite smoke summary: {output}")
    if not config_path.is_file():
        raise ValueError(f"frozen smoke config is missing: {config_path}")
    config = json.loads(config_path.read_text())
    results = {}
    inputs = []
    for raw_case, raw_path in args.case_result:
        case = int(raw_case)
        path = Path(raw_path).expanduser().resolve()
        if case in results or not path.is_file():
            raise ValueError(f"duplicate case or missing result: {case}/{path}")
        result = json.loads(path.read_text())
        if not isinstance(result, dict):
            raise ValueError(f"case result must contain a JSON object: {path}")
        results[case] = result
        inputs.append({"case": case, "path": str(path), "sha256": _sha256(path)})
    summary = summarize_smoke(config, results)
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    _write_json(
        output / "summary.json",
        {
            **summary,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "frozen_config": str(config_path),
            "frozen_config_sha256": _sha256(config_path),
            "inputs": inputs,
        },
    )
    (output / "summary.log").write_text(
        f"{summary['case_passes']}/{summary['case_total']} cases passed; "
        f"decision={summary['decision']}\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

