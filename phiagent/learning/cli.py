"""Installed command-line entry point for the experience ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phiagent.learning.experience import (
    ExperienceRecord,
    append_experience,
    load_experiences,
    read_status_inventory,
    summarize_experiences,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=Path("experiences/ledger.jsonl"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    summary = subparsers.add_parser("summary")
    summary.add_argument("--status-doc", type=Path, default=Path("docs/STATUS.md"))
    add = subparsers.add_parser("add")
    add.add_argument("--record", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    ledger = args.ledger.expanduser().resolve()
    if args.command == "add":
        raw = json.loads(args.record.expanduser().resolve().read_text())
        if not isinstance(raw, dict):
            raise ValueError("record must contain one JSON object")
        append_experience(ledger, ExperienceRecord.from_dict(raw))
    records = load_experiences(ledger)
    result = {"ledger": str(ledger), **summarize_experiences(records)}
    if args.command == "summary":
        inventory = read_status_inventory(args.status_doc.expanduser().resolve())
        result["status_source"] = str(inventory.source)
        result["status_inventory"] = dict(inventory.counts)
        result["status_total"] = inventory.total
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
