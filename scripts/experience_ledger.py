#!/usr/bin/env python3
"""Validate, append to, and summarize PhiAgent's experience ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.learning.experience import (  # noqa: E402
    ExperienceRecord,
    append_experience,
    load_experiences,
    read_status_inventory,
    summarize_experiences,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = PROJECT_ROOT / "experiences" / "ledger.jsonl"
DEFAULT_STATUS = PROJECT_ROOT / "docs" / "STATUS.md"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate")
    summary = subparsers.add_parser("summary")
    summary.add_argument("--status-doc", type=Path, default=DEFAULT_STATUS)

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


if __name__ == "__main__":
    raise SystemExit(main())
