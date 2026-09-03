#!/usr/bin/env python3
"""Validate checked-in PhiAgent-Bench v0.2 manifests with JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected object: {path}")
    return payload


def validate(schema_name: str, paths: list[Path]) -> None:
    schema = load(ROOT / "benchmark" / "schemas" / schema_name)
    validator = jsonschema.Draft202012Validator(schema)
    validator.check_schema(schema)
    for path in paths:
        validator.validate(load(path))


def main() -> int:
    validate(
        "policy-v0.2.schema.json",
        sorted((ROOT / "benchmark" / "policies").glob("*.json")),
    )
    validate(
        "method-v0.2.schema.json",
        [ROOT / "benchmark" / "examples" / "method-v0.2.example.json"],
    )
    validate(
        "physical-gate-trace-v0.2.schema.json",
        [ROOT / "benchmark" / "examples" / "physical-gate-trace-v0.2.example.json"],
    )
    for name in ("submission-v0.2.schema.json", "suite-v0.2.schema.json"):
        jsonschema.Draft202012Validator.check_schema(
            load(ROOT / "benchmark" / "schemas" / name)
        )
    print("PhiAgent-Bench v0.2 schemas, policies, and examples are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
