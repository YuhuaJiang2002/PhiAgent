#!/usr/bin/env python3
"""Migrate measured EPIC-Ego repair tournaments to demo-factory records."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.training.demo_factory import FactoryContract, FactoryRecord  # noqa: E402


DOMAIN = "epic-ego-bottle-repair-history"
METRICS = (
    "background_lock",
    "object_lock",
    "subject_replacement",
    "robot_identity",
    "motion_preservation",
    "temporal_consistency",
    "epl_minimum",
)


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


def _new_experiment(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = root / f"{stamp}-{uuid4().hex[:8]}"
    experiment.mkdir()
    return experiment


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _scorecard(payload: object, name: str) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    missing = set(METRICS) - set(payload)
    if missing:
        raise ValueError(f"{name} is missing metrics: {sorted(missing)}")
    result = {field: float(payload[field]) for field in METRICS}
    if any(not 0.0 <= value <= 1.0 for value in result.values()):
        raise ValueError(f"{name} has metrics outside [0, 1]")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/demo-factory-history-import"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    source_path = args.training_data.expanduser().resolve()
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise ValueError(f"Ego repair training data is missing: {source_path}")
    raw_rows = []
    for line_number, raw_line in enumerate(source_path.read_text().splitlines(), start=1):
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {source_path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"training row {line_number} is not an object")
        raw_rows.append(row)
    if not raw_rows:
        raise ValueError("Ego repair history is empty")

    recipe_order = []
    for row in raw_rows:
        repair = row.get("repair")
        if not isinstance(repair, dict) or not isinstance(repair.get("name"), str):
            raise ValueError("every history row requires a named repair")
        if repair["name"] not in recipe_order:
            recipe_order.append(repair["name"])
    if not recipe_order or recipe_order[0] != "raw-h3":
        raise ValueError("Ego history must begin with the raw-h3 baseline")

    evolution_cache: dict[Path, dict[str, object]] = {}
    thresholds = None
    tolerances = None
    records = []
    source_evidence = []
    for row in raw_rows:
        source = Path(str(row.get("source", ""))).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"source evolution does not exist: {source}")
        if source not in evolution_cache:
            evolution = _load_object(source)
            evolution_cache[source] = evolution
            current_thresholds = evolution.get("thresholds")
            if not isinstance(current_thresholds, dict):
                raise ValueError(f"evolution has no thresholds: {source}")
            normalized_thresholds = {
                field: float(current_thresholds[field]) for field in METRICS
            }
            if thresholds is None:
                thresholds = normalized_thresholds
            elif thresholds != normalized_thresholds:
                raise ValueError("history mixes incompatible quality thresholds")
            source_evidence.append({"path": str(source), "sha256": _sha256(source)})
        evolution = evolution_cache[source]
        repair = row["repair"]
        recipe_id = str(repair["name"])
        rounds = evolution.get("rounds")
        if not isinstance(rounds, list):
            raise ValueError(f"evolution has no measured rounds: {source}")
        matches = [
            item
            for item in rounds
            if isinstance(item, dict)
            and isinstance(item.get("repair"), dict)
            and item["repair"].get("name") == recipe_id
        ]
        if len(matches) != 1:
            raise ValueError(f"evolution does not uniquely bind recipe {recipe_id}: {source}")
        measured = matches[0]
        scorecard = _scorecard(row.get("scorecard"), "history scorecard")
        measured_scorecard = _scorecard(measured.get("scorecard"), "evolution scorecard")
        if scorecard != measured_scorecard:
            raise ValueError(f"cached training row drifted from evolution: {source}")
        raw_scorecard = _scorecard(row.get("raw_scorecard"), "raw scorecard")
        non_regression = row.get("non_regression")
        if not isinstance(non_regression, dict) or not isinstance(
            non_regression.get("tolerances"), dict
        ):
            raise ValueError("history row has no non-regression contract")
        current_tolerances = {
            str(key): float(value)
            for key, value in non_regression["tolerances"].items()
        }
        if tolerances is None:
            tolerances = current_tolerances
        elif tolerances != current_tolerances:
            raise ValueError("history mixes non-regression contracts")
        video = Path(str(measured.get("output", ""))).expanduser().resolve()
        if not video.is_file() or video.suffix.lower() != ".mp4":
            raise ValueError(f"measured repair video is missing: {video}")
        expected_hash = measured.get("output_sha256")
        actual_hash = _sha256(video)
        if expected_hash != actual_hash:
            raise ValueError(f"measured repair video hash mismatch: {video}")
        action = str(row.get("action", ""))
        campaign = str(row.get("campaign", ""))
        episode_id = f"{campaign}--{action}"
        failed = [
            field for field, threshold in thresholds.items() if scorecard[field] < threshold
        ]
        if non_regression.get("passed") is not True:
            failed.append("capability-non-regression")
        records.append(
            FactoryRecord(
                episode_id=episode_id,
                group_id=action,
                domain=DOMAIN,
                recipe_id=recipe_id,
                recipe_parameters={
                    str(key): value for key, value in repair.items() if key != "name"
                },
                context=raw_scorecard,
                metrics=scorecard,
                cost_units=1.0,
                human_review_passed=None,
                video=str(video),
                video_sha256=actual_hash,
                diagnoses=tuple(f"failed:{field}" for field in failed),
            )
        )
    assert thresholds is not None and tolerances is not None
    contract = FactoryContract.from_dict(
        {
            "domain": DOMAIN,
            "baseline_recipe_id": "raw-h3",
            "recipe_order": recipe_order,
            "context_fields": list(METRICS),
            "metric_weights": {field: 1.0 for field in METRICS},
            "hard_thresholds": thresholds,
            "non_regression_tolerances": tolerances,
            "cost_budget_units": float(len(recipe_order)),
            "cost_weight": 0.01,
            "rejection_penalty": 2.0,
            "human_review_required": True,
        }
    )
    experiment = _new_experiment(args.experiment_root.expanduser().resolve())
    dataset_path = experiment / "episodes.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    contract_path = experiment / "contract.json"
    _write_json(contract_path, {"schema_version": "1.0.0", "contract": contract.to_dict()})
    actions = sorted({record.group_id for record in records})
    episodes = sorted({record.episode_id for record in records})
    manifest = {
        "schema_version": "1.0.0",
        "method": "measured_epic_ego_repair_history_to_demo_factory",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "honest_status": "PARTIAL",
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": {"used": False, "reason": "immutable measured-history migration"},
        "input": {"path": str(source_path), "sha256": _sha256(source_path)},
        "source_evolutions": source_evidence,
        "records": len(records),
        "episodes": len(episodes),
        "held_groups": actions,
        "recipes": recipe_order,
        "artifacts": {
            "dataset": {"path": str(dataset_path), "sha256": _sha256(dataset_path)},
            "contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
        },
        "limitations": [
            "All episodes come from one EPIC-KITCHENS source interval and one bottle domain.",
            "Held groups are action labels, not independent scenes, subjects, or kitchens.",
            "Every historical human review is pending, so the display-ready contract cannot accept a candidate.",
            "Cost units count cached candidate evaluations and are not measured GPU seconds or currency.",
        ],
    }
    _write_json(experiment / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "experiment": str(experiment),
                "dataset": str(dataset_path),
                "contract": str(contract_path),
                "records": len(records),
                "episodes": len(episodes),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
