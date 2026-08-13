#!/usr/bin/env python3
"""Train and held-action-evaluate the bounded EPIC Ego repair router."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import socket
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.training.ego_repair_policy import (  # noqa: E402
    SCORE_FIELDS,
    EgoNonRegressionContract,
    EgoRepairPolicy,
    encode_features,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _git_state() -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "available": status.returncode == 0,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
        "error": status.stderr.strip() if status.returncode else None,
    }


def _utility(scorecard: dict[str, object]) -> float:
    values = [float(scorecard[field]) for field in SCORE_FIELDS]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("repair scorecard contains an invalid utility field")
    return sum(values) / len(values)


def _load_rows(
    paths: list[Path],
    contract: EgoNonRegressionContract,
    regression_penalty: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"evaluation must contain one JSON object: {path}")
        action = str(payload.get("action_label", "")).strip()
        rounds = payload.get("rounds")
        if not action or not isinstance(rounds, list) or len(rounds) < 3:
            raise ValueError(f"evaluation lacks action or repair tournament: {path}")
        raw_scorecard = rounds[0].get("scorecard")
        if not isinstance(raw_scorecard, dict):
            raise ValueError(f"raw repair round is missing from {path}")
        campaign = path.parent.parent.parent.parent.name
        for record in rounds:
            repair = record.get("repair")
            scorecard = record.get("scorecard")
            if not isinstance(repair, dict) or not isinstance(scorecard, dict):
                raise ValueError(f"malformed repair round in {path}")
            utility = _utility(scorecard)
            assessment = contract.assess(raw_scorecard, scorecard)
            constrained = (
                utility
                if bool(assessment["passed"])
                else utility
                - regression_penalty * (1.0 + float(assessment["total_excess"]))
            )
            rows.append(
                {
                    "campaign": campaign,
                    "action": action,
                    "source": str(path),
                    "repair": repair,
                    "raw_scorecard": raw_scorecard,
                    "scorecard": scorecard,
                    "utility": utility,
                    "constrained_utility": constrained,
                    "non_regression": assessment,
                    "features": list(encode_features(raw_scorecard, repair)),
                }
            )
    return rows


def _fit(
    rows: list[dict[str, object]],
    *,
    alpha: float,
    held_out_action: str | None,
    contract: EgoNonRegressionContract,
    regression_penalty: float,
) -> EgoRepairPolicy:
    import numpy as np

    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    targets = np.asarray(
        [row["constrained_utility"] for row in rows], dtype=np.float64
    )
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack(
        (np.ones(len(rows), dtype=np.float64), (features - mean) / scale)
    )
    regularizer = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
    regularizer[1:, 1:] = np.eye(design.shape[1] - 1) * math.sqrt(alpha)
    weights, _, _, _ = np.linalg.lstsq(
        np.vstack((design, regularizer)),
        np.concatenate((targets, np.zeros(design.shape[1]))),
        rcond=None,
    )
    return EgoRepairPolicy(
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        intercept=float(weights[0]),
        coefficients=tuple(float(value) for value in weights[1:]),
        alpha=alpha,
        training_actions=tuple(sorted({str(row["action"]) for row in rows})),
        held_out_action=held_out_action,
        non_regression_contract=contract,
        regression_penalty=regression_penalty,
    )


def _evaluate(
    policy: EgoRepairPolicy, rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["source"])].append(row)
    selections = []
    for source, candidates in sorted(groups.items()):
        raw = next(
            row for row in candidates if row["repair"]["name"] == "raw-h3"  # type: ignore[index]
        )
        ranked = sorted(
            candidates,
            key=lambda row: policy.predict(
                row["raw_scorecard"], row["repair"]  # type: ignore[arg-type]
            ),
            reverse=True,
        )
        passing = [row for row in ranked if row["non_regression"]["passed"]]  # type: ignore[index]
        selected = passing[0] if passing else raw
        eligible_oracle = [
            row for row in candidates if row["non_regression"]["passed"]  # type: ignore[index]
        ]
        oracle = max(eligible_oracle, key=lambda row: float(row["utility"]))
        selections.append(
            {
                "source": source,
                "action": selected["action"],
                "selected_repair": selected["repair"]["name"],  # type: ignore[index]
                "oracle_repair": oracle["repair"]["name"],  # type: ignore[index]
                "exact_selection": selected["repair"]["name"]  # type: ignore[index]
                == oracle["repair"]["name"],  # type: ignore[index]
                "selected_utility": selected["utility"],
                "oracle_utility": oracle["utility"],
                "regret": float(oracle["utility"]) - float(selected["utility"]),
                "selected_non_regression": selected["non_regression"],
            }
        )
    return selections


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evolution", type=Path, action="append", required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/epic-ego-bottle-repair-policy"),
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--regression-penalty", type=float, default=2.0)
    parser.add_argument("--minimum-selection-accuracy", type=float, default=0.5)
    parser.add_argument("--maximum-mean-regret", type=float, default=0.05)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.alpha <= 0 or args.regression_penalty <= 0:
        raise ValueError("training regularization values must be positive")
    if not 0.0 <= args.minimum_selection_accuracy <= 1.0:
        raise ValueError("minimum selection accuracy must be in [0, 1]")
    if args.maximum_mean_regret < 0:
        raise ValueError("maximum mean regret must be non-negative")
    paths = [path.expanduser().resolve() for path in args.evolution]
    if len(paths) != len(set(paths)) or len(paths) < 6:
        raise ValueError("training requires six unique two-window Ego evaluations")
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"training evaluation is missing: {path}")

    experiment = _new_experiment(args.experiment_root.expanduser().resolve())
    contract = EgoNonRegressionContract()
    rows = _load_rows(paths, contract, args.regression_penalty)
    actions = sorted({str(row["action"]) for row in rows})
    if len(actions) != 3:
        raise ValueError("held-action evaluation requires exactly three actions")

    dataset_path = experiment / "training-data.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    folds = []
    selections = []
    for held_out in actions:
        train_rows = [row for row in rows if row["action"] != held_out]
        test_rows = [row for row in rows if row["action"] == held_out]
        policy = _fit(
            train_rows,
            alpha=args.alpha,
            held_out_action=held_out,
            contract=contract,
            regression_penalty=args.regression_penalty,
        )
        fold_dir = experiment / "folds" / f"holdout-{held_out}"
        fold_selections = _evaluate(policy, test_rows)
        _write_json(fold_dir / "policy.json", policy.to_dict())
        _write_json(fold_dir / "evaluation.json", {"selections": fold_selections})
        folds.append(
            {
                "held_out_action": held_out,
                "training_rows": len(train_rows),
                "test_rows": len(test_rows),
                "policy": str(fold_dir / "policy.json"),
                "selections": fold_selections,
            }
        )
        selections.extend(fold_selections)

    selection_accuracy = sum(
        bool(record["exact_selection"]) for record in selections
    ) / len(selections)
    mean_regret = sum(float(record["regret"]) for record in selections) / len(
        selections
    )
    non_regression_rate = sum(
        bool(record["selected_non_regression"]["passed"])  # type: ignore[index]
        for record in selections
    ) / len(selections)
    gates = {
        "selection_accuracy": selection_accuracy >= args.minimum_selection_accuracy,
        "mean_regret": mean_regret <= args.maximum_mean_regret,
        "selected_non_regression": non_regression_rate == 1.0,
    }
    final_policy = _fit(
        rows,
        alpha=args.alpha,
        held_out_action=None,
        contract=contract,
        regression_penalty=args.regression_penalty,
    )
    checkpoint = experiment / "policy.json"
    _write_json(checkpoint, final_policy.to_dict())
    package_versions = {}
    for name in ("numpy",):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    manifest = {
        "schema_version": "1.0.0",
        "method": "held_action_epic_ego_bottle_repair_policy_training",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if all(gates.values()) else "rejected",
        "honest_status": "WORKING" if all(gates.values()) else "PARTIAL",
        "command": [sys.executable, *sys.argv],
        "config": {
            "alpha": args.alpha,
            "seed": args.seed,
            "regression_penalty": args.regression_penalty,
            "minimum_selection_accuracy": args.minimum_selection_accuracy,
            "maximum_mean_regret": args.maximum_mean_regret,
            "non_regression_contract": contract.to_dict(),
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": package_versions,
        "gpu": {"used": False, "reason": "small deterministic ridge router"},
        "git": _git_state(),
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "training_rows": len(rows),
        "actions": actions,
        "folds": folds,
        "metrics": {
            "selection_accuracy": selection_accuracy,
            "mean_regret": mean_regret,
            "selected_non_regression_rate": non_regression_rate,
        },
        "gates": gates,
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "limitations": [
            "Training covers only six generated windows from one public Ego video and three commands.",
            "The learned module ranks bounded deterministic image-space repairs; it does not fine-tune MiniMax-H3 or establish robot contact physics.",
            "Held-action folds share scene, model, seed, and repair recipes, so generalization to new kitchens or objects is not established.",
        ],
    }
    _write_json(experiment / "manifest.json", manifest)
    print(json.dumps({"experiment": str(experiment), "metrics": manifest["metrics"], "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
