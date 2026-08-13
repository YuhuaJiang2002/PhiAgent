#!/usr/bin/env python3
"""Train and held-action-evaluate a flower-repair utility ranker."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.training.flower_repair_policy import (  # noqa: E402
    FlowerRepairPolicy,
    NonRegressionContract,
    encode_features,
    feature_names,
)


DEFAULT_EVOLUTIONS = (
    "outputs/minimax-h3-action-control/*/variants/*/agent-evaluation/evolution.json"
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


def _git_state(root: Path) -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
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


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("numpy", "opencv-python", "opencv-python-headless"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _parse_source(path: Path, project_root: Path) -> tuple[str, str]:
    try:
        relative = path.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"training input is outside the project: {path}") from error
    parts = relative.parts
    if len(parts) < 7 or parts[:2] != ("outputs", "minimax-h3-action-control"):
        raise ValueError(f"unexpected evolution path contract: {relative}")
    return parts[2], parts[4]


def _load_rows(
    paths: list[Path],
    project_root: Path,
    contract: NonRegressionContract,
    regression_penalty: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"evolution must contain one JSON object: {path}")
        campaign, action = _parse_source(path, project_root)
        rounds = payload.get("rounds")
        if not isinstance(rounds, list) or len(rounds) != 5:
            raise ValueError(f"expected five repair rounds in {path}")
        raw = rounds[0].get("scorecard")
        if not isinstance(raw, dict):
            raise ValueError(f"raw scorecard missing from {path}")
        for round_record in rounds:
            repair = round_record.get("repair")
            scorecard = round_record.get("scorecard")
            if not isinstance(repair, dict) or not isinstance(scorecard, dict):
                raise ValueError(f"malformed repair round in {path}")
            utility = float(scorecard["mean_score"])
            if not math.isfinite(utility) or not 0.0 <= utility <= 1.0:
                raise ValueError(f"invalid candidate utility in {path}")
            non_regression = contract.assess(raw, scorecard)
            constrained_utility = (
                utility
                if non_regression.passed
                else utility - regression_penalty * (1.0 + non_regression.total_excess)
            )
            rows.append(
                {
                    "campaign": campaign,
                    "action": action,
                    "repair": repair,
                    "raw_scorecard": raw,
                    "utility": utility,
                    "constrained_utility": constrained_utility,
                    "non_regression": non_regression.to_dict(),
                    "source": str(path),
                    "features": list(encode_features(raw, repair)),
                }
            )
    return rows


def _fit_policy(
    rows: list[dict[str, object]],
    alpha: float,
    held_out_action: str | None,
    contract: NonRegressionContract,
    regression_penalty: float,
) -> FlowerRepairPolicy:
    import numpy as np

    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["constrained_utility"] for row in rows], dtype=np.float64)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack((np.ones(len(rows), dtype=np.float64), standardized))
    regularizer = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
    regularizer[1:, 1:] = np.eye(design.shape[1] - 1) * math.sqrt(alpha)
    augmented_design = np.vstack((design, regularizer))
    augmented_targets = np.concatenate((targets, np.zeros(design.shape[1])))
    weights, _, _, _ = np.linalg.lstsq(augmented_design, augmented_targets, rcond=None)
    actions = tuple(sorted({str(row["action"]) for row in rows}))
    return FlowerRepairPolicy(
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        intercept=float(weights[0]),
        coefficients=tuple(float(value) for value in weights[1:]),
        alpha=alpha,
        training_actions=actions,
        held_out_action=held_out_action,
        non_regression_contract=contract,
        regression_penalty=regression_penalty,
    )


def _evaluate_fold(
    policy: FlowerRepairPolicy, rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[float]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    squared_errors = []
    for row in rows:
        raw = row["raw_scorecard"]
        repair = row["repair"]
        assert isinstance(raw, dict) and isinstance(repair, dict)
        prediction = policy.predict(raw, repair)
        row = {**row, "predicted_constrained_utility": prediction}
        groups[(str(row["campaign"]), str(row["action"]))].append(row)
        squared_errors.append((prediction - float(row["constrained_utility"])) ** 2)
    selections = []
    for (campaign, action), candidates in sorted(groups.items()):
        eligible = [item for item in candidates if item["repair"]["name"] != "raw-h3"]  # type: ignore[index]
        ordered = sorted(
            eligible,
            key=lambda item: float(item["predicted_constrained_utility"]),
            reverse=True,
        )
        passing = [item for item in eligible if item["non_regression"]["passed"]]  # type: ignore[index]
        if not passing:
            raise ValueError(f"no non-regressing repair exists for {campaign}/{action}")
        selected = next(item for item in ordered if item["non_regression"]["passed"])  # type: ignore[index]
        oracle = max(passing, key=lambda item: float(item["utility"]))
        raw = next(item for item in candidates if item["repair"]["name"] == "raw-h3")  # type: ignore[index]
        generic = next(
            item
            for item in candidates
            if item["repair"]["name"] == "tracked-mask-background-lock"  # type: ignore[index]
        )
        selected_utility = float(selected["utility"])
        oracle_utility = float(oracle["utility"])
        selections.append(
            {
                "campaign": campaign,
                "action": action,
                "selected_repair": selected["repair"]["name"],  # type: ignore[index]
                "selected_utility": selected_utility,
                "predicted_constrained_utility": float(selected["predicted_constrained_utility"]),
                "first_choice_repair": ordered[0]["repair"]["name"],  # type: ignore[index]
                "first_choice_non_regression": ordered[0]["non_regression"]["passed"],  # type: ignore[index]
                "repair_candidates_evaluated": ordered.index(selected) + 1,
                "selected_non_regression": selected["non_regression"],
                "oracle_repair": oracle["repair"]["name"],  # type: ignore[index]
                "oracle_utility": oracle_utility,
                "exact_selection": selected["repair"]["name"] == oracle["repair"]["name"],  # type: ignore[index]
                "regret": oracle_utility - selected_utility,
                "raw_utility": float(raw["utility"]),
                "generic_one_step_utility": float(generic["utility"]),
                "gain_over_raw": selected_utility - float(raw["utility"]),
                "gain_over_generic_one_step": selected_utility - float(generic["utility"]),
                "selected_output": next(
                    round_record["output"]
                    for round_record in json.loads(Path(str(selected["source"])).read_text())[
                        "rounds"
                    ]
                    if round_record["repair"]["name"] == selected["repair"]["name"]  # type: ignore[index]
                ),
            }
        )
    return selections, squared_errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evolution-glob",
        default=DEFAULT_EVOLUTIONS,
        help="project-relative glob of prior five-round flower evaluations",
    )
    parser.add_argument(
        "--experiment-root", type=Path, default=Path("outputs/flower-repair-policy")
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--minimum-selection-accuracy", type=float, default=0.8)
    parser.add_argument("--minimum-gain-over-raw", type=float, default=0.1)
    parser.add_argument("--minimum-first-choice-non-regression", type=float, default=0.8)
    parser.add_argument("--regression-penalty", type=float, default=2.0)
    parser.add_argument("--motion-regression-tolerance", type=float, default=0.01)
    parser.add_argument("--epl-regression-tolerance", type=float, default=0.01)
    parser.add_argument("--temporal-regression-tolerance", type=float, default=0.01)
    parser.add_argument("--identity-regression-tolerance", type=float, default=0.01)
    parser.add_argument("--subject-regression-tolerance", type=float, default=0.02)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not 0 < args.minimum_selection_accuracy <= 1:
        raise ValueError("minimum-selection-accuracy must be in (0, 1]")
    if not math.isfinite(args.minimum_gain_over_raw):
        raise ValueError("minimum-gain-over-raw must be finite")
    if not 0 < args.minimum_first_choice_non_regression <= 1:
        raise ValueError("minimum-first-choice-non-regression must be in (0, 1]")
    if not math.isfinite(args.regression_penalty) or args.regression_penalty <= 0:
        raise ValueError("regression-penalty must be finite and positive")
    contract = NonRegressionContract(
        motion_preservation=args.motion_regression_tolerance,
        epl_minimum=args.epl_regression_tolerance,
        temporal_consistency=args.temporal_regression_tolerance,
        robot_identity=args.identity_regression_tolerance,
        subject_replacement=args.subject_regression_tolerance,
    )
    project_root = Path(__file__).resolve().parents[1]
    paths = sorted(project_root.glob(args.evolution_glob))
    if not paths:
        raise ValueError(f"no evolution inputs match {args.evolution_glob}")
    experiment = _new_experiment(args.experiment_root.expanduser().resolve())
    rows = _load_rows(paths, project_root, contract, args.regression_penalty)
    actions = sorted({str(row["action"]) for row in rows})
    campaigns = sorted({str(row["campaign"]) for row in rows})
    if len(actions) < 3 or len(campaigns) < 3:
        raise ValueError("held-action evaluation requires at least three actions and campaigns")

    dataset_path = experiment / "training-data.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    all_selections: list[dict[str, object]] = []
    all_squared_errors: list[float] = []
    fold_records = []
    for held_out_action in actions:
        train_rows = [row for row in rows if row["action"] != held_out_action]
        test_rows = [row for row in rows if row["action"] == held_out_action]
        policy = _fit_policy(
            train_rows,
            args.alpha,
            held_out_action,
            contract,
            args.regression_penalty,
        )
        fold_dir = experiment / "folds" / f"holdout-{held_out_action}"
        checkpoint = fold_dir / "policy.json"
        _write_json(checkpoint, policy.to_dict())
        selections, squared_errors = _evaluate_fold(policy, test_rows)
        all_selections.extend(selections)
        all_squared_errors.extend(squared_errors)
        fold = {
            "held_out_action": held_out_action,
            "training_actions": list(policy.training_actions),
            "training_rows": len(train_rows),
            "test_rows": len(test_rows),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "selections": selections,
        }
        _write_json(fold_dir / "evaluation.json", fold)
        fold_records.append(fold)

    final_policy = _fit_policy(rows, args.alpha, None, contract, args.regression_penalty)
    final_checkpoint = experiment / "policy.json"
    _write_json(final_checkpoint, final_policy.to_dict())
    selection_accuracy = sum(bool(item["exact_selection"]) for item in all_selections) / len(
        all_selections
    )
    mean_regret = sum(float(item["regret"]) for item in all_selections) / len(all_selections)
    mean_gain_raw = sum(float(item["gain_over_raw"]) for item in all_selections) / len(
        all_selections
    )
    mean_gain_generic = sum(
        float(item["gain_over_generic_one_step"]) for item in all_selections
    ) / len(all_selections)
    first_choice_non_regression_rate = sum(
        bool(item["first_choice_non_regression"]) for item in all_selections
    ) / len(all_selections)
    guarded_non_regression_rate = sum(
        bool(item["selected_non_regression"]["passed"])  # type: ignore[index]
        for item in all_selections
    ) / len(all_selections)
    mean_repair_candidates = sum(
        int(item["repair_candidates_evaluated"]) for item in all_selections
    ) / len(all_selections)
    maximum_regressions = {
        field: max(
            float(item["selected_non_regression"]["regressions"][field])  # type: ignore[index]
            for item in all_selections
        )
        for field in contract.to_dict()
    }
    rmse = math.sqrt(sum(all_squared_errors) / len(all_squared_errors))
    accepted = (
        selection_accuracy >= args.minimum_selection_accuracy
        and mean_gain_raw >= args.minimum_gain_over_raw
        and mean_regret <= 0.01
        and first_choice_non_regression_rate >= args.minimum_first_choice_non_regression
        and guarded_non_regression_rate == 1.0
        and all(
            maximum_regressions[field] <= tolerance + 1e-12
            for field, tolerance in contract.to_dict().items()
        )
    )
    metrics = {
        "accepted": accepted,
        "selection_cases": len(all_selections),
        "exact_selection_cases": sum(bool(item["exact_selection"]) for item in all_selections),
        "held_action_selection_accuracy": selection_accuracy,
        "mean_oracle_regret": mean_regret,
        "constrained_utility_rmse": rmse,
        "mean_gain_over_raw": mean_gain_raw,
        "mean_gain_over_generic_one_step": mean_gain_generic,
        "first_choice_non_regression_rate": first_choice_non_regression_rate,
        "guarded_non_regression_rate": guarded_non_regression_rate,
        "maximum_selected_regressions": maximum_regressions,
        "non_regression_contract": contract.to_dict(),
        "full_schedule_candidate_evaluations": 5,
        "mean_learned_route_candidate_evaluations": 1.0 + mean_repair_candidates,
        "candidate_evaluation_reduction_fraction": 1.0 - (1.0 + mean_repair_candidates) / 5.0,
        "acceptance": {
            "minimum_selection_accuracy": args.minimum_selection_accuracy,
            "minimum_gain_over_raw": args.minimum_gain_over_raw,
            "minimum_first_choice_non_regression": args.minimum_first_choice_non_regression,
            "maximum_mean_regret": 0.01,
            "required_guarded_non_regression_rate": 1.0,
        },
    }
    _write_json(experiment / "metrics.json", metrics)
    metadata = {
        "schema_version": "1.0.0",
        "method": "held_action_flower_repair_non_regression_ranking",
        "status": "accepted" if accepted else "rejected",
        "honest_status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "config": {
            "alpha": args.alpha,
            "seed": args.seed,
            "regression_penalty": args.regression_penalty,
            "non_regression_contract": contract.to_dict(),
            "evolution_glob": args.evolution_glob,
            "experiment_root": str(args.experiment_root),
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _package_versions(),
        "git": _git_state(project_root),
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "rows": len(rows),
            "candidate_groups": len(rows) // 5,
            "actions": actions,
            "campaigns": campaigns,
            "feature_names": list(feature_names()),
            "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in paths],
        },
        "metrics": metrics,
        "folds": fold_records,
        "checkpoint": {
            "path": str(final_checkpoint),
            "sha256": _sha256(final_checkpoint),
        },
        "coordinate_frames": {
            "diagnostics": "camera:H3_output_pixels from source-aligned video evaluations",
            "policy_output": "repair recipe selection only; no coordinate transform",
        },
        "limitations": [
            "Only 45 cached candidates from nine groups, one real source scene, three actions, and five fixed repair recipes are available.",
            "Held-action folds prevent action-label leakage but share the same source scene and repair implementations.",
            "The constrained target combines deterministic proxy mean score with hard capability-regression penalties; neither is human preference, contact physics, or real-robot success.",
            "The policy selects post-processing recipes; it does not fine-tune MiniMax-H3 or learn robot control.",
        ],
    }
    _write_json(experiment / "metadata.json", metadata)
    print(json.dumps({"experiment": str(experiment), **metrics}, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
