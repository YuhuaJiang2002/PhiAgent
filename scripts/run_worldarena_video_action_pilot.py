#!/usr/bin/env python3
"""Train and evaluate a task-disjoint WorldArena video-to-EEF ridge pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.promotion import paired_bootstrap_lower_bound  # noqa: E402
from phiagent.acwm.worldarena import attach_worldarena_lineage  # noqa: E402
from phiagent.labeling.video_action import (  # noqa: E402
    ROTATION_GROUPS,
    aggregate_video_action_groups,
    integrate_eef_deltas,
    video_action_episode_metrics,
    wrap_radians,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "phiagent_video_ridge_eef_delta_v2"
BASELINES = ("zero_delta", "train_mean_delta")
PRIMARY_METRICS = (
    ("normalized_delta_rmse", False),
    ("translation_delta_rmse_cm", False),
    ("rotation_delta_geodesic_deg", False),
    ("absolute_translation_rmse_cm", False),
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


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _git_state(commit: str | None, branch: str | None) -> dict[str, object]:
    if (commit is None) != (branch is None):
        raise ValueError("--git-commit and --git-branch must be supplied together")
    if commit is not None:
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError("--git-commit must be a lowercase 40-character SHA-1")
        return {
            "commit": commit,
            "branch": branch,
            "resolution": "explicit immutable source snapshot",
            "dirty": True,
        }
    return {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "resolution": "local Git checkout",
        "dirty": subprocess.run(
            ["git", "diff", "--quiet"], cwd=PROJECT_ROOT, check=False
        ).returncode
        != 0,
    }


def _package_versions() -> dict[str, str]:
    result = {}
    for name in ("numpy", "opencv-python", "opencv-python-headless", "pyarrow"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _decode(cv2: Any, path: Path) -> list[Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not decode video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    return frames


def _transition_features(cv2: Any, np: Any, previous: Any, current: Any) -> Any:
    size = (64, 48)
    previous_gray = cv2.resize(
        cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    current_gray = cv2.resize(
        cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    difference = (current_gray - previous_gray) / 255.0
    absolute = np.abs(difference)
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray.astype(np.uint8),
        current_gray.astype(np.uint8),
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    magnitude = np.linalg.norm(flow, axis=2)
    features = [
        float(np.mean(previous_gray) / 255.0),
        float(np.mean(current_gray) / 255.0),
        float(np.mean(difference)),
        float(np.mean(absolute)),
        float(np.std(absolute)),
        float(np.mean(flow[:, :, 0])),
        float(np.mean(flow[:, :, 1])),
        float(np.mean(magnitude)),
        float(np.std(magnitude)),
    ]
    height, width = previous_gray.shape
    for y0, y1 in ((0, height // 2), (height // 2, height)):
        for x0, x1 in ((0, width // 2), (width // 2, width)):
            region = np.s_[y0:y1, x0:x1]
            features.extend(
                (
                    float(np.mean(difference[region])),
                    float(np.mean(absolute[region])),
                    float(np.mean(flow[region][..., 0])),
                    float(np.mean(flow[region][..., 1])),
                    float(np.mean(magnitude[region])),
                    float(np.std(magnitude[region])),
                )
            )
    weights = absolute + 1e-8
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    features.extend(
        (
            float(np.sum(weights * x_grid) / np.sum(weights) / max(width - 1, 1)),
            float(np.sum(weights * y_grid) / np.sum(weights) / max(height - 1, 1)),
        )
    )
    grid_size = (8, 6)
    for field in (
        difference,
        absolute,
        flow[:, :, 0],
        flow[:, :, 1],
        magnitude,
    ):
        features.extend(
            cv2.resize(field, grid_size, interpolation=cv2.INTER_AREA)
            .astype(np.float64)
            .ravel()
            .tolist()
        )
    return np.asarray(features, dtype=np.float64)


def _target_deltas(np: Any, states: Any) -> Any:
    deltas = np.diff(states, axis=0)
    for group in ROTATION_GROUPS:
        for channel in group:
            deltas[:, channel] = np.vectorize(wrap_radians)(deltas[:, channel])
    return deltas


def _load_episode(cv2: Any, np: Any, pq: Any, root: Path, row: dict[str, Any]) -> dict:
    video = root / str(row["video"]["data"])
    action = root / str(row["action"]["data"])
    frames_all = _decode(cv2, video)
    frames = frames_all[
        int(row["video"]["start_frame"]) : int(row["video"]["end_frame"]) + 1
    ]
    table = pq.read_table(action, columns=["observation.state"])
    states_all = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    states = states_all[
        int(row["action"]["start_frame"]) : int(row["action"]["end_frame"]) + 1
    ]
    if len(frames) != len(states) or len(frames) != int(row["length"]):
        raise ValueError(f"video/state length mismatch for {row['source_episode']}")
    features = np.stack(
        [
            _transition_features(cv2, np, previous, current)
            for previous, current in zip(frames, frames[1:])
        ]
    )
    deltas = _target_deltas(np, states)
    return {
        "row": row,
        "features": features,
        "deltas": deltas,
        "states": states,
        "video_path": video,
        "video_sha256": _sha256(video),
        "action_path": action,
        "action_sha256": _sha256(action),
    }


def _ridge_fit(np: Any, features: Any, targets: Any, alpha: float) -> dict[str, Any]:
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge alpha must be finite and positive")
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (features - mean) / scale
    design = np.concatenate(
        [np.ones((len(standardized), 1), dtype=np.float64), standardized], axis=1
    )
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ targets,
    )
    return {
        "alpha": alpha,
        "feature_mean": mean,
        "feature_scale": scale,
        "coefficients": coefficients,
    }


def _ridge_predict(np: Any, model: dict[str, Any], features: Any) -> Any:
    standardized = (features - model["feature_mean"]) / model["feature_scale"]
    design = np.concatenate(
        [np.ones((len(standardized), 1), dtype=np.float64), standardized], axis=1
    )
    return design @ model["coefficients"]


def _method_records(
    np: Any,
    episodes: list[dict[str, Any]],
    model: dict[str, Any],
    train_mean: Any,
    channel_scale: Any,
) -> list[dict[str, Any]]:
    records = []
    for episode in episodes:
        targets = episode["deltas"]
        predictions = {
            MODEL_ID: _ridge_predict(np, model, episode["features"]),
            "zero_delta": np.zeros_like(targets),
            "train_mean_delta": np.repeat(train_mean[None, :], len(targets), axis=0),
        }
        for method, predicted in predictions.items():
            predicted_states = integrate_eef_deltas(
                episode["states"][0].tolist(),
                predicted.tolist(),
            )
            metrics = video_action_episode_metrics(
                predicted.tolist(),
                targets.tolist(),
                predicted_states,
                episode["states"].tolist(),
                channel_scale=channel_scale.tolist(),
            )
            row = episode["row"]
            records.append(
                {
                    "method": method,
                    "source_episode": row["source_episode"],
                    "independent_group_id": row["independent_group_id"],
                    "coordinate_frame": row["coordinate_frame"],
                    "metrics": metrics,
                }
            )
    return records


def _comparisons(
    aggregates: dict[str, dict[str, Any]],
    *,
    minimum_independent_groups: int,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    comparisons = []
    all_positive = True
    for baseline in BASELINES:
        for metric, higher_is_better in PRIMARY_METRICS:
            candidate = {
                group: values[metric]
                for group, values in aggregates[MODEL_ID]["per_group"].items()
            }
            baseline_values = {
                group: values[metric]
                for group, values in aggregates[baseline]["per_group"].items()
            }
            evidence = paired_bootstrap_lower_bound(
                candidate,
                baseline_values,
                higher_is_better=higher_is_better,
                seed=seed + len(comparisons),
                iterations=bootstrap_iterations,
                confidence=0.95,
            )
            positive = float(evidence["lower_confidence_bound"]) > 0.0
            all_positive = all_positive and positive
            comparisons.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "direction": "higher" if higher_is_better else "lower",
                    "positive_lower_bound": positive,
                    **evidence,
                }
            )
    independent = int(aggregates[MODEL_ID]["independent_groups"])
    eligible = independent >= minimum_independent_groups
    return {
        "independent_groups": independent,
        "minimum_independent_groups": minimum_independent_groups,
        "decision_eligible": eligible,
        "all_primary_metric_lower_bounds_positive": all_positive,
        "promotion_eligible": eligible and all_positive,
        "comparisons": comparisons,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--test-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        action="append",
        default=[0.001, 0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument("--minimum-independent-groups", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--git-commit")
    parser.add_argument("--git-branch")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.dataset_root.expanduser().resolve()
    dataset_manifest_path = args.dataset_manifest.expanduser().resolve()
    metadata_paths = {
        split: getattr(args, f"{split}_metadata").expanduser().resolve()
        for split in ("train", "validation", "test")
    }
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pilot: {output}")
    if not root.is_dir():
        raise ValueError(f"dataset root is missing: {root}")
    for path in (dataset_manifest_path, *metadata_paths.values()):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required pilot input is missing or empty: {path}")
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    _write_json(output / "git-state.json", _git_state(args.git_commit, args.git_branch))
    _write_json(
        output / "config.json",
        {
            "schema_version": "1.0.0",
            "status": "STARTED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(root),
            "dataset_manifest": str(dataset_manifest_path),
            "metadata": {split: str(path) for split, path in metadata_paths.items()},
            "alphas": sorted(set(args.alpha)),
            "minimum_independent_groups": args.minimum_independent_groups,
            "bootstrap_iterations": args.bootstrap_iterations,
            "seed": args.seed,
            "packages": _package_versions(),
        },
    )
    dataset_manifest = _json(dataset_manifest_path)
    split_rows = {
        split: list(
            attach_worldarena_lineage(_rows(path), dataset_manifest)
        )
        for split, path in metadata_paths.items()
    }
    task_sets = {
        split: {str(row["task"]) for row in rows}
        for split, rows in split_rows.items()
    }
    if (
        task_sets["train"] & task_sets["validation"]
        or task_sets["train"] & task_sets["test"]
        or task_sets["validation"] & task_sets["test"]
    ):
        raise ValueError("WorldArena video-action tasks must be split-disjoint")
    frames = {str(row["coordinate_frame"]) for rows in split_rows.values() for row in rows}
    if len(frames) != 1 or not next(iter(frames)).startswith("robot_base:"):
        raise ValueError("video-action labels require one explicit robot_base frame")

    import cv2
    import numpy as np
    import pyarrow.parquet as pq

    episodes = {
        split: [_load_episode(cv2, np, pq, root, row) for row in rows]
        for split, rows in split_rows.items()
    }
    train_features = np.concatenate(
        [episode["features"] for episode in episodes["train"]], axis=0
    )
    train_targets = np.concatenate(
        [episode["deltas"] for episode in episodes["train"]], axis=0
    )
    channel_scale = np.std(train_targets, axis=0)
    channel_scale = np.where(channel_scale < 1e-6, 1e-6, channel_scale)
    train_mean = np.mean(train_targets, axis=0)
    validation_scores = []
    fitted = {}
    validation_features = np.concatenate(
        [episode["features"] for episode in episodes["validation"]], axis=0
    )
    validation_targets = np.concatenate(
        [episode["deltas"] for episode in episodes["validation"]], axis=0
    )
    validation_channel_errors = {}
    for alpha in sorted(set(args.alpha)):
        model = _ridge_fit(np, train_features, train_targets, alpha)
        fitted[alpha] = model
        validation_prediction = _ridge_predict(np, model, validation_features)
        validation_channel_errors[alpha] = np.mean(
            ((validation_prediction - validation_targets) / channel_scale) ** 2,
            axis=0,
        )
        records = _method_records(
            np,
            episodes["validation"],
            model,
            train_mean,
            channel_scale,
        )
        aggregates = aggregate_video_action_groups(records)
        validation_scores.append(
            {
                "alpha": alpha,
                "normalized_delta_rmse": aggregates[MODEL_ID]["mean"][
                    "normalized_delta_rmse"
                ],
            }
        )
    selected_alphas = [
        min(
            validation_channel_errors,
            key=lambda alpha: (validation_channel_errors[alpha][channel], alpha),
        )
        for channel in range(14)
    ]
    selected = {
        "alpha": selected_alphas,
        "feature_mean": next(iter(fitted.values()))["feature_mean"],
        "feature_scale": next(iter(fitted.values()))["feature_scale"],
        "coefficients": np.stack(
            [
                fitted[alpha]["coefficients"][:, channel]
                for channel, alpha in enumerate(selected_alphas)
            ],
            axis=1,
        ),
    }
    test_records = _method_records(
        np,
        episodes["test"],
        selected,
        train_mean,
        channel_scale,
    )
    aggregates = aggregate_video_action_groups(test_records)
    comparison = _comparisons(
        aggregates,
        minimum_independent_groups=args.minimum_independent_groups,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    model_payload = {
        "schema_version": "1.0.0",
        "model_id": MODEL_ID,
        "status": "WORKING",
        "selected_alpha_by_channel": selected_alphas,
        "feature_count": int(len(selected["feature_mean"])),
        "output_channels": 14,
        "coordinate_frame": next(iter(frames)),
        "input_information": (
            "offline adjacent RGB frames plus one measured initial 14-D EEF state for "
            "absolute trajectory integration"
        ),
        "feature_mean": selected["feature_mean"].tolist(),
        "feature_scale": selected["feature_scale"].tolist(),
        "coefficients": selected["coefficients"].tolist(),
        "channel_scale": channel_scale.tolist(),
        "train_mean_delta": train_mean.tolist(),
        "validation_scores": validation_scores,
        "validation_normalized_mse_by_alpha": {
            str(alpha): errors.tolist()
            for alpha, errors in validation_channel_errors.items()
        },
    }
    _write_json(output / "model.json", model_payload)
    provenance = [
        {
            "split": split,
            "source_episode": episode["row"]["source_episode"],
            "independent_group_id": episode["row"]["independent_group_id"],
            "video": str(episode["video_path"]),
            "video_sha256": episode["video_sha256"],
            "action": str(episode["action_path"]),
            "action_sha256": episode["action_sha256"],
        }
        for split, split_episodes in episodes.items()
        for episode in split_episodes
    ]
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _package_versions(),
        "seed": args.seed,
        "dataset": {
            "root": str(root),
            "manifest": str(dataset_manifest_path),
            "manifest_sha256": _sha256(dataset_manifest_path),
            "metadata": {
                split: {"path": str(path), "sha256": _sha256(path)}
                for split, path in metadata_paths.items()
            },
            "tasks": {split: sorted(tasks) for split, tasks in task_sets.items()},
        },
        "provenance": provenance,
        "model": model_payload,
        "test_records": test_records,
        "test_aggregates": aggregates,
        "comparison": comparison,
        "abstentions": {
            "contact": "not predicted: no force/tactile ground truth in compiled lane",
            "phase": "not predicted in this geometric pilot",
            "robot_base_without_initial_state": (
                "abstain: video alone lacks a calibrated camera-to-base transform"
            ),
        },
        "claim_boundary": (
            "This is a task-disjoint, single-camera, single-embodiment pilot with only "
            f"{comparison['independent_groups']} independent test episodes. It evaluates "
            "offline RGB-to-realized-EEF labeling, not low-level commands, contact, phase, "
            "cross-scene generalization, or SOTA."
        ),
    }
    _write_json(output / "evaluation.json", result)
    (output / "run.log").write_text(
        f"selected_alpha_by_channel={selected_alphas}; "
        f"independent_test_groups={comparison['independent_groups']}; "
        f"promotion_eligible={comparison['promotion_eligible']}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
