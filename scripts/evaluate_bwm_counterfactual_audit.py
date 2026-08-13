#!/usr/bin/env python3
"""Evaluate matched BWM factual/history-preserving action-swap generations."""

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

from evaluate_bwm_heldout_pair import _decode, _flow_metrics, _ssim  # noqa: E402
from phiagent.acwm.counterfactual import compare_counterfactual_models  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_METRICS = {
    "factual_future_ssim": "higher",
    "factual_flow_endpoint_error_px_at_224x168": "lower",
    "wrong_action_ssim_margin": "higher",
    "wrong_action_flow_epe_margin": "higher",
}


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
    for name in ("numpy", "opencv-python", "opencv-python-headless"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def audit_limitations(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        raise ValueError("counterfactual audit requires records")
    independent_units = {
        str(record.get("independent_unit_id", "")).strip()
        for record in records
    }
    task_names = {
        str(record.get("trial_id", "")).split("/", maxsplit=1)[0].strip()
        for record in records
    }
    seeds = {int(record["seed"]) for record in records}
    if "" in independent_units or "" in task_names:
        raise ValueError("audit records require independent-unit and task lineage")
    task_label = "task" if len(task_names) == 1 else "tasks"
    limitations = [
        (
            f"The suite contains {len(independent_units)} independent source episodes "
            f"across {len(task_names)} {task_label}: {', '.join(sorted(task_names))}."
        )
    ]
    if len(seeds) == 1:
        limitations.append(
            f"The suite uses one inference seed ({next(iter(seeds))}); "
            "seed sensitivity is not estimated."
        )
    else:
        limitations.append(
            "Inference seeds are repeated measurements and are averaged before bootstrap."
        )
    limitations.extend(
        [
            "The wrong-action rollout has no physical reference and is diagnostic only.",
            "Conditioning is realized absolute EEF state, not low-level robot command.",
        ]
    )
    return limitations


def _video_index(root: Path) -> dict[int, Path]:
    if (root / "videos").is_dir():
        root = root / "videos"
    result: dict[int, Path] = {}
    for path in root.glob("**/episode*.mp4"):
        suffix = path.stem.removeprefix("episode")
        if not suffix.isdigit():
            continue
        episode_index = int(suffix)
        if episode_index in result:
            raise ValueError(f"duplicate generated episode{episode_index}.mp4 under {root}")
        result[episode_index] = path
    if not result:
        raise ValueError(f"no generated episode videos found under {root}")
    return result


def _resize(cv2: Any, frames: list[Any], size: tuple[int, int]) -> list[Any]:
    return [
        cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if frame.shape[1::-1] != size
        else frame
        for frame in frames
    ]


def _mean_ssim(cv2: Any, np: Any, first: list[Any], second: list[Any]) -> float:
    return float(
        np.mean([_ssim(cv2, np, left, right) for left, right in zip(first, second)])
    )


def _measure_pair(
    cv2: Any,
    np: Any,
    factual: list[Any],
    swapped: list[Any],
    reference: list[Any],
    history_frames: int,
) -> dict[str, float]:
    if not (len(factual) == len(swapped) == len(reference)):
        raise ValueError("generated and reference frame counts do not match")
    if not 0 < history_frames < len(reference):
        raise ValueError("invalid history frame count")
    future = slice(history_frames, None)
    flow_slice = slice(history_frames - 1, None)
    factual_future = factual[future]
    swapped_future = swapped[future]
    reference_future = reference[future]
    factual_ssim = _mean_ssim(cv2, np, factual_future, reference_future)
    swapped_ssim = _mean_ssim(cv2, np, swapped_future, reference_future)
    factual_epe, factual_cosine = _flow_metrics(
        cv2, np, factual[flow_slice], reference[flow_slice]
    )
    swapped_epe, swapped_cosine = _flow_metrics(
        cv2, np, swapped[flow_slice], reference[flow_slice]
    )
    pair_mad = float(
        np.mean(
            [
                np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32)))
                for left, right in zip(factual_future, swapped_future)
            ]
        )
        / 255.0
    )
    history_mad = float(
        np.mean(
            [
                np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32)))
                for left, right in zip(
                    factual[:history_frames], swapped[:history_frames]
                )
            ]
        )
        / 255.0
    )
    factual_endpoint = _ssim(cv2, np, factual[-1], reference[-1])
    swapped_endpoint = _ssim(cv2, np, swapped[-1], reference[-1])
    values = {
        "factual_future_ssim": factual_ssim,
        "swapped_future_ssim_to_factual_reference": swapped_ssim,
        "wrong_action_ssim_margin": factual_ssim - swapped_ssim,
        "action_effect_mad_0_1": pair_mad,
        "history_pair_mad_0_1": history_mad,
        "factual_flow_endpoint_error_px_at_224x168": factual_epe,
        "swapped_flow_endpoint_error_px_at_224x168": swapped_epe,
        "wrong_action_flow_epe_margin": swapped_epe - factual_epe,
        "factual_flow_direction_cosine": factual_cosine,
        "swapped_flow_direction_cosine": swapped_cosine,
        "wrong_action_flow_cosine_margin": factual_cosine - swapped_cosine,
        "factual_endpoint_ssim": factual_endpoint,
        "swapped_endpoint_ssim_to_factual_reference": swapped_endpoint,
        "wrong_action_endpoint_ssim_margin": factual_endpoint - swapped_endpoint,
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("counterfactual evaluation produced a non-finite metric")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("MODEL_ID", "SEED", "VIDEO_ROOT"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--minimum-independent-trials", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--git-commit")
    parser.add_argument("--git-branch")
    return parser


def main() -> int:
    args = _parser().parse_args()
    suite_path = args.suite_manifest.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    suite_manifest = _json(suite_path)
    if (args.run_manifest is None) == (args.run is None):
        raise ValueError("provide exactly one of --run-manifest or one or more --run")
    runs_path = (
        args.run_manifest.expanduser().resolve()
        if args.run_manifest is not None
        else None
    )
    if runs_path is not None:
        run_manifest = _json(runs_path)
    else:
        run_manifest = {
            "schema_version": "1.0.0",
            "runs": [
                {
                    "model_id": model_id,
                    "seed": int(seed),
                    "video_root": str(Path(video_root).expanduser().resolve()),
                }
                for model_id, seed, video_root in args.run
            ],
        }
    if suite_manifest.get("swap_mode") != "history_preserving_rebased_future":
        raise ValueError("only the history-preserving counterfactual suite is claim-eligible")
    suite = Path(str(suite_manifest["suite"]["path"])).expanduser().resolve()
    dataset_root = Path(str(suite_manifest["dataset_root"])).expanduser().resolve()
    if _sha256(suite) != suite_manifest["suite"]["sha256"]:
        raise ValueError("counterfactual suite hash does not match its manifest")
    rows = [json.loads(line) for line in suite.read_text().splitlines() if line.strip()]
    by_index = {int(row["episode_index"]): row for row in rows}
    runs = run_manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("run manifest requires a non-empty runs array")

    import cv2
    import numpy as np

    records = []
    artifacts = []
    reference_cache: dict[str, list[Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("run entries must be objects")
        model_id = str(run.get("model_id", "")).strip()
        seed = int(run["seed"])
        video_root = Path(str(run["video_root"])).expanduser().resolve()
        videos = _video_index(video_root)
        for factual_row in rows[::2]:
            metadata = factual_row.get("counterfactual")
            if not isinstance(metadata, dict) or metadata.get("variant") != "factual":
                raise ValueError("suite rows must be ordered factual, swapped")
            factual_index = int(factual_row["episode_index"])
            swapped_index = int(metadata["paired_episode_index"])
            swapped_row = by_index.get(swapped_index)
            if (
                not isinstance(swapped_row, dict)
                or swapped_row.get("counterfactual", {}).get("variant") != "swapped"
            ):
                raise ValueError("factual row lacks a valid swapped partner")
            if factual_index not in videos or swapped_index not in videos:
                raise ValueError(
                    f"{model_id}/seed-{seed} lacks generated videos for a trial pair"
                )
            video_payload = factual_row["video"]
            reference_path = dataset_root / str(video_payload["data"])
            cache_key = str(reference_path)
            if cache_key not in reference_cache:
                all_reference = _decode(cv2, reference_path)
                reference_cache[cache_key] = all_reference[
                    int(video_payload["start_frame"]) : int(video_payload["end_frame"]) + 1
                ]
            reference = reference_cache[cache_key]
            factual_path = videos[factual_index]
            swapped_path = videos[swapped_index]
            size = reference[0].shape[1], reference[0].shape[0]
            factual = _resize(cv2, _decode(cv2, factual_path), size)
            swapped = _resize(cv2, _decode(cv2, swapped_path), size)
            trial_id = str(metadata["trial_id"])
            records.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "trial_id": trial_id,
                    "independent_unit_id": metadata["independent_group_id"],
                    "source_episode": metadata["source_episode"],
                    "metrics": _measure_pair(
                        cv2,
                        np,
                        factual,
                        swapped,
                        reference,
                        int(factual_row["history_frames"]),
                    ),
                }
            )
            artifacts.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "trial_id": trial_id,
                    "factual_video": str(factual_path),
                    "factual_sha256": _sha256(factual_path),
                    "swapped_video": str(swapped_path),
                    "swapped_sha256": _sha256(swapped_path),
                    "reference_video": str(reference_path),
                    "reference_sha256": _sha256(reference_path),
                }
            )
    comparison = compare_counterfactual_models(
        records,
        candidate_model=args.candidate_model,
        baseline_model=args.baseline_model,
        primary_metrics=PRIMARY_METRICS,
        minimum_independent_trials=args.minimum_independent_trials,
        bootstrap_iterations=args.bootstrap_iterations,
        confidence=args.confidence,
        seed=args.seed,
    )
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    _write_json(output / "git-state.json", _git_state(args.git_commit, args.git_branch))
    _write_json(output / "runs.json", run_manifest)
    result = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _package_versions(),
        "suite_manifest": str(suite_path),
        "suite_manifest_sha256": _sha256(suite_path),
        "run_manifest": str(runs_path) if runs_path is not None else "inline --run arguments",
        "run_manifest_sha256": _sha256(runs_path) if runs_path is not None else None,
        "records": records,
        "artifacts": artifacts,
        "comparison": comparison,
        "limitations": audit_limitations(records),
    }
    _write_json(output / "evaluation.json", result)
    (output / "evaluation.log").write_text(
        f"evaluated {len(records)} model-seed-trial records; "
        f"decision_eligible={comparison['decision_eligible']}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
