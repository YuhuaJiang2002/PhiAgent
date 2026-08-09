"""Matched EPL-policy campaign aggregation."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

_IGNORED_MATCH_KEYS = {"experiment_root", "gpu", "mask_epl"}


def _single_json(arm: Path, name: str) -> tuple[Path, dict[str, Any]]:
    matches = tuple(arm.glob(f"*/{name}"))
    if len(matches) != 1:
        raise ValueError(f"expected one {name} under {arm}, found {len(matches)}")
    return matches[0], json.loads(matches[0].read_text())


def summarize_campaign(root: Path, minimum_mean_gain: float = 0.05) -> dict[str, Any]:
    if not math.isfinite(minimum_mean_gain) or minimum_mean_gain <= 0:
        raise ValueError("minimum_mean_gain must be finite and positive")
    seed_directories = sorted(path for path in root.glob("seed*") if path.is_dir())
    if not seed_directories:
        raise ValueError(f"no matched seed directories found under {root}")

    pairs: list[dict[str, Any]] = []
    for seed_directory in seed_directories:
        epl_metadata_path, epl_metadata = _single_json(
            seed_directory / "epl", "metadata.json"
        )
        mask_metadata_path, mask_metadata = _single_json(
            seed_directory / "mask", "metadata.json"
        )
        _, epl_metrics = _single_json(seed_directory / "epl", "metrics.json")
        _, mask_metrics = _single_json(seed_directory / "mask", "metrics.json")
        epl_config = {
            key: value
            for key, value in epl_metadata["config"].items()
            if key not in _IGNORED_MATCH_KEYS
        }
        mask_config = {
            key: value
            for key, value in mask_metadata["config"].items()
            if key not in _IGNORED_MATCH_KEYS
        }
        if epl_config != mask_config:
            raise ValueError(f"unmatched training config for {seed_directory.name}")
        for key in ("split_sizes", "label_counts"):
            if epl_metadata["dataset"][key] != mask_metadata["dataset"][key]:
                raise ValueError(
                    f"unmatched dataset {key} for {seed_directory.name}"
                )
        seed = int(epl_metadata["config"]["seed"])
        if seed != int(mask_metadata["config"]["seed"]):
            raise ValueError(f"unmatched seed for {seed_directory.name}")
        epl_accuracy = float(epl_metrics["test_accuracy"])
        mask_accuracy = float(mask_metrics["test_accuracy"])
        pairs.append(
            {
                "seed": seed,
                "epl_accuracy": epl_accuracy,
                "masked_accuracy": mask_accuracy,
                "gain": epl_accuracy - mask_accuracy,
                "majority_accuracy": float(epl_metrics["majority_accuracy"]),
                "epl_metadata": str(epl_metadata_path),
                "masked_metadata": str(mask_metadata_path),
            }
        )

    gains = [pair["gain"] for pair in pairs]
    epl_accuracies = [pair["epl_accuracy"] for pair in pairs]
    masked_accuracies = [pair["masked_accuracy"] for pair in pairs]
    mean_gain = statistics.fmean(gains)
    accepted = all(gain > 0 for gain in gains) and mean_gain >= minimum_mean_gain
    return {
        "schema_version": "1.0.0",
        "accepted": accepted,
        "matched_seeds": len(pairs),
        "minimum_mean_gain": minimum_mean_gain,
        "epl_mean_accuracy": statistics.fmean(epl_accuracies),
        "masked_mean_accuracy": statistics.fmean(masked_accuracies),
        "mean_gain": mean_gain,
        "gain_population_std": statistics.pstdev(gains),
        "pairs": pairs,
        "limitations": [
            "Evidence is from a deterministic synthetic repair-action classification task.",
            "Accuracy does not establish simulator, real-robot, or PhiZero performance.",
        ],
    }
