from __future__ import annotations

import json

import pytest

from phiagent.training.campaign import summarize_campaign


def _write_arm(root, seed: int, arm: str, accuracy: float) -> None:
    experiment = root / f"seed{seed}" / arm / "run"
    experiment.mkdir(parents=True)
    config = {
        "seed": seed,
        "examples": 1000,
        "epochs": 10,
        "gpu": 0 if arm == "epl" else 1,
        "mask_epl": arm == "mask",
        "experiment_root": str(experiment.parent),
    }
    metadata = {
        "config": config,
        "dataset": {
            "split_sizes": {"train": 700, "validation": 150, "test": 150},
            "label_counts": {"0": 500, "1": 500},
        },
    }
    (experiment / "metadata.json").write_text(json.dumps(metadata))
    (experiment / "metrics.json").write_text(
        json.dumps(
            {
                "test_accuracy": accuracy,
                "majority_accuracy": 0.5,
            }
        )
    )


def test_campaign_summary_requires_matched_positive_gain(tmp_path) -> None:
    for seed, masked in ((42, 0.82), (43, 0.84)):
        _write_arm(tmp_path, seed, "epl", 0.95)
        _write_arm(tmp_path, seed, "mask", masked)

    summary = summarize_campaign(tmp_path)

    assert summary["accepted"]
    assert summary["matched_seeds"] == 2
    assert summary["epl_mean_accuracy"] == pytest.approx(0.95)
    assert summary["mean_gain"] == pytest.approx(0.12)


def test_campaign_summary_rejects_unmatched_config(tmp_path) -> None:
    _write_arm(tmp_path, 42, "epl", 0.95)
    _write_arm(tmp_path, 42, "mask", 0.80)
    metadata_path = tmp_path / "seed42" / "mask" / "run" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["config"]["epochs"] = 11
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="unmatched training config"):
        summarize_campaign(tmp_path)
