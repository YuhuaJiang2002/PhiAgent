from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.recover_cosmos3_droid_sft_export import (
    _source_training_artifacts,
    _tree_inventory,
)
from scripts.run_cosmos3_droid_sft_training import write_single_process_export_config


def test_recovery_accepts_only_completed_iteration_500(tmp_path: Path) -> None:
    experiment = tmp_path / "training"
    run = tmp_path / "resolved-run"
    checkpoint = run / "checkpoints/iter_000000500"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.distcp").write_bytes(b"weights")
    (run / "checkpoints/latest_checkpoint.txt").write_text("iter_000000500\n")
    (run / "config.yaml").write_text("model: {}\n")
    experiment.mkdir()
    (experiment / "training.log").write_text("Done with training.\n")
    (experiment / "metadata.json").write_text(
        json.dumps({"expected_run_dir": str(run)})
    )
    result = _source_training_artifacts(experiment)
    assert result["checkpoint"] == checkpoint.resolve()
    assert _tree_inventory(checkpoint)["bytes"] == 7


def test_recovery_rejects_nonterminal_training(tmp_path: Path) -> None:
    experiment = tmp_path / "training"
    experiment.mkdir()
    (experiment / "training.log").write_text("still running\n")
    (experiment / "metadata.json").write_text(
        json.dumps({"expected_run_dir": str(tmp_path / "missing")})
    )
    with pytest.raises(ValueError):
        _source_training_artifacts(experiment)


def test_recovery_export_copy_matches_single_process_world_size(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text("context_parallel_shard_degree: 2\n")
    output = write_single_process_export_config(source, tmp_path / "export.yaml")
    assert output.read_text() == "context_parallel_shard_degree: 1\n"
    assert source.read_text() == "context_parallel_shard_degree: 2\n"
