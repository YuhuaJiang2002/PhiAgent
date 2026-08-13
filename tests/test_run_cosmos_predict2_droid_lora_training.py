from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_cosmos_predict2_droid_lora_training import (
    _parser,
    stage_training_overlay,
    validate_training_dataset,
)


def _make_dataset(tmp_path, *, embedding: bool = True):
    root = tmp_path / "composite"
    train = root / "train"
    for folder in ("videos", "metas", "t5_xxl"):
        (train / folder).mkdir(parents=True, exist_ok=True)
    (train / "videos/sample.mp4").write_bytes(b"video")
    (train / "metas/sample.txt").write_text("prompt")
    if embedding:
        (train / "t5_xxl/sample.pickle").write_bytes(b"embedding")
    (root / "dataset-contract.json").write_text(
        json.dumps(
            {
                "split_counts": {"train": 1},
                "video_contract": {"training_window_frames": 93},
                "leakage_checks": {"final_holdout_used_for_training": False},
            }
        )
    )
    return train


def test_training_dataset_requires_exact_video_meta_embedding_alignment(tmp_path) -> None:
    result = validate_training_dataset(_make_dataset(tmp_path))
    assert result["sample_count"] == 1


def test_training_default_uses_capacity_preserving_rank16() -> None:
    assert _parser().get_default("lora_rank") == 16
    assert _parser().get_default("train_frames") == 45
    assert _parser().get_default("learning_rate") == pytest.approx(1e-4)


def test_training_dataset_rejects_missing_embedding(tmp_path) -> None:
    with pytest.raises(ValueError, match="sample IDs differ"):
        validate_training_dataset(_make_dataset(tmp_path, embedding=False))


def test_overlay_staging_archives_prior_runtime_copies(tmp_path) -> None:
    repo = tmp_path / "cosmos"
    experiment_dir = repo / "cosmos_predict2/configs/base/experiment"
    experiment_dir.mkdir(parents=True)
    old = experiment_dir / "phiagent_droid_lora_old.py"
    old.write_text("old")
    source = tmp_path / "overlay.py"
    source.write_text("current")
    output = tmp_path / "run"
    output.mkdir()

    installed, retired = stage_training_overlay(repo, source, output)

    assert installed.read_text() == "current"
    assert not old.exists()
    assert (output / "retired-external-overlays/phiagent_droid_lora_old.py").read_text() == "old"
    assert retired[0]["original_path"] == str(old)


def test_overlay_matches_single_frame_deployment_conditioning() -> None:
    overlay = (
        Path(__file__).resolve().parents[1]
        / "third_party_overlays/cosmos_predict2/phiagent_droid_lora.py"
    ).read_text()
    assert "pipe_config.min_num_conditional_frames = 1" in overlay
    assert "pipe_config.max_num_conditional_frames = 1" in overlay
    assert 'MAX_ITER = _positive_env_int("PHIAGENT_MAX_ITER", 1500)' in overlay
    assert 'GPU_COUNT = _positive_env_int("PHIAGENT_GPU_COUNT", 8)' in overlay
    assert 'LORA_RANK = _positive_env_int("PHIAGENT_LORA_RANK", 16)' in overlay
    assert 'LEARNING_RATE = _bounded_env_float("PHIAGENT_LEARNING_RATE", 1e-4, 1e-3)' in overlay
    assert 'TRAIN_FRAMES = _positive_env_int("PHIAGENT_TRAIN_FRAMES", 45)' in overlay
    assert "num_frames=TRAIN_FRAMES" in overlay
    assert "pipe_config.state_t = TRAIN_STATE_T" in overlay
    assert "if TRAIN_STATE_T % GPU_COUNT" in overlay
    assert "optimizer=dict(lr=LEARNING_RATE" in overlay
    assert "weights[..., : height // 2, :] = 1.4" in overlay
    assert "weights[..., height // 2 :, : width // 2] = 0.8" in overlay
    assert "weights[..., height // 2 :, width // 2 :] = 0.1" in overlay
