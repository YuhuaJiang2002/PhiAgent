from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.run_cosmos3_droid_sft_training import (
    DATASET_PATH,
    bundle_text_tokenizer,
    build_command,
    build_export_command,
    build_tail_overrides,
    project_provenance,
    require_executable,
    resolve_training_profile,
    validate_dataset_lineage_audit,
    validate_sft_dataset,
    validate_text_tokenizer,
    validate_training_gpu_count,
    write_single_process_export_config,
)


def test_training_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    target = tmp_path / "runtime/python3"
    target.parent.mkdir()
    target.write_text("#!/bin/sh\n")
    target.chmod(0o755)
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(target)
    assert require_executable(python, "Python") == python.absolute()
    assert require_executable(python, "Python") != target.resolve()


def test_training_overrides_force_identity_focused_i2v() -> None:
    overrides = build_tail_overrides(
        run_name="test_run",
        seed=17,
        steps=25,
        save_every=5,
        learning_rate=2e-5,
        warmup_steps=5,
        grad_accum=2,
        max_sequence_length=16384,
        num_video_frames=93,
        enable_ema=False,
        enable_compile=False,
        text_tokenizer_root=Path("/verified/tokenizer"),
    )
    assert (
        f"{DATASET_PATH}.conditioning_config={{0:0.0,1:1.0,2:0.0}}"
        in overrides
    )
    assert "model.config.vlm_config.tokenizer.config_variant=hf" in overrides
    assert (
        "model.config.vlm_config.tokenizer.pretrained_model_name=/verified/tokenizer"
        in overrides
    )
    assert f"{DATASET_PATH}.cfg_dropout_rate=0.0" in overrides
    assert f"{DATASET_PATH}.num_video_frames=93" in overrides
    assert f"{DATASET_PATH}.resolution='480'" in overrides
    assert "trainer.cudnn.deterministic=true" in overrides
    assert "trainer.seed=17" in overrides
    assert "optimizer.lr=2e-05" in overrides


def test_training_binds_complete_local_text_tokenizer(tmp_path: Path) -> None:
    for name in ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text(name)
    vocab_sha256 = hashlib.sha256((tmp_path / "vocab.json").read_bytes()).hexdigest()
    result = validate_text_tokenizer(tmp_path, vocab_sha256)
    assert result["root"] == str(tmp_path.resolve())
    assert result["vocab_sha256"] == vocab_sha256
    assert len(result["files"]) == 4
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_text_tokenizer(tmp_path, "0" * 64)


def test_export_bundles_verified_local_text_tokenizer(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name in (
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.json",
    ):
        (tokenizer / name).write_text(name)
    vocab_sha256 = hashlib.sha256((tokenizer / "vocab.json").read_bytes()).hexdigest()
    model = tmp_path / "model"
    result = bundle_text_tokenizer(tokenizer, model, vocab_sha256)
    assert {row["name"] for row in result["files"]} == {
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.json",
    }
    assert (model / "vocab.json").read_text() == "vocab.json"


def test_formal_profile_matches_official_nano_full_sft_schedule() -> None:
    profile = resolve_training_profile(
        profile="formal",
        steps=None,
        save_every=None,
        learning_rate=None,
        warmup_steps=None,
        max_sequence_length=None,
        enable_ema=None,
        enable_compile=None,
    )
    assert profile == {
        "profile": "formal",
        "steps": 500,
        "save_every": 100,
        "learning_rate": 1e-4,
        "warmup_steps": 50,
        "max_sequence_length": 45056,
        "num_video_frames": 93,
        "enable_ema": True,
        "enable_compile": False,
        "lora_enabled": False,
        "activation_checkpointing_mode": "selective",
        "context_parallel_shard_degree": 1,
    }


def test_smoke_profile_can_be_overridden_without_becoming_formal() -> None:
    profile = resolve_training_profile(
        profile="smoke",
        steps=3,
        save_every=None,
        learning_rate=None,
        warmup_steps=None,
        max_sequence_length=None,
        enable_ema=True,
        enable_compile=None,
    )
    assert profile["profile"] == "smoke"
    assert profile["steps"] == 3
    assert profile["save_every"] == 1
    assert profile["num_video_frames"] == 33
    assert profile["enable_ema"] is True


def test_smoke_allows_two_gpus_but_formal_still_requires_eight() -> None:
    validate_training_gpu_count("smoke", 2)
    validate_training_gpu_count("formal_lora", 2)
    validate_training_gpu_count("formal", 8)
    with pytest.raises(ValueError, match="at least two"):
        validate_training_gpu_count("smoke", 1)
    with pytest.raises(ValueError, match="exactly eight"):
        validate_training_gpu_count("formal", 4)
    with pytest.raises(ValueError, match="even GPU count"):
        validate_training_gpu_count("formal_lora", 3)


def test_formal_lora_matches_official_generation_adapter_recipe() -> None:
    profile = resolve_training_profile(
        profile="formal_lora",
        steps=None,
        save_every=None,
        learning_rate=None,
        warmup_steps=None,
        max_sequence_length=None,
        enable_ema=None,
        enable_compile=None,
    )
    assert profile["steps"] == 500
    assert profile["num_video_frames"] == 93
    assert profile["learning_rate"] == 5e-4
    assert profile["lora_enabled"] is True
    assert profile["lora_rank"] == 16
    assert profile["lora_alpha"] == 32
    assert profile["activation_checkpointing_mode"] == "full"
    assert profile["context_parallel_shard_degree"] == 2


def test_formal_lora_overrides_train_only_generation_adapters() -> None:
    overrides = build_tail_overrides(
        run_name="lora_run",
        seed=17,
        steps=500,
        save_every=100,
        learning_rate=5e-4,
        warmup_steps=50,
        grad_accum=2,
        max_sequence_length=45056,
        num_video_frames=93,
        enable_ema=False,
        enable_compile=False,
        lora_enabled=True,
        lora_rank=16,
        lora_alpha=32,
        activation_checkpointing_mode="full",
        context_parallel_shard_degree=2,
    )
    assert "model.config.lora_enabled=true" in overrides
    assert "model.config.lora_rank=16" in overrides
    assert "model.config.lora_alpha=32" in overrides
    assert (
        "model.config.lora_target_modules='q_proj_moe_gen,k_proj_moe_gen,"
        "v_proj_moe_gen,o_proj_moe_gen'"
    ) in overrides
    assert "optimizer.keys_to_select=[lora_]" in overrides
    assert "checkpoint.keys_to_skip_loading=[net_ema.,lora_]" in overrides
    assert "model.config.activation_checkpointing.mode=full" in overrides
    assert "model.config.parallelism.context_parallel_shard_degree=2" in overrides
    assert "model.config.max_num_tokens_after_packing=45056" in overrides
    assert (
        "dataloader_train.dataloader.datasets.video.dataset."
        "conditioning_config={0:0.0,1:1.0,2:0.0}"
    ) in overrides


def test_training_overrides_reject_unsafe_schedule() -> None:
    with pytest.raises(ValueError, match="save-every"):
        build_tail_overrides(
            run_name="test",
            seed=1,
            steps=2,
            save_every=3,
            learning_rate=2e-5,
            warmup_steps=1,
            grad_accum=1,
            max_sequence_length=16384,
            num_video_frames=33,
            enable_ema=False,
            enable_compile=False,
        )


def test_training_command_uses_torchrun_and_tail_overrides(tmp_path: Path) -> None:
    python = tmp_path / ".venv/bin/python"
    torchrun = python.parent / "torchrun"
    torchrun.parent.mkdir(parents=True)
    torchrun.write_text("#!/bin/sh\n")
    command = build_command(
        python=python,
        gpu_count=4,
        master_port=29641,
        toml=tmp_path / "recipe.toml",
        overrides=["trainer.max_iter=2"],
    )
    assert command[0] == str(torchrun)
    assert "--nproc-per-node=4" in command
    assert "cosmos_framework.scripts.train" in command
    assert command[-2:] == ["--", "trainer.max_iter=2"]


def test_export_command_produces_generation_checkpoint(tmp_path: Path) -> None:
    command = build_export_command(
        python=Path("/cosmos/.venv/bin/python"),
        checkpoint=tmp_path / "iter_000000002",
        config=tmp_path / "config.yaml",
        output=tmp_path / "model",
    )
    assert command[:3] == [
        "/cosmos/.venv/bin/python",
        "-m",
        "cosmos_framework.scripts.export_model",
    ]
    assert command[3:7] == ["--cp-size", "1", "--no-use-torch-compile", "--checkpoint-path"]
    assert "--no-vit" in command
    assert command[-2:] == ["-o", str(tmp_path / "model")]


def test_export_config_rewrites_only_context_parallel_degree(tmp_path: Path) -> None:
    source = tmp_path / "training.yaml"
    source.write_text(
        "model:\n  config:\n    parallelism:\n      context_parallel_shard_degree: 2\n"
        "      data_parallel_replicate_degree: 1\n"
    )
    output = write_single_process_export_config(source, tmp_path / "export.yaml")
    assert "context_parallel_shard_degree: 1" in output.read_text()
    assert "data_parallel_replicate_degree: 1" in output.read_text()
    assert "context_parallel_shard_degree: 2" in source.read_text()


def test_training_records_explicit_project_source_revision() -> None:
    provenance = project_provenance("abc123+working-tree", "codex/test")
    assert provenance["commit"] == "abc123+working-tree"
    assert provenance["branch"] == "codex/test"
    assert provenance["i2v_launcher_sha256"]


def test_wrist_only_dataset_mode_rejects_anchor_condition_contract(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "dataset-contract.json").write_text(
        '{"method":"cosmos3_nano_droid_multiview_i2v_sft_dataset"}'
    )
    (root / "train/video_dataset_file.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="does not match condition mode"):
        validate_sft_dataset(root, "wrist_only")


def test_formal_lineage_audit_binds_all_dataset_records(tmp_path: Path) -> None:
    contract = tmp_path / "dataset-contract.json"
    contract.write_text("{}\n")
    contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
    audit = tmp_path / "audit.json"
    audit.write_text(
        """{
          "method": "phiagent_cosmos3_droid_wrist_only_pixel_lineage_audit",
          "status": "WORKING",
          "accepted": true,
          "dataset_contract_sha256": "%s",
          "aggregate": {"minimum_condition_to_wrist_ssim": 0.99},
          "records": [{"accepted": true}, {"accepted": true}]
        }\n"""
        % contract_sha
    )
    result = validate_dataset_lineage_audit(
        audit,
        {"contract_sha256": contract_sha, "total_records": 2},
    )
    assert result["accepted"] is True
    assert result["records"] == 2


def test_lineage_audit_rejects_contract_or_record_mismatch(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(
        """{
          "method": "phiagent_cosmos3_droid_wrist_only_pixel_lineage_audit",
          "status": "WORKING",
          "accepted": true,
          "dataset_contract_sha256": "wrong",
          "records": [{"accepted": true}]
        }\n"""
    )
    with pytest.raises(ValueError, match="does not bind"):
        validate_dataset_lineage_audit(
            audit,
            {"contract_sha256": "expected", "total_records": 1},
        )
