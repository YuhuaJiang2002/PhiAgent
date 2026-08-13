from pathlib import Path

from scripts.train_agentic_bwm import build_train_command


def test_action_adapter_command_is_action_conditioned_and_frame_explicit(tmp_path: Path) -> None:
    repository = tmp_path / "bwm"
    command = build_train_command(
        repository=repository,
        base_model=tmp_path / "base",
        checkpoint=tmp_path / "bwm.safetensors",
        dataset_root=tmp_path / "dataset",
        metadata=tmp_path / "train.jsonl",
        action_stats=tmp_path / "stats.json",
        output=tmp_path / "output",
        stage="action-adapter",
        seed=7,
        learning_rate=1e-5,
        epochs=1,
        dataset_repeat=1,
        workers=2,
        gradient_accumulation=8,
        physical_gpu_index=7,
        accelerate_config=tmp_path / "accelerate.yaml",
    )

    assert command[command.index("--data_file_keys") + 1] == "video,action"
    assert command[command.index("--action_type") + 1] == "eef_abs"
    assert command[command.index("--action_dim") + 1] == "14"
    assert command[command.index("--trainable_models") + 1] == "action_encoder"
    assert command[command.index("--gpu_ids") + 1] == "7"
    assert command[command.index("--config_file") + 1] == str(tmp_path / "accelerate.yaml")
    assert command[command.index("--dataset_base_path") + 1] == str(tmp_path / "dataset")
    assert command[command.index("--seed") + 1] == "7"


def test_joint_finetune_is_an_explicit_separate_stage(tmp_path: Path) -> None:
    command = build_train_command(
        repository=tmp_path / "bwm",
        base_model=tmp_path / "base",
        checkpoint=tmp_path / "bwm.safetensors",
        dataset_root=tmp_path / "dataset",
        metadata=tmp_path / "train.jsonl",
        action_stats=tmp_path / "stats.json",
        output=tmp_path / "output",
        stage="joint-finetune",
        seed=1,
        learning_rate=2e-6,
        epochs=2,
        dataset_repeat=1,
        workers=1,
        gradient_accumulation=16,
        physical_gpu_index=3,
        accelerate_config=tmp_path / "accelerate.yaml",
    )

    assert command[command.index("--trainable_models") + 1] == "dit,action_encoder"
    assert command[command.index("--gpu_ids") + 1] == "3"
