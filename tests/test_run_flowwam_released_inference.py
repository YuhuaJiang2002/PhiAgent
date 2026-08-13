from pathlib import Path

from scripts.run_flowwam_released_inference import build_flowwam_command


def test_flowwam_command_is_stage1_and_geometry_explicit(tmp_path: Path) -> None:
    command = build_flowwam_command(
        python=tmp_path / "python",
        repository=tmp_path / "flowwam",
        test_dataset_dir=tmp_path / "test",
        robot_only_dir=tmp_path / "robot-only",
        embodiment_root=tmp_path / "embodiments",
        output_dir=tmp_path / "output",
        base_model_root=tmp_path / "base",
        checkpoint=tmp_path / "flowwam.safetensors",
        episode="episode0",
        num_frames=57,
        num_inference_steps=20,
        fps=24,
        width=640,
        height=480,
        seed=7,
        flow_method="raft",
    )

    assert "--robot_only_dir" in command
    assert "--embodiment_dir" in command
    assert command[command.index("--num_output_frames") + 1] == "57"
    assert command[command.index("--flow_method") + 1] == "raft"
    assert command[command.index("--flow_device") + 1] == "cuda:0"
    assert "--disable_refiner" in command
    assert command[command.index("--seed") + 1] == "7"
