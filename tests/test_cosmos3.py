from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from phiagent.rendering.alignment import parse_ssim_stats
from phiagent.rendering.cosmos3 import Cosmos3Config, Cosmos3TrajectoryRenderer
from test_trajectory_rendering import _request


def test_cosmos_config_rejects_invalid_resolution(tmp_path: Path) -> None:
    config = Cosmos3Config(
        framework_repo=tmp_path / "cosmos-framework",
        checkpoint_dir=tmp_path / "checkpoint",
        resolution=512,
    )
    with pytest.raises(ValueError, match="256, 480, or 720"):
        config.validate()


def test_cosmos_config_resolves_pinned_wan_vae_cache(tmp_path: Path) -> None:
    config = Cosmos3Config(
        framework_repo=tmp_path / "cosmos-framework",
        checkpoint_dir=tmp_path / "checkpoint",
        hf_home=tmp_path / "hf",
    )
    assert config.wan_vae_path == (
        tmp_path
        / "hf"
        / "hub"
        / "models--Wan-AI--Wan2.2-TI2V-5B"
        / "snapshots"
        / "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
        / "Wan2.2_VAE.pth"
    )


def test_cosmos_config_accepts_explicit_verified_wan_vae(tmp_path: Path) -> None:
    override = tmp_path / "Wan2.2_VAE.pth"
    config = Cosmos3Config(
        framework_repo=tmp_path / "cosmos-framework",
        checkpoint_dir=tmp_path / "checkpoint",
        wan_vae_override=override,
    )
    assert config.wan_vae_path == override.resolve()


def test_build_spec_binds_verified_control_and_camera(tmp_path: Path) -> None:
    request = _request(tmp_path)
    timestamps = tuple(index / 30 for index in range(5))
    request = replace(
        request,
        robot_trajectory=replace(request.robot_trajectory, timestamps_s=timestamps,
                                 joint_positions_rad=((0.0,),) * 5),
        object_trajectories=(
            replace(
                request.object_trajectories[0],
                timestamps_s=timestamps,
                poses=(request.object_trajectories[0].poses[0],) * 5,
            ),
        ),
    )
    renderer = Cosmos3TrajectoryRenderer(
        Cosmos3Config(
            framework_repo=tmp_path / "cosmos-framework",
            checkpoint_dir=tmp_path / "checkpoint",
            fps=30,
        )
    )
    spec = renderer.build_spec(
        request,
        tmp_path / "control.mp4",
        tmp_path / "prompt.json",
    )
    assert spec["model_mode"] == "video2video"
    assert spec["num_frames"] == 5
    assert spec["aspect_ratio"] == "4,3"
    assert spec["vision_path"].endswith("control.mp4")
    assert spec["edge"]["weight"] == 1.0
    assert spec["emphasize_control_in_prompt"] is True


def test_build_spec_rejects_unsupported_trajectory_length(tmp_path: Path) -> None:
    renderer = Cosmos3TrajectoryRenderer(
        Cosmos3Config(
            framework_repo=tmp_path / "cosmos-framework",
            checkpoint_dir=tmp_path / "checkpoint",
        )
    )
    with pytest.raises(RuntimeError, match="between 5 and 300"):
        renderer.build_spec(
            _request(tmp_path),
            tmp_path / "control.mp4",
            tmp_path / "prompt.json",
        )


def test_build_spec_rejects_unsupported_aspect_ratio(tmp_path: Path) -> None:
    request = _request(tmp_path)
    timestamps = tuple(index / 30 for index in range(5))
    request = replace(
        request,
        camera_intrinsics=replace(request.camera_intrinsics, width=832, height=480),
        robot_trajectory=replace(
            request.robot_trajectory,
            timestamps_s=timestamps,
            joint_positions_rad=((0.0,),) * 5,
        ),
        object_trajectories=(
            replace(
                request.object_trajectories[0],
                timestamps_s=timestamps,
                poses=(request.object_trajectories[0].poses[0],) * 5,
            ),
        ),
    )
    renderer = Cosmos3TrajectoryRenderer(
        Cosmos3Config(
            framework_repo=tmp_path / "cosmos-framework",
            checkpoint_dir=tmp_path / "checkpoint",
        )
    )
    with pytest.raises(RuntimeError, match="does not support aspect ratio 26,15"):
        renderer.build_spec(
            request,
            tmp_path / "control.mp4",
            tmp_path / "prompt.json",
        )


def test_command_uses_official_framework_entrypoint(tmp_path: Path) -> None:
    renderer = Cosmos3TrajectoryRenderer(
        Cosmos3Config(
            framework_repo=tmp_path / "cosmos-framework",
            checkpoint_dir=tmp_path / "checkpoint",
            python_executable=Path("/cosmos/.venv/bin/python"),
            guardrails=False,
        )
    )
    command = renderer.build_command(
        tmp_path / "spec.json",
        tmp_path / "outputs",
        seed=17,
    )
    assert command[:3] == [
        "/cosmos/.venv/bin/python",
        "-m",
        "cosmos_framework.scripts.inference",
    ]
    assert command[command.index("--checkpoint-path") + 1].endswith("checkpoint")
    assert command[command.index("--seed") + 1] == "17"
    assert "--no-use-torch-compile" in command
    assert "--no-guardrails" in command


def test_vision_only_model_config_disables_unused_outputs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"model":{"config":{"vision_gen":true,"sound_gen":true,"action_gen":true,'
        '"tokenizer":{"vae_path":"pretrained/Wan2.2_VAE.pth"},'
        '"vlm_config":{"tokenizer":{"config_variant":"gcp",'
        '"pretrained_model_name":"Qwen/Qwen3-VL-8B-Instruct"}}}}}'
    )
    renderer = Cosmos3TrajectoryRenderer(
        Cosmos3Config(
            framework_repo=tmp_path / "cosmos-framework",
            checkpoint_dir=checkpoint,
        )
    )
    payload = renderer.build_model_config()
    assert payload["model"]["config"]["vision_gen"] is True
    assert payload["model"]["config"]["sound_gen"] is False
    assert payload["model"]["config"]["action_gen"] is False
    assert payload["model"]["config"]["tokenizer"]["vae_path"].endswith(
        "Wan2.2_VAE.pth"
    )
    tokenizer = payload["model"]["config"]["vlm_config"]["tokenizer"]
    assert tokenizer["config_variant"] == "hf"
    assert tokenizer["pretrained_model_name"] == str(checkpoint)


def test_parse_per_frame_edge_ssim_stats() -> None:
    values = parse_ssim_stats(
        "n:1 Y:0.8 All:0.8 (6.9)\n"
        "n:2 Y:0.6 All:0.6 (4.0)\n"
    )
    assert values == (0.8, 0.6)
