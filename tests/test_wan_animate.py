from __future__ import annotations

import os
from pathlib import Path

import pytest

from phiagent.rendering.wan_animate import (
    GPUInfo,
    PreflightError,
    WanAnimateConfig,
    WanAnimateRenderer,
    _assert_sharded_onnx,
    _select_runtime_profile,
    parse_nvidia_smi_csv,
    select_gpu,
)


def test_parse_and_select_freest_gpu() -> None:
    inventory = (
        "0, NVIDIA A800-SXM4-80GB, 81920, 10000, 71920\n"
        "1, NVIDIA A800, 81920, 50, 81870"
    )
    gpus = parse_nvidia_smi_csv(inventory)
    assert gpus[0] == GPUInfo(0, "NVIDIA A800-SXM4-80GB", 81920, 10000, 71920)
    assert select_gpu(gpus, None, 60 * 1024).physical_index == 1


def test_requested_busy_gpu_fails_loudly() -> None:
    gpus = [GPUInfo(3, "NVIDIA A800", 81920, 40000, 41920)]
    with pytest.raises(PreflightError, match="at least 61440 MiB"):
        select_gpu(gpus, requested_index=3, minimum_free_mib=60 * 1024)


def test_cpu_preflight_environment_does_not_claim_a_gpu(tmp_path: Path) -> None:
    renderer = WanAnimateRenderer(
        WanAnimateConfig(
            wan_repo=tmp_path / "wan",
            checkpoint_dir=tmp_path / "checkpoint",
        )
    )

    environment = renderer._execution_environment(None, seed=7)

    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert environment["PYTHONHASHSEED"] == "7"


@pytest.mark.parametrize(
    ("torch_cuda", "torch_version", "torchvision_version", "torchaudio_version", "flash_version", "profile"),
    (
        ("12.4", "2.6.0+cu124", "0.21.0+cu124", "2.6.0+cu124", "2.7.4.post1", "cuda12.4-torch2.6"),
        ("12.8", "2.7.1+cu128", "0.22.1+cu128", "2.7.1+cu128", "2.8.3", "cuda12.8-torch2.7-blackwell"),
    ),
)
def test_runtime_profiles_accept_pinned_cuda_stacks(
    torch_cuda: str,
    torch_version: str,
    torchvision_version: str,
    torchaudio_version: str,
    flash_version: str,
    profile: str,
) -> None:
    packages = {
        "torch": torch_version,
        "torchvision": torchvision_version,
        "torchaudio": torchaudio_version,
        "diffusers": "0.36.0",
        "transformers": "4.51.3",
        "peft": "0.17.1",
        "moviepy": "2.2.1",
        "librosa": "0.11.0",
        "accelerate": "1.5.2",
        "onnxruntime-gpu": "1.20.2",
        "flash-attn": flash_version,
    }
    assert _select_runtime_profile({"torch_cuda": torch_cuda, "packages": packages}) == profile


def test_runtime_profiles_reject_mixed_cuda_stack() -> None:
    packages = {
        "torch": "2.7.1+cu128",
        "torchvision": "0.22.1+cu128",
        "torchaudio": "2.7.1+cu128",
        "diffusers": "0.36.0",
        "transformers": "4.51.3",
        "peft": "0.17.1",
        "moviepy": "2.2.1",
        "librosa": "0.11.0",
        "accelerate": "1.5.2",
        "onnxruntime-gpu": "1.20.2",
        "flash-attn": "2.7.4.post1",
    }
    with pytest.raises(PreflightError, match="supported runtime profile"):
        _select_runtime_profile({"torch_cuda": "12.8", "packages": packages})


def test_config_rejects_invalid_temporal_length(tmp_path: Path) -> None:
    config = WanAnimateConfig(
        wan_repo=tmp_path / "wan", checkpoint_dir=tmp_path / "checkpoint", frame_num=76
    )
    with pytest.raises(ValueError, match=r"4n \+ 1"):
        config.validate()


def test_config_rejects_invalid_inference_clip_length(tmp_path: Path) -> None:
    config = WanAnimateConfig(
        wan_repo=tmp_path / "wan",
        checkpoint_dir=tmp_path / "checkpoint",
        infer_frames=98,
    )
    with pytest.raises(ValueError, match="infer_frames"):
        config.validate()


def test_commands_preserve_inputs_and_reproducibility(tmp_path: Path) -> None:
    config = WanAnimateConfig(
        wan_repo=tmp_path / "Wan2.2",
        checkpoint_dir=tmp_path / "checkpoint",
        python_executable=Path("/env/bin/python"),
        fps=24,
        frame_num=77,
        infer_frames=100,
        reference_frames=5,
    )
    renderer = WanAnimateRenderer(config)
    preprocess = renderer.build_preprocess_command(
        tmp_path / "human.mp4", tmp_path / "robot.png", tmp_path / "processed"
    )
    generate = renderer.build_generate_command(
        tmp_path / "processed", tmp_path / "generated.mp4", "robot picks up cup", 123
    )
    assert preprocess[preprocess.index("--video_path") + 1].endswith("human.mp4")
    assert preprocess[preprocess.index("--refer_path") + 1].endswith("robot.png")
    assert "--retarget_flag" in preprocess
    assert preprocess[preprocess.index("--fps") + 1] == "24"
    assert generate[generate.index("--base_seed") + 1] == "123"
    assert generate[generate.index("--infer_frames") + 1] == "100"
    assert generate[generate.index("--refert_num") + 1] == "5"
    assert generate[generate.index("--prompt") + 1] == "robot picks up cup"
    assert generate[generate.index("--save_file") + 1].endswith("generated.mp4")


def test_replacement_commands_preserve_scene_and_enable_relighting(tmp_path: Path) -> None:
    python = tmp_path / "venv" / "bin" / "python"
    cudnn = (
        tmp_path
        / "venv"
        / "lib"
        / "python3.10"
        / "site-packages"
        / "nvidia"
        / "cudnn"
        / "lib"
    )
    cudnn.mkdir(parents=True)
    config = WanAnimateConfig(
        wan_repo=tmp_path / "Wan2.2",
        checkpoint_dir=tmp_path / "checkpoint",
        python_executable=python,
        mode="replacement",
        retarget=False,
        t5_cpu=True,
        object_roi=(0.36, 0.72, 0.31, 0.20),
    )
    renderer = WanAnimateRenderer(config)
    preprocess = renderer.build_preprocess_command(
        tmp_path / "human.mp4", tmp_path / "robot.png", tmp_path / "processed"
    )
    generate = renderer.build_generate_command(
        tmp_path / "processed", tmp_path / "generated.mp4", "ignored prompt", 42
    )

    assert "--replace_flag" in preprocess
    assert "--retarget_flag" not in preprocess
    assert "--use_flux" not in preprocess
    assert preprocess[preprocess.index("--iterations") + 1] == "3"
    assert "--replace_flag" in generate
    assert "--use_relighting_lora" in generate
    assert "--t5_cpu" in generate
    environment = renderer._execution_environment(7, 42)
    assert environment["CUDA_VISIBLE_DEVICES"] == "7"
    assert environment["PYTHONHASHSEED"] == "42"
    assert environment.get("PYTHONPATH") == os.environ.get("PYTHONPATH")
    assert environment["LD_LIBRARY_PATH"].split(":")[0] == str(cudnn.resolve())


def test_replacement_rejects_animation_retargeting(tmp_path: Path) -> None:
    config = WanAnimateConfig(
        wan_repo=tmp_path / "Wan2.2",
        checkpoint_dir=tmp_path / "checkpoint",
        mode="replacement",
        object_roi=(0.36, 0.72, 0.31, 0.20),
    )
    with pytest.raises(ValueError, match="does not support pose retargeting"):
        config.validate()


def test_preflight_reports_missing_source(tmp_path: Path) -> None:
    renderer = WanAnimateRenderer(
        WanAnimateConfig(
            wan_repo=tmp_path / "missing", checkpoint_dir=tmp_path / "missing-ckpt"
        )
    )
    with pytest.raises(PreflightError, match="Wan2.2 source"):
        renderer.preflight(select_cuda_device=False)


def test_sharded_pose_onnx_requires_graph_and_external_data(tmp_path: Path) -> None:
    checkpoint = tmp_path / "vitpose_h_wholebody.onnx"
    checkpoint.mkdir()
    (checkpoint / "end2end.onnx").write_bytes(b"graph")
    with pytest.raises(PreflightError, match="external tensor"):
        _assert_sharded_onnx(checkpoint)
    (checkpoint / "backbone.weight").write_bytes(b"tensor")
    _assert_sharded_onnx(checkpoint)
