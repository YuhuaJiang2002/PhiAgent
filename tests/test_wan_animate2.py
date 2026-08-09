from __future__ import annotations

from pathlib import Path

import pytest

from phiagent.rendering.wan_animate import GPUInfo
from phiagent.rendering.wan_animate2 import (
    WAN_ANIMATE2_MODEL_REVISION,
    WAN_ANIMATE2_MODELSCOPE_REVISION,
    select_wan_animate2_gpus,
    verify_wan_animate2_checkpoint,
    write_runtime_config,
)


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    files = (
        "videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth",
        "videomodel/Wan-AI/umt5-xxl/tokenizer.json",
        "videomodel/Wan-AI/vae.pth",
        "videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "videomodel/Wan-AI/xlm-roberta-large/tokenizer.json",
        "wan_animate_2/wan_animate_2_bf16.safetensors",
        "wan_animate_2/wan_animate_2_bf16_distillation.safetensors",
    )
    for relative in files:
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    (checkpoint / ".phiagent-model-revision").write_text(
        WAN_ANIMATE2_MODEL_REVISION + "\n"
    )
    return checkpoint


def test_checkpoint_requires_pinned_revision(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)

    hashes = verify_wan_animate2_checkpoint(checkpoint)

    assert "wan_animate_2/wan_animate_2_bf16.safetensors" in hashes
    (checkpoint / ".phiagent-model-revision").write_text("wrong\n")
    with pytest.raises(ValueError, match="revision marker"):
        verify_wan_animate2_checkpoint(checkpoint)
    (checkpoint / ".phiagent-model-revision").write_text(
        f"modelscope:{WAN_ANIMATE2_MODELSCOPE_REVISION}\n"
    )
    assert verify_wan_animate2_checkpoint(checkpoint)
    assert verify_wan_animate2_checkpoint(checkpoint, distilled=True)


def test_runtime_config_uses_only_absolute_checkpoint_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    template = Path("external/Wan-Animate-2/infer/wan_animate_2.yaml").read_text()
    config = repo / "infer" / "wan_animate_2.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(template)
    checkpoint = _checkpoint(tmp_path)
    output = tmp_path / "experiment" / "config" / "wan_animate_2.yaml"

    write_runtime_config(repo, checkpoint, output)

    rendered = output.read_text()
    assert "../ckpts" not in rendered
    assert str(checkpoint.resolve()) in rendered

    distilled_output = tmp_path / "distilled" / "config.yaml"
    distilled_template = (
        Path("external/Wan-Animate-2/infer/wan_animate_2_distillation.yaml").read_text()
    )
    (repo / "infer" / "wan_animate_2_distillation.yaml").write_text(distilled_template)
    write_runtime_config(repo, checkpoint, distilled_output, distilled=True)
    assert "wan_animate_2_bf16_distillation.safetensors" in distilled_output.read_text()


def test_select_two_free_physical_gpus() -> None:
    gpus = (
        GPUInfo(0, "A800", 81920, 1000, 80920),
        GPUInfo(1, "A800", 81920, 2000, 79920),
        GPUInfo(2, "A800", 81920, 40000, 41920),
    )

    selected = select_wan_animate2_gpus(gpus, (), 60000)

    assert tuple(gpu.physical_index for gpu in selected) == (0, 1)
    with pytest.raises(ValueError, match="exactly two distinct"):
        select_wan_animate2_gpus(gpus, (0, 0), 60000)
