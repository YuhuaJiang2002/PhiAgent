from __future__ import annotations

from pathlib import Path
import json

import pytest

from phiagent.data.adaptation import (
    AdaptationArm,
    AdaptationAsset,
    AdaptationAssetKind,
    AdaptationManifest,
    AdaptationSplit,
    VaceTrainingExample,
    file_sha256,
)
from phiagent.training.diffsynth_vace import (
    VACE_MODEL_REVISION,
    build_vace_training_command,
    write_vace_metadata,
)


def test_vace_training_entrypoint_persists_pinned_provenance() -> None:
    source = Path("scripts/train_sharpa_vace_lora.py").read_text()

    assert '"diffsynth_commit": diffsynth_commit' in source
    assert '"checkpoint_files": [' in source
    assert '"manifest_sha256": _sha256' in source
    assert 'training_env["PYTHONPATH"]' in source
    assert '"training_pythonpath_prefix"' in source


def _asset(path: Path, asset_id: str, kind: AdaptationAssetKind) -> AdaptationAsset:
    return AdaptationAsset(
        asset_id,
        str(path),
        AdaptationSplit.TRAIN,
        kind,
        f"local://{asset_id}",
        "project-generated development pseudo-target",
        file_sha256(path),
        path.stat().st_size,
        True,
    )


def _manifest(tmp_path: Path) -> AdaptationManifest:
    files = {}
    for asset_id, suffix in (("target", ".mp4"), ("control", ".mp4"), ("reference", ".png")):
        path = tmp_path / f"{asset_id}{suffix}"
        path.write_bytes(asset_id.encode())
        files[asset_id] = path
    return AdaptationManifest(
        experiment_id="vace-smoke",
        arm=AdaptationArm.VACE_LORA,
        assets=(
            _asset(files["target"], "target", AdaptationAssetKind.TARGET_VIDEO),
            _asset(files["control"], "control", AdaptationAssetKind.VACE_CONTROL_VIDEO),
            _asset(files["reference"], "reference", AdaptationAssetKind.VACE_REFERENCE_IMAGE),
        ),
        vace_examples=(VaceTrainingExample("example", "target", "control", "reference", "Sharpa"),),
    )


def test_vace_metadata_uses_reviewed_columns(tmp_path: Path) -> None:
    metadata = tmp_path / "dataset" / "metadata.csv"
    write_vace_metadata(_manifest(tmp_path), metadata)
    text = metadata.read_text()
    assert text.splitlines()[0] == "video,prompt,vace_video,vace_reference_image"
    assert "Sharpa" in text


def test_vace_command_is_single_gpu_and_region_controlled(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    for relative in (
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "google/umt5-xxl/tokenizer.json",
    ):
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (checkpoint / ".phiagent-model-revision").write_text(VACE_MODEL_REVISION)
    monkeypatch.setattr(
        "phiagent.training.diffsynth_vace.verify_diffsynth_checkout",
        lambda repo: "pinned",
    )
    command = build_vace_training_command(
        tmp_path / "accelerate",
        tmp_path / "DiffSynth",
        tmp_path / "metadata.csv",
        checkpoint,
        tmp_path / "output",
        rank=4,
        learning_rate=1e-4,
        epochs=1,
        dataset_repeat=1,
        height=256,
        width=448,
        num_frames=17,
    )
    assert command[command.index("--num_processes") + 1] == "1"
    assert command[command.index("--lora_base_model") + 1] == "vace"
    assert command[command.index("--extra_inputs") + 1] == "vace_video,vace_reference_image"
    assert len(json.loads(command[command.index("--model_paths") + 1])) == 3


def test_vace_command_rejects_invalid_temporal_length(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"4n\+1"):
        build_vace_training_command(
            tmp_path / "accelerate",
            tmp_path / "repo",
            tmp_path / "metadata.csv",
            tmp_path / "checkpoint",
            tmp_path / "output",
            rank=4,
            learning_rate=1e-4,
            epochs=1,
            dataset_repeat=1,
            height=256,
            width=448,
            num_frames=18,
        )
