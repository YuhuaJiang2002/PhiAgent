from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.convert_cosmos3_checkpoint_to_dcp import (
    build_conversion_command,
    require_executable,
    stage_local_first_checkpoint,
    validate_file_sha256,
    validate_verification,
)


def test_conversion_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    target = tmp_path / "runtime/python3"
    target.parent.mkdir()
    target.write_text("#!/bin/sh\n")
    target.chmod(0o755)
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(target)
    assert require_executable(python, "Python") == python.absolute()
    assert require_executable(python, "Python") != target.resolve()


def _working_report(checkpoint: Path, revision: str) -> dict[str, object]:
    return {
        "status": "WORKING",
        "nano": {
            "status": "WORKING",
            "checkpoint": str(checkpoint.resolve()),
            "revision": revision,
            "indexes": [
                {
                    "actual_total_size_bytes": 31_500_114_912,
                    "expected_total_size_bytes": 31_500_114_912,
                }
            ],
        },
    }


def test_conversion_requires_matching_working_verification(tmp_path: Path) -> None:
    checkpoint = tmp_path / "nano"
    checkpoint.mkdir()
    revision = "411f42a8"
    validate_verification(_working_report(checkpoint, revision), checkpoint, revision)
    bad = _working_report(checkpoint, revision)
    bad["status"] = "PARTIAL"
    with pytest.raises(ValueError, match="not WORKING"):
        validate_verification(bad, checkpoint, revision)


def test_conversion_rejects_different_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "nano"
    checkpoint.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="does not bind"):
        validate_verification(_working_report(other, "rev"), checkpoint, "rev")


def test_conversion_command_uses_official_module(tmp_path: Path) -> None:
    command = build_conversion_command(
        Path("/cosmos/.venv/bin/python"), tmp_path / "nano", tmp_path / "dcp"
    )
    assert command[:3] == [
        "/cosmos/.venv/bin/python",
        "-m",
        "cosmos_framework.scripts.convert_model_to_dcp",
    ]
    assert command[-2:] == ["-o", str(tmp_path / "dcp")]


def test_conversion_binds_required_sound_tokenizer_digest(tmp_path: Path) -> None:
    sound = tmp_path / "sound.safetensors"
    sound.write_bytes(b"verified sound dependency")
    digest = "0aeb4ce4f774003200c2c65fa2418116f35399e1f4015b9719b51d73fea0816f"
    assert validate_file_sha256(sound, digest, "sound") == digest
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_file_sha256(sound, "0" * 64, "sound")


def test_conversion_stages_bundled_text_tokenizer_without_copying_weights(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "nano"
    checkpoint.mkdir()
    config = {
        "model": {
            "config": {
                "vlm_config": {
                    "tokenizer": {
                        "_target": "create_qwen2_tokenizer_with_download",
                        "config_variant": "gcp",
                        "pretrained_model_name": "Qwen/Qwen3-VL-8B-Instruct",
                    }
                },
                "tokenizer": {
                    "_target": "wan2pt2_vae_interface",
                    "bucket_name": "bucket",
                    "object_store_credential_path_pretrained": "credentials/gcp_training.secret",
                    "vae_path": "remote/vae.pth",
                },
            }
        }
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    for name in ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"):
        (checkpoint / name).write_text(name)
    weights = checkpoint / "transformer"
    weights.mkdir()
    vocab_sha = hashlib.sha256((checkpoint / "vocab.json").read_bytes()).hexdigest()
    vae = tmp_path / "Wan2.2_VAE.pth"
    vae.write_bytes(b"verified local vae")
    vae_sha = hashlib.sha256(vae.read_bytes()).hexdigest()
    staging = tmp_path / "experiment/local-first-hf"
    result = stage_local_first_checkpoint(
        checkpoint,
        staging,
        vocab_sha,
        vae,
        vae_sha,
    )
    staged = json.loads((staging / "config.json").read_text())
    tokenizer = staged["model"]["config"]["vlm_config"]["tokenizer"]
    assert tokenizer["config_variant"] == "hf"
    assert tokenizer["pretrained_model_name"] == str(checkpoint.resolve())
    vision_tokenizer = staged["model"]["config"]["tokenizer"]
    assert vision_tokenizer["vae_path"] == str(vae.resolve())
    assert vision_tokenizer["bucket_name"] == ""
    assert vision_tokenizer["object_store_credential_path_pretrained"] == ""
    assert (staging / "transformer").is_symlink()
    assert result["weight_payloads_copied"] is False
    assert result["wan_vae_sha256"] == vae_sha
    assert json.loads((checkpoint / "config.json").read_text()) == config
