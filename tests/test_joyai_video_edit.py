from __future__ import annotations

from pathlib import Path

import pytest

from phiagent.rendering.joyai_video_edit import (
    DEFAULT_FLOWER_WINDOWS,
    JOYAI_LARGE_FILE_CONTRACT,
    JOYAI_MODEL_REVISION,
    JOYAI_MODELSCOPE_MODEL_REVISION,
    JOYAI_REPOSITORY_REVISION,
    JOYAI_SOURCE_REVISION_MARKER,
    JOYAI_TEXT_ENCODER_REVISION,
    JOYAI_TEXT_ENCODER_MODELSCOPE_REVISION,
    JoyAIFlowerEditContract,
    JoyAIPreflightError,
    JoyAIWindow,
    build_server_argv,
    validate_checkpoint_layout,
    validate_upstream_checkout,
)
from scripts.compose_joyai_flower_repairs import (
    center_crop_proposal_to_source,
    isotropic_proposal_to_source,
    temporal_weight,
)
from scripts.prepare_joyai_flower_windows import build_extract_command
from scripts.setup_joyai_video_edit_runtime import write_effective_requirements


def _complete_sparse_checkpoint(tmp_path: Path) -> Path:
    for relative, (size, _) in JOYAI_LARGE_FILE_CONTRACT.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.truncate(size)
    model = tmp_path / "JoyAI-Video-Edit"
    text_encoder = tmp_path / "MiMo-VL-7B-RL-2508"
    (model / "vae/config.json").write_text('{"sample_size": 720}\n')
    (model / ".phiagent-model-revision").write_text(JOYAI_MODEL_REVISION + "\n")
    (text_encoder / ".phiagent-model-revision").write_text(
        JOYAI_TEXT_ENCODER_REVISION + "\n"
    )
    (text_encoder / "model.safetensors.index.json").write_text('{"weight_map": {}}\n')
    return tmp_path


def test_default_flower_windows_are_exact_causal_chunk_sequences() -> None:
    contract = JoyAIFlowerEditContract()
    contract.validate()
    assert [(row.start_frame, row.end_frame, row.frame_count) for row in contract.windows] == [
        (463, 495, 33),
        (543, 575, 33),
    ]
    assert all((window.frame_count - 1) % 8 == 0 for window in DEFAULT_FLOWER_WINDOWS)
    manifest = contract.to_manifest()
    assert manifest["model"]["weights_revision"] == JOYAI_MODEL_REVISION
    assert manifest["model_authority"] == "proposal_only"
    assert manifest["physical_evidence"] is False


def test_window_rejects_length_not_aligned_to_eight_frame_causal_chunks() -> None:
    with pytest.raises(ValueError, match=r"1 \+ 8n"):
        JoyAIWindow(start_frame=10, end_frame=41, seam_frame=25).validate()


def test_integer_crop_command_never_rescales_source_geometry(tmp_path: Path) -> None:
    contract = JoyAIFlowerEditContract()
    command = build_extract_command(
        ffmpeg=Path("/usr/bin/ffmpeg"),
        video=tmp_path / "candidate.mkv",
        output=tmp_path / "window.mkv",
        window=DEFAULT_FLOWER_WINDOWS[0],
        contract=contract,
    )
    filtergraph = command[command.index("-vf") + 1]
    assert "crop=1248:720:16:0" in filtergraph
    assert "scale=" not in filtergraph
    assert "between(n\\,463\\,495)" in filtergraph


def test_low_resolution_transform_is_explicit_isotropic_resize_then_crop(tmp_path: Path) -> None:
    contract = JoyAIFlowerEditContract(
        source_width=624,
        source_height=352,
        crop_left=14,
        transform_kind="isotropic_fit_height_then_center_crop",
        resized_width=1276,
    )
    contract.validate()
    command = build_extract_command(
        ffmpeg=Path("/usr/bin/ffmpeg"),
        video=tmp_path / "candidate.mp4",
        output=tmp_path / "window.mkv",
        window=DEFAULT_FLOWER_WINDOWS[0],
        contract=contract,
    )
    filtergraph = command[command.index("-vf") + 1]
    assert "scale=1276:720:flags=lanczos,crop=1248:720:14:0" in filtergraph
    transform = contract.to_manifest()["coordinate_frames"]["source_to_joyai"]
    assert transform["kind"] == "isotropic_fit_height_then_center_crop"
    assert transform["resized_width_px"] == 1276


def test_temporal_blend_is_endpoint_exact_and_full_strength_interior() -> None:
    assert temporal_weight(462, 463, 495, 4) == 0.0
    assert temporal_weight(463, 463, 495, 4) == 0.0
    assert temporal_weight(464, 463, 495, 4) == 0.25
    assert temporal_weight(467, 463, 495, 4) == 1.0
    assert temporal_weight(491, 463, 495, 4) == 1.0
    assert temporal_weight(494, 463, 495, 4) == 0.25
    assert temporal_weight(495, 463, 495, 4) == 0.0


def test_proposal_to_native_transform_is_named_and_aspect_preserving() -> None:
    transform = isotropic_proposal_to_source(
        source_width=832,
        source_height=480,
        proposal_width=1248,
        proposal_height=720,
    )

    assert transform == {
        "kind": "isotropic_rational_scale",
        "x_source": "x_joyai * (832/1248)",
        "y_source": "y_joyai * (480/720)",
        "interpolation": "Lanczos4",
    }
    with pytest.raises(ValueError, match="equal aspect ratios"):
        isotropic_proposal_to_source(
            source_width=832,
            source_height=480,
            proposal_width=1280,
            proposal_height=720,
        )


def test_native_proposal_is_inverse_center_crop_without_rescaling() -> None:
    transform = center_crop_proposal_to_source(
        source_width=1280,
        source_height=720,
        proposal_width=1248,
        proposal_height=720,
        crop_left=16,
        crop_top=0,
    )

    assert transform == {
        "kind": "center_crop_inverse_no_rescale",
        "x_source": "x_joyai + 16",
        "y_source": "y_joyai + 0",
        "crop_left_px": 16,
        "crop_top_px": 0,
        "interpolation": "none",
    }
    with pytest.raises(ValueError, match="exceeds source width"):
        center_crop_proposal_to_source(
            source_width=1280,
            source_height=720,
            proposal_width=1248,
            proposal_height=720,
            crop_left=33,
            crop_top=0,
        )


def test_a800_bf16_runtime_excludes_only_unused_fp8_build_packages(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requirements.txt"
    effective = tmp_path / "requirements-effective.txt"
    source.write_text(
        "--extra-index-url https://download.pytorch.org/whl/cu128\n"
        "torch==2.9.1+cu128\n"
        "flash-attn-4==4.0.0b13\n"
        "nvidia-cutlass-dsl==4.5.1\n"
        "nvidia-cutlass-dsl-libs-base==4.5.1\n"
        "cuda-python==13.3.1\n"
        "cuda-bindings==13.3.1\n"
        "cuda-core==1.0.1\n"
        "cuda-pathfinder==1.5.6\n",
        encoding="utf-8",
    )

    record = write_effective_requirements(
        source, effective, include_fp8_build_deps=False
    )

    text = effective.read_text(encoding="utf-8")
    assert "torch==2.9.1+cu128" in text
    assert "flash-attn-4==4.0.0b13" in text
    assert "cutlass" not in text
    assert "cuda-python" not in text
    assert len(record["removed_requirements"]) == 6


def test_checkpoint_preflight_rejects_lfs_pointer_or_missing_layout(tmp_path: Path) -> None:
    (tmp_path / "JoyAI-Video-Edit/dit").mkdir(parents=True)
    (tmp_path / "JoyAI-Video-Edit/dit/joyai_video_edit_dit_0804.pth").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
    )
    with pytest.raises(JoyAIPreflightError, match="incomplete"):
        validate_checkpoint_layout(tmp_path)


def test_source_archive_requires_exact_revision_marker(tmp_path: Path) -> None:
    repository = tmp_path / "JoyAI-Video-Edit"
    server = repository / "deploy/xvideo/serving/serve_joyomni_streaming.py"
    server.parent.mkdir(parents=True)
    server.write_text("# pinned server\n")
    (repository / "LICENSE").write_text("Apache License\nVersion 2.0\n")
    marker = repository / JOYAI_SOURCE_REVISION_MARKER
    marker.write_text(JOYAI_REPOSITORY_REVISION + "\n")

    observed = validate_upstream_checkout(repository)

    assert observed["revision"] == JOYAI_REPOSITORY_REVISION
    assert observed["source_kind"] == "revision-marked-archive"
    marker.write_text("wrong\n")
    with pytest.raises(JoyAIPreflightError, match="revision mismatch"):
        validate_upstream_checkout(repository)


def test_checkpoint_preflight_requires_pinned_markers_and_exact_sizes(
    tmp_path: Path,
) -> None:
    root = _complete_sparse_checkpoint(tmp_path)

    observed = validate_checkpoint_layout(root)

    assert observed["model_revision"] == JOYAI_MODEL_REVISION
    assert observed["text_encoder_revision"] == JOYAI_TEXT_ENCODER_REVISION
    assert observed["large_file_hashes"] == {}
    (root / "JoyAI-Video-Edit/.phiagent-model-revision").write_text(
        f"modelscope:{JOYAI_MODELSCOPE_MODEL_REVISION}\n"
    )
    marker = root / "MiMo-VL-7B-RL-2508/.phiagent-model-revision"
    marker.write_text(f"modelscope:{JOYAI_TEXT_ENCODER_MODELSCOPE_REVISION}\n")
    mirrored = validate_checkpoint_layout(root)
    assert mirrored["observed_revision_markers"]["model"].startswith("modelscope:")
    marker.write_text("wrong\n")
    with pytest.raises(JoyAIPreflightError, match="revision markers"):
        validate_checkpoint_layout(root)


def test_server_command_splits_primary_and_vae_devices_and_extends_a800_timeout(
    tmp_path: Path,
) -> None:
    base_python = tmp_path / "base/python3.10"
    base_python.parent.mkdir()
    base_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)
    argv = build_server_argv(
        python_executable=venv_python,
        repository=tmp_path / "JoyAI",
        checkpoint_root=tmp_path / "checkpoints",
        record_dir=tmp_path / "records",
        host="127.0.0.1",
        port=18080,
    )
    assert argv[0] == str(venv_python)
    assert argv[0] != str(base_python)
    assert argv[argv.index("--device") + 1] == "cuda:0"
    assert argv[argv.index("--vae-encode-device") + 1] == "cuda:1"
    assert argv[argv.index("--holder-idle-timeout-s") + 1] == "1800"
    assert "--no-use-pe" in argv
    assert "--no-online-gate" in argv
