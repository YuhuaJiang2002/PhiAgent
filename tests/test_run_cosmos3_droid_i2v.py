from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_cosmos3_droid_i2v import (
    build_i2v_spec,
    build_inference_command,
    project_provenance,
    require_executable,
    validate_generation_shape,
    validate_gpu_selection,
)


def test_inference_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    target = tmp_path / "runtime/python3"
    target.parent.mkdir()
    target.write_text("#!/bin/sh\n")
    target.chmod(0o755)
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(target)
    assert require_executable(python, "Python") == python.absolute()
    assert require_executable(python, "Python") != target.resolve()


def _inventory() -> list[dict[str, object]]:
    return [
        {
            "physical_index": index,
            "uuid": f"GPU-{index}",
            "name": "A800",
            "memory_total_mib": 81920,
            "memory_used_mib": 10000,
            "memory_free_mib": 71920,
            "utilization_gpu_percent": 0,
        }
        for index in range(4)
    ]


def test_generation_shape_enforces_cosmos_cadence_and_resolution_limit() -> None:
    validate_generation_shape(480, 93)
    with pytest.raises(ValueError, match="4n\\+1"):
        validate_generation_shape(480, 92)
    with pytest.raises(ValueError, match="maximum"):
        validate_generation_shape(480, 301)


def test_gpu_selection_records_physical_devices_and_free_memory() -> None:
    selected = validate_gpu_selection(_inventory(), [1, 3], 60_000)
    assert [row["physical_index"] for row in selected] == [1, 3]
    with pytest.raises(RuntimeError, match="below"):
        validate_gpu_selection(_inventory(), [0], 75_000)
    with pytest.raises(ValueError, match="unique"):
        validate_gpu_selection(_inventory(), [0, 0], 1)


def test_i2v_spec_discloses_only_first_frame_condition(tmp_path: Path) -> None:
    spec = build_i2v_spec(
        sample_id="ep012-cosmos3-base",
        prompt="A synchronized DROID robot closes a drawer.",
        condition_image=tmp_path / "real-condition.png",
        resolution=480,
        aspect_ratio="16,9",
        num_frames=93,
        fps=16,
        num_steps=35,
        guidance=6.0,
        shift=10.0,
        seed=17,
    )
    assert spec["model_mode"] == "image2video"
    assert spec["vision_path"].endswith("real-condition.png")
    assert spec["num_frames"] == 93
    assert spec["enable_sound"] is False
    assert "target" not in spec


def test_single_gpu_command_uses_official_entrypoint(tmp_path: Path) -> None:
    command = build_inference_command(
        python=Path("/cosmos/.venv/bin/python"),
        framework_repo=tmp_path,
        spec_path=tmp_path / "i2v.json",
        model_config_path=tmp_path / "model.json",
        checkpoint=tmp_path / "checkpoint",
        output_dir=tmp_path / "outputs",
        seed=11,
        gpu_count=1,
        master_port=29631,
        offload_guardrails=True,
        no_guardrails=False,
    )
    assert command[:3] == [
        "/cosmos/.venv/bin/python",
        "-m",
        "cosmos_framework.scripts.inference",
    ]
    assert "--parallelism-preset=latency" in command
    assert "--offload-guardrail-models" in command


def test_multi_gpu_command_uses_torchrun_and_no_guardrails(tmp_path: Path) -> None:
    python = tmp_path / ".venv/bin/python"
    torchrun = python.parent / "torchrun"
    torchrun.parent.mkdir(parents=True)
    torchrun.write_text("#!/bin/sh\n")
    command = build_inference_command(
        python=python,
        framework_repo=tmp_path,
        spec_path=tmp_path / "i2v.json",
        model_config_path=tmp_path / "model.json",
        checkpoint=tmp_path / "checkpoint",
        output_dir=tmp_path / "outputs",
        seed=11,
        gpu_count=4,
        master_port=29631,
        offload_guardrails=False,
        no_guardrails=True,
    )
    assert command[0] == str(torchrun)
    assert "--nproc-per-node=4" in command
    assert "--parallelism-preset=throughput" in command
    assert "--no-guardrails" in command


def test_explicit_project_source_revision_is_recorded() -> None:
    provenance = project_provenance("abc123+working-tree", "codex/test")
    assert provenance["commit"] == "abc123+working-tree"
    assert provenance["branch"] == "codex/test"
    assert provenance["launcher_sha256"]
