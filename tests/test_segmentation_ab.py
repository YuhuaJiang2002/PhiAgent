from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from phiagent.evaluation.segmentation_ab import (
    MaskGeometryThresholds,
    compare_tracker_results,
    effective_component_area_threshold,
    save_packed_masks,
    score_attachment_distance,
    score_centroid_continuity,
    score_mask_geometry,
    validate_repository_revision,
    validate_result_mask_artifact,
    validate_sam31_config,
    validate_task_config,
)
from phiagent.evaluation.video_proxy import file_sha256
from phiagent.rendering.wan_animate import GPUInfo, PreflightError
from scripts.evaluate_joyai_tshirt_segmentation_ab import (
    _launch_workers,
    _prepare_inputs,
    _require_python,
    build_worker_command,
    classify_worker_outcome,
    hold_parallel_gpu_leases,
    select_parallel_gpus,
)
from scripts.evaluate_joyai_tshirt_segmentation_worker import (
    _validate_prepared_input,
)


def _thresholds() -> MaskGeometryThresholds:
    return MaskGeometryThresholds(
        baseline_hold_frames=1,
        minimum_area_pixels=2,
        minimum_component_area_pixels=1,
        maximum_connected_components=2,
        maximum_major_axis_cv=0.08,
        maximum_major_axis_relative_deviation=0.15,
        maximum_terminal_major_axis_relative_deviation=0.12,
        maximum_area_cv=0.12,
        maximum_area_relative_deviation=0.2,
        maximum_terminal_area_relative_deviation=0.18,
        minimum_component_area_fraction=0.005,
        maximum_centroid_step_pixels=6.0,
    )


def _task_config() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "coordinate_frame": "camera:test",
        "frame_count": 2,
        "frame_size": [4, 3],
        "fps": 24.0,
        "initial_frame_index": 0,
        "ab_policy": {
            "authoritative_model": "sam2",
            "shadow_model": "sam3.1",
            "decision_mode": "sam2_authoritative_sam31_shadow",
            "reuse_incumbent_thresholds_for_shadow": False,
        },
        "sam2": {},
        "objects": {
            "left_sleeve": {
                "object_id": 1,
                "initial_polygon_xy": [[0, 0], [1, 0], [0, 1]],
            },
            "right_sleeve": {
                "object_id": 2,
                "initial_polygon_xy": [[2, 0], [3, 0], [3, 1]],
            },
        },
        "thresholds": {
            "baseline_hold_frames": 1,
            "minimum_area_pixels": 2,
            "minimum_component_area_pixels": 1,
            "maximum_connected_components": 2,
            "maximum_major_axis_cv": 0.08,
            "maximum_major_axis_relative_deviation": 0.15,
            "maximum_terminal_major_axis_relative_deviation": 0.12,
            "maximum_area_cv": 0.12,
            "maximum_area_relative_deviation": 0.2,
            "maximum_terminal_area_relative_deviation": 0.18,
        },
        "initial_mask_refinement": {
            "maximum_luma": 115,
            "closing_kernel_pixels": 3,
            "keep_largest_component": True,
            "minimum_area_pixels": 1,
        },
    }


def _sam31_config() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "model_id": "sam3.1",
        "role": "shadow",
        "repository_commit": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "huggingface_revision": "c" * 64,
        "multiplex_count": 16,
        "thresholds": None,
    }


def test_mask_geometry_and_continuity_keep_frozen_behavior() -> None:
    score = score_mask_geometry(
        [10, 10, 9],
        [5.0, 5.0, 4.8],
        [1, 1, 1],
        thresholds=_thresholds(),
    )
    continuity = score_centroid_continuity(
        [(0.0, 0.0), (3.0, 4.0), (10.0, 10.0)],
        maximum_step_pixels=6.0,
    )

    assert score["passed"] is True
    assert continuity["passed"] is False
    assert continuity["maximum_step_frame"] == 2
    assert effective_component_area_threshold(1_000, _thresholds()) == 5


def test_mask_geometry_rejects_tracks_without_meaningful_components() -> None:
    score = score_mask_geometry(
        [900, 900, 900],
        [30.0, 30.0, 30.0],
        [0, 0, 0],
        thresholds=_thresholds(),
    )

    assert score["passed"] is False
    assert score["gate_results"]["mask_persistent"] is True
    assert score["gate_results"]["mask_connected"] is False


def test_attachment_distance_rejects_separation() -> None:
    score = score_attachment_distance(
        [2.0, 3.0, 8.0],
        baseline_hold_frames=2,
        maximum_distance_increase_pixels=4.0,
    )

    assert score["passed"] is False


def test_task_config_requires_non_decision_bearing_shadow() -> None:
    config = _task_config()
    validate_task_config(config)

    config["ab_policy"]["reuse_incumbent_thresholds_for_shadow"] = True

    with pytest.raises(ValueError, match="must not reuse"):
        validate_task_config(config)


def test_sam31_shadow_rejects_unregistered_thresholds() -> None:
    config = _sam31_config()
    validate_sam31_config(config)

    config["thresholds"] = {"maximum_area_cv": 1.0}

    with pytest.raises(ValueError, match="must not contain decision thresholds"):
        validate_sam31_config(config)


def test_committed_ab_configs_preserve_shadow_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    task = json.loads(
        (
            root / "configs" / "physical_video" / "tshirt_left_stage_segmentation_ab_v1.json"
        ).read_text(encoding="utf-8")
    )
    sam31 = json.loads(
        (root / "configs" / "physical_video" / "sam31_multiplex_shadow_v1.json").read_text(
            encoding="utf-8"
        )
    )

    validate_task_config(task)
    validate_sam31_config(sam31)
    assert task["sam2"]["role"] == "authoritative"
    assert sam31["role"] == "shadow"
    assert sam31["thresholds"] is None


def test_parallel_gpu_selection_is_distinct_and_respects_memory() -> None:
    gpus = (
        GPUInfo(0, "GPU-0", 80_000, 10_000, 70_000),
        GPUInfo(1, "GPU-1", 80_000, 20_000, 60_000),
        GPUInfo(2, "GPU-2", 24_000, 20_000, 4_000),
    )

    sam2, sam31 = select_parallel_gpus(
        gpus,
        sam2_requested=None,
        sam31_requested=None,
        sam2_minimum_free_mib=12_000,
        sam31_minimum_free_mib=32_000,
    )

    assert sam31.physical_index == 0
    assert sam2.physical_index == 1

    with pytest.raises(ValueError, match="distinct"):
        select_parallel_gpus(
            gpus,
            sam2_requested=0,
            sam31_requested=0,
            sam2_minimum_free_mib=12_000,
            sam31_minimum_free_mib=32_000,
        )
    with pytest.raises(PreflightError):
        select_parallel_gpus(
            gpus[:1],
            sam2_requested=None,
            sam31_requested=None,
            sam2_minimum_free_mib=12_000,
            sam31_minimum_free_mib=32_000,
        )


def test_shadow_failure_does_not_override_authoritative_outcome() -> None:
    assert classify_worker_outcome({"sam2": 0, "sam3.1": 7}) == "shadow_failed"
    assert classify_worker_outcome({"sam2": 7, "sam3.1": 0}) == "authoritative_failed"
    assert classify_worker_outcome({"sam2": 0, "sam3.1": 0}) == "complete"


def test_parallel_gpu_leases_are_sorted_and_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class Lease:
        def __init__(self, physical_index: int) -> None:
            self.physical_index = physical_index

        def close(self) -> None:
            events.append(("close", self.physical_index))

    def acquire(physical_index: int) -> tuple[Path, Lease]:
        events.append(("acquire", physical_index))
        return Path(f"/tmp/gpu-{physical_index}.lock"), Lease(physical_index)

    monkeypatch.setattr(
        "scripts.evaluate_joyai_tshirt_segmentation_ab.acquire_gpu_lease",
        acquire,
    )

    with hold_parallel_gpu_leases((7, 2)) as paths:
        assert list(paths) == [2, 7]
        assert events == [("acquire", 2), ("acquire", 7)]

    assert events[-2:] == [("close", 7), ("close", 2)]


def test_shadow_launch_failure_does_not_terminate_sam2(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    return_codes, launch_errors, _ = _launch_workers(
        {
            "sam2": [
                sys.executable,
                "-c",
                "import time; time.sleep(0.2); print('authoritative-ok')",
            ],
            "sam3.1": [str(tmp_path / "missing-python")],
        },
        logs,
    )

    assert return_codes["sam2"] == 0
    assert return_codes["sam3.1"] == 127
    assert "sam3.1" in launch_errors
    assert "authoritative-ok" in (logs / "sam2.stdout.log").read_text(encoding="utf-8")


def test_python_validation_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    base_python = tmp_path / "base" / "python"
    base_python.parent.mkdir()
    base_python.write_text("#!/bin/sh\n", encoding="utf-8")
    base_python.chmod(0o755)
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)

    selected = _require_python(venv_python, "test Python")

    assert selected == venv_python.absolute()
    assert selected.is_symlink()


def test_shared_input_binds_tracking_and_lossless_scoring_frames(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        24.0,
        (16, 16),
    )
    assert writer.isOpened()
    for _ in range(2):
        frame = np.full((16, 16, 3), 255, dtype=np.uint8)
        frame[2:7, 1:6] = 0
        frame[2:7, 10:15] = 0
        writer.write(frame)
    writer.release()
    config = _task_config()
    config["frame_size"] = [16, 16]
    config["objects"] = {
        "left_sleeve": {
            "object_id": 1,
            "initial_polygon_xy": [[0, 1], [7, 1], [7, 8], [0, 8]],
        },
        "right_sleeve": {
            "object_id": 2,
            "initial_polygon_xy": [[9, 1], [15, 1], [15, 8], [9, 8]],
        },
    }
    config_path = tmp_path / "task.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    prepared_dir = tmp_path / "prepared"

    manifest_path = _prepare_inputs(
        video=video,
        task_config_path=config_path,
        task_config=config,
        prepared_dir=prepared_dir,
    )
    manifest, manifest_sha256 = _validate_prepared_input(prepared_dir, config_path)

    assert manifest_sha256 == file_sha256(manifest_path)
    assert manifest["frame_count"] == 2
    assert all(Path(frame["path"]).suffix == ".jpg" for frame in manifest["frames"])
    assert all(Path(frame["scoring_path"]).suffix == ".png" for frame in manifest["frames"])
    scoring_frame = Path(manifest["frames"][0]["scoring_path"])
    scoring_frame.write_bytes(scoring_frame.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="scoring-frame hash differs"):
        _validate_prepared_input(prepared_dir, config_path)


def test_model_repository_must_be_clean(tmp_path: Path) -> None:
    repository = tmp_path / "model"
    repository.mkdir()
    subprocess.run(("git", "init"), cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.com"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Test"),
        cwd=repository,
        check=True,
    )
    source = repository / "model.py"
    source.write_text("MODEL = 1\n", encoding="utf-8")
    subprocess.run(("git", "add", "model.py"), cwd=repository, check=True)
    subprocess.run(
        ("git", "commit", "-m", "pin model"),
        cwd=repository,
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert validate_repository_revision(repository, revision, "test") == revision

    source.write_text("MODEL = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted"):
        validate_repository_revision(repository, revision, "test")


def test_worker_command_binds_model_gpu_and_shared_input(tmp_path: Path) -> None:
    command = build_worker_command(
        python=Path("/envs/sam31/bin/python"),
        model_id="sam3.1",
        prepared_dir=tmp_path / "shared",
        task_config=tmp_path / "task.json",
        model_config=tmp_path / "sam31.json",
        output_dir=tmp_path / "sam3.1",
        gpu=GPUInfo(7, "GPU-7", 80_000, 1_000, 79_000),
        minimum_free_gpu_mib=32_000,
        seed=17,
    )

    assert command[0] == "/envs/sam31/bin/python"
    assert command[command.index("--model") + 1] == "sam3.1"
    assert command[command.index("--gpu") + 1] == "7"
    assert command[command.index("--seed") + 1] == "17"
    assert command[command.index("--prepared-dir") + 1] == str(tmp_path / "shared")


def test_comparison_detects_potential_label_takeover_without_promotion(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    left = np.zeros((2, 2, 3), dtype=np.uint8)
    right = np.zeros((2, 2, 3), dtype=np.uint8)
    left[:, :, 0] = 1
    right[:, :, 2] = 1
    shadow_left = left.copy()
    shadow_right = right.copy()
    shadow_left[1] = right[1]
    shadow_right[1] = left[1]
    sam2_path = tmp_path / "sam2.npz"
    sam31_path = tmp_path / "sam31.npz"
    sam2_keys = save_packed_masks(
        sam2_path,
        {"left_sleeve": left, "right_sleeve": right},
        frame_count=2,
        height=2,
        width=3,
    )
    sam31_keys = save_packed_masks(
        sam31_path,
        {"left_sleeve": shadow_left, "right_sleeve": shadow_right},
        frame_count=2,
        height=2,
        width=3,
    )
    common = {
        "prepared_input_sha256": "d" * 64,
        "runtime": {
            "elapsed_seconds": 2.0,
            "peak_cuda_memory_mib": 100.0,
        },
        "incumbent_threshold_diagnostics": {},
    }
    authoritative = {
        **common,
        "model_id": "sam2",
        "masks": str(sam2_path),
        "masks_sha256": file_sha256(sam2_path),
        "mask_keys": sam2_keys,
        "hard_gates_passed": True,
    }
    shadow = {
        **common,
        "model_id": "sam3.1",
        "masks": str(sam31_path),
        "masks_sha256": file_sha256(sam31_path),
        "mask_keys": sam31_keys,
        "hard_gates_passed": None,
    }

    comparison = compare_tracker_results(authoritative, shadow)
    mask_path, mask_sha256 = validate_result_mask_artifact(
        authoritative,
        expected_model_id="sam2",
    )

    assert comparison["potential_label_takeover_frames"]["left_sleeve"] == [1]
    assert comparison["potential_label_takeover_frames"]["right_sleeve"] == [1]
    assert comparison["promotion_eligible"] is False
    assert comparison["authoritative_hard_gates_passed"] is True
    assert comparison["shadow_hard_gates_passed"] is None
    assert mask_path == sam2_path
    assert mask_sha256 == file_sha256(sam2_path)

    tampered_shadow = {**shadow, "masks_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="differs from its bound"):
        compare_tracker_results(authoritative, tampered_shadow)
