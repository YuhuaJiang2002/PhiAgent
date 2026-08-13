from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_flower_task_adapter import (
    _contact_observation,
    _motion_similarity,
    _similarity,
    _validate_evaluation_lineage,
)


def test_similarity_is_one_for_equal_pixels_and_decreases_with_error() -> None:
    target = np.zeros((8, 8, 3), dtype=np.uint8)
    close = np.full_like(target, 8)
    far = np.full_like(target, 64)

    assert _similarity(np, target, target) == 1.0
    assert _similarity(np, target, close) > _similarity(np, target, far)


def test_motion_similarity_rewards_matching_contact_motion() -> None:
    target = [np.zeros((5, 5, 3), dtype=np.uint8) for _ in range(3)]
    target[1][2, 2] = 100
    target[2][2, 3] = 100
    matching = [frame.copy() for frame in target]
    static = [np.zeros((5, 5, 3), dtype=np.uint8) for _ in range(3)]
    masks = [np.ones((5, 5), dtype=bool) for _ in range(3)]

    assert _motion_similarity(np, target, matching, masks) == 1.0
    assert _motion_similarity(np, target, matching, masks) > _motion_similarity(
        np, target, static, masks
    )


def test_contact_observation_accepts_metric_control_schema() -> None:
    assert _contact_observation(
        {"contact_xy": [12, 34], "contact_active": True}
    ) == (12.0, 34.0, True)


def _lineage_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name in (
        "target",
        "zero_shot",
        "adapted",
        "trajectory",
        "control",
        "reference",
        "adapter",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        paths[name] = path

    def digest(name: str) -> str:
        return hashlib.sha256(paths[name].read_bytes()).hexdigest()

    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "passed": True,
                "clip_records": [
                    {
                        "split": "validation",
                        "target_sha256": digest("target"),
                        "control_sha256": digest("control"),
                        "trajectory_sha256": digest("trajectory"),
                        "reference_sha256": digest("reference"),
                    }
                ],
            }
        )
    )
    paths["dataset_validation"] = validation
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "sha256": digest("target"),
                        "kind": "target_video",
                        "split": "validation",
                    },
                    {
                        "sha256": digest("control"),
                        "kind": "vace_control_video",
                        "split": "validation",
                    },
                    {
                        "sha256": digest("reference"),
                        "kind": "vace_reference_image",
                        "split": "train",
                    },
                ]
            }
        )
    )
    paths["frozen_manifest"] = manifest
    shared_config = {
        "checkpoint_dir": "/checkpoint",
        "control_video": str(paths["control"]),
        "reference_image": str(paths["reference"]),
        "denoising_strength": 1.0,
        "prompt": "flower",
        "gpu": 1,
        "minimum_free_gpu_mib": 30000,
        "seed": 42,
        "height": 256,
        "width": 448,
        "num_frames": 17,
        "fps": 8,
        "steps": 20,
    }
    checkpoint_files = [{"path": "/checkpoint/model", "sha256": "a" * 64}]
    source_git = {
        "source_git_head": "b" * 40,
        "source_git_status_sha256": "c" * 64,
    }
    for arm, output_name in (("zero", "zero_shot"), ("adapted", "adapted")):
        metadata_path = tmp_path / f"{arm}-metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "output": str(paths[output_name]),
                    "output_sha256": digest(output_name),
                    "config": shared_config,
                    "git": source_git,
                    "inputs": {
                        "control_sha256": digest("control"),
                        "reference_sha256": digest("reference"),
                        "lora_sha256": (
                            None if arm == "zero" else digest("adapter")
                        ),
                        "checkpoint_files": checkpoint_files,
                    },
                }
            )
        )
        paths[f"{arm}_metadata"] = metadata_path
    return paths


def test_evaluation_lineage_requires_matched_heldout_generation(
    tmp_path: Path,
) -> None:
    paths = _lineage_fixture(tmp_path)
    result = _validate_evaluation_lineage(paths)

    assert result["passed"] is True
    assert all(result["matched_config_gates"].values())


def test_evaluation_lineage_rejects_seed_mismatch(tmp_path: Path) -> None:
    paths = _lineage_fixture(tmp_path)
    metadata = json.loads(paths["adapted_metadata"].read_text())
    metadata["config"]["seed"] = 314159
    paths["adapted_metadata"].write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="not matched"):
        _validate_evaluation_lineage(paths)
