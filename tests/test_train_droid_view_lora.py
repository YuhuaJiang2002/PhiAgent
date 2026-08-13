from __future__ import annotations

import pytest

from phiagent.training.diffsynth_animate import load_frozen_manifest
from scripts.train_droid_view_lora import validate_dataset_contract


def _contract() -> dict:
    return {
        "method": "phiagent_droid_wrist_to_exterior_vace_lora_dataset",
        "status": "WORKING",
        "leakage_checks": {
            "episode_disjoint": True,
            "heldout_anchors_used_for_training": False,
            "heldout_targets_used_for_training": False,
        },
        "split": {
            "train": [0, 1],
            "validation": [12],
            "holdout": [21, 60, 77],
        },
        "adaptation_manifest_sha256": "a" * 64,
        "training_example_count": 1,
        "conditioning_contract": {
            "real_condition": [
                "full wrist-camera clip",
                "one exterior-camera anchor frame at the requested target viewpoint",
            ]
        },
        "holdout_records": [
            {"episode_index": value, "training_use": False, "targets": {"target_a": {}}}
            for value in (21, 60, 77)
        ],
    }


def _manifest() -> dict:
    return {
        "assets": [{"path": "dataset/train/ep000/condition.mp4"}],
        "vace_examples": [{}],
    }


def test_contract_validation_accepts_disjoint_heldout_data() -> None:
    summary = validate_dataset_contract(_contract(), _manifest(), "a" * 64)
    assert summary["holdout_episodes"] == [21, 60, 77]
    assert summary["target_anchor_disclosed"] is True


def test_contract_validation_rejects_target_leakage() -> None:
    contract = _contract()
    contract["leakage_checks"]["heldout_targets_used_for_training"] = True
    with pytest.raises(ValueError, match="leakage"):
        validate_dataset_contract(contract, _manifest(), "a" * 64)


def test_frozen_manifest_resolves_relative_asset_paths(tmp_path) -> None:
    asset = tmp_path / "dataset" / "train" / "condition.mp4"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"not-empty")
    import hashlib
    import json

    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "0.1.0",
        "method": "sharpa_lightweight_adaptation_not_official_phizero",
        "experiment_id": "relative-vace",
        "arm": "vace_lora",
        "evidence_scope": "development_only",
        "assets": [
            {
                "asset_id": "condition",
                "path": "dataset/train/condition.mp4",
                "split": "train",
                "kind": "vace_control_video",
                "source_uri": "local://condition",
                "rights_basis": "test fixture",
                "sha256": digest,
                "size_bytes": asset.stat().st_size,
                "training_authorized": True,
            },
            {
                "asset_id": "target",
                "path": "dataset/train/condition.mp4",
                "split": "train",
                "kind": "target_video",
                "source_uri": "local://target",
                "rights_basis": "test fixture",
                "sha256": digest,
                "size_bytes": asset.stat().st_size,
                "training_authorized": True,
            },
            {
                "asset_id": "anchor",
                "path": "dataset/train/condition.mp4",
                "split": "train",
                "kind": "vace_reference_image",
                "source_uri": "local://anchor",
                "rights_basis": "test fixture",
                "sha256": digest,
                "size_bytes": asset.stat().st_size,
                "training_authorized": True,
            },
        ],
        "vace_examples": [
            {
                "example_id": "example",
                "target_video_asset_id": "target",
                "control_video_asset_id": "condition",
                "reference_image_asset_id": "anchor",
                "prompt": "test",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate asset content"):
        load_frozen_manifest(path)
