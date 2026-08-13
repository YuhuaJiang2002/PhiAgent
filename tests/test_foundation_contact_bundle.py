from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.compile_foundation_contact_pipeline import _bundle_lineage_report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(tmp_path: Path, *, force_bundle_id: str = "bundle-1") -> dict[str, Path]:
    frames = np.arange(6, dtype=np.int64)
    common = {
        "source_video_sha256": np.asarray("a" * 64),
        "source_frame_indices": frames,
        "fps": np.asarray(24.0),
        "timeline": np.asarray("frame:source_video"),
    }
    camera_path = tmp_path / "camera.npz"
    np.savez_compressed(
        camera_path,
        **common,
        bundle_id=np.asarray("bundle-1"),
        world_frame=np.asarray("robot_base:g1"),
    )
    robot_path = tmp_path / "robot.npz"
    np.savez_compressed(
        robot_path,
        **common,
        bundle_id=np.asarray("bundle-1"),
        robot_base_frame=np.asarray("robot_base:g1"),
    )
    stem_path = tmp_path / "stems.npz"
    np.savez_compressed(
        stem_path,
        **common,
        bundle_id=np.asarray("bundle-1"),
        coordinate_frame=np.asarray("robot_base:g1"),
        instance_ids=np.asarray(("stem-1",)),
    )
    force_path = tmp_path / "forces.npz"
    np.savez_compressed(
        force_path,
        **common,
        bundle_id=np.asarray(force_bundle_id),
        coordinate_frame=np.asarray("robot_base:g1"),
        instance_ids=np.asarray(("stem-1",)),
    )
    camera_report_path = tmp_path / "camera.json"
    camera_report_path.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-1",
                "source_video_sha256": "a" * 64,
            }
        )
    )
    manifest_path = tmp_path / "manifest.json"
    artifacts = {
        name: {"sha256": _sha256(path)}
        for name, path in {
            "metric_camera_samples": camera_path,
            "metric_camera_report": camera_report_path,
            "robot_trajectory": robot_path,
            "stem_centerlines": stem_path,
            "contact_forces": force_path,
        }.items()
    }
    manifest_path.write_text(
        json.dumps(
            {
                "bundle_id": "bundle-1",
                "source_video_sha256": "a" * 64,
                "artifacts": artifacts,
            }
        )
    )
    return {
        "camera": camera_path,
        "camera_report": camera_report_path,
        "robot": robot_path,
        "stems": stem_path,
        "forces": force_path,
        "manifest": manifest_path,
    }


def _lineage(paths: dict[str, Path]) -> dict[str, object]:
    camera = np.load(paths["camera"], allow_pickle=False)
    return _bundle_lineage_report(
        np,
        camera_samples=camera,
        camera_report=json.loads(paths["camera_report"].read_text()),
        camera_samples_path=paths["camera"],
        camera_report_path=paths["camera_report"],
        robot_path=paths["robot"],
        stem_path=paths["stems"],
        force_path=paths["forces"],
        bundle_manifest_path=paths["manifest"],
    )


def test_bundle_lineage_accepts_one_hash_bound_timeline(tmp_path: Path) -> None:
    result = _lineage(_write_bundle(tmp_path))

    assert result["passed"] is True
    assert all(result["gates"].values())


def test_bundle_lineage_rejects_cross_run_force_artifact(tmp_path: Path) -> None:
    result = _lineage(_write_bundle(tmp_path, force_bundle_id="bundle-2"))

    assert result["passed"] is False
    assert result["gates"]["common_bundle_id"] is False
    assert result["gates"]["bundle_manifest_identity_matches"] is False
