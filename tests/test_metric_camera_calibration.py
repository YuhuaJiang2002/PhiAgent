from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from phiagent.perception.metric_camera_calibration import (
    MetricDepthCalibrationContract,
    calibrate_metric_camera_sequence,
)
from scripts.compile_foundation_contact_pipeline import _direct_camera_report


def _fixture(
    *, evidence: str = "sensor_measurement", groups: int = 2
) -> dict[str, object]:
    frame_indices = np.arange(6, dtype=np.int64)
    height, width = 8, 10
    yy, xx = np.mgrid[:height, :width]
    predicted = np.stack(
        [0.7 + 0.15 * frame + 0.02 * xx + 0.01 * yy for frame in frame_indices]
    )
    anchor_frames = np.repeat(frame_indices, 4)
    anchor_xy = np.tile(np.asarray(((2, 2), (4, 3), (6, 4), (7, 5))), (6, 1))
    sampled = predicted[
        anchor_frames,
        anchor_xy[:, 1],
        anchor_xy[:, 0],
    ]
    metric = 1.0 / (0.82 / sampled + 0.025)
    group_ids = [f"sensor-{index % groups}" for index in range(len(anchor_frames))]
    poses = np.repeat(np.eye(4)[None], len(frame_indices), axis=0)
    intrinsics = np.repeat(
        np.asarray(((8.0, 0.0, 5.0), (0.0, 8.0, 4.0), (0.0, 0.0, 1.0)))[
            None
        ],
        len(frame_indices),
        axis=0,
    )
    return {
        "frame_indices": frame_indices,
        "intrinsics_px": intrinsics,
        "world_from_camera": poses,
        "predicted_depth_m": predicted,
        "depth_confidence": np.ones_like(predicted),
        "anchor_frame_indices": anchor_frames,
        "anchor_xy_px": anchor_xy,
        "anchor_metric_depth_m": metric,
        "anchor_metric_depth_std_m": np.full(len(anchor_frames), 0.002),
        "anchor_group_ids": group_ids,
        "anchor_evidence_classes": [evidence] * len(anchor_frames),
    }


def test_direct_metric_camera_bundle_binds_calibrated_rgbd(tmp_path: Path) -> None:
    fixture = _fixture()
    samples_path = tmp_path / "metric-camera.npz"
    depth = np.ones((6, 8, 10), dtype=np.float32)
    np.savez_compressed(
        samples_path,
        source_frame_indices=fixture["frame_indices"],
        intrinsics_px=fixture["intrinsics_px"],
        world_from_camera=fixture["world_from_camera"],
        depth_m=depth,
        confidence=np.ones_like(depth),
        bundle_id=np.asarray("metric-camera-test-bundle"),
        source_video_sha256=np.asarray("a" * 64),
        fps=np.asarray(24.0),
        camera_frame=np.asarray("camera:simulated_rgbd"),
        world_frame=np.asarray("robot_base:simulated"),
        timeline=np.asarray("frame:simulation"),
    )
    samples_sha256 = hashlib.sha256(samples_path.read_bytes()).hexdigest()
    report = {
        "passed": True,
        "bundle_id": "metric-camera-test-bundle",
        "samples_sha256": samples_sha256,
        "source_video_sha256": "a" * 64,
        "camera_frame": "camera:simulated_rgbd",
        "world_frame": "robot_base:simulated",
        "timeline": "frame:simulation",
        "fps": 24.0,
        "image_width": 10,
        "image_height": 8,
        "intrinsics_evidence": "calibrated_geometry",
        "depth_evidence": "calibrated_geometry",
        "metric_scale_source": "mujoco exact scene units",
        "absolute_scale_standard_deviation_fraction": 0.0,
        "independent_calibration_groups": 2,
    }

    with np.load(samples_path, allow_pickle=False) as samples:
        result = _direct_camera_report(
            np,
            samples,
            report,
            report_sha256="b" * 64,
            samples_sha256=samples_sha256,
        )

    assert result["passed"] is True
    assert result["gates"]["absolute_metric_scale_calibrated"] is True
    assert result["direct_metric_camera"]["bound"] is True


def test_calibration_cli_rejects_samples_not_bound_to_manifest(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    samples_path = tmp_path / "samples.npz"
    np.savez_compressed(samples_path, value=np.asarray((1.0,)))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input": {"sha256": "a" * 64},
                "outputs": {"samples": {"sha256": "b" * 64}},
            }
        )
    )
    observations_path = tmp_path / "observations.json"
    observations_path.write_text(
        json.dumps(
            {
                "source_video_sha256": "a" * 64,
                "observations": [],
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/calibrate_foundation_metric_camera.py"),
            "--da3-samples",
            str(samples_path),
            "--da3-manifest",
            str(manifest_path),
            "--observations",
            str(observations_path),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "does not match the supplied manifest" in completed.stderr


def _contract() -> MetricDepthCalibrationContract:
    return MetricDepthCalibrationContract(
        camera_frame="camera:processed",
        world_frame="world:calibrated",
        timeline="frame:source",
        source_video_sha256="a" * 64,
        bootstrap_samples=64,
    )


def test_independent_sparse_metric_anchors_calibrate_depth() -> None:
    fixture = _fixture()
    result = calibrate_metric_camera_sequence(
        np, contract=_contract(), **fixture, seed=7
    )

    assert result["passed"] is True
    assert result["anchors_admissible"] == 24
    assert len(result["independent_group_ids"]) == 2
    assert result["anchor_relative_error_p95"] < 1e-9
    assert result["group_holdout_relative_error_p95_max"] < 1e-9
    assert result["scale_standard_deviation_fraction"] < 0.02
    calibrated = result["calibrated_depth_m"]
    predicted = fixture["predicted_depth_m"]
    expected = 1.0 / (0.82 / predicted + 0.025)
    np.testing.assert_allclose(calibrated, expected, rtol=1e-6, atol=1e-6)


def test_foundation_model_anchors_cannot_self_calibrate() -> None:
    result = calibrate_metric_camera_sequence(
        np,
        contract=_contract(),
        **_fixture(evidence="foundation_model_estimate"),
    )

    assert result["passed"] is False
    assert result["anchors_admissible"] == 0
    assert (
        result["gates"]["anchor_evidence_independent_of_foundation_model"] is False
    )


def test_one_sensor_group_cannot_establish_independence() -> None:
    result = calibrate_metric_camera_sequence(
        np, contract=_contract(), **_fixture(groups=1)
    )

    assert result["passed"] is False
    assert result["gates"]["minimum_independent_groups"] is False


def test_exact_asset_anchor_requires_complete_q_and_reprojection() -> None:
    fixture = _fixture(evidence="exact_asset")
    count = len(fixture["anchor_group_ids"])
    result = calibrate_metric_camera_sequence(
        np,
        contract=_contract(),
        **fixture,
        anchor_complete_q=np.zeros(count, dtype=bool),
        anchor_reprojection_rmse_px=np.full(count, 2.0),
    )

    assert result["passed"] is False
    assert result["anchors_admissible"] == 0


def test_hash_bound_exact_asset_with_complete_q_can_calibrate() -> None:
    fixture = _fixture(evidence="exact_asset")
    count = len(fixture["anchor_group_ids"])
    digest = "c" * 64
    result = calibrate_metric_camera_sequence(
        np,
        contract=replace(_contract(), allowed_exact_asset_sha256=(digest,)),
        **fixture,
        anchor_complete_q=np.ones(count, dtype=bool),
        anchor_asset_sha256=[digest] * count,
        anchor_reprojection_rmse_px=np.full(count, 2.0),
    )

    assert result["passed"] is True


def test_invalid_depth_rank_fails_cleanly() -> None:
    fixture = _fixture()
    fixture["predicted_depth_m"] = np.ones((6, 8))
    try:
        calibrate_metric_camera_sequence(np, contract=_contract(), **fixture)
    except ValueError as error:
        assert "TxHxW" in str(error)
    else:
        raise AssertionError("rank-2 predicted depth must be rejected")


def test_calibration_artifact_binds_into_pipeline_compiler(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    fixture = _fixture()
    da3_path = tmp_path / "da3.npz"
    np.savez_compressed(
        da3_path,
        source_frame_indices=fixture["frame_indices"],
        intrinsics_px=fixture["intrinsics_px"],
        world_from_camera=fixture["world_from_camera"],
        depth_m=fixture["predicted_depth_m"],
        confidence=fixture["depth_confidence"],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input": {"sha256": "a" * 64, "fps": 24.0},
                "sampling": {"processed_height": 8, "processed_width": 10},
                "outputs": {
                    "samples": {
                        "sha256": hashlib.sha256(da3_path.read_bytes()).hexdigest()
                    }
                },
            }
        )
    )
    observations_path = tmp_path / "observations.json"
    observations_path.write_text(
        json.dumps(
            {
                "source_video_sha256": "a" * 64,
                "observations": [
                    {
                        "frame_index": int(frame),
                        "pixel_x": float(xy[0]),
                        "pixel_y": float(xy[1]),
                        "metric_depth_m": float(metric),
                        "standard_deviation_m": 0.002,
                        "group_id": group,
                        "evidence_class": evidence,
                    }
                    for frame, xy, metric, group, evidence in zip(
                        fixture["anchor_frame_indices"],
                        fixture["anchor_xy_px"],
                        fixture["anchor_metric_depth_m"],
                        fixture["anchor_group_ids"],
                        fixture["anchor_evidence_classes"],
                        strict=True,
                    )
                ],
            }
        )
    )
    config_path = tmp_path / "calibration-config.json"
    config_path.write_text(
        json.dumps(
            {
                "camera_frame": "camera:processed",
                "world_frame": "world:calibrated",
                "timeline": "frame:source",
                "minimum_anchors": 20,
                "minimum_independent_groups": 2,
                "maximum_anchor_relative_error_p95": 0.04,
                "maximum_group_holdout_relative_error_p95": 0.06,
                "maximum_scale_standard_deviation_fraction": 0.02,
                "minimum_robust_inlier_fraction": 0.8,
                "maximum_unscaled_camera_motion_m": 0.01,
                "maximum_exact_asset_reprojection_rmse_px": 8.0,
                "allowed_exact_asset_sha256": ["c" * 64],
                "bootstrap_samples": 64,
            }
        )
    )
    calibration_dir = tmp_path / "calibration"
    calibration = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/calibrate_foundation_metric_camera.py"),
            "--da3-samples",
            str(da3_path),
            "--da3-manifest",
            str(manifest_path),
            "--observations",
            str(observations_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(calibration_dir),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert calibration.returncode == 0, calibration.stderr
    calibration_report = json.loads(
        (calibration_dir / "calibration-report.json").read_text()
    )
    assert calibration_report["passed"] is True

    context_path = tmp_path / "context.json"
    context_path.write_text(
        json.dumps(
            {
                "passed": True,
                "summary": {"context_scale_variation_fraction_p95": 0.001},
            }
        )
    )
    asset_paths = []
    for name in ("g1.xml", "left.xml", "right.xml"):
        path = tmp_path / name
        path.write_text("<mujoco/>")
        asset_paths.append(path)
    pipeline_dir = tmp_path / "pipeline"
    compiled = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/compile_foundation_contact_pipeline.py"),
            "--da3-samples",
            str(da3_path),
            "--da3-manifest",
            str(manifest_path),
            "--context-scale-report",
            str(context_path),
            "--g1-model",
            str(asset_paths[0]),
            "--sharpa-left-model",
            str(asset_paths[1]),
            "--sharpa-right-model",
            str(asset_paths[2]),
            "--camera-calibration-report",
            str(calibration_dir / "calibration-report.json"),
            "--calibrated-camera-samples",
            str(calibration_dir / "calibrated-camera-samples.npz"),
            "--output-dir",
            str(pipeline_dir),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 2, compiled.stderr
    pipeline_report = json.loads((pipeline_dir / "pipeline-report.json").read_text())
    camera = pipeline_report["stages"]["metric_camera"]
    assert camera["calibration_bridge"]["bound"] is True
    assert camera["gates"]["absolute_metric_scale_calibrated"] is True
    assert camera["passed"] is True
    assert pipeline_report["status"] == "PARTIAL"
