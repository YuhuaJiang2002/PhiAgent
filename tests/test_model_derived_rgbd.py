from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from phiagent.perception.model_derived_rgbd import (
    ModelDerivedRGBDContract,
    audit_model_derived_rgbd,
    depth_splat_rgbd,
)
from scripts.compile_foundation_contact_pipeline import _model_derived_rgbd_diagnostic


def _contract() -> ModelDerivedRGBDContract:
    return ModelDerivedRGBDContract(
        source_video_sha256="a" * 64,
        timeline="frame:source_video",
        fps=24.0,
        model_name="DA3 Nested",
        model_revision="revision",
        checkpoint_sha256="b" * 64,
        source_group_frames=("world:run-0", "world:run-1"),
        virtual_camera_frame="camera:virtual-4cm",
        minimum_mean_virtual_view_coverage=0.50,
    )


def test_identity_depth_splat_preserves_rgbd() -> None:
    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    depth = np.full((4, 5), 2.0, dtype=np.float64)
    result = depth_splat_rgbd(
        np,
        source_rgb=rgb,
        source_depth_m=depth,
        source_confidence=np.ones_like(depth),
        intrinsics_px=np.asarray([[8.0, 0.0, 2.0], [0.0, 8.0, 1.5], [0.0, 0.0, 1.0]]),
        target_camera_from_source_camera=np.eye(4),
    )

    np.testing.assert_array_equal(result["rgb"], rgb)
    np.testing.assert_allclose(result["depth_m"], depth)
    assert bool(np.all(result["valid_mask"]))


def test_translated_virtual_view_is_visible_surface_only() -> None:
    rgb = np.full((8, 10, 3), 127, dtype=np.uint8)
    transform = np.eye(4)
    transform[0, 3] = 0.25
    result = depth_splat_rgbd(
        np,
        source_rgb=rgb,
        source_depth_m=np.ones((8, 10)),
        source_confidence=np.ones((8, 10)),
        intrinsics_px=np.asarray([[10.0, 0.0, 4.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]]),
        target_camera_from_source_camera=transform,
    )

    assert 0 < float(np.mean(result["valid_mask"])) < 1
    assert bool(np.all(result["source_flat_index"][result["valid_mask"]] >= 0))


def test_full_da3_resolution_splat_has_finite_depth() -> None:
    height, width = 280, 504
    result = depth_splat_rgbd(
        np,
        source_rgb=np.zeros((height, width, 3), dtype=np.uint8),
        source_depth_m=np.full((height, width), 4.0),
        source_confidence=np.ones((height, width)),
        intrinsics_px=np.asarray(
            [[1165.0, 0.0, 252.0], [0.0, 1158.0, 140.0], [0.0, 0.0, 1.0]]
        ),
        target_camera_from_source_camera=np.eye(4),
    )

    assert bool(np.all(np.isfinite(result["depth_m"])))


def test_model_rgbd_proposal_never_becomes_physical_calibration() -> None:
    frames = np.arange(0, 24, 3)
    result = audit_model_derived_rgbd(
        np,
        contract=_contract(),
        source_frame_indices=frames,
        source_group_indices=np.arange(len(frames)) % 2,
        depth_m=np.ones((len(frames), 3, 4)),
        virtual_view_coverage=np.full(4, 0.8),
        cycle_depth_relative_error_p95=np.zeros(4),
        group_median_depth_m=np.asarray([1.0, 1.005]),
    )

    assert result["proposal_passed"] is True
    assert result["physical_calibration_passed"] is False
    assert result["independent_physical_groups"] == 0
    assert not any(result["physical_gates"].values())


def test_model_rgbd_audit_rejects_implicit_world_frame_mix() -> None:
    frames = np.arange(0, 24, 3)
    contract = replace(_contract(), source_group_frames=("world:same", "world:same"))
    with pytest.raises(ValueError, match="distinct world frame"):
        audit_model_derived_rgbd(
            np,
            contract=contract,
            source_frame_indices=frames,
            source_group_indices=np.arange(len(frames)) % 2,
            depth_m=np.ones((len(frames), 3, 4)),
            virtual_view_coverage=np.full(4, 0.8),
            cycle_depth_relative_error_p95=np.zeros(4),
            group_median_depth_m=np.asarray([1.0, 1.0]),
        )


def test_compiler_binds_model_rgbd_without_upgrading_authority(tmp_path) -> None:
    artifact = tmp_path / "rgbd.npz"
    artifact.write_bytes(b"model-rgbd")
    report = {
        "passed": True,
        "source_video_sha256": "a" * 64,
        "evidence_class": "foundation_model_estimate",
        "physical_calibration_passed": False,
        "outputs": {
            "rgbd": {
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        },
        "audit": {
            "proposal_passed": True,
            "independent_physical_groups": 0,
            "reason": "model_derived_same_video_not_independent_calibration",
            "metrics": {"samples": 110},
        },
    }
    result = _model_derived_rgbd_diagnostic(
        report,
        report_sha256="c" * 64,
        source_video_sha256="a" * 64,
    )

    assert result["bound"] is True
    assert result["proposal_passed"] is True
    assert result["physical_calibration_passed"] is False
