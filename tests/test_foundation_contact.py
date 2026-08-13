from __future__ import annotations

import numpy as np
import pytest

from phiagent.agent.contact_dynamics_evolution import (
    derive_foundation_pipeline_experiments,
)
from phiagent.perception.foundation_contact import (
    ContactForceContract,
    EvidenceClass,
    MetricCameraContract,
    RobotTrajectoryContract,
    StemCenterlineContract,
    decide_foundation_contact_status,
    validate_contact_force_sequence,
    validate_metric_camera_sequence,
    validate_robot_trajectory,
    validate_stem_centerlines,
)


def test_foundation_depth_does_not_masquerade_as_bounded_metric_scale() -> None:
    contract = MetricCameraContract(
        camera_frame="camera:video",
        world_frame="world:reconstruction",
        timeline="frame:source",
        fps=24.0,
        image_width=4,
        image_height=3,
        intrinsics_evidence=EvidenceClass.FOUNDATION_MODEL_ESTIMATE,
        depth_evidence=EvidenceClass.FOUNDATION_MODEL_ESTIMATE,
        metric_scale_source="UniDepthV2 learned metric prior",
    )
    result = validate_metric_camera_sequence(
        np,
        contract=contract,
        frame_indices=np.asarray([0, 24]),
        intrinsics_px=np.asarray([[3.0, 0, 2.0], [0, 3.0, 1.0], [0, 0, 1.0]]),
        world_from_camera=np.repeat(np.eye(4)[None], 2, axis=0),
        depth_m=np.ones((2, 3, 4)),
        depth_confidence=np.ones((2, 3, 4)),
    )
    assert result["passed"] is False
    assert result["proposal_passed"] is False
    assert result["gates"]["context_scale_stability_bounded_or_calibrated"] is False
    assert result["calibrated_scale"] is False


def test_context_stable_learned_metric_camera_is_a_proposal_not_calibration() -> None:
    contract = MetricCameraContract(
        camera_frame="camera:video",
        world_frame="world:reconstruction",
        timeline="frame:source",
        fps=24.0,
        image_width=4,
        image_height=3,
        intrinsics_evidence=EvidenceClass.FOUNDATION_MODEL_ESTIMATE,
        depth_evidence=EvidenceClass.FOUNDATION_MODEL_ESTIMATE,
        metric_scale_source="DA3 Nested learned metric prior",
        learned_context_scale_variation_fraction=0.0032,
    )
    result = validate_metric_camera_sequence(
        np,
        contract=contract,
        frame_indices=np.asarray([0, 24]),
        intrinsics_px=np.asarray([[3.0, 0, 2.0], [0, 3.0, 1.0], [0, 0, 1.0]]),
        world_from_camera=np.repeat(np.eye(4)[None], 2, axis=0),
        depth_m=np.ones((2, 3, 4)),
        depth_confidence=np.ones((2, 3, 4)),
    )
    assert result["proposal_passed"] is True
    assert result["gates"]["absolute_metric_scale_calibrated"] is False
    assert result["passed"] is False


def test_calibrated_metric_camera_sequence_passes() -> None:
    contract = MetricCameraContract(
        camera_frame="camera:video",
        world_frame="world:reconstruction",
        timeline="frame:source",
        fps=24.0,
        image_width=4,
        image_height=3,
        intrinsics_evidence=EvidenceClass.CALIBRATED_GEOMETRY,
        depth_evidence=EvidenceClass.SENSOR_MEASUREMENT,
        metric_scale_source="registered RGB-D sensor",
    )
    result = validate_metric_camera_sequence(
        np,
        contract=contract,
        frame_indices=np.asarray([0, 24]),
        intrinsics_px=np.asarray([[3.0, 0, 2.0], [0, 3.0, 1.0], [0, 0, 1.0]]),
        world_from_camera=np.repeat(np.eye(4)[None], 2, axis=0),
        depth_m=np.ones((2, 3, 4)),
        depth_confidence=np.ones((2, 3, 4)),
    )
    assert result["passed"] is True


def test_calibrated_geometry_requires_uncertainty_groups_and_report_binding() -> None:
    common = {
        "camera_frame": "camera:video",
        "world_frame": "world:reconstruction",
        "timeline": "frame:source",
        "fps": 24.0,
        "image_width": 4,
        "image_height": 3,
        "intrinsics_evidence": EvidenceClass.CALIBRATED_GEOMETRY,
        "depth_evidence": EvidenceClass.CALIBRATED_GEOMETRY,
        "metric_scale_source": "sparse independent metric anchors",
    }
    arrays = {
        "frame_indices": np.asarray([0, 24]),
        "intrinsics_px": np.asarray(
            [[3.0, 0, 2.0], [0, 3.0, 1.0], [0, 0, 1.0]]
        ),
        "world_from_camera": np.repeat(np.eye(4)[None], 2, axis=0),
        "depth_m": np.ones((2, 3, 4)),
        "depth_confidence": np.ones((2, 3, 4)),
    }
    missing_metadata = validate_metric_camera_sequence(
        np, contract=MetricCameraContract(**common), **arrays
    )
    assert missing_metadata["passed"] is False
    assert missing_metadata["gates"]["absolute_metric_scale_calibrated"] is False

    bound = validate_metric_camera_sequence(
        np,
        contract=MetricCameraContract(
            **common,
            absolute_scale_standard_deviation_fraction=0.01,
            independent_calibration_groups=2,
            calibration_report_sha256="b" * 64,
        ),
        **arrays,
    )
    assert bound["passed"] is True


def test_robot_trajectory_requires_complete_q_and_render_validation() -> None:
    contract = RobotTrajectoryContract(
        embodiment_id="unitree-g1-sharpa-wave",
        robot_base_frame="robot_base:g1",
        timeline="frame:source",
        fps=24.0,
        joint_names=("shoulder", "elbow"),
        joint_limits_rad=((-1.0, 1.0), (-1.5, 1.5)),
        asset_sha256={"g1.xml": "a" * 64, "sharpa.xml": "b" * 64},
        trajectory_evidence=EvidenceClass.PHYSICS_SOLVER_ESTIMATE,
    )
    result = validate_robot_trajectory(
        np,
        contract=contract,
        frame_indices=np.asarray([0, 1, 2]),
        joint_positions_rad=np.zeros((3, 2)),
    )
    assert result["passed"] is False
    assert result["gates"]["render_reprojection_validated"] is False
    with pytest.raises(ValueError, match="shape"):
        validate_robot_trajectory(
            np,
            contract=contract,
            frame_indices=np.asarray([0, 1, 2]),
            joint_positions_rad=np.zeros((3, 1)),
        )


def test_stem_centerlines_require_persistent_metric_geometry() -> None:
    contract = StemCenterlineContract(
        instance_ids=("pink-stem-01",),
        coordinate_frame="world:reconstruction",
        timeline="frame:source",
        nodes_per_stem=4,
        geometry_evidence=EvidenceClass.FOUNDATION_MODEL_ESTIMATE,
    )
    base = np.stack((np.zeros(4), np.linspace(0.0, 0.3, 4), np.ones(4)), axis=1)
    centerlines = np.repeat(base[None, None], 3, axis=0)
    result = validate_stem_centerlines(
        np,
        contract=contract,
        frame_indices=np.asarray([0, 1, 2]),
        centerlines_m=centerlines,
        confidence=np.ones((3, 1, 4)),
    )
    assert result["passed"] is True


def test_visual_model_cannot_be_declared_a_force_source() -> None:
    with pytest.raises(ValueError, match="never a visual model"):
        ContactForceContract(
            coordinate_frame="world:reconstruction",
            timeline="frame:source",
            instance_ids=("stem-1",),
            force_evidence=EvidenceClass.FOUNDATION_MODEL_ESTIMATE,
            source_name="vision transformer",
        ).validate_metadata()


def test_physics_force_sequence_propagates_covariance() -> None:
    contract = ContactForceContract(
        coordinate_frame="world:reconstruction",
        timeline="frame:source",
        instance_ids=("stem-1",),
        force_evidence=EvidenceClass.PHYSICS_SOLVER_ESTIMATE,
        source_name="damped rod inverse dynamics",
    )
    forces = np.zeros((3, 1, 2, 3))
    covariance = np.repeat(np.eye(3)[None, None, None], 3, axis=0)
    covariance = np.repeat(covariance, 2, axis=2) * 1e-4
    result = validate_contact_force_sequence(
        np,
        contract=contract,
        forces_n=forces,
        solver_residual_n=np.zeros((3, 1)),
        covariance_n2=covariance,
    )
    assert result["passed"] is True


def test_end_to_end_status_is_fail_closed() -> None:
    result = decide_foundation_contact_status(
        {
            "metric_camera": {"passed": True},
            "robot_trajectory": {"passed": True},
            "stem_centerlines": {"passed": False},
            "contact_forces": None,
        }
    )
    assert result["status"] == "PARTIAL"
    assert result["missing_or_rejected_stages"] == ["stem_centerlines", "contact_forces"]


def test_foundation_evolution_changes_architecture_without_relaxing_gates() -> None:
    report = {
        "status": "PARTIAL",
        "stages": {
            "metric_camera": {"passed": True},
            "robot_trajectory": {"passed": False},
            "stem_centerlines": {
                "passed": False,
                "maximum_segment_length_cv": 1.857,
            },
            "contact_forces": {"passed": False},
        },
    }
    result = derive_foundation_pipeline_experiments(report)
    assert result["promotable"] is False
    experiments = {row["failed_stage"]: row for row in result["experiments"]}
    assert set(experiments) == {"robot_trajectory", "stem_centerlines", "contact_forces"}
    stem = experiments["stem_centerlines"]
    assert "V-DPM/SpaTracker" in stem["architecture_change"]
    assert "threshold is unchanged" in " ".join(stem["promotion_gates"])
    assert experiments["contact_forces"]["blocked_by"] == [
        "robot_trajectory",
        "stem_centerlines",
    ]
    assert all(
        row["mutation_class"] == "architecture_not_hyperparameter"
        for row in result["experiments"]
    )


def test_foundation_evolution_promotes_only_complete_pipeline() -> None:
    stages = {
        name: {"passed": True}
        for name in (
            "metric_camera",
            "robot_trajectory",
            "stem_centerlines",
            "contact_forces",
        )
    }
    result = derive_foundation_pipeline_experiments(
        {"status": "WORKING", "stages": stages}
    )
    assert result["promotable"] is True
    assert result["experiments"] == []
