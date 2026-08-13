from __future__ import annotations

import numpy as np
import pytest

from phiagent.rendering.contact_dynamics import (
    ArticulatedHandContract,
    InteractionFrameContract,
    MetricContactContract,
    StemRodContract,
    assess_metric_force_closure,
    causal_motion_audit,
    infer_stem_contact_forces,
    simulate_damped_stem,
    validate_kinematic_sequence,
)


def _hand() -> ArticulatedHandContract:
    return ArticulatedHandContract(
        embodiment_id="test-hand",
        coordinate_frame="robot_base:test",
        joint_names=("root", "finger-a", "finger-b"),
        parent_indices=(-1, 0, 0),
        joint_limits_rad=((-1.0, 1.0), (-0.5, 1.5), (-0.5, 1.5)),
        fingertip_indices=(1, 2),
        palm_index=0,
    )


def test_hand_contract_rejects_a_broken_tree() -> None:
    broken = ArticulatedHandContract(
        embodiment_id="broken",
        coordinate_frame="robot_base:test",
        joint_names=("root", "a", "b"),
        parent_indices=(-1, 2, 0),
        joint_limits_rad=((-1.0, 1.0),) * 3,
        fingertip_indices=(1, 2),
        palm_index=0,
    )
    with pytest.raises(ValueError, match="precede"):
        broken.validate()


def test_kinematic_sequence_rejects_bone_breathing_and_limits() -> None:
    joints = np.asarray(
        [
            [[0, 0, 0], [1, 0, 0], [-1, 0, 0]],
            [[0, 0, 0], [2, 0, 0], [-1, 0, 0]],
        ],
        dtype=float,
    )
    angles = np.zeros((2, 3))
    angles[1, 1] = 2.0
    result = validate_kinematic_sequence(
        np, joints_xyz_m=joints, joint_angles_rad=angles, contract=_hand()
    )
    assert result["passed"] is False
    assert result["gates"]["bone_lengths_rigid"] is False
    assert result["joint_limit_violations"] == 1


def test_metric_contact_fails_closed_without_depth_or_force() -> None:
    frame = InteractionFrameContract(
        camera_frame="camera:pixels",
        metric_frame="camera:metric",
        timeline="frame:source",
        fps=24.0,
    )
    result = assess_metric_force_closure(
        np,
        contact_points_m=None,
        surface_gaps_m=None,
        contact_normals=None,
        contact_forces_n=None,
        object_center_m=None,
        external_force_n=None,
        external_moment_nm=None,
        fingertip_indices=None,
        frame_contract=frame,
        contact_contract=MetricContactContract(),
        depth_source=None,
        force_source=None,
        occlusion_order_known=False,
    )
    assert result["passed"] is False
    assert "missing_metric_camera" in result["reasons"]
    assert "missing_force_source" in result["reasons"]


def test_metric_six_contact_force_closure_passes() -> None:
    frame = InteractionFrameContract(
        camera_frame="camera:pixels",
        metric_frame="camera:metric",
        timeline="frame:source",
        fps=24.0,
        fx_pixels=500.0,
        fy_pixels=500.0,
        cx_pixels=320.0,
        cy_pixels=240.0,
        metric_scale_source="calibrated-rgbd",
    )
    result = assess_metric_force_closure(
        np,
        contact_points_m=np.asarray(
            [
                [-0.002, 0, 0],
                [0.002, 0, 0],
                [0, -0.002, 0],
                [0, 0.002, 0],
                [0, 0, -0.002],
                [0, 0, 0.002],
            ]
        ),
        surface_gaps_m=np.full(6, 0.001),
        contact_normals=np.asarray(
            [
                [-1.0, 0, 0],
                [1.0, 0, 0],
                [0, -1.0, 0],
                [0, 1.0, 0],
                [0, 0, -1.0],
                [0, 0, 1.0],
            ]
        ),
        contact_forces_n=np.asarray(
            [
                [0.05, 0, 0],
                [-0.05, 0, 0],
                [0, 0.05, 0],
                [0, -0.05, 0],
                [0, 0, 0.05],
                [0, 0, -0.05],
            ]
        ),
        object_center_m=np.zeros(3),
        external_force_n=np.zeros(3),
        external_moment_nm=np.zeros(3),
        fingertip_indices=(1, 2, 3, 4, 5, 6),
        frame_contract=frame,
        contact_contract=MetricContactContract(),
        depth_source="registered-depth",
        force_source="mujoco-contact-solver",
        occlusion_order_known=True,
    )
    assert result["passed"] is True
    assert result["friction_cone_violations"] == 0
    assert result["force_closure"]["linearized_grasp_matrix_rank"] == 6


def test_balanced_two_finger_contact_is_not_misreported_as_3d_force_closure() -> None:
    frame = InteractionFrameContract(
        camera_frame="camera:pixels",
        metric_frame="camera:metric",
        timeline="frame:source",
        fps=24.0,
        fx_pixels=500.0,
        fy_pixels=500.0,
        cx_pixels=320.0,
        cy_pixels=240.0,
        metric_scale_source="calibrated-rgbd",
    )
    result = assess_metric_force_closure(
        np,
        contact_points_m=np.asarray([[-0.002, 0, 0], [0.002, 0, 0]]),
        surface_gaps_m=np.asarray([0.001, 0.001]),
        contact_normals=np.asarray([[-1.0, 0, 0], [1.0, 0, 0]]),
        contact_forces_n=np.asarray([[0.05, 0, 0], [-0.05, 0, 0]]),
        object_center_m=np.zeros(3),
        external_force_n=np.zeros(3),
        external_moment_nm=np.zeros(3),
        fingertip_indices=(1, 2),
        frame_contract=frame,
        contact_contract=MetricContactContract(),
        depth_source="registered-depth",
        force_source="mujoco-contact-solver",
        occlusion_order_known=True,
    )
    assert result["force_balance_residual_n"] == pytest.approx(0.0)
    assert result["passed"] is False
    assert "force_closure_certificate_failed" in result["reasons"]


def test_stem_simulator_is_rooted_and_responds_to_contact() -> None:
    contract = StemRodContract(
        instance_id="stem-1",
        coordinate_frame="camera:metric",
        node_count=5,
        root_node=0,
        linear_density_kg_m=0.01,
        axial_stiffness_n_m=4.0,
        bending_stiffness_n_m=0.2,
        damping_n_s_m=0.08,
    )
    rest = np.stack((np.zeros(5), np.linspace(0.0, 0.2, 5), np.zeros(5)), axis=1)
    targets = np.repeat(rest[-1][None, :], 12, axis=0)
    targets[3:, 0] += 0.02
    result = simulate_damped_stem(
        np,
        rest_nodes_m=rest,
        contact_targets_m=targets,
        contact_active=np.asarray([False] * 3 + [True] * 9),
        contact_node=4,
        contract=contract,
        fps=24.0,
    )
    assert result["passed"] is True
    assert result["maximum_root_error_m"] == pytest.approx(0.0)
    assert result["nodes_m"][-1, -1, 0] > 0.0


def test_causal_motion_detects_frozen_grasp_and_attack() -> None:
    grasp = np.asarray([False, True, True, True, True, False])
    hand = np.asarray([0.0, 2.0, 2.0, 2.0, 2.0, 0.0])
    good = causal_motion_audit(
        np,
        grasp_active=grasp,
        hand_speed=hand,
        stem_speed=np.asarray([0.0, 0.0, 1.0, 1.0, 1.0, 0.0]),
        hand_motion_floor=1.0,
        stem_motion_floor=0.5,
        maximum_response_lag_frames=1,
        maximum_frozen_run_frames=0,
    )
    attacked = causal_motion_audit(
        np,
        grasp_active=grasp,
        hand_speed=hand,
        stem_speed=np.zeros(6),
        hand_motion_floor=1.0,
        stem_motion_floor=0.5,
        maximum_response_lag_frames=1,
        maximum_frozen_run_frames=0,
    )
    assert good["passed"] is True
    assert attacked["passed"] is False
    assert attacked["maximum_frozen_run_frames"] == 4


def test_inverse_rod_dynamics_exports_force_uncertainty_and_residual() -> None:
    contract = StemRodContract(
        instance_id="stem-1",
        coordinate_frame="world:test",
        node_count=4,
        root_node=0,
        linear_density_kg_m=0.01,
        axial_stiffness_n_m=2.0,
        bending_stiffness_n_m=0.1,
        damping_n_s_m=0.04,
    )
    rest = np.stack((np.zeros(4), np.linspace(0.0, 0.3, 4), np.zeros(4)), axis=1)
    targets = np.repeat(rest[-1][None], 8, axis=0)
    targets[2:, 0] += np.linspace(0.0, 0.01, 6)
    simulated = simulate_damped_stem(
        np,
        rest_nodes_m=rest,
        contact_targets_m=targets,
        contact_active=np.asarray([False, False, True, True, True, True, True, True]),
        contact_node=3,
        contract=contract,
        fps=24.0,
    )
    inverse = infer_stem_contact_forces(
        np,
        nodes_m=simulated["nodes_m"],
        position_sigma_m=np.full((8, 4), 0.0005),
        contact_nodes=np.full(8, 3),
        contact_active=np.asarray([False, False, True, True, True, True, True, True]),
        contract=contract,
        fps=24.0,
    )
    assert inverse["finite"] is True
    assert inverse["hand_on_stem_forces_n"].shape == (8, 1, 3)
    assert inverse["force_covariance_n2"].shape == (8, 1, 3, 3)
    assert inverse["unexplained_force_residual_n"].shape == (8,)
    assert np.linalg.norm(inverse["hand_on_stem_forces_n"][3:]) > 0
