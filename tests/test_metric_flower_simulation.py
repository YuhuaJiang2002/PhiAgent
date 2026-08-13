from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from phiagent.rendering.contact_dynamics import (
    InteractionFrameContract,
    MetricContactContract,
    assess_metric_force_closure,
    couple_contact_patch_to_required_wrench,
)
from phiagent.rendering.metric_flower_simulation import (
    MetricFlowerSimulationContract,
    articulated_hand_points,
    build_metric_flower_schedule,
    camera_calibration_from_mujoco_scene,
    exact_pad_stem_contact_state,
    project_world_points,
)
from scripts.generate_metric_flower_simulation import _transport_grasp_quaternion


def _rest_stem(nodes: int) -> np.ndarray:
    return np.stack(
        (
            np.full(nodes, 0.32),
            np.full(nodes, -0.18),
            np.linspace(0.55, 1.15, nodes),
        ),
        axis=1,
    )


def test_metric_flower_schedule_has_one_bounded_contact_interval() -> None:
    contract = MetricFlowerSimulationContract(
        frames=33,
        fps=24.0,
        nodes_per_stem=8,
        contact_node=5,
        approach_end_frame=9,
        release_frame=25,
    )
    schedule = build_metric_flower_schedule(
        np,
        rest_nodes_m=_rest_stem(contract.nodes_per_stem),
        contract=contract,
    )

    assert schedule["contact_active"].shape == (33,)
    assert np.flatnonzero(schedule["contact_active"]).tolist() == list(range(9, 25))
    assert schedule["right_hand_closure"][9] == pytest.approx(1.0)
    assert schedule["right_hand_closure"][24] == pytest.approx(1.0)
    assert schedule["right_hand_closure"][-1] == pytest.approx(0.0)
    assert set(schedule["phases"]) == {
        "approach",
        "grasp",
        "manipulate",
        "release",
        "retract",
    }
    np.testing.assert_allclose(
        schedule["right_wrist_targets_m"][9]
        + schedule["right_pad_offset_robot_base_m"],
        schedule["contact_targets_m"][9],
    )


def test_articulated_hand_points_close_without_changing_topology() -> None:
    opened = articulated_hand_points(np, 0.0)
    closed = articulated_hand_points(np, 1.0)

    assert opened.shape == closed.shape == (21, 3)
    assert np.all(np.isfinite(closed))
    assert np.linalg.norm(closed[8] - closed[5]) < np.linalg.norm(
        opened[8] - opened[5]
    )


def test_scene_camera_calibration_round_trips_world_points() -> None:
    scene_camera = [
        SimpleNamespace(
            pos=np.asarray((0.0, 0.0, 0.0)),
            forward=np.asarray((0.0, 0.0, 1.0)),
            up=np.asarray((0.0, 1.0, 0.0)),
        ),
        SimpleNamespace(
            pos=np.asarray((0.0, 0.0, 0.0)),
            forward=np.asarray((0.0, 0.0, 1.0)),
            up=np.asarray((0.0, 1.0, 0.0)),
        ),
    ]
    intrinsics, world_from_camera = camera_calibration_from_mujoco_scene(
        np,
        scene_camera=scene_camera,
        width=640,
        height=480,
        vertical_fov_degrees=60.0,
    )
    pixels, depth = project_world_points(
        np,
        points_world_m=np.asarray(((0.0, 0.0, 2.0), (0.2, -0.1, 2.0))),
        intrinsics_px=intrinsics,
        world_from_camera=world_from_camera,
    )

    assert pixels[0] == pytest.approx((319.5, 239.5))
    assert pixels[1, 0] < pixels[0, 0]
    assert pixels[1, 1] > pixels[0, 1]
    assert depth == pytest.approx((2.0, 2.0))


def test_exact_three_pad_state_passes_metric_force_closure() -> None:
    center = np.asarray((0.3, -0.1, 0.9))
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    vertices = np.stack(
        (
            center[0] + 0.0022 * np.cos(angles),
            center[1] + 0.0022 * np.sin(angles),
            center[2] + np.asarray((-0.006, 0.006, -0.006, 0.006, -0.006, 0.006)),
        ),
        axis=1,
    )
    state = exact_pad_stem_contact_state(
        np,
        pad_vertices_by_fingertip={
            fingertip: vertices[start : start + 2]
            for fingertip, start in zip((4, 7, 10), (0, 2, 4))
        },
        stem_nodes_m=np.asarray(
            (
                (0.3, -0.1, 0.85),
                (0.3, -0.1, 0.90),
                (0.3, -0.1, 0.95),
            )
        ),
    )
    state.update(
        couple_contact_patch_to_required_wrench(
            np,
            contact_points_m=state["contact_points_m"],
            contact_normals=state["contact_normals"],
            fingertip_indices=state["fingertip_indices"],
            object_center_m=state["object_center_m"],
            required_force_n=np.asarray((0.01, -0.005, 0.002)),
            required_moment_nm=np.zeros(3),
        )
    )
    frame = InteractionFrameContract(
        camera_frame="camera:simulated_rgbd",
        metric_frame="robot_base:simulated",
        timeline="frame:simulation",
        fps=24.0,
        fx_pixels=500.0,
        fy_pixels=500.0,
        cx_pixels=320.0,
        cy_pixels=240.0,
        metric_scale_source="calibrated-simulator-camera",
    )

    result = assess_metric_force_closure(
        np,
        **{
            name: state[name]
            for name in (
                "contact_points_m",
                "surface_gaps_m",
                "contact_normals",
                "contact_forces_n",
                "object_center_m",
                "external_force_n",
                "external_moment_nm",
                "fingertip_indices",
            )
        },
        frame_contract=frame,
        contact_contract=MetricContactContract(),
        depth_source="mujoco-depth-buffer",
        force_source="metric-articulated-rod-residual-v1",
        occlusion_order_known=True,
    )

    assert result["passed"] is True
    assert set(state["fingertip_indices"]) == {4, 7, 10}
    assert state["contacting_fingertips"] == 3
    assert state["contact_patch_points"] == 6
    assert state["coupled_force_residual_n"] < 1e-6
    assert state["coupled_moment_residual_nm"] < 1e-8
    assert result["force_closure"]["linearized_grasp_matrix_rank"] == 6


def test_grasp_orientation_parallel_transports_with_stem_tangent() -> None:
    transported = _transport_grasp_quaternion(
        np,
        initial_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
        initial_stem_tangent=np.asarray((0.0, 0.0, 1.0)),
        current_stem_tangent=np.asarray((1.0, 0.0, 0.0)),
    )
    vector = np.asarray((0.0, 0.0, 1.0))
    quaternion_vector = transported[1:]
    rotated = (
        vector
        + 2.0 * transported[0] * np.cross(quaternion_vector, vector)
        + 2.0
        * np.cross(
            quaternion_vector,
            np.cross(quaternion_vector, vector),
        )
    )

    np.testing.assert_allclose(
        rotated,
        (1.0, 0.0, 0.0),
        atol=1e-9,
    )
