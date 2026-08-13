from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from phiagent.perception.exact_asset_trajectory import (
    ExactAssetTrajectoryContract,
    _project_points,
    _rodrigues,
    fit_articulated_keypoints_frame,
    validate_exact_asset_trajectory,
)
from scripts.compile_foundation_contact_pipeline import _robot_report
from scripts.fit_foundation_exact_asset_trajectory import _joint_schema


def _forward_articulated(q: np.ndarray) -> np.ndarray:
    fixed = np.asarray(
        [
            [-0.18, -0.22, -0.10],
            [-0.18, -0.22, 0.10],
            [-0.18, 0.22, -0.10],
            [-0.18, 0.22, 0.10],
            [0.18, -0.22, -0.10],
            [0.18, -0.22, 0.10],
            [0.18, 0.22, -0.10],
            [0.18, 0.22, 0.10],
        ],
        dtype=np.float64,
    )
    shoulder = np.asarray([0.16, -0.18, 0.06], dtype=np.float64)
    first_offsets = np.asarray(
        [[0.08, 0.00, 0.00], [0.14, 0.02, 0.01], [0.20, -0.02, -0.01]],
        dtype=np.float64,
    )
    first_rotation = _rodrigues(np, np.asarray([0.0, q[0], 0.0]))
    first = shoulder[None, :] + (first_rotation @ first_offsets.T).T
    elbow = first[-1]
    second_offsets = np.asarray(
        [[0.06, 0.01, 0.00], [0.12, -0.02, 0.02], [0.17, 0.02, -0.01]],
        dtype=np.float64,
    )
    second_rotation = _rodrigues(
        np, np.asarray([q[1], q[0] + 0.35 * q[1], 0.0])
    )
    second = elbow[None, :] + (second_rotation @ second_offsets.T).T
    return np.concatenate((fixed, first, second), axis=0)


def _contract(*, minimum_keypoints: int = 12) -> ExactAssetTrajectoryContract:
    digest = "a" * 64
    return ExactAssetTrajectoryContract(
        embodiment_id="unitree-g1-sharpa",
        camera_frame="camera:generated_video",
        robot_base_frame="robot_base:g1",
        timeline="frame:source_video",
        source_video_sha256="b" * 64,
        fps=24.0,
        joint_names=("shoulder_pitch", "elbow_pitch"),
        joint_limits_rad=((-1.5, 1.5), (-1.5, 1.5)),
        asset_sha256={"g1_model": digest},
        expected_asset_sha256={"g1_model": digest},
        minimum_visible_keypoints_per_frame=minimum_keypoints,
    )


def _passing_validation_inputs() -> dict[str, object]:
    frames = np.arange(24, dtype=np.int64) * 3
    q = np.zeros((24, 2), dtype=np.float64)
    q[:, 0] = np.linspace(0.0, 0.20, 24)
    q[:, 1] = np.linspace(0.0, -0.15, 24)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 24, axis=0)
    poses[:, 2, 3] = 2.4
    observed = np.zeros((24, 12, 2), dtype=np.float64)
    observed[:, :, 0] = np.linspace(120.0, 420.0, 12)[None, :]
    observed[:, :, 1] = np.linspace(80.0, 330.0, 12)[None, :]
    rendered = observed + 0.25
    fit_mask = np.ones(24, dtype=bool)
    fit_mask[16:] = False
    groups = tuple(
        "fit" if index < 16 else ("late-a" if index % 2 else "late-b")
        for index in range(24)
    )
    return {
        "contract": _contract(),
        "evidence_source_video_sha256": "b" * 64,
        "frame_indices": frames,
        "joint_positions_rad": q,
        "camera_from_robot_base": poses,
        "observed_keypoints_px": observed,
        "rendered_keypoints_px": rendered,
        "keypoint_confidence": np.ones((24, 12), dtype=np.float64),
        "fit_frame_mask": fit_mask,
        "heldout_group_ids": groups,
        "silhouette_iou": np.full(24, 0.82, dtype=np.float64),
        "joint_standard_deviation_rad": np.full((24, 2), 0.015, dtype=np.float64),
        "base_translation_standard_deviation_m": np.full(
            (24, 3), 0.004, dtype=np.float64
        ),
        "alternative_asset_reprojection_rmse_px": np.full(
            (24, 2), 10.0, dtype=np.float64
        ),
    }


def test_exact_asset_fit_recovers_full_q_and_camera_pose() -> None:
    intrinsics = np.asarray(
        [[650.0, 0.0, 320.0], [0.0, 645.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    truth_q = np.asarray([0.36, -0.44], dtype=np.float64)
    truth_rotation = np.asarray([0.08, -0.11, 0.035], dtype=np.float64)
    truth_translation = np.asarray([0.04, -0.025, 2.45], dtype=np.float64)
    observed, _ = _project_points(
        np,
        points_robot_base_m=_forward_articulated(truth_q),
        intrinsics_px=intrinsics,
        rotation_vector=truth_rotation,
        translation_m=truth_translation,
    )

    result = fit_articulated_keypoints_frame(
        np,
        forward_keypoints_robot_base_m=_forward_articulated,
        intrinsics_px=intrinsics,
        observed_keypoints_px=observed,
        keypoint_confidence=np.ones(len(observed), dtype=np.float64),
        initial_joint_positions_rad=np.asarray([0.25, -0.30]),
        joint_limits_rad=np.asarray([[-1.5, 1.5], [-1.5, 1.5]]),
        initial_rotation_vector=np.asarray([0.04, -0.07, 0.02]),
        initial_translation_m=np.asarray([0.02, -0.01, 2.30]),
        maximum_iterations=80,
    )

    assert result["identifiable"] is True
    assert result["reprojection_rmse_px"] < 0.05
    np.testing.assert_allclose(result["joint_positions_rad"], truth_q, atol=2e-3)
    np.testing.assert_allclose(
        result["camera_from_robot_base"][:3, 3], truth_translation, atol=2e-3
    )


def test_exact_asset_fit_marks_unseen_joints_unidentifiable() -> None:
    fixed = _forward_articulated(np.zeros(2))
    intrinsics = np.asarray(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    )
    observed, _ = _project_points(
        np,
        points_robot_base_m=fixed,
        intrinsics_px=intrinsics,
        rotation_vector=np.zeros(3),
        translation_m=np.asarray([0.0, 0.0, 2.5]),
    )

    result = fit_articulated_keypoints_frame(
        np,
        forward_keypoints_robot_base_m=lambda q: fixed,
        intrinsics_px=intrinsics,
        observed_keypoints_px=observed,
        keypoint_confidence=np.ones(len(observed)),
        initial_joint_positions_rad=np.zeros(2),
        joint_limits_rad=np.asarray([[-1.5, 1.5], [-1.5, 1.5]]),
        initial_rotation_vector=np.zeros(3),
        initial_translation_m=np.asarray([0.0, 0.0, 2.5]),
    )

    assert result["identifiable"] is False
    assert np.all(np.isinf(result["parameter_standard_deviation"][6:]))


def test_exact_asset_validator_accepts_only_complete_heldout_evidence() -> None:
    result = validate_exact_asset_trajectory(np, **_passing_validation_inputs())

    assert result["passed"] is True
    assert result["proposal_passed"] is True
    assert result["heldout_groups"] == ["late-a", "late-b"]


@pytest.mark.parametrize(
    ("attack", "expected_reason"),
    [
        ("wrong_hash", "exact_asset_hashes_match_registry"),
        ("unobservable_q", "full_q_posterior_observable"),
        ("identity_ambiguous", "selected_asset_beats_alternatives"),
        ("no_heldout_groups", "heldout_group_count"),
    ],
)
def test_exact_asset_validator_rejects_adversarial_shortcuts(
    attack: str, expected_reason: str
) -> None:
    inputs = _passing_validation_inputs()
    if attack == "wrong_hash":
        inputs["contract"] = replace(
            inputs["contract"], asset_sha256={"g1_model": "c" * 64}
        )
    elif attack == "unobservable_q":
        inputs["joint_standard_deviation_rad"] = np.full((24, 2), np.inf)
    elif attack == "identity_ambiguous":
        inputs["alternative_asset_reprojection_rmse_px"] = np.full(
            (24, 1), 1.0
        )
    elif attack == "no_heldout_groups":
        inputs["heldout_group_ids"] = tuple("" for _ in range(24))

    result = validate_exact_asset_trajectory(np, **inputs)

    assert result["passed"] is False
    assert expected_reason in result["reasons"]


def test_exact_asset_validator_rejects_partial_q_shape() -> None:
    inputs = _passing_validation_inputs()
    inputs["joint_positions_rad"] = np.zeros((24, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="complete TxJ"):
        validate_exact_asset_trajectory(np, **inputs)


def test_joint_schema_is_derived_from_hashed_assets(tmp_path) -> None:
    first = tmp_path / "first.xml"
    first.write_text(
        '<mujoco><worldbody><body><joint name="a" range="-1 2"/>'
        '<joint name="b" range="0 3"/></body></worldbody></mujoco>'
    )
    second = tmp_path / "second.xml"
    second.write_text(
        '<mujoco><worldbody><body><joint name="c" range="-0.5 0.5"/>'
        "</body></worldbody></mujoco>"
    )

    names, limits, counts = _joint_schema({"first": first, "second": second})

    assert names == ("a", "b", "c")
    assert limits == ((-1.0, 2.0), (0.0, 3.0), (-0.5, 0.5))
    assert counts == {"first": 2, "second": 1}


def test_pipeline_compiler_preserves_failed_analysis_by_synthesis_reason() -> None:
    fit_report = {
        "validation": {
            "passed": False,
            "reasons": ["full_q_posterior_observable"],
        }
    }
    assets = {
        "g1_model": {"hash_matches_registry": True},
        "left": {"hash_matches_registry": True},
    }

    result = _robot_report(
        np,
        None,
        assets,
        24.0,
        fit_report=fit_report,
    )

    assert result["passed"] is False
    assert result["exact_asset_registry_passed"] is True
    assert "full_q_posterior_observable" in result["reasons"]
    assert result["analysis_by_synthesis"] == fit_report["validation"]
