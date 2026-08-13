from __future__ import annotations

from scripts.build_rigid_part_robot_replacement import (
    _anisotropic_segment_transform,
    _fill_missing,
    _fixed_scale_anchor_transform,
    _fixed_scale_hand_transform,
    _flower_restore_mask,
    _load_reusable_pose_trajectory,
    _piece_mask_overlap_metrics,
    _robust_pose_filter,
    _robot_rig_reference,
    _series_statistics,
    _similarity,
    _stable_segment_angles,
    _trajectory_correspondence_metrics,
    _zero_phase_bounded_steps,
    _zero_phase_bounded_vector_steps,
)


def test_fixed_scale_anchor_transform_maps_grip_without_size_breathing() -> None:
    import numpy as np

    source_anchor = np.asarray((100.0, 200.0))
    target_anchor = np.asarray((640.0, 360.0))
    transform = _fixed_scale_anchor_transform(
        np,
        source_anchor,
        target_anchor,
        scale=0.25,
        angle_degrees=0.0,
    )

    mapped = transform[:, :2] @ source_anchor + transform[:, 2]
    assert np.allclose(mapped, target_anchor)
    assert np.hypot(transform[0, 0], transform[0, 1]) == 0.25


def test_zero_phase_vector_bound_removes_spatial_jump_without_axis_bias() -> None:
    import numpy as np

    source = np.asarray(((0.0, 0.0), (1.0, 1.0), (30.0, 30.0), (31.0, 31.0)))
    bounded = _zero_phase_bounded_vector_steps(np, source, maximum_step=5.0)

    steps = np.linalg.norm(np.diff(bounded, axis=0), axis=1)
    assert np.max(steps) <= 5.0 + 1e-9
    assert np.allclose(bounded[:, 0], bounded[:, 1])


def test_flower_restore_mask_keeps_flowers_and_rejects_skin() -> None:
    import cv2
    import numpy as np

    source = np.zeros((16, 16, 3), dtype=np.uint8)
    source[2:6, 2:6] = (30, 180, 30)
    source[9:13, 9:13] = (80, 120, 180)
    instance = np.zeros((16, 16), dtype=np.uint8)
    instance[2:6, 2:6] = 1
    instance[9:13, 9:13] = 1
    safety = np.ones((16, 16), dtype=np.uint8) * 255
    changed = np.ones((16, 16), dtype=np.uint8) * 255

    restore, metrics = _flower_restore_mask(
        cv2,
        np,
        source_frame=source,
        instance_mask=instance,
        safety_mask=safety,
        changed_support=changed,
        skin_dilation_pixels=0,
    )

    assert np.all(restore[2:6, 2:6])
    assert not np.any(restore[9:13, 9:13])
    assert metrics["restore_pixels"] == 16
    assert metrics["restore_skin_overlap_pixels"] == 0


def test_similarity_maps_segment_endpoints() -> None:
    import numpy as np

    source_a = np.asarray((10.0, 20.0))
    source_b = np.asarray((30.0, 20.0))
    target_a = np.asarray((15.0, 35.0))
    target_b = np.asarray((15.0, 55.0))

    transform = _similarity(np, source_a, source_b, target_a, target_b)

    mapped_a = transform[:, :2] @ source_a + transform[:, 2]
    mapped_b = transform[:, :2] @ source_b + transform[:, 2]
    assert np.allclose(mapped_a, target_a)
    assert np.allclose(mapped_b, target_b)


def test_fill_missing_interpolates_and_smooths_tracks() -> None:
    import numpy as np

    first = np.zeros((33, 2), dtype=np.float64)
    last = np.full((33, 2), 10.0, dtype=np.float64)

    tracks = _fill_missing(np, [first, None, last])

    assert tracks.shape == (3, 33, 2)
    assert np.all(np.isfinite(tracks))
    assert np.all(tracks[0] < tracks[-1])


def test_reusable_pose_trajectory_preserves_selected_tracks(tmp_path) -> None:
    import json

    import numpy as np

    indices = [11, 12, 13, 14, 15, 16, 19, 20]
    selected = np.arange(3 * len(indices) * 2, dtype=np.float64).reshape(
        3, len(indices), 2
    )
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "coordinate_frame": "camera:source_pixels",
                "frame_count": 3,
                "landmark_indices": indices,
                "robot_target_xy": selected.tolist(),
                "raw_interpolated_xy": selected.tolist(),
                "robust_median_xy": selected.tolist(),
                "temporal_index_map": [0, 1, 2],
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "metrics": {
                    "decoded_frames": 3,
                    "missing_pose_frames": 0,
                    "low_confidence_frames": 0,
                    "smoothing": {},
                    "outlier_repair": {},
                    "correspondence": {},
                }
            }
        )
    )

    tracks, tracking, raw, robust = _load_reusable_pose_trajectory(
        np, trajectory, manifest, expected_frames=3
    )

    assert np.array_equal(tracks[:, indices], selected)
    assert np.array_equal(raw[:, indices], selected)
    assert np.array_equal(robust[:, indices], selected)
    assert np.all(tracks[:, 23:25, 1] > tracks[:, 11:13, 1])
    assert tracking["trajectory_reuse"]["used"]


def test_exact_similarity_maps_unclipped_segment_length() -> None:
    import numpy as np

    source_a = np.asarray((0.0, 0.0))
    source_b = np.asarray((10.0, 0.0))
    target_a = np.asarray((5.0, 7.0))
    target_b = np.asarray((35.0, 7.0))

    transform = _similarity(
        np,
        source_a,
        source_b,
        target_a,
        target_b,
        exact_scale=True,
    )

    assert np.allclose(transform[:, :2] @ source_a + transform[:, 2], target_a)
    assert np.allclose(transform[:, :2] @ source_b + transform[:, 2], target_b)


def test_anisotropic_transform_maps_joints_and_preserves_thickness_scale() -> None:
    import numpy as np

    source_a = np.asarray((10.0, 10.0))
    source_b = np.asarray((30.0, 10.0))
    target_a = np.asarray((50.0, 60.0))
    target_b = np.asarray((50.0, 100.0))
    transform = _anisotropic_segment_transform(
        np,
        source_a,
        source_b,
        target_a,
        target_b,
        transverse_scale=0.75,
    )

    assert np.allclose(transform[:, :2] @ source_a + transform[:, 2], target_a)
    assert np.allclose(transform[:, :2] @ source_b + transform[:, 2], target_b)
    mapped_transverse = transform[:, :2] @ np.asarray((0.0, 1.0))
    assert np.linalg.norm(mapped_transverse) == 0.75


def test_piece_mask_overlap_metrics_reject_cross_side_duplication() -> None:
    import numpy as np

    pieces = {
        "left_hand": np.asarray(((255, 0), (0, 0)), dtype=np.uint8),
        "left_lower": np.asarray(((255, 0), (0, 0)), dtype=np.uint8),
        "right_hand": np.asarray(((0, 0), (0, 255)), dtype=np.uint8),
    }
    clean = _piece_mask_overlap_metrics(np, pieces)
    pieces["right_hand"][0, 0] = 255
    contaminated = _piece_mask_overlap_metrics(np, pieces)

    assert clean["pairwise_overlap_pixels"] == 1
    assert clean["cross_side_overlap_pixels"] == 0
    assert contaminated["cross_side_overlap_pixels"] == 2


def test_robot_rig_reference_uses_explicit_coordinate_frame() -> None:
    import numpy as np

    payload = {
        "robot_rig_reference_xy": {
            "left": {
                "shoulder": [1, 2],
                "elbow": [3, 4],
                "wrist": [5, 6],
                "hand": [7, 8],
            },
            "right": {
                "shoulder": [9, 10],
                "elbow": [11, 12],
                "wrist": [13, 14],
                "hand": [15, 16],
            },
        }
    }

    reference = _robot_rig_reference(np, payload, np.zeros((33, 2)))

    assert np.array_equal(reference[11], (1, 2))
    assert np.array_equal(reference[20], (15, 16))


def test_fixed_scale_hand_transform_maps_wrist_without_scale_breathing() -> None:
    import numpy as np

    source_wrist = np.asarray((10.0, 20.0))
    source_hand = np.asarray((17.0, 20.0))
    target_wrist = np.asarray((80.0, 40.0))
    transform = _fixed_scale_hand_transform(
        np,
        source_wrist,
        source_hand,
        target_wrist,
        np.pi / 2.0,
    )

    mapped_wrist = transform[:, :2] @ source_wrist + transform[:, 2]
    mapped_hand = transform[:, :2] @ source_hand + transform[:, 2]
    assert np.allclose(mapped_wrist, target_wrist)
    assert np.allclose(mapped_hand, (80.0, 47.0))
    assert np.hypot(transform[0, 0], transform[0, 1]) == 1.0


def test_stable_segment_angles_bridge_degenerate_direction_without_flip() -> None:
    import numpy as np

    starts = np.zeros((21, 2), dtype=np.float64)
    angles = np.linspace(0.0, 0.4, 21)
    lengths = np.full(21, 12.0)
    lengths[8:13] = 0.1
    ends = np.column_stack((np.cos(angles), np.sin(angles))) * lengths[:, None]
    ends[10] = (-0.1, 0.0)

    stable, record = _stable_segment_angles(
        np,
        starts,
        ends,
        minimum_length_pixels=8.0,
        median_radius=2,
        smoothing_sigma=1.5,
    )

    assert record["interpolated_frames"] == 5
    assert np.max(np.abs(np.diff(stable))) < 0.05
    assert stable[-1] > stable[0]


def test_series_statistics_exposes_scale_breathing() -> None:
    import numpy as np

    stable = _series_statistics(np, [1.0] * 20)
    breathing = _series_statistics(np, [1.0] * 18 + [2.0, 3.0])

    assert stable["p99_to_p01_ratio"] == 1.0
    assert stable["maximum_frame_step"] == 0.0
    assert breathing["p99_to_p01_ratio"] > 2.0
    assert breathing["maximum_frame_step"] == 1.0


def test_zero_phase_bounded_steps_limits_jump_without_directional_endpoint_bias() -> None:
    import numpy as np

    source = np.asarray((0.0, 0.1, 4.0, 4.1, 4.2), dtype=np.float64)
    bounded = _zero_phase_bounded_steps(np, source, maximum_step=0.25)

    assert np.max(np.abs(np.diff(bounded))) <= 0.25 + 1e-12
    assert bounded[2] == 2.175


def test_correspondence_metrics_preserve_timeline_and_reduce_jerk() -> None:
    import numpy as np

    raw = np.zeros((24, 33, 2), dtype=np.float64)
    raw[:, :, 0] = np.arange(24, dtype=np.float64)[:, None]
    raw[12, :, 0] += 3.0
    smooth = _fill_missing(np, [frame for frame in raw], smoothing_sigma=2.0)

    metrics = _trajectory_correspondence_metrics(
        np,
        raw,
        smooth,
        width=128,
        height=72,
        smoothing_sigma=2.0,
    )

    assert metrics["temporal_index_map_mismatch_frames"] == 0
    assert (
        metrics["maximum_smoothed_jerk_fraction_of_diagonal"]
        < metrics["maximum_raw_jerk_fraction_of_diagonal"]
    )


def test_robust_pose_filter_removes_single_frame_landmark_jump() -> None:
    import numpy as np

    raw = np.zeros((31, 33, 2), dtype=np.float64)
    raw[:, :, 0] = np.arange(31, dtype=np.float64)[:, None]
    raw[15, 19] = (300.0, -250.0)
    original = raw.copy()

    smoothed, robust, outliers = _robust_pose_filter(
        np,
        raw,
        median_radius=3,
        smoothing_sigma=2.0,
        outlier_threshold_pixels=20.0,
    )

    assert outliers[15, 19]
    assert np.array_equal(raw, original)
    assert np.linalg.norm(robust[15, 19] - (15.0, 0.0)) <= 1.0
    assert np.linalg.norm(smoothed[15, 19] - (15.0, 0.0)) <= 1.0
