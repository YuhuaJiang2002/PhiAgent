from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from phiagent.perception.adaptive_vda import (
    AdaptiveVDAConfig,
    apply_root_depth_correction,
    is_hard_sequence,
    map_relative_depth,
    robust_temporal_correction,
)


def test_frozen_v1_v2_router_uses_only_gt_free_scale_evidence() -> None:
    config = AdaptiveVDAConfig()
    assert is_hard_sequence(3, 0.01, config)
    assert is_hard_sequence(8, 0.21, config)
    assert not is_hard_sequence(4, 0.20, config)


def test_metric_mapping_supports_depth_and_inverse_depth() -> None:
    relative = np.asarray([[1.0, 2.0]], dtype=np.float32)
    direct = map_relative_depth(
        relative, {"kind": "direct_depth", "slope": 2.0, "intercept": 1.0}
    )
    inverse = map_relative_depth(
        relative, {"kind": "inverse_depth", "slope": 1.0, "intercept": 0.0}
    )
    assert np.allclose(direct, [[3.0, 5.0]])
    assert np.allclose(inverse, [[1.0, 0.5]])


def test_temporal_correction_removes_sequence_bias_and_respects_bound() -> None:
    config = AdaptiveVDAConfig()
    estimates = [
        (0, 1.00, {"patch_iqr_m": 0.001}),
        (2, 1.02, {"patch_iqr_m": 0.001}),
        (4, 0.98, {"patch_iqr_m": 0.002}),
        (6, 1.01, {"patch_iqr_m": 0.002}),
    ]
    correction, detail = robust_temporal_correction(7, estimates, config)
    assert detail["accepted"] is True
    assert correction.shape == (7,)
    assert np.max(np.abs(correction)) <= config.residual_clip_m
    assert abs(np.median(np.asarray([item[1] for item in estimates])) - detail["center_m"]) < 1e-12


def _payload() -> dict[str, np.ndarray]:
    frames = 3
    joints = np.zeros((frames, 21, 3), dtype=np.float32)
    joints[..., 2] = 1.0
    joints[..., 0] = np.linspace(-0.02, 0.02, 21)[None]
    vertices = np.zeros((frames, 8, 3), dtype=np.float32)
    vertices[..., 2] = 1.0
    return {
        "frame_index": np.arange(frames, dtype=np.int64),
        "transl": np.zeros((frames, 3), dtype=np.float32),
        "joints_3d_world": joints.copy(),
        "joints_3d_camera": joints.copy(),
        "vertices_world": vertices.copy(),
        "vertices_camera": vertices.copy(),
        "joints_2d": np.zeros((frames, 21, 2), dtype=np.float32),
        "joints_in_frame": np.ones((frames, 21), dtype=bool),
        "camera_R_c2w": np.repeat(np.eye(3, dtype=np.float32)[None], frames, axis=0),
        "camera_R_w2c": np.repeat(np.eye(3, dtype=np.float32)[None], frames, axis=0),
        "camera_t_c2w": np.zeros((frames, 3), dtype=np.float32),
        "camera_t_w2c": np.zeros((frames, 3), dtype=np.float32),
        "camera_intrinsics": np.asarray(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "image_size": np.asarray([640, 480], dtype=np.int64),
        "valid": np.ones(frames, dtype=bool),
        "hand_pose": np.zeros((frames, 45), dtype=np.float32),
    }


def test_root_depth_correction_preserves_camera_and_local_hand_geometry() -> None:
    payload = _payload()
    delta = np.asarray([0.0, 0.01, -0.01], dtype=np.float64)
    detector = {
        "penetration_detected": np.zeros(3, dtype=bool),
        "penetration_energy": np.zeros(3, dtype=np.float32),
        "max_nearest_vertex_depth_proxy_m": np.zeros(3, dtype=np.float32),
    }
    candidate = apply_root_depth_correction(
        payload, delta, detector, AdaptiveVDAConfig()
    )
    assert np.array_equal(candidate["camera_R_c2w"], payload["camera_R_c2w"])
    assert np.array_equal(candidate["camera_t_c2w"], payload["camera_t_c2w"])
    assert np.array_equal(candidate["hand_pose"], payload["hand_pose"])
    assert np.allclose(candidate["joints_3d_camera"][..., 2] - 1.0, delta[:, None])
    before_local = payload["joints_3d_world"] - payload["joints_3d_world"][:, :1]
    after_local = candidate["joints_3d_world"] - candidate["joints_3d_world"][:, :1]
    assert np.allclose(after_local, before_local, atol=1e-7)
    assert bool(candidate["adaptive_vda_candidate_generated_without_gt"])


def test_zero_root_depth_correction_is_an_exact_native_fallback() -> None:
    payload = _payload()
    payload["joints_2d"][:] = 123.0
    payload["joints_in_frame"][:] = False
    detector = {
        "penetration_detected": np.zeros(3, dtype=bool),
        "penetration_energy": np.zeros(3, dtype=np.float32),
        "max_nearest_vertex_depth_proxy_m": np.zeros(3, dtype=np.float32),
    }
    candidate = apply_root_depth_correction(
        payload, np.zeros(3, dtype=np.float64), detector, AdaptiveVDAConfig()
    )
    for key, value in payload.items():
        assert np.array_equal(candidate[key], value, equal_nan=True)


def test_importing_adaptive_vda_does_not_import_torch() -> None:
    probe = (
        "import json, sys; import phiagent.perception.adaptive_vda; "
        "print(json.dumps({'torch': 'torch' in sys.modules, 'cv2': 'cv2' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == {"torch": False, "cv2": False}
