from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.build_viewpoint_generalization_demo import (
    _homography_points,
    _parse_labeled_paths,
    _roundtrip_metrics,
    _warp,
)


def test_homography_names_camera_direction_explicitly() -> None:
    _src, left = _homography_points(np, 832, 480, -12.0)
    _src, right = _homography_points(np, 832, 480, 12.0)

    assert left[1, 0] < right[1, 0]
    assert left[1, 1] > right[1, 1]
    assert right[0, 0] > left[0, 0]
    assert right[0, 1] > left[0, 1]


def test_roundtrip_identity_metric_passes_for_bounded_warp() -> None:
    x = np.linspace(0, 255, 320, dtype=np.uint8)
    y = np.linspace(0, 255, 180, dtype=np.uint8)[:, None]
    frame = np.dstack(
        (
            np.broadcast_to(x, (180, 320)),
            np.broadcast_to(y, (180, 320)),
            np.full((180, 320), 127, dtype=np.uint8),
        )
    )
    warped, matrix = _warp(cv2, np, frame, 12.0)
    metrics = _roundtrip_metrics(cv2, np, frame, warped, matrix)

    assert metrics["roundtrip_mae"] < 2.0
    assert metrics["roundtrip_psnr_db"] > 35.0
    assert metrics["luma_similarity"] > 0.99


def test_parse_labeled_paths_requires_three_real_inputs(tmp_path: Path) -> None:
    paths = []
    for label in ("handover", "unscrew", "rinse"):
        path = tmp_path / f"{label}.mp4"
        path.write_bytes(b"video")
        paths.append(f"{label}={path}")

    parsed = _parse_labeled_paths(paths)
    assert set(parsed) == {"handover", "unscrew", "rinse"}

    with pytest.raises(ValueError, match="exactly three"):
        _parse_labeled_paths(paths[:2])
