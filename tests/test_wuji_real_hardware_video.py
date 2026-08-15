from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_reference_filter_is_exact_crop_without_generation() -> None:
    module = load_script("prepare_wuji_real_hardware_reference")
    assert module.extract_reference_filter(
        frame_index=30,
        crop=(640, 360, 0, 390),
        output_size=(640, 352),
    ) == "select=eq(n\\,30),crop=640:360:0:390,scale=640:352:flags=lanczos"


def test_crop_parser_rejects_implicit_or_negative_coordinates() -> None:
    module = load_script("prepare_wuji_real_hardware_reference")
    assert module.parse_crop("640:360:0:390") == (640, 360, 0, 390)
    for invalid in ("640:360", "640:360:-1:0", "0:360:0:0"):
        try:
            module.parse_crop(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid crop was accepted: {invalid}")


def test_real_reference_is_bound_to_the_declared_source_camera_support() -> None:
    module = load_script("prepare_wuji_real_hardware_reference")
    left, top, width, height = module.SOURCE_SCENE_REFERENCE_PLACEMENT
    placed = (left, top, left + width, top + height)
    assert module.box_iou(placed, module.SOURCE_FRAME_ZERO_HAND_ENVELOPE) > 0.75
    assert module.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_30_to_24_fps_mask_mapping_is_time_aligned_and_bounded() -> None:
    module = load_script("stitch_wuji_real_hardware_video")
    observed = [
        module.source_frame_for_output(
            index, output_fps=24.0, source_fps=30.0, source_frames=621
        )
        for index in (0, 1, 24, 496, 1000)
    ]
    assert observed == [0, 1, 30, 620, 620]


def test_alpha_composite_preserves_every_zero_alpha_pixel_exactly() -> None:
    module = load_script("stitch_wuji_real_hardware_video")
    foreground = np.full((5, 7, 3), 240, dtype=np.uint8)
    background = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    alpha = np.zeros((5, 7), dtype=np.float32)
    alpha[2, 3] = 1.0
    result = module.composite_under_alpha(np, foreground, background, alpha)
    assert np.array_equal(result[alpha == 0], background[alpha == 0])
    assert np.array_equal(result[2, 3], foreground[2, 3])


def test_overlap_stitch_covers_the_timeline_without_frame_loss() -> None:
    module = load_script("stitch_wuji_real_hardware_video")
    first = [np.full((4, 4, 3), index, dtype=np.uint8) for index in range(5)]
    second = [np.full((4, 4, 3), 3 + index, dtype=np.uint8) for index in range(5)]
    merged, seams = module.merge_windows(np, [(0, first), (3, second)], blend_radius=0)
    assert len(merged) == 8
    assert len(seams) == 1
    assert 4 <= seams[0]["seam_frame"] <= 4


def test_real_appearance_hand_is_filled_and_affine_motion_is_not_degenerate() -> None:
    import cv2

    module = load_script("build_wuji_real_hardware_appearance_comparison")
    points = np.asarray(
        [
            (100, 180),
            (118, 165), (132, 150), (145, 135), (158, 120),
            (82, 150), (80, 120), (78, 90), (76, 60),
            (100, 145), (100, 110), (100, 75), (100, 40),
            (118, 150), (123, 120), (128, 90), (133, 62),
            (135, 160), (147, 137), (158, 115), (168, 95),
        ],
        dtype=np.float32,
    )
    mask = module.hand_silhouette(cv2, np, points, (220, 240))
    assert int(np.count_nonzero(mask >= 128)) > 3_000
    for tip in (4, 8, 12, 16, 20):
        x, y = np.rint(points[tip]).astype(int)
        assert mask[y, x] >= 128
    target = points + np.asarray([17.0, -9.0], dtype=np.float32)
    matrix = module.affine_from_landmarks(cv2, points, target)
    mapped = matrix[:, :2] @ points[0] + matrix[:, 2]
    assert np.allclose(mapped, target[0], atol=1e-4)


def test_real_appearance_scene_lock_preserves_zero_alpha_pixels() -> None:
    module = load_script("build_wuji_real_hardware_appearance_comparison")
    background = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    foreground = np.full_like(background, 220)
    alpha = np.zeros((8, 9), dtype=np.uint8)
    alpha[3:5, 4:6] = 255
    result = module.composite(np, background, foreground, alpha)
    assert np.array_equal(result[alpha == 0], background[alpha == 0])
    assert np.array_equal(result[alpha == 255], foreground[alpha == 255])
