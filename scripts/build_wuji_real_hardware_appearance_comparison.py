#!/usr/bin/env python3
"""Build a source-locked Wuji replacement from real hardware appearance pixels.

The black compliant-hand material comes from a hash-bound physical Wuji video.
Image-plane landmarks drive a filled five-finger glove instead of a simulation
mesh.  A separate fixed-width wrist/forearm layer is drawn behind the hand, and
the result is clipped to a declared edit mask over one immutable clean plate.
Heavy MediaPipe/OpenCV/SciPy imports stay behind the command-line entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
PALM_INDICES = (0, 1, 2, 5, 9, 13, 17)
AFFINE_INDICES = (0, 5, 17)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def affine_from_landmarks(cv2: Any, source: Any, target: Any) -> Any:
    """Return a non-degenerate palm affine transform in one camera frame."""

    if source.shape != (21, 2) or target.shape != (21, 2):
        raise ValueError("source and target landmarks must have shape (21, 2)")
    matrix = cv2.getAffineTransform(
        source[list(AFFINE_INDICES)].astype("float32"),
        target[list(AFFINE_INDICES)].astype("float32"),
    )
    determinant = float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    if abs(determinant) < 1e-5:
        raise RuntimeError("palm affine transform is degenerate")
    return matrix


def hand_silhouette(cv2: Any, np: Any, points: Any, shape: tuple[int, int]) -> Any:
    """Construct one complete, filled five-finger compliant-hand silhouette."""

    height, width = shape
    if points.shape != (21, 2):
        raise ValueError("points must have shape (21, 2)")
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    if palm_width < 20:
        raise RuntimeError("detected palm is implausibly small")
    mask = np.zeros((height, width), dtype=np.uint8)
    palm = cv2.convexHull(np.rint(points[list(PALM_INDICES)]).astype(np.int32))
    cv2.fillConvexPoly(mask, palm, 255, lineType=cv2.LINE_AA)
    # The physical Wuji covering is visibly broader and softer than a human
    # skeleton or an exposed-link simulation.  Keep every phalanx filled and
    # overlap adjacent segments so no kinematic rods or joint markers appear.
    widths = (0.24, 0.225, 0.205, 0.185)
    for chain in FINGER_CHAINS:
        for segment, (start, end) in enumerate(zip(chain[:-1], chain[1:])):
            thickness = max(16, int(round(widths[segment] * palm_width)))
            cv2.line(
                mask,
                tuple(np.rint(points[start]).astype(int)),
                tuple(np.rint(points[end]).astype(int)),
                255,
                thickness,
                cv2.LINE_AA,
            )
            cv2.circle(
                mask,
                tuple(np.rint(points[end]).astype(int)),
                max(6, thickness // 2),
                255,
                -1,
                cv2.LINE_AA,
            )
    wrist_radius = max(18, int(round(0.21 * palm_width)))
    cv2.circle(
        mask,
        tuple(np.rint(points[0]).astype(int)),
        wrist_radius,
        255,
        -1,
        cv2.LINE_AA,
    )
    return cv2.GaussianBlur(mask, (0, 0), 0.55)


def forearm_geometry(np: Any, wrist: Any, *, end_x: float, height: int) -> dict[str, Any]:
    """Return the fixed-width source-camera wrist/forearm geometry."""

    end = np.asarray([end_x, height + 45.0], dtype=np.float32)
    axis = end - wrist
    axis_norm = max(float(np.linalg.norm(axis)), 1.0)
    axis /= axis_norm
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    wrist_half, end_half = 25.0, 30.0
    top = wrist + axis * 19.0
    polygon = np.rint(
        np.stack(
            (
                top - normal * wrist_half,
                top + normal * wrist_half,
                end + normal * end_half,
                end - normal * end_half,
            )
        )
    ).astype(np.int32)
    return {
        "wrist": wrist,
        "end": end,
        "axis": axis,
        "normal": normal,
        "axis_norm": axis_norm,
        "wrist_half": wrist_half,
        "end_half": end_half,
        "top": top,
        "polygon": polygon,
    }


def render_forearm(
    cv2: Any,
    np: Any,
    shape: tuple[int, int],
    geometry: dict[str, Any],
) -> tuple[Any, Any]:
    """Render a photographic white shell with a black wrist bearing."""

    height, width = shape
    alpha = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(alpha, geometry["polygon"], 255, lineType=cv2.LINE_AA)
    yy, xx = np.mgrid[0:height, 0:width]
    wrist = geometry["wrist"]
    normal = geometry["normal"]
    lateral = (xx - wrist[0]) * normal[0] + (yy - wrist[1]) * normal[1]
    highlight = np.exp(-((lateral + 5.0) / 21.0) ** 2)
    edge_shadow = np.clip(np.abs(lateral) / 31.0, 0.0, 1.0)
    shell_value = 198.0 + 42.0 * highlight - 27.0 * edge_shadow
    material = np.empty((height, width, 3), dtype=np.uint8)
    material[..., 0] = np.clip(shell_value + 5.0, 0, 255).astype(np.uint8)
    material[..., 1] = np.clip(shell_value + 3.0, 0, 255).astype(np.uint8)
    material[..., 2] = np.clip(shell_value, 0, 255).astype(np.uint8)
    collar_center = tuple(np.rint(wrist + geometry["axis"] * 8.0).astype(int))
    collar_angle = float(np.degrees(np.arctan2(geometry["axis"][1], geometry["axis"][0])))
    cv2.ellipse(
        material,
        collar_center,
        (34, 25),
        collar_angle,
        0,
        360,
        (224, 226, 228),
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        material,
        collar_center,
        (27, 19),
        collar_angle,
        0,
        360,
        (28, 29, 31),
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(alpha, collar_center, (35, 26), collar_angle, 0, 360, 255, -1, cv2.LINE_AA)
    for sign in (-1.0, 1.0):
        bolt = wrist + geometry["axis"] * 8.0 + geometry["normal"] * sign * 25.0
        cv2.circle(material, tuple(np.rint(bolt).astype(int)), 3, (72, 74, 76), -1, cv2.LINE_AA)
    seam_center = wrist + geometry["axis"] * min(210.0, 0.43 * geometry["axis_norm"])
    seam_a = tuple(np.rint(seam_center - geometry["normal"] * 27.0).astype(int))
    seam_b = tuple(np.rint(seam_center + geometry["normal"] * 27.0).astype(int))
    cv2.line(material, seam_a, seam_b, (151, 156, 160), 2, cv2.LINE_AA)
    screw = seam_center + geometry["normal"] * 17.0
    cv2.circle(material, tuple(np.rint(screw).astype(int)), 2, (89, 91, 93), -1, cv2.LINE_AA)
    return material, cv2.GaussianBlur(alpha, (0, 0), 0.6)


def composite(np: Any, background: Any, foreground: Any, alpha: Any) -> Any:
    if background.shape != foreground.shape or alpha.shape != background.shape[:2]:
        raise ValueError("composite shapes do not match")
    weight = alpha.astype(np.float32)[..., None] / 255.0
    return np.clip(
        np.rint(background.astype(np.float32) * (1.0 - weight) + foreground * weight),
        0,
        255,
    ).astype(np.uint8)


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    info: dict[str, float | int] = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "reported_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    info["decoded_frames"] = len(frames)
    return frames, info


def _writer(ffmpeg: Path, path: Path, *, width: int, height: int, fps: float, crf: int) -> Any:
    return subprocess.Popen(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def encoded_exterior_audit(
    cv2: Any, np: Any, video: Path, masks: list[Any], clean_plate: Any
) -> dict[str, float | int]:
    """Measure lossy-codec propagation outside a six-pixel mask guard."""

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode encoded audit video: {video}")
    frame_mae = []
    maximum = 0
    decoded = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded >= len(masks):
                raise RuntimeError("encoded video has more frames than its audit mask")
            mask = cv2.cvtColor(masks[decoded], cv2.COLOR_BGR2GRAY)
            guard = cv2.dilate(np.where(mask >= 128, 255, 0).astype(np.uint8), kernel)
            exterior = guard == 0
            difference = np.abs(frame.astype(np.int16) - clean_plate.astype(np.int16))
            frame_mae.append(float(np.mean(difference[exterior])))
            maximum = max(maximum, int(np.max(difference[exterior])))
            decoded += 1
    finally:
        capture.release()
    if decoded != len(masks):
        raise RuntimeError(f"encoded audit decoded {decoded}/{len(masks)} frames")
    return {
        "guard_dilation_pixels": 6,
        "decoded_frames": decoded,
        "mean_frame_mae": float(np.mean(frame_mae)),
        "p95_frame_mae": float(np.percentile(frame_mae, 95)),
        "maximum_channel_difference": maximum,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--reference-alpha", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--clean-plate", type=Path, required=True)
    parser.add_argument("--edit-mask", type=Path, required=True)
    parser.add_argument("--wuji-retargeting-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--expected-reference-alpha-sha256", required=True)
    parser.add_argument("--expected-clean-plate-sha256", required=True)
    parser.add_argument("--expected-mask-sha256", required=True)
    parser.add_argument("--landmark-sigma", type=float, default=2.0)
    parser.add_argument("--forearm-end-x", type=float, default=900.0)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument(
        "--human-review",
        choices=("pending", "passed", "failed"),
        default="pending",
    )
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = {
        name: value.expanduser().resolve()
        for name, value in {
            "source": args.source_video,
            "trajectory": args.trajectory,
            "reference": args.reference_image,
            "reference_alpha": args.reference_alpha,
            "reference_manifest": args.reference_manifest,
            "clean_plate": args.clean_plate,
            "edit_mask": args.edit_mask,
            "retargeting_root": args.wuji_retargeting_root,
            "output": args.output_dir,
            "ffmpeg": args.ffmpeg,
        }.items()
    }
    for label in (
        "source",
        "trajectory",
        "reference",
        "reference_alpha",
        "reference_manifest",
        "clean_plate",
        "edit_mask",
        "ffmpeg",
    ):
        if not paths[label].is_file() or paths[label].stat().st_size == 0:
            raise ValueError(f"missing or empty {label}: {paths[label]}")
    if paths["output"].exists():
        raise FileExistsError(f"output directory exists: {paths['output']}")
    paths["output"].mkdir(parents=True)
    expected = {
        "source": args.expected_source_sha256,
        "reference": args.expected_reference_sha256,
        "reference_alpha": args.expected_reference_alpha_sha256,
        "clean_plate": args.expected_clean_plate_sha256,
        "edit_mask": args.expected_mask_sha256,
    }
    observed = {name: sha256_file(paths[name]) for name in expected}
    mismatch = {
        name: (expected[name], observed[name])
        for name in expected
        if expected[name] != observed[name]
    }
    if mismatch:
        raise ValueError(f"input SHA-256 mismatch: {mismatch}")

    import cv2
    import numpy as np
    import yaml
    from scipy.ndimage import gaussian_filter1d

    sys.path.insert(0, str(paths["retargeting_root"] / "example"))
    from input_devices.video_mediapipe import VideoMediaPipe

    started = time.perf_counter()
    config_path = paths["retargeting_root"] / "example/config/adaptive_analytical_video.yaml"
    config = yaml.safe_load(config_path.read_text()).get("video_input", {})
    device = VideoMediaPipe(
        str(paths["source"]),
        hand_side="right",
        loop=False,
        video_config=config,
        show_video=False,
    )
    raw_landmarks = list(device._raw_landmarks)
    if not raw_landmarks or any(item is None for item in raw_landmarks):
        raise RuntimeError("complete direct image-plane hand landmarks are required")
    landmarks = gaussian_filter1d(
        np.asarray(raw_landmarks, dtype=np.float32),
        sigma=args.landmark_sigma,
        axis=0,
        mode="nearest",
    )
    source_frames, source_info = _decode(cv2, paths["source"])
    mask_frames, mask_info = _decode(cv2, paths["edit_mask"])
    frame_count = len(source_frames)
    if landmarks.shape != (frame_count, 21, 2) or len(mask_frames) != frame_count:
        raise RuntimeError("source, landmarks, and edit mask do not share one frame clock")
    height, width = source_frames[0].shape[:2]
    if (height, width) != (720, 1280):
        raise RuntimeError(f"unexpected source dimensions: {width}x{height}")
    clean_plate = cv2.imread(str(paths["clean_plate"]), cv2.IMREAD_COLOR)
    reference = cv2.imread(str(paths["reference"]), cv2.IMREAD_COLOR)
    reference_alpha = cv2.imread(str(paths["reference_alpha"]), cv2.IMREAD_GRAYSCALE)
    if clean_plate is None or reference is None or reference_alpha is None:
        raise RuntimeError("could not decode reference assets")
    if any(item.shape[:2] != (height, width) for item in (clean_plate, reference, reference_alpha)):
        raise RuntimeError("reference assets do not match the source camera")
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    reference_dark_alpha = np.where(
        (reference_alpha > 8) & (reference_gray < 108), reference_alpha, 0
    ).astype(np.uint8)
    reference_foreground = np.zeros_like(reference)
    reference_foreground[reference_dark_alpha > 0] = reference[reference_dark_alpha > 0]
    reference_manifest = json.loads(paths["reference_manifest"].read_text())

    trajectory = np.load(paths["trajectory"], allow_pickle=False)
    q = np.asarray(trajectory["q"], dtype=np.float32)
    qdot = np.asarray(trajectory["qdot"], dtype=np.float32)
    if q.shape != (frame_count, 20) or qdot.shape != q.shape:
        raise RuntimeError(f"incomplete q/qdot trajectory: q={q.shape}, qdot={qdot.shape}")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qdot)):
        raise RuntimeError("q/qdot contain non-finite values")

    fps = float(source_info["fps"])
    replacement_path = paths["output"] / "wuji-real-hardware-appearance-replacement-20p7s.mp4"
    comparison_path = (
        paths["output"]
        / "human-to-wuji-real-hardware-appearance-comparison-20p7s.mp4"
    )
    replacement_writer = _writer(
        paths["ffmpeg"], replacement_path, width=width, height=height, fps=fps, crf=args.crf
    )
    comparison_writer = _writer(
        paths["ffmpeg"],
        comparison_path,
        width=width * 2,
        height=height + 48,
        fps=fps,
        crf=args.crf,
    )
    assert replacement_writer.stdin is not None and comparison_writer.stdin is not None
    frame_zero_points = landmarks[0] * np.asarray([width, height], dtype=np.float32)
    poster_result = None
    poster_index = min(frame_count - 1, 210)
    outside_maximum = 0
    texture_fractions = []
    hand_areas = []
    hand_centers = []
    forearm_widths = []
    for index, (source, edit_frame) in enumerate(zip(source_frames, mask_frames)):
        points = landmarks[index] * np.asarray([width, height], dtype=np.float32)
        hand_alpha = hand_silhouette(cv2, np, points, (height, width))
        matrix = affine_from_landmarks(cv2, frame_zero_points, points)
        photo = cv2.warpAffine(
            reference_foreground,
            matrix,
            (width, height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        photo_alpha = cv2.warpAffine(
            reference_dark_alpha,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        source_luma = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
        shade = np.clip(9.0 + 61.0 * (source_luma / 255.0) ** 1.7, 8, 76)
        glove = np.empty_like(source)
        glove[..., 0] = np.clip(shade * 0.92, 0, 255).astype(np.uint8)
        glove[..., 1] = np.clip(shade * 0.94, 0, 255).astype(np.uint8)
        glove[..., 2] = np.clip(shade, 0, 255).astype(np.uint8)
        real_weight = np.minimum(photo_alpha.astype(np.float32) / 255.0, 0.88)[..., None]
        hand_material = np.clip(
            glove.astype(np.float32) * (1.0 - real_weight) + photo * real_weight,
            0,
            255,
        ).astype(np.uint8)
        wrist = points[0].astype(np.float32)
        geometry = forearm_geometry(np, wrist, end_x=args.forearm_end_x, height=height)
        forearm_material, forearm_alpha = render_forearm(
            cv2, np, (height, width), geometry
        )
        staged = composite(np, clean_plate, forearm_material, forearm_alpha)
        staged = composite(np, staged, hand_material, hand_alpha)
        edit_gray = (
            cv2.cvtColor(edit_frame, cv2.COLOR_BGR2GRAY)
            if edit_frame.ndim == 3
            else edit_frame
        )
        allowed = np.where(edit_gray >= 128, 255, 0).astype(np.uint8)
        robot_alpha = np.maximum(forearm_alpha, hand_alpha)
        robot_alpha = cv2.bitwise_and(robot_alpha, allowed)
        result = composite(np, clean_plate, staged, robot_alpha)
        outside = allowed == 0
        outside_maximum = max(
            outside_maximum,
            int(np.max(np.abs(result[outside].astype(np.int16) - clean_plate[outside]))),
        )
        visible_hand = (hand_alpha >= 128) & (allowed > 0)
        photo_supported = visible_hand & (photo_alpha >= 32)
        hand_area = int(np.count_nonzero(visible_hand))
        if hand_area < 2_000:
            raise RuntimeError(f"hand coverage collapsed at frame {index}: {hand_area}")
        texture_fractions.append(float(np.count_nonzero(photo_supported) / hand_area))
        hand_areas.append(hand_area)
        ys, xs = np.nonzero(visible_hand)
        hand_centers.append([float(np.mean(xs)), float(np.mean(ys))])
        forearm_widths.append(2.0 * geometry["wrist_half"])
        replacement_writer.stdin.write(result.tobytes())
        canvas = np.zeros((height + 48, width * 2, 3), dtype=np.uint8)
        canvas[48:, :width] = source
        canvas[48:, width:] = result
        cv2.putText(
            canvas,
            "HUMAN SOURCE",
            (24, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (238, 242, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "REAL WUJI HARDWARE APPEARANCE | SYNTHETIC MOTION",
            (width + 24, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{index + 1:03d}/{frame_count}",
            (width * 2 - 120, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (225, 231, 244),
            1,
            cv2.LINE_AA,
        )
        comparison_writer.stdin.write(canvas.tobytes())
        if index == poster_index:
            poster_result = result.copy()
    replacement_writer.stdin.close()
    comparison_writer.stdin.close()
    if replacement_writer.wait() != 0 or comparison_writer.wait() != 0:
        raise RuntimeError("FFmpeg encoder failed")
    if outside_maximum != 0:
        raise RuntimeError("scene-lock invariant failed outside the declared edit mask")

    poster_path = paths["output"] / "human-to-wuji-real-hardware-appearance-poster.jpg"
    if poster_result is None:
        raise RuntimeError("poster frame was not retained")
    poster = np.hstack((source_frames[poster_index], poster_result))
    if not cv2.imwrite(str(poster_path), poster, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError("could not write poster")
    centers = np.asarray(hand_centers, dtype=np.float32)
    motion = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    encoded_audit = encoded_exterior_audit(
        cv2, np, replacement_path, mask_frames, clean_plate
    )
    elapsed = time.perf_counter() - started
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "accepted_visual_demo"
            if args.human_review == "passed"
            else "rejected"
            if args.human_review == "failed"
            else "review_required"
        ),
        "honest_status": "PARTIAL",
        "claim": (
            "Official physical-Wuji pixels and material statistics supply the black "
            "compliant-hand appearance; complete image landmarks drive synthetic motion "
            "inside a declared source-camera edit mask."
        ),
        "not_claimed": [
            "This is not footage of physical Wuji hardware executing the gesture.",
            "The source human skeleton supplies motion; q/qdot are validation "
            "evidence rather than robot telemetry for these pixels.",
            "The comparison does not establish metric depth, force closure, or tactile contact.",
        ],
        "inputs": {
            name: {"path": str(paths[name]), "sha256": sha256_file(paths[name])}
            for name in (
                "source",
                "trajectory",
                "reference",
                "reference_alpha",
                "reference_manifest",
                "clean_plate",
                "edit_mask",
            )
        },
        "hardware_source": reference_manifest["hardware_appearance_source"],
        "motion": {
            "direct_image_landmark_frames": frame_count,
            "q_shape": list(q.shape),
            "qdot_shape": list(qdot.shape),
            "hand_centroid_step_median_pixels": float(np.median(motion)),
            "hand_centroid_step_maximum_pixels": float(np.max(motion)),
            "hand_area_minimum_pixels": int(np.min(hand_areas)),
            "hand_area_maximum_pixels": int(np.max(hand_areas)),
        },
        "appearance": {
            "real_photo_texture_fraction_minimum": float(np.min(texture_fractions)),
            "real_photo_texture_fraction_median": float(np.median(texture_fractions)),
            "real_photo_texture_fraction_maximum": float(np.max(texture_fractions)),
            "filled_fingers": 5,
            "exposed_simulation_mesh": False,
            "forearm_width_pixels_minimum": float(np.min(forearm_widths)),
            "forearm_width_pixels_maximum": float(np.max(forearm_widths)),
            "source_human_rgb_copied_into_hand": False,
            "maximum_hand_area_relative_step": float(
                np.max(
                    np.abs(np.diff(np.asarray(hand_areas, dtype=np.float32)))
                    / np.maximum(np.asarray(hand_areas[:-1], dtype=np.float32), 1.0)
                )
            ),
        },
        "scene_lock": {
            "preencode_outside_edit_mask_max_rgb_difference": outside_maximum,
            "clean_plate_is_identical_for_every_frame": True,
            "mask_info": mask_info,
            "encoded_exterior_audit": encoded_audit,
        },
        "throughput": {
            "wall_seconds": elapsed,
            "effective_fps": frame_count / elapsed,
            "realtime_factor": elapsed / (frame_count / fps),
        },
        "human_review": args.human_review,
        "outputs": {
            "replacement": str(replacement_path),
            "replacement_sha256": sha256_file(replacement_path),
            "comparison": str(comparison_path),
            "comparison_sha256": sha256_file(comparison_path),
            "poster": str(poster_path),
            "poster_sha256": sha256_file(poster_path),
            "frames": frame_count,
            "fps": fps,
            "duration_seconds": frame_count / fps,
            "width": width,
            "height": height,
        },
        "runtime": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
        },
    }
    manifest_path = paths["output"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"outputs": manifest["outputs"], "throughput": manifest["throughput"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
