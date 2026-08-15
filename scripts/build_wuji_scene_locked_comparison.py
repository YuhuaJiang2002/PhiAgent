#!/usr/bin/env python3
"""Build a source-scene-locked human-to-Wuji Hand comparison.

The official Wuji Hand MJCF and a complete q/qdot trajectory drive the hand.
Only a declared hand/forearm replacement mask may change source-scene pixels.
Heavy runtime dependencies remain optional and are imported inside ``main``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path


HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def git_state(repo: Path) -> dict[str, object]:
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1"], text=True
    ).splitlines()
    return {
        "head": git_head(repo),
        "branch": subprocess.check_output(
            ["git", "-C", str(repo), "branch", "--show-current"], text=True
        ).strip(),
        "dirty": bool(status),
        "status": status,
    }


def ffprobe_video(path: Path) -> dict[str, object]:
    payload = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_read_frames",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(payload)


def direct_detection_mask(frames: list[object | None]) -> list[bool]:
    result: list[bool] = []
    previous: object | None = None
    for frame in frames:
        direct = frame is not None and frame is not previous
        result.append(direct)
        if frame is not None:
            previous = frame
    return result


def urdf_velocity_limits(path: Path, joint_names: list[str]) -> list[float]:
    root = ET.parse(path).getroot()
    by_name: dict[str, float] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and "velocity" in limit.attrib:
            by_name[joint.attrib["name"]] = float(limit.attrib["velocity"])
    missing = [name for name in joint_names if name not in by_name]
    if missing:
        raise ValueError(f"URDF velocity limits missing for joints: {missing}")
    return [by_name[name] for name in joint_names]


def expand_row_spans(mask, np):
    """Make each occupied mask row contiguous and return the exact edit mask."""
    expanded = np.zeros_like(mask)
    for row in np.flatnonzero(np.any(mask, axis=1)):
        columns = np.flatnonzero(mask[row])
        expanded[row, columns[0] : columns[-1] + 1] = 255
    return expanded


def fill_from_row_context(frame, mask, np):
    """Replace a foreground span by interpolation from same-row scene pixels."""
    output = frame.copy()
    width = frame.shape[1]
    for row in np.flatnonzero(np.any(mask, axis=1)):
        columns = np.flatnonzero(mask[row])
        left, right = int(columns[0]), int(columns[-1])
        left_slice = frame[row, max(0, left - 70) : max(1, left - 35)]
        right_slice = frame[row, min(width - 1, right + 35) : min(width, right + 70)]
        if left_slice.size and right_slice.size:
            left_color = np.median(left_slice, axis=0)
            right_color = np.median(right_slice, axis=0)
        elif left_slice.size:
            left_color = right_color = np.median(left_slice, axis=0)
        elif right_slice.size:
            left_color = right_color = np.median(right_slice, axis=0)
        else:
            continue
        mix = np.linspace(0.0, 1.0, right - left + 1, dtype=np.float32)[:, None]
        values = left_color[None, :] * (1.0 - mix) + right_color[None, :] * mix
        output[row, left : right + 1] = np.clip(values, 0, 255).astype(np.uint8)
    return output


def stable_forearm_half_widths(palm_width: float) -> tuple[float, float]:
    """Return fixed Shadow-style link widths derived once per full video."""
    wrist_half = min(30.0, max(21.0, 0.24 * palm_width))
    end_half = min(34.0, max(24.0, 0.28 * palm_width))
    return wrist_half, end_half


def build_static_clean_plate(
    cv2,
    np,
    source: Path,
    smoothed_points,
    width: int,
    height: int,
    frame_count: int,
):
    """Build one immutable background reconstruction for every output frame."""
    capture = cv2.VideoCapture(str(source))
    union_mask = np.zeros((height, width), dtype=np.uint8)
    samples = []
    sample_masks = []
    sample_stride = max(1, frame_count // 20)
    fixed_end_x = None
    try:
        for index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"clean-plate decode stopped at frame {index}")
            mask, _wrist, end, _normal, _wrist_half, _end_half = human_replacement_mask(
                cv2,
                np,
                frame,
                smoothed_points[index],
                width,
                height,
                fixed_end_x,
            )
            if fixed_end_x is None:
                fixed_end_x = float(end[0])
            union_mask = np.maximum(union_mask, mask)
            if index % sample_stride == 0 or index == frame_count - 1:
                samples.append(frame)
                sample_masks.append(mask > 0)
    finally:
        capture.release()
    if fixed_end_x is None or not samples:
        raise RuntimeError("could not construct a static source-scene clean plate")
    sample_stack = np.stack(samples, axis=0)
    sample_mask_stack = np.stack(sample_masks, axis=0)
    valid_counts = np.sum(~sample_mask_stack, axis=0)
    masked_samples = np.where(
        sample_mask_stack[:, :, :, None],
        np.nan,
        sample_stack.astype(np.float32),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        with np.errstate(invalid="ignore"):
            clean_float = np.nanmedian(masked_samples, axis=0)
    temporal_median = np.median(sample_stack, axis=0).astype(np.uint8)
    observed = valid_counts > 0
    clean_seed = temporal_median.copy()
    clean_seed[observed] = np.clip(clean_float[observed], 0, 255).astype(np.uint8)
    unobserved_mask = (~observed).astype(np.uint8) * 255
    unobserved_mask = cv2.dilate(
        unobserved_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    union_mask = cv2.dilate(
        union_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    clean_plate = fill_from_row_context(clean_seed, unobserved_mask, np)
    return (
        clean_plate,
        union_mask,
        fixed_end_x,
        len(samples),
        float(np.mean(observed)),
        float(np.mean(unobserved_mask > 0)),
    )


def human_replacement_mask(
    cv2,
    np,
    frame,
    points,
    width: int,
    height: int,
    fixed_end_x: float | None = None,
):
    pixels = points.copy()
    pixels[:, 0] *= width
    pixels[:, 1] *= height
    pixels = np.rint(pixels).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(pixels[:, :2])
    cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_AA)
    palm_width = float(np.linalg.norm(pixels[5, :2] - pixels[17, :2]))
    line_width = max(24, int(round(0.22 * palm_width)))
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            mask,
            tuple(pixels[start, :2]),
            tuple(pixels[end, :2]),
            255,
            line_width,
            cv2.LINE_AA,
        )
    wrist = pixels[0, :2].astype(np.float32)
    cv2.circle(
        mask,
        tuple(np.rint(wrist).astype(int)),
        max(68, int(round(0.45 * palm_width))),
        255,
        -1,
        cv2.LINE_AA,
    )
    dilation = max(26, int(round(0.18 * palm_width)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
    )
    mask = cv2.dilate(mask, kernel)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    wrist_x, wrist_y = np.rint(wrist).astype(int)
    reference_center = np.rint(np.mean(pixels[[0, 5, 9, 13, 17], :2], axis=0)).astype(
        int
    )
    reference_x, reference_y = int(reference_center[0]), int(reference_center[1])
    patch = lab[
        max(0, reference_y - 10) : min(height, reference_y + 11),
        max(0, reference_x - 10) : min(width, reference_x + 11),
    ]
    if patch.size:
        reference = np.median(patch.reshape(-1, 3), axis=0)
    else:
        reference = lab[np.clip(wrist_y, 0, height - 1), np.clip(wrist_x, 0, width - 1)]
    delta = lab - reference[None, None, :]
    color_distance = np.sqrt(0.16 * delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2 + delta[:, :, 2] ** 2)
    skin = (color_distance <= 24.0).astype(np.uint8) * 255
    roi_top = max(0, int(round(wrist[1] - 36)))
    roi_left = max(0, int(round(wrist[0] - 190)))
    roi_right = min(width, int(round(wrist[0] + 230)))
    skin_roi = skin[roi_top:, roi_left:roi_right]
    skin_roi = cv2.morphologyEx(
        skin_roi,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (skin_roi > 0).astype(np.uint8), connectivity=8
    )
    seed = np.zeros_like(skin_roi)
    local_wrist = wrist - np.array([roi_left, roi_top], dtype=np.float32)
    cv2.circle(seed, tuple(np.rint(local_wrist).astype(int)), 34, 255, -1)
    candidate_labels = labels[seed > 0]
    candidate_labels = candidate_labels[candidate_labels > 0]
    if candidate_labels.size:
        unique, counts = np.unique(candidate_labels, return_counts=True)
        arm_label = int(unique[np.argmax(counts)])
    elif labels_count > 1:
        arm_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        arm_label = 0
    arm_skin_roi = (labels == arm_label).astype(np.uint8) * 255 if arm_label else seed
    local_cutoff = max(0, int(round(local_wrist[1] - 18)))
    arm_skin_roi[:local_cutoff] = 0
    arm_skin_roi = cv2.dilate(
        arm_skin_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    )
    arm_y, arm_x = np.nonzero(arm_skin_roi)
    arm_y = arm_y + roi_top
    arm_x = arm_x + roi_left
    fit = arm_y >= wrist[1] + 12
    bottom_skin = skin[max(0, height - 110) : height, roi_left:roi_right]
    bottom_y, bottom_x = np.nonzero(bottom_skin)
    if bottom_x.size >= 80:
        measured_end_x = float(np.median(bottom_x) + roi_left)
        bottom_residual = np.abs(bottom_x + roi_left - measured_end_x)
        detected_half = min(150.0, float(np.percentile(bottom_residual, 98)) + 20.0)
    elif int(np.count_nonzero(fit)) >= 80:
        slope, intercept = np.polyfit(arm_y[fit], arm_x[fit], deg=1)
        measured_end_x = float(slope * (height + 60.0) + intercept)
        measured_end_x = float(
            np.clip(measured_end_x, wrist[0] - 170.0, wrist[0] + 170.0)
        )
        residual = np.abs(arm_x[fit] - (slope * arm_y[fit] + intercept))
        detected_half = min(150.0, float(np.percentile(residual, 98)) + 20.0)
    else:
        measured_end_x = float(wrist[0] + 0.18 * (height - wrist[1]))
        detected_half = 72.0
    end_x = measured_end_x if fixed_end_x is None else fixed_end_x
    end = np.array([end_x, height + 60.0], dtype=np.float32)
    direction = end - wrist
    direction /= max(float(np.linalg.norm(direction)), 1.0)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    wrist_half = max(58.0, 0.38 * palm_width)
    end_half = max(78.0, 0.48 * palm_width, detected_half)
    arm_polygon = np.rint(
        np.stack(
            [
                wrist - normal * wrist_half,
                wrist + normal * wrist_half,
                end + normal * end_half,
                end - normal * end_half,
            ]
        )
    ).astype(np.int32)
    cv2.fillConvexPoly(mask, arm_polygon, 255, lineType=cv2.LINE_AA)
    hand_skin = np.zeros_like(mask)
    hand_left = max(0, int(np.min(pixels[:, 0]) - 70))
    hand_right = min(width, int(np.max(pixels[:, 0]) + 71))
    hand_top = max(0, int(np.min(pixels[:, 1]) - 70))
    hand_bottom = min(height, int(np.max(pixels[:, 1]) + 71))
    hand_skin[hand_top:hand_bottom, hand_left:hand_right] = skin[
        hand_top:hand_bottom, hand_left:hand_right
    ]
    hand_skin = cv2.dilate(
        hand_skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    )
    mask = np.maximum(mask, hand_skin)
    return expand_row_spans(mask, np), wrist, end, normal, wrist_half, end_half


def draw_robot_forearm(
    cv2,
    np,
    frame,
    wrist,
    end,
    normal,
    wrist_half: float,
    end_half: float,
):
    """Draw a thin, rigid wrist link modeled after the accepted Shadow demo."""
    height, width = frame.shape[:2]
    alpha = np.zeros((height, width), dtype=np.uint8)
    axis = end - wrist
    axis_norm = max(float(np.linalg.norm(axis)), 1.0)
    axis /= axis_norm
    arm_top = wrist + axis * 28.0
    connector_top = wrist - axis * 5.0
    connector_half = 0.68 * wrist_half
    main_polygon = np.rint(
        np.stack(
            [
                arm_top - normal * wrist_half,
                arm_top + normal * wrist_half,
                end + normal * end_half,
                end - normal * end_half,
            ]
        )
    ).astype(np.int32)
    connector_polygon = np.rint(
        np.stack(
            [
                connector_top - normal * connector_half,
                connector_top + normal * connector_half,
                arm_top + normal * wrist_half * 0.88,
                arm_top - normal * wrist_half * 0.88,
            ]
        )
    ).astype(np.int32)
    cv2.fillConvexPoly(alpha, main_polygon, 255, lineType=cv2.LINE_AA)
    cv2.fillConvexPoly(alpha, connector_polygon, 255, lineType=cv2.LINE_AA)
    metal = np.zeros_like(frame)
    yy, xx = np.mgrid[0:height, 0:width]
    lateral = (xx - wrist[0]) * normal[0] + (yy - wrist[1]) * normal[1]
    longitudinal = (xx - wrist[0]) * axis[0] + (yy - wrist[1]) * axis[1]
    highlight = np.exp(
        -((lateral + wrist_half * 0.18) / max(wrist_half * 0.62, 1.0)) ** 2
    )
    length_shade = np.clip(longitudinal / max(axis_norm, 1.0), 0.0, 1.0)
    base = 44.0 + 55.0 * highlight - 8.0 * length_shade
    metal[:, :, 0] = np.clip(base * 1.18, 0, 255)
    metal[:, :, 1] = np.clip(base * 1.10, 0, 255)
    metal[:, :, 2] = np.clip(base * 0.98, 0, 255)
    edge = cv2.Canny(alpha, 30, 90)
    metal[edge > 0] = (26, 34, 39)
    blend = (
        cv2.GaussianBlur(alpha, (0, 0), 0.55).astype(np.float32)[:, :, None]
        / 255.0
    )
    output = np.clip(frame.astype(np.float32) * (1.0 - blend) + metal * blend, 0, 255).astype(
        np.uint8
    )
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    joint_center = wrist + axis * 13.0
    joint_axes = (max(7, int(round(connector_half))), max(5, int(round(wrist_half * 0.34))))
    cv2.ellipse(
        output,
        tuple(np.rint(joint_center).astype(int)),
        joint_axes,
        angle,
        0,
        360,
        (37, 50, 58),
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        output,
        tuple(np.rint(joint_center).astype(int)),
        joint_axes,
        angle,
        0,
        360,
        (121, 151, 162),
        2,
        cv2.LINE_AA,
    )
    collar_a = np.rint(arm_top - normal * wrist_half).astype(int)
    collar_b = np.rint(arm_top + normal * wrist_half).astype(int)
    cv2.line(output, tuple(collar_a), tuple(collar_b), (137, 161, 170), 4, cv2.LINE_AA)
    for sign in (-1.0, 1.0):
        bolt = joint_center + normal * connector_half * 0.55
        if sign < 0:
            bolt = joint_center - normal * connector_half * 0.55
        cv2.circle(output, tuple(np.rint(bolt).astype(int)), 3, (190, 160, 83), -1, cv2.LINE_AA)
    return output, alpha


def warp_robot_to_hand(
    cv2,
    np,
    robot_rgb,
    raw_alpha,
    points,
    width: int,
    height: int,
    fixed_scale: float | None = None,
):
    support = np.argwhere(raw_alpha > 4)
    if not support.size:
        raise RuntimeError("official Wuji render has no visible pixels")
    y0, x0 = support.min(axis=0)
    y1, x1 = support.max(axis=0)
    source = points.copy()
    source[:, 0] *= width
    source[:, 1] *= height
    target_anchor = source[0]
    direction = np.mean(source[[9, 12]], axis=0) - target_anchor
    direction_norm = max(float(np.linalg.norm(direction)), 1.0)
    direction /= direction_norm
    input_anchor = np.array([(x0 + x1) * 0.5, y1 - 0.025 * (y1 - y0)], dtype=np.float32)
    render_height = max(float(y1 - y0), 1.0)
    render_width = max(float(x1 - x0), 1.0)
    target_height = max(float(np.max(np.linalg.norm(source - target_anchor, axis=1))), 1.0)
    target_width = max(float(np.ptp(source[:, 0])), 1.0)
    measured_scale = max(
        1.06 * target_height / render_height, 1.03 * target_width / render_width
    )
    scale = float(np.clip(measured_scale, 0.35, 1.15)) if fixed_scale is None else fixed_scale
    ux, uy = float(direction[0]), float(direction[1])
    transform = scale * np.array([[-uy, -ux], [ux, -uy]], dtype=np.float32)
    translation = target_anchor - transform @ input_anchor
    matrix = np.column_stack([transform, translation]).astype(np.float32)
    warped_rgb = cv2.warpAffine(
        robot_rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_alpha = cv2.warpAffine(
        raw_alpha,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_alpha = cv2.GaussianBlur(warped_alpha, (0, 0), 0.55)
    return warped_rgb, warped_alpha, scale


def label_comparison(cv2, frame, frame_index: int, total_frames: int):
    header = 48
    width = frame.shape[1] // 2
    cv2.rectangle(frame, (0, 0), (frame.shape[1], header), (7, 12, 25), -1)
    cv2.line(frame, (width, 0), (width, frame.shape[0]), (91, 108, 147), 2)
    cv2.putText(
        frame,
        "HUMAN SOURCE",
        (24, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (238, 242, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "WUJI HAND  |  SHADOW-STYLE SOURCE-SCENE LOCK",
        (width + 24, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (126, 231, 200),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{frame_index + 1:03d}/{total_frames}",
        (width * 2 - 112, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (225, 231, 244),
        1,
        cv2.LINE_AA,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--wuji-retargeting-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--landmark-sigma", type=float, default=2.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--camera-azimuth", type=float, default=180.0)
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    parser.add_argument("--camera-distance", type=float, default=0.48)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = args.source_video.expanduser().resolve()
    trajectory = args.trajectory.expanduser().resolve()
    root = args.wuji_retargeting_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_root = root / "wuji_retargeting" / "wuji-description" / "hand" / "body"
    mjcf = model_root / "mjcf" / f"{args.hand_side}.xml"
    urdf = model_root / "urdf" / f"{args.hand_side}.urdf"
    config = root / "example" / "config" / "adaptive_analytical_video.yaml"
    missing = [path for path in (source, trajectory, mjcf, urdf, config) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")
    output_dir.mkdir(parents=True, exist_ok=False)
    workspace_root = Path(__file__).resolve().parents[1]

    try:
        import cv2
        import mediapipe
        import mujoco
        import numpy as np
        import scipy
        import yaml
        from scipy.ndimage import gaussian_filter1d
    except ImportError as exc:
        raise SystemExit(
            "Install the optional Wuji, MuJoCo, MediaPipe, OpenCV, NumPy, SciPy, and PyYAML stack. "
            f"Missing import: {exc}"
        ) from exc

    sys.path.insert(0, str(root / "example"))
    from input_devices.video_mediapipe import VideoMediaPipe

    started = time.perf_counter()
    device = VideoMediaPipe(
        str(source),
        hand_side=args.hand_side,
        loop=False,
        video_config=yaml.safe_load(config.read_text()).get("video_input", {}),
        show_video=False,
    )
    raw_frames = list(device._raw_landmarks)
    detection_mask = direct_detection_mask(raw_frames)
    if not raw_frames or any(frame is None for frame in raw_frames):
        raise RuntimeError("complete image-plane hand landmarks are required")
    raw_points = np.asarray(raw_frames, dtype=np.float32)
    smoothed_points = gaussian_filter1d(raw_points, sigma=args.landmark_sigma, axis=0, mode="nearest")

    arrays = np.load(trajectory)
    q = np.asarray(arrays["q"], dtype=np.float32)
    qdot = np.asarray(arrays["qdot"], dtype=np.float32)
    joint_names = [str(name) for name in arrays["joint_names"]]
    if q.shape != qdot.shape or q.shape != (len(raw_frames), 20):
        raise RuntimeError(f"trajectory shape mismatch: q={q.shape}, qdot={qdot.shape}")

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    model_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ]
    if model_joint_names != joint_names:
        raise RuntimeError("trajectory joint order does not match the official Wuji MJCF")
    lower = model.jnt_range[:, 0]
    upper = model.jnt_range[:, 1]
    joint_limit_violations = int(
        np.count_nonzero((q < lower[None, :] - 1e-5) | (q > upper[None, :] + 1e-5))
    )
    velocity_limits = np.asarray(urdf_velocity_limits(urdf, joint_names), dtype=np.float32)
    velocity_limit_violations = int(
        np.count_nonzero(np.abs(qdot) > velocity_limits[None, :] + 1e-5)
    )
    if joint_limit_violations or velocity_limit_violations:
        raise RuntimeError(
            f"physical gate failed: position={joint_limit_violations}, velocity={velocity_limit_violations}"
        )

    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count != q.shape[0]:
        raise RuntimeError(f"source/trajectory frame mismatch: {frame_count} != {q.shape[0]}")
    capture.release()

    (
        clean_plate,
        clean_plate_union,
        calibrated_end_x,
        clean_plate_samples,
        clean_plate_observed_fraction,
        clean_plate_unobserved_fill_fraction,
    ) = build_static_clean_plate(
        cv2,
        np,
        source,
        smoothed_points,
        width,
        height,
        frame_count,
    )
    clean_plate_path = output_dir / "wuji-source-scene-static-clean-plate.png"
    if not cv2.imwrite(str(clean_plate_path), clean_plate):
        raise RuntimeError(f"could not write clean plate: {clean_plate_path}")
    palm_widths = np.linalg.norm(
        (smoothed_points[:, 5, :2] - smoothed_points[:, 17, :2])
        * np.asarray([width, height], dtype=np.float32)[None, :],
        axis=1,
    )
    median_palm_width = float(np.median(palm_widths))
    forearm_half_wrist, forearm_half_end = stable_forearm_half_widths(
        median_palm_width
    )
    capture = cv2.VideoCapture(str(source))

    render_size = 720
    model.vis.global_.offwidth = render_size
    model.vis.global_.offheight = render_size
    renderer = mujoco.Renderer(model, height=render_size, width=render_size)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.distance = args.camera_distance
    camera.lookat[:] = [0.0, 0.0, 0.055]
    render_options = mujoco.MjvOption()
    render_options.geomgroup[:] = 0
    render_options.geomgroup[1] = 1

    video_path = (
        output_dir
        / "human-to-wuji-hand-shadow-style-scene-locked-comparison-20p7s.mp4"
    )
    mask_path = output_dir / "source-scene-replacement-mask.mkv"
    video_command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width * 2}x{height + 48}",
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
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    mask_command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "ffv1",
        str(mask_path),
    ]
    video_encoder = subprocess.Popen(video_command, stdin=subprocess.PIPE)
    mask_encoder = subprocess.Popen(mask_command, stdin=subprocess.PIPE)
    render_started = time.perf_counter()
    poster = None
    edit_fractions = []
    scales = []
    wrist_overlap_pixels = []
    static_plate_interior_mae = []
    static_plate_interior_max = 0
    calibrated_scale = None
    outside_max = 0
    try:
        for index in range(frame_count):
            ok, source_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"source decode stopped at frame {index}")
            human_mask, wrist, end, normal, _wrist_half, _end_half = human_replacement_mask(
                cv2,
                np,
                source_bgr,
                smoothed_points[index],
                width,
                height,
                calibrated_end_x,
            )
            erase_alpha = cv2.GaussianBlur(human_mask, (0, 0), 1.15)
            erase_blend = erase_alpha.astype(np.float32)[:, :, None] / 255.0
            candidate = np.clip(
                source_bgr.astype(np.float32) * (1.0 - erase_blend)
                + clean_plate.astype(np.float32) * erase_blend,
                0,
                255,
            ).astype(np.uint8)
            plate_interior = erase_alpha >= 254
            plate_residual = np.abs(
                candidate.astype(np.int16) - clean_plate.astype(np.int16)
            )[plate_interior]
            static_plate_interior_mae.append(float(np.mean(plate_residual)))
            static_plate_interior_max = max(
                static_plate_interior_max, int(plate_residual.max(initial=0))
            )
            candidate, forearm_alpha = draw_robot_forearm(
                cv2,
                np,
                candidate,
                wrist,
                end,
                normal,
                forearm_half_wrist,
                forearm_half_end,
            )

            data.qpos[:] = q[index]
            data.qvel[:] = qdot[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=render_options)
            robot_rgb = renderer.render()
            raw_alpha = np.clip(np.max(robot_rgb, axis=2).astype(np.float32) * 2.8, 0, 255).astype(
                np.uint8
            )
            robot_bgr = cv2.cvtColor(robot_rgb, cv2.COLOR_RGB2BGR)
            warped_bgr, robot_alpha, scale = warp_robot_to_hand(
                cv2,
                np,
                robot_bgr,
                raw_alpha,
                smoothed_points[index],
                width,
                height,
                calibrated_scale,
            )
            if calibrated_scale is None:
                calibrated_scale = scale
            blend = robot_alpha.astype(np.float32)[:, :, None] / 255.0
            candidate = np.clip(
                candidate.astype(np.float32) * (1.0 - blend) + warped_bgr * blend, 0, 255
            ).astype(np.uint8)
            wrist_overlap_pixels.append(
                int(np.count_nonzero((forearm_alpha > 0) & (robot_alpha > 16)))
            )
            erase_support = (erase_alpha > 0).astype(np.uint8) * 255
            edit_mask = np.maximum(
                erase_support, np.maximum(forearm_alpha, robot_alpha)
            )
            candidate[edit_mask == 0] = source_bgr[edit_mask == 0]
            outside = np.abs(candidate.astype(np.int16) - source_bgr.astype(np.int16))[
                edit_mask == 0
            ]
            outside_max = max(outside_max, int(outside.max(initial=0)))
            edit_fractions.append(float(np.mean(edit_mask > 0)))
            scales.append(scale)

            comparison = np.zeros((height + 48, width * 2, 3), dtype=np.uint8)
            comparison[48:, :width] = source_bgr
            comparison[48:, width:] = candidate
            label_comparison(cv2, comparison, index, frame_count)
            if index == frame_count // 2:
                poster = comparison.copy()
            assert video_encoder.stdin is not None
            assert mask_encoder.stdin is not None
            video_encoder.stdin.write(comparison.tobytes())
            mask_encoder.stdin.write(edit_mask.tobytes())
    finally:
        capture.release()
        renderer.close()
        if video_encoder.stdin is not None:
            video_encoder.stdin.close()
        if mask_encoder.stdin is not None:
            mask_encoder.stdin.close()
        video_return = video_encoder.wait()
        mask_return = mask_encoder.wait()
    if video_return or mask_return:
        raise RuntimeError(f"encoder failure: video={video_return}, mask={mask_return}")
    render_seconds = time.perf_counter() - render_started
    if min(wrist_overlap_pixels) < 20:
        raise RuntimeError(
            "wrist-link continuity gate failed: "
            f"minimum overlap={min(wrist_overlap_pixels)} pixels"
        )
    if static_plate_interior_max > 1:
        raise RuntimeError(
            "static clean-plate gate failed: "
            f"maximum interior deviation={static_plate_interior_max}"
        )

    if poster is None:
        raise RuntimeError("poster frame was not captured")
    poster_path = output_dir / "human-to-wuji-hand-shadow-style-scene-locked-poster.jpg"
    cv2.imwrite(str(poster_path), poster, [cv2.IMWRITE_JPEG_QUALITY, 92])

    probe = ffprobe_video(video_path)
    decoded_frames = int(probe["streams"][0]["nb_read_frames"])
    if decoded_frames != frame_count:
        raise RuntimeError(f"decoded frame mismatch: {decoded_frames} != {frame_count}")

    decoded = cv2.VideoCapture(str(video_path))
    masks = cv2.VideoCapture(str(mask_path))
    background_mae = []
    background_max = 0
    decoded_count = 0
    guard_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    while True:
        ok_video, frame = decoded.read()
        ok_mask, mask_frame = masks.read()
        if not ok_video or not ok_mask:
            break
        left = frame[48:, :width]
        right = frame[48:, width:]
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        guard = cv2.dilate((mask_gray > 0).astype(np.uint8), guard_kernel) > 0
        difference = np.abs(left.astype(np.int16) - right.astype(np.int16))
        exterior = difference[~guard]
        background_mae.append(float(np.mean(exterior)))
        background_max = max(background_max, int(exterior.max(initial=0)))
        decoded_count += 1
    decoded.release()
    masks.release()
    if decoded_count != frame_count:
        raise RuntimeError(f"post-decode audit stopped at {decoded_count}/{frame_count}")

    total_seconds = time.perf_counter() - started
    manifest = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "claim": "Official Wuji Hand model with a thin rigid wrist link composited into a static reconstruction of the occluded source scene under a declared edit mask.",
        "non_claims": [
            "This is not footage of physical Wuji hardware.",
            "The thin metallic forearm and wrist connector are procedural because the official Wuji asset is a hand model.",
            "Monocular image landmarks do not establish metric depth, contact force, or real execution.",
        ],
        "source": {
            "path": source.name,
            "sha256": sha256_file(source),
            "frames": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
        },
        "upstream": {
            "repository": "https://github.com/wuji-technology/wuji-retargeting",
            "commit": git_head(root),
            "description_commit": git_head(root / "wuji_retargeting" / "wuji-description"),
            "config_sha256": sha256_file(config),
            "mjcf_sha256": sha256_file(mjcf),
            "urdf_sha256": sha256_file(urdf),
            "model": "Wuji Hand official MJCF/URDF",
        },
        "trajectory": {
            "artifact": trajectory.name,
            "artifact_sha256": sha256_file(trajectory),
            "q_shape": list(q.shape),
            "qdot_shape": list(qdot.shape),
            "direct_detection_frames": int(sum(detection_mask)),
            "held_observation_frames": int(len(detection_mask) - sum(detection_mask)),
            "joint_limit_violations": joint_limit_violations,
            "velocity_limit_violations": velocity_limit_violations,
            "max_velocity_limit_ratio": float(
                np.max(np.abs(qdot) / velocity_limits[None, :])
            ),
        },
        "scene_lock": {
            "coordinate_frame": "camera:source_video_pixels",
            "landmark_smoothing": {
                "method": "zero-phase Gaussian",
                "sigma_frames": args.landmark_sigma,
            },
            "mask_artifact": mask_path.name,
            "mask_sha256": sha256_file(mask_path),
            "clean_plate_artifact": clean_plate_path.name,
            "clean_plate_sha256": sha256_file(clean_plate_path),
            "clean_plate_temporal_samples": clean_plate_samples,
            "clean_plate_observed_background_fraction": clean_plate_observed_fraction,
            "clean_plate_unobserved_fill_fraction": clean_plate_unobserved_fill_fraction,
            "clean_plate_union_fraction": float(np.mean(clean_plate_union > 0)),
            "clean_plate_reused_without_framewise_reestimation": True,
            "clean_plate_interior_mae_mean": float(
                np.mean(static_plate_interior_mae)
            ),
            "clean_plate_interior_mae_p95": float(
                np.percentile(static_plate_interior_mae, 95)
            ),
            "clean_plate_interior_max_abs_rgb": static_plate_interior_max,
            "pre_encode_max_abs_rgb_outside_declared_mask": outside_max,
            "edit_fraction_mean": float(np.mean(edit_fractions)),
            "edit_fraction_p95": float(np.percentile(edit_fractions, 95)),
            "render_scale_min": float(np.min(scales)),
            "render_scale_max": float(np.max(scales)),
            "render_scale_max_frame_step": float(np.max(np.abs(np.diff(scales)))),
            "forearm_half_width_at_wrist_pixels": forearm_half_wrist,
            "forearm_half_width_at_frame_exit_pixels": forearm_half_end,
            "forearm_width_frame_step_pixels": 0.0,
            "wrist_link_overlap_pixels_min": int(np.min(wrist_overlap_pixels)),
            "wrist_link_overlap_pixels_p05": float(
                np.percentile(wrist_overlap_pixels, 5)
            ),
            "post_decode_exterior_mae_mean": float(np.mean(background_mae)),
            "post_decode_exterior_mae_p95": float(np.percentile(background_mae, 95)),
            "post_decode_exterior_max_abs_rgb": background_max,
            "post_decode_guard_dilation_pixels": 6,
        },
        "output": {
            "video": video_path.name,
            "video_sha256": sha256_file(video_path),
            "poster": poster_path.name,
            "poster_sha256": sha256_file(poster_path),
            "probe": probe,
            "decoded_frames": decoded_frames,
            "render_seconds": render_seconds,
            "render_fps": float(frame_count / render_seconds),
            "total_seconds": total_seconds,
            "end_to_end_fps": float(frame_count / total_seconds),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "mediapipe": mediapipe.__version__,
            "opencv": cv2.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "experiment": {
            "workspace_git": git_state(workspace_root),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "randomness_used": False,
            "configuration": {
                "hand_side": args.hand_side,
                "landmark_sigma": args.landmark_sigma,
                "crf": args.crf,
                "camera_azimuth": args.camera_azimuth,
                "camera_elevation": args.camera_elevation,
                "camera_distance": args.camera_distance,
                "clean_plate_method": "21-sample mask-excluded temporal mosaic plus fixed row-context fill only where never observed",
                "forearm_style": "thin rigid Shadow-style link with explicit overlapping wrist connector",
            },
        },
        "command": sys.argv,
    }
    manifest_path = (
        output_dir / "human-to-wuji-hand-shadow-style-scene-locked-manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), **manifest["output"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
