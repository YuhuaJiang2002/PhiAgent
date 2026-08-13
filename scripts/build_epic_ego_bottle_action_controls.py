#!/usr/bin/env python3
"""Compile long egocentric bottle tasks into explicit control videos."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAMERA_PIXEL_FRAME = "camera:epic_kitchens_p03_28_pixels"
ACTION_LABELS = ("pour-bottle", "shake-bottle", "handover-bottle")
MULTITASK_ACTION_LABELS = (
    "place-bottle-rack",
    "unscrew-bottle-cap",
    "rinse-bottle",
)
SUPPORTED_ACTION_LABELS = ACTION_LABELS + MULTITASK_ACTION_LABELS


@dataclass(frozen=True)
class EgoControlState:
    bottle_x: float
    bottle_y: float
    bottle_rotation_degrees: float
    bottle_scale: float
    left_wrist_x: float
    left_wrist_y: float
    right_wrist_x: float
    right_wrist_y: float
    left_grasp: bool
    right_grasp: bool
    holder: str
    coordinate_frame: str = CAMERA_PIXEL_FRAME

    def validate(self, width: int, height: int) -> None:
        values = (
            self.bottle_x,
            self.bottle_y,
            self.bottle_rotation_degrees,
            self.bottle_scale,
            self.left_wrist_x,
            self.left_wrist_y,
            self.right_wrist_x,
            self.right_wrist_y,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Ego control state must contain finite values")
        if not 0 <= self.bottle_x < width or not 0 <= self.bottle_y < height:
            raise ValueError("bottle center is outside the named camera frame")
        if not 0.5 <= self.bottle_scale <= 1.5:
            raise ValueError("bottle scale must be in [0.5, 1.5]")
        if self.coordinate_frame != CAMERA_PIXEL_FRAME:
            raise ValueError(f"control state frame must be {CAMERA_PIXEL_FRAME}")


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _mix(first: float, second: float, progress: float) -> float:
    return first + (second - first) * smoothstep(progress)


def _segment(progress: float, start: float, end: float) -> float:
    return (progress - start) / max(1e-9, end - start)


def ego_bottle_state(
    label: str,
    progress: float,
    *,
    width: int,
    height: int,
    start_x: float,
    start_y: float,
) -> EgoControlState:
    """Return an explicit, mutually distinguishable state in camera pixels."""

    if label not in SUPPORTED_ACTION_LABELS:
        raise ValueError(f"unsupported Ego bottle action: {label}")
    progress = min(1.0, max(0.0, progress))
    left_home = (0.22 * width, 0.88 * height)
    right_home = (0.78 * width, 0.88 * height)
    left = left_home
    right = right_home
    bottle = (start_x, start_y)
    rotation = 0.0
    scale = 1.0
    left_grasp = False
    right_grasp = False
    holder = "counter"

    if progress < 0.10:
        right = (start_x + 0.08 * width, start_y + 0.05 * height)
        right_grasp = True
        holder = "right_hand"
    elif label == "pour-bottle":
        right_grasp = True
        holder = "right_hand"
        if progress < 0.24:
            local = _segment(progress, 0.10, 0.24)
            bottle = (_mix(start_x, 0.62 * width, local), _mix(start_y, 0.43 * height, local))
            rotation = _mix(0.0, -12.0, local)
        elif progress < 0.58:
            local = _segment(progress, 0.24, 0.58)
            bottle = (0.66 * width, 0.40 * height)
            rotation = _mix(-12.0, 70.0, min(1.0, local / 0.35))
            left = (0.49 * width, 0.64 * height)
        elif progress < 0.70:
            local = _segment(progress, 0.58, 0.70)
            bottle = (_mix(0.66 * width, 0.63 * width, local), _mix(0.40 * height, 0.48 * height, local))
            rotation = _mix(70.0, 0.0, local)
        elif progress < 0.90:
            local = _segment(progress, 0.70, 0.90)
            bottle = (_mix(0.63 * width, start_x, local), _mix(0.48 * height, start_y, local))
            if progress >= 0.86:
                right_grasp = False
                holder = "counter"
        else:
            holder = "counter"
            right_grasp = False
            bottle = (start_x, start_y)
            local = _segment(progress, 0.90, 1.0)
            right = (_mix(start_x + 0.08 * width, right_home[0], local), _mix(start_y, right_home[1], local))
        if right_grasp:
            right = (bottle[0] + 0.08 * width, bottle[1] + 0.05 * height)
    elif label == "shake-bottle":
        center = (0.50 * width, 0.43 * height)
        if progress < 0.22:
            local = _segment(progress, 0.10, 0.22)
            bottle = (_mix(start_x, center[0], local), _mix(start_y, center[1], local))
            right_grasp = True
            holder = "right_hand"
            left = (_mix(left_home[0], center[0] - 0.09 * width, local), _mix(left_home[1], center[1], local))
        elif progress < 0.68:
            local = _segment(progress, 0.22, 0.68)
            offset = math.sin(local * 8.0 * math.pi) * 0.10 * width
            bottle = (center[0] + offset, center[1])
            rotation = math.sin(local * 8.0 * math.pi) * 12.0
            left_grasp = right_grasp = True
            holder = "both_hands"
            left = (bottle[0] - 0.10 * width, bottle[1] + 0.02 * height)
            right = (bottle[0] + 0.10 * width, bottle[1] + 0.02 * height)
        elif progress < 0.77:
            bottle = center
            right_grasp = True
            holder = "right_hand"
            left = (center[0] - 0.10 * width, center[1] + 0.02 * height)
            right = (center[0] + 0.10 * width, center[1] + 0.02 * height)
        elif progress < 0.91:
            local = _segment(progress, 0.77, 0.91)
            bottle = (_mix(center[0], start_x, local), _mix(center[1], start_y, local))
            right_grasp = progress < 0.88
            holder = "right_hand" if right_grasp else "counter"
            right = (bottle[0] + 0.08 * width, bottle[1] + 0.05 * height)
        else:
            bottle = (start_x, start_y)
            holder = "counter"
    elif label == "handover-bottle":
        center = (0.52 * width, 0.46 * height)
        if progress < 0.38:
            local = _segment(progress, 0.10, 0.38)
            bottle = (_mix(start_x, center[0], local), _mix(start_y, center[1], local))
            right_grasp = True
            holder = "right_hand"
            right = (bottle[0] + 0.08 * width, bottle[1] + 0.04 * height)
            left = (_mix(left_home[0], center[0] - 0.10 * width, local), _mix(left_home[1], center[1], local))
        elif progress < 0.53:
            bottle = center
            left_grasp = right_grasp = True
            holder = "both_hands"
            left = (center[0] - 0.10 * width, center[1] + 0.02 * height)
            right = (center[0] + 0.10 * width, center[1] + 0.02 * height)
        elif progress < 0.68:
            bottle = center
            left_grasp = True
            right_grasp = progress < 0.61
            holder = "both_hands" if right_grasp else "left_hand"
            left = (center[0] - 0.10 * width, center[1] + 0.02 * height)
            right = (center[0] + 0.10 * width, center[1] + 0.02 * height)
        else:
            local = _segment(progress, 0.68, 1.0)
            bottle = (_mix(center[0], 0.34 * width, local), _mix(center[1], 0.48 * height, local))
            left_grasp = True
            holder = "left_hand"
            left = (bottle[0] - 0.09 * width, bottle[1] + 0.03 * height)
            right = (_mix(center[0] + 0.10 * width, right_home[0], local), _mix(center[1], right_home[1], local))
    elif label == "place-bottle-rack":
        approach = (0.54 * width, 0.42 * height)
        rack_motion = smoothstep(_segment(progress, 0.26, 1.0))
        rack_x = _mix(0.68 * width, 0.95 * width, rack_motion)
        rack_top = (rack_x, 0.31 * height)
        rack_supported = (rack_x, 0.42 * height)
        if progress < 0.26:
            local = _segment(progress, 0.10, 0.26)
            bottle = (_mix(start_x, approach[0], local), _mix(start_y, approach[1], local))
            right_grasp = True
            holder = "right_hand"
        elif progress < 0.55:
            local = _segment(progress, 0.26, 0.55)
            bottle = (_mix(approach[0], rack_top[0], local), _mix(approach[1], rack_top[1], local))
            right_grasp = True
            holder = "right_hand"
            left = (
                _mix(left_home[0], rack_top[0] - 0.10 * width, local),
                _mix(left_home[1], rack_top[1] + 0.04 * height, local),
            )
        elif progress < 0.72:
            local = _segment(progress, 0.55, 0.72)
            bottle = (
                _mix(rack_top[0], rack_supported[0], local),
                _mix(rack_top[1], rack_supported[1], local),
            )
            left_grasp = right_grasp = True
            holder = "both_hands"
            left = (bottle[0] - 0.10 * width, bottle[1] + 0.01 * height)
        elif progress < 0.86:
            local = _segment(progress, 0.72, 0.86)
            bottle = rack_supported
            left_grasp = progress < 0.78
            right_grasp = progress < 0.81
            holder = "both_hands" if left_grasp else ("right_hand" if right_grasp else "dish_rack")
            left = (
                _mix(rack_supported[0] - 0.10 * width, left_home[0], local),
                _mix(rack_supported[1], left_home[1], local),
            )
        else:
            local = _segment(progress, 0.86, 1.0)
            bottle = rack_supported
            holder = "dish_rack"
            left = left_home
            right = (
                _mix(rack_supported[0] + 0.09 * width, right_home[0], local),
                _mix(rack_supported[1] + 0.03 * height, right_home[1], local),
            )
        if right_grasp:
            right = (bottle[0] + 0.09 * width, bottle[1] + 0.03 * height)
    elif label == "unscrew-bottle-cap":
        center = (0.52 * width, 0.49 * height)
        cap = (center[0], center[1] - 0.13 * height)
        if progress < 0.25:
            local = _segment(progress, 0.10, 0.25)
            bottle = (_mix(start_x, center[0], local), _mix(start_y, center[1], local))
            right_grasp = True
            holder = "right_hand"
        elif progress < 0.38:
            local = _segment(progress, 0.25, 0.38)
            bottle = center
            right_grasp = True
            holder = "right_hand"
            left = (_mix(left_home[0], cap[0], local), _mix(left_home[1], cap[1], local))
        elif progress < 0.70:
            local = _segment(progress, 0.38, 0.70)
            bottle = center
            rotation = 4.0 * math.sin(local * 6.0 * math.pi)
            left_grasp = right_grasp = True
            holder = "both_hands"
            left = (
                cap[0] + math.sin(local * 6.0 * math.pi) * 0.025 * width,
                cap[1] + math.cos(local * 6.0 * math.pi) * 0.015 * height,
            )
        elif progress < 0.82:
            local = _segment(progress, 0.70, 0.82)
            bottle = center
            right_grasp = True
            left_grasp = True
            holder = "right_hand"
            left = (cap[0], _mix(cap[1], cap[1] - 0.16 * height, local))
        else:
            local = _segment(progress, 0.82, 1.0)
            bottle = center
            right_grasp = True
            left_grasp = True
            holder = "right_hand"
            left = (
                _mix(cap[0], 0.31 * width, local),
                _mix(cap[1] - 0.16 * height, 0.39 * height, local),
            )
        right = (bottle[0] + 0.10 * width, bottle[1] + 0.02 * height)
    else:  # rinse-bottle
        faucet = (0.68 * width, 0.40 * height)
        if progress < 0.27:
            local = _segment(progress, 0.10, 0.27)
            bottle = (_mix(start_x, faucet[0], local), _mix(start_y, faucet[1], local))
            right_grasp = True
            holder = "right_hand"
        elif progress < 0.38:
            local = _segment(progress, 0.27, 0.38)
            bottle = faucet
            right_grasp = True
            holder = "right_hand"
            left = (
                _mix(left_home[0], 0.72 * width, local),
                _mix(left_home[1], 0.27 * height, local),
            )
        elif progress < 0.72:
            local = _segment(progress, 0.38, 0.72)
            bottle = (
                faucet[0] + math.sin(local * 6.0 * math.pi) * 0.025 * width,
                faucet[1] + math.cos(local * 4.0 * math.pi) * 0.025 * height,
            )
            rotation = 30.0 * math.sin(local * 6.0 * math.pi)
            right_grasp = True
            holder = "right_hand"
            left = (0.72 * width, 0.27 * height)
        elif progress < 0.86:
            local = _segment(progress, 0.72, 0.86)
            bottle = (
                _mix(faucet[0], 0.60 * width, local),
                _mix(faucet[1], 0.50 * height, local),
            )
            rotation = _mix(18.0, 0.0, local)
            right_grasp = True
            holder = "right_hand"
            left = (
                _mix(0.72 * width, left_home[0], local),
                _mix(0.27 * height, left_home[1], local),
            )
        else:
            bottle = (0.60 * width, 0.50 * height)
            right_grasp = True
            holder = "right_hand"
            left = left_home
        right = (bottle[0] + 0.10 * width, bottle[1] + 0.03 * height)

    state = EgoControlState(
        bottle_x=bottle[0],
        bottle_y=bottle[1],
        bottle_rotation_degrees=rotation,
        bottle_scale=scale,
        left_wrist_x=left[0],
        left_wrist_y=left[1],
        right_wrist_x=right[0],
        right_wrist_y=right[1],
        left_grasp=left_grasp,
        right_grasp=right_grasp,
        holder=holder,
    )
    state.validate(width, height)
    return state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parse_box(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(item.strip()) for item in value.split(","))
    if len(parts) != 4 or any(item < 0 for item in parts) or parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError("box must be x,y,width,height with positive size")
    return parts


def _parse_point(value: str) -> tuple[float, float]:
    parts = tuple(float(item.strip()) for item in value.split(","))
    if len(parts) != 2 or not all(math.isfinite(item) for item in parts):
        raise argparse.ArgumentTypeError("point must be finite x,y")
    return parts


def _parse_progress_range(value: str) -> tuple[str, float, float]:
    """Parse ``label=start,end`` without changing the named camera frame."""

    try:
        label, bounds = value.split("=", maxsplit=1)
        start_text, end_text = bounds.split(",", maxsplit=1)
        start = float(start_text)
        end = float(end_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "progress range must be label=start,end"
        ) from error
    if not label or label not in SUPPORTED_ACTION_LABELS:
        raise argparse.ArgumentTypeError(f"unsupported progress-range label: {label}")
    if not all(math.isfinite(item) for item in (start, end)):
        raise argparse.ArgumentTypeError("progress bounds must be finite")
    if not 0.0 <= start < end <= 1.0:
        raise argparse.ArgumentTypeError(
            "progress bounds must satisfy 0 <= start < end <= 1"
        )
    return label, start, end


def _decode(cv2: Any, path: Path) -> tuple[list[Any], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    return frames, fps


def _writer(ffmpeg: Path, output: Path, width: int, height: int, fps: int) -> Any:
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
            "-c:v", "libx264", "-crf", "12", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _render_robot_hand(
    cv2: Any,
    np: Any,
    frame: Any,
    *,
    wrist: tuple[float, float],
    base: tuple[float, float],
    grasp: bool,
) -> None:
    wrist_i = tuple(int(round(value)) for value in wrist)
    base_i = tuple(int(round(value)) for value in base)
    cv2.line(frame, base_i, wrist_i, (58, 65, 72), 60, cv2.LINE_AA)
    cv2.line(frame, base_i, wrist_i, (195, 205, 212), 43, cv2.LINE_AA)
    cv2.circle(frame, wrist_i, 31, (50, 57, 64), -1, cv2.LINE_AA)
    cv2.circle(frame, wrist_i, 24, (205, 215, 222), -1, cv2.LINE_AA)
    direction = np.asarray(wrist, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    direction = direction / max(norm, 1e-6)
    normal = np.asarray((-direction[1], direction[0]))
    spread = 7.0 if grasp else 15.0
    length = 44.0
    for index in range(5):
        offset = (index - 2) * spread
        start = np.asarray(wrist) + normal * offset
        end = start + direction * length + normal * (index - 2) * (1.5 if grasp else 3.0)
        cv2.line(
            frame,
            tuple(np.rint(start).astype(int)),
            tuple(np.rint(end).astype(int)),
            (226, 231, 235),
            8,
            cv2.LINE_AA,
        )


def _render_bottle(
    cv2: Any,
    np: Any,
    frame: Any,
    patch: Any,
    patch_mask: Any,
    *,
    center: tuple[float, float],
    rotation: float,
    scale: float,
) -> None:
    height, width = patch.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation, scale)
    transformed = cv2.warpAffine(
        patch, matrix, (width, height), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask = cv2.warpAffine(
        patch_mask,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, min(width, height) * 0.018))
    x0 = int(round(center[0] - width / 2))
    y0 = int(round(center[1] - height / 2))
    x1, y1 = x0 + width, y0 + height
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    px0, py0 = fx0 - x0, fy0 - y0
    px1, py1 = px0 + (fx1 - fx0), py0 + (fy1 - fy0)
    alpha = mask[py0:py1, px0:px1].astype(np.float32) / 255.0
    target = frame[fy0:fy1, fx0:fx1]
    source = transformed[py0:py1, px0:px1]
    frame[fy0:fy1, fx0:fx1] = np.rint(
        target.astype(np.float32) * (1.0 - alpha[..., None])
        + source.astype(np.float32) * alpha[..., None]
    ).astype(np.uint8)


def _extract_bottle_patch(
    cv2: Any,
    np: Any,
    frame: Any,
    box: tuple[int, int, int, int],
) -> tuple[Any, Any]:
    """Extract a bounded object matte without requiring a heavyweight segmenter."""

    x, y, width, height = box
    patch = frame[y : y + height, x : x + width].copy()
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (82, 60, 20), (132, 255, 255))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    blue_fraction = float(np.mean(mask > 0))
    if 0.04 <= blue_fraction <= 0.85:
        return patch, mask

    segmentation = np.zeros(frame.shape[:2], dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        frame,
        segmentation,
        (x, y, width, height),
        background,
        foreground,
        5,
        cv2.GC_INIT_WITH_RECT,
    )
    mask = np.where(
        (segmentation == cv2.GC_FGD) | (segmentation == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)[y : y + height, x : x + width]
    active_fraction = float(np.mean(mask > 0))
    if not 0.04 <= active_fraction <= 0.92:
        mask.fill(0)
        cv2.ellipse(
            mask,
            (width // 2, height // 2),
            (max(2, width // 3), max(2, height // 2 - 2)),
            0,
            0,
            360,
            255,
            -1,
        )
    return patch, mask


def _clean_interaction_frame(cv2: Any, np: Any, frame: Any) -> Any:
    """Soften source hands/bottle while retaining head-camera scene motion."""

    height, width = frame.shape[:2]
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(ycrcb, (0, 134, 80), (255, 180, 132))
    skin = cv2.bitwise_and(skin, cv2.inRange(hsv, (0, 35, 40), (25, 210, 255)))
    blue = cv2.inRange(hsv, (82, 75, 30), (132, 255, 255))
    skin[: int(0.38 * height)] = 0
    blue[: int(0.38 * height)] = 0
    blue[:, int(0.82 * width) :] = 0
    blue = cv2.morphologyEx(
        blue,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    blue = cv2.dilate(
        blue,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    cleaned = cv2.inpaint(frame, blue, 7, cv2.INPAINT_TELEA)
    skin = cv2.morphologyEx(
        skin,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    skin = cv2.dilate(
        skin,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
    )
    alpha = cv2.GaussianBlur(skin, (0, 0), 5.0).astype(np.float32) / 255.0
    alpha[: int(0.32 * height)] = 0.0
    blurred = cv2.GaussianBlur(cleaned, (0, 0), 9.0)
    return np.rint(
        cleaned.astype(np.float32) * (1.0 - alpha[..., None])
        + blurred.astype(np.float32) * alpha[..., None]
    ).astype(np.uint8)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--action-manifest", type=Path, required=True)
    parser.add_argument("--bottle-box", type=_parse_box, required=True)
    parser.add_argument("--bottle-frame", type=int, default=0)
    parser.add_argument("--start-center", type=_parse_point)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument(
        "--progress-range",
        type=_parse_progress_range,
        action="append",
        default=[],
        metavar="LABEL=START,END",
        help=(
            "Render one action over a bounded portion of its normalized trajectory; "
            "repeat once per label when preparing a continuation window."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    manifest_path = experiment / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Ego control experiment already exists: {manifest_path}")
    source = args.source_video.expanduser().resolve()
    action_manifest = args.action_manifest.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    for path in (source, action_manifest, ffmpeg):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required input is missing or empty: {path}")
    requested = json.loads(action_manifest.read_text())
    action_items = requested["actions"]
    action_labels = tuple(str(item["label"]) for item in action_items)
    actions = {str(item["label"]): item for item in action_items}
    if len(actions) != len(action_labels):
        raise ValueError("action manifest contains duplicate labels")
    unsupported = set(action_labels) - set(SUPPORTED_ACTION_LABELS)
    if unsupported:
        raise ValueError(f"unsupported Ego bottle plans: {sorted(unsupported)}")
    if not action_labels:
        raise ValueError("action manifest contains no Ego bottle plans")
    progress_ranges: dict[str, tuple[float, float]] = {}
    for label, start, end in args.progress_range:
        if label in progress_ranges:
            raise ValueError(f"duplicate progress range for {label}")
        if label not in actions:
            raise ValueError(f"progress range has no matching requested action: {label}")
        progress_ranges[label] = (start, end)

    import cv2
    import numpy as np

    np.random.seed(args.seed)
    frames, source_fps = _decode(cv2, source)
    if len(frames) != args.num_frames or abs(source_fps - args.fps) > 1e-6:
        raise ValueError("Ego source must match the requested exact frame count and FPS")
    height, width = frames[0].shape[:2]
    if not 0 <= args.bottle_frame < len(frames):
        raise ValueError("bottle-frame is outside the decoded source")
    x, y, box_width, box_height = args.bottle_box
    if x + box_width > width or y + box_height > height:
        raise ValueError("bottle box exceeds the source camera frame")
    anchor = frames[args.bottle_frame]
    bottle_patch, bottle_mask = _extract_bottle_patch(
        cv2, np, anchor, args.bottle_box
    )
    start_x, start_y = args.start_center or (
        x + box_width / 2,
        y + box_height / 2,
    )
    if not 0 <= start_x < width or not 0 <= start_y < height:
        raise ValueError("start-center is outside the source camera frame")

    records = []
    traces: dict[str, Any] = {}
    for label in action_labels:
        output = experiment / "variants" / label / "action-control.mp4"
        writer = _writer(ffmpeg, output, width, height, args.fps)
        trace = []
        try:
            for frame_index in range(args.num_frames):
                local_progress = frame_index / max(1, args.num_frames - 1)
                progress_start, progress_end = progress_ranges.get(label, (0.0, 1.0))
                progress = _mix(progress_start, progress_end, local_progress)
                state = ego_bottle_state(
                    label,
                    progress,
                    width=width,
                    height=height,
                    start_x=start_x,
                    start_y=start_y,
                )
                candidate = _clean_interaction_frame(
                    cv2, np, frames[frame_index]
                )
                _render_bottle(
                    cv2,
                    np,
                    candidate,
                    bottle_patch,
                    bottle_mask,
                    center=(state.bottle_x, state.bottle_y),
                    rotation=state.bottle_rotation_degrees,
                    scale=state.bottle_scale,
                )
                _render_robot_hand(
                    cv2,
                    np,
                    candidate,
                    wrist=(state.left_wrist_x, state.left_wrist_y),
                    base=(0.08 * width, 1.02 * height),
                    grasp=state.left_grasp,
                )
                _render_robot_hand(
                    cv2,
                    np,
                    candidate,
                    wrist=(state.right_wrist_x, state.right_wrist_y),
                    base=(0.92 * width, 1.02 * height),
                    grasp=state.right_grasp,
                )
                assert writer.stdin is not None
                writer.stdin.write(candidate.tobytes())
                trace.append({"frame": frame_index, "progress": progress, **asdict(state)})
        finally:
            if writer.stdin is not None:
                writer.stdin.close()
            if writer.wait():
                raise RuntimeError(f"ffmpeg failed for {label}")
        subprocess.run([str(ffmpeg), "-v", "error", "-i", str(output), "-f", "null", "-"], check=True)
        subprocess.run(
            [
                str(ffmpeg), "-y", "-v", "error", "-i", str(output),
                "-vf", "fps=1,scale=320:180,tile=5x1:padding=3:margin=3:color=black",
                "-frames:v", "1", str(output.parent / "contact-sheet.jpg"),
            ],
            check=True,
        )
        trace_path = output.parent / "trajectory.json"
        _write_json(trace_path, {"label": label, "trace": trace})
        traces[label] = np.asarray(
            [
                (
                    item["bottle_x"], item["bottle_y"], item["left_wrist_x"],
                    item["left_wrist_y"], item["right_wrist_x"], item["right_wrist_y"],
                )
                for item in trace
            ],
            dtype=np.float64,
        )
        records.append(
            {
                "label": label,
                "instruction": actions[label]["instruction"],
                "output": str(output),
                "output_sha256": _sha256(output),
                "trajectory": str(trace_path),
                "trajectory_sha256": _sha256(trace_path),
                "terminal_holder": trace[-1]["holder"],
            }
        )

    separations = []
    for left_index, left in enumerate(action_labels):
        for right in action_labels[left_index + 1 :]:
            difference = traces[left] - traces[right]
            rms = float(np.sqrt(np.mean(np.square(difference))))
            separations.append({"left": left, "right": right, "state_rms_pixels": rms})
    separation_floor = min(item["state_rms_pixels"] for item in separations)
    manifest = {
        "schema_version": "1.0.0",
        "method": "language_to_explicit_egocentric_two_hand_bottle_state_control_video",
        "status": "completed",
        "honest_status": "WORKING" if separation_floor >= 35.0 else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "opencv-python")
        },
        "gpu": {"used": False, "reason": "deterministic camera-frame control compilation"},
        "seed": args.seed,
        "coordinate_frame": CAMERA_PIXEL_FRAME,
        "inputs": {
            "source_video": {"path": str(source), "sha256": _sha256(source)},
            "action_manifest": {"path": str(action_manifest), "sha256": _sha256(action_manifest)},
            "bottle_box_xywh": list(args.bottle_box),
            "bottle_frame": args.bottle_frame,
            "start_center_xy": [start_x, start_y],
            "progress_ranges": {
                label: [start, end]
                for label, (start, end) in sorted(progress_ranges.items())
            },
        },
        "variants": records,
        "trajectory_separation": separations,
        "acceptance": {
            "all_outputs_decoded": True,
            "all_trajectories_explicit": True,
            "terminal_holders": {item["label"]: item["terminal_holder"] for item in records},
            "minimum_pairwise_state_rms_pixels": separation_floor,
            "trajectory_separation_passed": separation_floor >= 35.0,
        },
        "limitations": [
            "These are simplified deterministic control videos, not final generated outputs.",
            "Bottle and robot-hand graphics encode camera-frame motion and holder state, not 3D kinematics, liquid simulation or force/contact physics.",
            "Source head-camera motion is retained, while a bounded blur is only a control-stage softening of detected human-skin/blue-bottle pixels.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"experiment": str(experiment), "acceptance": manifest["acceptance"]}, indent=2))
    return 0 if manifest["honest_status"] == "WORKING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
