#!/usr/bin/env python3
"""Compose H3 body, H3 limb, flower, and human-negative layers independently."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.minimax_h3 import file_sha256  # noqa: E402


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state() -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "available": status.returncode == 0,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
        "error": status.stderr.strip() if status.returncode != 0 else None,
    }


def _video_info(cv2: Any, path: Path, *, decode: bool = False) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    result: dict[str, int | float] = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "reported_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    if decode:
        count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
        result["decoded_frames"] = count
    capture.release()
    return result


def _align_mask(cv2: Any, mask: Any, width: int, height: int) -> Any:
    scale = max(width / mask.shape[1], height / mask.shape[0])
    resized = cv2.resize(
        mask,
        (round(mask.shape[1] * scale), round(mask.shape[0] * scale)),
        interpolation=cv2.INTER_NEAREST,
    )
    left = max(0, (resized.shape[1] - width) // 2)
    top = max(0, (resized.shape[0] - height) // 2)
    return resized[top : top + height, left : left + width]


def _load_packed(np: Any, path: Path, key: str) -> Any:
    payload = np.load(path)
    height, width = int(payload["height"]), int(payload["width"])
    packed = payload[key]
    unpacked = np.unpackbits(packed, axis=1, bitorder="little")
    return unpacked[:, : height * width].reshape(len(packed), height, width).astype(bool)


def _fill_mask_holes(cv2: Any, np: Any, mask: Any) -> Any:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros(mask.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled.astype(bool)


def _drop_small_components(cv2: Any, np: Any, mask: Any, minimum: int) -> Any:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    result = np.zeros(mask.shape, dtype=bool)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= minimum:
            result = np.logical_or(result, labels == component)
    return result


def _stabilize_area_outliers(
    np: Any,
    masks: Any,
    *,
    ratio: float,
) -> list[int]:
    """Replace isolated one-frame mask expansions with temporal majority masks."""

    areas = np.count_nonzero(masks, axis=(1, 2)).astype(np.float64)
    replacements: list[tuple[int, Any]] = []
    for index in range(1, len(masks) - 1):
        neighbor_indices = [
            item
            for item in (index - 2, index - 1, index + 1, index + 2)
            if 0 <= item < len(masks)
        ]
        reference = float(np.median(areas[neighbor_indices]))
        if reference <= 0.0 or areas[index] <= ratio * reference:
            continue
        previous = masks[index - 1]
        current = masks[index]
        following = masks[index + 1]
        majority = np.logical_or(
            np.logical_and(previous, current),
            np.logical_or(
                np.logical_and(previous, following),
                np.logical_and(current, following),
            ),
        )
        replacements.append((index, majority))
    for index, replacement in replacements:
        masks[index] = replacement
    return [index for index, _ in replacements]


def _strict_flower_seed(cv2: Any, np: Any, frame: Any, safety: Any) -> Any:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue >= 28) & (hue <= 91) & (saturation >= 67) & (value >= 28)
    pink = (hue >= 145) & (hue <= 179) & (saturation >= 105) & (value >= 55)
    yellow = (hue >= 12) & (hue <= 35) & (saturation >= 105) & (value >= 70)
    colors = np.logical_or(np.logical_or(green, pink), yellow)
    mask = np.logical_and(colors, safety).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def _skin_like(cv2: Any, np: Any, frame: Any) -> Any:
    _, cr, cb = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb))
    blue, green, red = cv2.split(frame.astype(np.float32))
    return (
        (cr >= 132)
        & (cr <= 180)
        & (cb >= 75)
        & (cb <= 135)
        & (red > green * 1.03)
        & (green > blue * 0.90)
    )


def select_arm_skin_components(
    cv2: Any,
    np: Any,
    *,
    skin_mask: Any,
    arm_mask: Any,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    arm_dilation: int,
    minimum_area: int,
    maximum_area: int | None,
    minimum_arm_overlap: float,
) -> Any:
    """Keep skin components centred in a bounded arm-contact zone.

    This is a negative-mask selector, not a generic foreground segmenter.  A
    component must be large enough, have its centroid inside the declared
    camera-frame ROI, and overlap the tracked arm neighborhood.  Those three
    constraints prevent the wide skin reach from touching unrelated ribbons,
    wall decor, table flowers, or the right-side bouquet.
    """

    if skin_mask.shape != arm_mask.shape:
        raise ValueError("skin and arm masks must have identical geometry")
    height, width = skin_mask.shape
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ValueError("skin component ROI is outside the camera frame")
    if arm_dilation < 0 or minimum_area <= 0:
        raise ValueError("component dilation/area parameters are invalid")
    if maximum_area is not None and maximum_area < minimum_area:
        raise ValueError("maximum component area must exceed minimum area")
    if not 0.0 <= minimum_arm_overlap <= 1.0:
        raise ValueError("minimum arm overlap must be in [0, 1]")
    if arm_dilation:
        arm_support = cv2.dilate(
            arm_mask.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (arm_dilation * 2 + 1, arm_dilation * 2 + 1),
            ),
        ) > 0
    else:
        arm_support = arm_mask.astype(bool)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        skin_mask.astype(np.uint8), connectivity=8
    )
    selected = np.zeros(skin_mask.shape, dtype=bool)
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        if maximum_area is not None and area > maximum_area:
            continue
        centroid_x, centroid_y = centroids[component]
        if not (
            x_min <= centroid_x < x_max
            and y_min <= centroid_y < y_max
        ):
            continue
        current = labels == component
        overlap = float(np.count_nonzero(np.logical_and(current, arm_support))) / area
        if overlap >= minimum_arm_overlap:
            selected = np.logical_or(selected, current)
    return selected


def fill_selected_component_hulls(cv2: Any, np: Any, mask: Any) -> Any:
    """Fill each already-selected component's convex hull independently."""

    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    result = np.zeros(mask.shape, dtype=np.uint8)
    for component in range(1, count):
        current = (labels == component).astype(np.uint8)
        contours, _ = cv2.findContours(
            current, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        points = np.concatenate(contours, axis=0)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(result, hull, 1)
    return result.astype(bool)


def build_residual_arm_skin_support(
    cv2: Any,
    np: Any,
    *,
    frame: Any,
    search_alpha: Any,
    close_width: int,
    close_height: int,
    minimum_area: int,
    dilation: int,
) -> Any:
    """Find one smooth skin-connected forearm inside a broad reviewed track."""

    if frame.shape[:2] != search_alpha.shape:
        raise ValueError("frame and residual-arm search alpha must align")
    if close_width <= 0 or close_height <= 0:
        raise ValueError("residual-arm skin closing dimensions must be positive")
    if minimum_area <= 0:
        raise ValueError("residual-arm skin minimum area must be positive")
    if dilation < 0:
        raise ValueError("residual-arm skin dilation must be non-negative")
    search = search_alpha >= 0.02
    candidate = np.logical_and(_skin_like(cv2, np, frame), search)
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_width, close_height)
        ),
    ) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    eligible = [
        component
        for component in range(1, count)
        if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_area
    ]
    if not eligible:
        return np.zeros(search_alpha.shape, dtype=bool)
    largest = max(
        eligible,
        key=lambda component: int(stats[component, cv2.CC_STAT_AREA]),
    )
    support = labels == largest
    contours, _ = cv2.findContours(
        support.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    filled = np.zeros(search_alpha.shape, dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    if dilation:
        filled = cv2.dilate(
            filled,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilation * 2 + 1, dilation * 2 + 1),
            ),
        )
    return np.logical_and(filled > 0, search)


def interpolate_polygon_keyframes(
    np: Any,
    keyframes: list[dict[str, Any]],
    frame_index: int,
) -> tuple[Any | None, float]:
    """Interpolate one reviewed camera-frame polygon track.

    Each keyframe contains ``frame``, ``points`` and an optional ``strength``.
    All points remain in the declared camera pixel frame; no pose/world-frame
    transform is implied.  Zero-strength endpoint keyframes provide a smooth
    temporal fade without changing the polygon vertex count.
    """

    if not keyframes:
        return None, 0.0
    ordered = sorted(keyframes, key=lambda item: int(item["frame"]))
    frames = [int(item["frame"]) for item in ordered]
    if len(set(frames)) != len(frames):
        raise ValueError("polygon keyframe indices must be unique")
    points = [np.asarray(item["points"], dtype=np.float32) for item in ordered]
    if any(value.ndim != 2 or value.shape[1] != 2 for value in points):
        raise ValueError("polygon points must have Nx2 geometry")
    if any(value.shape != points[0].shape for value in points[1:]):
        raise ValueError("polygon keyframes must use a constant vertex count")
    strengths = [float(item.get("strength", 1.0)) for item in ordered]
    if any(not 0.0 <= value <= 1.0 for value in strengths):
        raise ValueError("polygon keyframe strengths must be in [0, 1]")
    if frame_index < frames[0] or frame_index > frames[-1]:
        return None, 0.0
    if frame_index in frames:
        index = frames.index(frame_index)
        return points[index].copy(), strengths[index]
    right = next(index for index, value in enumerate(frames) if value > frame_index)
    left = right - 1
    fraction = (frame_index - frames[left]) / (frames[right] - frames[left])
    polygon = points[left] * (1.0 - fraction) + points[right] * fraction
    strength = strengths[left] * (1.0 - fraction) + strengths[right] * fraction
    return polygon, float(strength)


def build_tracked_polygon_alpha(
    cv2: Any,
    np: Any,
    *,
    shape: tuple[int, int],
    tracks: list[dict[str, Any]],
    frame_index: int,
    feather_sigma: float,
) -> Any:
    """Build the union alpha of reviewed, temporally interpolated polygons."""

    if feather_sigma < 0:
        raise ValueError("polygon feather sigma must be non-negative")
    height, width = shape
    alpha = np.zeros((height, width), dtype=np.float32)
    for track in tracks:
        keyframes = track.get("keyframes")
        if not isinstance(keyframes, list):
            raise ValueError("each polygon track must contain a keyframes list")
        polygon, strength = interpolate_polygon_keyframes(
            np, keyframes, frame_index
        )
        if polygon is None or strength <= 0.0:
            continue
        if (
            np.any(polygon[:, 0] < 0)
            or np.any(polygon[:, 0] >= width)
            or np.any(polygon[:, 1] < 0)
            or np.any(polygon[:, 1] >= height)
        ):
            raise ValueError("polygon keyframe lies outside the camera frame")
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
        current = mask.astype(np.float32) / 255.0
        if feather_sigma:
            current = cv2.GaussianBlur(current, (0, 0), feather_sigma)
            current[mask > 0] = 1.0
        alpha = np.maximum(alpha, current * strength)
    return np.clip(alpha, 0.0, 1.0)


def build_tracked_robot_arm_material(
    cv2: Any,
    np: Any,
    *,
    frame: Any,
    tracks: list[dict[str, Any]],
    frame_index: int,
    style: str,
) -> Any:
    """Render a deterministic silver/graphite material inside tracked arms.

    The source frame supplies only luminance, so motion and illumination stay
    temporally coherent while skin chroma is replaced by a cool metal palette.
    Soft dark edge and longitudinal seams make the reviewed polygon read as an
    articulated robot forearm instead of a flat background patch.
    """

    if style not in {"silver", "graphite"}:
        raise ValueError("robot arm material style must be silver or graphite")
    height, width = frame.shape[:2]
    material = frame.astype(np.float32).copy()
    luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    yy, xx = np.indices((height, width), dtype=np.float32)
    for track in tracks:
        polygon, strength = interpolate_polygon_keyframes(
            np, track["keyframes"], frame_index
        )
        if polygon is None or strength <= 0.0:
            continue
        points = np.rint(polygon).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)
        u = np.clip((xx - x_min) / max(1.0, float(x_max - x_min)), 0.0, 1.0)
        v = np.clip((yy - y_min) / max(1.0, float(y_max - y_min)), 0.0, 1.0)
        if style == "silver":
            base = np.clip(
                luminance * 0.62
                + 72.0
                + 22.0 * np.exp(-((v - 0.30) / 0.18) ** 2),
                48.0,
                225.0,
            )
        else:
            base = np.clip(
                luminance * 0.48
                + 58.0
                + 16.0 * np.exp(-((v - 0.28) / 0.20) ** 2),
                35.0,
                190.0,
            )
        target = np.stack(
            [
                np.clip(base * 1.08 + 4.0, 0.0, 255.0),
                base,
                np.clip(base * 0.92, 0.0, 255.0),
            ],
            axis=2,
        )
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        edge = np.logical_and(mask > 0, distance < 1.6)
        seam = np.logical_and(
            mask > 0,
            np.logical_or(np.abs(u - 0.42) < 0.010, np.abs(u - 0.73) < 0.009),
        )
        target[edge] = 0.55 * target[edge] + 0.45 * np.asarray(
            [58.0, 62.0, 66.0], dtype=np.float32
        )
        target[seam] = 0.65 * target[seam] + 0.35 * np.asarray(
            [52.0, 57.0, 62.0], dtype=np.float32
        )
        material[mask > 0] = target[mask > 0]
    return np.clip(material, 0.0, 255.0).astype(np.uint8)


def build_layer_masks(
    cv2: Any,
    np: Any,
    *,
    source: Any,
    generated: Any,
    body_mask: Any,
    wrist_mask: Any,
    robot_limb_mask: Any,
    generated_flower_instance_mask: Any,
    source_person_semantic_mask: Any,
    source_arms: Any,
    source_hands: Any,
    flower_instance_mask: Any,
    safety_mask: Any,
    limb_corridor_dilation: int,
    body_neighborhood_dilation: int,
    limb_consensus_dilation: int = 3,
    generated_flower_support_radius: int = 24,
    generated_flower_limb_radius: int = 56,
    generated_flower_min_component: int = 12,
) -> tuple[Any, Any, Any, Any, dict[str, float]]:
    """Build robot, flower, and source-skin-negative masks without conflation."""

    if robot_limb_mask is None:
        if limb_corridor_dilation < 0:
            arm_corridor = np.ones(source_arms.shape, dtype=bool)
        else:
            arm_corridor = cv2.dilate(
                source_arms.astype(np.uint8) * 255,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        limb_corridor_dilation * 2 + 1,
                        limb_corridor_dilation * 2 + 1,
                    ),
                ),
            ) > 0
        body_neighborhood = cv2.dilate(
            body_mask.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    body_neighborhood_dilation * 2 + 1,
                    body_neighborhood_dilation * 2 + 1,
                ),
            ),
        ) > 0
        limb_extra = np.logical_and(wrist_mask.astype(bool), arm_corridor)
        limb_extra = np.logical_and(limb_extra, body_neighborhood)
    else:
        raw_limb = robot_limb_mask.astype(bool).copy()
        consensus = cv2.dilate(
            wrist_mask.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    limb_consensus_dilation * 2 + 1,
                    limb_consensus_dilation * 2 + 1,
                ),
            ),
        ) > 0
        # Independent limb tracking occasionally expands into the blocky H3
        # human remnant. Keep only pixels also supported by the separately
        # tracked robot/wrist object.
        limb_extra = np.logical_and(raw_limb, consensus)
    if generated_flower_instance_mask is not None:
        generated_flower_core = _strict_flower_seed(
            cv2, np, generated, safety_mask
        )
        generated_flower_support = cv2.dilate(
            generated_flower_core.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    generated_flower_support_radius * 2 + 1,
                    generated_flower_support_radius * 2 + 1,
                ),
            ),
        ) > 0
        limb_neighborhood = cv2.dilate(
            limb_extra.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    generated_flower_limb_radius * 2 + 1,
                    generated_flower_limb_radius * 2 + 1,
                ),
            ),
        ) > 0
        # Instance tracking supplies temporal continuity, while the colour and
        # limb gates prevent static plants and flat grey/teal H3 blocks from
        # being reintroduced as foreground flowers.
        generated_flowers = np.logical_and(
            generated_flower_instance_mask.astype(bool),
            generated_flower_support,
        )
        generated_flowers = np.logical_and(
            generated_flowers,
            limb_neighborhood,
        )
        generated_flowers = _drop_small_components(
            cv2,
            np,
            generated_flowers,
            generated_flower_min_component,
        )
    else:
        generated_flower_core = _strict_flower_seed(
            cv2, np, generated, safety_mask
        )
        generated_flower_support = cv2.dilate(
            generated_flower_core.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49)),
        ) > 0
        flower_reference = np.logical_or(wrist_mask.astype(bool), limb_extra)
        generated_flowers = np.logical_and(
            flower_reference, generated_flower_support
        )
    robot = np.logical_or(body_mask.astype(bool), limb_extra)
    robot = np.logical_or(robot, generated_flowers)

    strict_flowers = _strict_flower_seed(cv2, np, source, safety_mask)
    human_geometry = np.logical_or(
        source_person_semantic_mask.astype(bool), source_arms.astype(bool)
    )
    source_skin = np.logical_and(
        _skin_like(cv2, np, source), human_geometry
    )
    skin_negative = cv2.dilate(
        source_skin.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    flower_candidates = np.logical_or(
        flower_instance_mask.astype(bool), strict_flowers
    )
    flowers = np.logical_and(
        flower_candidates, np.logical_not(skin_negative.copy())
    )
    flowers = np.logical_and(
        flowers, np.logical_not(generated_flowers.copy())
    )
    return robot, flowers, skin_negative, generated_flowers, {
        "body_fraction": float(np.mean(body_mask)),
        "raw_limb_fraction": float(
            np.mean(robot_limb_mask) if robot_limb_mask is not None else np.mean(wrist_mask)
        ),
        "limb_extra_fraction": float(np.mean(limb_extra)),
        "robot_fraction": float(np.mean(robot)),
        "generated_flower_fraction": float(np.mean(generated_flowers)),
        "generated_flower_instance_fraction": float(
            np.mean(generated_flower_instance_mask)
        ),
        "flower_instance_fraction": float(np.mean(flower_instance_mask)),
        "strict_flower_fraction": float(np.mean(strict_flowers)),
        "flower_fraction": float(np.mean(flowers)),
        "skin_negative_fraction": float(np.mean(skin_negative)),
    }


def compose_layered_frame(
    cv2: Any,
    np: Any,
    *,
    source: Any,
    generated: Any,
    clean_plate: Any,
    source_person_mask: Any,
    robot_mask: Any,
    flower_mask: Any,
    robot_dilation: int,
    person_feather_sigma: float,
    background_method: str = "alpha",
    person_dilation: int = 0,
) -> tuple[Any, Any]:
    person = source_person_mask.astype(bool)
    if person_dilation:
        person = cv2.dilate(
            person.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (person_dilation * 2 + 1, person_dilation * 2 + 1),
            ),
        ) > 0
    if background_method == "seamless":
        ys, xs = np.where(person)
        if len(xs):
            center = (
                int(round((int(xs.min()) + int(xs.max())) / 2)),
                int(round((int(ys.min()) + int(ys.max())) / 2)),
            )
            background = cv2.seamlessClone(
                clean_plate,
                source,
                person.astype(np.uint8) * 255,
                center,
                cv2.NORMAL_CLONE,
            )
        else:
            background = source.copy()
    elif background_method == "alpha":
        person_alpha = person.astype(np.float32)
        if person_feather_sigma:
            person_alpha = cv2.GaussianBlur(person_alpha, (0, 0), person_feather_sigma)
            person_alpha[person] = 1.0
        background = np.rint(
            clean_plate.astype(np.float32) * person_alpha[..., None]
            + source.astype(np.float32) * (1.0 - person_alpha[..., None])
        ).astype(np.uint8)
    else:
        raise ValueError(f"unknown background method: {background_method}")
    keep = robot_mask.astype(np.uint8) * 255
    if robot_dilation:
        keep = cv2.dilate(
            keep,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (robot_dilation * 2 + 1, robot_dilation * 2 + 1)
            ),
        )
    robot_alpha = cv2.GaussianBlur(keep, (5, 5), 0.8).astype(np.float32) / 255.0
    robot_alpha[robot_mask.astype(bool)] = 1.0
    result = np.rint(
        generated.astype(np.float32) * robot_alpha[..., None]
        + background.astype(np.float32) * (1.0 - robot_alpha[..., None])
    ).astype(np.uint8)
    result[flower_mask.astype(bool)] = source[flower_mask.astype(bool)]
    return result, background


def build_conservative_arm_shadow_alpha(
    cv2: Any,
    np: Any,
    *,
    generated: Any,
    clean_plate: Any,
    safety_mask: Any,
    protected_mask: Any,
    arm_mask: Any,
    protect_radius: int,
    cleanup_radius: int,
    maximum_strength: float,
    neutral_chroma_limit: float,
    difference_threshold: float,
    feather_sigma: float,
    skin_strength: float = 0.0,
    skin_to_core_mask: Any | None = None,
    skin_safety_mask: Any | None = None,
    skin_cleanup_radius: int | None = None,
    skin_candidate_mask: Any | None = None,
    skin_arm_neighborhood_mask: Any | None = None,
) -> Any:
    """Return a bounded alpha that only attenuates neutral arm-adjacent shadow.

    The generated frame stays authoritative.  Pixels are eligible only when
    they are inside the fixed person safety region, close to a tracked arm, far
    enough outside the temporally protected robot/object matte, neutral rather
    than flower-coloured, and measurably different from the fixed clean plate.
    """

    if generated.shape != clean_plate.shape or generated.ndim != 3:
        raise ValueError("generated and clean plate must have identical HxWx3 geometry")
    if safety_mask.shape != generated.shape[:2]:
        raise ValueError("safety mask geometry does not match frame")
    if protected_mask.shape != safety_mask.shape or arm_mask.shape != safety_mask.shape:
        raise ValueError("protected/arm mask geometry does not match frame")
    if skin_to_core_mask is not None and skin_to_core_mask.shape != safety_mask.shape:
        raise ValueError("skin-to-core mask geometry does not match frame")
    if skin_safety_mask is not None and skin_safety_mask.shape != safety_mask.shape:
        raise ValueError("skin safety mask geometry does not match frame")
    if skin_candidate_mask is not None and skin_candidate_mask.shape != safety_mask.shape:
        raise ValueError("skin candidate mask geometry does not match frame")
    if (
        skin_arm_neighborhood_mask is not None
        and skin_arm_neighborhood_mask.shape != safety_mask.shape
    ):
        raise ValueError("skin arm-neighborhood mask geometry does not match frame")
    if protect_radius < 0 or cleanup_radius <= protect_radius:
        raise ValueError("cleanup radius must exceed a non-negative protect radius")
    if skin_cleanup_radius is not None and skin_cleanup_radius <= protect_radius:
        raise ValueError("skin cleanup radius must exceed protect radius")
    if not 0.0 <= maximum_strength <= 1.0:
        raise ValueError("maximum strength must be in [0, 1]")
    if not 0.0 <= skin_strength <= 1.0:
        raise ValueError("skin strength must be in [0, 1]")
    if neutral_chroma_limit <= 0 or difference_threshold < 0 or feather_sigma < 0:
        raise ValueError("colour and feather parameters must be non-negative")

    def dilate(mask: Any, radius: int) -> Any:
        if radius == 0:
            return mask.astype(bool).copy()
        return cv2.dilate(
            mask.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
            ),
        ) > 0

    protected_core = protected_mask.astype(bool)
    protected = dilate(protected_core, protect_radius)
    arm_neighborhood = dilate(arm_mask, cleanup_radius)
    skin_arm_neighborhood = (
        dilate(
            arm_mask,
            cleanup_radius if skin_cleanup_radius is None else skin_cleanup_radius,
        )
        if skin_arm_neighborhood_mask is None
        else skin_arm_neighborhood_mask.astype(bool)
    )
    skin_safety = (
        safety_mask.astype(bool)
        if skin_safety_mask is None
        else skin_safety_mask.astype(bool)
    )
    pixels = generated.astype(np.float32)
    plate = clean_plate.astype(np.float32)
    chroma = np.max(pixels, axis=2) - np.min(pixels, axis=2)
    brightness = np.mean(pixels, axis=2)
    plate_difference = np.mean(np.abs(pixels - plate), axis=2)
    neutral_confidence = np.clip(
        (neutral_chroma_limit + 20.0 - chroma) / 40.0,
        0.0,
        1.0,
    )
    difference_confidence = np.clip(
        (plate_difference - difference_threshold) / 25.0,
        0.0,
        1.0,
    )
    common_eligible = np.logical_and(safety_mask.astype(bool), arm_neighborhood)
    common_eligible = np.logical_and(common_eligible, brightness >= 24.0)
    common_eligible = np.logical_and(common_eligible, brightness <= 242.0)
    neutral_eligible = np.logical_and(
        common_eligible, np.logical_not(protected.copy())
    )
    neutral_alpha = (
        neutral_eligible.astype(np.float32)
        * neutral_confidence
        * difference_confidence
        * float(maximum_strength)
    )
    if skin_candidate_mask is None:
        skin = _skin_like(cv2, np, generated)
        skin = cv2.morphologyEx(
            skin.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        ) > 0
    else:
        skin = skin_candidate_mask.astype(bool)
    skin_to_core = (
        np.zeros(safety_mask.shape, dtype=bool)
        if skin_to_core_mask is None
        else np.logical_and(
            skin_to_core_mask.astype(bool), safety_mask.astype(bool)
        )
    )
    skin_blocked = np.logical_and(
        protected, np.logical_not(skin_to_core.copy())
    )
    skin_common_eligible = np.logical_and(skin_safety, skin_arm_neighborhood)
    skin_common_eligible = np.logical_and(skin_common_eligible, brightness >= 24.0)
    skin_common_eligible = np.logical_and(skin_common_eligible, brightness <= 242.0)
    skin_eligible = np.logical_and(
        skin_common_eligible, np.logical_not(skin_blocked.copy())
    )
    skin_eligible = np.logical_and(
        skin_eligible, np.logical_not(protected_core.copy())
    )
    skin_alpha = (
        skin_eligible.astype(np.float32)
        * skin.astype(np.float32)
        * np.maximum(difference_confidence, 0.45)
        * float(skin_strength)
    )
    if feather_sigma:
        neutral_alpha = cv2.GaussianBlur(neutral_alpha, (0, 0), feather_sigma)
        skin_alpha = cv2.GaussianBlur(skin_alpha, (0, 0), feather_sigma)
    neutral_alpha[protected] = 0.0
    skin_alpha[skin_blocked] = 0.0
    skin_alpha[protected_core] = 0.0
    alpha = np.maximum(neutral_alpha, skin_alpha)
    alpha[protected_core] = 0.0
    edit_safety = np.logical_or(safety_mask.astype(bool), skin_safety)
    alpha[np.logical_not(edit_safety)] = 0.0
    return np.clip(alpha, 0.0, max(maximum_strength, skin_strength))


def apply_conservative_arm_shadow_cleanup(
    np: Any,
    *,
    generated: Any,
    clean_plate: Any,
    alpha: Any,
    protected_mask: Any,
) -> Any:
    """Blend a fixed plate only under the bounded alpha and restore protection."""

    if generated.shape != clean_plate.shape or alpha.shape != generated.shape[:2]:
        raise ValueError("cleanup inputs have incompatible geometry")
    result = np.rint(
        generated.astype(np.float32) * (1.0 - alpha[..., None])
        + clean_plate.astype(np.float32) * alpha[..., None]
    ).astype(np.uint8)
    result[protected_mask.astype(bool)] = generated[protected_mask.astype(bool)]
    return result


def _writer(ffmpeg: Path, output: Path, width: int, height: int, fps: float) -> Any:
    return subprocess.Popen(
        [
            str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _annotate(cv2: Any, frame: Any, label: str) -> Any:
    result = frame.copy()
    cv2.putText(
        result, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return result


def _sheet(cv2: Any, rows: list[list[Any]]) -> Any:
    return cv2.vconcat([cv2.hconcat(row) for row in rows])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-video", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--source-safety-mask", type=Path, required=True)
    parser.add_argument("--clean-plate", type=Path, required=True)
    parser.add_argument("--source-person-masks", type=Path, required=True)
    parser.add_argument("--source-limb-masks", type=Path, required=True)
    parser.add_argument("--robot-body-masks", type=Path, required=True)
    parser.add_argument("--robot-wrist-masks", type=Path, required=True)
    parser.add_argument("--robot-limb-masks", type=Path, required=True)
    parser.add_argument("--generated-flower-instance-masks", type=Path, required=True)
    parser.add_argument("--flower-instance-masks", type=Path, required=True)
    parser.add_argument("--previous-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-name",
        default="minimax-h3-epl-full-27s-layered.mp4",
    )
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument(
        "--limb-corridor-dilation",
        type=int,
        default=-1,
        help="Negative disables the pose-arm corridor and uses the body-neighborhood gate.",
    )
    parser.add_argument("--body-neighborhood-dilation", type=int, default=48)
    parser.add_argument("--limb-consensus-dilation", type=int, default=3)
    parser.add_argument("--generated-flower-support-radius", type=int, default=24)
    parser.add_argument("--generated-flower-limb-radius", type=int, default=56)
    parser.add_argument("--generated-flower-min-component", type=int, default=150)
    parser.add_argument("--generated-flower-outlier-ratio", type=float, default=1.65)
    parser.add_argument("--robot-dilation", type=int, default=1)
    parser.add_argument("--person-feather-sigma", type=float, default=7.0)
    parser.add_argument(
        "--background-method",
        choices=("alpha", "seamless"),
        default="alpha",
    )
    parser.add_argument("--person-dilation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--human-review", choices=("pending", "passed", "failed"), default="pending")
    return parser


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"experiment already exists: {output_dir}")
    for relative in ("logs", "review", "provenance/execution-sources"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    frozen_source = output_dir / "provenance/execution-sources" / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), frozen_source)
    paths = {
        "generated_video": args.generated_video.expanduser().resolve(),
        "source_video": args.source_video.expanduser().resolve(),
        "source_safety_mask": args.source_safety_mask.expanduser().resolve(),
        "clean_plate": args.clean_plate.expanduser().resolve(),
        "source_person_masks": args.source_person_masks.expanduser().resolve(),
        "source_limb_masks": args.source_limb_masks.expanduser().resolve(),
        "robot_body_masks": args.robot_body_masks.expanduser().resolve(),
        "robot_wrist_masks": args.robot_wrist_masks.expanduser().resolve(),
        "robot_limb_masks": args.robot_limb_masks.expanduser().resolve(),
        "generated_flower_instance_masks": (
            args.generated_flower_instance_masks.expanduser().resolve()
        ),
        "flower_instance_masks": args.flower_instance_masks.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    if args.previous_output:
        paths["previous_output"] = args.previous_output.expanduser().resolve()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    command = [sys.executable, *sys.argv]
    (output_dir / "command.sh").write_text(shlex.join(command) + "\n")
    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "honest_status": "PARTIAL",
        "scope": "full 660-frame H3 layered shadow/hand/flower recomposition",
        "command": command,
        "command_shell": shlex.join(command),
        "seed": args.seed,
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "execution_source": {"path": str(frozen_source), "sha256": file_sha256(frozen_source)},
        "git": _git_state(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": {
            "used": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mode": "CPU composition from pinned mask layers",
        },
        "packages": {
            name: _package_version(name) for name in ("numpy", "opencv-python")
        },
    }
    _write_json(output_dir / "manifest.json", record)

    generated_info = _video_info(cv2, paths["generated_video"])
    source_info = _video_info(cv2, paths["source_video"])
    if generated_info != source_info:
        raise RuntimeError("generated and source videos have different geometry/timing")
    if int(generated_info["reported_frames"]) != 660:
        raise RuntimeError("expected 660-frame videos")
    width = int(generated_info["width"])
    height = int(generated_info["height"])
    fps = float(generated_info["fps"])

    body = _load_packed(np, paths["robot_body_masks"], "packed")
    wrist = _load_packed(np, paths["robot_wrist_masks"], "packed")
    robot_limbs = _load_packed(np, paths["robot_limb_masks"], "packed")
    generated_flower_instances = _load_packed(
        np, paths["generated_flower_instance_masks"], "packed"
    )
    generated_flower_outlier_frames = _stabilize_area_outliers(
        np,
        generated_flower_instances,
        ratio=args.generated_flower_outlier_ratio,
    )
    source_person = _load_packed(np, paths["source_person_masks"], "packed")
    source_arms = _load_packed(np, paths["source_limb_masks"], "arms_packed")
    source_hands = _load_packed(np, paths["source_limb_masks"], "hands_packed")
    flower_instances = _load_packed(np, paths["flower_instance_masks"], "packed")
    expected_shape = (660, height, width)
    for name, masks in (
        ("body", body), ("wrist", wrist), ("robot_limbs", robot_limbs),
        ("generated_flower_instances", generated_flower_instances),
        ("source_person", source_person),
        ("source_arms", source_arms), ("source_hands", source_hands),
        ("flower_instances", flower_instances),
    ):
        if masks.shape != expected_shape:
            raise RuntimeError(f"{name} masks have wrong geometry: {masks.shape}")
    safety_raw = cv2.imread(str(paths["source_safety_mask"]), cv2.IMREAD_GRAYSCALE)
    clean_plate = cv2.imread(str(paths["clean_plate"]), cv2.IMREAD_COLOR)
    if safety_raw is None or clean_plate is None:
        raise RuntimeError("cannot decode safety mask or clean plate")
    safety = _align_mask(cv2, safety_raw, width, height) >= 127
    flower_safety = cv2.dilate(
        safety.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)),
    ) > 0
    if clean_plate.shape[:2] != (height, width):
        raise RuntimeError("clean plate geometry does not match videos")
    source_person_semantic = source_person.copy()
    source_person = np.stack(
        [
            np.logical_or(
                np.logical_or(_fill_mask_holes(cv2, np, mask), safety),
                source_arms[index],
            )
            for index, mask in enumerate(source_person_semantic)
        ],
        axis=0,
    )

    generated_capture = cv2.VideoCapture(str(paths["generated_video"]))
    source_capture = cv2.VideoCapture(str(paths["source_video"]))
    previous_capture = (
        cv2.VideoCapture(str(paths["previous_output"])) if "previous_output" in paths else None
    )
    output_video = output_dir / args.output_name
    writer = _writer(paths["ffmpeg"], output_video, width, height, fps)
    review_indices = set(int(item) for item in np.linspace(0, 659, 28, dtype=np.int32))
    risk_indices = {48, 72, 96, 120, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 659}
    review_frames: dict[int, list[Any]] = {}
    risk_frames: dict[int, list[Any]] = {}
    metric_rows = []
    prior_robot = None
    for index in range(660):
        ok_generated, generated = generated_capture.read()
        ok_source, source = source_capture.read()
        if not ok_generated or not ok_source:
            raise RuntimeError(f"video decode ended on frame {index}")
        previous = None
        if previous_capture is not None:
            ok_previous, previous = previous_capture.read()
            if not ok_previous:
                raise RuntimeError(f"previous output decode ended on frame {index}")
        robot, flowers, skin_negative, generated_flowers, layer_metrics = (
            build_layer_masks(
                cv2,
                np,
                source=source,
                generated=generated,
                body_mask=body[index],
                wrist_mask=wrist[index],
                robot_limb_mask=robot_limbs[index],
                generated_flower_instance_mask=generated_flower_instances[index],
                source_person_semantic_mask=source_person_semantic[index],
                source_arms=source_arms[index],
                source_hands=source_hands[index],
                flower_instance_mask=flower_instances[index],
                safety_mask=flower_safety,
                limb_corridor_dilation=args.limb_corridor_dilation,
                body_neighborhood_dilation=args.body_neighborhood_dilation,
                limb_consensus_dilation=args.limb_consensus_dilation,
                generated_flower_support_radius=args.generated_flower_support_radius,
                generated_flower_limb_radius=args.generated_flower_limb_radius,
                generated_flower_min_component=args.generated_flower_min_component,
            )
        )
        result, background = compose_layered_frame(
            cv2,
            np,
            source=source,
            generated=generated,
            clean_plate=clean_plate,
            source_person_mask=source_person[index],
            robot_mask=robot,
            flower_mask=flowers,
            robot_dilation=args.robot_dilation,
            person_feather_sigma=args.person_feather_sigma,
            background_method=args.background_method,
            person_dilation=args.person_dilation,
        )
        assert writer.stdin is not None
        writer.stdin.write(result.tobytes())

        strict = np.logical_and(
            _strict_flower_seed(cv2, np, source, flower_safety),
            np.logical_not(skin_negative.copy()),
        )
        strict = np.logical_and(
            strict, np.logical_not(generated_flowers.copy())
        )
        limb_expected = np.logical_and(robot, np.logical_not(body[index].copy()))
        limb_expected = np.logical_and(
            limb_expected, np.logical_not(flowers.copy())
        )
        body_expected = np.logical_and(
            body[index], np.logical_not(flowers.copy())
        )
        human_residual = np.logical_and(
            skin_negative, np.logical_not(robot.copy())
        )
        human_residual = np.logical_and(
            human_residual, np.logical_not(flowers.copy())
        )
        robot_halo = (
            cv2.dilate(
                robot.astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            ) > 0
        )
        halo_region = np.logical_and(safety, np.logical_not(robot_halo))
        halo_region = np.logical_and(
            halo_region, np.logical_not(flowers.copy())
        )
        body_exact = (
            float(np.mean(np.all(result[body_expected] == generated[body_expected], axis=1)))
            if np.any(body_expected) else 1.0
        )
        limb_exact = (
            float(np.mean(np.all(result[limb_expected] == generated[limb_expected], axis=1)))
            if np.any(limb_expected) else 1.0
        )
        flower_exact = (
            float(np.mean(np.all(result[flowers] == source[flowers], axis=1)))
            if np.any(flowers) else 1.0
        )
        generated_flower_exact = (
            float(
                np.mean(
                    np.all(
                        result[generated_flowers] == generated[generated_flowers],
                        axis=1,
                    )
                )
            )
            if np.any(generated_flowers) else 1.0
        )
        strict_recall = (
            float(np.mean(np.all(result[strict] == source[strict], axis=1)))
            if np.any(strict) else 1.0
        )
        human_retained = (
            float(np.mean(np.all(result[human_residual] == source[human_residual], axis=1)))
            if np.any(human_residual) else 0.0
        )
        halo_mae = (
            float(np.mean(np.abs(result[halo_region].astype(np.float32) - background[halo_region].astype(np.float32))))
            if np.any(halo_region) else 0.0
        )
        robot_iou = 1.0
        if prior_robot is not None:
            union_mask = np.logical_or(robot, prior_robot)
            intersection_mask = np.logical_and(robot, prior_robot)
            union = np.count_nonzero(union_mask)
            robot_iou = (
                float(np.count_nonzero(intersection_mask) / union) if union else 1.0
            )
        prior_robot = robot
        metric_rows.append(
            {
                "frame": index,
                **layer_metrics,
                "body_exact": body_exact,
                "limb_exact": limb_exact,
                "flower_exact": flower_exact,
                "generated_flower_exact": generated_flower_exact,
                "strict_flower_recall": strict_recall,
                "human_skin_retained": human_retained,
                "halo_background_mae": halo_mae,
                "robot_mask_iou_previous": robot_iou,
            }
        )
        if index in review_indices or index in risk_indices:
            overlay = result.copy()
            overlay[robot] = np.rint(
                0.55 * overlay[robot] + 0.45 * np.asarray([255, 80, 40])
            ).astype(np.uint8)
            overlay[flowers] = np.rint(
                0.35 * overlay[flowers] + 0.65 * np.asarray([40, 40, 255])
            ).astype(np.uint8)
            cells = [
                _annotate(cv2, source, f"source {index}"),
                _annotate(cv2, generated, "H3 generated"),
            ]
            if previous is not None:
                cells.append(_annotate(cv2, previous, "v25 rejected"))
            cells.extend(
                (
                    _annotate(cv2, result, Path(args.output_name).stem),
                    _annotate(cv2, overlay, "robot blue / flowers red"),
                )
            )
            if index in review_indices:
                review_frames[index] = cells
            if index in risk_indices:
                risk_frames[index] = [cell[110:470, 210:760] for cell in cells]
    generated_capture.release()
    source_capture.release()
    if previous_capture is not None:
        previous_capture.release()
    assert writer.stdin is not None
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding layered output")

    dense_path = output_dir / "review/dense-layer-review.jpg"
    risk_path = output_dir / "review/interaction-crop-review.jpg"
    cv2.imwrite(
        str(dense_path), _sheet(cv2, [review_frames[index] for index in sorted(review_frames)]),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    cv2.imwrite(
        str(risk_path), _sheet(cv2, [risk_frames[index] for index in sorted(risk_frames)]),
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )
    metrics_path = output_dir / "logs/frame-metrics.jsonl"
    metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in metric_rows))
    aggregate = {
        "frames": len(metric_rows),
        "body_exact_min": min(row["body_exact"] for row in metric_rows),
        "limb_exact_min": min(row["limb_exact"] for row in metric_rows),
        "flower_exact_min": min(row["flower_exact"] for row in metric_rows),
        "generated_flower_exact_min": min(
            row["generated_flower_exact"] for row in metric_rows
        ),
        "strict_flower_recall_min": min(row["strict_flower_recall"] for row in metric_rows),
        "human_skin_retained_max": max(row["human_skin_retained"] for row in metric_rows),
        "halo_background_mae_max": max(row["halo_background_mae"] for row in metric_rows),
        "robot_mask_iou_previous_min": min(row["robot_mask_iou_previous"] for row in metric_rows[1:]),
        "robot_fraction_mean": float(np.mean([row["robot_fraction"] for row in metric_rows])),
        "limb_extra_fraction_mean": float(np.mean([row["limb_extra_fraction"] for row in metric_rows])),
        "raw_limb_fraction_mean": float(np.mean([row["raw_limb_fraction"] for row in metric_rows])),
        "flower_fraction_mean": float(np.mean([row["flower_fraction"] for row in metric_rows])),
        "generated_flower_fraction_mean": float(
            np.mean([row["generated_flower_fraction"] for row in metric_rows])
        ),
    }
    gates = {
        "full_decode": aggregate["frames"] == 660,
        "body_preserved": aggregate["body_exact_min"] >= 0.999,
        "limbs_preserved": aggregate["limb_exact_min"] >= 0.999,
        "flowers_preserved": aggregate["flower_exact_min"] >= 0.999,
        "generated_flowers_preserved": (
            aggregate["generated_flower_exact_min"] >= 0.999
        ),
        "strict_flowers_preserved": aggregate["strict_flower_recall_min"] >= 0.999,
        "human_skin_removed": aggregate["human_skin_retained_max"] <= 0.005,
        "halo_removed": aggregate["halo_background_mae_max"] <= 0.05,
        "human_review": args.human_review == "passed",
    }
    encoded_info = _video_info(cv2, output_video, decode=True)
    gates["encoded_decode"] = int(encoded_info.get("decoded_frames", 0)) == 660
    automatic_passed = all(value for key, value in gates.items() if key != "human_review")
    final_status = "WORKING" if automatic_passed and gates["human_review"] else "PARTIAL"
    record.update(
        {
            "status": final_status,
            "honest_status": (
                f"{final_status}: automatic semantic preservation/removal gates "
                + ("and human review passed." if final_status == "WORKING" else "passed where shown; human review remains pending or failed.")
            ),
            "video": generated_info,
            "encoded_video": encoded_info,
            "method": {
                "layer_order": [
                    "clean background",
                    "H3 body plus constrained wrist track and generated flowers",
                    "non-overlapping source flower instances",
                ],
                "human_negative": "per-frame pose-arm skin pixels, never a direct whole-arm erasure",
                "limb_corridor_dilation": args.limb_corridor_dilation,
                "body_neighborhood_dilation": args.body_neighborhood_dilation,
                "limb_consensus_dilation": args.limb_consensus_dilation,
                "generated_flower_support_radius": args.generated_flower_support_radius,
                "generated_flower_limb_radius": args.generated_flower_limb_radius,
                "generated_flower_min_component": args.generated_flower_min_component,
                "generated_flower_outlier_ratio": args.generated_flower_outlier_ratio,
                "generated_flower_outlier_frames": generated_flower_outlier_frames,
                "robot_limb_source": str(paths["robot_limb_masks"]),
                "generated_flower_source": str(
                    paths["generated_flower_instance_masks"]
                ),
                "robot_dilation": args.robot_dilation,
                "person_feather_sigma": args.person_feather_sigma,
                "background_method": args.background_method,
                "person_dilation": args.person_dilation,
            },
            "metrics": aggregate,
            "acceptance_gates": gates,
            "outputs": {
                "video": {"path": str(output_video), "sha256": file_sha256(output_video)},
                "dense_review": {"path": str(dense_path), "sha256": file_sha256(dense_path)},
                "interaction_review": {"path": str(risk_path), "sha256": file_sha256(risk_path)},
                "frame_metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
            },
        }
    )
    _write_json(output_dir / "manifest.json", record)
    (output_dir / "logs/run.log").write_text(
        json.dumps({"status": final_status, "metrics": aggregate, "gates": gates}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output_dir": str(output_dir), "status": final_status, "metrics": aggregate, "gates": gates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
