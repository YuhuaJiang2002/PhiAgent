#!/usr/bin/env python3
"""Render a background-locked robot replacement from multiple pose anchors.

Each anchor pairs a source-video frame with an image-generated robot frame in
the same camera coordinate system.  Dense optical flow maps the two anchors
nearest each source frame into that frame before the robot appearances are
blended.  The current source frame is always the compositing base, so pixels
outside the tracked replacement support remain unchanged before encoding.
"""

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnchorSpec:
    frame: int
    source: Path
    robot: Path


@dataclass
class Anchor:
    frame: int
    source: Any
    robot: Any
    mask: Any
    source_gray_small: Any
    # Source-person support in the anchor camera frame.  Keep it separate from
    # ``mask``: the latter also contains robot silhouette extensions and image
    # generation differences, which must not be mistaken for the source person
    # when deciding which foreground objects should occlude the robot.
    person_mask: Any | None = None


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


def _git_state(project_root: Path) -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=project_root,
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


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in (
        "mediapipe",
        "numpy",
        "opencv-python",
        "opencv-python-headless",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _source_info(cv2: Any, source: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {source}")
    result: dict[str, int | float] = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    if (
        int(result["width"]) <= 0
        or int(result["height"]) <= 0
        or float(result["fps"]) <= 0
        or int(result["frames"]) <= 1
    ):
        raise RuntimeError("source video metadata is invalid")
    return result


def _load_specs(config: Path) -> tuple[Path, Path, Path, tuple[AnchorSpec, ...]]:
    payload = json.loads(config.read_text())
    source = Path(payload["source"]).expanduser().resolve()
    person_union_mask = Path(payload["person_union_mask"]).expanduser().resolve()
    semantic_person_mask = Path(payload["semantic_person_mask"]).expanduser().resolve()
    specs = tuple(
        AnchorSpec(
            frame=int(item["frame"]),
            source=Path(item["source"]).expanduser().resolve(),
            robot=Path(item["robot"]).expanduser().resolve(),
        )
        for item in payload["anchors"]
    )
    if len(specs) < 3:
        raise ValueError("at least three motion anchors are required")
    if tuple(sorted(item.frame for item in specs)) != tuple(item.frame for item in specs):
        raise ValueError("anchor frames must be strictly increasing")
    if len({item.frame for item in specs}) != len(specs):
        raise ValueError("anchor frames must be unique")
    for path in (source, person_union_mask, semantic_person_mask):
        if not path.is_file():
            raise ValueError(f"input does not exist: {path}")
    for item in specs:
        for path in (item.source, item.robot):
            if not path.is_file():
                raise ValueError(f"anchor input does not exist: {path}")
    return source, person_union_mask, semantic_person_mask, specs


def _odd(value: int) -> int:
    return max(1, value if value % 2 else value + 1)


def _largest_components(cv2: Any, np: Any, mask: Any, minimum_area: int) -> Any:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    result = np.zeros(mask.shape, dtype=np.uint8)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_area:
            result[labels == component] = 255
    return result


def _flow_map(
    cv2: Any,
    np: Any,
    current_gray_small: Any,
    anchor_gray_small: Any,
    width: int,
    height: int,
    flow_clip_pixels: float,
    flow_strength: float = 1.0,
) -> tuple[Any, Any, float]:
    flow_width = int(current_gray_small.shape[1])
    flow = cv2.calcOpticalFlowFarneback(
        current_gray_small,
        anchor_gray_small,
        None,
        0.5,
        5,
        31,
        5,
        7,
        1.5,
        cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    flow = cv2.GaussianBlur(flow, (5, 5), 0)
    magnitude = np.linalg.norm(flow, axis=2)
    maximum = float(magnitude.max() * width / flow_width)
    clip_at_scale = flow_clip_pixels * flow_width / width
    scale = np.minimum(1.0, clip_at_scale / np.maximum(magnitude, 1e-6))
    flow *= scale[..., None]
    full_flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    full_flow[..., 0] *= width / flow_width
    full_flow[..., 1] *= height / int(current_gray_small.shape[0])
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    return (
        grid_x + full_flow[..., 0] * flow_strength,
        grid_y + full_flow[..., 1] * flow_strength,
        maximum,
    )


def _warp_anchor(
    cv2: Any,
    np: Any,
    anchor: Anchor,
    current_gray_small: Any,
    width: int,
    height: int,
    flow_clip_pixels: float,
    flow_strength: float,
) -> tuple[Any, Any, float]:
    robot, mask, _, maximum = _warp_anchor_layers(
        cv2,
        np,
        anchor,
        current_gray_small,
        width,
        height,
        flow_clip_pixels,
        flow_strength,
    )
    return robot, mask, maximum


def _warp_anchor_layers(
    cv2: Any,
    np: Any,
    anchor: Anchor,
    current_gray_small: Any,
    width: int,
    height: int,
    flow_clip_pixels: float,
    flow_strength: float,
) -> tuple[Any, Any, Any, float]:
    """Warp robot, robot support, and source-person support consistently.

    Dense flow maps pixels in the current source frame to the anchor source
    frame.  Robot pixels and their silhouette use the same attenuated map;
    applying full flow to the mask but partial flow to the robot creates the
    bright tearing visible around hands and flowers in the old compositor.
    The source-person layer uses full flow because it is a source-to-source
    correspondence used only for occlusion and human-removal decisions.
    """

    full_map_x, full_map_y, maximum = _flow_map(
        cv2,
        np,
        current_gray_small,
        anchor.source_gray_small,
        width,
        height,
        flow_clip_pixels,
        1.0,
    )
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    robot_map_x = grid_x + (full_map_x - grid_x) * flow_strength
    robot_map_y = grid_y + (full_map_y - grid_y) * flow_strength
    robot = cv2.remap(
        anchor.robot,
        robot_map_x,
        robot_map_y,
        cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT101,
    )
    mask = cv2.remap(
        anchor.mask,
        robot_map_x,
        robot_map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    person_source = anchor.person_mask if anchor.person_mask is not None else anchor.mask
    person_mask = cv2.remap(
        person_source,
        full_map_x,
        full_map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return robot, mask, person_mask, maximum


def _build_anchor_mask(
    cv2: Any,
    np: Any,
    source: Any,
    robot: Any,
    person_union: Any,
    semantic_proxy: Any,
) -> tuple[Any, dict[str, float]]:
    delta = cv2.absdiff(source, robot)
    delta_gray = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
    delta_gray = cv2.GaussianBlur(delta_gray, (5, 5), 0)
    changed = (delta_gray >= 13).astype(np.uint8) * 255
    height, width = changed.shape
    subject_roi = np.zeros(changed.shape, dtype=np.uint8)
    subject_roi[
        round(height * 0.02) : round(height * 0.96),
        round(width * 0.45) : round(width * 0.90),
    ] = 255
    changed = cv2.bitwise_and(changed, subject_roi)
    semantic_proxy = cv2.bitwise_and(semantic_proxy, subject_roi)
    semantic_proxy = (semantic_proxy >= 96).astype(np.uint8) * 255
    semantic_proxy = cv2.morphologyEx(
        semantic_proxy,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    semantic_proxy = _largest_components(
        cv2,
        np,
        semantic_proxy,
        minimum_area=max(64, round(semantic_proxy.size * 0.008)),
    )
    semantic_neighborhood = cv2.dilate(
        semantic_proxy,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
    )
    # Generated pixels supply robot-silhouette evidence, while the transferred
    # semantic mask guarantees coverage of the original person.  Restrict the
    # generated difference to the semantic neighborhood so global ImageGen
    # color drift cannot pull shelves and the tabletop into the replacement.
    changed = cv2.bitwise_and(changed, semantic_neighborhood)
    mask = cv2.bitwise_or(changed, semantic_proxy)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    mask = _largest_components(
        cv2,
        np,
        mask,
        minimum_area=max(48, round(mask.size * 0.00008)),
    )
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    mask = cv2.bitwise_and(mask, subject_roi)
    coverage = float(np.count_nonzero(mask) / mask.size)
    changed_coverage = float(np.count_nonzero(changed) / changed.size)
    if not 0.06 <= coverage <= 0.30:
        raise RuntimeError(f"anchor mask coverage is implausible: {coverage:.4f}")
    return mask, {
        "coverage": coverage,
        "changed_coverage": changed_coverage,
        "person_union_coverage": float(np.count_nonzero(person_union) / person_union.size),
    }


def _object_mask(cv2: Any, np: Any, frame: Any, dilation: int = 2) -> Any:
    """Conservative source-pixel restoration for flowers, leaves, and stems."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue >= 26) & (hue <= 96) & (saturation >= 44) & (value >= 25)
    pink = (hue >= 145) & (hue <= 179) & (saturation >= 85) & (value >= 45)
    yellow = (hue >= 12) & (hue <= 35) & (saturation >= 80) & (value >= 65)
    height, width = frame.shape[:2]
    scene_region = np.zeros((height, width), dtype=bool)
    scene_region[round(height * 0.26) :, round(width * 0.30) :] = True
    scene_region[round(height * 0.60) :, :] = True
    mask = ((green | pink | yellow) & scene_region).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if dilation:
        size = _odd(dilation * 2 + 1)
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
    return mask


def _green_object_mask(cv2: Any, np: Any, frame: Any, dilation: int = 2) -> Any:
    """Green-only protection safe to apply inside the person boundary."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue >= 28) & (hue <= 94) & (saturation >= 58) & (value >= 28)
    height, width = frame.shape[:2]
    scene_region = np.zeros((height, width), dtype=bool)
    scene_region[round(height * 0.26) :, round(width * 0.30) :] = True
    scene_region[round(height * 0.60) :, :] = True
    mask = (green & scene_region).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if dilation:
        size = _odd(dilation * 2 + 1)
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
    return mask


def _source_person_mask(
    cv2: Any,
    np: Any,
    segmenter: Any,
    frame: Any,
    threshold: float = 0.22,
) -> Any:
    """Segment the current source person instead of warping a stale mask."""

    score = segmenter.process(frame[:, :, ::-1]).segmentation_mask
    mask = (score >= threshold).astype(np.uint8) * 255
    height, width = mask.shape
    subject_roi = np.zeros(mask.shape, dtype=np.uint8)
    subject_roi[
        round(height * 0.02) : round(height * 0.97),
        round(width * 0.44) : round(width * 0.91),
    ] = 255
    mask = cv2.bitwise_and(mask, subject_roi)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    return _largest_components(
        cv2,
        np,
        mask,
        minimum_area=max(128, round(mask.size * 0.008)),
    )


def _writer(ffmpeg: Path, output: Path, width: int, height: int, fps: float) -> Any:
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            str(ffmpeg),
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
            "medium",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _cosine(np: Any, first: Any, second: Any) -> float:
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-9:
        return 1.0 if float(np.linalg.norm(a - b)) < 1e-9 else 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / denominator)))


def _bracket(anchors: tuple[Anchor, ...], frame_index: int) -> tuple[Anchor, Anchor, float]:
    if frame_index <= anchors[0].frame:
        return anchors[0], anchors[0], 0.0
    if frame_index >= anchors[-1].frame:
        return anchors[-1], anchors[-1], 0.0
    for left, right in zip(anchors, anchors[1:]):
        if left.frame <= frame_index <= right.frame:
            weight = (frame_index - left.frame) / (right.frame - left.frame)
            weight = weight * weight * (3.0 - 2.0 * weight)
            return left, right, weight
    raise AssertionError("frame was not bracketed")


def _render(
    *,
    cv2: Any,
    np: Any,
    source: Path,
    output: Path,
    ffmpeg: Path,
    source_info: dict[str, int | float],
    anchors: tuple[Anchor, ...],
    person_union: Any,
    flow_width: int,
    flow_clip_pixels: float,
    flow_strength: float,
    robot_expansion_pixels: int,
    person_segmenter: Any | None,
) -> dict[str, float | int]:
    width = int(source_info["width"])
    height = int(source_info["height"])
    fps = float(source_info["fps"])
    frame_count = int(source_info["frames"])
    flow_height = max(2, round(height * flow_width / width))
    # Retained for provenance and a sanity check only.  Applying the temporal
    # union to every frame freezes all poses into one large patch and lowers
    # motion fidelity.  Per-frame person support is flow-propagated below.
    person_union_coverage = float(np.count_nonzero(person_union) / person_union.size)
    capture = cv2.VideoCapture(str(source))
    writer = _writer(ffmpeg, output, width, height, fps)
    decoded = 0
    maximum_flow = 0.0
    background_scores: list[float] = []
    replacement_scores: list[float] = []
    replacement_mae_scores: list[float] = []
    motion_scores: list[float] = []
    temporal_scores: list[float] = []
    object_lock_scores: list[float] = []
    support_coverages: list[float] = []
    protected_coverages: list[float] = []
    human_leakage_risks: list[float] = []
    previous_source_gray = None
    previous_candidate_gray = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_small = cv2.resize(
                frame, (flow_width, flow_height), interpolation=cv2.INTER_AREA
            )
            current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
            left, right, weight = _bracket(anchors, decoded)
            left_robot, left_mask, left_person, left_maximum = _warp_anchor_layers(
                cv2,
                np,
                left,
                current_gray,
                width,
                height,
                flow_clip_pixels,
                flow_strength,
            )
            maximum_flow = max(maximum_flow, left_maximum)
            if right.frame == left.frame:
                right_robot, right_mask, right_person = (
                    left_robot,
                    left_mask,
                    left_person,
                )
            else:
                right_robot, right_mask, right_person, right_maximum = _warp_anchor_layers(
                    cv2,
                    np,
                    right,
                    current_gray,
                    width,
                    height,
                    flow_clip_pixels,
                    flow_strength,
                )
                maximum_flow = max(maximum_flow, right_maximum)

            left_alpha = left_mask.astype(np.float32) / 255.0
            right_alpha = right_mask.astype(np.float32) / 255.0
            left_weight = (1.0 - weight) * left_alpha
            right_weight = weight * right_alpha
            denominator = left_weight + right_weight
            # Use a current-frame-aligned anchor as the fallback.  The previous
            # implementation used an unwarped image for the upper body, which
            # froze the head and snapped it at every anchor midpoint.
            nearest_scene = left_robot if weight < 0.5 else right_robot
            warped_blend = nearest_scene.astype(np.float32)
            active = denominator > 1e-4
            warped_blend[active] = (
                left_robot[active].astype(np.float32) * left_weight[active, None]
                + right_robot[active].astype(np.float32) * right_weight[active, None]
            ) / denominator[active, None]
            blended_robot = warped_blend

            support = (denominator >= 0.035).astype(np.uint8) * 255
            support = cv2.morphologyEx(
                support,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            )
            support = cv2.dilate(
                support,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            if person_segmenter is None:
                person_confidence = (
                    (1.0 - weight) * left_person.astype(np.float32)
                    + weight * right_person.astype(np.float32)
                ) / 255.0
                person_support = (person_confidence >= 0.32).astype(np.uint8) * 255
            else:
                person_support = _source_person_mask(
                    cv2,
                    np,
                    person_segmenter,
                    frame,
                )
            person_support = cv2.morphologyEx(
                person_support,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            )
            # The source person must always be replaced, but unlike the old
            # temporal union this support follows the current pose.
            support = cv2.bitwise_or(support, person_support)
            alpha = cv2.GaussianBlur(support, (0, 0), 0.8).astype(np.float32) / 255.0
            candidate = np.rint(
                frame.astype(np.float32) * (1.0 - alpha[..., None])
                + blended_robot * alpha[..., None]
            ).astype(np.uint8)
            # Preserve every source pixel outside a small expansion of the
            # tracked person.  This protects pale flowers, thin stems, scissors,
            # and table details that the old HSV-only mask missed, while still
            # allowing the robot silhouette to be modestly wider than the human.
            expansion_size = _odd(robot_expansion_pixels * 2 + 1)
            robot_allowance = cv2.dilate(
                person_support,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (expansion_size, expansion_size)
                ),
            )
            structural_protection = cv2.bitwise_and(
                support,
                cv2.bitwise_not(robot_allowance),
            )
            protected = cv2.bitwise_or(
                structural_protection,
                _object_mask(cv2, np, frame),
            )
            # Color-based protection must not resurrect blond hair, pink
            # clothing, or warm skin inside the tracked source person.
            protected[person_support > 0] = 0
            green_inside = _green_object_mask(cv2, np, frame)
            protected[(person_support > 0) & (green_inside > 0)] = 255
            candidate[protected > 0] = frame[protected > 0]

            protected_pixels = protected > 0
            if np.count_nonzero(protected_pixels):
                object_lock_scores.append(
                    float(
                        np.count_nonzero(
                            np.all(
                                candidate[protected_pixels] == frame[protected_pixels],
                                axis=1,
                            )
                        )
                    )
                    / int(np.count_nonzero(protected_pixels))
                )
            support_coverages.append(float(np.count_nonzero(support) / support.size))
            protected_coverages.append(
                float(np.count_nonzero(protected) / protected.size)
            )
            allowed = cv2.dilate(
                support,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            )
            outside = allowed == 0
            background_scores.append(
                float(np.count_nonzero(np.all(candidate[outside] == frame[outside], axis=1)))
                / max(1, int(np.count_nonzero(outside)))
            )
            subject = (support > 0) & ~(protected > 0)
            subject_delta = np.abs(
                candidate[subject].astype(np.int16) - frame[subject].astype(np.int16)
            )
            if subject_delta.size:
                visibly_changed = np.max(subject_delta, axis=1) >= 12
                replacement_scores.append(float(np.mean(visibly_changed)))
                replacement_mae_scores.append(float(np.mean(subject_delta)))
            person_pixels = (person_support > 0) & ~(green_inside > 0)
            person_delta = np.abs(
                candidate[person_pixels].astype(np.int16)
                - frame[person_pixels].astype(np.int16)
            )
            if person_delta.size:
                # Source-equal pixels inside the segmented human are direct
                # evidence that the compositor resurrected the performer.  A
                # binary support-coverage metric was tautological because the
                # person support is explicitly OR'ed into ``support``.
                source_like = np.max(person_delta, axis=1) < 12
                human_leakage_risks.append(float(np.mean(source_like)))

            candidate_small = cv2.resize(
                candidate, (flow_width, flow_height), interpolation=cv2.INTER_AREA
            )
            candidate_gray = cv2.cvtColor(candidate_small, cv2.COLOR_BGR2GRAY)
            if previous_source_gray is not None and previous_candidate_gray is not None:
                small_support = cv2.resize(
                    support, (flow_width, flow_height), interpolation=cv2.INTER_NEAREST
                ) > 0
                source_motion = cv2.absdiff(current_gray, previous_source_gray)
                candidate_motion = cv2.absdiff(candidate_gray, previous_candidate_gray)
                if np.count_nonzero(small_support):
                    source_values = source_motion[small_support]
                    candidate_values = candidate_motion[small_support]
                    cosine = _cosine(np, source_values, candidate_values)
                    source_energy = float(np.mean(source_values))
                    candidate_energy = float(np.mean(candidate_values))
                    energy_ratio = min(
                        (source_energy + 1e-3) / (candidate_energy + 1e-3),
                        (candidate_energy + 1e-3) / (source_energy + 1e-3),
                    )
                    motion_scores.append(math.sqrt(max(0.0, cosine * energy_ratio)))
                    residual = float(
                        np.mean(
                            np.abs(
                                source_values.astype(np.float32)
                                - candidate_values.astype(np.float32)
                            )
                        )
                    )
                    temporal_scores.append(math.exp(-residual / 32.0))
            previous_source_gray = current_gray
            previous_candidate_gray = candidate_gray

            assert writer.stdin is not None
            writer.stdin.write(candidate.tobytes())
            decoded += 1
    finally:
        capture.release()
        if writer.stdin is not None:
            writer.stdin.close()
        return_code = writer.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg writer failed with code {return_code}")

    if decoded != frame_count:
        raise RuntimeError(f"decoded {decoded}/{frame_count} source frames")
    return {
        "decoded_frames": decoded,
        "background_lock": sum(background_scores) / len(background_scores),
        "subject_change": sum(replacement_scores) / len(replacement_scores),
        "subject_mean_absolute_change": sum(replacement_mae_scores)
        / len(replacement_mae_scores),
        "motion_preservation": sum(motion_scores) / len(motion_scores),
        "temporal_consistency": sum(temporal_scores) / len(temporal_scores),
        "object_lock": sum(object_lock_scores) / len(object_lock_scores),
        "mean_dynamic_support_coverage": sum(support_coverages) / len(support_coverages),
        "mean_protected_coverage": sum(protected_coverages) / len(protected_coverages),
        "human_leakage_risk": sum(human_leakage_risks) / len(human_leakage_risks),
        "person_union_evidence_coverage": person_union_coverage,
        "maximum_unclipped_flow_pixels": maximum_flow,
    }


def _review_assets(ffmpeg: Path, video: Path, output_dir: Path) -> None:
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-ss",
            "13.5",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_dir / "poster.jpg"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            "fps=1/3.4,scale=480:-2,tile=4x2:padding=4:margin=4:color=black",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_dir / "storyboard.jpg"),
        ],
        check=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--flow-width", type=int, default=384)
    parser.add_argument("--flow-clip-pixels", type=float, default=84.0)
    parser.add_argument("--flow-strength", type=float, default=0.30)
    parser.add_argument("--robot-expansion-pixels", type=int, default=18)
    parser.add_argument(
        "--person-segmentation",
        choices=("mediapipe", "flow"),
        default="mediapipe",
    )
    parser.add_argument(
        "--human-review",
        choices=("pending", "passed", "failed"),
        default="pending",
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.flow_width < 128:
        raise ValueError("flow-width must be at least 128")
    if not math.isfinite(args.flow_clip_pixels) or args.flow_clip_pixels <= 0:
        raise ValueError("flow-clip-pixels must be positive and finite")
    if not math.isfinite(args.flow_strength) or not 0.0 <= args.flow_strength <= 1.0:
        raise ValueError("flow-strength must be finite and in [0, 1]")
    if not 0 <= args.robot_expansion_pixels <= 64:
        raise ValueError("robot-expansion-pixels must be in [0, 64]")
    config = args.config.expanduser().resolve()
    experiment = args.experiment_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if not config.is_file() or not ffmpeg.is_file():
        raise ValueError("config and ffmpeg must exist")
    experiment.mkdir(parents=True, exist_ok=True)
    assets = experiment / "assets"
    final_dir = experiment / "final"
    assets.mkdir(exist_ok=True)
    final_dir.mkdir(exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    trace_path = experiment / "trace.json"

    try:
        import cv2
        import numpy as np

        np.random.seed(args.seed)
        source, person_union_path, semantic_path, specs = _load_specs(config)
        source_info = _source_info(cv2, source)
        width = int(source_info["width"])
        height = int(source_info["height"])
        frame_count = int(source_info["frames"])
        if specs[0].frame != 0 or specs[-1].frame != frame_count - 1:
            raise ValueError("anchors must cover the first and last source frames")

        person_union = cv2.imread(str(person_union_path), cv2.IMREAD_GRAYSCALE)
        semantic = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE)
        if person_union is None or semantic is None:
            raise RuntimeError("cannot decode person masks")
        person_union = cv2.resize(
            person_union, (width, height), interpolation=cv2.INTER_NEAREST
        )
        person_union = (person_union >= 127).astype(np.uint8) * 255
        semantic = cv2.resize(semantic, (width, height), interpolation=cv2.INTER_LANCZOS4)
        semantic = (semantic >= 127).astype(np.uint8) * 255
        semantic_anchor_spec = min(specs, key=lambda item: abs(item.frame - 276))
        semantic_anchor_source = cv2.imread(
            str(semantic_anchor_spec.source), cv2.IMREAD_COLOR
        )
        if semantic_anchor_source is None:
            raise RuntimeError("cannot decode semantic anchor source")
        semantic_anchor_source = cv2.resize(
            semantic_anchor_source, (width, height), interpolation=cv2.INTER_LANCZOS4
        )
        flow_height = max(2, round(height * args.flow_width / width))
        semantic_anchor_gray = cv2.cvtColor(
            cv2.resize(
                semantic_anchor_source,
                (args.flow_width, flow_height),
                interpolation=cv2.INTER_AREA,
            ),
            cv2.COLOR_BGR2GRAY,
        )

        anchors: list[Anchor] = []
        mask_evidence: list[dict[str, float | int]] = []
        anchor_union = np.zeros((height, width), dtype=np.uint8)
        for item in specs:
            source_anchor = cv2.imread(str(item.source), cv2.IMREAD_COLOR)
            robot_anchor = cv2.imread(str(item.robot), cv2.IMREAD_COLOR)
            if source_anchor is None or robot_anchor is None:
                raise RuntimeError(f"cannot decode anchor {item.frame}")
            source_anchor = cv2.resize(
                source_anchor, (width, height), interpolation=cv2.INTER_LANCZOS4
            )
            robot_anchor = cv2.resize(
                robot_anchor, (width, height), interpolation=cv2.INTER_LANCZOS4
            )
            source_small = cv2.resize(
                source_anchor,
                (args.flow_width, flow_height),
                interpolation=cv2.INTER_AREA,
            )
            source_gray_small = cv2.cvtColor(source_small, cv2.COLOR_BGR2GRAY)
            map_x, map_y, _ = _flow_map(
                cv2,
                np,
                source_gray_small,
                semantic_anchor_gray,
                width,
                height,
                args.flow_clip_pixels,
                1.0,
            )
            semantic_proxy = cv2.remap(
                semantic,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            cv2.imwrite(
                str(assets / f"semantic-proxy-{item.frame:04d}.png"),
                semantic_proxy,
            )
            anchor_mask, evidence = _build_anchor_mask(
                cv2,
                np,
                source_anchor,
                robot_anchor,
                person_union,
                semantic_proxy,
            )
            anchors.append(
                Anchor(
                    item.frame,
                    source_anchor,
                    robot_anchor,
                    anchor_mask,
                    source_gray_small,
                    semantic_proxy,
                )
            )
            anchor_union = cv2.bitwise_or(anchor_union, anchor_mask)
            mask_evidence.append({"frame": item.frame, **evidence})
            cv2.imwrite(str(assets / f"mask-{item.frame:04d}.png"), anchor_mask)
            overlay = source_anchor.copy()
            overlay[anchor_mask > 0] = np.rint(
                0.42 * source_anchor[anchor_mask > 0]
                + 0.58 * np.asarray((40, 70, 245), dtype=np.float32)
            ).astype(np.uint8)
            cv2.imwrite(str(assets / f"mask-overlay-{item.frame:04d}.jpg"), overlay)

        # The historical union mask is retained as evidence only.  It is not a
        # hard compositing boundary because it is horizontally misregistered
        # for this source.  The union of all eight camera-aligned anchor masks
        # is the conservative safety boundary used by the renderer.
        person_safety_union = anchor_union
        cv2.imwrite(str(assets / "anchor-mask-union.png"), anchor_union)
        cv2.imwrite(str(assets / "person-safety-union.png"), person_safety_union)

        trace: dict[str, object] = {
            "schema_version": "1.0.0",
            "status": "preflight" if args.preflight_only else "running",
            "honest_status": "PARTIAL",
            "method": "occlusion_aware_multi_anchor_flow_v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": _package_versions(),
            "seed": args.seed,
            "gpu": {
                "used": False,
                "cuda_visible_devices": None,
                "reason": "deterministic OpenCV compositor runs on CPU; image anchors were generated before this entry point",
            },
            "git": _git_state(project_root),
            "config": str(config),
            "config_sha256": _sha256(config),
            "source": str(source),
            "source_sha256": _sha256(source),
            "source_video": source_info,
            "coordinate_frames": {
                "source": "camera:source_pixels",
                "anchors": "camera:source_pixels after explicit resize",
                "optical_flow": "current-frame camera pixels -> anchor-frame camera pixels",
            },
            "parameters": {
                "flow_width": args.flow_width,
                "flow_clip_pixels": args.flow_clip_pixels,
                "flow_strength": args.flow_strength,
                "robot_expansion_pixels": args.robot_expansion_pixels,
                "person_segmentation": args.person_segmentation,
                "anchor_count": len(anchors),
                "person_safety_union_coverage": float(
                    np.count_nonzero(person_safety_union) / person_safety_union.size
                ),
            },
            "anchors": [
                {
                    "frame": item.frame,
                    "source": str(item.source),
                    "source_sha256": _sha256(item.source),
                    "robot": str(item.robot),
                    "robot_sha256": _sha256(item.robot),
                }
                for item in specs
            ],
            "mask_evidence": mask_evidence,
        }
        _write_json(trace_path, trace)
        if args.preflight_only:
            trace.update(
                {
                    "status": "preflight_complete",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "limitations": ["No video was rendered in preflight-only mode."],
                }
            )
            _write_json(trace_path, trace)
            print(json.dumps({"experiment": str(experiment), "mask_evidence": mask_evidence}, indent=2))
            return 0

        final_video = final_dir / "robot-motion-replacement.mp4"
        person_segmenter = None
        if args.person_segmentation == "mediapipe":
            try:
                import mediapipe as mp
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "--person-segmentation mediapipe requires the optional "
                    "MediaPipe experiment environment; install it separately or "
                    "use --person-segmentation flow"
                ) from error

            person_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1
            )
        try:
            metrics = _render(
                cv2=cv2,
                np=np,
                source=source,
                output=final_video,
                ffmpeg=ffmpeg,
                source_info=source_info,
                anchors=tuple(anchors),
                person_union=person_safety_union,
                flow_width=args.flow_width,
                flow_clip_pixels=args.flow_clip_pixels,
                flow_strength=args.flow_strength,
                robot_expansion_pixels=args.robot_expansion_pixels,
                person_segmenter=person_segmenter,
            )
        finally:
            if person_segmenter is not None:
                person_segmenter.close()
        subprocess.run(
            [str(ffmpeg), "-v", "error", "-i", str(final_video), "-f", "null", "-"],
            check=True,
        )
        _review_assets(ffmpeg, final_video, final_dir)
        automated_passed = (
            int(metrics["decoded_frames"]) == frame_count
            and float(metrics["background_lock"]) >= 0.99999
            and float(metrics["subject_change"]) >= 0.985
            and float(metrics["subject_mean_absolute_change"]) >= 12.0
            and float(metrics["motion_preservation"]) >= 0.55
            and float(metrics["object_lock"]) >= 0.99999
            and float(metrics["human_leakage_risk"]) <= 1e-9
        )
        accepted = automated_passed and args.human_review == "passed"
        status = (
            "accepted"
            if accepted
            else "review_required"
            if automated_passed and args.human_review == "pending"
            else "rejected"
        )
        trace.update(
            {
                "status": status,
                "honest_status": "WORKING" if accepted else "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "acceptance": {
                    "automated_gates_passed": automated_passed,
                    "human_review": args.human_review,
                    "full_clip_decoded": int(metrics["decoded_frames"]) == frame_count,
                    "background_lock_passed": float(metrics["background_lock"]) >= 0.99999,
                    "subject_change_passed": float(metrics["subject_change"]) >= 0.985,
                    "subject_mean_absolute_change_passed": float(
                        metrics["subject_mean_absolute_change"]
                    )
                    >= 12.0,
                    "motion_preservation_passed": float(metrics["motion_preservation"]) >= 0.55,
                    "object_lock_passed": float(metrics["object_lock"]) >= 0.99999,
                    "human_leakage_risk_passed": float(metrics["human_leakage_risk"]) <= 1e-9,
                },
                "outputs": {
                    "video": str(final_video),
                    "video_sha256": _sha256(final_video),
                    "poster": str(final_dir / "poster.jpg"),
                    "storyboard": str(final_dir / "storyboard.jpg"),
                },
                "limitations": [
                    "This is an image-anchor/optical-flow proxy, not official PhiZero inference or real-robot execution.",
                    "Robot articulation is anchored at eight real source poses and flow-interpolated between them; it is not frame-wise video diffusion.",
                    (
                        "Person support is estimated per frame with MediaPipe Selfie Segmentation; "
                        "SAM 3.1 multi-object tracks have not yet replaced this proxy."
                        if args.person_segmentation == "mediapipe"
                        else "Person support is propagated from one semantic mask by dense flow; "
                        "SAM 3.1 multi-object tracks have not yet replaced this proxy."
                    ),
                    "Source pixels outside the expanded person support are restored, but fully occluded flower geometry cannot be recovered from RGB alone.",
                    "Pixel-lock metrics are computed before lossy H.264 encoding.",
                ],
            }
        )
        _write_json(trace_path, trace)
        _write_json(final_dir / "manifest.json", trace)
        print(
            json.dumps(
                {
                    "experiment": str(experiment),
                    "status": trace["status"],
                    "honest_status": trace["honest_status"],
                    "video": str(final_video),
                    "metrics": metrics,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if accepted else 2
    except Exception as exc:
        payload = json.loads(trace_path.read_text()) if trace_path.exists() else {}
        payload.update(
            {
                "status": "failed",
                "honest_status": "BLOCKED",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(trace_path, payload)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
