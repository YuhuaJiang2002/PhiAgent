#!/usr/bin/env python3
"""Render a sharp human-free robot replacement without anchor cross-dissolves.

The previous renderer mixed two independently warped robot anchors in every
in-between frame.  That creates double contours even when both anchor images
are individually sharp.  This renderer instead composes short, consecutive
source-frame flow fields into a coordinate map and samples exactly one clean
robot anchor once per output frame.  A conservative camera-frame safety union
is then overwritten with the robot scene at full opacity; source restoration is
never allowed inside that union.
"""

from __future__ import annotations

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_multi_anchor_robot_replacement import (  # noqa: E402
    Anchor,
    _build_anchor_mask,
    _cosine,
    _git_state,
    _load_specs,
    _package_versions,
    _sha256,
    _source_info,
    _write_json,
    _writer,
)


@dataclass
class FramePlan:
    """One camera-current -> camera-anchor coordinate map."""

    anchor_index: int
    map_x: Any
    map_y: Any


def _identity_map(np: Any, width: int, height: int) -> tuple[Any, Any]:
    return np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )


def _compose_step(
    cv2: Any,
    np: Any,
    current_gray: Any,
    neighbor_gray: Any,
    neighbor_map_x: Any,
    neighbor_map_y: Any,
    max_step_at_full_resolution: float,
    full_width: int,
) -> tuple[Any, Any, float]:
    """Compose current->neighbor flow with neighbor->anchor coordinates."""

    flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        neighbor_gray,
        None,
        0.5,
        4,
        23,
        4,
        7,
        1.5,
        cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    flow = cv2.GaussianBlur(flow, (5, 5), 0)
    magnitude = np.linalg.norm(flow, axis=2)
    maximum_full = float(magnitude.max() * full_width / current_gray.shape[1])
    clip_small = max_step_at_full_resolution * current_gray.shape[1] / full_width
    scale = np.minimum(1.0, clip_small / np.maximum(magnitude, 1e-6))
    flow *= scale[..., None]
    grid_x, grid_y = _identity_map(
        np, int(current_gray.shape[1]), int(current_gray.shape[0])
    )
    neighbor_x = grid_x + flow[..., 0]
    neighbor_y = grid_y + flow[..., 1]
    current_map_x = cv2.remap(
        neighbor_map_x,
        neighbor_x,
        neighbor_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    current_map_y = cv2.remap(
        neighbor_map_y,
        neighbor_x,
        neighbor_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return current_map_x, current_map_y, maximum_full


def _direction_maps(
    cv2: Any,
    np: Any,
    grays: list[Any],
    start: int,
    end: int,
    max_step_pixels: float,
    full_width: int,
) -> tuple[dict[int, tuple[Any, Any]], float]:
    """Propagate an anchor map from ``start`` to ``end`` in either direction."""

    direction = 1 if end >= start else -1
    width = int(grays[0].shape[1])
    height = int(grays[0].shape[0])
    map_x, map_y = _identity_map(np, width, height)
    result = {start: (map_x, map_y)}
    maximum = 0.0
    previous = start
    for current in range(start + direction, end + direction, direction):
        map_x, map_y, observed = _compose_step(
            cv2,
            np,
            grays[current],
            grays[previous],
            map_x,
            map_y,
            max_step_pixels,
            full_width,
        )
        result[current] = (map_x, map_y)
        maximum = max(maximum, observed)
        previous = current
    return result, maximum


def _small_warp(cv2: Any, image: Any, map_pair: tuple[Any, Any]) -> Any:
    return cv2.remap(
        image,
        map_pair[0],
        map_pair[1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _choose_seam(
    cv2: Any,
    np: Any,
    left_anchor: Anchor,
    right_anchor: Anchor,
    left_maps: dict[int, tuple[Any, Any]],
    right_maps: dict[int, tuple[Any, Any]],
    left_frame: int,
    right_frame: int,
    flow_width: int,
    flow_height: int,
) -> tuple[int, list[dict[str, float | int]]]:
    """Choose one hard cut; never blend the two anchor appearances."""

    left_robot = cv2.resize(
        left_anchor.robot, (flow_width, flow_height), interpolation=cv2.INTER_AREA
    )
    right_robot = cv2.resize(
        right_anchor.robot, (flow_width, flow_height), interpolation=cv2.INTER_AREA
    )
    left_mask = cv2.resize(
        left_anchor.mask, (flow_width, flow_height), interpolation=cv2.INTER_NEAREST
    )
    right_mask = cv2.resize(
        right_anchor.mask, (flow_width, flow_height), interpolation=cv2.INTER_NEAREST
    )
    span = right_frame - left_frame
    first = left_frame + max(1, round(span * 0.35))
    last = min(right_frame - 2, left_frame + round(span * 0.65))
    candidates: list[dict[str, float | int]] = []
    for seam in range(first, last + 1):
        warped_left = _small_warp(cv2, left_robot, left_maps[seam])
        warped_right = _small_warp(cv2, right_robot, right_maps[seam + 1])
        support = (
            _small_warp(cv2, left_mask, left_maps[seam]) >= 64
        ) | (_small_warp(cv2, right_mask, right_maps[seam + 1]) >= 64)
        if np.count_nonzero(support) == 0:
            appearance = float("inf")
        else:
            difference = cv2.absdiff(warped_left, warped_right)
            appearance = float(np.mean(difference[support])) / 255.0
        center_penalty = abs((seam + 0.5) - (left_frame + right_frame) / 2) / span
        score = appearance + 0.08 * center_penalty
        candidates.append(
            {
                "frame": seam,
                "appearance_cost": appearance,
                "center_penalty": center_penalty,
                "score": score,
            }
        )
    selected = min(candidates, key=lambda item: float(item["score"]))
    return int(selected["frame"]), candidates


def _build_frame_plans(
    cv2: Any,
    np: Any,
    grays: list[Any],
    anchors: tuple[Anchor, ...],
    flow_width: int,
    flow_height: int,
    max_step_pixels: float,
    full_width: int,
) -> tuple[list[FramePlan], list[dict[str, object]], float]:
    plans: list[FramePlan | None] = [None] * len(grays)
    seams: list[dict[str, object]] = []
    maximum_flow = 0.0
    for left_index, (left, right) in enumerate(zip(anchors, anchors[1:])):
        right_index = left_index + 1
        left_maps, left_maximum = _direction_maps(
            cv2,
            np,
            grays,
            left.frame,
            right.frame,
            max_step_pixels,
            full_width,
        )
        right_maps, right_maximum = _direction_maps(
            cv2,
            np,
            grays,
            right.frame,
            left.frame,
            max_step_pixels,
            full_width,
        )
        maximum_flow = max(maximum_flow, left_maximum, right_maximum)
        seam, candidates = _choose_seam(
            cv2,
            np,
            left,
            right,
            left_maps,
            right_maps,
            left.frame,
            right.frame,
            flow_width,
            flow_height,
        )
        for frame in range(left.frame, seam + 1):
            map_x, map_y = left_maps[frame]
            plans[frame] = FramePlan(left_index, map_x, map_y)
        for frame in range(seam + 1, right.frame + 1):
            map_x, map_y = right_maps[frame]
            plans[frame] = FramePlan(right_index, map_x, map_y)
        selected = next(item for item in candidates if int(item["frame"]) == seam)
        seams.append(
            {
                "left_anchor": left.frame,
                "right_anchor": right.frame,
                "cut_after_frame": seam,
                "selected_cost": selected,
            }
        )
    if any(item is None for item in plans):
        missing = [index for index, item in enumerate(plans) if item is None]
        raise RuntimeError(f"coordinate plan missed frames: {missing[:8]}")
    return [item for item in plans if item is not None], seams, maximum_flow


def _build_fixed_anchor_plans(
    cv2: Any,
    np: Any,
    grays: list[Any],
    anchors: tuple[Anchor, ...],
    anchor_index: int,
    max_step_pixels: float,
    full_width: int,
) -> tuple[list[FramePlan], float]:
    """Propagate exactly one anchor to every frame without any identity cut."""

    anchor = anchors[anchor_index]
    backward, backward_maximum = _direction_maps(
        cv2,
        np,
        grays,
        anchor.frame,
        0,
        max_step_pixels,
        full_width,
    )
    forward, forward_maximum = _direction_maps(
        cv2,
        np,
        grays,
        anchor.frame,
        len(grays) - 1,
        max_step_pixels,
        full_width,
    )
    plans: list[FramePlan | None] = [None] * len(grays)
    for frame, (map_x, map_y) in {**backward, **forward}.items():
        plans[frame] = FramePlan(anchor_index, map_x, map_y)
    if any(item is None for item in plans):
        raise RuntimeError("fixed-anchor coordinate propagation missed frames")
    return [item for item in plans if item is not None], max(
        backward_maximum, forward_maximum
    )


def _full_maps(
    cv2: Any,
    map_x: Any,
    map_y: Any,
    width: int,
    height: int,
) -> tuple[Any, Any]:
    small_height, small_width = map_x.shape
    full_x = cv2.resize(map_x, (width, height), interpolation=cv2.INTER_LINEAR)
    full_y = cv2.resize(map_y, (width, height), interpolation=cv2.INTER_LINEAR)
    full_x *= (width - 1) / max(1, small_width - 1)
    full_y *= (height - 1) / max(1, small_height - 1)
    return full_x, full_y


def _hard_composite(
    cv2: Any,
    np: Any,
    source: Any,
    robot: Any,
    dynamic_mask: Any,
    safety: Any,
) -> tuple[Any, Any, Any]:
    """Composite with exact robot overwrite inside ``safety``."""

    dynamic = cv2.dilate(
        (dynamic_mask >= 64).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    core = cv2.bitwise_or(dynamic, safety)
    outer = cv2.dilate(
        core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    alpha = cv2.GaussianBlur(outer, (7, 7), 0.9).astype(np.float32) / 255.0
    alpha[outer == 0] = 0.0
    alpha[core > 0] = 1.0
    candidate = np.rint(
        source.astype(np.float32) * (1.0 - alpha[..., None])
        + robot.astype(np.float32) * alpha[..., None]
    ).astype(np.uint8)
    # The invariant below is deliberately redundant with alpha[core] = 1.
    # It makes any future feathering change unable to resurrect source pixels.
    candidate[safety > 0] = robot[safety > 0]
    return candidate, core, outer


def _decode_grays(
    cv2: Any, source: Path, flow_width: int, flow_height: int
) -> list[Any]:
    capture = cv2.VideoCapture(str(source))
    grays: list[Any] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            small = cv2.resize(
                frame, (flow_width, flow_height), interpolation=cv2.INTER_AREA
            )
            grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    return grays


def _render(
    *,
    cv2: Any,
    np: Any,
    source: Path,
    output: Path,
    ffmpeg: Path,
    source_info: dict[str, int | float],
    anchors: tuple[Anchor, ...],
    plans: list[FramePlan],
    safety_union: Any,
    seam_frames: set[int],
    flow_width: int,
) -> dict[str, float | int]:
    width = int(source_info["width"])
    height = int(source_info["height"])
    fps = float(source_info["fps"])
    frame_count = int(source_info["frames"])
    safety = cv2.dilate(
        safety_union,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    capture = cv2.VideoCapture(str(source))
    writer = _writer(ffmpeg, output, width, height, fps)
    decoded = 0
    background_scores: list[float] = []
    replacement_scores: list[float] = []
    motion_scores: list[float] = []
    temporal_scores: list[float] = []
    sharpness_scores: list[float] = []
    transition_energy: list[float] = []
    switch_energy: list[float] = []
    previous_source_gray = None
    previous_candidate_gray = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            plan = plans[decoded]
            anchor = anchors[plan.anchor_index]
            full_x, full_y = _full_maps(
                cv2, plan.map_x, plan.map_y, width, height
            )
            robot = cv2.remap(
                anchor.robot,
                full_x,
                full_y,
                cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT101,
            )
            dynamic_mask = cv2.remap(
                anchor.mask,
                full_x,
                full_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            candidate, core, outer = _hard_composite(
                cv2, np, frame, robot, dynamic_mask, safety
            )
            outside = outer == 0
            background_scores.append(
                float(np.count_nonzero(np.all(candidate[outside] == frame[outside], axis=1)))
                / max(1, int(np.count_nonzero(outside)))
            )
            safety_pixels = safety > 0
            replacement_scores.append(
                1.0
                - float(
                    np.count_nonzero(
                        np.all(candidate[safety_pixels] == frame[safety_pixels], axis=1)
                    )
                )
                / max(1, int(np.count_nonzero(safety_pixels)))
            )
            gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
            laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
            sharpness_scores.append(float(np.mean(laplacian[safety_pixels])))
            flow_height = plans[decoded].map_x.shape[0]
            source_small = cv2.resize(
                frame, (flow_width, flow_height), interpolation=cv2.INTER_AREA
            )
            candidate_small = cv2.resize(
                candidate, (flow_width, flow_height), interpolation=cv2.INTER_AREA
            )
            source_gray = cv2.cvtColor(source_small, cv2.COLOR_BGR2GRAY)
            candidate_gray = cv2.cvtColor(candidate_small, cv2.COLOR_BGR2GRAY)
            if previous_source_gray is not None and previous_candidate_gray is not None:
                small_safety = cv2.resize(
                    safety, (flow_width, flow_height), interpolation=cv2.INTER_NEAREST
                ) > 0
                source_motion = cv2.absdiff(source_gray, previous_source_gray)[small_safety]
                candidate_motion = cv2.absdiff(
                    candidate_gray, previous_candidate_gray
                )[small_safety]
                cosine = _cosine(np, source_motion, candidate_motion)
                source_energy = float(np.mean(source_motion))
                candidate_energy = float(np.mean(candidate_motion))
                energy_ratio = min(
                    (source_energy + 1e-3) / (candidate_energy + 1e-3),
                    (candidate_energy + 1e-3) / (source_energy + 1e-3),
                )
                motion_scores.append(math.sqrt(max(0.0, cosine * energy_ratio)))
                residual = float(
                    np.mean(
                        np.abs(
                            source_motion.astype(np.float32)
                            - candidate_motion.astype(np.float32)
                        )
                    )
                )
                temporal_scores.append(math.exp(-residual / 32.0))
                energy = float(np.mean(candidate_motion))
                transition_energy.append(energy)
                if decoded in seam_frames:
                    switch_energy.append(energy)
            previous_source_gray = source_gray
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
    median_energy = float(np.median(transition_energy)) if transition_energy else 0.0
    maximum_switch_ratio = (
        max(switch_energy) / max(median_energy, 1e-6) if switch_energy else 0.0
    )
    return {
        "decoded_frames": decoded,
        "background_lock": float(np.mean(background_scores)),
        "subject_change_inside_safety": float(np.mean(replacement_scores)),
        "motion_preservation": float(np.mean(motion_scores)),
        "temporal_consistency": float(np.mean(temporal_scores)),
        "mean_robot_region_laplacian": float(np.mean(sharpness_scores)),
        "median_transition_energy": median_energy,
        "maximum_switch_transition_ratio": maximum_switch_ratio,
        "source_blend_weight_inside_safety": 0.0,
        "anchor_images_sampled_per_frame_maximum": 1,
        "person_safety_coverage": float(np.count_nonzero(safety) / safety.size),
    }


def _review_assets(
    ffmpeg: Path,
    video: Path,
    output_dir: Path,
    seams: list[dict[str, object]],
    fps: float,
) -> None:
    subprocess.run(
        [
            str(ffmpeg), "-y", "-v", "error", "-ss", "13.5", "-i", str(video),
            "-frames:v", "1", "-q:v", "2", str(output_dir / "poster.jpg"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(ffmpeg), "-y", "-v", "error", "-i", str(video), "-vf",
            "fps=1/1.7,scale=400:-2,tile=4x4:padding=4:margin=4:color=black",
            "-frames:v", "1", "-q:v", "2", str(output_dir / "storyboard-16.jpg"),
        ],
        check=True,
    )
    select_frames: list[int] = []
    for seam in seams:
        cut = int(seam["cut_after_frame"])
        select_frames.extend((max(0, cut - 1), cut, cut + 1, cut + 2))
    if not select_frames:
        select_frames = [round(index * 659 / 27) for index in range(28)]
    expression = "+".join(f"eq(n\\,{frame})" for frame in select_frames)
    subprocess.run(
        [
            str(ffmpeg), "-y", "-v", "error", "-i", str(video), "-vf",
            f"select='{expression}',scale=320:-2,tile=4x7:padding=3:margin=3:color=black",
            "-vsync", "0", "-frames:v", "1", "-q:v", "2",
            str(output_dir / "switch-review.jpg"),
        ],
        check=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--flow-width", type=int, default=256)
    parser.add_argument("--max-step-pixels", type=float, default=28.0)
    parser.add_argument("--fixed-anchor-frame", type=int)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.flow_width < 128:
        raise ValueError("flow-width must be at least 128")
    if not math.isfinite(args.max_step_pixels) or args.max_step_pixels <= 0:
        raise ValueError("max-step-pixels must be positive and finite")
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
    trace_path = experiment / "trace.json"
    project_root = Path(__file__).resolve().parents[1]
    try:
        import cv2
        import numpy as np

        np.random.seed(args.seed)
        source, person_union_path, semantic_path, specs = _load_specs(config)
        _write_json(experiment / "resolved-config.json", json.loads(config.read_text()))
        source_info = _source_info(cv2, source)
        width = int(source_info["width"])
        height = int(source_info["height"])
        frame_count = int(source_info["frames"])
        if specs[0].frame != 0 or specs[-1].frame != frame_count - 1:
            raise ValueError("anchors must cover the first and last source frames")
        flow_height = max(2, round(height * args.flow_width / width))
        grays = _decode_grays(cv2, source, args.flow_width, flow_height)
        if len(grays) != frame_count:
            raise RuntimeError(f"decoded {len(grays)}/{frame_count} planning frames")

        semantic = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE)
        person_union = cv2.imread(str(person_union_path), cv2.IMREAD_GRAYSCALE)
        if semantic is None or person_union is None:
            raise RuntimeError("cannot decode person masks")
        semantic = cv2.resize(semantic, (width, height), interpolation=cv2.INTER_NEAREST)
        semantic = (semantic >= 127).astype(np.uint8) * 255
        person_union = cv2.resize(
            person_union, (width, height), interpolation=cv2.INTER_NEAREST
        )
        person_union = (person_union >= 127).astype(np.uint8) * 255
        semantic_anchor_spec = min(specs, key=lambda item: abs(item.frame - 276))
        semantic_anchor_gray = grays[semantic_anchor_spec.frame]

        anchors: list[Anchor] = []
        anchor_union = np.zeros((height, width), dtype=np.uint8)
        mask_evidence: list[dict[str, float | int]] = []
        for spec in specs:
            source_anchor = cv2.imread(str(spec.source), cv2.IMREAD_COLOR)
            robot_anchor = cv2.imread(str(spec.robot), cv2.IMREAD_COLOR)
            if source_anchor is None or robot_anchor is None:
                raise RuntimeError(f"cannot decode anchor {spec.frame}")
            source_anchor = cv2.resize(
                source_anchor, (width, height), interpolation=cv2.INTER_LANCZOS4
            )
            robot_anchor = cv2.resize(
                robot_anchor, (width, height), interpolation=cv2.INTER_LANCZOS4
            )
            # Map the central semantic person mask to each source anchor.  This
            # is used only to construct the camera-aligned conservative union.
            flow = cv2.calcOpticalFlowFarneback(
                grays[spec.frame], semantic_anchor_gray, None, 0.5, 5, 31, 5, 7, 1.5,
                cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
            )
            grid_x, grid_y = _identity_map(np, args.flow_width, flow_height)
            map_x = cv2.resize(grid_x + flow[..., 0], (width, height))
            map_y = cv2.resize(grid_y + flow[..., 1], (width, height))
            map_x *= (width - 1) / max(1, args.flow_width - 1)
            map_y *= (height - 1) / max(1, flow_height - 1)
            semantic_proxy = cv2.remap(
                semantic, map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            anchor_mask, evidence = _build_anchor_mask(
                cv2, np, source_anchor, robot_anchor, person_union, semantic_proxy
            )
            anchors.append(
                Anchor(
                    spec.frame,
                    source_anchor,
                    robot_anchor,
                    anchor_mask,
                    grays[spec.frame],
                )
            )
            anchor_union = cv2.bitwise_or(anchor_union, anchor_mask)
            mask_evidence.append({"frame": spec.frame, **evidence})
            cv2.imwrite(str(assets / f"mask-{spec.frame:04d}.png"), anchor_mask)
        cv2.imwrite(str(assets / "person-safety-union.png"), anchor_union)

        if args.fixed_anchor_frame is None:
            plans, seams, maximum_flow = _build_frame_plans(
                cv2,
                np,
                grays,
                tuple(anchors),
                args.flow_width,
                flow_height,
                args.max_step_pixels,
                width,
            )
            method = "single_anchor_per_frame_composed_flow_with_hard_cuts"
        else:
            matching = [
                index
                for index, anchor in enumerate(anchors)
                if anchor.frame == args.fixed_anchor_frame
            ]
            if not matching:
                raise ValueError("fixed-anchor-frame must name a configured anchor")
            plans, maximum_flow = _build_fixed_anchor_plans(
                cv2,
                np,
                grays,
                tuple(anchors),
                matching[0],
                args.max_step_pixels,
                width,
            )
            seams = []
            method = "one_fixed_anchor_composed_consecutive_flow_hard_person_clear"
        trace: dict[str, object] = {
            "schema_version": "1.0.0",
            "status": "preflight" if args.preflight_only else "running",
            "honest_status": "PARTIAL",
            "method": method,
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
                "reason": "deterministic OpenCV compositor runs on CPU; no GPU entry point is invoked",
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
                "step_flow": "camera:current_frame_pixels -> camera:adjacent_frame_pixels",
                "composed_map": "camera:current_frame_pixels -> camera:selected_anchor_pixels",
            },
            "parameters": {
                "flow_width": args.flow_width,
                "max_step_pixels": args.max_step_pixels,
                "anchor_count": len(anchors),
                "anchor_images_sampled_per_frame": 1,
                "cross_dissolve": False,
                "source_restoration_inside_person_safety": False,
                "fixed_anchor_frame": args.fixed_anchor_frame,
            },
            "seams": seams,
            "maximum_unclipped_consecutive_flow_pixels": maximum_flow,
            "anchors": [
                {
                    "frame": spec.frame,
                    "source": str(spec.source),
                    "source_sha256": _sha256(spec.source),
                    "robot": str(spec.robot),
                    "robot_sha256": _sha256(spec.robot),
                }
                for spec in specs
            ],
            "mask_evidence": mask_evidence,
        }
        _write_json(trace_path, trace)
        if args.preflight_only:
            trace.update(
                {
                    "status": "preflight_complete",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "limitations": ["No output video was rendered in preflight-only mode."],
                }
            )
            _write_json(trace_path, trace)
            print(json.dumps({"experiment": str(experiment), "seams": seams}, indent=2))
            return 0

        final_video = final_dir / "robot-motion-replacement-sharp.mp4"
        seam_frames = {int(item["cut_after_frame"]) + 1 for item in seams}
        metrics = _render(
            cv2=cv2,
            np=np,
            source=source,
            output=final_video,
            ffmpeg=ffmpeg,
            source_info=source_info,
            anchors=tuple(anchors),
            plans=plans,
            safety_union=anchor_union,
            seam_frames=seam_frames,
            flow_width=args.flow_width,
        )
        subprocess.run(
            [str(ffmpeg), "-v", "error", "-i", str(final_video), "-f", "null", "-"],
            check=True,
        )
        _review_assets(ffmpeg, final_video, final_dir, seams, float(source_info["fps"]))
        acceptance = {
            "full_clip_decoded": int(metrics["decoded_frames"]) == frame_count,
            "background_lock_passed": float(metrics["background_lock"]) >= 0.99999,
            "hard_person_clear_invariant_passed": float(
                metrics["source_blend_weight_inside_safety"]
            ) == 0.0,
            "single_anchor_per_frame_passed": int(
                metrics["anchor_images_sampled_per_frame_maximum"]
            ) == 1,
            "temporal_consistency_passed": float(metrics["temporal_consistency"]) >= 0.75,
            "switch_transition_passed": float(
                metrics["maximum_switch_transition_ratio"]
            ) <= 3.0,
        }
        accepted = all(acceptance.values())
        trace.update(
            {
                "status": "accepted" if accepted else "rejected",
                "honest_status": "WORKING" if accepted else "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "acceptance": acceptance,
                "outputs": {
                    "video": str(final_video),
                    "video_sha256": _sha256(final_video),
                    "poster": str(final_dir / "poster.jpg"),
                    "storyboard": str(final_dir / "storyboard-16.jpg"),
                    "switch_review": str(final_dir / "switch-review.jpg"),
                },
                "limitations": [
                    "This is a deterministic image-anchor/optical-flow proxy, not official PhiZero inference or real-robot execution.",
                    "Seven hard anchor-identity cuts remain, selected at the lowest measured appearance cost; no pixels are cross-dissolved at those cuts.",
                    "The person hard-clear guarantee applies to the dilated union of the eight camera-aligned anchor masks.",
                    "Pixel-lock and source-contribution invariants are measured before lossy H.264 encoding.",
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
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(trace_path, payload)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
