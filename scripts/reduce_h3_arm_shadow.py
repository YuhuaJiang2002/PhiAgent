#!/usr/bin/env python3
"""Conservatively reduce only neutral shadow beside H3 robot arms."""

from __future__ import annotations

import argparse
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
from scripts.compose_h3_layered_replacement import (  # noqa: E402
    _align_mask,
    _annotate,
    _git_state,
    _load_packed,
    _package_version,
    _sheet,
    _stabilize_area_outliers,
    _skin_like,
    _video_info,
    _writer,
    apply_conservative_arm_shadow_cleanup,
    build_conservative_arm_shadow_alpha,
    build_residual_arm_skin_support,
    build_tracked_robot_arm_material,
    build_tracked_polygon_alpha,
    fill_selected_component_hulls,
    select_arm_skin_components,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _mask_centroids(np: Any, masks: Any) -> list[tuple[float, float] | None]:
    """Return camera-frame centroids without inventing a track through empty masks."""
    centroids: list[tuple[float, float] | None] = []
    for mask in masks:
        y_coordinates, x_coordinates = np.nonzero(mask)
        if x_coordinates.size == 0:
            centroids.append(None)
        else:
            centroids.append(
                (float(np.mean(x_coordinates)), float(np.mean(y_coordinates)))
            )
    return centroids


def _smooth_centroids(
    np: Any,
    centroids: list[tuple[float, float] | None],
    *,
    radius: int,
) -> list[tuple[float, float] | None]:
    """Median-filter observed centroids while keeping absent current frames absent."""
    if radius == 0:
        return list(centroids)
    smoothed: list[tuple[float, float] | None] = []
    for index, centroid in enumerate(centroids):
        if centroid is None:
            smoothed.append(None)
            continue
        neighborhood = [
            item
            for item in centroids[
                max(0, index - radius) : min(len(centroids), index + radius + 1)
            ]
            if item is not None
        ]
        smoothed.append(
            (
                float(np.median([item[0] for item in neighborhood])),
                float(np.median([item[1] for item in neighborhood])),
            )
        )
    return smoothed


def _align_alpha_to_motion(
    cv2: Any,
    np: Any,
    alpha: Any,
    *,
    source_centroid: tuple[float, float] | None,
    target_centroid: tuple[float, float] | None,
    threshold_pixels: float,
    maximum_shift_pixels: float,
    blend_ramp_pixels: float,
    protect_threshold: float,
) -> tuple[Any, bool, float]:
    """Translate a neighboring alpha into the current arm's camera-frame position."""
    if source_centroid is None or target_centroid is None:
        return alpha, False, 0.0
    delta_x = target_centroid[0] - source_centroid[0]
    delta_y = target_centroid[1] - source_centroid[1]
    distance = float(np.hypot(delta_x, delta_y))
    if distance <= threshold_pixels or distance == 0.0:
        return alpha, False, distance
    scale = min(1.0, maximum_shift_pixels / distance)
    confidence = min(1.0, (distance - threshold_pixels) / blend_ramp_pixels)
    matrix = np.asarray(
        [[1.0, 0.0, delta_x * scale], [0.0, 1.0, delta_y * scale]],
        dtype=np.float32,
    )
    shadow_alpha = np.where(alpha < protect_threshold, alpha, 0.0)
    warped_shadow = cv2.warpAffine(
        shadow_alpha,
        matrix,
        (alpha.shape[1], alpha.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    aligned = alpha * (1.0 - confidence) + warped_shadow * confidence
    aligned[alpha >= protect_threshold] = alpha[alpha >= protect_threshold]
    return aligned, True, distance * scale


def _apply_plate_alpha_gain(
    np: Any,
    alpha: Any,
    *,
    gain: float,
    cap: float,
    protect_threshold: float,
) -> Any:
    """Lighten low-alpha shadow support without weakening reviewed hard negatives."""
    result = alpha.copy()
    lightening_band = alpha < protect_threshold
    result[lightening_band] = np.clip(
        alpha[lightening_band] * gain,
        0.0,
        cap,
    )
    return result


def _temporal_union_masks(np: Any, masks: Any, *, radius: int) -> Any:
    """Expand a camera-frame mask track without inventing cross-frame motion."""
    if not 0 <= radius <= 4:
        raise ValueError("temporal union radius must be between 0 and 4")
    result = masks.copy()
    for offset in range(1, radius + 1):
        result[offset:] = np.logical_or(result[offset:], masks[:-offset])
        result[:-offset] = np.logical_or(result[:-offset], masks[offset:])
    return result


def _build_wide_person_coverage_alpha(
    cv2: Any,
    np: Any,
    *,
    person_mask: Any,
    protected_mask: Any,
    edit_safety: Any,
    coverage_region: Any | None = None,
    dilation: int,
    feather_sigma: float,
    strength: float,
) -> Any:
    """Build a broad, soft plate matte for residual source-person pixels.

    The semantic source-person track supplies the core. Robot and flower masks
    remain authoritative, while the safety mask prevents background-wide edits.
    """
    if person_mask.shape != protected_mask.shape or person_mask.shape != edit_safety.shape:
        raise ValueError("wide person masks must share camera-frame geometry")
    if dilation < 0 or feather_sigma < 0:
        raise ValueError("wide person dilation and feather must be non-negative")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("wide person strength must be in [0, 1]")
    core = person_mask.astype(bool)
    if dilation:
        core = cv2.dilate(
            core.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilation * 2 + 1, dilation * 2 + 1),
            ),
        ) > 0
    core = np.logical_and(core, edit_safety)
    if coverage_region is not None:
        if coverage_region.shape != core.shape:
            raise ValueError("wide person coverage region has incorrect geometry")
        core = np.logical_and(core, coverage_region)
    core = np.logical_and(core, np.logical_not(protected_mask))
    alpha = core.astype(np.float32)
    if feather_sigma:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather_sigma)
        alpha[core] = 1.0
    alpha *= float(strength)
    alpha[protected_mask] = 0.0
    alpha[np.logical_not(edit_safety)] = 0.0
    return np.clip(alpha, 0.0, 1.0)


def _build_wide_person_graphite_material(cv2: Any, np: Any, frame: Any) -> Any:
    """Neutralize source clothing into a light graphite shadow material."""
    luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Keep most local luminance so a deliberately wide matte reads as a soft
    # shadow instead of a dark duplicate silhouette. Chroma removal, rather
    # than aggressive darkening, is what hides lavender clothing remnants.
    value = np.clip(luminance * 0.84 + 14.0, 32.0, 190.0)
    return np.stack(
        [
            np.clip(value * 1.025, 0.0, 255.0),
            value,
            np.clip(value * 0.975, 0.0, 255.0),
        ],
        axis=2,
    ).astype(np.uint8)


def _temporal_metrics(cv2: Any, np: Any, path: Path, roi: Any) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(
            cv2.cvtColor(
                cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            ).astype(np.float32)
        )
    capture.release()
    array = np.stack(frames)
    transition = np.abs(np.diff(array, axis=0))
    jerk = np.abs(array[2:] - 2 * array[1:-1] + array[:-2])
    roi_small = cv2.resize(
        roi.astype(np.uint8), (256, 144), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    def stats(values: Any, mask: Any | None = None) -> dict[str, float]:
        if mask is not None:
            values = values[:, mask]
            reduced = np.mean(values, axis=1)
        else:
            reduced = np.mean(values, axis=(1, 2))
        return {
            "mean": float(np.mean(reduced)),
            "p95": float(np.percentile(reduced, 95)),
            "maximum": float(np.max(reduced)),
        }

    return {
        "frames": len(array),
        "transition_full": stats(transition),
        "jerk_full": stats(jerk),
        "transition_roi": stats(transition, roi_small),
        "jerk_roi": stats(jerk, roi_small),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-video", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--clean-plate", type=Path, required=True)
    parser.add_argument("--source-safety-mask", type=Path, required=True)
    parser.add_argument("--source-person-masks", type=Path)
    parser.add_argument("--robot-body-masks", type=Path, required=True)
    parser.add_argument("--robot-wrist-masks", type=Path, required=True)
    parser.add_argument("--robot-limb-masks", type=Path, required=True)
    parser.add_argument("--generated-flower-masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="minimax-h3-epl-arm-shadow-reduced.mp4")
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--protect-radius", type=int, default=8)
    parser.add_argument("--cleanup-radius", type=int, default=44)
    parser.add_argument("--maximum-strength", type=float, default=0.55)
    parser.add_argument("--skin-strength", type=float, default=0.0)
    parser.add_argument("--neutral-chroma-limit", type=float, default=90.0)
    parser.add_argument("--difference-threshold", type=float, default=10.0)
    parser.add_argument("--feather-sigma", type=float, default=1.8)
    parser.add_argument("--temporal-radius", type=int, default=2)
    parser.add_argument("--limb-consensus-radius", type=int, default=3)
    parser.add_argument("--maximum-edit-fraction", type=float, default=0.04)
    parser.add_argument("--flower-protect-radius", type=int, default=3)
    parser.add_argument("--safety-dilation", type=int, default=0)
    parser.add_argument("--skin-safety-dilation", type=int)
    parser.add_argument("--skin-cleanup-radius", type=int)
    parser.add_argument("--skin-edge-override", action="store_true")
    parser.add_argument("--skin-component-filter", action="store_true")
    parser.add_argument("--skin-component-roi", default="360,150,620,340")
    parser.add_argument("--skin-component-arm-dilation", type=int, default=16)
    parser.add_argument("--skin-component-min-area", type=int, default=300)
    parser.add_argument("--skin-component-max-area", type=int)
    parser.add_argument("--skin-component-min-arm-overlap", type=float, default=0.12)
    parser.add_argument("--skin-component-hulls", action="store_true")
    parser.add_argument("--high-alpha-temporal-maximum", action="store_true")
    parser.add_argument("--motion-compensated-temporal", action="store_true")
    parser.add_argument("--motion-threshold-pixels", type=float, default=6.0)
    parser.add_argument("--motion-maximum-shift-pixels", type=float, default=32.0)
    parser.add_argument("--motion-blend-ramp-pixels", type=float, default=12.0)
    parser.add_argument("--motion-centroid-smoothing-radius", type=int, default=2)
    parser.add_argument("--plate-alpha-gain", type=float, default=1.0)
    parser.add_argument("--plate-alpha-cap", type=float, default=1.0)
    parser.add_argument("--plate-alpha-protect-threshold", type=float, default=0.70)
    parser.add_argument("--wide-person-strength", type=float, default=0.0)
    parser.add_argument("--wide-person-dilation", type=int, default=0)
    parser.add_argument("--wide-person-feather-sigma", type=float, default=0.0)
    parser.add_argument("--wide-person-temporal-radius", type=int, default=2)
    parser.add_argument("--wide-person-protect-radius", type=int, default=0)
    parser.add_argument("--wide-person-green-protect-radius", type=int, default=0)
    parser.add_argument("--wide-person-maximum-chroma-ratio", type=float, default=0.65)
    parser.add_argument("--wide-person-roi", default="0,0,832,480")
    parser.add_argument(
        "--wide-person-mode",
        choices=("plate", "graphite"),
        default="graphite",
    )
    parser.add_argument("--selected-skin-core-override", action="store_true")
    parser.add_argument("--residual-arm-keyframes", type=Path)
    parser.add_argument("--residual-arm-strength", type=float, default=1.0)
    parser.add_argument(
        "--residual-arm-mode",
        choices=("plate", "silver", "graphite"),
        default="plate",
    )
    parser.add_argument("--residual-arm-feather-sigma", type=float, default=4.0)
    parser.add_argument("--residual-arm-green-protect-radius", type=int, default=1)
    parser.add_argument("--residual-arm-green-min-area", type=int, default=0)
    parser.add_argument("--residual-arm-skin-constrained", action="store_true")
    parser.add_argument("--residual-arm-skin-close-width", type=int, default=25)
    parser.add_argument("--residual-arm-skin-close-height", type=int, default=11)
    parser.add_argument("--residual-arm-skin-min-area", type=int, default=500)
    parser.add_argument("--residual-arm-skin-dilation", type=int, default=6)
    parser.add_argument("--residual-arm-skin-temporal-radius", type=int, default=2)
    parser.add_argument("--residual-arm-skin-feather-sigma", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--human-review", choices=("pending", "passed", "failed"), default="pending")
    parser.add_argument("--review-notes", default="")
    parser.add_argument("--showcase-output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"immutable experiment already exists: {output_dir}")
    if args.temporal_radius < 0 or args.temporal_radius > 4:
        raise ValueError("temporal radius must be between 0 and 4")
    if args.motion_threshold_pixels < 0:
        raise ValueError("motion threshold must be non-negative")
    if args.motion_maximum_shift_pixels <= 0:
        raise ValueError("motion maximum shift must be positive")
    if args.motion_blend_ramp_pixels <= 0:
        raise ValueError("motion blend ramp must be positive")
    if not 0 <= args.motion_centroid_smoothing_radius <= 4:
        raise ValueError("motion centroid smoothing radius must be between 0 and 4")
    if args.plate_alpha_gain <= 0:
        raise ValueError("plate alpha gain must be positive")
    if not 0 < args.plate_alpha_cap <= 1:
        raise ValueError("plate alpha cap must be in (0, 1]")
    if not 0 < args.plate_alpha_protect_threshold <= 1:
        raise ValueError("plate alpha protect threshold must be in (0, 1]")
    if not 0.0 <= args.wide_person_strength <= 1.0:
        raise ValueError("wide person strength must be in [0, 1]")
    if args.wide_person_dilation < 0 or args.wide_person_feather_sigma < 0:
        raise ValueError("wide person dilation and feather must be non-negative")
    if args.wide_person_protect_radius < 0:
        raise ValueError("wide person protection radius must be non-negative")
    if args.wide_person_green_protect_radius < 0:
        raise ValueError("wide person green protection radius must be non-negative")
    if not 0 < args.wide_person_maximum_chroma_ratio <= 1:
        raise ValueError("wide person maximum chroma ratio must be in (0, 1]")
    if not 0 <= args.wide_person_temporal_radius <= 4:
        raise ValueError("wide person temporal radius must be between 0 and 4")
    if args.wide_person_strength > 0 and args.source_person_masks is None:
        raise ValueError("wide person coverage requires --source-person-masks")
    for relative in ("logs", "review", "final", "provenance/execution-sources"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    for script in (Path(__file__), PROJECT_ROOT / "scripts/compose_h3_layered_replacement.py"):
        shutil.copy2(script.resolve(), output_dir / "provenance/execution-sources" / script.name)

    paths = {
        "generated_video": args.generated_video.expanduser().resolve(),
        "source_video": args.source_video.expanduser().resolve(),
        "clean_plate": args.clean_plate.expanduser().resolve(),
        "source_safety_mask": args.source_safety_mask.expanduser().resolve(),
        "robot_body_masks": args.robot_body_masks.expanduser().resolve(),
        "robot_wrist_masks": args.robot_wrist_masks.expanduser().resolve(),
        "robot_limb_masks": args.robot_limb_masks.expanduser().resolve(),
        "generated_flower_masks": args.generated_flower_masks.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    if args.residual_arm_keyframes is not None:
        paths["residual_arm_keyframes"] = (
            args.residual_arm_keyframes.expanduser().resolve()
        )
    if args.source_person_masks is not None:
        paths["source_person_masks"] = args.source_person_masks.expanduser().resolve()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    command = [sys.executable, *sys.argv]
    (output_dir / "command.sh").write_text(shlex.join(command) + "\n")
    config = {
        "schema_version": "1.0.0",
        "method": "raw H3 authority with conservative neutral shadow attenuation and optional semantic person-wide plate coverage",
        "coordinate_frame": "camera:H3_output_pixels",
        "protect_radius": args.protect_radius,
        "cleanup_radius": args.cleanup_radius,
        "maximum_strength": args.maximum_strength,
        "skin_strength": args.skin_strength,
        "neutral_chroma_limit": args.neutral_chroma_limit,
        "difference_threshold": args.difference_threshold,
        "feather_sigma": args.feather_sigma,
        "temporal_radius": args.temporal_radius,
        "limb_consensus_radius": args.limb_consensus_radius,
        "maximum_edit_fraction": args.maximum_edit_fraction,
        "flower_protect_radius": args.flower_protect_radius,
        "safety_dilation": args.safety_dilation,
        "skin_safety_dilation": args.skin_safety_dilation,
        "skin_cleanup_radius": args.skin_cleanup_radius,
        "skin_edge_override": args.skin_edge_override,
        "skin_component_filter": args.skin_component_filter,
        "skin_component_roi": args.skin_component_roi,
        "skin_component_arm_dilation": args.skin_component_arm_dilation,
        "skin_component_min_area": args.skin_component_min_area,
        "skin_component_max_area": args.skin_component_max_area,
        "skin_component_min_arm_overlap": args.skin_component_min_arm_overlap,
        "skin_component_hulls": args.skin_component_hulls,
        "high_alpha_temporal_maximum": args.high_alpha_temporal_maximum,
        "motion_compensated_temporal": args.motion_compensated_temporal,
        "motion_threshold_pixels": args.motion_threshold_pixels,
        "motion_maximum_shift_pixels": args.motion_maximum_shift_pixels,
        "motion_blend_ramp_pixels": args.motion_blend_ramp_pixels,
        "motion_centroid_smoothing_radius": args.motion_centroid_smoothing_radius,
        "plate_alpha_gain": args.plate_alpha_gain,
        "plate_alpha_cap": args.plate_alpha_cap,
        "plate_alpha_protect_threshold": args.plate_alpha_protect_threshold,
        "source_person_masks": (
            str(paths["source_person_masks"])
            if "source_person_masks" in paths
            else None
        ),
        "wide_person_strength": args.wide_person_strength,
        "wide_person_dilation": args.wide_person_dilation,
        "wide_person_feather_sigma": args.wide_person_feather_sigma,
        "wide_person_temporal_radius": args.wide_person_temporal_radius,
        "wide_person_protect_radius": args.wide_person_protect_radius,
        "wide_person_green_protect_radius": args.wide_person_green_protect_radius,
        "wide_person_maximum_chroma_ratio": args.wide_person_maximum_chroma_ratio,
        "wide_person_mode": args.wide_person_mode,
        "wide_person_roi": args.wide_person_roi,
        "selected_skin_core_override": args.selected_skin_core_override,
        "residual_arm_keyframes": (
            str(paths["residual_arm_keyframes"])
            if "residual_arm_keyframes" in paths
            else None
        ),
        "residual_arm_strength": args.residual_arm_strength,
        "residual_arm_mode": args.residual_arm_mode,
        "residual_arm_feather_sigma": args.residual_arm_feather_sigma,
        "residual_arm_green_protect_radius": args.residual_arm_green_protect_radius,
        "residual_arm_green_min_area": args.residual_arm_green_min_area,
        "residual_arm_skin_constrained": args.residual_arm_skin_constrained,
        "residual_arm_skin_close_width": args.residual_arm_skin_close_width,
        "residual_arm_skin_close_height": args.residual_arm_skin_close_height,
        "residual_arm_skin_min_area": args.residual_arm_skin_min_area,
        "residual_arm_skin_dilation": args.residual_arm_skin_dilation,
        "residual_arm_skin_temporal_radius": args.residual_arm_skin_temporal_radius,
        "residual_arm_skin_feather_sigma": args.residual_arm_skin_feather_sigma,
        "seed": args.seed,
        "candidate_policy": "preserve raw frame except bounded neutral alpha, reviewed residual-arm material, and optional source-person-wide plate alpha",
    }
    _write_json(output_dir / "config.json", config)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "honest_status": "PARTIAL",
        "scope": "localized shadow attenuation on the user-preferred 660-frame H3+EPL track",
        "command": command,
        "seed": args.seed,
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "git": _git_state(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": {
            "used": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mode": "CPU-only composition from pinned masks",
        },
        "packages": {
            name: _package_version(name) for name in ("numpy", "opencv-python")
        },
        "config": config,
    }
    _write_json(output_dir / "manifest.json", manifest)

    generated_info = _video_info(cv2, paths["generated_video"])
    source_info = _video_info(cv2, paths["source_video"])
    if generated_info != source_info or int(generated_info["reported_frames"]) != 660:
        raise RuntimeError(f"video timeline mismatch: {generated_info} vs {source_info}")
    width, height = int(generated_info["width"]), int(generated_info["height"])
    fps = float(generated_info["fps"])
    try:
        wide_person_roi = tuple(
            int(value) for value in args.wide_person_roi.split(",")
        )
    except ValueError as error:
        raise ValueError(
            "wide person ROI must contain four comma-separated integers"
        ) from error
    if len(wide_person_roi) != 4:
        raise ValueError("wide person ROI must be x_min,y_min,x_max,y_max")
    x_min, y_min, x_max, y_max = wide_person_roi
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ValueError(
            f"wide person ROI {wide_person_roi} is outside {width}x{height}"
        )
    wide_person_region = np.zeros((height, width), dtype=bool)
    wide_person_region[y_min:y_max, x_min:x_max] = True
    if not 0.0 <= args.residual_arm_strength <= 1.0:
        raise ValueError("residual arm strength must be in [0, 1]")
    if args.residual_arm_feather_sigma < 0:
        raise ValueError("residual arm feather sigma must be non-negative")
    if args.residual_arm_green_protect_radius < 0:
        raise ValueError("residual arm green protection radius must be non-negative")
    if args.residual_arm_green_min_area < 0:
        raise ValueError("residual arm green minimum area must be non-negative")
    if args.residual_arm_skin_close_width <= 0 or args.residual_arm_skin_close_height <= 0:
        raise ValueError("residual arm skin closing dimensions must be positive")
    if args.residual_arm_skin_min_area <= 0:
        raise ValueError("residual arm skin minimum area must be positive")
    if args.residual_arm_skin_dilation < 0:
        raise ValueError("residual arm skin dilation must be non-negative")
    if not 0 <= args.residual_arm_skin_temporal_radius <= 4:
        raise ValueError("residual arm skin temporal radius must be between 0 and 4")
    if args.residual_arm_skin_feather_sigma < 0:
        raise ValueError("residual arm skin feather sigma must be non-negative")
    residual_arm_tracks: list[dict[str, Any]] = []
    residual_arm_union = np.zeros((height, width), dtype=bool)
    if "residual_arm_keyframes" in paths:
        residual_payload = json.loads(paths["residual_arm_keyframes"].read_text())
        if residual_payload.get("coordinate_frame") != "camera:H3_output_pixels":
            raise ValueError("residual arm keyframes must name camera:H3_output_pixels")
        residual_arm_tracks = residual_payload.get("tracks", [])
        if not isinstance(residual_arm_tracks, list) or not residual_arm_tracks:
            raise ValueError("residual arm keyframes must contain at least one track")
        for check_index in range(660):
            checked_alpha = build_tracked_polygon_alpha(
                cv2,
                np,
                shape=(height, width),
                tracks=residual_arm_tracks,
                frame_index=check_index,
                feather_sigma=args.residual_arm_feather_sigma,
            )
            residual_arm_union = np.logical_or(
                residual_arm_union, checked_alpha >= 0.001
            )
    try:
        skin_component_roi = tuple(int(value) for value in args.skin_component_roi.split(","))
    except ValueError as error:
        raise ValueError("skin component ROI must contain four comma-separated integers") from error
    if len(skin_component_roi) != 4:
        raise ValueError("skin component ROI must be x_min,y_min,x_max,y_max")
    clean_plate = cv2.imread(str(paths["clean_plate"]), cv2.IMREAD_COLOR)
    safety_raw = cv2.imread(str(paths["source_safety_mask"]), cv2.IMREAD_GRAYSCALE)
    if clean_plate is None or safety_raw is None or clean_plate.shape[:2] != (height, width):
        raise RuntimeError("clean plate or safety mask cannot be decoded/aligned")
    base_safety = _align_mask(cv2, safety_raw, width, height) >= 127
    safety = base_safety.copy()
    if args.safety_dilation:
        safety = cv2.dilate(
            safety.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (args.safety_dilation * 2 + 1, args.safety_dilation * 2 + 1),
            ),
        ) > 0
    skin_safety_dilation = (
        args.safety_dilation
        if args.skin_safety_dilation is None
        else args.skin_safety_dilation
    )
    if skin_safety_dilation < 0:
        raise ValueError("skin safety dilation must be non-negative")
    skin_safety = base_safety.copy()
    if skin_safety_dilation:
        skin_safety = cv2.dilate(
            skin_safety.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (skin_safety_dilation * 2 + 1, skin_safety_dilation * 2 + 1),
            ),
        ) > 0
    edit_safety = np.logical_or(safety, skin_safety)
    metric_edit_safety = np.logical_or(edit_safety, residual_arm_union)
    skin_to_core = (
        np.logical_and(skin_safety, np.logical_not(base_safety.copy()))
        if args.skin_edge_override
        else None
    )

    body = _load_packed(np, paths["robot_body_masks"], "packed")
    wrist = _load_packed(np, paths["robot_wrist_masks"], "packed")
    raw_limbs = _load_packed(np, paths["robot_limb_masks"], "packed")
    flowers = _load_packed(np, paths["generated_flower_masks"], "packed")
    source_person = (
        _load_packed(np, paths["source_person_masks"], "packed")
        if "source_person_masks" in paths
        else None
    )
    flower_outlier_frames = _stabilize_area_outliers(np, flowers, ratio=1.65)
    expected_shape = (660, height, width)
    if any(item.shape != expected_shape for item in (body, wrist, raw_limbs, flowers)):
        raise RuntimeError("one or more mask tracks have incorrect geometry")
    if source_person is not None and source_person.shape != expected_shape:
        raise RuntimeError("source person masks have incorrect camera-frame geometry")
    wide_person_support = (
        _temporal_union_masks(
            np,
            source_person,
            radius=args.wide_person_temporal_radius,
        )
        if source_person is not None and args.wide_person_strength > 0
        else None
    )

    protected = np.zeros(expected_shape, dtype=bool)
    arms = np.zeros(expected_shape, dtype=bool)
    consensus_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.limb_consensus_radius * 2 + 1, args.limb_consensus_radius * 2 + 1),
    )
    skin_cleanup_radius = (
        args.cleanup_radius
        if args.skin_cleanup_radius is None
        else args.skin_cleanup_radius
    )
    if skin_cleanup_radius <= args.protect_radius:
        raise ValueError("skin cleanup radius must exceed protect radius")
    for index in range(660):
        wrist_support = cv2.dilate(
            wrist[index].astype(np.uint8) * 255, consensus_kernel
        ) > 0
        arms[index] = np.logical_and(raw_limbs[index], wrist_support)
        protected[index] = np.logical_or(body[index], arms[index])
        protected[index] = np.logical_or(protected[index], flowers[index])
    arm_centroids = _smooth_centroids(
        np,
        _mask_centroids(np, arms),
        radius=args.motion_centroid_smoothing_radius,
    )
    stable_protected = protected.copy()
    for offset in range(1, args.temporal_radius + 1):
        stable_protected[offset:] = np.logical_or(
            stable_protected[offset:], protected[:-offset]
        )
        stable_protected[:-offset] = np.logical_or(
            stable_protected[:-offset], protected[offset:]
        )
    wide_person_protected = stable_protected
    if wide_person_support is not None:
        wide_person_protected = np.logical_or(body, raw_limbs)
        wide_person_protected = np.logical_or(wide_person_protected, wrist)
        wide_person_protected = np.logical_or(wide_person_protected, flowers)
        if args.wide_person_protect_radius:
            wide_protect_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    args.wide_person_protect_radius * 2 + 1,
                    args.wide_person_protect_radius * 2 + 1,
                ),
            )
            wide_person_protected = np.stack(
                [
                    cv2.dilate(mask.astype(np.uint8) * 255, wide_protect_kernel)
                    > 0
                    for mask in wide_person_protected
                ]
            )
        wide_person_protected = _temporal_union_masks(
            np,
            wide_person_protected,
            radius=args.wide_person_temporal_radius,
        )

    alpha_seed = np.zeros(expected_shape, dtype=np.uint8)
    residual_skin_seed = (
        np.zeros(expected_shape, dtype=np.uint8)
        if args.residual_arm_skin_constrained
        else None
    )
    capture = cv2.VideoCapture(str(paths["generated_video"]))
    for index in range(660):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"generated decode ended on frame {index}")
        flower_buffer = cv2.dilate(
            flowers[index].astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    args.flower_protect_radius * 2 + 1,
                    args.flower_protect_radius * 2 + 1,
                ),
            ),
        ) > 0
        skin_component = _skin_like(cv2, np, frame)
        skin_component = cv2.morphologyEx(
            skin_component.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        ) > 0
        if args.skin_component_filter:
            skin_component = select_arm_skin_components(
                cv2,
                np,
                skin_mask=skin_component,
                arm_mask=arms[index],
                x_min=skin_component_roi[0],
                y_min=skin_component_roi[1],
                x_max=skin_component_roi[2],
                y_max=skin_component_roi[3],
                arm_dilation=args.skin_component_arm_dilation,
                minimum_area=args.skin_component_min_area,
                maximum_area=args.skin_component_max_area,
                minimum_arm_overlap=args.skin_component_min_arm_overlap,
            )
        skin_candidate = (
            fill_selected_component_hulls(cv2, np, skin_component)
            if args.skin_component_hulls
            else skin_component
        )
        core_override = np.zeros((height, width), dtype=bool)
        if args.selected_skin_core_override:
            hue, saturation, value = cv2.split(
                cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            )
            green_core = (
                (hue >= 28)
                & (hue <= 91)
                & (saturation >= 67)
                & (value >= 28)
            )
            yellow_core = (
                (hue >= 12)
                & (hue <= 35)
                & (saturation >= 105)
                & (value >= 70)
            )
            strict_flower = np.logical_and(
                np.logical_or(green_core, yellow_core), edit_safety
            )
            strict_flower = cv2.dilate(
                strict_flower.astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            ) > 0
            semantic_robot = np.logical_or(body[index], wrist[index])
            semantic_robot = np.logical_or(semantic_robot, arms[index])
            pixels = frame.astype(np.float32)
            chroma = np.max(pixels, axis=2) - np.min(pixels, axis=2)
            coloured_hull = np.logical_and(skin_candidate, chroma >= 18.0)
            core_negative = np.logical_or(skin_component, coloured_hull)
            core_override = np.logical_and(core_negative, semantic_robot)
            core_override = np.logical_and(
                core_override, np.logical_not(strict_flower.copy())
            )
            flower_buffer = np.logical_and(
                flower_buffer, np.logical_not(core_override.copy())
            )
        skin_override = cv2.dilate(
            skin_component.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        ) > 0
        skin_arm_neighborhood = cv2.dilate(
            arms[index].astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (skin_cleanup_radius * 2 + 1, skin_cleanup_radius * 2 + 1),
            ),
        ) > 0
        skin_override = np.logical_and(skin_override, skin_safety)
        skin_override = np.logical_and(skin_override, skin_arm_neighborhood)
        skin_override = np.logical_and(
            skin_override, np.logical_not(flower_buffer.copy())
        )
        skin_override = np.logical_or(skin_override, core_override)
        stable_protected[index] = np.logical_and(
            stable_protected[index], np.logical_not(skin_override.copy())
        )
        stable_protected[index] = np.logical_or(
            stable_protected[index], flower_buffer
        )
        alpha = build_conservative_arm_shadow_alpha(
            cv2,
            np,
            generated=frame,
            clean_plate=clean_plate,
            safety_mask=safety,
            protected_mask=stable_protected[index],
            arm_mask=arms[index],
            protect_radius=args.protect_radius,
            cleanup_radius=args.cleanup_radius,
            maximum_strength=args.maximum_strength,
            neutral_chroma_limit=args.neutral_chroma_limit,
            difference_threshold=args.difference_threshold,
            feather_sigma=args.feather_sigma,
            skin_strength=args.skin_strength,
            skin_to_core_mask=skin_to_core,
            skin_safety_mask=skin_safety,
            skin_cleanup_radius=skin_cleanup_radius,
            skin_candidate_mask=skin_candidate,
            skin_arm_neighborhood_mask=skin_arm_neighborhood,
        )
        alpha_seed[index] = np.rint(alpha * 255.0).astype(np.uint8)
        if residual_skin_seed is not None:
            residual_search_alpha = build_tracked_polygon_alpha(
                cv2,
                np,
                shape=(height, width),
                tracks=residual_arm_tracks,
                frame_index=index,
                feather_sigma=0.0,
            )
            residual_skin_seed[index] = (
                build_residual_arm_skin_support(
                    cv2,
                    np,
                    frame=frame,
                    search_alpha=residual_search_alpha,
                    close_width=args.residual_arm_skin_close_width,
                    close_height=args.residual_arm_skin_close_height,
                    minimum_area=args.residual_arm_skin_min_area,
                    dilation=args.residual_arm_skin_dilation,
                ).astype(np.uint8)
                * 255
            )
    capture.release()

    output = output_dir / "final" / args.output_name
    writer = _writer(paths["ffmpeg"], output, width, height, fps)
    capture = cv2.VideoCapture(str(paths["generated_video"]))
    review_indices = set(int(item) for item in np.linspace(0, 659, 28, dtype=np.int32))
    for track in residual_arm_tracks:
        for keyframe in track["keyframes"]:
            keyframe_index = int(keyframe["frame"])
            review_indices.update(
                range(max(0, keyframe_index - 2), min(660, keyframe_index + 3))
            )
    review_rows: dict[int, list[Any]] = {}
    metrics = []
    temporal_weights = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32)
    if args.temporal_radius != 2:
        temporal_weights = np.ones(args.temporal_radius * 2 + 1, dtype=np.float32)
    temporal_weights /= temporal_weights.sum()
    for index in range(660):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"generated decode ended on frame {index}")
        start = max(0, index - args.temporal_radius)
        end = min(660, index + args.temporal_radius + 1)
        weight_start = start - (index - args.temporal_radius)
        weight_end = weight_start + (end - start)
        weights = temporal_weights[weight_start:weight_end]
        weights = weights / weights.sum()
        temporal_stack = alpha_seed[start:end].astype(np.float32) / 255.0
        motion_aligned = False
        maximum_motion_shift = 0.0
        if args.motion_compensated_temporal:
            aligned_stack = []
            for local_index, neighbor_index in enumerate(range(start, end)):
                aligned, applied, shift = _align_alpha_to_motion(
                    cv2,
                    np,
                    temporal_stack[local_index],
                    source_centroid=arm_centroids[neighbor_index],
                    target_centroid=arm_centroids[index],
                    threshold_pixels=args.motion_threshold_pixels,
                    maximum_shift_pixels=args.motion_maximum_shift_pixels,
                    blend_ramp_pixels=args.motion_blend_ramp_pixels,
                    protect_threshold=args.plate_alpha_protect_threshold,
                )
                aligned_stack.append(aligned)
                motion_aligned = motion_aligned or applied
                maximum_motion_shift = max(maximum_motion_shift, shift)
            temporal_stack = np.stack(aligned_stack)
        alpha = np.tensordot(
            weights,
            temporal_stack,
            axes=(0, 0),
        )
        if args.high_alpha_temporal_maximum:
            temporal_peak = np.max(temporal_stack, axis=0)
            high_alpha = temporal_peak >= 0.70
            alpha[high_alpha] = temporal_peak[high_alpha]
        plate_alpha = _apply_plate_alpha_gain(
            np,
            alpha,
            gain=float(args.plate_alpha_gain),
            cap=float(args.plate_alpha_cap),
            protect_threshold=float(args.plate_alpha_protect_threshold),
        )
        plate_alpha[stable_protected[index]] = 0.0
        plate_alpha[np.logical_not(edit_safety)] = 0.0
        wide_person_alpha = np.zeros((height, width), dtype=np.float32)
        wide_person_colour_protect = np.zeros((height, width), dtype=bool)
        if wide_person_support is not None:
            hue, saturation, value = cv2.split(
                cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            )
            wide_person_colour_protect = (
                (hue >= 28)
                & (hue <= 91)
                & (saturation >= 55)
                & (value >= 28)
            )
            if args.wide_person_green_protect_radius:
                colour_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        args.wide_person_green_protect_radius * 2 + 1,
                        args.wide_person_green_protect_radius * 2 + 1,
                    ),
                )
                wide_person_colour_protect = cv2.dilate(
                    wide_person_colour_protect.astype(np.uint8) * 255,
                    colour_kernel,
                ) > 0
            frame_wide_protected = np.logical_or(
                wide_person_protected[index], wide_person_colour_protect
            )
            wide_person_alpha = _build_wide_person_coverage_alpha(
                cv2,
                np,
                person_mask=wide_person_support[index],
                protected_mask=frame_wide_protected,
                edit_safety=edit_safety,
                coverage_region=wide_person_region,
                dilation=args.wide_person_dilation,
                feather_sigma=args.wide_person_feather_sigma,
                strength=args.wide_person_strength,
            )
        composite_plate_alpha = (
            np.maximum(plate_alpha, wide_person_alpha)
            if args.wide_person_mode == "plate"
            else plate_alpha
        )
        effective_protected = stable_protected[index].copy()
        residual_alpha = build_tracked_polygon_alpha(
            cv2,
            np,
            shape=(height, width),
            tracks=residual_arm_tracks,
            frame_index=index,
            feather_sigma=args.residual_arm_feather_sigma,
        )
        residual_alpha *= float(args.residual_arm_strength)
        if residual_skin_seed is not None:
            skin_start = max(0, index - args.residual_arm_skin_temporal_radius)
            skin_end = min(660, index + args.residual_arm_skin_temporal_radius + 1)
            skin_core = np.max(residual_skin_seed[skin_start:skin_end], axis=0) > 0
            skin_alpha = skin_core.astype(np.float32)
            if args.residual_arm_skin_feather_sigma:
                skin_alpha = cv2.GaussianBlur(
                    skin_alpha,
                    (0, 0),
                    args.residual_arm_skin_feather_sigma,
                )
                skin_alpha[skin_core] = 1.0
            residual_alpha *= np.clip(skin_alpha, 0.0, 1.0)
        residual_active = residual_alpha >= 0.02
        if np.any(residual_active):
            hue, saturation, value = cv2.split(
                cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            )
            green_core = (
                (hue >= 28)
                & (hue <= 91)
                & (saturation >= 67)
                & (value >= 28)
            )
            pink_core = (
                (hue >= 145)
                & (hue <= 179)
                & (saturation >= 115)
                & (value >= 55)
            )
            green_core = np.logical_or(green_core, pink_core)
            if args.residual_arm_green_protect_radius:
                green_core = cv2.dilate(
                    green_core.astype(np.uint8) * 255,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (
                            args.residual_arm_green_protect_radius * 2 + 1,
                            args.residual_arm_green_protect_radius * 2 + 1,
                        ),
                    ),
                ) > 0
            if args.residual_arm_green_min_area:
                count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    green_core.astype(np.uint8), connectivity=8
                )
                coherent_green = np.zeros((height, width), dtype=bool)
                for component in range(1, count):
                    if (
                        int(stats[component, cv2.CC_STAT_AREA])
                        >= args.residual_arm_green_min_area
                    ):
                        coherent_green = np.logical_or(
                            coherent_green, labels == component
                        )
                green_core = coherent_green
            residual_preserve = green_core
            effective_protected[residual_active] = False
            effective_protected[residual_preserve] = True
            residual_alpha[residual_preserve] = 0.0
        residual_alpha[np.logical_not(metric_edit_safety)] = 0.0
        plate_result = apply_conservative_arm_shadow_cleanup(
            np,
            generated=frame,
            clean_plate=clean_plate,
            alpha=composite_plate_alpha,
            protected_mask=stable_protected[index],
        )
        wide_result = plate_result
        if args.wide_person_mode == "graphite" and np.any(wide_person_alpha >= 0.02):
            wide_material = _build_wide_person_graphite_material(cv2, np, frame)
            wide_result = np.rint(
                plate_result.astype(np.float32)
                * (1.0 - wide_person_alpha[..., None])
                + wide_material.astype(np.float32)
                * wide_person_alpha[..., None]
            ).astype(np.uint8)
        if np.any(residual_active):
            if args.residual_arm_mode == "plate":
                alpha = np.maximum(composite_plate_alpha, residual_alpha)
                alpha[effective_protected] = 0.0
                result = apply_conservative_arm_shadow_cleanup(
                    np,
                    generated=frame,
                    clean_plate=clean_plate,
                    alpha=alpha,
                    protected_mask=effective_protected,
                )
                if args.wide_person_mode == "graphite":
                    wide_material = _build_wide_person_graphite_material(
                        cv2, np, frame
                    )
                    result = np.rint(
                        result.astype(np.float32)
                        * (1.0 - wide_person_alpha[..., None])
                        + wide_material.astype(np.float32)
                        * wide_person_alpha[..., None]
                    ).astype(np.uint8)
                    alpha = np.maximum(alpha, wide_person_alpha)
            else:
                material = build_tracked_robot_arm_material(
                    cv2,
                    np,
                    frame=frame,
                    tracks=residual_arm_tracks,
                    frame_index=index,
                    style=args.residual_arm_mode,
                )
                result = np.rint(
                    wide_result.astype(np.float32)
                    * (1.0 - residual_alpha[..., None])
                    + material.astype(np.float32) * residual_alpha[..., None]
                ).astype(np.uint8)
                result[residual_preserve] = frame[residual_preserve]
                alpha = np.maximum(composite_plate_alpha, wide_person_alpha)
                alpha = np.maximum(alpha, residual_alpha)
        else:
            alpha = np.maximum(composite_plate_alpha, wide_person_alpha)
            result = wide_result
        assert writer.stdin is not None
        writer.stdin.write(result.tobytes())
        active = alpha >= 0.02
        plate_active = composite_plate_alpha >= 0.02
        before_mae = (
            float(np.mean(np.abs(frame[plate_active].astype(np.float32) - clean_plate[plate_active].astype(np.float32))))
            if np.any(plate_active) else 0.0
        )
        after_mae = (
            float(np.mean(np.abs(plate_result[plate_active].astype(np.float32) - clean_plate[plate_active].astype(np.float32))))
            if np.any(plate_active) else 0.0
        )
        wide_active = wide_person_alpha >= 0.02
        wide_chroma_before = (
            float(
                np.mean(
                    np.ptp(frame[wide_active].astype(np.float32), axis=1)
                )
            )
            if np.any(wide_active)
            else 0.0
        )
        wide_chroma_after = (
            float(
                np.mean(
                    np.ptp(result[wide_active].astype(np.float32), axis=1)
                )
            )
            if np.any(wide_active)
            else 0.0
        )
        metrics.append(
            {
                "frame": index,
                "modified_fraction": float(np.mean(active)),
                "mean_alpha": float(np.mean(alpha)),
                "maximum_alpha": float(np.max(alpha)),
                "plate_mean_alpha": float(np.mean(composite_plate_alpha)),
                "wide_person_active_fraction": float(
                    np.mean(wide_active)
                ),
                "wide_person_chroma_before": wide_chroma_before,
                "wide_person_chroma_after": wide_chroma_after,
                "wide_person_colour_protected_exact": bool(
                    np.array_equal(
                        wide_result[wide_person_colour_protect],
                        plate_result[wide_person_colour_protect],
                    )
                ),
                "motion_aligned": motion_aligned,
                "maximum_motion_shift_pixels": maximum_motion_shift,
                "plate_mae_before": before_mae,
                "plate_mae_after": after_mae,
                "protected_exact": bool(np.array_equal(result[effective_protected], frame[effective_protected])),
                "outside_safety_exact": bool(np.array_equal(result[np.logical_not(metric_edit_safety)], frame[np.logical_not(metric_edit_safety)])),
                "residual_arm_active_fraction": float(np.mean(residual_active)),
                "skin_edited_fraction": float(
                    np.mean(
                        np.logical_and(
                            _skin_like(cv2, np, frame), alpha >= 0.10
                        )
                    )
                ),
            }
        )
        if index in review_indices:
            overlay = frame.copy()
            overlay[active] = np.rint(
                0.45 * overlay[active] + 0.55 * np.asarray([255, 80, 40])
            ).astype(np.uint8)
            review_rows[index] = [
                _annotate(cv2, frame, f"raw H3 {index}"),
                _annotate(cv2, result, "small shadow reduction"),
                _annotate(cv2, overlay, "edited band in blue"),
            ]
    capture.release()
    assert writer.stdin is not None
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("ffmpeg failed during output encoding")

    review = output_dir / "review/dense-raw-vs-reduced.jpg"
    cv2.imwrite(
        str(review),
        _sheet(cv2, [review_rows[index] for index in sorted(review_rows)]),
        [cv2.IMWRITE_JPEG_QUALITY, 93],
    )
    metrics_path = output_dir / "logs/frame-metrics.jsonl"
    metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics))
    valid_before = [row["plate_mae_before"] for row in metrics if row["plate_mae_before"] > 0]
    valid_after = [row["plate_mae_after"] for row in metrics if row["plate_mae_before"] > 0]
    aggregate = {
        "frames": len(metrics),
        "modified_fraction_mean": float(np.mean([row["modified_fraction"] for row in metrics])),
        "modified_fraction_max": float(np.max([row["modified_fraction"] for row in metrics])),
        "mean_alpha": float(np.mean([row["mean_alpha"] for row in metrics])),
        "maximum_alpha": float(np.max([row["maximum_alpha"] for row in metrics])),
        "plate_mean_alpha": float(
            np.mean([row["plate_mean_alpha"] for row in metrics])
        ),
        "wide_person_active_fraction_mean": float(
            np.mean([row["wide_person_active_fraction"] for row in metrics])
        ),
        "wide_person_active_fraction_max": float(
            np.max([row["wide_person_active_fraction"] for row in metrics])
        ),
        "wide_person_chroma_before_mean": float(
            np.mean([row["wide_person_chroma_before"] for row in metrics])
        ),
        "wide_person_chroma_after_mean": float(
            np.mean([row["wide_person_chroma_after"] for row in metrics])
        ),
        "wide_person_colour_protected_exact_all_frames": all(
            row["wide_person_colour_protected_exact"] for row in metrics
        ),
        "motion_aligned_frames": int(
            sum(bool(row["motion_aligned"]) for row in metrics)
        ),
        "maximum_motion_shift_pixels": float(
            np.max([row["maximum_motion_shift_pixels"] for row in metrics])
        ),
        "skin_edited_fraction_mean": float(
            np.mean([row["skin_edited_fraction"] for row in metrics])
        ),
        "residual_arm_active_fraction_mean": float(
            np.mean([row["residual_arm_active_fraction"] for row in metrics])
        ),
        "residual_arm_active_fraction_max": float(
            np.max([row["residual_arm_active_fraction"] for row in metrics])
        ),
        "plate_mae_before_mean": float(np.mean(valid_before)),
        "plate_mae_after_mean": float(np.mean(valid_after)),
        "plate_mae_reduction_fraction": float(1.0 - np.mean(valid_after) / np.mean(valid_before)),
        "protected_exact_all_frames": all(row["protected_exact"] for row in metrics),
        "outside_safety_exact_all_frames": all(row["outside_safety_exact"] for row in metrics),
    }
    raw_temporal = _temporal_metrics(
        cv2, np, paths["generated_video"], metric_edit_safety
    )
    output_temporal = _temporal_metrics(cv2, np, output, metric_edit_safety)
    gates = {
        "full_decode": output_temporal["frames"] == 660,
        "protected_content_exact_preencode": aggregate["protected_exact_all_frames"],
        "outside_safety_exact_preencode": aggregate["outside_safety_exact_all_frames"],
        "edit_scope_small": aggregate["modified_fraction_mean"] <= args.maximum_edit_fraction,
        "shadow_reduced": aggregate["plate_mae_reduction_fraction"] >= 0.10,
        "wide_person_chroma_reduced": (
            args.wide_person_strength == 0
            or aggregate["wide_person_chroma_after_mean"]
            <= aggregate["wide_person_chroma_before_mean"]
            * args.wide_person_maximum_chroma_ratio
        ),
        "wide_person_colour_protected": aggregate[
            "wide_person_colour_protected_exact_all_frames"
        ],
        "full_frame_jerk_bounded": output_temporal["jerk_full"]["mean"] <= raw_temporal["jerk_full"]["mean"] * 1.12,
        "roi_jerk_bounded": output_temporal["jerk_roi"]["mean"] <= raw_temporal["jerk_roi"]["mean"] * 1.18,
        "human_review": args.human_review == "passed",
    }
    automatic_passed = all(value for key, value in gates.items() if key != "human_review")
    final_status = "WORKING" if automatic_passed and gates["human_review"] else "PARTIAL"

    comparison = output_dir / "final/human-vs-h3-arm-shadow-reduced-vertical.mp4"
    subprocess.run(
        [
            str(paths["ffmpeg"]), "-y", "-v", "error", "-i", str(paths["source_video"]),
            "-i", str(output), "-filter_complex",
            "[0:v]scale=672:378:force_original_aspect_ratio=decrease:flags=lanczos,pad=672:384:(ow-iw)/2:(oh-ih)/2:black[v0];"
            "[1:v]scale=672:378:force_original_aspect_ratio=decrease:flags=lanczos,pad=672:384:(ow-iw)/2:(oh-ih)/2:black[v1];"
            "[v0][v1]vstack=inputs=2[out]",
            "-map", "[out]", "-frames:v", "660", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "15", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(comparison),
        ],
        check=True,
    )
    comparison_info = _video_info(cv2, comparison, decode=True)
    gates["comparison_full_decode"] = int(comparison_info.get("decoded_frames", 0)) == 660
    if not gates["comparison_full_decode"]:
        final_status = "PARTIAL"

    manifest.update(
        {
            "status": final_status,
            "honest_status": f"{final_status}: conservative cleanup gates " + ("and review passed." if final_status == "WORKING" else "reported; review pending or a gate failed."),
            "config": config,
            "video": generated_info,
            "metrics": aggregate,
            "temporal": {"raw_h3": raw_temporal, "output": output_temporal},
            "acceptance_gates": gates,
            "review_notes": args.review_notes,
            "flower_mask_outlier_frames": flower_outlier_frames,
            "outputs": {
                "video": {"path": str(output), "sha256": file_sha256(output)},
                "comparison": {"path": str(comparison), "sha256": file_sha256(comparison)},
                "review": {"path": str(review), "sha256": file_sha256(review)},
                "frame_metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
            },
            "limitations": [
                "This attenuates a bounded neutral halo and, when supplied, applies reviewed camera-frame residual-arm negatives plus a protected semantic source-person coverage matte; it does not regenerate the H3 robot.",
                "The output remains generated 2D imagery, not physical robot execution.",
            ],
        }
    )
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "logs/run.log").write_text(json.dumps({"status": final_status, "metrics": aggregate, "temporal": manifest["temporal"], "gates": gates}, indent=2, sort_keys=True) + "\n")

    if args.showcase_output is not None:
        if final_status != "WORKING":
            raise RuntimeError("showcase publication requires WORKING status")
        showcase = args.showcase_output.expanduser().resolve()
        if showcase.exists():
            raise FileExistsError(f"showcase output exists: {showcase}")
        showcase.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(comparison, showcase)
        _write_json(
            showcase.with_suffix(".json"),
            {
                "schema_version": "1.0.0",
                "experiment": str(output_dir),
                "video_sha256": file_sha256(showcase),
                "frames": 660,
                "fps": 24,
                "layout": ["top: real human input", "bottom: H3+EPL with conservative arm-shadow attenuation"],
            },
        )
        manifest["showcase"] = {"path": str(showcase), "sha256": file_sha256(showcase)}
        _write_json(output_dir / "manifest.json", manifest)

    print(json.dumps({"output_dir": str(output_dir), "status": final_status, "metrics": aggregate, "temporal": manifest["temporal"], "gates": gates}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
