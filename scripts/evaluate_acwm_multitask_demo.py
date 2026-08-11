#!/usr/bin/env python3
"""Evaluate same-scene, instruction-conditioned AC-WM task videos."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_acwm_visual_recovery import (  # noqa: E402
    _decode,
    _labeled_paths,
    _measure,
    _sha256,
)


def _metadata_paths(values: list[str]) -> dict[str, Path]:
    return _labeled_paths(values, "--generation-metadata")


def _align_support(cv2: Any, masks: list[Any], frames: list[Any]) -> list[Any]:
    """Align binary camera-frame support without softening its boundary."""
    if not frames:
        raise ValueError("cannot align support to an empty video")
    height, width = frames[0].shape[:2]
    aligned = []
    for mask in masks:
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(
                mask.astype("uint8"),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        aligned.append(mask >= 1)
    return aligned


def _merge_manifest_items(
    payloads: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for item in payload[key]:
            label = str(item["label"])
            if label in merged:
                raise ValueError(f"duplicate {key} label across manifests: {label}")
            merged[label] = item
    return merged


def _measure_handover_transfer(
    cv2: Any,
    np: Any,
    frames: list[Any],
    *,
    final_x_max: float,
    final_p90_x_max: float,
    leftward_shift_min: float,
    valid_fraction_min: float,
) -> dict[str, Any]:
    """Track the blue bottle and fail closed unless it finishes screen-left.

    The Ego camera makes anatomical left/right ambiguous, so the measurable
    contract is stated in image coordinates.  The action driver begins with the
    bottle near the centre/right, establishes dual contact, and ends with the
    bottle in the screen-left gripper.  Restricting the HSV support to the
    manipulation workspace avoids the blue/green clutter at the image border.
    A separate dense human review is still required to verify finger contact,
    release, and the empty screen-right gripper.
    """
    if not frames:
        raise ValueError("cannot evaluate handover on an empty video")
    normalized_x: list[float | None] = []
    component_areas: list[int] = []
    for frame in frames:
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        support = cv2.inRange(hsv, (85, 100, 35), (130, 255, 255))
        support[: int(0.25 * height), :] = 0
        support[int(0.98 * height) :, :] = 0
        support[:, : int(0.12 * width)] = 0
        support[:, int(0.88 * width) :] = 0
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            support
        )
        minimum_area = max(100, int(round(0.0002 * width * height)))
        candidates = [
            index
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area
        ]
        if not candidates:
            normalized_x.append(None)
            component_areas.append(0)
            continue
        selected = max(
            candidates,
            key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        )
        normalized_x.append(float(centroids[selected, 0] / width))
        component_areas.append(int(stats[selected, cv2.CC_STAT_AREA]))

    window = min(48, max(1, len(frames) // 5))
    first = np.asarray(
        [value for value in normalized_x[:window] if value is not None],
        dtype=np.float64,
    )
    final = np.asarray(
        [value for value in normalized_x[-window:] if value is not None],
        dtype=np.float64,
    )
    valid_fraction = float(
        np.mean([value is not None for value in normalized_x])
    )
    if first.size == 0 or final.size == 0:
        return {
            "passed": False,
            "valid_fraction": valid_fraction,
            "failure": "blue bottle was not trackable in the first/final window",
        }
    first_median = float(np.median(first))
    final_median = float(np.median(final))
    final_p90 = float(np.percentile(final, 90))
    leftward_shift = first_median - final_median
    checks = {
        "track_valid": valid_fraction >= valid_fraction_min,
        "final_median_is_screen_left": final_median <= final_x_max,
        "final_p90_stays_screen_left": final_p90 <= final_p90_x_max,
        "net_leftward_transfer": leftward_shift >= leftward_shift_min,
    }
    positive_areas = [area for area in component_areas if area > 0]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "valid_fraction": valid_fraction,
        "first_window_median_x": first_median,
        "final_window_median_x": final_median,
        "final_window_p90_x": final_p90,
        "leftward_shift": leftward_shift,
        "component_area_p10": (
            float(np.percentile(positive_areas, 10))
            if positive_areas else 0.0
        ),
        "coordinate_frame": "camera:normalized_x (0=screen-left, 1=screen-right)",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cabbage-reference", type=Path, required=True)
    parser.add_argument("--quality-baseline", action="append", default=[])
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--candidate-mask", action="append", default=[])
    parser.add_argument("--generation-metadata", action="append", default=[])
    parser.add_argument("--action-manifest", type=Path, required=True)
    parser.add_argument("--control-manifest", action="append", type=Path, required=True)
    parser.add_argument("--driver-manifest", action="append", type=Path, required=True)
    parser.add_argument("--human-review", choices=("pending", "passed", "failed"), default="pending")
    parser.add_argument("--blur-review", choices=("pending", "passed", "failed"), default="pending")
    parser.add_argument("--task-review", choices=("pending", "passed", "failed"), default="pending")
    parser.add_argument("--handover-review", choices=("pending", "passed", "failed"), default="pending")
    parser.add_argument("--foreground-ratio-min", type=float, default=0.75)
    parser.add_argument("--background-ratio-min", type=float, default=0.70)
    parser.add_argument("--handover-final-x-max", type=float, default=0.47)
    parser.add_argument("--handover-final-p90-x-max", type=float, default=0.52)
    parser.add_argument("--handover-leftward-shift-min", type=float, default=0.04)
    parser.add_argument("--handover-track-valid-min", type=float, default=0.95)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    import numpy as np

    source_path = args.source.expanduser().resolve()
    cabbage_path = args.cabbage_reference.expanduser().resolve()
    action_manifest_path = args.action_manifest.expanduser().resolve()
    control_paths = [path.expanduser().resolve() for path in args.control_manifest]
    driver_paths = [path.expanduser().resolve() for path in args.driver_manifest]
    baselines = _labeled_paths(args.quality_baseline, "--quality-baseline")
    candidates = _labeled_paths(args.candidate, "--candidate")
    masks = _labeled_paths(args.candidate_mask, "--candidate-mask")
    metadata_paths = _metadata_paths(args.generation_metadata)
    labels = set(candidates)
    if labels != set(baselines) or labels != set(masks) or labels != set(metadata_paths):
        raise ValueError("baseline, candidate, mask, and metadata labels must match")
    for path in (
        source_path,
        cabbage_path,
        action_manifest_path,
        *control_paths,
        *driver_paths,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required evaluation input is missing: {path}")

    action_manifest = json.loads(action_manifest_path.read_text())
    requested_labels = {str(item["label"]) for item in action_manifest["actions"]}
    controls = [json.loads(path.read_text()) for path in control_paths]
    drivers = [json.loads(path.read_text()) for path in driver_paths]
    control_variants = _merge_manifest_items(controls, "variants")
    driver_variants = _merge_manifest_items(drivers, "actions")
    if labels != requested_labels:
        raise ValueError("action manifest must exactly match candidate labels")
    if not labels.issubset(control_variants) or not labels.issubset(driver_variants):
        raise ValueError("control and raw-driver manifests must cover candidate labels")

    source_frames, source_info = _decode(cv2, source_path)
    cabbage_frames, cabbage_info = _decode(cv2, cabbage_path)
    source_metrics = _measure(cv2, np, source_frames, None)
    cabbage_metrics = _measure(cv2, np, cabbage_frames, None)
    action_records: dict[str, Any] = {}
    candidate_frames: dict[str, list[Any]] = {}
    geometry_passes = []
    quality_passes = []
    lineage_passes = []
    for label in sorted(labels):
        baseline_frames, baseline_info = _decode(cv2, baselines[label])
        generated_frames, generated_info = _decode(cv2, candidates[label])
        candidate_sha256 = _sha256(candidates[label])
        raw_mask_frames, mask_info = _decode(cv2, masks[label])
        mask_support = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) >= 127
            for frame in raw_mask_frames
        ]
        baseline_metrics = _measure(
            cv2, np, baseline_frames,
            _align_support(cv2, mask_support, baseline_frames),
        )
        generated_metrics = _measure(
            cv2, np, generated_frames,
            _align_support(cv2, mask_support, generated_frames),
        )
        source_on_support = _measure(
            cv2, np, source_frames,
            _align_support(cv2, mask_support, source_frames),
        )
        foreground_ratio = (
            generated_metrics["foreground_absolute_laplacian"]["p10"]
            / max(baseline_metrics["foreground_absolute_laplacian"]["p10"], 1e-6)
        )
        background_ratio = (
            generated_metrics["background_absolute_laplacian"]["p10"]
            / max(source_on_support["background_absolute_laplacian"]["p10"], 1e-6)
        )
        cabbage_ratio = (
            generated_metrics["full_frame_absolute_laplacian"]["mean"]
            / max(cabbage_metrics["full_frame_absolute_laplacian"]["mean"], 1e-6)
        )
        geometry = generated_info["frames"] == 240 and abs(float(generated_info["fps"]) - 24.0) <= 1e-6
        generation = json.loads(metadata_paths[label].read_text())
        method = str(generation.get("method", ""))
        wan_seed = generation.get("seed")
        wan_lineage = (
            method.startswith("robot_factored_wan22")
            and isinstance(wan_seed, int)
            and generation.get("preflight", {}).get("model_id")
            == "Wan-AI/Wan2.2-Animate-14B"
        )
        if label == "handover-bottle" and wan_lineage:
            generation_prompt = " ".join(
                str(item) for item in generation.get("command", [])
            ).lower()
            wan_lineage = all(
                phrase in generation_prompt
                for phrase in (
                    "screen-left robot gripper",
                    "screen-right gripper is empty",
                    "final two seconds",
                )
            )
        h3_hold_lineage = (
            method == "h3_nf4_terminal_state_hold_no_interpolation_no_blur"
            and generation.get("seed") == 20260811
            and generation.get("postprocessing", {}).get("cross_dissolve") is False
            and generation.get("postprocessing", {}).get("source_person_restore") is False
        )
        h3_continuous_lineage = (
            method == "task_state_valid_continuation_crop_with_nearest_frame_retiming"
            and generation.get("status") == "succeeded"
            and generation.get("seed") == 20260811
            and generation.get("human_review") == "passed"
            and generation.get("postprocessing", {}).get("frame_interpolation") is False
            and generation.get("postprocessing", {}).get("cross_dissolve") is False
            and generation.get("postprocessing", {}).get("source_person_restore") is False
        )
        h3_semantic_handover_lineage = (
            label == "handover-bottle"
            and method
            == "minimax_h3_nf4_two_window_semantic_handover_with_detail_recovery"
            and generation.get("status") == "succeeded"
            and generation.get("seed") == 20260811
            and generation.get("human_review") == "passed"
            and generation.get("blur_review") == "passed"
            and generation.get("semantic_review") == "passed"
            and generation.get("seam", {}).get("frame") == 123
            and generation.get("postprocessing", {}).get("frame_interpolation") is False
            and generation.get("postprocessing", {}).get("cross_dissolve") is False
            and generation.get("postprocessing", {}).get("source_person_restore") is False
        )
        continuous_lineage_match = (
            h3_continuous_lineage
            and generation.get("final_output_sha256")
            == driver_variants[label].get("output_sha256")
        )
        semantic_handover_lineage_match = (
            h3_semantic_handover_lineage
            and generation.get("final_output_sha256")
            == candidate_sha256
            and generation.get("inputs", {}).get("action_driver", {}).get("sha256")
            == driver_variants[label].get("output_sha256")
        )
        lineage = (
            generation.get("status") == "succeeded"
            and (
                continuous_lineage_match
                or semantic_handover_lineage_match
                or (
                    (wan_lineage or h3_hold_lineage)
                    and generation.get("inputs", {}).get("action_driver", {}).get("sha256")
                    == driver_variants[label]["output_sha256"]
                )
            )
            and generation.get("postprocessing", {}).get("blur") is False
            and generation.get("postprocessing", {}).get("alpha_repair") is False
        )
        quality = foreground_ratio >= args.foreground_ratio_min and background_ratio >= args.background_ratio_min
        geometry_passes.append(geometry)
        quality_passes.append(quality)
        lineage_passes.append(lineage)
        candidate_frames[label] = generated_frames
        action_records[label] = {
            "quality_baseline": {"path": str(baselines[label]), "video": baseline_info, "metrics": baseline_metrics},
            "candidate": {
                "path": str(candidates[label]),
                "sha256": candidate_sha256,
                "video": generated_info,
                "metrics": generated_metrics,
            },
            "candidate_mask": {"path": str(masks[label]), "video": mask_info},
            "generation_metadata": {"path": str(metadata_paths[label]), "sha256": _sha256(metadata_paths[label])},
            "source_on_candidate_support": source_on_support,
            "comparison": {
                "candidate_to_accepted_baseline_foreground_p10_ratio": foreground_ratio,
                "candidate_background_to_source_p10_ratio": background_ratio,
                "candidate_to_cabbage_full_mean_ratio_diagnostic_only": cabbage_ratio,
                "geometry_passed": geometry,
                "quality_non_regression_passed": quality,
                "generation_lineage_passed": lineage,
            },
        }

    distinctness: dict[str, Any] = {}
    for left, right in combinations(sorted(candidate_frames), 2):
        values = [
            float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))))
            for first, second in zip(candidate_frames[left], candidate_frames[right])
        ]
        distinctness[f"{left}__{right}"] = {
            "mean_full_frame_mad": float(np.mean(values)),
            "fraction_frames_above_2_mad": float(np.mean(np.asarray(values) >= 2.0)),
        }
    distinctness_passed = all(
        item["mean_full_frame_mad"] >= 2.0
        and item["fraction_frames_above_2_mad"] >= 0.95
        for item in distinctness.values()
    )
    selected_controls = [
        payload
        for payload in controls
        if labels.intersection(str(item["label"]) for item in payload["variants"])
    ]
    control_passed = all(
        payload.get("acceptance", {}).get("trajectory_separation_passed") is True
        and payload.get("acceptance", {}).get(
            "minimum_pairwise_state_rms_pixels", 0,
        ) >= 35.0
        for payload in selected_controls
    )
    handover_transfer: dict[str, Any] | None = None
    if "handover-bottle" in candidate_frames:
        handover_transfer = _measure_handover_transfer(
            cv2,
            np,
            candidate_frames["handover-bottle"],
            final_x_max=args.handover_final_x_max,
            final_p90_x_max=args.handover_final_p90_x_max,
            leftward_shift_min=args.handover_leftward_shift_min,
            valid_fraction_min=args.handover_track_valid_min,
        )
        action_records["handover-bottle"]["handover_transfer"] = handover_transfer
    gates: dict[str, bool | str] = {
        "exact_240_frames_24_fps": all(geometry_passes),
        "quality_non_regression": all(quality_passes),
        "action_distinctness": distinctness_passed,
        "control_trajectory_separation": control_passed,
        "reproducible_generation_lineage": all(lineage_passes),
        "handover_blue_bottle_left_transfer": (
            handover_transfer is None or handover_transfer["passed"] is True
        ),
        "dense_human_residual_review": args.human_review,
        "dense_blur_review": args.blur_review,
        "dense_task_adherence_review": args.task_review,
        "dense_handover_contact_release_review": args.handover_review,
    }
    automatic_gates = (
        "exact_240_frames_24_fps",
        "quality_non_regression",
        "action_distinctness",
        "control_trajectory_separation",
        "reproducible_generation_lineage",
        "handover_blue_bottle_left_transfer",
    )
    accepted = all(gates[key] is True for key in automatic_gates) and all(
        review == "passed"
        for review in (
            args.human_review,
            args.blur_review,
            args.task_review,
            args.handover_review,
        )
    )
    payload = {
        "schema_version": "1.0.0",
        "status": "WORKING" if accepted else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "opencv-python")
        },
        "thresholds": {
            "foreground_ratio_min": args.foreground_ratio_min,
            "background_ratio_min": args.background_ratio_min,
            "handover_final_x_max": args.handover_final_x_max,
            "handover_final_p90_x_max": args.handover_final_p90_x_max,
            "handover_leftward_shift_min": args.handover_leftward_shift_min,
            "handover_track_valid_min": args.handover_track_valid_min,
        },
        "source": {"path": str(source_path), "video": source_info, "metrics": source_metrics},
        "cabbage_reference": {
            "path": str(cabbage_path), "video": cabbage_info,
            "metrics": cabbage_metrics,
            "use": "visual-quality diagnostic only; different scene and duration",
        },
        "action_manifest": {
            "path": str(action_manifest_path),
            "sha256": _sha256(action_manifest_path),
        },
        "control_manifests": [
            {"path": str(path), "sha256": _sha256(path)} for path in control_paths
        ],
        "driver_manifests": [
            {"path": str(path), "sha256": _sha256(path)} for path in driver_paths
        ],
        "actions": action_records,
        "action_distinctness": distinctness,
        "gates": gates,
        "accepted": accepted,
        "claim_boundary": "WORKING means visual instruction control in this recorded Ego scene, not physical-robot execution.",
    }
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
