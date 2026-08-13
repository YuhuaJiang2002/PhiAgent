#!/usr/bin/env python3
"""Evaluate sharpness and action separation for a recovered AC-WM Ego demo.

Human-removal and blur acceptance remain explicit dense-review gates. Color and
edge metrics are diagnostics and never auto-promote a video with a visible hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    info: dict[str, float | int] = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (416, 240), interpolation=cv2.INTER_AREA))
    capture.release()
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    info["frames"] = len(frames)
    info["sha256"] = _sha256(path)
    return frames, info


def _percentiles(np: Any, values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "minimum": float(np.min(values)),
    }


def _measure(cv2: Any, np: Any, frames: list[Any], masks: list[Any] | None) -> dict[str, Any]:
    full_sharpness = []
    foreground_sharpness = []
    background_sharpness = []
    skin_fractions = []
    transitions = []
    previous = None
    for index, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        full_sharpness.append(float(np.mean(edges)))
        if masks is not None:
            mask = masks[min(index, len(masks) - 1)]
            foreground_sharpness.append(float(np.mean(edges[mask])))
            background_sharpness.append(float(np.mean(edges[~mask])))
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            _, cr, cb = cv2.split(ycrcb)
            skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
            skin_fractions.append(float(np.mean(skin[mask])))
        if previous is not None:
            transitions.append(float(np.mean(cv2.absdiff(gray, previous))))
        previous = gray
    result: dict[str, Any] = {
        "full_frame_absolute_laplacian": _percentiles(np, full_sharpness),
        "transition_energy": {
            "median": float(np.median(transitions)),
            "maximum": float(np.max(transitions)),
            "maximum_to_median": float(np.max(transitions) / max(np.median(transitions), 1e-6)),
        },
    }
    if masks is not None:
        result.update(
            {
                "foreground_absolute_laplacian": _percentiles(np, foreground_sharpness),
                "background_absolute_laplacian": _percentiles(np, background_sharpness),
                "skin_tone_fraction_in_generated_support": _percentiles(np, skin_fractions),
                "skin_metric_warning": "Color heuristic only; it cannot pass human-residual review.",
            }
        )
    return result


def _labeled_paths(values: list[str], option: str) -> dict[str, Path]:
    result = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or label in result:
            raise ValueError(f"{option} values must be unique LABEL=PATH pairs")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing {option} input: {path}")
        result[label] = path
    if len(result) != 3:
        raise ValueError(f"{option} must be supplied exactly three times")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cabbage-reference", type=Path, required=True)
    parser.add_argument("--old-action", action="append", default=[])
    parser.add_argument("--new-action", action="append", default=[])
    parser.add_argument("--new-mask", action="append", default=[])
    parser.add_argument("--human-review", choices=("pending", "passed", "failed"), default="pending")
    parser.add_argument("--blur-review", choices=("pending", "passed", "failed"), default="pending")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    import numpy as np

    source_path = args.source.expanduser().resolve()
    cabbage_path = args.cabbage_reference.expanduser().resolve()
    old_paths = _labeled_paths(args.old_action, "--old-action")
    new_paths = _labeled_paths(args.new_action, "--new-action")
    mask_paths = _labeled_paths(args.new_mask, "--new-mask")
    if set(old_paths) != set(new_paths) or set(new_paths) != set(mask_paths):
        raise ValueError("old, new, and mask action labels must match")
    source, source_info = _decode(cv2, source_path)
    cabbage, cabbage_info = _decode(cv2, cabbage_path)
    cabbage_metrics = _measure(cv2, np, cabbage, None)
    source_metrics = _measure(cv2, np, source, None)
    actions: dict[str, Any] = {}
    new_frames: dict[str, list[Any]] = {}
    exact_geometry = True
    sharpness_passes = []
    for label in sorted(new_paths):
        old, old_info = _decode(cv2, old_paths[label])
        new, new_info = _decode(cv2, new_paths[label])
        raw_masks, mask_info = _decode(cv2, mask_paths[label])
        masks = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) >= 127 for frame in raw_masks]
        source_action_metrics = _measure(cv2, np, source, masks)
        old_metrics = _measure(cv2, np, old, masks)
        new_metrics = _measure(cv2, np, new, masks)
        foreground_ratio = (
            new_metrics["foreground_absolute_laplacian"]["p10"]
            / max(old_metrics["foreground_absolute_laplacian"]["p10"], 1e-6)
        )
        background_ratio = (
            new_metrics["background_absolute_laplacian"]["p10"]
            / max(source_action_metrics["background_absolute_laplacian"]["p10"], 1e-6)
        )
        reference_ratio = (
            new_metrics["full_frame_absolute_laplacian"]["mean"]
            / max(cabbage_metrics["full_frame_absolute_laplacian"]["mean"], 1e-6)
        )
        geometry_passed = (
            new_info["frames"] == 240 and abs(float(new_info["fps"]) - 24.0) <= 1e-6
        )
        exact_geometry &= geometry_passed
        sharpness_passes.append(foreground_ratio >= 1.10 and background_ratio >= 0.70)
        actions[label] = {
            "old": {"path": str(old_paths[label]), "video": old_info, "metrics": old_metrics},
            "new": {"path": str(new_paths[label]), "video": new_info, "metrics": new_metrics},
            "mask": {"path": str(mask_paths[label]), "video": mask_info},
            "source_on_action_mask": source_action_metrics,
            "comparison": {
                "new_to_old_foreground_p10_sharpness_ratio": foreground_ratio,
                "new_background_to_source_p10_sharpness_ratio": background_ratio,
                "new_to_cabbage_full_mean_sharpness_ratio_diagnostic_only": reference_ratio,
                "geometry_passed": geometry_passed,
            },
        }
        new_frames[label] = new

    distinctness = {}
    for left, right in combinations(sorted(new_frames), 2):
        values = [
            float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
            for a, b in zip(new_frames[left], new_frames[right])
        ]
        distinctness[f"{left}__{right}"] = {
            "mean_full_frame_mad": float(np.mean(values)),
            "fraction_frames_above_2_mad": float(np.mean(np.asarray(values) >= 2.0)),
        }
    distinctness_passed = all(
        item["mean_full_frame_mad"] >= 2.0 for item in distinctness.values()
    )
    gates = {
        "exact_240_frames_24_fps": exact_geometry,
        "sharpness_non_regression": all(sharpness_passes),
        "action_distinctness": distinctness_passed,
        "dense_human_residual_review": args.human_review,
        "dense_blur_review": args.blur_review,
    }
    accepted = (
        all(value is True for value in list(gates.values())[:3])
        and args.human_review == "passed"
        and args.blur_review == "passed"
    )
    payload = {
        "schema_version": "1.0.0",
        "status": "WORKING" if accepted else "PARTIAL",
        "source": {"path": str(source_path), "video": source_info, "metrics": source_metrics},
        "cabbage_reference": {
            "path": str(cabbage_path),
            "video": cabbage_info,
            "metrics": cabbage_metrics,
            "use": "visual-quality reference only; different scene and duration",
        },
        "actions": actions,
        "action_distinctness": distinctness,
        "gates": gates,
        "accepted": accepted,
    }
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
