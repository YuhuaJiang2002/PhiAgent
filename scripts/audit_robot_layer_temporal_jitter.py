#!/usr/bin/env python3
"""Compare source-motion-aligned robot-layer jitter on frozen risk windows."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.temporal_appearance import (  # noqa: E402
    bidirectional_flow_state,
    warp_with_flow,
)
from scripts.compose_joyai_flower_repairs import (  # noqa: E402
    _load_packed,
    _unpack,
)
from scripts.stabilize_joyai_appearance_state import (  # noqa: E402
    _git_state,
    _packages,
    _probe,
    _sha256,
)
from phiagent.rendering.temporal_masks import build_torso_head_whitelist  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--incumbent-video", type=Path, required=True)
    parser.add_argument("--challenger-video", type=Path, required=True)
    parser.add_argument("--robot-masks", type=Path, required=True)
    parser.add_argument("--limb-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--window", nargs=2, type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--evaluation-width", type=int, default=320)
    parser.add_argument("--evaluation-height", type=int, default=180)
    parser.add_argument("--flow-scale", type=float, default=0.5)
    parser.add_argument(
        "--flow-reference",
        choices=("source", "candidate"),
        default="source",
        help=(
            "Use one source-motion field for action-following residuals or "
            "one self-motion field per generated candidate for visual flicker."
        ),
    )
    parser.add_argument("--minimum-confidence", type=float, default=0.2)
    parser.add_argument("--limb-dilation-pixels", type=int, default=5)
    parser.add_argument("--torso-erosion-pixels", type=int, default=3)
    parser.add_argument("--contact-dilation-pixels", type=int, default=7)
    parser.add_argument(
        "--high-jitter-threshold",
        type=float,
        default=20.0,
        help="Frozen layer-warp MAE threshold used to count visible jitter spikes.",
    )
    return parser


def _mask_to_evaluation(cv2: Any, np: Any, mask: Any, width: int, height: int) -> Any:
    if mask.shape != (480, 832):
        raise ValueError("legacy jitter audit requires 832x480 packed masks")
    canvas = np.zeros((480, 854), dtype=np.uint8)
    canvas[:, 11:843] = np.asarray(mask, dtype=np.uint8)
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _summary(np: Any, values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _high_jitter_count(values: list[float], threshold: float) -> int:
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("high-jitter threshold must be finite and positive")
    return sum(value > threshold for value in values)


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    paths = {
        "source": args.source_video.expanduser().resolve(),
        "incumbent": args.incumbent_video.expanduser().resolve(),
        "challenger": args.challenger_video.expanduser().resolve(),
        "robot_masks": args.robot_masks.expanduser().resolve(),
        "limb_masks": args.limb_masks.expanduser().resolve(),
        "flower_masks": args.flower_masks.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{name}: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    windows = sorted((int(start), int(end)) for start, end in args.window)
    if any(not 0 <= start < end < args.expected_frames for start, end in windows):
        raise ValueError("risk windows must be valid inclusive intervals")
    if any(left[1] >= right[0] for left, right in zip(windows, windows[1:])):
        raise ValueError("risk windows must not overlap")
    if not math.isfinite(args.high_jitter_threshold) or args.high_jitter_threshold <= 0:
        raise ValueError("high-jitter threshold must be finite and positive")
    active = {frame for start, end in windows for frame in range(start, end + 1)}

    probes = {name: _probe(cv2, paths[name]) for name in ("source", "incumbent", "challenger")}
    for name, probe in probes.items():
        if probe["frames"] != args.expected_frames or abs(probe["fps"] - args.fps) > 0.01:
            raise ValueError(f"{name} violates the full timeline: {probe}")
    robot_payload = _load_packed(np, paths["robot_masks"])
    limb_payload = _load_packed(np, paths["limb_masks"])
    flower_payload = _load_packed(np, paths["flower_masks"])
    if any(len(value[0]) != args.expected_frames for value in (robot_payload, limb_payload, flower_payload)):
        raise ValueError("packed masks do not cover the full timeline")

    captures = {
        name: cv2.VideoCapture(str(paths[name]))
        for name in ("source", "incumbent", "challenger")
    }
    previous: dict[str, Any] | None = None
    per_frame = []
    values = {"incumbent": [], "challenger": []}
    started = time.perf_counter()
    for index in range(args.expected_frames):
        frames = {}
        for name, capture in captures.items():
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"{name} decode stopped at frame {index}")
            frames[name] = cv2.resize(
                frame,
                (args.evaluation_width, args.evaluation_height),
                interpolation=cv2.INTER_AREA,
            )
        robot = _mask_to_evaluation(
            cv2, np, _unpack(np, robot_payload, index),
            args.evaluation_width, args.evaluation_height,
        )
        limbs = _mask_to_evaluation(
            cv2, np, _unpack(np, limb_payload, index),
            args.evaluation_width, args.evaluation_height,
        )
        flower = _mask_to_evaluation(
            cv2, np, _unpack(np, flower_payload, index),
            args.evaluation_width, args.evaluation_height,
        )
        editable, _, _ = build_torso_head_whitelist(
            cv2,
            np,
            robot=robot,
            limbs=limbs,
            flower=flower,
            limb_dilation_pixels=args.limb_dilation_pixels,
            torso_erosion_pixels=args.torso_erosion_pixels,
            contact_dilation_pixels=args.contact_dilation_pixels,
        )
        same_window = any(start < index <= end for start, end in windows)
        if index in active and same_window and previous is not None:
            if args.flow_reference == "source":
                shared_flow = bidirectional_flow_state(
                    cv2,
                    np,
                    previous["source"],
                    frames["source"],
                    scale=args.flow_scale,
                )
                flows = {name: shared_flow for name in ("incumbent", "challenger")}
            else:
                flows = {
                    name: bidirectional_flow_state(
                        cv2,
                        np,
                        previous[name],
                        frames[name],
                        scale=args.flow_scale,
                    )
                    for name in ("incumbent", "challenger")
                }
            support = editable.copy()
            for flow in flows.values():
                warped_editable = warp_with_flow(
                    cv2,
                    previous["editable"].astype(np.uint8),
                    flow,
                    nearest=True,
                ) > 0
                support &= warped_editable & (
                    flow.confidence >= args.minimum_confidence
                )
            row: dict[str, Any] = {
                "frame": index,
                "support_pixels": int(np.count_nonzero(support)),
            }
            if not np.any(support):
                raise RuntimeError(f"empty jitter support at frame {index}")
            for name in ("incumbent", "challenger"):
                current_residual = (
                    frames[name].astype(np.float32)
                    - frames["source"].astype(np.float32)
                )
                previous_residual = (
                    previous[name].astype(np.float32)
                    - previous["source"].astype(np.float32)
                )
                warped_residual = warp_with_flow(
                    cv2, previous_residual, flows[name]
                )
                mae = float(np.abs(current_residual - warped_residual)[support].mean())
                row[f"{name}_layer_warp_mae"] = mae
                values[name].append(mae)
            per_frame.append(row)
        previous = {**frames, "editable": editable}
    for capture in captures.values():
        capture.release()
    wall_seconds = time.perf_counter() - started

    incumbent = _summary(np, values["incumbent"])
    challenger = _summary(np, values["challenger"])
    relative = 1.0 - challenger["mean"] / incumbent["mean"]
    incumbent_high_jitter = _high_jitter_count(
        values["incumbent"], args.high_jitter_threshold
    )
    challenger_high_jitter = _high_jitter_count(
        values["challenger"], args.high_jitter_threshold
    )
    metrics = {
        "evaluated_transitions": len(per_frame),
        "incumbent_layer_warp_mae": incumbent,
        "challenger_layer_warp_mae": challenger,
        "relative_mean_layer_warp_reduction": relative,
        "high_jitter_threshold": args.high_jitter_threshold,
        "incumbent_high_jitter_transitions": incumbent_high_jitter,
        "challenger_high_jitter_transitions": challenger_high_jitter,
        "wall_seconds": wall_seconds,
        "full_timeline_processing_fps": args.expected_frames / wall_seconds,
        "evaluated_transition_fps": len(per_frame) / wall_seconds,
    }
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "scope": (
            f"{args.flow_reference}-motion-aligned robot head/torso residual "
            "jitter diagnostic"
        ),
        "physical_evidence": False,
        "command": [sys.executable, *sys.argv],
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "coordinate_frames": {
            "source": "camera:source_native_1280x720",
            "packed_masks": "camera:source_aligned_832x480",
            "evaluation": f"camera:source_downsampled_{args.evaluation_width}x{args.evaluation_height}",
            "timeline": "absolute_frame_index:full_source_660",
        },
        "windows_inclusive": [list(window) for window in windows],
        "config": {
            "high_jitter_threshold": args.high_jitter_threshold,
            "minimum_confidence": args.minimum_confidence,
            "flow_scale": args.flow_scale,
            "flow_reference": args.flow_reference,
        },
        "metrics": metrics,
        "gates": {
            "challenger_mean_jitter_lower": challenger["mean"] < incumbent["mean"],
            "challenger_p95_jitter_lower": challenger["p95"] < incumbent["p95"],
            "challenger_high_jitter_count_lower": (
                challenger_high_jitter < incumbent_high_jitter
            ),
            "complete_risk_transition_coverage": len(per_frame)
            == sum(end - start for start, end in windows),
        },
        "runtime": {"python": sys.version, "packages": _packages(), "git": _git_state()},
        "limitations": [
            "This is a 2-D source-motion-aligned diagnostic, not a physical trajectory measurement.",
            "The result covers one flower-arranging scene and the declared risk windows only."
        ],
    }
    frame_path = output / "frame-metrics.json"
    frame_path.write_text(json.dumps(per_frame, indent=2, sort_keys=True) + "\n")
    report["outputs"] = {
        "frame_metrics": {"path": str(frame_path), "sha256": _sha256(frame_path)}
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(report_path), "metrics": metrics, "gates": report["gates"]}, indent=2))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
