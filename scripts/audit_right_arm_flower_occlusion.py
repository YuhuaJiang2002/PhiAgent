#!/usr/bin/env python3
"""Audit anatomical-right-arm flicker and source-grounded flower z-order."""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
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
from phiagent.rendering.temporal_occlusion import (  # noqa: E402
    right_arm_flower_partition,
)
from scripts.audit_robot_layer_long_video import _resolve_frame_masks  # noqa: E402
from scripts.compose_joyai_flower_repairs import _mask_to_native  # noqa: E402
from scripts.stabilize_joyai_appearance_state import (  # noqa: E402
    _git_state,
    _packages,
    _probe,
    _sha256,
    _summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--incumbent-video", type=Path, required=True)
    parser.add_argument("--challenger-video", type=Path, required=True)
    parser.add_argument("--robot-limb-masks", type=Path, required=True)
    parser.add_argument("--person-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument(
        "--flower-mask-contract",
        choices=("resolved_visibility", "tracked_front_layer_with_human_negatives"),
        default="resolved_visibility",
    )
    parser.add_argument("--pose-limb-masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--evaluation-width", type=int, default=320)
    parser.add_argument("--evaluation-height", type=int, default=180)
    parser.add_argument(
        "--mask-projection",
        choices=("source_native", "legacy_832x480_to_native_1280x720"),
        default="legacy_832x480_to_native_1280x720",
    )
    parser.add_argument("--right-arm-key", default="right_packed")
    parser.add_argument("--corridor-dilation-pixels", type=int, default=31)
    parser.add_argument("--hand-dilation-pixels", type=int, default=13)
    parser.add_argument("--flow-scale", type=float, default=0.5)
    parser.add_argument("--minimum-confidence", type=float, default=0.2)
    parser.add_argument("--replacement-threshold", type=float, default=12.0)
    parser.add_argument("--high-flicker-threshold", type=float, default=20.0)
    parser.add_argument("--minimum-arm-support-pixels", type=int, default=32)
    parser.add_argument("--minimum-flower-support-pixels", type=int, default=2)
    parser.add_argument("--person-dilation", type=int, default=10)
    parser.add_argument("--skin-negative-dilation", type=int, default=2)
    parser.add_argument("--person-core-negative-erosion", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser


def _load_key(np: Any, path: Path, key: str) -> tuple[Any, int, int, str]:
    payload = np.load(path, allow_pickle=False)
    if key not in payload.files:
        raise ValueError(f"{path} has no packed mask key {key!r}")
    return (
        payload[key],
        int(payload["height"]),
        int(payload["width"]),
        str(payload["bitorder"]),
    )


def _native_mask(cv2: Any, np: Any, payload: Any, index: int, args: Any) -> Any:
    packed, height, width, bitorder = payload
    flat = np.unpackbits(packed[index], bitorder=bitorder)[: height * width]
    mask = flat.reshape(height, width).astype(np.uint8)
    return _mask_to_native(
        cv2,
        np,
        mask,
        width=args.width,
        height=args.height,
        projection=args.mask_projection,
    )


def _count_above(values: list[float], threshold: float) -> int:
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("high-flicker threshold must be finite and positive")
    return sum(value > threshold for value in values)


def _lower_summary(np: Any, challenger: dict[str, float], incumbent: dict[str, float]) -> dict[str, bool]:
    return {
        "mean_lower": challenger["mean"] < incumbent["mean"],
        "p95_lower": challenger["p95"] < incumbent["p95"],
    }


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    paths = {
        "source": args.source_video.expanduser().resolve(),
        "incumbent": args.incumbent_video.expanduser().resolve(),
        "challenger": args.challenger_video.expanduser().resolve(),
        "robot_limb_masks": args.robot_limb_masks.expanduser().resolve(),
        "person_masks": args.person_masks.expanduser().resolve(),
        "flower_masks": args.flower_masks.expanduser().resolve(),
        "pose_limb_masks": args.pose_limb_masks.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{name}: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    if not math.isfinite(args.replacement_threshold) or args.replacement_threshold <= 0:
        raise ValueError("replacement threshold must be finite and positive")
    _count_above([], args.high_flicker_threshold)

    required = {
        "frames": args.expected_frames,
        "width": args.width,
        "height": args.height,
    }
    for name in ("source", "incumbent", "challenger"):
        probe = _probe(cv2, paths[name])
        if any(probe[key] != value for key, value in required.items()):
            raise ValueError(f"{name} video does not match the timeline: {probe}")
        if abs(float(probe["fps"]) - args.fps) > 0.01:
            raise ValueError(f"{name} FPS mismatch: {probe}")

    right_payload = _load_key(np, paths["robot_limb_masks"], args.right_arm_key)
    person_payload = _load_key(np, paths["person_masks"], "packed")
    flower_payload = _load_key(np, paths["flower_masks"], "packed")
    hands_payload = _load_key(np, paths["pose_limb_masks"], "hands_packed")
    if any(
        len(payload[0]) != args.expected_frames
        for payload in (right_payload, person_payload, flower_payload, hands_payload)
    ):
        raise ValueError("all packed masks must cover the complete timeline")

    captures = {
        name: cv2.VideoCapture(str(paths[name]))
        for name in ("source", "incumbent", "challenger")
    }
    names = ("incumbent", "challenger")
    values = {
        name: {
            "self_flow_arm_mae": [],
            "source_flow_residual_mae": [],
            "wrong_flower_occlusion_fraction": [],
            "flower_owner_flip_fraction": [],
        }
        for name in names
    }
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    started = time.perf_counter()
    for index in range(args.expected_frames):
        frames: dict[str, Any] = {}
        for name, capture in captures.items():
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"{name} decode stopped at frame {index}")
            frames[name] = frame
        right_arm = _native_mask(cv2, np, right_payload, index, args)
        person = _native_mask(cv2, np, person_payload, index, args)
        tracked_flower = _native_mask(cv2, np, flower_payload, index, args)
        hands = _native_mask(cv2, np, hands_payload, index, args)
        _, flower_visible = _resolve_frame_masks(
            np,
            source_rgb=cv2.cvtColor(frames["source"], cv2.COLOR_BGR2RGB),
            person=person,
            tracked_flower=tracked_flower,
            hands=hands,
            person_dilation=args.person_dilation,
            skin_negative_dilation=args.skin_negative_dilation,
            person_core_negative_erosion=args.person_core_negative_erosion,
            flower_mask_contract=args.flower_mask_contract,
        )
        arm_editable, flower_owner, _ = right_arm_flower_partition(
            cv2,
            np,
            right_arm=right_arm,
            flower_visible=flower_visible,
            hand_support=hands,
            corridor_dilation_pixels=args.corridor_dilation_pixels,
            hand_dilation_pixels=args.hand_dilation_pixels,
        )
        evaluation_size = (args.evaluation_width, args.evaluation_height)
        eval_frames = {
            name: cv2.resize(frame, evaluation_size, interpolation=cv2.INTER_AREA)
            for name, frame in frames.items()
        }
        eval_arm = cv2.resize(
            arm_editable.astype(np.uint8), evaluation_size,
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        eval_flower = cv2.resize(
            flower_owner.astype(np.uint8), evaluation_size,
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        replacement = {
            name: np.max(
                np.abs(
                    eval_frames[name].astype(np.int16)
                    - eval_frames["source"].astype(np.int16)
                ),
                axis=2,
            ) > args.replacement_threshold
            for name in names
        }
        row: dict[str, Any] = {
            "frame": index,
            "arm_pixels": int(np.count_nonzero(eval_arm)),
            "flower_owner_pixels": int(np.count_nonzero(eval_flower)),
        }
        for name in names:
            wrong = (
                float(np.mean(replacement[name][eval_flower]))
                if np.count_nonzero(eval_flower) >= args.minimum_flower_support_pixels
                else None
            )
            row[f"{name}_wrong_flower_occlusion_fraction"] = wrong
            if wrong is not None:
                values[name]["wrong_flower_occlusion_fraction"].append(wrong)

        if previous is not None:
            source_flow = bidirectional_flow_state(
                cv2,
                np,
                previous["frames"]["source"],
                eval_frames["source"],
                scale=args.flow_scale,
            )
            flows = {
                name: bidirectional_flow_state(
                    cv2,
                    np,
                    previous["frames"][name],
                    eval_frames[name],
                    scale=args.flow_scale,
                )
                for name in names
            }
            common = eval_arm.copy()
            for flow in (source_flow, *flows.values()):
                warped_arm = warp_with_flow(
                    cv2, previous["arm"].astype(np.uint8), flow, nearest=True
                ) > 0
                common &= warped_arm & (flow.confidence >= args.minimum_confidence)
            row["common_arm_support_pixels"] = int(np.count_nonzero(common))
            if np.count_nonzero(common) >= args.minimum_arm_support_pixels:
                for name in names:
                    warped_candidate = warp_with_flow(
                        cv2, previous["frames"][name], flows[name]
                    )
                    self_mae = float(
                        np.abs(
                            eval_frames[name].astype(np.float32)
                            - warped_candidate.astype(np.float32)
                        )[common].mean()
                    )
                    current_residual = (
                        eval_frames[name].astype(np.float32)
                        - eval_frames["source"].astype(np.float32)
                    )
                    previous_residual = (
                        previous["frames"][name].astype(np.float32)
                        - previous["frames"]["source"].astype(np.float32)
                    )
                    source_mae = float(
                        np.abs(
                            current_residual
                            - warp_with_flow(cv2, previous_residual, source_flow)
                        )[common].mean()
                    )
                    row[f"{name}_self_flow_arm_mae"] = self_mae
                    row[f"{name}_source_flow_residual_mae"] = source_mae
                    values[name]["self_flow_arm_mae"].append(self_mae)
                    values[name]["source_flow_residual_mae"].append(source_mae)

            warped_flower = warp_with_flow(
                cv2, previous["flower"].astype(np.uint8), source_flow, nearest=True
            ) > 0
            stable_flower = (
                eval_flower
                & warped_flower
                & (source_flow.confidence >= args.minimum_confidence)
            )
            row["stable_flower_owner_pixels"] = int(np.count_nonzero(stable_flower))
            if np.count_nonzero(stable_flower) >= args.minimum_flower_support_pixels:
                for name in names:
                    warped_replacement = warp_with_flow(
                        cv2,
                        previous["replacement"][name].astype(np.uint8),
                        source_flow,
                        nearest=True,
                    ) > 0
                    flip = float(
                        np.mean(
                            np.logical_xor(replacement[name], warped_replacement)[
                                stable_flower
                            ]
                        )
                    )
                    row[f"{name}_flower_owner_flip_fraction"] = flip
                    values[name]["flower_owner_flip_fraction"].append(flip)
        rows.append(row)
        previous = {
            "frames": eval_frames,
            "arm": eval_arm,
            "flower": eval_flower,
            "replacement": replacement,
        }

    for capture in captures.values():
        capture.release()
    wall_seconds = time.perf_counter() - started
    summaries = {}
    for name, metrics in values.items():
        summaries[name] = {}
        for metric, metric_values in metrics.items():
            summary = _summary(np, metric_values)
            summary["count"] = len(metric_values)
            summaries[name][metric] = summary
    high_counts = {
        name: {
            metric: _count_above(values[name][metric], args.high_flicker_threshold)
            for metric in ("self_flow_arm_mae", "source_flow_residual_mae")
        }
        for name in names
    }
    gates: dict[str, bool] = {}
    for metric in (
        "self_flow_arm_mae",
        "source_flow_residual_mae",
        "wrong_flower_occlusion_fraction",
        "flower_owner_flip_fraction",
    ):
        for suffix, passed in _lower_summary(
            np, summaries["challenger"][metric], summaries["incumbent"][metric]
        ).items():
            gates[f"challenger_{metric}_{suffix}"] = passed
        gates[f"matched_{metric}_coverage"] = (
            summaries["challenger"][metric]["count"]
            == summaries["incumbent"][metric]["count"]
            and summaries["incumbent"][metric]["count"] > 0
        )
    for metric in ("self_flow_arm_mae", "source_flow_residual_mae"):
        gates[f"challenger_{metric}_high_count_lower"] = (
            high_counts["challenger"][metric] < high_counts["incumbent"][metric]
        )

    frame_metrics_path = output / "frame-metrics.json"
    frame_metrics_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "scope": "2D anatomical-right-arm temporal appearance and source-grounded flower ownership audit",
        "physical_evidence": False,
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _packages(),
        "git": _git_state(),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "coordinate_frames": {
            "video": f"camera:source_native_{args.width}x{args.height}",
            "evaluation": (
                f"camera:source_downsampled_{args.evaluation_width}x"
                f"{args.evaluation_height}"
            ),
            "timeline": f"absolute_frame_index:full_source_{args.expected_frames}",
        },
        "config": {
            "right_arm_key": args.right_arm_key,
            "flow_scale": args.flow_scale,
            "minimum_confidence": args.minimum_confidence,
            "replacement_threshold": args.replacement_threshold,
            "high_flicker_threshold": args.high_flicker_threshold,
            "corridor_dilation_pixels": args.corridor_dilation_pixels,
            "hand_dilation_pixels": args.hand_dilation_pixels,
            "seed": args.seed,
        },
        "metrics": {
            "incumbent": summaries["incumbent"],
            "challenger": summaries["challenger"],
            "high_flicker_counts": high_counts,
            "wall_seconds": wall_seconds,
            "processing_fps": args.expected_frames / wall_seconds,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "outputs": {
            "frame_metrics": {
                "path": str(frame_metrics_path),
                "sha256": _sha256(frame_metrics_path),
            }
        },
        "limitations": [
            "Flower ownership is source-grounded 2-D visibility, not metric depth.",
            "Optical-flow residuals are temporal visual diagnostics, not force evidence.",
            "The full native-resolution long-video and adversarial audits remain mandatory."
        ],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(report_path), "gates": gates, "metrics": report["metrics"]}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
