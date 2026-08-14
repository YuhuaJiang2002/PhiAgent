#!/usr/bin/env python3
"""Suppress one-frame robot-appearance outliers with past/future consensus."""

from __future__ import annotations

import argparse
import json
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
    bidirectional_residual_consensus_update,
    warp_with_flow,
)
from scripts.compose_joyai_flower_repairs import _load_packed  # noqa: E402
from scripts.stabilize_joyai_appearance_state import (  # noqa: E402
    _editable_mask,
    _finish,
    _git_state,
    _packages,
    _probe,
    _sha256,
    _summary,
    _window_for_frame,
    _writer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--incumbent-video", type=Path, required=True)
    parser.add_argument("--robot-masks", type=Path, required=True)
    parser.add_argument("--limb-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--window", nargs=2, type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--mask-projection",
        choices=("source_native", "legacy_832x480_to_native_1280x720"),
        default="legacy_832x480_to_native_1280x720",
    )
    parser.add_argument("--limb-dilation-pixels", type=int, default=23)
    parser.add_argument("--torso-erosion-pixels", type=int, default=9)
    parser.add_argument("--contact-dilation-pixels", type=int, default=31)
    parser.add_argument("--state-boundary-erosion-pixels", type=int, default=11)
    parser.add_argument("--flow-scale", type=float, default=0.5)
    parser.add_argument("--minimum-confidence", type=float, default=0.2)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--maximum-residual-delta", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser


def _read_record(
    cv2: Any,
    np: Any,
    *,
    index: int,
    candidate_capture: Any,
    incumbent_capture: Any,
    windows: tuple[tuple[int, int], ...],
    robot_payload: tuple[Any, int, int, str],
    limb_payload: tuple[Any, int, int, str],
    flower_payload: tuple[Any, int, int, str],
    boundary_kernel: Any,
    args: Any,
) -> dict[str, Any]:
    candidate_ok, candidate = candidate_capture.read()
    incumbent_ok, incumbent = incumbent_capture.read()
    if not candidate_ok or not incumbent_ok:
        raise RuntimeError(f"video decode stopped at frame {index}")
    window = _window_for_frame(windows, index)
    editable = _editable_mask(
        cv2,
        np,
        index=index,
        robot_payload=robot_payload,
        limb_payload=limb_payload,
        flower_payload=flower_payload,
        width=args.width,
        height=args.height,
        limb_dilation_pixels=args.limb_dilation_pixels,
        torso_erosion_pixels=args.torso_erosion_pixels,
        contact_dilation_pixels=args.contact_dilation_pixels,
        include_adjacent_envelope=window is not None,
        mask_projection=args.mask_projection,
    )
    interior = cv2.erode(editable.astype(np.uint8), boundary_kernel) > 0
    return {
        "index": index,
        "candidate": candidate,
        "incumbent": incumbent,
        "window": window,
        "editable": editable,
        "interior": interior,
    }


def _repair_middle(
    cv2: Any,
    np: Any,
    *,
    previous: dict[str, Any],
    current: dict[str, Any],
    following: dict[str, Any],
    args: Any,
) -> tuple[Any, dict[str, Any]]:
    index = int(current["index"])
    window = current["window"]
    repaired = current["candidate"].copy()
    metrics: dict[str, Any] = {
        "frame": index,
        "window": list(window) if window is not None else None,
        "editable_fraction": float(np.mean(current["editable"])),
    }
    if window is None or index in window:
        return repaired, metrics

    previous_flow = bidirectional_flow_state(
        cv2,
        np,
        previous["candidate"],
        current["candidate"],
        scale=args.flow_scale,
    )
    next_flow = bidirectional_flow_state(
        cv2,
        np,
        following["candidate"],
        current["candidate"],
        scale=args.flow_scale,
    )
    previous_editable = warp_with_flow(
        cv2, previous["editable"].astype(np.uint8), previous_flow, nearest=True
    ) > 0
    next_editable = warp_with_flow(
        cv2, following["editable"].astype(np.uint8), next_flow, nearest=True
    ) > 0
    reliable = (
        current["interior"]
        & previous_editable
        & next_editable
        & (previous_flow.confidence >= args.minimum_confidence)
        & (next_flow.confidence >= args.minimum_confidence)
    )
    current_residual = (
        current["candidate"].astype(np.float32)
        - current["incumbent"].astype(np.float32)
    )
    previous_residual = (
        previous["candidate"].astype(np.float32)
        - previous["incumbent"].astype(np.float32)
    )
    next_residual = (
        following["candidate"].astype(np.float32)
        - following["incumbent"].astype(np.float32)
    )
    warped_previous = warp_with_flow(cv2, previous_residual, previous_flow)
    warped_next = warp_with_flow(cv2, next_residual, next_flow)
    target = np.median(
        np.stack((warped_previous, current_residual, warped_next), axis=0),
        axis=0,
    )
    if np.any(reliable):
        metrics["baseline_consensus_mae"] = float(
            np.abs(current_residual - target)[reliable].mean()
        )
    repaired, update_metrics = bidirectional_residual_consensus_update(
        np,
        current_candidate=current["candidate"],
        current_residual=current_residual,
        warped_previous_residual=warped_previous,
        warped_next_residual=warped_next,
        previous_confidence=previous_flow.confidence,
        next_confidence=next_flow.confidence,
        reliable=reliable,
        strength=args.strength,
        maximum_residual_delta=args.maximum_residual_delta,
    )
    immutable = np.logical_not(np.asarray(current["editable"], dtype=np.bool_))
    repaired[immutable] = current["candidate"][immutable]
    repaired_residual = (
        repaired.astype(np.float32) - current["incumbent"].astype(np.float32)
    )
    if np.any(reliable):
        metrics["repaired_consensus_mae"] = float(
            np.abs(repaired_residual - target)[reliable].mean()
        )
    metrics.update(update_metrics)
    metrics["reliable_fraction_of_editable"] = float(np.count_nonzero(reliable)) / max(
        1, int(np.count_nonzero(current["editable"]))
    )
    return repaired, metrics


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    paths = {
        "candidate": args.candidate_video.expanduser().resolve(),
        "incumbent": args.incumbent_video.expanduser().resolve(),
        "robot_masks": args.robot_masks.expanduser().resolve(),
        "limb_masks": args.limb_masks.expanduser().resolve(),
        "flower_masks": args.flower_masks.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{name}: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    windows = tuple(sorted((int(start), int(end)) for start, end in args.window))
    if any(not 0 <= start < end < args.expected_frames for start, end in windows):
        raise ValueError("windows must be valid inclusive intervals")
    if any(left[1] >= right[0] for left, right in zip(windows, windows[1:])):
        raise ValueError("windows must not overlap")
    for pixels, label in (
        (args.limb_dilation_pixels, "limb dilation"),
        (args.torso_erosion_pixels, "torso erosion"),
        (args.contact_dilation_pixels, "contact dilation"),
        (args.state_boundary_erosion_pixels, "state boundary erosion"),
    ):
        if pixels < 1 or pixels % 2 == 0:
            raise ValueError(f"{label} must be a positive odd integer")

    required_probe = {
        "frames": args.expected_frames,
        "width": args.width,
        "height": args.height,
    }
    for name in ("candidate", "incumbent"):
        probe = _probe(cv2, paths[name])
        if any(probe[key] != value for key, value in required_probe.items()):
            raise ValueError(f"{name} video does not match timeline: {probe}")
        if abs(float(probe["fps"]) - args.fps) > 0.01:
            raise ValueError(f"{name} FPS mismatch: {probe}")

    robot_payload = _load_packed(np, paths["robot_masks"])
    limb_payload = _load_packed(np, paths["limb_masks"])
    flower_payload = _load_packed(np, paths["flower_masks"])
    if any(
        len(payload[0]) != args.expected_frames
        for payload in (robot_payload, limb_payload, flower_payload)
    ):
        raise ValueError("packed masks must cover the complete timeline")

    lossless = output / "appearance-bidirectional-consensus-lossless.mkv"
    review = output / "appearance-bidirectional-consensus.mp4"
    lossless_writer = _writer(
        paths["ffmpeg"], lossless, width=args.width, height=args.height,
        fps=args.fps, lossless=True,
    )
    review_writer = _writer(
        paths["ffmpeg"], review, width=args.width, height=args.height,
        fps=args.fps, lossless=False,
    )
    assert lossless_writer.stdin is not None and review_writer.stdin is not None
    candidate_capture = cv2.VideoCapture(str(paths["candidate"]))
    incumbent_capture = cv2.VideoCapture(str(paths["incumbent"]))
    boundary_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.state_boundary_erosion_pixels, args.state_boundary_erosion_pixels),
    )

    def read(index: int) -> dict[str, Any]:
        return _read_record(
            cv2,
            np,
            index=index,
            candidate_capture=candidate_capture,
            incumbent_capture=incumbent_capture,
            windows=windows,
            robot_payload=robot_payload,
            limb_payload=limb_payload,
            flower_payload=flower_payload,
            boundary_kernel=boundary_kernel,
            args=args,
        )

    started = time.perf_counter()
    previous = read(0)
    current = read(1)
    lossless_writer.stdin.write(previous["candidate"].tobytes())
    review_writer.stdin.write(previous["candidate"].tobytes())
    per_frame: list[dict[str, Any]] = [
        {"frame": 0, "window": None, "endpoint_exact": True}
    ]
    for index in range(2, args.expected_frames):
        following = read(index)
        repaired, metrics = _repair_middle(
            cv2,
            np,
            previous=previous,
            current=current,
            following=following,
            args=args,
        )
        lossless_writer.stdin.write(repaired.tobytes())
        review_writer.stdin.write(repaired.tobytes())
        per_frame.append(metrics)
        previous, current = current, following
    lossless_writer.stdin.write(current["candidate"].tobytes())
    review_writer.stdin.write(current["candidate"].tobytes())
    per_frame.append(
        {
            "frame": args.expected_frames - 1,
            "window": list(current["window"]) if current["window"] else None,
            "endpoint_exact": True,
        }
    )
    candidate_capture.release()
    incumbent_capture.release()
    lossless_log = _finish(lossless_writer, "lossless")
    review_log = _finish(review_writer, "review")
    wall_seconds = time.perf_counter() - started

    decoded = cv2.VideoCapture(str(lossless))
    original = cv2.VideoCapture(str(paths["candidate"]))
    outside_exact = outside_total = 0
    outside_window_exact = outside_window_total = 0
    endpoint_exact = endpoint_total = 0
    window_frames = {
        frame for start, end in windows for frame in range(start, end + 1)
    }
    endpoints = {value for window in windows for value in window}
    for index in range(args.expected_frames):
        decoded_ok, decoded_frame = decoded.read()
        original_ok, original_frame = original.read()
        if not decoded_ok or not original_ok:
            raise RuntimeError(f"post-decode audit stopped at frame {index}")
        editable = _editable_mask(
            cv2,
            np,
            index=index,
            robot_payload=robot_payload,
            limb_payload=limb_payload,
            flower_payload=flower_payload,
            width=args.width,
            height=args.height,
            limb_dilation_pixels=args.limb_dilation_pixels,
            torso_erosion_pixels=args.torso_erosion_pixels,
            contact_dilation_pixels=args.contact_dilation_pixels,
            include_adjacent_envelope=index in window_frames,
            mask_projection=args.mask_projection,
        )
        exact = np.all(decoded_frame == original_frame, axis=2)
        immutable = np.logical_not(editable)
        outside_exact += int(np.count_nonzero(exact[immutable]))
        outside_total += int(np.count_nonzero(immutable))
        if index not in window_frames:
            outside_window_exact += int(np.count_nonzero(exact))
            outside_window_total += exact.size
        if index in endpoints:
            endpoint_exact += int(np.count_nonzero(exact))
            endpoint_total += exact.size
    decoded.release()
    original.release()

    baseline = [
        float(row["baseline_consensus_mae"])
        for row in per_frame if "baseline_consensus_mae" in row
    ]
    repaired_values = [
        float(row["repaired_consensus_mae"])
        for row in per_frame if "repaired_consensus_mae" in row
    ]
    applied = [
        float(row["mean_abs_applied_correction"])
        for row in per_frame if "mean_abs_applied_correction" in row
    ]
    metrics = {
        "baseline_consensus_mae": _summary(np, baseline),
        "repaired_consensus_mae": _summary(np, repaired_values),
        "relative_mean_consensus_reduction": (
            1.0 - float(np.mean(repaired_values)) / float(np.mean(baseline))
            if baseline and float(np.mean(baseline)) > 0 else None
        ),
        "mean_applied_correction": _summary(np, applied),
        "postdecode_outside_editable_exact_fraction": outside_exact / outside_total,
        "postdecode_outside_windows_exact_fraction": (
            outside_window_exact / outside_window_total
        ),
        "postdecode_window_endpoints_exact_fraction": endpoint_exact / endpoint_total,
        "frames": args.expected_frames,
        "video_seconds": args.expected_frames / args.fps,
        "wall_seconds": wall_seconds,
        "processing_fps": args.expected_frames / wall_seconds,
    }
    gates = {
        "full_timeline_decodes": _probe(cv2, lossless)["frames"] == args.expected_frames,
        "outside_editable_exact": metrics["postdecode_outside_editable_exact_fraction"] == 1.0,
        "outside_windows_exact": metrics["postdecode_outside_windows_exact_fraction"] == 1.0,
        "window_endpoints_exact": metrics["postdecode_window_endpoints_exact_fraction"] == 1.0,
        "bidirectional_consensus_non_regression": (
            metrics["relative_mean_consensus_reduction"] is not None
            and metrics["relative_mean_consensus_reduction"] >= 0
        ),
    }
    frame_metrics = output / "frame-metrics.json"
    frame_metrics.write_text(json.dumps(per_frame, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "decision": "AWAITING_FROZEN_EXTERNAL_AUDITS_AND_NATIVE_REVIEW",
        "method": "candidate_motion_aligned_bidirectional_temporal_residual_median",
        "physical_evidence": False,
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _packages(),
        "git": _git_state(),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items() if name != "ffmpeg"
        },
        "windows": [list(window) for window in windows],
        "config": {
            "flow_reference": "candidate",
            "flow_scale": args.flow_scale,
            "minimum_confidence": args.minimum_confidence,
            "strength": args.strength,
            "maximum_residual_delta": args.maximum_residual_delta,
            "mask_projection": args.mask_projection,
            "seed": args.seed,
        },
        "metrics": metrics,
        "gates": gates,
        "outputs": {
            "lossless": {"path": str(lossless), "sha256": _sha256(lossless)},
            "review": {"path": str(review), "sha256": _sha256(review)},
            "frame_metrics": {"path": str(frame_metrics), "sha256": _sha256(frame_metrics)},
        },
        "encoder_logs": {"lossless": lossless_log, "review": review_log},
        "limitations": [
            "Past/future consensus is an offline visual smoother, not metric robot dynamics.",
            "Hand, flower, contact, boundary, and low-confidence pixels remain immutable.",
            "The complete frozen external jitter and adversarial audits remain authoritative."
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "metrics": metrics, "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
