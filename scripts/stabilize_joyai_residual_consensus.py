#!/usr/bin/env python3
"""Build an offline bidirectional consensus over a bounded appearance residual."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compose_joyai_flower_repairs import _load_packed  # noqa: E402
from phiagent.rendering.temporal_appearance import (  # noqa: E402
    bidirectional_flow_state,
    warp_with_flow,
    weighted_residual_consensus,
)
from stabilize_joyai_appearance_state import (  # noqa: E402
    _editable_mask,
    _finish,
    _git_state,
    _packages,
    _probe,
    _sha256,
    _sheet,
    _writer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--incumbent-video", type=Path, required=True)
    parser.add_argument("--robot-masks", type=Path, required=True)
    parser.add_argument("--limb-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument(
        "--window",
        nargs=2,
        type=int,
        action="append",
        metavar=("START", "END"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--mask-projection",
        choices=("source_native", "legacy_832x480_to_native_1280x720"),
        default="source_native",
        help=(
            "Named packed-mask-to-video camera transform.  The legacy option "
            "is the audited scale/crop inverse for 832x480 masks on a "
            "1280x720 source timeline."
        ),
    )
    parser.add_argument("--edge-ramp-frames", type=int, default=4)
    parser.add_argument("--limb-dilation-pixels", type=int, default=15)
    parser.add_argument("--torso-erosion-pixels", type=int, default=5)
    parser.add_argument("--contact-dilation-pixels", type=int, default=21)
    parser.add_argument("--state-boundary-erosion-pixels", type=int, default=7)
    parser.add_argument("--flow-scale", type=float, default=0.5)
    parser.add_argument("--minimum-confidence", type=float, default=0.20)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--minimum-observations", type=int, default=3)
    parser.add_argument("--maximum-channel-mad", type=float, default=8.0)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--maximum-residual-delta", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser


def _decode_selected(
    cv2: Any,
    path: Path,
    selected: set[int],
    expected_frames: int,
) -> dict[int, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    result = {}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in selected:
            result[index] = frame
        index += 1
    capture.release()
    if index != expected_frames or set(result) != selected:
        raise RuntimeError(
            f"selected decode mismatch for {path}: decoded={index}, selected={len(result)}"
        )
    return result


def _validate_windows(
    windows: tuple[tuple[int, int], ...],
    expected_frames: int,
) -> None:
    if any(not 0 <= start < end < expected_frames for start, end in windows):
        raise ValueError("consensus windows must be valid inclusive intervals")
    if any(left[1] >= right[0] for left, right in zip(windows, windows[1:])):
        raise ValueError("consensus windows must not overlap")


def _summary(np: Any, values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


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
    (output / "review").mkdir()

    windows = tuple(sorted((int(start), int(end)) for start, end in args.window))
    _validate_windows(windows, args.expected_frames)
    if args.radius < 1:
        raise ValueError("radius must be positive")
    if args.minimum_observations > 2 * args.radius + 1:
        raise ValueError("minimum observations exceed the temporal neighborhood")
    if not 0 < args.strength <= 1:
        raise ValueError("strength must be in (0, 1]")
    if not 0 <= args.minimum_confidence <= 1:
        raise ValueError("minimum confidence must be in [0, 1]")
    if args.maximum_residual_delta <= 0 or args.maximum_channel_mad <= 0:
        raise ValueError("residual limits must be positive")
    for pixels, name in (
        (args.limb_dilation_pixels, "limb dilation"),
        (args.torso_erosion_pixels, "torso erosion"),
        (args.contact_dilation_pixels, "contact dilation"),
        (args.state_boundary_erosion_pixels, "state boundary erosion"),
    ):
        if pixels < 1 or pixels % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer")

    probes = {
        name: _probe(cv2, paths[name])
        for name in ("candidate", "incumbent")
    }
    for name, probe in probes.items():
        if (
            probe["frames"] != args.expected_frames
            or probe["width"] != args.width
            or probe["height"] != args.height
            or abs(float(probe["fps"]) - args.fps) > 0.01
        ):
            raise ValueError(f"{name} video does not match timeline: {probe}")

    selected = {
        frame
        for start, end in windows
        for frame in range(start, end + 1)
    }
    candidate_frames = _decode_selected(
        cv2, paths["candidate"], selected, args.expected_frames
    )
    incumbent_frames = _decode_selected(
        cv2, paths["incumbent"], selected, args.expected_frames
    )
    robot_payload = _load_packed(np, paths["robot_masks"])
    limb_payload = _load_packed(np, paths["limb_masks"])
    flower_payload = _load_packed(np, paths["flower_masks"])
    if any(
        len(payload[0]) != args.expected_frames
        for payload in (robot_payload, limb_payload, flower_payload)
    ):
        raise ValueError("mask archives must cover the complete timeline")
    editables = {
        frame: _editable_mask(
            cv2,
            np,
            index=frame,
            robot_payload=robot_payload,
            limb_payload=limb_payload,
            flower_payload=flower_payload,
            width=args.width,
            height=args.height,
            limb_dilation_pixels=args.limb_dilation_pixels,
            torso_erosion_pixels=args.torso_erosion_pixels,
            contact_dilation_pixels=args.contact_dilation_pixels,
            include_adjacent_envelope=True,
            mask_projection=args.mask_projection,
        )
        for frame in selected
    }
    boundary_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.state_boundary_erosion_pixels, args.state_boundary_erosion_pixels),
    )
    interiors = {
        frame: cv2.erode(editables[frame].astype(np.uint8), boundary_kernel) > 0
        for frame in selected
    }

    repaired_frames = {frame: candidate_frames[frame].copy() for frame in selected}
    per_frame = []
    correction_values = []
    reliable_fractions = []
    support_values = []
    mad_values = []
    started = time.perf_counter()
    for start, end in windows:
        core_start = start + args.edge_ramp_frames + 1
        core_end = end - args.edge_ramp_frames - 1
        for frame in range(core_start, core_end + 1):
            current_residual = (
                candidate_frames[frame].astype(np.float32)
                - incumbent_frames[frame].astype(np.float32)
            )
            residuals = [current_residual]
            weights = [interiors[frame].astype(np.float32)]
            neighbors = []
            for neighbor in range(
                max(core_start, frame - args.radius),
                min(core_end, frame + args.radius) + 1,
            ):
                if neighbor == frame:
                    continue
                flow = bidirectional_flow_state(
                    cv2,
                    np,
                    incumbent_frames[neighbor],
                    incumbent_frames[frame],
                    scale=args.flow_scale,
                )
                neighbor_residual = (
                    candidate_frames[neighbor].astype(np.float32)
                    - incumbent_frames[neighbor].astype(np.float32)
                )
                warped_residual = warp_with_flow(
                    cv2,
                    neighbor_residual,
                    flow,
                )
                warped_editable = warp_with_flow(
                    cv2,
                    editables[neighbor].astype(np.uint8),
                    flow,
                    nearest=True,
                ) > 0
                weight = flow.confidence * (
                    interiors[frame] & warped_editable
                ).astype(np.float32)
                weight[weight < args.minimum_confidence] = 0.0
                residuals.append(warped_residual)
                weights.append(weight)
                neighbors.append(neighbor)

            if len(residuals) < args.minimum_observations:
                per_frame.append(
                    {
                        "frame": frame,
                        "neighbors": neighbors,
                        "reliable_fraction_of_editable": 0.0,
                        "mean_support_count": float(len(residuals)),
                        "mean_maximum_channel_mad": 0.0,
                        "mean_abs_applied_correction": 0.0,
                        "decision": "unchanged_insufficient_observations",
                    }
                )
                continue
            consensus = weighted_residual_consensus(
                np,
                residuals=np.stack(residuals),
                weights=np.stack(weights),
                minimum_observations=args.minimum_observations,
                maximum_channel_mad=args.maximum_channel_mad,
            )
            reliable = np.logical_and(
                np.asarray(consensus.reliable, dtype=np.bool_),
                np.asarray(interiors[frame], dtype=np.bool_),
            )
            correction = np.clip(
                consensus.value - current_residual,
                -args.maximum_residual_delta,
                args.maximum_residual_delta,
            )
            normalized_weight = np.clip(
                (consensus.weight_sum - 1.0)
                / np.maximum(consensus.support_count - 1, 1),
                0.0,
                1.0,
            )
            repaired = candidate_frames[frame].astype(np.float32)
            repaired += (
                correction
                * (args.strength * normalized_weight[..., None])
                * reliable[..., None]
            )
            repaired = np.clip(np.rint(repaired), 0, 255).astype(np.uint8)
            immutable = np.logical_not(reliable)
            repaired[immutable] = candidate_frames[frame][immutable]
            repaired_frames[frame] = repaired
            active_correction = np.abs(
                repaired.astype(np.float32)
                - candidate_frames[frame].astype(np.float32)
            )
            reliable_fraction = float(
                np.count_nonzero(reliable)
                / max(1, np.count_nonzero(editables[frame]))
            )
            reliable_fractions.append(reliable_fraction)
            if np.any(reliable):
                correction_values.append(float(np.mean(active_correction[reliable])))
                support_values.append(float(np.mean(consensus.support_count[reliable])))
                mad_values.append(float(np.mean(consensus.maximum_channel_mad[reliable])))
            per_frame.append(
                {
                    "frame": frame,
                    "neighbors": neighbors,
                    "reliable_fraction_of_editable": reliable_fraction,
                    "mean_support_count": (
                        float(np.mean(consensus.support_count[reliable]))
                        if np.any(reliable)
                        else 0.0
                    ),
                    "mean_maximum_channel_mad": (
                        float(np.mean(consensus.maximum_channel_mad[reliable]))
                        if np.any(reliable)
                        else 0.0
                    ),
                    "mean_abs_applied_correction": (
                        float(np.mean(active_correction[reliable]))
                        if np.any(reliable)
                        else 0.0
                    ),
                }
            )

    lossless = output / "appearance-consensus-lossless.mkv"
    review = output / "appearance-consensus.mp4"
    lossless_writer = _writer(
        paths["ffmpeg"],
        lossless,
        width=args.width,
        height=args.height,
        fps=args.fps,
        lossless=True,
    )
    review_writer = _writer(
        paths["ffmpeg"],
        review,
        width=args.width,
        height=args.height,
        fps=args.fps,
        lossless=False,
    )
    assert lossless_writer.stdin is not None and review_writer.stdin is not None
    candidate_capture = cv2.VideoCapture(str(paths["candidate"]))
    for frame in range(args.expected_frames):
        ok, original = candidate_capture.read()
        if not ok:
            raise RuntimeError(f"candidate decode stopped at frame {frame}")
        value = repaired_frames.get(frame, original)
        lossless_writer.stdin.write(value.tobytes())
        review_writer.stdin.write(value.tobytes())
    candidate_capture.release()
    lossless_log = _finish(lossless_writer, "lossless")
    review_log = _finish(review_writer, "review")
    wall_seconds = time.perf_counter() - started

    lossless_probe = _probe(cv2, lossless)
    decoded = cv2.VideoCapture(str(lossless))
    original = cv2.VideoCapture(str(paths["candidate"]))
    outside_exact = outside_total = 0
    outside_window_exact = outside_window_total = 0
    endpoint_exact = endpoint_total = 0
    endpoints = {value for window in windows for value in window}
    for frame in range(args.expected_frames):
        decoded_ok, decoded_frame = decoded.read()
        original_ok, original_frame = original.read()
        if not decoded_ok or not original_ok:
            raise RuntimeError(f"postdecode audit stopped at frame {frame}")
        exact = np.all(decoded_frame == original_frame, axis=2)
        if frame in selected:
            outside = np.logical_not(
                np.asarray(editables[frame], dtype=np.bool_)
            )
            outside_exact += int(np.count_nonzero(exact[outside]))
            outside_total += int(np.count_nonzero(outside))
        else:
            outside_window_exact += int(np.count_nonzero(exact))
            outside_window_total += exact.size
        if frame in endpoints:
            endpoint_exact += int(np.count_nonzero(exact))
            endpoint_total += exact.size
    decoded.release()
    original.release()

    review_frames = sorted(
        set(
            [value for window in windows for value in window]
            + [
                int(row["frame"])
                for row in sorted(
                    per_frame,
                    key=lambda row: float(row["mean_abs_applied_correction"]),
                    reverse=True,
                )[:12]
            ]
        )
    )
    sheet_path = output / "review" / "incumbent-before-consensus.jpg"
    sheet = _sheet(
        cv2,
        np,
        [
            (
                frame,
                incumbent_frames[frame],
                candidate_frames[frame],
                repaired_frames[frame],
            )
            for frame in review_frames
        ],
    )
    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"failed to write {sheet_path}")
    frame_metrics_path = output / "frame-metrics.json"
    frame_metrics_path.write_text(
        json.dumps(per_frame, indent=2, sort_keys=True) + "\n"
    )
    metrics = {
        "reliable_fraction_of_editable": _summary(np, reliable_fractions),
        "support_count": _summary(np, support_values),
        "maximum_channel_mad": _summary(np, mad_values),
        "mean_applied_correction": _summary(np, correction_values),
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
        "full_timeline_decodes": lossless_probe["frames"] == args.expected_frames,
        "outside_editable_exact": metrics[
            "postdecode_outside_editable_exact_fraction"
        ]
        == 1.0,
        "outside_windows_exact": metrics[
            "postdecode_outside_windows_exact_fraction"
        ]
        == 1.0,
        "window_endpoints_exact": metrics[
            "postdecode_window_endpoints_exact_fraction"
        ]
        == 1.0,
        "consensus_nonempty": (
            metrics["reliable_fraction_of_editable"]["mean"] is not None
            and metrics["reliable_fraction_of_editable"]["mean"] > 0
        ),
    }
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "decision": "AWAITING_FROZEN_FLICKER_EVALUATION_AND_NATIVE_REVIEW",
        "method": "bidirectional_multi_anchor_weighted_median_residual_consensus",
        "physical_evidence": False,
        "command": [sys.executable, *sys.argv],
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
            if name != "ffmpeg"
        },
        "windows": [list(window) for window in windows],
        "config": {
            "edge_ramp_frames": args.edge_ramp_frames,
            "limb_dilation_pixels": args.limb_dilation_pixels,
            "torso_erosion_pixels": args.torso_erosion_pixels,
            "contact_dilation_pixels": args.contact_dilation_pixels,
            "state_boundary_erosion_pixels": args.state_boundary_erosion_pixels,
            "flow_scale": args.flow_scale,
            "minimum_confidence": args.minimum_confidence,
            "radius": args.radius,
            "minimum_observations": args.minimum_observations,
            "maximum_channel_mad": args.maximum_channel_mad,
            "strength": args.strength,
            "maximum_residual_delta": args.maximum_residual_delta,
            "mask_projection": args.mask_projection,
            "seed": args.seed,
        },
        "metrics": metrics,
        "gates": gates,
        "runtime": {
            "python": sys.version,
            "packages": _packages(),
            "git": _git_state(),
        },
        "outputs": {
            "lossless": {"path": str(lossless), "sha256": _sha256(lossless)},
            "review": {"path": str(review), "sha256": _sha256(review)},
            "review_sheet": {
                "path": str(sheet_path),
                "sha256": _sha256(sheet_path),
            },
            "frame_metrics": {
                "path": str(frame_metrics_path),
                "sha256": _sha256(frame_metrics_path),
            },
        },
        "encoder_logs": {"lossless": lossless_log, "review": review_log},
        "limitations": [
            "Consensus uses image-space incumbent correspondence, not exact material coordinates.",
            "Only pixels with sufficient mutually consistent observations are changed.",
            "Hands, contact, visibility boundaries, and low-confidence points remain outside repair authority.",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "metrics": metrics,
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
