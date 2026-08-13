#!/usr/bin/env python3
"""Stabilize a bounded JoyAI residual without blending current-frame geometry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from compose_joyai_flower_repairs import (  # noqa: E402
    _load_packed,
    _mask_to_native,
    _unpack,
    apply_temporal_lock_envelope,
    build_limb_contact_locks,
    build_torso_head_whitelist,
)
from phiagent.rendering.temporal_appearance import (  # noqa: E402
    bidirectional_flow_state,
    residual_state_update,
    warp_with_flow,
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
    parser.add_argument(
        "--mode",
        choices=("none", "global", "naive", "gated"),
        required=True,
    )
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
    parser.add_argument("--minimum-state-age", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--gaussian-sigma", type=float, default=3.0)
    parser.add_argument("--maximum-residual-delta", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(cv2: Any, path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    value = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return value


def _writer(
    ffmpeg: Path,
    output: Path,
    *,
    width: int,
    height: int,
    fps: float,
    lossless: bool,
) -> Any:
    codec = (
        ["-c:v", "ffv1", "-level", "3", "-g", "1"]
        if lossless
        else [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "8",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )
    return subprocess.Popen(
        [
            str(ffmpeg),
            "-y",
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
            *codec,
            str(output),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _finish(process: Any, label: str) -> str:
    assert process.stdin is not None
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    code = process.wait()
    if code:
        raise RuntimeError(f"{label} encoder returned {code}: {stderr}")
    return stderr


def _packages() -> dict[str, str | None]:
    versions = {}
    for name in ("numpy", "opencv-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_state() -> dict[str, Any]:
    result = {}
    for name, command in {
        "head": ["git", "rev-parse", "--verify", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        result[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
        }
    return result


def _window_for_frame(
    windows: tuple[tuple[int, int], ...], frame: int
) -> tuple[int, int] | None:
    for start, end in windows:
        if start <= frame <= end:
            return start, end
    return None


def _editable_mask(
    cv2: Any,
    np: Any,
    *,
    index: int,
    robot_payload: tuple[Any, int, int, str],
    limb_payload: tuple[Any, int, int, str],
    flower_payload: tuple[Any, int, int, str],
    width: int,
    height: int,
    limb_dilation_pixels: int,
    torso_erosion_pixels: int,
    contact_dilation_pixels: int,
    include_adjacent_envelope: bool,
    mask_projection: str = "source_native",
) -> Any:
    def native(payload: tuple[Any, int, int, str], frame: int) -> Any:
        return _mask_to_native(
            cv2,
            np,
            _unpack(np, payload, frame),
            width=width,
            height=height,
            projection=mask_projection,
        )

    flower = native(flower_payload, index)
    limbs = native(limb_payload, index)
    editable, _, _ = build_torso_head_whitelist(
        cv2,
        np,
        robot=native(robot_payload, index),
        limbs=limbs,
        flower=flower,
        limb_dilation_pixels=limb_dilation_pixels,
        torso_erosion_pixels=torso_erosion_pixels,
        contact_dilation_pixels=contact_dilation_pixels,
    )
    if not include_adjacent_envelope:
        return editable
    adjacent_locks = []
    for adjacent in (index - 1, index + 1):
        if not 0 <= adjacent < len(robot_payload[0]):
            continue
        adjacent_flower = native(flower_payload, adjacent)
        adjacent_limb = native(limb_payload, adjacent)
        limb_lock, contact_lock = build_limb_contact_locks(
            cv2,
            np,
            limbs=adjacent_limb,
            flower=adjacent_flower,
            limb_dilation_pixels=limb_dilation_pixels,
            contact_dilation_pixels=contact_dilation_pixels,
        )
        adjacent_locks.append(limb_lock | adjacent_flower | contact_lock)
    return apply_temporal_lock_envelope(
        np,
        editable=editable,
        adjacent_locked_masks=adjacent_locks,
    )


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


def _sheet(cv2: Any, np: Any, rows: list[tuple[int, Any, Any, Any]]) -> Any:
    tiles = []
    for frame, incumbent, candidate, repaired in rows:
        panels = []
        for label, value in (
            ("INCUMBENT", incumbent),
            ("BEFORE", candidate),
            ("AFTER", repaired),
        ):
            panel = cv2.resize(value, (416, 240), interpolation=cv2.INTER_AREA)
            cv2.rectangle(panel, (0, 0), (416, 34), (0, 0, 0), -1)
            cv2.putText(
                panel,
                f"{label} f{frame:03d}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)
        tiles.append(cv2.hconcat(panels))
    blank = np.zeros_like(tiles[0])
    while len(tiles) % 2:
        tiles.append(blank)
    return cv2.vconcat(
        [cv2.hconcat(tiles[index : index + 2]) for index in range(0, len(tiles), 2)]
    )


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
    if any(
        not 0 <= start < end < args.expected_frames
        for start, end in windows
    ):
        raise ValueError("appearance windows must be valid inclusive intervals")
    if any(left[1] >= right[0] for left, right in zip(windows, windows[1:])):
        raise ValueError("appearance windows must not overlap")
    for pixels, name in (
        (args.limb_dilation_pixels, "limb dilation"),
        (args.torso_erosion_pixels, "torso erosion"),
        (args.contact_dilation_pixels, "contact dilation"),
        (args.state_boundary_erosion_pixels, "state boundary erosion"),
    ):
        if pixels < 1 or pixels % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer")
    if not 0 <= args.minimum_confidence <= 1:
        raise ValueError("minimum confidence must be in [0, 1]")
    if args.minimum_state_age < 0:
        raise ValueError("minimum state age must be non-negative")

    required_probe = {
        "frames": args.expected_frames,
        "width": args.width,
        "height": args.height,
    }
    probes = {
        name: _probe(cv2, paths[name])
        for name in ("candidate", "incumbent")
    }
    for name, probe in probes.items():
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
        raise ValueError("all mask archives must cover the complete timeline")

    lossless = output / f"appearance-state-{args.mode}-lossless.mkv"
    review = output / f"appearance-state-{args.mode}.mp4"
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
    incumbent_capture = cv2.VideoCapture(str(paths["incumbent"]))
    boundary_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.state_boundary_erosion_pixels, args.state_boundary_erosion_pixels),
    )
    previous_candidate = previous_incumbent = previous_state = None
    previous_editable = previous_age = None
    previous_window: tuple[int, int] | None = None
    frozen_global_mean: Any = None
    per_frame = []
    active_rows: dict[int, tuple[Any, Any, Any]] = {}
    correction_metrics = []
    baseline_warp = []
    repaired_warp = []
    confidence_coverage = []
    started = time.perf_counter()
    for index in range(args.expected_frames):
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
        core = (
            window is not None
            and window[0] + args.edge_ramp_frames + 1
            <= index
            <= window[1] - args.edge_ramp_frames - 1
        )
        repaired = candidate.copy()
        frame_metrics: dict[str, Any] = {
            "frame": index,
            "window": list(window) if window is not None else None,
            "core": core,
            "editable_fraction": float(np.mean(editable)),
        }
        same_chain = core and window == previous_window
        if (
            same_chain
            and previous_candidate is not None
            and previous_incumbent is not None
            and previous_state is not None
            and previous_editable is not None
            and previous_age is not None
        ):
            flow = bidirectional_flow_state(
                cv2,
                np,
                previous_incumbent,
                incumbent,
                scale=args.flow_scale,
            )
            warped_previous_candidate = warp_with_flow(
                cv2, previous_candidate, flow
            )
            warped_previous_incumbent = warp_with_flow(
                cv2, previous_incumbent, flow
            )
            warped_previous_state = warp_with_flow(cv2, previous_state, flow)
            warped_editable = warp_with_flow(
                cv2,
                previous_editable.astype(np.uint8),
                flow,
                nearest=True,
            ) > 0
            warped_age = warp_with_flow(
                cv2,
                previous_age.astype(np.float32),
                flow,
                nearest=True,
            )
            base_overlap = editable & warped_editable
            metric_mask = (
                interior
                & warped_editable
                & (flow.confidence >= args.minimum_confidence)
            )
            reliable = metric_mask & (
                warped_age >= args.minimum_state_age
            )
            confidence_coverage.append(
                float(np.count_nonzero(metric_mask))
                / max(1, int(np.count_nonzero(editable)))
            )
            current_layer = (
                candidate.astype(np.float32) - incumbent.astype(np.float32)
            )
            warped_candidate_layer = (
                warped_previous_candidate.astype(np.float32)
                - warped_previous_incumbent.astype(np.float32)
            )
            if np.any(metric_mask):
                baseline_value = float(
                    np.mean(
                        np.abs(
                            current_layer[metric_mask]
                            - warped_candidate_layer[metric_mask]
                        )
                    )
                )
                baseline_warp.append(baseline_value)
                frame_metrics["baseline_layer_warp_mae"] = baseline_value

            if args.mode == "global":
                if frozen_global_mean is None and np.any(interior):
                    frozen_global_mean = np.mean(
                        current_layer[interior],
                        axis=0,
                    )
                if frozen_global_mean is not None and np.any(interior):
                    current_mean = np.mean(current_layer[interior], axis=0)
                    correction = np.clip(
                        frozen_global_mean - current_mean,
                        -args.maximum_residual_delta,
                        args.maximum_residual_delta,
                    )
                    repaired_float = candidate.astype(np.float32)
                    repaired_float[interior] += (
                        args.strength * correction
                    )
                    repaired = np.clip(
                        np.rint(repaired_float), 0, 255
                    ).astype(np.uint8)
                    immutable = np.logical_not(
                        np.asarray(editable, dtype=np.bool_)
                    )
                    repaired[immutable] = candidate[immutable]
            elif args.mode in {"naive", "gated"}:
                mode_reliable = base_overlap if args.mode == "naive" else reliable
                mode_confidence = (
                    np.ones_like(flow.confidence)
                    if args.mode == "naive"
                    else flow.confidence
                )
                repaired, applied = residual_state_update(
                    cv2,
                    np,
                    current_incumbent=incumbent,
                    current_candidate=candidate,
                    warped_previous_incumbent=warped_previous_incumbent,
                    warped_previous_state=warped_previous_state,
                    confidence=mode_confidence,
                    reliable=mode_reliable,
                    strength=args.strength,
                    gaussian_sigma=args.gaussian_sigma,
                    maximum_residual_delta=args.maximum_residual_delta,
                )
                correction_metrics.append(applied)
                frame_metrics["applied"] = applied

            repaired_layer = (
                repaired.astype(np.float32) - incumbent.astype(np.float32)
            )
            warped_state_layer = (
                warped_previous_state.astype(np.float32)
                - warped_previous_incumbent.astype(np.float32)
            )
            if np.any(metric_mask):
                repaired_value = float(
                    np.mean(
                        np.abs(
                            repaired_layer[metric_mask]
                            - warped_state_layer[metric_mask]
                        )
                    )
                )
                repaired_warp.append(repaired_value)
                frame_metrics["repaired_layer_warp_mae"] = repaired_value
            valid_age = base_overlap & (
                flow.confidence >= args.minimum_confidence
            )
            age = np.where(
                editable,
                np.where(valid_age, warped_age + 1.0, 1.0),
                0.0,
            ).astype(np.float32)
        else:
            age = editable.astype(np.float32)
            if window != previous_window:
                frozen_global_mean = None

        immutable = np.logical_not(np.asarray(editable, dtype=np.bool_))
        repaired[immutable] = candidate[immutable]
        lossless_writer.stdin.write(repaired.tobytes())
        review_writer.stdin.write(repaired.tobytes())
        if window is not None:
            active_rows[index] = (
                incumbent.copy(),
                candidate.copy(),
                repaired.copy(),
            )
        per_frame.append(frame_metrics)
        previous_candidate = candidate
        previous_incumbent = incumbent
        previous_state = repaired
        previous_editable = editable
        previous_age = age
        previous_window = window if core else None
    candidate_capture.release()
    incumbent_capture.release()
    lossless_log = _finish(lossless_writer, "lossless")
    review_log = _finish(review_writer, "review")
    wall_seconds = time.perf_counter() - started

    lossless_probe = _probe(cv2, lossless)
    if lossless_probe["frames"] != args.expected_frames:
        raise RuntimeError(f"lossless output frame mismatch: {lossless_probe}")

    decoded = cv2.VideoCapture(str(lossless))
    original = cv2.VideoCapture(str(paths["candidate"]))
    outside_exact = outside_total = outside_window_exact = outside_window_total = 0
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
        immutable = np.logical_not(np.asarray(editable, dtype=np.bool_))
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

    ranked = sorted(
        (
            row
            for row in per_frame
            if "baseline_layer_warp_mae" in row
        ),
        key=lambda row: float(row["baseline_layer_warp_mae"]),
        reverse=True,
    )
    review_frames = sorted(
        set(
            [start for start, _ in windows]
            + [end for _, end in windows]
            + [int(row["frame"]) for row in ranked[:12]]
        )
    )
    sheet = _sheet(
        cv2,
        np,
        [
            (frame, *active_rows[frame])
            for frame in review_frames
            if frame in active_rows
        ],
    )
    sheet_path = output / "review" / "incumbent-before-after.jpg"
    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"failed to write review sheet: {sheet_path}")
    frame_metrics_path = output / "frame-metrics.json"
    frame_metrics_path.write_text(
        json.dumps(per_frame, indent=2, sort_keys=True) + "\n"
    )
    metrics = {
        "baseline_layer_warp_mae": _summary(np, baseline_warp),
        "repaired_layer_warp_mae": _summary(np, repaired_warp),
        "relative_mean_layer_warp_reduction": (
            1.0 - float(np.mean(repaired_warp)) / float(np.mean(baseline_warp))
            if baseline_warp and repaired_warp and float(np.mean(baseline_warp)) > 0
            else None
        ),
        "confidence_coverage_of_editable": _summary(
            np, confidence_coverage
        ),
        "mean_applied_correction": _summary(
            np,
            [
                float(row["mean_abs_applied_correction"])
                for row in correction_metrics
            ],
        ),
        "postdecode_outside_editable_exact_fraction": (
            outside_exact / outside_total
        ),
        "postdecode_outside_windows_exact_fraction": (
            outside_window_exact / outside_window_total
        ),
        "postdecode_window_endpoints_exact_fraction": (
            endpoint_exact / endpoint_total
        ),
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
        "flicker_non_regression": (
            metrics["relative_mean_layer_warp_reduction"] is not None
            and metrics["relative_mean_layer_warp_reduction"] >= 0
        ),
    }
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "decision": "AWAITING_MATCHED_ABLATION_AND_NATIVE_REVIEW",
        "method": "confidence_gated_low_frequency_appearance_residual_state",
        "mode": args.mode,
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
            "minimum_state_age": args.minimum_state_age,
            "strength": args.strength,
            "gaussian_sigma": args.gaussian_sigma,
            "maximum_residual_delta": args.maximum_residual_delta,
            "mask_projection": args.mask_projection,
            "seed": args.seed,
        },
        "metrics": metrics,
        "gates": gates,
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
            "The state follows incumbent optical correspondence, not metric robot geometry.",
            "Low-confidence, hand, flower, contact, and visibility-boundary pixels are never repaired.",
            "Appearance stabilization cannot repair finger topology, contact physics, or hidden geometry.",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "mode": args.mode,
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
