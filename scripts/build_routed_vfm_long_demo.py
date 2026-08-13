#!/usr/bin/env python3
"""Route reviewed VFM windows into a source-locked long-video demo.

This is a perceptual compositor, not a physical-state estimator.  The source
video owns the background and flowers.  A full-length incumbent supplies the
default robot layer, while independently reviewed foundation-model windows
replace only bounded time ranges inside the source-person support.  Later
windows have higher priority and enter/leave through deterministic ramps.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.agent.perceptual_video_harness import foundation_model_roles  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--person-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--pose-limbs", type=Path, required=True)
    parser.add_argument(
        "--window",
        nargs=3,
        action="append",
        metavar=("START", "END", "VIDEO"),
        required=True,
        help="reviewed inclusive global frame range and its exact video",
    )
    parser.add_argument("--review-frame", type=int, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--route-ramp-frames", type=int, default=6)
    parser.add_argument(
        "--optimize-hard-route-switches",
        action="store_true",
        help=(
            "For zero-ramp routing, choose each switch from its valid overlap by "
            "minimizing source-aligned appearance and temporal discontinuity."
        ),
    )
    parser.add_argument("--person-dilation-pixels", type=int, default=51)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_strength(
    index: int,
    start: int,
    end: int,
    ramp_frames: int,
    final_frame: int,
) -> float:
    """Return the bounded route weight for one inclusive reviewed window."""

    if ramp_frames < 0 or not 0 <= start <= end <= final_frame:
        raise ValueError("invalid route bounds or ramp")
    if not start <= index <= end:
        return 0.0
    incoming = 1.0 if start == 0 else min(1.0, (index - start + 1) / (ramp_frames + 1))
    outgoing = 1.0 if end == final_frame else min(1.0, (end - index + 1) / (ramp_frames + 1))
    return min(incoming, outgoing)


def switch_search_interval(routes: list[dict[str, Any]], route_index: int) -> tuple[int, int]:
    """Return the interval in which one route can take over without a gap."""

    route = routes[route_index]
    upper = int(route["end"])
    if route_index + 1 < len(routes):
        upper = min(upper, int(routes[route_index + 1]["start"]) - 1)
    if route_index:
        upper = min(upper, int(routes[route_index - 1]["end"]) + 1)
    lower = int(route["start"])
    if upper < lower:
        raise ValueError(
            f"route {route_index} has no continuity-safe switch interval: {lower}>{upper}"
        )
    return lower, upper


def _git_state() -> dict[str, Any]:
    result = {}
    for label, command in {
        "head": ["git", "rev-parse", "--verify", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        result[label] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    return result


def _package_versions() -> dict[str, str | None]:
    result = {}
    for name in ("numpy", "opencv-python", "opencv-python-headless"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _load_packed(np: Any, path: Path) -> tuple[Any, int, int, str]:
    payload = np.load(path, allow_pickle=False)
    return (
        payload["packed"],
        int(payload["height"]),
        int(payload["width"]),
        str(payload["bitorder"]),
    )


def _unpack(np: Any, payload: tuple[Any, int, int, str], index: int) -> Any:
    packed, height, width, bitorder = payload
    flat = np.unpackbits(packed[index], bitorder=bitorder)[: height * width]
    return flat.reshape(height, width).astype(np.uint8)


def _mask_to_native(cv2: Any, np: Any, mask: Any) -> Any:
    if mask.shape != (480, 832):
        raise ValueError(f"packed masks must be 832x480, received {mask.shape[::-1]}")
    canvas = np.zeros((480, 854), dtype=np.uint8)
    canvas[:, 11:843] = mask
    return cv2.resize(canvas, (1280, 720), interpolation=cv2.INTER_NEAREST) > 0


def _locked_and_editable_regions(np: Any, support: Any, flower: Any) -> tuple[Any, Any]:
    """Build disjoint audit regions without mutating either input mask.

    Explicit logical ufuncs are intentional here.  They avoid temporary-array
    reuse around chained ``~``/``&`` expressions in the Python 3.14 + NumPy
    runtime used by the experiment and make the lock contract auditable.
    """

    support_bool = np.asarray(support, dtype=np.bool_).copy()
    flower_bool = np.asarray(flower, dtype=np.bool_).copy()
    not_flower = np.logical_not(flower_bool)
    background = np.logical_and(np.logical_not(support_bool), not_flower)
    editable = np.logical_and(support_bool, not_flower)
    return background, editable


def _points_to_native(points: Any) -> Any:
    result = points.astype("float32").copy()
    result[..., 0] = (result[..., 0] + 11.0) * (1280.0 / 854.0)
    result[..., 1] *= 720.0 / 480.0
    return result


def _decode_all(cv2: Any, path: Path) -> tuple[list[Any], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not decode {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    info = {
        "frames": len(frames),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if not frames:
        raise RuntimeError(f"video has no decodable frames: {path}")
    return frames, info


def _encoder(ffmpeg: Path, output: Path, fps: float, *, lossless: bool) -> Any:
    codec = (
        ["-c:v", "ffv1", "-level", "3"]
        if lossless
        else ["-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p"]
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
            "1280x720",
            "-r",
            str(fps),
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
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if process.wait():
        raise RuntimeError(f"{label} encoder failed: {stderr[-2000:]}")
    return stderr


def _sheet(cv2: Any, np: Any, frames: list[tuple[int, Any]], columns: int) -> Any:
    tiles = []
    for index, frame in frames:
        tile = frame.copy()
        text = f"f{index:03d}  {index / 24.0:05.2f}s"
        cv2.putText(tile, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            tile, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        tiles.append(tile)
    if not tiles:
        raise ValueError("review frame list must not be empty")
    blank = np.zeros_like(tiles[0])
    while len(tiles) % columns:
        tiles.append(blank)
    return cv2.vconcat(
        [cv2.hconcat(tiles[index : index + columns]) for index in range(0, len(tiles), columns)]
    )


def _postencode_lock_audit(
    cv2: Any,
    np: Any,
    source_path: Path,
    candidate_path: Path,
    flower_payload: tuple[Any, int, int, str],
    person_payload: tuple[Any, int, int, str],
    frames: int,
    person_kernel: Any,
) -> dict[str, float]:
    """Re-decode the lossless artifact and measure locks independently."""

    source = cv2.VideoCapture(str(source_path))
    candidate = cv2.VideoCapture(str(candidate_path))
    if not source.isOpened() or not candidate.isOpened():
        raise RuntimeError("post-encode audit could not open source/candidate")
    flower_exact = flower_total = background_exact = background_total = 0
    changed = changed_total = decoded = 0
    try:
        for index in range(frames):
            source_ok, source_frame = source.read()
            candidate_ok, candidate_frame = candidate.read()
            if not source_ok or not candidate_ok:
                raise RuntimeError(f"post-encode audit stopped at frame {index}")
            decoded += 1
            person = _mask_to_native(cv2, np, _unpack(np, person_payload, index))
            flower = _mask_to_native(cv2, np, _unpack(np, flower_payload, index))
            support = cv2.dilate(person.astype(np.uint8) * 255, person_kernel) > 0
            delta = np.max(
                np.abs(candidate_frame.astype(np.int16) - source_frame.astype(np.int16)), axis=2
            )
            flower_exact += int(np.count_nonzero(delta[flower] == 0))
            flower_total += int(np.count_nonzero(flower))
            background, editable = _locked_and_editable_regions(np, support, flower)
            background_exact += int(np.count_nonzero(delta[background] == 0))
            background_total += int(np.count_nonzero(background))
            changed += int(np.count_nonzero((delta >= 12) & editable))
            changed_total += int(np.count_nonzero(editable))
        if candidate.read()[0]:
            raise RuntimeError("post-encode candidate has more frames than expected")
    finally:
        source.release()
        candidate.release()
    return {
        "decoded_frames": decoded,
        "flower_exact_fraction": flower_exact / flower_total,
        "native_background_exact_fraction": background_exact / background_total,
        "editable_support_changed_fraction": changed / changed_total,
    }


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    fixed_paths = {
        "source_video": args.source_video.expanduser().resolve(),
        "base_candidate": args.base_candidate.expanduser().resolve(),
        "person_masks": args.person_masks.expanduser().resolve(),
        "flower_masks": args.flower_masks.expanduser().resolve(),
        "pose_limbs": args.pose_limbs.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    for label, path in fixed_paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{label}: {path}")
    if args.expected_frames < 2 or args.fps <= 0 or args.route_ramp_frames < 0:
        raise ValueError("invalid frame/FPS/ramp configuration")
    if args.person_dilation_pixels < 1 or args.person_dilation_pixels % 2 == 0:
        raise ValueError("person dilation must be a positive odd integer")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)
    (output / "review").mkdir()
    (output / "logs").mkdir()

    routes = []
    for priority, raw in enumerate(args.window):
        start, end, video = int(raw[0]), int(raw[1]), Path(raw[2]).expanduser().resolve()
        if not 0 <= start <= end < args.expected_frames or not video.is_file():
            raise ValueError(f"invalid reviewed window: {raw}")
        frames, info = _decode_all(cv2, video)
        if len(frames) != end - start + 1 or abs(info["fps"] - args.fps) > 0.01:
            raise ValueError(f"window timeline mismatch for {video}: {info}")
        routes.append(
            {
                "priority": priority,
                "start": start,
                "end": end,
                "video": video,
                "frames": frames,
                "info": info,
            }
        )

    source_frames, source_info = _decode_all(cv2, fixed_paths["source_video"])
    base_frames, base_info = _decode_all(cv2, fixed_paths["base_candidate"])
    if len(source_frames) != args.expected_frames or len(base_frames) != args.expected_frames:
        raise ValueError("source and base must match the expected full timeline")
    if source_info["width"] != 1280 or source_info["height"] != 720:
        raise ValueError(f"source must be native 1280x720: {source_info}")

    person_payload = _load_packed(np, fixed_paths["person_masks"])
    flower_payload = _load_packed(np, fixed_paths["flower_masks"])
    if (
        len(person_payload[0]) != args.expected_frames
        or len(flower_payload[0]) != args.expected_frames
    ):
        raise ValueError("packed masks must cover the full timeline")
    pose = np.load(fixed_paths["pose_limbs"], allow_pickle=False)
    landmarks = _points_to_native(pose["landmarks_xy"])
    landmark_ids = [int(value) for value in pose["landmark_ids"]]
    landmark_map = {value: index for index, value in enumerate(landmark_ids)}
    hand_ids = [15, 16, 17, 18, 19, 20, 21, 22]
    if any(value not in landmark_map for value in hand_ids):
        raise ValueError("pose archive lacks required wrist/hand landmarks")

    if args.optimize_hard_route_switches and args.route_ramp_frames != 0:
        raise ValueError("optimized route switches require --route-ramp-frames 0")
    route_switch_evidence = []
    if args.optimize_hard_route_switches:
        chosen_switches: list[int] = []
        switch_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.person_dilation_pixels, args.person_dilation_pixels),
        )

        def provider_frame(provider: int, index: int) -> Any:
            if provider < 0:
                raw = base_frames[index]
            else:
                provider_route = routes[provider]
                raw = provider_route["frames"][index - provider_route["start"]]
            selected = cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
            source = source_frames[index]
            person = _mask_to_native(cv2, np, _unpack(np, person_payload, index))
            flower = _mask_to_native(cv2, np, _unpack(np, flower_payload, index))
            support = cv2.dilate(person.astype(np.uint8) * 255, kernel=switch_kernel)
            alpha = cv2.GaussianBlur(support, (0, 0), 2.0).astype(np.float32) / 255.0
            alpha[flower] = 0.0
            composed = np.rint(
                selected.astype(np.float32) * alpha[..., None]
                + source.astype(np.float32) * (1.0 - alpha[..., None])
            ).astype(np.uint8)
            composed[flower] = source[flower]
            return composed

        def current_provider(index: int) -> int:
            provider = -1
            for prior_index, prior in enumerate(routes[: len(chosen_switches)]):
                if chosen_switches[prior_index] <= index <= prior["end"]:
                    provider = prior_index
            return provider

        for route_index, route in enumerate(routes):
            lower, upper = switch_search_interval(routes, route_index)
            candidates = []
            for switch in range(lower, upper + 1):
                old_provider = current_provider(switch)
                previous_provider = current_provider(switch - 1)
                old_same = provider_frame(old_provider, switch).astype(np.float32)
                old_previous = provider_frame(previous_provider, switch - 1).astype(np.float32)
                incoming = provider_frame(route_index, switch).astype(np.float32)
                person = _mask_to_native(cv2, np, _unpack(np, person_payload, switch))
                flower = _mask_to_native(cv2, np, _unpack(np, flower_payload, switch))
                support = cv2.dilate(person.astype(np.uint8) * 255, kernel=switch_kernel) > 0
                _, editable = _locked_and_editable_regions(np, support, flower)
                same_frame_l1 = float(np.mean(np.abs(incoming[editable] - old_same[editable])))
                incoming_transition_l1 = float(
                    np.mean(np.abs(incoming[editable] - old_previous[editable]))
                )
                natural_transition_l1 = float(
                    np.mean(np.abs(old_same[editable] - old_previous[editable]))
                )
                score = same_frame_l1 + abs(incoming_transition_l1 - natural_transition_l1)
                candidates.append(
                    {
                        "frame": switch,
                        "score": score,
                        "same_frame_l1": same_frame_l1,
                        "incoming_transition_l1": incoming_transition_l1,
                        "natural_transition_l1": natural_transition_l1,
                        "outgoing_provider": old_provider,
                    }
                )
            selected_switch = min(candidates, key=lambda row: (row["score"], row["frame"]))
            chosen_switches.append(int(selected_switch["frame"]))
            route["switch_start"] = int(selected_switch["frame"])
            route_switch_evidence.append(
                {
                    "route_priority": route_index,
                    "search_interval_inclusive": [lower, upper],
                    "selected": selected_switch,
                    "candidates": candidates,
                }
            )
    else:
        for route in routes:
            route["switch_start"] = route["start"]

    lossless_path = output / "routed-vfm-demo-27p5s-lossless.mkv"
    review_path = output / "routed-vfm-demo-27p5s-720p.mp4"
    lossless_writer = _encoder(fixed_paths["ffmpeg"], lossless_path, args.fps, lossless=True)
    review_writer = _encoder(fixed_paths["ffmpeg"], review_path, args.fps, lossless=False)
    assert lossless_writer.stdin is not None and review_writer.stdin is not None

    uniform_indices = set(int(value) for value in np.linspace(0, args.expected_frames - 1, 24))
    review_indices = set(args.review_frame)
    seam_indices = set()
    for route in routes:
        for anchor in (route["switch_start"], route["end"]):
            seam_indices.update(range(max(0, anchor - 3), min(args.expected_frames, anchor + 4)))
    review_indices.update(seam_indices)
    full_review = []
    hand_review = []
    route_rows = []
    transition_deltas = []
    flower_motion = []
    flower_exact = flower_total = background_exact = background_total = 0
    support_changed = support_total = 0
    previous_output = previous_source = None
    started = time.perf_counter()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.person_dilation_pixels, args.person_dilation_pixels)
    )
    for index in range(args.expected_frames):
        source = source_frames[index]
        selected = cv2.resize(
            base_frames[index], (1280, 720), interpolation=cv2.INTER_LANCZOS4
        ).astype(np.float32)
        weights = []
        for route in routes:
            weight = route_strength(
                index,
                route["switch_start"],
                route["end"],
                args.route_ramp_frames,
                args.expected_frames - 1,
            )
            if weight > 0:
                incoming = cv2.resize(
                    route["frames"][index - route["start"]],
                    (1280, 720),
                    interpolation=cv2.INTER_LANCZOS4,
                ).astype(np.float32)
                selected = selected * (1.0 - weight) + incoming * weight
            weights.append(weight)
        person = _mask_to_native(cv2, np, _unpack(np, person_payload, index))
        flower = _mask_to_native(cv2, np, _unpack(np, flower_payload, index))
        support = cv2.dilate(person.astype(np.uint8) * 255, kernel)
        alpha = cv2.GaussianBlur(support, (0, 0), 2.0).astype(np.float32) / 255.0
        alpha[flower] = 0.0
        composed = np.rint(
            selected * alpha[..., None] + source.astype(np.float32) * (1.0 - alpha[..., None])
        ).astype(np.uint8)
        composed[flower] = source[flower]

        flower_exact += int(np.count_nonzero(np.all(composed[flower] == source[flower], axis=1)))
        flower_total += int(np.count_nonzero(flower))
        protected, editable = _locked_and_editable_regions(np, support > 0, flower)
        background_exact += int(
            np.count_nonzero(np.all(composed[protected] == source[protected], axis=1))
        )
        background_total += int(np.count_nonzero(protected))
        delta = np.max(np.abs(composed.astype(np.int16) - source.astype(np.int16)), axis=2)
        support_changed += int(np.count_nonzero((delta >= 12) & editable))
        support_total += int(np.count_nonzero(editable))
        if previous_output is not None:
            transition_deltas.append(
                float(
                    np.mean(
                        np.abs(composed.astype(np.float32) - previous_output.astype(np.float32))
                    )
                )
            )
        if previous_source is not None and np.any(flower):
            flower_motion.append(
                float(
                    np.mean(
                        np.abs(
                            source[flower].astype(np.float32)
                            - previous_source[flower].astype(np.float32)
                        )
                    )
                )
            )
        previous_output = composed
        previous_source = source
        lossless_writer.stdin.write(composed.tobytes())
        review_writer.stdin.write(composed.tobytes())
        route_rows.append({"frame": index, "window_weights_by_priority": weights})

        if index in uniform_indices:
            full_review.append(
                (index, cv2.resize(composed, (480, 270), interpolation=cv2.INTER_AREA))
            )
        if index in review_indices:
            points = np.asarray(
                [landmarks[index, landmark_map[value]] for value in hand_ids], dtype=np.float32
            )
            center = np.nanmean(points, axis=0)
            if not np.all(np.isfinite(center)):
                center = np.asarray([640.0, 360.0])
            x0 = max(0, min(640, int(round(center[0])) - 320))
            y0 = max(0, min(360, int(round(center[1])) - 180))
            hand_review.append((index, composed[y0 : y0 + 360, x0 : x0 + 640]))

    lossless_stderr = _finish(lossless_writer, "lossless")
    review_stderr = _finish(review_writer, "review")
    wall_seconds = time.perf_counter() - started
    postencode = _postencode_lock_audit(
        cv2,
        np,
        fixed_paths["source_video"],
        lossless_path,
        flower_payload,
        person_payload,
        args.expected_frames,
        kernel,
    )
    full_sheet = _sheet(cv2, np, full_review, 4)
    hand_sheet = _sheet(cv2, np, hand_review, 4)
    full_sheet_path = output / "review" / "uniform-full-24.jpg"
    hand_sheet_path = output / "review" / "hands-and-seams.jpg"
    cv2.imwrite(str(full_sheet_path), full_sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(hand_sheet_path), hand_sheet, [cv2.IMWRITE_JPEG_QUALITY, 96])

    transitions = np.asarray(transition_deltas, dtype=np.float64)
    motion = np.asarray(flower_motion, dtype=np.float64)
    route_boundaries = sorted(
        {
            value
            for route in routes
            for value in (route["switch_start"], route["end"] + 1)
            if 0 < value < args.expected_frames
        }
    )
    boundary_deltas = {str(index): float(transitions[index - 1]) for index in route_boundaries}
    route_manifest = [
        {
            "priority": route["priority"],
            "global_range_inclusive": [route["start"], route["end"]],
            "selected_switch_frame": route["switch_start"],
            "video": str(route["video"]),
            "video_sha256": _sha256(route["video"]),
            "probe": route["info"],
        }
        for route in routes
    ]
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "decision": "AWAITING_HIGH_RESOLUTION_HUMAN_AND_ADVERSARIAL_REVIEW",
        "claim_scope": "perceptually plausible synthetic display data",
        "physical_evidence": False,
        "method": "reviewed_vfm_window_routing_plus_native_source_object_lock",
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _package_versions(),
        "git": _git_state(),
        "foundation_model_roles": list(foundation_model_roles()),
        "coordinate_frames": {
            "source": "camera:source_native_1280x720",
            "packed_masks": "camera:source_aligned_832x480",
            "timeline": f"absolute_frame_index:full_source_{args.expected_frames}",
        },
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in fixed_paths.items()
            if label != "ffmpeg"
        },
        "routes": route_manifest,
        "config": {
            "expected_frames": args.expected_frames,
            "fps": args.fps,
            "route_ramp_frames": args.route_ramp_frames,
            "optimize_hard_route_switches": args.optimize_hard_route_switches,
            "route_switch_evidence": route_switch_evidence,
            "person_dilation_pixels": args.person_dilation_pixels,
            "review_frames": sorted(review_indices),
        },
        "metrics": {
            "video_seconds": args.expected_frames / args.fps,
            "frames": args.expected_frames,
            "native_width": 1280,
            "native_height": 720,
            "compositor_wall_seconds": wall_seconds,
            "compositor_fps": args.expected_frames / wall_seconds,
            "compositor_realtime_factor": wall_seconds / (args.expected_frames / args.fps),
            "flower_exact_fraction_before_encode": flower_exact / flower_total,
            "native_background_exact_fraction_before_encode": background_exact / background_total,
            "editable_support_changed_fraction_before_encode": support_changed / support_total,
            "postencode_lossless_lock_audit": postencode,
            "source_flower_motion_delta_mean": float(motion.mean()),
            "source_flower_motion_delta_p05": float(np.quantile(motion, 0.05)),
            "source_flower_dynamic_frame_fraction": float(np.mean(motion >= 1.0)),
            "frame_transition_delta_mean": float(transitions.mean()),
            "frame_transition_delta_p95": float(np.quantile(transitions, 0.95)),
            "frame_transition_delta_max": float(transitions.max()),
            "route_boundary_transition_deltas": boundary_deltas,
        },
        "outputs": {
            "lossless": {"path": str(lossless_path), "sha256": _sha256(lossless_path)},
            "review_video": {"path": str(review_path), "sha256": _sha256(review_path)},
            "uniform_review": {"path": str(full_sheet_path), "sha256": _sha256(full_sheet_path)},
            "hand_and_seam_review": {
                "path": str(hand_sheet_path),
                "sha256": _sha256(hand_sheet_path),
            },
        },
        "limitations": [
            "Reviewed windows and masks are image-space evidence, not metric geometry or contact force.",
            "Source flowers and their response motion are preserved rather than predicted by the robot model.",
            "High-resolution human review is a hard veto before DISPLAY_READY.",
            "No depth, force, telemetry, force closure, or real-robot executability is claimed.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "routing.json").write_text(json.dumps(route_rows, indent=2) + "\n")
    (output / "logs" / "encoders.json").write_text(
        json.dumps({"lossless": lossless_stderr, "review": review_stderr}, indent=2) + "\n"
    )
    shutil.copy2(Path(__file__).resolve(), output / "execution-source.py")
    print(json.dumps({"output": str(output), **manifest["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
