#!/usr/bin/env python3
"""Insert JoyAI proposals only inside locked flower-task edit support."""

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

from phiagent.rendering.joyai_video_edit import (  # noqa: E402
    JOYAI_MODEL_ID,
    JOYAI_MODEL_REVISION,
    TIMELINE_FRAME,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--incumbent-video", type=Path, required=True)
    parser.add_argument("--person-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument(
        "--window",
        nargs=3,
        action="append",
        required=True,
        metavar=("START", "END", "JOYAI_VIDEO"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--source-width", type=int, default=1280)
    parser.add_argument("--source-height", type=int, default=720)
    parser.add_argument("--proposal-width", type=int, default=1248)
    parser.add_argument("--proposal-height", type=int, default=720)
    parser.add_argument(
        "--proposal-transform",
        choices=("center_crop_inverse_no_rescale", "isotropic_scale"),
        default="center_crop_inverse_no_rescale",
    )
    parser.add_argument("--proposal-crop-left", type=int, default=16)
    parser.add_argument("--proposal-crop-top", type=int, default=0)
    parser.add_argument(
        "--mask-projection",
        choices=("legacy_832x480_to_native_1280x720", "source_native"),
        default="legacy_832x480_to_native_1280x720",
    )
    parser.add_argument("--edge-ramp-frames", type=int, default=4)
    parser.add_argument("--person-dilation-pixels", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser


def temporal_weight(index: int, start: int, end: int, ramp: int) -> float:
    """Endpoint-exact trapezoid used to join a proposal without latent guessing."""

    if not 0 <= start <= end or ramp < 1:
        raise ValueError("invalid temporal blend bounds")
    if not start <= index <= end:
        return 0.0
    if index in (start, end):
        return 0.0
    return min(1.0, (index - start) / ramp, (end - index) / ramp)


def isotropic_proposal_to_source(
    *,
    source_width: int,
    source_height: int,
    proposal_width: int,
    proposal_height: int,
) -> dict[str, str]:
    dimensions = (source_width, source_height, proposal_width, proposal_height)
    if min(dimensions) <= 0:
        raise ValueError("source and proposal dimensions must be positive")
    if source_width * proposal_height != proposal_width * source_height:
        raise ValueError(
            "proposal-to-source projection requires an explicit isotropic scale "
            "between equal aspect ratios"
        )
    return {
        "kind": "isotropic_rational_scale",
        "x_source": f"x_joyai * ({source_width}/{proposal_width})",
        "y_source": f"y_joyai * ({source_height}/{proposal_height})",
        "interpolation": "Lanczos4",
    }


def center_crop_proposal_to_source(
    *,
    source_width: int,
    source_height: int,
    proposal_width: int,
    proposal_height: int,
    crop_left: int,
    crop_top: int,
) -> dict[str, Any]:
    dimensions = (source_width, source_height, proposal_width, proposal_height)
    if min(dimensions) <= 0 or min(crop_left, crop_top) < 0:
        raise ValueError("source/proposal dimensions and crop offsets are invalid")
    if crop_left + proposal_width > source_width:
        raise ValueError("proposal crop exceeds source width")
    if crop_top + proposal_height > source_height:
        raise ValueError("proposal crop exceeds source height")
    return {
        "kind": "center_crop_inverse_no_rescale",
        "x_source": f"x_joyai + {crop_left}",
        "y_source": f"y_joyai + {crop_top}",
        "crop_left_px": crop_left,
        "crop_top_px": crop_top,
        "interpolation": "none",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_all(cv2: Any, path: Path) -> tuple[list[Any], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not decode video: {path}")
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
        raise RuntimeError(f"video has no frames: {path}")
    return frames, info


def _load_packed(np: Any, path: Path) -> tuple[Any, int, int, str]:
    payload = np.load(path, allow_pickle=False)
    return payload["packed"], int(payload["height"]), int(payload["width"]), str(payload["bitorder"])


def _unpack(np: Any, payload: tuple[Any, int, int, str], index: int) -> Any:
    packed, height, width, bitorder = payload
    flat = np.unpackbits(packed[index], bitorder=bitorder)[: height * width]
    return flat.reshape(height, width).astype(np.uint8)


def _mask_to_native(
    cv2: Any,
    np: Any,
    mask: Any,
    *,
    width: int,
    height: int,
    projection: str,
) -> Any:
    if projection == "source_native":
        if mask.shape != (height, width):
            raise ValueError(
                f"native packed mask must be {width}x{height}, "
                f"received {mask.shape[1]}x{mask.shape[0]}"
            )
        return mask > 0
    if projection == "legacy_832x480_to_native_1280x720":
        if (width, height) != (1280, 720) or mask.shape != (480, 832):
            raise ValueError(
                "legacy mask projection requires packed 832x480 masks and a "
                "1280x720 source"
            )
        canvas = np.zeros((480, 854), dtype=np.uint8)
        canvas[:, 11:843] = mask
        return cv2.resize(canvas, (1280, 720), interpolation=cv2.INTER_NEAREST) > 0
    raise ValueError(f"unsupported mask projection: {projection}")


def _encoder(
    ffmpeg: Path,
    output: Path,
    fps: float,
    *,
    width: int,
    height: int,
    lossless: bool,
) -> Any:
    codec = (
        ["-c:v", "ffv1", "-level", "3"]
        if lossless
        else ["-c:v", "libx264", "-preset", "medium", "-crf", "8", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    )
    return subprocess.Popen(
        [
            str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", *codec,
            str(output),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _finish(process: Any, label: str) -> str:
    assert process.stdin is not None
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    code = process.wait()
    if code:
        raise RuntimeError(f"{label} encoder returned {code}: {stderr}")
    return stderr


def _git_state() -> dict[str, Any]:
    result = {}
    for label, command in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        result[label] = {"returncode": completed.returncode, "stdout": completed.stdout.strip()}
    return result


def main() -> int:
    args = _parser().parse_args()
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("JoyAI compositor requires the optional numpy/OpenCV runtime") from exc

    fixed = {
        "source_video": args.source_video.expanduser().resolve(),
        "incumbent_video": args.incumbent_video.expanduser().resolve(),
        "person_masks": args.person_masks.expanduser().resolve(),
        "flower_masks": args.flower_masks.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    for label, path in fixed.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"JoyAI composition experiment already exists: {output}")
    if args.person_dilation_pixels < 1 or args.person_dilation_pixels % 2 == 0:
        raise ValueError("person dilation must be a positive odd integer")
    if args.edge_ramp_frames < 1:
        raise ValueError("edge ramp must be positive")
    if args.proposal_transform == "center_crop_inverse_no_rescale":
        proposal_to_source = center_crop_proposal_to_source(
            source_width=args.source_width,
            source_height=args.source_height,
            proposal_width=args.proposal_width,
            proposal_height=args.proposal_height,
            crop_left=args.proposal_crop_left,
            crop_top=args.proposal_crop_top,
        )
    else:
        proposal_to_source = isotropic_proposal_to_source(
            source_width=args.source_width,
            source_height=args.source_height,
            proposal_width=args.proposal_width,
            proposal_height=args.proposal_height,
        )
    output.mkdir(parents=True)
    (output / "review").mkdir()

    source_frames, source_info = _decode_all(cv2, fixed["source_video"])
    incumbent_frames, incumbent_info = _decode_all(cv2, fixed["incumbent_video"])
    required_info = {
        "frames": args.expected_frames,
        "width": args.source_width,
        "height": args.source_height,
    }
    for label, info in (("source", source_info), ("incumbent", incumbent_info)):
        if any(info[key] != value for key, value in required_info.items()):
            raise ValueError(f"{label} video does not match full-timeline contract: {info}")
        if abs(info["fps"] - args.fps) > 0.01:
            raise ValueError(f"{label} FPS {info['fps']} != {args.fps}")

    routes = []
    for raw_start, raw_end, raw_path in args.window:
        start, end = int(raw_start), int(raw_end)
        proposal_path = Path(raw_path).expanduser().resolve()
        if not 0 <= start <= end < args.expected_frames or not proposal_path.is_file():
            raise ValueError(f"invalid JoyAI route: {raw_start}, {raw_end}, {proposal_path}")
        if (end - start) % 8:
            raise ValueError("JoyAI route length must satisfy frame_count = 1 + 8n")
        proposal, info = _decode_all(cv2, proposal_path)
        if info["frames"] != end - start + 1 or (
            info["width"],
            info["height"],
        ) != (args.proposal_width, args.proposal_height):
            raise ValueError(f"JoyAI proposal does not match its route: {info}")
        routes.append({"start": start, "end": end, "path": proposal_path, "frames": proposal, "probe": info})
    routes.sort(key=lambda row: row["start"])
    if any(left["end"] >= right["start"] for left, right in zip(routes, routes[1:])):
        raise ValueError("JoyAI routes must not overlap")

    person_payload = _load_packed(np, fixed["person_masks"])
    flower_payload = _load_packed(np, fixed["flower_masks"])
    if len(person_payload[0]) != args.expected_frames or len(flower_payload[0]) != args.expected_frames:
        raise ValueError("packed masks must cover every full-timeline frame")

    lossless = output / "joyai-flower-repaired-27p5s-lossless.mkv"
    review = output / "joyai-flower-repaired-27p5s-native.mp4"
    lossless_writer = _encoder(
        fixed["ffmpeg"],
        lossless,
        args.fps,
        width=args.source_width,
        height=args.source_height,
        lossless=True,
    )
    review_writer = _encoder(
        fixed["ffmpeg"],
        review,
        args.fps,
        width=args.source_width,
        height=args.source_height,
        lossless=False,
    )
    assert lossless_writer.stdin is not None and review_writer.stdin is not None

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.person_dilation_pixels, args.person_dilation_pixels)
    )
    route_by_frame = {
        index: route
        for route in routes
        for index in range(int(route["start"]), int(route["end"]) + 1)
    }
    edited_pixels = editable_pixels = 0
    flower_exact = flower_total = 0
    background_exact = background_total = 0
    outside_exact = outside_total = 0
    endpoint_exact = endpoint_total = 0
    transition_deltas = []
    previous = None
    review_rows = []
    started = time.perf_counter()
    for index in range(args.expected_frames):
        source = source_frames[index]
        incumbent = incumbent_frames[index]
        flower = _mask_to_native(
            cv2,
            np,
            _unpack(np, flower_payload, index),
            width=args.source_width,
            height=args.source_height,
            projection=args.mask_projection,
        )
        person = _mask_to_native(
            cv2,
            np,
            _unpack(np, person_payload, index),
            width=args.source_width,
            height=args.source_height,
            projection=args.mask_projection,
        )
        support = cv2.dilate(person.astype(np.uint8) * 255, kernel) > 0
        editable = np.logical_and(support, np.logical_not(flower))
        background = np.logical_and(np.logical_not(support), np.logical_not(flower))
        composed = incumbent.copy()
        route = route_by_frame.get(index)
        weight = 0.0
        if route is not None:
            start, end = int(route["start"]), int(route["end"])
            weight = temporal_weight(index, start, end, args.edge_ramp_frames)
            proposal_frame = route["frames"][index - start]
            if args.proposal_transform == "isotropic_scale":
                proposal = cv2.resize(
                    proposal_frame,
                    (args.source_width, args.source_height),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            else:
                proposal = incumbent.copy()
                left = args.proposal_crop_left
                top = args.proposal_crop_top
                proposal[
                    top : top + args.proposal_height,
                    left : left + args.proposal_width,
                ] = proposal_frame
            alpha_mask = editable
            if weight > 0:
                alpha = cv2.GaussianBlur(alpha_mask.astype(np.uint8) * 255, (0, 0), 2.0).astype(np.float32) / 255.0
                alpha *= weight
                composed = np.rint(
                    proposal.astype(np.float32) * alpha[..., None]
                    + incumbent.astype(np.float32) * (1.0 - alpha[..., None])
                ).astype(np.uint8)
        # Projection makes the foundation model incapable of changing these states.
        composed[background] = source[background]
        composed[flower] = source[flower]

        delta_source = np.max(np.abs(composed.astype(np.int16) - source.astype(np.int16)), axis=2)
        delta_incumbent = np.max(np.abs(composed.astype(np.int16) - incumbent.astype(np.int16)), axis=2)
        flower_exact += int(np.count_nonzero(delta_source[flower] == 0))
        flower_total += int(np.count_nonzero(flower))
        background_exact += int(np.count_nonzero(delta_source[background] == 0))
        background_total += int(np.count_nonzero(background))
        edited_pixels += int(np.count_nonzero((delta_incumbent >= 2) & editable))
        editable_pixels += int(np.count_nonzero(editable))
        if route is None:
            outside_exact += int(np.count_nonzero(delta_incumbent == 0))
            outside_total += delta_incumbent.size
        if route is not None and index in (int(route["start"]), int(route["end"])):
            endpoint_exact += int(np.count_nonzero(delta_incumbent == 0))
            endpoint_total += delta_incumbent.size
        if previous is not None:
            transition_deltas.append(
                float(np.mean(np.abs(composed.astype(np.float32) - previous.astype(np.float32))))
            )
        previous = composed
        lossless_writer.stdin.write(composed.tobytes())
        review_writer.stdin.write(composed.tobytes())
        if route is not None and (index % 4 == 0 or index in (int(route["start"]), int(route["end"]))):
            panel = np.concatenate(
                [
                    cv2.resize(incumbent, (480, 270), interpolation=cv2.INTER_AREA),
                    cv2.resize(composed, (480, 270), interpolation=cv2.INTER_AREA),
                ],
                axis=1,
            )
            cv2.putText(panel, f"frame {index} weight {weight:.2f} incumbent | JoyAI locked", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            review_rows.append(panel)
    lossless_stderr = _finish(lossless_writer, "lossless")
    review_stderr = _finish(review_writer, "review")
    wall_seconds = time.perf_counter() - started

    sheet = np.concatenate(review_rows, axis=0) if review_rows else np.zeros((270, 960, 3), np.uint8)
    sheet_path = output / "review/joyai-window-comparison.jpg"
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 96])
    transitions = np.asarray(transition_deltas, dtype=np.float64)
    boundary_frames = sorted(
        {frame for route in routes for frame in (int(route["start"]), int(route["end"]) + 1) if 0 < frame < args.expected_frames}
    )
    boundary_deltas = {str(frame): float(transitions[frame - 1]) for frame in boundary_frames}
    metrics = {
        "frames": args.expected_frames,
        "video_seconds": args.expected_frames / args.fps,
        "compositor_wall_seconds": wall_seconds,
        "compositor_fps": args.expected_frames / wall_seconds,
        "flower_exact_fraction": flower_exact / max(flower_total, 1),
        "native_background_exact_fraction": background_exact / max(background_total, 1),
        "outside_windows_incumbent_exact_fraction": outside_exact / max(outside_total, 1),
        "window_endpoint_incumbent_exact_fraction": endpoint_exact / max(endpoint_total, 1),
        "editable_support_changed_fraction": edited_pixels / max(editable_pixels, 1),
        "frame_transition_delta_p95": float(np.percentile(transitions, 95)),
        "route_boundary_transition_deltas": boundary_deltas,
    }
    gates = {
        "flower_pixels_locked": metrics["flower_exact_fraction"] == 1.0,
        "native_background_locked": metrics["native_background_exact_fraction"] == 1.0,
        "outside_windows_locked": metrics["outside_windows_incumbent_exact_fraction"] == 1.0,
        "window_endpoints_locked": metrics["window_endpoint_incumbent_exact_fraction"] == 1.0,
        "proposal_changed_editable_support": metrics["editable_support_changed_fraction"] > 0.0,
        "route_boundaries_below_global_p95": all(
            value <= metrics["frame_transition_delta_p95"] for value in boundary_deltas.values()
        ),
    }
    packages = {}
    for name in ("numpy", "opencv-python", "opencv-python-headless"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "reason": "JoyAI proposal is generated and state-locked but still requires adversarial and native-resolution review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "command": sys.argv,
        "git": _git_state(),
        "packages": packages,
        "seed": args.seed,
        "model": {"id": JOYAI_MODEL_ID, "revision": JOYAI_MODEL_REVISION, "authority": "proposal_only"},
        "coordinate_frames": {
            "source": (
                f"camera:source_native_{args.source_width}x{args.source_height}"
            ),
            "joyai": (
                f"camera:joyai_model_{args.proposal_width}x{args.proposal_height}"
            ),
            "timeline": TIMELINE_FRAME,
            "joyai_to_source": proposal_to_source,
            "packed_masks_to_source": {
                "kind": args.mask_projection,
                "legacy_canvas": (
                    {
                        "input": "camera:source_aligned_832x480",
                        "padded_canvas": "camera:source_aligned_854x480",
                        "insert_x_range_half_open": [11, 843],
                        "output": "camera:source_native_1280x720",
                        "interpolation": "nearest",
                    }
                    if args.mask_projection == "legacy_832x480_to_native_1280x720"
                    else None
                ),
            },
        },
        "inputs": {label: {"path": str(path), "sha256": _sha256(path)} for label, path in fixed.items()},
        "routes": [
            {
                "range_inclusive": [int(route["start"]), int(route["end"])],
                "path": str(route["path"]),
                "sha256": _sha256(route["path"]),
                "probe": route["probe"],
            }
            for route in routes
        ],
        "metrics": metrics,
        "deterministic_gates": gates,
        "outputs": {
            "lossless": {"path": str(lossless), "sha256": _sha256(lossless)},
            "review": {"path": str(review), "sha256": _sha256(review)},
            "comparison_sheet": {"path": str(sheet_path), "sha256": _sha256(sheet_path)},
        },
        "encoder_logs": {"lossless": lossless_stderr, "review": review_stderr},
        "physical_evidence": False,
        "promotion_status": "NOT_EVALUATED",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"experiment": str(output), "status": manifest["status"], "metrics": metrics, "gates": gates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
