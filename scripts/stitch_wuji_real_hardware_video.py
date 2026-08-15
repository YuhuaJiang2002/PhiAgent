#!/usr/bin/env python3
"""Stitch Wan windows and lock a real-reference Wuji replacement to the scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_frame_for_output(
    output_index: int, *, output_fps: float, source_fps: float, source_frames: int
) -> int:
    if output_index < 0 or min(output_fps, source_fps, source_frames) <= 0:
        raise ValueError("frame indices, frame rates, and frame counts must be valid")
    return min(source_frames - 1, int(round(output_index * source_fps / output_fps)))


def alpha_from_mask(cv2: Any, np: Any, mask: Any, *, dilation: int, feather: int) -> Any:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = np.where(mask >= 128, 255, 0).astype(np.uint8)
    if dilation:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1)
        )
        binary = cv2.dilate(binary, kernel)
    if feather:
        kernel_size = 2 * feather + 1
        binary = cv2.GaussianBlur(binary, (kernel_size, kernel_size), 0)
    return binary.astype(np.float32) / 255.0


def composite_under_alpha(np: Any, foreground: Any, background: Any, alpha: Any) -> Any:
    if foreground.shape != background.shape:
        raise ValueError("foreground and background shapes must match")
    if alpha.shape != foreground.shape[:2]:
        raise ValueError("alpha shape must match the frame")
    weight = alpha[..., None]
    return np.clip(
        np.rint(foreground.astype(np.float32) * weight + background * (1.0 - weight)),
        0,
        255,
    ).astype(np.uint8)


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    info: dict[str, float | int] = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "reported_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    info["decoded_frames"] = len(frames)
    return frames, info


def _transition_cost(np: Any, first: Any, second: Any) -> float:
    height, width = first.shape[:2]
    y0, y1 = round(0.05 * height), round(0.98 * height)
    x0, x1 = round(0.30 * width), round(0.98 * width)
    full = float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))))
    subject = float(
        np.mean(
            np.abs(
                first[y0:y1, x0:x1].astype(np.float32)
                - second[y0:y1, x0:x1].astype(np.float32)
            )
        )
    )
    return full + 2.0 * subject


def merge_windows(
    np: Any, ordered: list[tuple[int, list[Any]]], *, blend_radius: int
) -> tuple[list[Any], list[dict[str, float | int]]]:
    if not ordered:
        raise ValueError("at least one window is required")
    if blend_radius < 0:
        raise ValueError("blend_radius must be non-negative")
    start, merged = ordered[0][0], list(ordered[0][1])
    seams: list[dict[str, float | int]] = []
    for following_start, following in ordered[1:]:
        overlap_start = following_start
        overlap_end = min(start + len(merged), following_start + len(following))
        candidates = range(overlap_start + 1, overlap_end)
        scored = [
            (
                _transition_cost(
                    np,
                    merged[index - 1 - start],
                    following[index - following_start],
                ),
                index,
            )
            for index in candidates
        ]
        if not scored:
            raise ValueError("adjacent windows do not overlap")
        cost, seam = min(scored)
        blend_start = max(overlap_start, seam - blend_radius)
        blend_end = min(overlap_end, seam + blend_radius)
        blended = []
        for offset, index in enumerate(range(blend_start, blend_end)):
            progress = (offset + 1) / (blend_end - blend_start + 1)
            weight = 0.5 - 0.5 * math.cos(math.pi * progress)
            first = merged[index - start].astype(np.float32)
            second = following[index - following_start].astype(np.float32)
            blended.append(
                np.clip(np.rint(first * (1.0 - weight) + second * weight), 0, 255).astype(
                    np.uint8
                )
            )
        merged = (
            merged[: blend_start - start]
            + blended
            + following[blend_end - following_start :]
        )
        seams.append(
            {
                "following_start_frame": following_start,
                "seam_frame": seam,
                "blend_start_frame": blend_start,
                "blend_end_frame_exclusive": blend_end,
                "cost": cost,
            }
        )
    return merged, seams


def _writer(ffmpeg: Path, output: Path, *, width: int, height: int, fps: float) -> Any:
    output.parent.mkdir(parents=True, exist_ok=True)
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
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--mask-video", type=Path, required=True)
    parser.add_argument("--clean-plate", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--blend-radius", type=int, default=3)
    parser.add_argument("--mask-dilation", type=int, default=3)
    parser.add_argument("--mask-feather", type=int, default=3)
    parser.add_argument(
        "--human-review",
        choices=("pending", "passed", "failed"),
        default="pending",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    experiment_dir = args.experiment_dir.expanduser().resolve()
    source_video = args.source_video.expanduser().resolve()
    mask_video = args.mask_video.expanduser().resolve()
    clean_plate_path = args.clean_plate.expanduser().resolve()
    reference_manifest_path = args.reference_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    metadata_path = experiment_dir / "metadata.json"
    for label, path in (
        ("generation metadata", metadata_path),
        ("source video", source_video),
        ("mask video", mask_video),
        ("clean plate", clean_plate_path),
        ("reference manifest", reference_manifest_path),
        ("FFmpeg", ffmpeg),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} does not exist or is empty: {path}")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    import cv2
    import numpy as np

    generation = json.loads(metadata_path.read_text())
    reference_manifest = json.loads(reference_manifest_path.read_text())
    if generation.get("status") != "completed":
        raise RuntimeError("generation experiment is not complete")
    expected_reference = reference_manifest["outputs"]["reference_sha256"]
    if generation["reference"]["sha256"] != expected_reference:
        raise RuntimeError("generation reference does not match the real-hardware manifest")

    windows: list[tuple[int, list[Any]]] = []
    window_records = []
    for item in generation["windows"]:
        start = int(item["start_frame"])
        result = Path(item["result"])
        if not result.is_file():
            result = (
                experiment_dir
                / "windows"
                / f"window-{int(item['index']):02d}-{start:04d}"
                / "result.mp4"
            )
        frames, info = _decode(cv2, result)
        windows.append((start, frames))
        window_records.append(
            {
                "index": int(item["index"]),
                "start_frame": start,
                "path": str(result),
                "sha256": sha256_file(result),
                "info": info,
                "reference": item.get("reference"),
            }
        )
    windows.sort(key=lambda row: row[0])
    stitched, seams = merge_windows(np, windows, blend_radius=args.blend_radius)
    source, source_info = _decode(cv2, source_video)
    masks, mask_info = _decode(cv2, mask_video)
    if len(stitched) != len(source):
        raise RuntimeError(
            f"stitched {len(stitched)} frames but source has {len(source)} frames"
        )
    clean_plate = cv2.imread(str(clean_plate_path), cv2.IMREAD_COLOR)
    if clean_plate is None:
        raise RuntimeError("cannot decode clean plate")
    height, width = stitched[0].shape[:2]
    clean_plate = cv2.resize(clean_plate, (width, height), interpolation=cv2.INTER_AREA)
    source_resized = [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in source
    ]
    source_fps = float(source_info["fps"])
    mask_fps = float(mask_info["fps"])
    alpha_frames = []
    composited = []
    outside_maximum = 0
    for index, generated in enumerate(stitched):
        mask_index = source_frame_for_output(
            index,
            output_fps=source_fps,
            source_fps=mask_fps,
            source_frames=len(masks),
        )
        mask = cv2.resize(masks[mask_index], (width, height), interpolation=cv2.INTER_NEAREST)
        alpha = alpha_from_mask(
            cv2,
            np,
            mask,
            dilation=args.mask_dilation,
            feather=args.mask_feather,
        )
        frame = composite_under_alpha(np, generated, clean_plate, alpha)
        outside = alpha == 0
        if np.any(outside):
            outside_maximum = max(
                outside_maximum,
                int(np.max(np.abs(frame[outside].astype(np.int16) - clean_plate[outside]))),
            )
        alpha_frames.append(alpha)
        composited.append(frame)
    if outside_maximum != 0:
        raise RuntimeError("pre-encode scene-lock invariant failed")

    output_dir.mkdir(parents=True)
    replacement_path = output_dir / "wuji-real-hardware-appearance-replacement-20p7s.mp4"
    writer = _writer(ffmpeg, replacement_path, width=width, height=height, fps=source_fps)
    assert writer.stdin is not None
    for frame in composited:
        writer.stdin.write(frame.tobytes())
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("replacement encoder failed")

    panel_height = height
    panel_width = width
    comparison_path = output_dir / "human-to-wuji-real-hardware-appearance-20p7s.mp4"
    comparison_writer = _writer(
        ffmpeg,
        comparison_path,
        width=panel_width * 2,
        height=panel_height + 42,
        fps=source_fps,
    )
    assert comparison_writer.stdin is not None
    font = cv2.FONT_HERSHEY_SIMPLEX
    for source_frame, replacement in zip(source_resized, composited):
        canvas = np.zeros((panel_height + 42, panel_width * 2, 3), dtype=np.uint8)
        canvas[42:, :panel_width] = source_frame
        canvas[42:, panel_width:] = replacement
        cv2.putText(canvas, "HUMAN SOURCE", (12, 28), font, 0.65, (245, 245, 245), 2)
        cv2.putText(
            canvas,
            "REAL WUJI APPEARANCE | SYNTHETIC MOTION",
            (panel_width + 12, 28),
            font,
            0.52,
            (80, 220, 255),
            2,
        )
        comparison_writer.stdin.write(canvas.tobytes())
    comparison_writer.stdin.close()
    if comparison_writer.wait() != 0:
        raise RuntimeError("comparison encoder failed")

    poster_path = output_dir / "human-to-wuji-real-hardware-appearance-poster.jpg"
    poster_index = min(len(composited) - 1, len(composited) // 3)
    poster = np.hstack((source_resized[poster_index], composited[poster_index]))
    if not cv2.imwrite(str(poster_path), poster, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError("failed to write poster")

    motion_steps = []
    for index in range(1, len(composited)):
        motion_steps.append(
            float(
                np.mean(
                    np.abs(
                        composited[index].astype(np.float32)
                        - composited[index - 1].astype(np.float32)
                    )
                )
            )
        )
    median_motion = float(np.median(motion_steps))
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "accepted"
            if args.human_review == "passed"
            else "rejected"
            if args.human_review == "failed"
            else "review_required"
        ),
        "honest_status": "WORKING" if args.human_review == "passed" else "PARTIAL",
        "claim": (
            "A real physical Wuji recording provides appearance conditioning; Wan-Animate-2 "
            "synthesizes motion, and a declared mask restores the static source scene."
        ),
        "not_claimed": [
            "This is not footage of the Wuji hardware executing the source gesture.",
            "The video does not prove metric depth, contact force, or physical execution.",
            "Real-reference conditioning does not imply that every generated "
            "pixel came from the hardware recording.",
        ],
        "reference_manifest": {
            "path": str(reference_manifest_path),
            "sha256": sha256_file(reference_manifest_path),
            "hardware_source": reference_manifest["hardware_appearance_source"],
        },
        "generation": {
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "windows": window_records,
            "throughput": generation.get("throughput"),
        },
        "scene_lock": {
            "mask": str(mask_video),
            "mask_sha256": sha256_file(mask_video),
            "clean_plate": str(clean_plate_path),
            "clean_plate_sha256": sha256_file(clean_plate_path),
            "preencode_outside_alpha_max_rgb_difference": outside_maximum,
            "mask_dilation_px_at_generated_resolution": args.mask_dilation,
            "mask_feather_px_at_generated_resolution": args.mask_feather,
        },
        "stitch": {"blend_radius": args.blend_radius, "seams": seams},
        "temporal_audit": {
            "median_full_frame_step": median_motion,
            "maximum_full_frame_step": max(motion_steps),
            "maximum_to_median_ratio": max(motion_steps) / max(median_motion, 1e-6),
        },
        "outputs": {
            "replacement": str(replacement_path),
            "replacement_sha256": sha256_file(replacement_path),
            "comparison": str(comparison_path),
            "comparison_sha256": sha256_file(comparison_path),
            "poster": str(poster_path),
            "poster_sha256": sha256_file(poster_path),
            "frames": len(composited),
            "fps": source_fps,
            "duration_seconds": len(composited) / source_fps,
            "width": width,
            "height": height,
        },
        "human_review": args.human_review,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
