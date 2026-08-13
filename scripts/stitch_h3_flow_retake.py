#!/usr/bin/env python3
"""Stitch protected H3 flow-retake windows into one exact-background video."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import socket
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.h3_flow_retake import (  # noqa: E402
    plan_h3_flow_retake_windows,
    window_temporal_weight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-video", type=Path, required=True)
    parser.add_argument("--windows-root", type=Path, required=True)
    parser.add_argument("--robot-mask", type=Path, action="append", required=True)
    parser.add_argument("--flower-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=124)
    parser.add_argument("--overlap-frames", type=int, default=22)
    parser.add_argument("--mask-dilation", type=int, default=18)
    parser.add_argument("--mask-close", type=int, default=9)
    parser.add_argument("--mask-temporal-radius", type=int, default=2)
    parser.add_argument("--flower-protect-radius", type=int, default=5)
    parser.add_argument("--composite-feather-sigma", type=float, default=3.0)
    parser.add_argument("--base-anchor-frame", type=int, action="append", default=[])
    parser.add_argument("--base-anchor-sigma", type=float, default=3.0)
    parser.add_argument("--base-anchor-strength", type=float, default=0.8)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--crf", type=int, default=10)
    parser.add_argument("--showcase-output", type=Path)
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def _video_info(path: Path) -> dict[str, int | float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/", maxsplit=1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(numerator) / int(denominator),
        "frames": int(stream["nb_frames"]),
        "duration": float(payload["format"]["duration"]),
    }


class PackedMaskReader:
    def __init__(self, np: Any, path: Path):
        data = np.load(path)
        required = {"packed", "height", "width", "bitorder"}
        if not required.issubset(data.files):
            raise ValueError(f"packed mask file has the wrong schema: {path}")
        self.np = np
        self.packed = data["packed"]
        self.height = int(data["height"])
        self.width = int(data["width"])
        self.bitorder = str(data["bitorder"])
        self.size = self.height * self.width

    def __len__(self) -> int:
        return int(self.packed.shape[0])

    def __getitem__(self, index: int) -> Any:
        unpacked = self.np.unpackbits(
            self.packed[index], count=self.size, bitorder=self.bitorder
        )
        return (unpacked.reshape(self.height, self.width) * 255).astype(self.np.uint8)


def _decode_all(cv2: Any, path: Path, expected_frames: int) -> list[Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != expected_frames:
        raise RuntimeError(f"decoded {len(frames)}/{expected_frames} frames from {path}")
    return frames


def _prepare_global_masks(
    cv2: Any,
    np: Any,
    robot_readers: list[PackedMaskReader],
    flower_reader: PackedMaskReader,
    *,
    total_frames: int,
    dilation: int,
    close: int,
    temporal_radius: int,
    flower_protect_radius: int,
) -> tuple[list[Any], list[Any]]:
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
    )
    flower_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (flower_protect_radius * 2 + 1, flower_protect_radius * 2 + 1),
    )
    packed_processed = []
    packed_flowers = []
    for frame_index in range(total_frames):
        union = np.maximum.reduce([reader[frame_index] for reader in robot_readers])
        union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, close_kernel)
        union = cv2.dilate(union, dilation_kernel)
        flower = cv2.dilate(flower_reader[frame_index], flower_kernel)
        union[flower > 0] = 0
        packed_processed.append(np.packbits(union.reshape(-1) > 0, bitorder="little"))
        packed_flowers.append(np.packbits(flower.reshape(-1) > 0, bitorder="little"))
    height, width = robot_readers[0].height, robot_readers[0].width
    size = height * width

    def unpack(payload: Any) -> Any:
        return (
            np.unpackbits(payload, count=size, bitorder="little")
            .reshape(height, width)
            .astype(np.uint8)
            * 255
        )

    masks, flowers = [], []
    for frame_index in range(total_frames):
        start = max(0, frame_index - temporal_radius)
        end = min(total_frames, frame_index + temporal_radius + 1)
        mask = np.maximum.reduce([unpack(packed_processed[i]) for i in range(start, end)])
        flower = unpack(packed_flowers[frame_index])
        mask[flower > 0] = 0
        masks.append(mask)
        flowers.append(flower)
    return masks, flowers


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"stitch experiment already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "raised_cosine_h3_flow_windows_exact_background_flower_protected",
        "status": "running",
        "honest_status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
    }
    _write_json(manifest_path, record)
    try:
        import cv2
        import imageio.v2 as imageio
        import numpy as np

        base_video = args.base_video.expanduser().resolve()
        windows_root = args.windows_root.expanduser().resolve()
        robot_paths = [path.expanduser().resolve() for path in args.robot_mask]
        flower_path = args.flower_mask.expanduser().resolve()
        for path in [base_video, flower_path, *robot_paths]:
            if not path.is_file():
                raise FileNotFoundError(path)
        base_info = _video_info(base_video)
        total_frames = int(base_info["frames"])
        if abs(float(base_info["fps"]) - args.fps) > 1e-6:
            raise ValueError("base video FPS does not match the requested FPS")
        windows = plan_h3_flow_retake_windows(
            total_frames,
            window_frames=args.window_frames,
            overlap_frames=args.overlap_frames,
        )
        if args.base_anchor_sigma <= 0:
            raise ValueError("base-anchor-sigma must be positive")
        if not 0.0 <= args.base_anchor_strength <= 1.0:
            raise ValueError("base-anchor-strength must be in [0, 1]")
        if any(not 0 <= frame < total_frames for frame in args.base_anchor_frame):
            raise ValueError("base-anchor-frame must be inside the timeline")
        window_frames: list[list[Any]] = []
        window_records = []
        for window in windows:
            window_dir = windows_root / f"window-{window.index:02d}-f{window.start_frame:04d}"
            metadata_path = window_dir / "metadata.json"
            candidate_path = window_dir / "protected-h3-flow-refinement.mp4"
            if not metadata_path.is_file() or not candidate_path.is_file():
                raise FileNotFoundError(f"incomplete flow window: {window_dir}")
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("status") != "completed":
                raise ValueError(f"window {window.index} did not complete")
            config = metadata.get("config", {})
            if int(config.get("start_frame", -1)) != window.start_frame:
                raise ValueError(f"window {window.index} start frame mismatch")
            if int(config.get("source_frames", -1)) != window.source_frames:
                raise ValueError(f"window {window.index} source-frame mismatch")
            acceptance = metadata.get("acceptance", {})
            required = (
                "full_decode",
                "edit_mask_fraction_bounded",
                "automatic_temporal_gate",
            )
            if not all(bool(acceptance.get(gate)) for gate in required):
                raise ValueError(f"window {window.index} failed an automatic gate")
            decoded = _decode_all(cv2, candidate_path, window.source_frames)
            if decoded[0].shape[:2] != (int(base_info["height"]), int(base_info["width"])):
                raise ValueError(f"window {window.index} dimensions do not match the base")
            window_frames.append(decoded)
            window_records.append(
                {
                    **asdict(window),
                    "directory": str(window_dir),
                    "metadata": str(metadata_path),
                    "metadata_sha256": _sha256(metadata_path),
                    "candidate": str(candidate_path),
                    "candidate_sha256": _sha256(candidate_path),
                    "acceptance": acceptance,
                }
            )

        robot_readers = [PackedMaskReader(np, path) for path in robot_paths]
        flower_reader = PackedMaskReader(np, flower_path)
        if any(len(reader) != total_frames for reader in [*robot_readers, flower_reader]):
            raise ValueError("mask timeline does not match the base video")
        edit_masks, flower_masks = _prepare_global_masks(
            cv2,
            np,
            robot_readers,
            flower_reader,
            total_frames=total_frames,
            dilation=args.mask_dilation,
            close=args.mask_close,
            temporal_radius=args.mask_temporal_radius,
            flower_protect_radius=args.flower_protect_radius,
        )
        frame_sources: list[list[tuple[int, int, float]]] = [[] for _ in range(total_frames)]
        for window in windows:
            for local_frame in range(window.source_frames):
                global_frame = window.start_frame + local_frame
                frame_sources[global_frame].append(
                    (
                        window.index,
                        local_frame,
                        window_temporal_weight(windows, window.index, local_frame),
                    )
                )
        for frame_index, sources in enumerate(frame_sources):
            total_weight = sum(source[2] for source in sources)
            if not sources or abs(total_weight - 1.0) > 1e-6:
                raise ValueError(
                    f"invalid temporal stitch weights at frame {frame_index}: {sources}"
                )

        result_path = output_dir / "h3-flow-inversion-full-27s.mp4"
        writer = imageio.get_writer(
            str(result_path),
            fps=args.fps,
            codec="libx264",
            quality=10,
            ffmpeg_params=["-crf", str(args.crf), "-preset", "medium", "-pix_fmt", "yuv420p"],
        )
        capture = cv2.VideoCapture(str(base_video))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open base video: {base_video}")
        previous_base: list[Any] = []
        previous_result: list[Any] = []
        base_delta, result_delta, base_jerk, result_jerk = [], [], [], []
        outside_changed_pixels = 0
        flower_changed_pixels = 0
        outside_change_frames: list[dict[str, int]] = []
        safety_radius = max(1, round(args.composite_feather_sigma * 4))
        safety_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (safety_radius * 2 + 1, safety_radius * 2 + 1)
        )
        seam_frames = {window.start_frame for window in windows[1:]}
        seam_metrics: dict[str, dict[str, float]] = {}
        mask_fractions = []
        for frame_index in range(total_frames):
            ok, base = capture.read()
            if not ok:
                raise RuntimeError(f"base video stopped at frame {frame_index}")
            sources = frame_sources[frame_index]
            candidate = np.zeros_like(base, dtype=np.float32)
            for window_index, local_frame, weight in sources:
                candidate += window_frames[window_index][local_frame].astype(np.float32) * weight
            mask = edit_masks[frame_index]
            flower = flower_masks[frame_index]
            alpha = mask.astype(np.float32) / 255.0
            if args.composite_feather_sigma > 0:
                alpha = cv2.GaussianBlur(alpha, (0, 0), args.composite_feather_sigma)
            alpha = alpha.clip(0.0, 1.0)
            support = cv2.dilate(mask, safety_kernel) > 0
            alpha[~support] = 0.0
            alpha[flower > 0] = 0.0
            if args.base_anchor_frame:
                anchor_attenuation = max(
                    math.exp(
                        -0.5
                        * ((frame_index - anchor) / args.base_anchor_sigma) ** 2
                    )
                    for anchor in args.base_anchor_frame
                )
                alpha *= 1.0 - args.base_anchor_strength * anchor_attenuation
            result = np.rint(
                candidate * alpha[..., None]
                + base.astype(np.float32) * (1.0 - alpha[..., None])
            ).clip(0, 255).astype(np.uint8)
            safety = cv2.dilate(mask, safety_kernel) > 0
            result = np.where(safety[..., None], result, base).astype(np.uint8)
            result = np.where((flower > 0)[..., None], base, result).astype(np.uint8)
            changed = np.any(result != base, axis=2)
            frame_outside_changes = int(np.count_nonzero(changed & ~safety))
            outside_changed_pixels += frame_outside_changes
            if frame_outside_changes and len(outside_change_frames) < 20:
                outside_change_frames.append(
                    {"frame": frame_index, "changed_pixels": frame_outside_changes}
                )
            flower_changed_pixels += int(np.count_nonzero(changed & (flower > 0)))
            mask_fractions.append(float(np.mean(mask > 0)))
            writer.append_data(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

            base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
            previous_base.append(base_gray)
            previous_result.append(result_gray)
            if len(previous_base) > 3:
                previous_base.pop(0)
                previous_result.pop(0)
            if len(previous_base) >= 2:
                roi = (edit_masks[frame_index] > 0) | (edit_masks[frame_index - 1] > 0)
                b_delta = float(np.abs(previous_base[-1] - previous_base[-2])[roi].mean())
                r_delta = float(np.abs(previous_result[-1] - previous_result[-2])[roi].mean())
                base_delta.append(b_delta)
                result_delta.append(r_delta)
            if len(previous_base) >= 3:
                roi = (
                    (edit_masks[frame_index] > 0)
                    | (edit_masks[frame_index - 1] > 0)
                    | (edit_masks[frame_index - 2] > 0)
                )
                b_jerk = float(
                    np.abs(previous_base[-1] - 2 * previous_base[-2] + previous_base[-3])[roi].mean()
                )
                r_jerk = float(
                    np.abs(previous_result[-1] - 2 * previous_result[-2] + previous_result[-3])[roi].mean()
                )
                base_jerk.append(b_jerk)
                result_jerk.append(r_jerk)
                if frame_index in seam_frames or frame_index - 1 in seam_frames:
                    seam_metrics[str(frame_index)] = {
                        "base_jerk": b_jerk,
                        "result_jerk": r_jerk,
                        "ratio": r_jerk / max(b_jerk, 1e-6),
                    }
        capture.release()
        writer.close()
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(result_path), "-f", "null", "-"],
            check=True,
        )
        result_info = _video_info(result_path)
        metrics = {
            "base_mean_abs_delta": float(np.mean(base_delta)),
            "result_mean_abs_delta": float(np.mean(result_delta)),
            "delta_ratio": float(np.mean(result_delta) / max(np.mean(base_delta), 1e-6)),
            "base_mean_abs_jerk": float(np.mean(base_jerk)),
            "result_mean_abs_jerk": float(np.mean(result_jerk)),
            "jerk_ratio": float(np.mean(result_jerk) / max(np.mean(base_jerk), 1e-6)),
            "mean_edit_mask_fraction": float(np.mean(mask_fractions)),
            "maximum_edit_mask_fraction": float(np.max(mask_fractions)),
            "outside_safety_changed_pixels_preencode": outside_changed_pixels,
            "outside_safety_change_frames": outside_change_frames,
            "flower_changed_pixels_preencode": flower_changed_pixels,
            "seams": seam_metrics,
        }
        acceptance = {
            "all_windows_completed": True,
            "all_window_temporal_gates": all(
                bool(item["acceptance"].get("automatic_temporal_gate"))
                for item in window_records
            ),
            "full_decode": int(result_info["frames"]) == total_frames,
            "outside_safety_exact_preencode": outside_changed_pixels == 0,
            "flowers_exact_preencode": flower_changed_pixels == 0,
            "edit_scope_bounded": float(np.max(mask_fractions)) < 0.35,
            "full_roi_jerk_bounded": metrics["jerk_ratio"] <= 1.02,
            "seam_jerk_bounded": all(item["ratio"] <= 1.25 for item in seam_metrics.values()),
            "human_review": False,
        }
        record.update(
            {
                "status": "completed",
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "window_frames": args.window_frames,
                    "overlap_frames": args.overlap_frames,
                    "mask_dilation": args.mask_dilation,
                    "mask_close": args.mask_close,
                    "mask_temporal_radius": args.mask_temporal_radius,
                    "flower_protect_radius": args.flower_protect_radius,
                    "composite_feather_sigma": args.composite_feather_sigma,
                    "base_anchor_frames": args.base_anchor_frame,
                    "base_anchor_sigma": args.base_anchor_sigma,
                    "base_anchor_strength": args.base_anchor_strength,
                    "fps": args.fps,
                    "crf": args.crf,
                    "coordinate_frame": "camera:H3_832x480_pixels and absolute source frame",
                },
                "inputs": {
                    "base_video": str(base_video),
                    "base_sha256": _sha256(base_video),
                    "base_info": base_info,
                    "windows_root": str(windows_root),
                    "robot_masks": [
                        {"path": str(path), "sha256": _sha256(path)} for path in robot_paths
                    ],
                    "flower_mask": {"path": str(flower_path), "sha256": _sha256(flower_path)},
                },
                "windows": window_records,
                "output": {
                    "path": str(result_path),
                    "sha256": _sha256(result_path),
                    "info": result_info,
                },
                "metrics": metrics,
                "acceptance": acceptance,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "opencv-python", "imageio")
                },
                "git": _git_state(),
            }
        )
        if args.showcase_output is not None:
            showcase = args.showcase_output.expanduser().resolve()
            showcase.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result_path, showcase)
            record["showcase_output"] = str(showcase)
            record["showcase_sha256"] = _sha256(showcase)
        _write_json(manifest_path, record)
        print(json.dumps({"status": "completed", "output": str(result_path), "acceptance": acceptance}))
        return 0
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "honest_status": "PARTIAL",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(manifest_path, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
