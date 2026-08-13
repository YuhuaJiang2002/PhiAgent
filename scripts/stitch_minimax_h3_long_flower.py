#!/usr/bin/env python3
"""Repair, deflicker, and stitch a full MiniMax-H3 flower replacement."""

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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.agent.epl_video_evolution import ReplacementThresholds  # noqa: E402
from phiagent.rendering.h3_long_video import (  # noqa: E402
    apply_subject_color_offset,
    estimate_subject_color_offset,
    merge_at_masked_seam,
)
from scripts.build_multi_anchor_robot_replacement import _git_state  # noqa: E402
from scripts.build_rigid_part_robot_replacement import (  # noqa: E402
    _pose_clear_mask,
    _track_pose,
)
from scripts.evaluate_minimax_h3_flower_validation import (  # noqa: E402
    FACE_REPLACEMENT_THRESHOLD,
    _expand_with_face_box,
    _flower_mask,
    _preprocess_reference,
    _score,
)


class PackedMasks:
    """Read-only sequence of bit-packed uint8 masks."""

    def __init__(self, np: Any, masks: list[Any]):
        if not masks:
            raise ValueError("PackedMasks requires at least one mask")
        self._np = np
        self.shape = masks[0].shape
        self._size = int(masks[0].size)
        self._payload = [
            np.packbits((mask > 0).reshape(-1), bitorder="little")
            for mask in masks
        ]

    @classmethod
    def from_packed(
        cls,
        np: Any,
        shape: tuple[int, int],
        payload: list[Any],
    ) -> "PackedMasks":
        instance = cls.__new__(cls)
        instance._np = np
        instance.shape = shape
        instance._size = int(shape[0] * shape[1])
        instance._payload = payload
        return instance

    def __len__(self) -> int:
        return len(self._payload)

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self._payload)
        if not 0 <= index < len(self._payload):
            raise IndexError(index)
        unpacked = self._np.unpackbits(
            self._payload[index],
            count=self._size,
            bitorder="little",
        )
        return (unpacked.reshape(self.shape) * 255).astype(self._np.uint8)

    def __iter__(self) -> Any:
        for index in range(len(self)):
            yield self[index]


class MemmapFrames:
    """Read-only sequence backed by a raw BGR NumPy memory map."""

    def __init__(
        self,
        np: Any,
        path: Path,
        frame_count: int,
        height: int,
        width: int,
        *,
        mode: str,
    ):
        self._array = np.memmap(
            path,
            dtype=np.uint8,
            mode=mode,
            shape=(frame_count, height, width, 3),
        )

    def __len__(self) -> int:
        return int(self._array.shape[0])

    def __getitem__(self, index: int) -> Any:
        return self._array[index]

    def __iter__(self) -> Any:
        for index in range(len(self)):
            yield self[index]


def _pose_face_boxes(
    np: Any,
    tracks: Any,
    width: int,
    height: int,
) -> tuple[list[tuple[int, int, int, int]], dict[str, object]]:
    """Build dense face boxes from MediaPipe's eleven pose face landmarks."""

    boxes = []
    for points in tracks:
        face = points[:11]
        shoulder_width = float(np.linalg.norm(points[11] - points[12]))
        center = np.mean(face, axis=0)
        box_width = max(36.0, shoulder_width * 0.62, float(np.ptp(face[:, 0])) * 1.45)
        box_height = max(42.0, shoulder_width * 0.78, float(np.ptp(face[:, 1])) * 1.55)
        left = max(0, min(width - 1, round(center[0] - box_width / 2)))
        top = max(0, min(height - 1, round(center[1] - box_height * 0.48)))
        right = max(left + 1, min(width, round(center[0] + box_width / 2)))
        bottom = max(top + 1, min(height, round(center[1] + box_height * 0.52)))
        boxes.append((left, top, right - left, bottom - top))
    return boxes, {
        "detector": "MediaPipe Pose landmarks 0-10",
        "detected_frames": len(boxes),
        "total_frames": len(boxes),
        "selection": "dense pose-derived source-face support in camera:H3_output_pixels",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _freeze_execution_sources(output_dir: Path) -> list[dict[str, str]]:
    destination = output_dir / "provenance" / "execution-sources"
    destination.mkdir(parents=True, exist_ok=True)
    sources = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "phiagent" / "rendering" / "h3_long_video.py",
        PROJECT_ROOT / "phiagent" / "rendering" / "minimax_h3.py",
        PROJECT_ROOT / "phiagent" / "agent" / "epl_video_evolution.py",
        PROJECT_ROOT / "scripts" / "evaluate_minimax_h3_flower_validation.py",
        PROJECT_ROOT / "scripts" / "build_rigid_part_robot_replacement.py",
    )
    records = []
    for source in sources:
        target = destination / source.name
        shutil.copy2(source, target)
        records.append(
            {
                "source": str(source),
                "frozen_copy": str(target),
                "sha256": _sha256(target),
            }
        )
    return records


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, int | float]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    info: dict[str, int | float] = {
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


def _decode_to_memmap(
    cv2: Any,
    np: Any,
    video: Path,
    output: Path,
) -> tuple[MemmapFrames, dict[str, int | float]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if reported < 1:
        capture.release()
        raise RuntimeError(f"video has no exact frame count: {video}")
    frames = MemmapFrames(np, output, reported, height, width, mode="w+")
    decoded = 0
    while decoded < reported:
        ok, frame = capture.read()
        if not ok:
            break
        frames._array[decoded] = frame
        decoded += 1
    capture.release()
    frames._array.flush()
    if decoded != reported:
        raise RuntimeError(f"decoded {decoded}/{reported} frames from {video}")
    return frames, {
        "width": width,
        "height": height,
        "fps": fps,
        "reported_frames": reported,
        "decoded_frames": decoded,
        "storage": "raw BGR NumPy memmap",
        "memmap_path": str(output),
    }


def _count_decoded_frames(cv2: Any, video: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    info: dict[str, int | float] = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "reported_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    info["decoded_frames"] = decoded
    return info


def _writer(ffmpeg: Path, output: Path, width: int, height: int, fps: float) -> Any:
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "12",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _write_video(ffmpeg: Path, output: Path, frames: list[Any], fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = _writer(ffmpeg, output, width, height, fps)
    try:
        assert writer.stdin is not None
        for frame in frames:
            writer.stdin.write(frame.tobytes())
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
        if writer.wait():
            raise RuntimeError(f"FFmpeg failed to encode {output}")


def _align_source(
    ffmpeg: Path,
    source: Path,
    output: Path,
    width: int,
    height: int,
) -> list[str]:
    command = [
        str(ffmpeg), "-y", "-v", "error", "-i", str(source), "-vf",
        (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        ),
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "12",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    subprocess.run(command, check=True)
    return command


def _align_mask(cv2: Any, mask: Any, width: int, height: int) -> Any:
    scale = max(width / mask.shape[1], height / mask.shape[0])
    scaled_width = round(mask.shape[1] * scale)
    scaled_height = round(mask.shape[0] * scale)
    resized = cv2.resize(mask, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
    left = max(0, (scaled_width - width) // 2)
    top = max(0, (scaled_height - height) // 2)
    return resized[top : top + height, left : left + width]


def _build_supports(
    cv2: Any,
    np: Any,
    tracks: Any,
    safety: Any,
    face_boxes: list[tuple[int, int, int, int] | None],
    source: list[Any],
) -> tuple[PackedMasks, PackedMasks, PackedMasks, dict[str, float]]:
    subject_payload, allowed_payload, object_payload = [], [], []
    kernel_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_31 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    coverages = []
    for frame, points, face_box in zip(source, tracks, face_boxes):
        dynamic = _pose_clear_mask(cv2, np, frame.shape[:2], points)
        local_safety = cv2.bitwise_and(safety, cv2.dilate(dynamic, kernel_31))
        core = cv2.bitwise_or(dynamic, local_safety)
        core = _expand_with_face_box(cv2, core, face_box, 12)
        core = cv2.dilate(core, kernel_9)
        allowed = cv2.dilate(core, kernel_9)
        flowers = _flower_mask(cv2, np, frame, dilation=2, exclude_skin_like=True)
        flowers = cv2.bitwise_and(flowers, allowed)
        if face_box is not None:
            face = np.zeros(core.shape, dtype=np.uint8)
            face = _expand_with_face_box(cv2, face, face_box, 12)
            flowers[face > 0] = 0
        subject_payload.append(np.packbits((core > 0).reshape(-1), bitorder="little"))
        allowed_payload.append(np.packbits((allowed > 0).reshape(-1), bitorder="little"))
        object_payload.append(np.packbits((flowers > 0).reshape(-1), bitorder="little"))
        coverages.append(float(np.count_nonzero(allowed) / allowed.size))
    record = {
        "minimum_allowed_coverage": min(coverages),
        "mean_allowed_coverage": float(np.mean(coverages)),
        "maximum_allowed_coverage": max(coverages),
        "coordinate_frame": "camera:H3_output_pixels",
        "storage": "one-bit packed masks; unpacked on indexed access",
    }
    return (
        PackedMasks.from_packed(np, source[0].shape[:2], subject_payload),
        PackedMasks.from_packed(np, source[0].shape[:2], allowed_payload),
        PackedMasks.from_packed(np, source[0].shape[:2], object_payload),
        record,
    )


def _lock_candidate(
    cv2: Any,
    np: Any,
    source: list[Any],
    generated: list[Any],
    start: int,
    subject_masks: list[Any],
    allowed_masks: list[Any],
    object_masks: list[Any],
) -> list[Any]:
    result = []
    for local, raw in enumerate(generated):
        absolute = start + local
        source_frame = source[absolute]
        core = subject_masks[absolute]
        allowed = allowed_masks[absolute]
        alpha = cv2.GaussianBlur(allowed, (9, 9), 1.1).astype(np.float32) / 255.0
        alpha[allowed == 0] = 0.0
        alpha[core > 0] = 1.0
        frame = np.rint(
            raw.astype(np.float32) * alpha[..., None]
            + source_frame.astype(np.float32) * (1.0 - alpha[..., None])
        ).astype(np.uint8)
        frame[core > 0] = raw[core > 0]
        objects = object_masks[absolute] > 0
        frame[objects] = source_frame[objects]
        result.append(frame)
    return result


def hard_relock_in_place(
    frames: list[Any],
    source: list[Any],
    allowed_masks: list[Any],
    object_masks: list[Any],
) -> dict[str, object]:
    """Restore protected pixels after any lossy intermediate encode/decode."""

    background_pixels = 0
    object_pixels = 0
    for index, frame in enumerate(frames):
        outside = allowed_masks[index] == 0
        objects = object_masks[index] > 0
        frame[outside] = source[index][outside]
        frame[objects] = source[index][objects]
        background_pixels += int(outside.sum())
        object_pixels += int(objects.sum())
    return {
        "mutation": "exact_source_pixel_relock_after_intermediate_decode",
        "background_pixels": background_pixels,
        "object_pixels": object_pixels,
    }


def _smooth_series(np: Any, values: Any, sigma: float) -> Any:
    radius = max(1, round(3 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets**2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, channel], kernel, mode="valid") for channel in range(3)],
        axis=1,
    )


def stabilize_subject_color(
    np: Any,
    frames: list[Any],
    source: list[Any],
    subject_masks: list[Any],
    object_masks: list[Any],
    *,
    sigma: float = 2.0,
    maximum_offset: float = 6.0,
    frozen_interval: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Remove only high-frequency per-frame robot color drift."""

    medians = []
    for frame, source_frame, subject, objects in zip(
        frames, source, subject_masks, object_masks
    ):
        changed = np.max(np.abs(frame.astype(np.int16) - source_frame.astype(np.int16)), axis=2) >= 8
        support = (subject > 0) & ~(objects > 0) & changed
        if np.count_nonzero(support) < 64:
            medians.append(medians[-1] if medians else np.zeros(3, dtype=np.float32))
        else:
            medians.append(np.median(frame[support].astype(np.float32), axis=0))
    medians_array = np.asarray(medians, dtype=np.float32)
    smooth = _smooth_series(np, medians_array, sigma)
    offsets = np.clip(smooth - medians_array, -maximum_offset, maximum_offset)
    for index, (frame, source_frame, subject, objects, offset) in enumerate(
        zip(frames, source, subject_masks, object_masks, offsets)
    ):
        if frozen_interval is not None and frozen_interval[0] <= index < frozen_interval[1]:
            continue
        changed = np.max(
            np.abs(frame.astype(np.int16) - source_frame.astype(np.int16)), axis=2
        ) >= 8
        support = (subject > 0) & ~(objects > 0) & changed
        frame[support] = np.clip(
            frame[support].astype(np.float32) + offset,
            0,
            255,
        ).astype(np.uint8)
    return {
        "sigma_frames": sigma,
        "maximum_offset": maximum_offset,
        "maximum_applied_bgr_offset": [float(value) for value in np.max(np.abs(offsets), axis=0)],
        "frozen_contact_interval": list(frozen_interval) if frozen_interval else None,
        "mutation": "in_place_after_previous_round_was_encoded",
    }


def low_motion_flow_deflicker(
    cv2: Any,
    np: Any,
    frames: list[Any],
    source: list[Any],
    subject_masks: list[Any],
    object_masks: list[Any],
    *,
    weight: float = 0.12,
    frozen_interval: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Blend a flow-aligned neighbor only on low-motion robot interiors."""

    blended_pixels = 0
    eligible_pixels = 0
    height, width = frames[0].shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    previous_source_gray = cv2.cvtColor(source[0], cv2.COLOR_BGR2GRAY)
    previous_frame = frames[0].copy()
    for index in range(1, len(frames)):
        current_original = frames[index].copy()
        current_source_gray = cv2.cvtColor(source[index], cv2.COLOR_BGR2GRAY)
        backward = cv2.calcOpticalFlowFarneback(
            current_source_gray,
            previous_source_gray,
            None,
            0.5,
            3,
            19,
            3,
            5,
            1.2,
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        warped = cv2.remap(
            previous_frame,
            grid_x + backward[..., 0],
            grid_y + backward[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        source_motion = cv2.absdiff(current_source_gray, previous_source_gray)
        color_delta = np.mean(
            np.abs(frames[index].astype(np.float32) - warped.astype(np.float32)),
            axis=2,
        )
        interior = cv2.erode(
            subject_masks[index],
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        ) > 0
        eligible = interior & ~(object_masks[index] > 0)
        blend = eligible & (source_motion < 7) & (color_delta < 32)
        frozen = frozen_interval is not None and frozen_interval[0] <= index < frozen_interval[1]
        if not frozen:
            frames[index][blend] = np.rint(
                frames[index][blend].astype(np.float32) * (1.0 - weight)
                + warped[blend].astype(np.float32) * weight
            ).astype(np.uint8)
            blended_pixels += int(np.count_nonzero(blend))
        eligible_pixels += int(np.count_nonzero(eligible))
        previous_source_gray = current_source_gray
        previous_frame = current_original
    return {
        "weight": weight,
        "blended_pixel_fraction_of_eligible": blended_pixels / max(1, eligible_pixels),
        "frozen_contact_interval": list(frozen_interval) if frozen_interval else None,
        "mutation": "in_place_after_previous_round_was_encoded",
    }


def _transition_metrics(cv2: Any, np: Any, frames: list[Any]) -> dict[str, object]:
    grays = [
        cv2.cvtColor(cv2.resize(frame, (256, 148), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        for frame in frames
    ]
    energy = [float(np.mean(cv2.absdiff(grays[index], grays[index - 1]))) for index in range(1, len(grays))]
    median = float(np.median(energy))
    return {
        "median_full_frame_transition_energy": median,
        "maximum_full_frame_transition_energy": max(energy),
        "maximum_full_frame_transition_ratio": max(energy) / max(median, 1e-6),
        "transition_energy": energy,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--quality-anchor", type=Path, required=True)
    parser.add_argument("--quality-anchor-start", type=int, default=216)
    parser.add_argument(
        "--quality-anchor-mode",
        choices=("hard", "evidence-only"),
        default="hard",
    )
    parser.add_argument("--contact-start", type=int, default=236)
    parser.add_argument("--contact-end-exclusive", type=int, default=316)
    parser.add_argument("--robot-reference", type=Path, required=True)
    parser.add_argument("--safety-mask", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--physics-manifest", type=Path, required=True)
    parser.add_argument("--physics-trajectory", type=Path, required=True)
    parser.add_argument("--physics-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--human-review", choices=("pending", "passed", "failed"), default="pending")
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"stitch experiment already exists: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_dir = args.generation_dir.expanduser().resolve()
    paths = {
        "generation_metadata": generation_dir / "metadata.json",
        "source": args.source.expanduser().resolve(),
        "quality_anchor": args.quality_anchor.expanduser().resolve(),
        "robot_reference": args.robot_reference.expanduser().resolve(),
        "safety_mask": args.safety_mask.expanduser().resolve(),
        "pose_model": args.pose_model.expanduser().resolve(),
        "physics_manifest": args.physics_manifest.expanduser().resolve(),
        "physics_trajectory": args.physics_trajectory.expanduser().resolve(),
        "physics_result": args.physics_result.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    generation = json.loads(paths["generation_metadata"].read_text())
    if generation.get("status") != "completed":
        raise RuntimeError(f"generation is not completed: {generation.get('status')}")
    physics = json.loads(paths["physics_result"].read_text())
    physics_manifest = json.loads(paths["physics_manifest"].read_text())
    physics_trajectory = json.loads(paths["physics_trajectory"].read_text())
    if not physics.get("task_success") or not physics.get("physically_valid"):
        raise RuntimeError("recorded flower-insertion physics evidence is not successful")
    if physics_manifest.get("status") != "WORKING":
        raise RuntimeError("flower-insertion manifest is not marked WORKING")

    import cv2
    import mediapipe as mp
    import numpy as np

    np.random.seed(args.seed)
    width = int(generation["config"]["width"])
    height = int(generation["config"]["height"])
    fps = float(generation["config"]["fps"])
    source_aligned_path = output_dir / "source-aligned-660.mp4"
    align_command = _align_source(paths["ffmpeg"], paths["source"], source_aligned_path, width, height)
    source, source_info = _decode_to_memmap(
        cv2,
        np,
        source_aligned_path,
        output_dir / "source-aligned-660.bgr",
    )
    if len(source) != 660:
        raise RuntimeError(f"aligned source has {len(source)} frames, expected 660")
    anchor_capture = cv2.VideoCapture(str(paths["quality_anchor"]))
    if not anchor_capture.isOpened():
        raise RuntimeError("cannot open quality anchor")
    anchor_count = int(anchor_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    anchor_info = {
        "width": int(anchor_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(anchor_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(anchor_capture.get(cv2.CAP_PROP_FPS)),
        "reported_frames": anchor_count,
    }
    anchor_capture.release()
    anchor_start = args.quality_anchor_start
    anchor_end = anchor_start + anchor_count
    if not (anchor_start <= args.contact_start < args.contact_end_exclusive <= anchor_end):
        raise ValueError("hard contact interval must be contained in the quality anchor")

    tracks, pose_tracking = _track_pose(
        cv2=cv2,
        np=np,
        mp=mp,
        source=source_aligned_path,
        model=paths["pose_model"],
        fps=fps,
        width=width,
        height=height,
    )
    face_boxes, face_tracking = _pose_face_boxes(np, tracks, width, height)
    safety_raw = cv2.imread(str(paths["safety_mask"]), cv2.IMREAD_GRAYSCALE)
    if safety_raw is None:
        raise RuntimeError("cannot decode safety mask")
    safety = (_align_mask(cv2, safety_raw, width, height) >= 127).astype(np.uint8) * 255
    subject_masks, allowed_masks, object_masks, support_tracking = _build_supports(
        cv2, np, tracks, safety, face_boxes, source
    )

    locked_dir = output_dir / "locked-windows"
    aligned_dir = output_dir / "aligned-windows"
    locked_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    locked_paths: dict[int, Path] = {}
    window_lengths: dict[int, int] = {}
    window_inputs = []
    for item in generation["windows"]:
        start = int(item["start_frame"])
        path = Path(str(item["result"]))
        if not path.is_file():
            path = generation_dir / "windows" / f"window-{int(item['index']):02d}-{start:04d}" / "raw-h3-nf4.mp4"
        raw, info = _decode(cv2, path)
        if len(raw) != int(item["frame_count"]):
            raise RuntimeError(f"window {item['index']} has {len(raw)} decoded frames")
        locked = _lock_candidate(
            cv2, np, source, raw, start, subject_masks, allowed_masks, object_masks
        )
        locked_path = locked_dir / f"window-{int(item['index']):02d}-{start:04d}.mp4"
        _write_video(paths["ffmpeg"], locked_path, locked, fps)
        locked_paths[start] = locked_path
        window_lengths[start] = len(locked)
        window_inputs.append(
            {
                "index": int(item["index"]),
                "start_frame": start,
                "path": str(path),
                "sha256": _sha256(path),
                "info": info,
                "locked_path": str(locked_path),
                "locked_sha256": _sha256(locked_path),
            }
        )
        del raw, locked
    anchor_raw, decoded_anchor_info = _decode(cv2, paths["quality_anchor"])
    anchor_info = decoded_anchor_info
    anchor = _lock_candidate(
        cv2, np, source, anchor_raw, anchor_start, subject_masks, allowed_masks, object_masks
    )
    anchor_locked_path = locked_dir / "quality-anchor-locked.mp4"
    _write_video(paths["ffmpeg"], anchor_locked_path, anchor, fps)
    del anchor_raw, anchor
    starts = sorted(locked_paths)
    left_starts = [start for start in starts if start < anchor_start]
    right_starts = [
        start
        for start in starts
        if start < anchor_end < start + window_lengths[start]
    ]
    right_starts += [start for start in starts if start >= anchor_end]
    right_starts = sorted(set(right_starts))
    if not left_starts or not right_starts:
        raise RuntimeError("window plan does not bracket the quality anchor")

    offsets = []
    reference_path, reference_start = anchor_locked_path, anchor_start
    for start in reversed(left_starts):
        reference, _ = _decode(cv2, reference_path)
        candidate, _ = _decode(cv2, locked_paths[start])
        offset = estimate_subject_color_offset(
            np,
            reference=reference,
            reference_start=reference_start,
            candidate=candidate,
            candidate_start=start,
            subject_masks=subject_masks,
            object_masks=object_masks,
        )
        aligned = apply_subject_color_offset(
            np,
            frames=candidate,
            start_frame=start,
            subject_masks=subject_masks,
            object_masks=object_masks,
            offset=offset,
        )
        aligned_path = aligned_dir / f"window-{start:04d}.mp4"
        _write_video(paths["ffmpeg"], aligned_path, aligned, fps)
        locked_paths[start] = aligned_path
        offsets.append({"start_frame": start, "direction": "left", "bgr": offset})
        reference_path, reference_start = aligned_path, start
        del reference, candidate, aligned
    reference_path, reference_start = anchor_locked_path, anchor_start
    for start in right_starts:
        reference, _ = _decode(cv2, reference_path)
        candidate, _ = _decode(cv2, locked_paths[start])
        offset = estimate_subject_color_offset(
            np,
            reference=reference,
            reference_start=reference_start,
            candidate=candidate,
            candidate_start=start,
            subject_masks=subject_masks,
            object_masks=object_masks,
        )
        aligned = apply_subject_color_offset(
            np,
            frames=candidate,
            start_frame=start,
            subject_masks=subject_masks,
            object_masks=object_masks,
            offset=offset,
        )
        aligned_path = aligned_dir / f"window-{start:04d}.mp4"
        _write_video(paths["ffmpeg"], aligned_path, aligned, fps)
        locked_paths[start] = aligned_path
        offsets.append({"start_frame": start, "direction": "right", "bgr": offset})
        reference_path, reference_start = aligned_path, start
        del reference, candidate, aligned

    seams = []
    stitched_start = starts[0]
    if args.quality_anchor_mode == "evidence-only":
        stitched, _ = _decode(cv2, locked_paths[starts[0]])
        for start in starts[1:]:
            following, _ = _decode(cv2, locked_paths[start])
            stitched, seam = merge_at_masked_seam(
                np,
                current=stitched,
                current_start=stitched_start,
                following=following,
                following_start=start,
                source=source,
                subject_masks=allowed_masks,
            )
            seams.append({**seam, "role": "recursive_window"})
            del following
        contact_matches_anchor: bool | None = None
    else:
        stitched, _ = _decode(cv2, locked_paths[left_starts[0]])
        for start in left_starts[1:]:
            following, _ = _decode(cv2, locked_paths[start])
            stitched, seam = merge_at_masked_seam(
                np,
                current=stitched,
                current_start=stitched_start,
                following=following,
                following_start=start,
                source=source,
                subject_masks=allowed_masks,
            )
            seams.append({**seam, "role": "left_window"})
            del following
        anchor, _ = _decode(cv2, anchor_locked_path)
        stitched, seam = merge_at_masked_seam(
            np,
            current=stitched,
            current_start=stitched_start,
            following=anchor,
            following_start=anchor_start,
            source=source,
            subject_masks=allowed_masks,
            maximum_frame_exclusive=args.contact_start + 1,
        )
        seams.append({**seam, "role": "enter_quality_anchor"})
        first_right = right_starts[0]
        following, _ = _decode(cv2, locked_paths[first_right])
        stitched, seam = merge_at_masked_seam(
            np,
            current=stitched,
            current_start=stitched_start,
            following=following,
            following_start=first_right,
            source=source,
            subject_masks=allowed_masks,
            minimum_frame=args.contact_end_exclusive,
        )
        seams.append({**seam, "role": "leave_quality_anchor"})
        del following
        for start in right_starts[1:]:
            following, _ = _decode(cv2, locked_paths[start])
            stitched, seam = merge_at_masked_seam(
                np,
                current=stitched,
                current_start=stitched_start,
                following=following,
                following_start=start,
                source=source,
                subject_masks=allowed_masks,
            )
            seams.append({**seam, "role": "right_window"})
            del following
        contact_matches_anchor = all(
            np.array_equal(stitched[index], anchor[index - anchor_start])
            for index in range(args.contact_start, args.contact_end_exclusive)
        )
        if not contact_matches_anchor:
            raise RuntimeError("hard contact quality anchor changed during seam selection")
        del anchor
    if len(stitched) != len(source):
        raise RuntimeError(f"stitched {len(stitched)} frames, expected {len(source)}")
    relock_record = hard_relock_in_place(stitched, source, allowed_masks, object_masks)
    contact_digest = hashlib.sha256(
        b"".join(
            stitched[index].tobytes()
            for index in range(args.contact_start, args.contact_end_exclusive)
        )
    ).hexdigest()
    robot_image = cv2.imread(str(paths["robot_reference"]), cv2.IMREAD_COLOR)
    if robot_image is None:
        raise RuntimeError("cannot decode robot reference")
    robot_reference = _preprocess_reference(cv2, robot_image, width, height, False)
    thresholds = ReplacementThresholds()
    round_records = []
    scored = []
    round_names = (
        "background-object-locked-seam",
        "subject-color-stabilized",
        "low-motion-flow-deflicker",
        "stronger-low-motion-flow-deflicker",
    )
    frozen_interval = (
        (args.contact_start, args.contact_end_exclusive)
        if args.quality_anchor_mode == "hard"
        else None
    )
    for index, name in enumerate(round_names):
        if index == 0:
            mutation: dict[str, object] = {}
        elif index == 1:
            mutation = stabilize_subject_color(
                np,
                stitched,
                source,
                subject_masks,
                object_masks,
                frozen_interval=frozen_interval,
            )
        elif index == 2:
            mutation = low_motion_flow_deflicker(
                cv2,
                np,
                stitched,
                source,
                subject_masks,
                object_masks,
                frozen_interval=frozen_interval,
            )
        else:
            mutation = low_motion_flow_deflicker(
                cv2,
                np,
                stitched,
                source,
                subject_masks,
                object_masks,
                weight=0.18,
                frozen_interval=frozen_interval,
            )
        mutation["protected_pixel_relock"] = hard_relock_in_place(
            stitched, source, allowed_masks, object_masks
        )
        round_contact_digest = hashlib.sha256(
            b"".join(
                stitched[frame].tobytes()
                for frame in range(args.contact_start, args.contact_end_exclusive)
            )
        ).hexdigest()
        if args.quality_anchor_mode == "hard" and round_contact_digest != contact_digest:
            raise RuntimeError(f"deflicker round {index} changed the hard contact anchor")
        scorecard, evidence = _score(
            cv2,
            np,
            source,
            stitched,
            subject_masks,
            allowed_masks,
            object_masks,
            robot_reference,
            face_boxes,
            276,
            0,
            len(source),
            208,
            3,
        )
        transition = _transition_metrics(cv2, np, stitched)
        transition_energy = transition.pop("transition_energy")
        seam_ratios = {
            str(item["seam_frame"]): transition_energy[int(item["seam_frame"]) - 1]
            / max(float(transition["median_full_frame_transition_energy"]), 1e-6)
            for item in seams
        }
        output = output_dir / "rounds" / f"round-{index:02d}-{name}.mp4"
        _write_video(paths["ffmpeg"], output, stitched, fps)
        record = {
            "round": index,
            "name": name,
            "mutation": mutation,
            "scorecard": scorecard.to_dict(),
            "evidence": evidence,
            "transition": transition,
            "seam_transition_ratios": seam_ratios,
            "output": str(output),
            "output_sha256": _sha256(output),
            "contact_anchor_digest_preencode": round_contact_digest,
        }
        round_records.append(record)
        scored.append((scorecard, evidence, transition, output, index))
    baseline_motion = scored[0][0].motion_preservation
    eligible = [
        item
        for item in scored
        if item[0].background_lock >= thresholds.background_lock
        and item[0].object_lock >= thresholds.object_lock
        and float(item[1]["source_face_replacement"]) >= FACE_REPLACEMENT_THRESHOLD
        and item[0].motion_preservation >= baseline_motion - 0.03
    ]
    pool = eligible or scored
    selected = max(
        pool,
        key=lambda item: (
            max(round_records[item[4]]["seam_transition_ratios"].values()) <= 4.0,
            -max(round_records[item[4]]["seam_transition_ratios"].values()),
            item[0].temporal_consistency,
            item[0].epl_minimum,
            item[0].motion_preservation,
            item[0].robot_identity,
        ),
    )
    final_score, final_evidence, final_transition, selected_path, selected_round = selected
    selected_contact_unchanged = (
        round_records[selected_round]["contact_anchor_digest_preencode"]
        == contact_digest
    )
    if args.quality_anchor_mode == "hard" and not selected_contact_unchanged:
        raise RuntimeError("selected deflicker round changed the hard contact anchor")
    final = output_dir / "minimax-h3-epl-full-27s-deflickered.mp4"
    shutil.copy2(selected_path, final)
    subprocess.run(
        [str(paths["ffmpeg"]), "-v", "error", "-i", str(final), "-f", "null", "-"],
        check=True,
    )
    final_info = _count_decoded_frames(cv2, final)
    comparison = output_dir / "human-vs-minimax-h3-epl-full-27s.mp4"
    subprocess.run(
        [
            str(paths["ffmpeg"]), "-y", "-v", "error", "-i", str(source_aligned_path),
            "-i", str(final), "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
            "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(comparison),
        ],
        check=True,
    )
    comparison_info = _count_decoded_frames(cv2, comparison)
    subprocess.run(
        [
            str(paths["ffmpeg"]), "-y", "-v", "error", "-i", str(final), "-vf",
            "fps=28/27.5,scale=416:-2,tile=4x7:padding=4:margin=4:color=black",
            "-frames:v", "1", "-q:v", "2", str(output_dir / "dense-review.jpg"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(paths["ffmpeg"]), "-y", "-v", "error", "-i", str(comparison), "-vf",
            "fps=16/27.5,scale=832:-2,tile=4x4:padding=4:margin=4:color=black",
            "-frames:v", "1", "-q:v", "2", str(output_dir / "comparison-storyboard.jpg"),
        ],
        check=True,
    )
    review_passed = args.human_review == "passed"
    automatic_acceptance = {
        "full_clip_decoded": int(final_info["decoded_frames"]) == 660,
        "background_lock_passed": final_score.background_lock >= thresholds.background_lock,
        "object_lock_passed": final_score.object_lock >= thresholds.object_lock,
        "subject_replacement_passed": final_score.subject_replacement >= thresholds.subject_replacement,
        "source_face_replacement_passed": float(final_evidence["source_face_replacement"]) >= FACE_REPLACEMENT_THRESHOLD,
        "robot_identity_passed": final_score.robot_identity >= thresholds.robot_identity,
        "motion_preservation_passed": final_score.motion_preservation >= thresholds.motion_preservation,
        "temporal_consistency_passed": final_score.temporal_consistency >= thresholds.temporal_consistency,
        "epl_minimum_passed": final_score.epl_minimum >= thresholds.epl_minimum,
        "seam_transition_passed": max(
            round_records[selected_round]["seam_transition_ratios"].values()
        )
        <= 4.0,
    }
    if args.quality_anchor_mode == "hard":
        automatic_acceptance["contact_interval_unchanged_preencode"] = (
            selected_contact_unchanged
        )
        automatic_acceptance["hard_contact_anchor_exact_preencode"] = bool(
            contact_matches_anchor
        )
    accepted = all(automatic_acceptance.values()) and review_passed
    if accepted:
        review_status = "accepted"
    elif args.human_review == "pending":
        review_status = "review_required"
    else:
        review_status = "rejected"
    packages = {}
    for name in ("numpy", "opencv-contrib-python", "mediapipe"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": "1.0.0",
        "method": "minimax_h3_nf4_epl_recursive_continuation_agentic_deflicker_v2",
        "status": review_status,
        "honest_status": "WORKING" if accepted else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "packages": packages,
        "seed": args.seed,
        "gpu": {"used": False, "cuda_visible_devices": None, "reason": "OpenCV/MediaPipe CPU post-processing"},
        "git": _git_state(PROJECT_ROOT),
        "execution_sources": _freeze_execution_sources(output_dir),
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in paths.items()
        },
        "source_alignment_command": align_command,
        "source_info": source_info,
        "quality_anchor": {
            "mode": args.quality_anchor_mode,
            "applied_as_pixel_source": args.quality_anchor_mode == "hard",
            "start_frame": anchor_start,
            "end_frame_exclusive": anchor_end,
            "contact_interval": [args.contact_start, args.contact_end_exclusive],
            "decoded_info": anchor_info,
            "contact_matches_anchor_preencode": contact_matches_anchor,
            "contact_interval_unchanged_across_rounds": selected_contact_unchanged,
        },
        "physics_evidence": {
            "manifest_status": physics_manifest["status"],
            "task_success": physics["task_success"],
            "physically_valid": physics["physically_valid"],
            "metrics": physics["metrics"],
            "trajectory_source_epl": physics_trajectory["source_epl"],
            "trajectory_duration_s": physics_trajectory["timestamps_s"][-1],
            "trajectory_keyframes": len(physics_trajectory["timestamps_s"]),
            "end_effector_frame": physics_trajectory["embodiment"]["end_effector_frame"],
            "contact_events": physics["contact_events"],
            "scope": "separate authored MuJoCo trajectory; used for phase/contact gates, not as a visual pixel source",
        },
        "coordinate_frames": {
            "source": "camera:H3_output_pixels after recorded scale-and-center-crop",
            "pose": "camera:H3_output_pixels",
            "timeline": "absolute_frame_index:full_source_660",
        },
        "pose_tracking": pose_tracking,
        "face_tracking": face_tracking,
        "support_tracking": support_tracking,
        "windows": window_inputs,
        "subject_color_offsets": offsets,
        "protected_pixel_relock": relock_record,
        "seams": seams,
        "rounds": round_records,
        "selected_round": selected_round,
        "selected_scorecard": final_score.to_dict(),
        "selected_evidence": final_evidence,
        "selected_transition": final_transition,
        "thresholds": asdict(thresholds),
        "acceptance": {**automatic_acceptance, "human_review": args.human_review},
        "outputs": {
            "video": str(final),
            "video_sha256": _sha256(final),
            "video_info": final_info,
            "comparison": str(comparison),
            "comparison_sha256": _sha256(comparison),
            "comparison_info": comparison_info,
            "dense_review": str(output_dir / "dense-review.jpg"),
            "comparison_storyboard": str(output_dir / "comparison-storyboard.jpg"),
        },
        "limitations": [
            "The visual generator is third-party NF4 MiniMax-H3, not official BF16 H3, PhiZero, or a real-robot policy.",
            "The successful MuJoCo insertion is a separate pre-grasped authored 5-second trajectory; it validates phase/contact criteria, not this video's physical executability.",
            "H3 windows share only recursive Picture-2 continuation, not diffusion state; identity consistency and residual humans still require dense visual review.",
            "Flower preservation is conservative HSV segmentation and can miss pale or fully occluded petals.",
            "Pixel locks and contact-anchor equality are measured before final lossy H.264 encoding.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output": str(final),
                "comparison": str(comparison),
                "status": manifest["status"],
                "selected_round": selected_round,
                "scorecard": final_score.to_dict(),
                "acceptance": manifest["acceptance"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
