"""Mask-targeted temporal deghosting for generated robot videos."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MaskedDeghostConfig:
    strength: float = 7.0
    crf: int = 16
    preset: str = "slow"

    def __post_init__(self) -> None:
        if self.strength <= 0:
            raise ValueError("deghost strength must be positive")
        if not 0 <= self.crf <= 51:
            raise ValueError("CRF must be between 0 and 51")
        if not self.preset.strip():
            raise ValueError("encoder preset must be non-empty")


@dataclass(frozen=True)
class MaskedDeghostResult:
    output: Path
    metadata: Path


@dataclass(frozen=True)
class ObjectGhostRepairConfig:
    strength: float = 8.0
    intervals_s: tuple[tuple[float, float], ...] = ((1.47, 1.93), (2.33, 2.57))
    transition_frames: int = 3
    prior_dilation_pixels: int = 50

    def __post_init__(self) -> None:
        if self.strength <= 0:
            raise ValueError("repair strength must be positive")
        if self.transition_frames < 1:
            raise ValueError("transition_frames must be positive")
        if self.prior_dilation_pixels < 0:
            raise ValueError("prior_dilation_pixels cannot be negative")
        if not self.intervals_s or any(
            start < 0 or end <= start for start, end in self.intervals_s
        ):
            raise ValueError("repair intervals must be non-empty positive ranges")


def interval_weight(
    frame_index: int,
    fps: float,
    intervals_s: tuple[tuple[float, float], ...],
    transition_frames: int,
) -> float:
    """Return a ramped [0, 1] repair weight for one frame."""

    for start_s, end_s in intervals_s:
        start = round(start_s * fps)
        end = round(end_s * fps)
        if start <= frame_index <= end:
            return min(
                1.0,
                (frame_index - start + 1) / transition_frames,
                (end - frame_index + 1) / transition_frames,
            )
    return 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_deghost_filter(strength: float) -> str:
    return (
        "[0:v]split=2[base][filtered_input];"
        f"[filtered_input]hqdn3d={strength:g},split=2[filtered_character][filtered_object];"
        "[1:v]format=gray[character_mask];"
        "[base][filtered_character][character_mask]maskedmerge[character_filtered];"
        "[2:v]format=gray[object_mask];"
        "[character_filtered][filtered_object][object_mask]maskedmerge[out]"
    )


def deghost_video(
    *,
    candidate: Path,
    character_mask: Path,
    object_mask: Path,
    output: Path,
    config: MaskedDeghostConfig,
    overwrite: bool = False,
) -> MaskedDeghostResult:
    inputs = {
        "candidate": candidate.expanduser().resolve(),
        "character_mask": character_mask.expanduser().resolve(),
        "object_mask": object_mask.expanduser().resolve(),
    }
    for label, path in inputs.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    destination = output.expanduser().resolve()
    if destination.suffix.lower() != ".mp4":
        raise ValueError("deghost output must be an .mp4 file")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for mask-targeted deghosting")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-v",
        "error",
        "-i",
        str(inputs["candidate"]),
        "-i",
        str(inputs["character_mask"]),
        "-i",
        str(inputs["object_mask"]),
        "-filter_complex",
        build_deghost_filter(config.strength),
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-preset",
        config.preset,
        "-crf",
        str(config.crf),
        "-an",
        str(destination),
    ]
    subprocess.run(command, check=True)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create a non-empty output: {destination}")
    metadata_path = destination.with_suffix(".deghost.json")
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "command": command,
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in inputs.items()
        },
        "output": {
            "path": str(destination),
            "sha256": _sha256(destination),
        },
        "method": "masked_hqdn3d_character_and_object",
    }
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(metadata_path)
    return MaskedDeghostResult(destination, metadata_path)


def repair_object_ghosts(
    *,
    candidate: Path,
    source_video: Path,
    character_mask: Path,
    object_prior_mask: Path,
    output: Path,
    config: ObjectGhostRepairConfig,
    overwrite: bool = False,
) -> MaskedDeghostResult:
    """Restore clean source-object pixels only in diagnosed ghost intervals."""

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("object ghost repair requires OpenCV and NumPy") from exc
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for object ghost repair")
    paths = {
        "candidate": candidate.expanduser().resolve(),
        "source_video": source_video.expanduser().resolve(),
        "character_mask": character_mask.expanduser().resolve(),
        "object_prior_mask": object_prior_mask.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    destination = output.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = destination.parent / f"{destination.stem}-repair"
    work.mkdir(exist_ok=False)
    aligned_source = work / "source.mp4"
    repair_mask = work / "repair_mask.mp4"

    candidate_capture = cv2.VideoCapture(str(paths["candidate"]))
    fps = float(candidate_capture.get(cv2.CAP_PROP_FPS))
    width = int(candidate_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(candidate_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(candidate_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    candidate_capture.release()
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError("candidate video has invalid media properties")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(paths["source_video"]),
            "-vf",
            f"fps={fps:g},scale={width}:{height}",
            "-frames:v",
            str(frame_count),
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-an",
            str(aligned_source),
        ],
        check=True,
    )

    source_capture = cv2.VideoCapture(str(aligned_source))
    prior_capture = cv2.VideoCapture(str(paths["object_prior_mask"]))
    writer = cv2.VideoWriter(
        str(repair_mask),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        False,
    )
    close_kernel = np.ones((5, 5), np.uint8)
    dilation_size = 2 * config.prior_dilation_pixels + 1
    prior_kernel = np.ones((dilation_size, dilation_size), np.uint8)
    for frame_index in range(frame_count):
        source_ok, source_frame = source_capture.read()
        prior_ok, prior_frame = prior_capture.read()
        if not (source_ok and prior_ok):
            raise RuntimeError("source/object mask ended before the candidate")
        hsv = cv2.cvtColor(source_frame, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (12, 90, 70), (42, 255, 255))
        prior = cv2.cvtColor(prior_frame, cv2.COLOR_BGR2GRAY)
        allowed = cv2.dilate(prior, prior_kernel, iterations=1) > 10
        tracked = np.where(allowed, yellow, 0).astype(np.uint8)
        tracked = cv2.morphologyEx(
            tracked, cv2.MORPH_CLOSE, close_kernel, iterations=2
        )
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(tracked)
        clean = np.zeros_like(tracked)
        if component_count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            clean[labels == largest] = 255
        clean = cv2.dilate(clean, np.ones((3, 3), np.uint8), iterations=1)
        writer.write(clean)
    source_capture.release()
    prior_capture.release()
    writer.release()

    filtered_video = work / "filtered.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(paths["candidate"]),
            "-vf",
            f"hqdn3d={config.strength:g}",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-an",
            str(filtered_video),
        ],
        check=True,
    )
    raw_output = work / "rebuilt.mp4"
    captures = {
        "candidate": cv2.VideoCapture(str(paths["candidate"])),
        "filtered": cv2.VideoCapture(str(filtered_video)),
        "source": cv2.VideoCapture(str(aligned_source)),
        "repair_mask": cv2.VideoCapture(str(repair_mask)),
        "prior": cv2.VideoCapture(str(paths["object_prior_mask"])),
        "character": cv2.VideoCapture(str(paths["character_mask"])),
    }
    output_writer = cv2.VideoWriter(
        str(raw_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    broad_kernel = np.ones(
        (2 * config.prior_dilation_pixels + 1,) * 2, np.uint8
    )
    for frame_index in range(frame_count):
        decoded: dict[str, Any] = {}
        for label, capture in captures.items():
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"{label} ended before the candidate")
            decoded[label] = frame
        candidate_frame = decoded["candidate"]
        filtered_frame = decoded["filtered"]
        character = cv2.cvtColor(decoded["character"], cv2.COLOR_BGR2GRAY) > 127
        composed = candidate_frame.copy()
        composed[character] = filtered_frame[character]
        linear_weight = interval_weight(
            frame_index, fps, config.intervals_s, config.transition_frames
        )
        if linear_weight > 0:
            prior = cv2.cvtColor(decoded["prior"], cv2.COLOR_BGR2GRAY) > 20
            ys, xs = np.nonzero(prior)
            if len(xs):
                hsv = cv2.cvtColor(composed, cv2.COLOR_BGR2HSV)
                yellow = cv2.inRange(hsv, (12, 60, 45), (42, 255, 255)) > 0
                allowed = cv2.dilate(
                    prior.astype(np.uint8), broad_kernel, iterations=1
                ) > 0
                candidate_object = (yellow & allowed).astype(np.uint8) * 255
                count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    candidate_object
                )
                object_mask = np.zeros_like(candidate_object)
                if count > 1:
                    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    object_mask[labels == largest] = 255
                erase = cv2.dilate(
                    object_mask, np.ones((11, 11), np.uint8), iterations=1
                )
                rebuilt = cv2.inpaint(composed, erase, 5, cv2.INPAINT_TELEA)
                clean_object = (
                    cv2.cvtColor(decoded["repair_mask"], cv2.COLOR_BGR2GRAY) > 20
                )
                rebuilt[clean_object] = decoded["source"][clean_object]
                center_x = int(xs.mean())
                x_coordinates = np.arange(width)[None, :]
                robot = (
                    character
                    & (x_coordinates > center_x)
                    & (~yellow)
                )
                robot = cv2.morphologyEx(
                    robot.astype(np.uint8),
                    cv2.MORPH_CLOSE,
                    np.ones((5, 5), np.uint8),
                    iterations=1,
                ).astype(bool)
                rebuilt[robot] = composed[robot]
                weight = 0.5 - 0.5 * math.cos(math.pi * linear_weight)
                composed = cv2.addWeighted(
                    composed, 1 - weight, rebuilt, weight, 0
                )
        output_writer.write(composed)
    for capture in captures.values():
        capture.release()
    output_writer.release()

    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-v",
        "error",
        "-i",
        str(raw_output),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-an",
        str(destination),
    ]
    subprocess.run(command, check=True)
    metadata_path = destination.with_suffix(".deghost.json")
    payload = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "command": command,
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in paths.items()
        },
        "derived": {
            "aligned_source_sha256": _sha256(aligned_source),
            "repair_mask_sha256": _sha256(repair_mask),
        },
        "output": {"path": str(destination), "sha256": _sha256(destination)},
        "method": "layered_object_rebuild_with_cosine_transitions",
    }
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(metadata_path)
    return MaskedDeghostResult(destination, metadata_path)
