"""Apply a material-only edit through a tracked hand-instance mask."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from phiagent.evaluation.object_instance import (
    NormalizedROI,
    ObjectTrackerConfig,
    RGBFrames,
    route_object_preservation,
)


@dataclass(frozen=True)
class GraphiteHandConfig:
    blue_scale: float = 0.18
    green_scale: float = 0.22
    red_scale: float = 0.25
    opacity: float = 0.94

    def __post_init__(self) -> None:
        values = (self.blue_scale, self.green_scale, self.red_scale, self.opacity)
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("graphite material values must be between zero and one")
        if self.opacity == 0:
            raise ValueError("graphite material opacity must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_graphite_hand_style(
    *,
    candidate: Path,
    hand_mask: Path,
    output: Path,
    object_roi: NormalizedROI,
    config: GraphiteHandConfig,
) -> Path:
    inputs = {
        "candidate": candidate.expanduser().resolve(),
        "hand_mask": hand_mask.expanduser().resolve(),
    }
    for label, path in inputs.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("styled hand output must be an .mp4 file")

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("tracked hand styling requires OpenCV and NumPy") from exc

    def read_video(path: Path) -> tuple[list[object], float, int, int]:
        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames: list[object] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if not frames or fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"could not decode video: {path}")
        return frames, fps, width, height

    frames, fps, width, height = read_video(inputs["candidate"])
    masks, mask_fps, mask_width, mask_height = read_video(inputs["hand_mask"])
    if (mask_width, mask_height) != (width, height) or abs(mask_fps - fps) > 1e-3:
        raise ValueError("hand mask dimensions and FPS must match the candidate")
    if len(masks) < len(frames):
        raise ValueError("hand mask has fewer frames than the candidate")

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    output_frames: list[object] = []
    for frame, mask_frame in zip(frames, masks):
        mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )
        alpha = (
            cv2.GaussianBlur(mask, (9, 9), 0).astype(np.float32)[..., None]
            / 255
            * config.opacity
        )
        luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        graphite = np.stack(
            (
                luminance * config.blue_scale,
                luminance * config.green_scale,
                luminance * config.red_scale,
            ),
            axis=2,
        )
        styled = np.clip(
            frame.astype(np.float32) * (1 - alpha) + graphite * alpha,
            0,
            255,
        ).astype(np.uint8)
        writer.write(styled)
        output_frames.append(styled)
    writer.release()

    source_rgb = RGBFrames(
        tuple(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes() for frame in frames),
        width,
        height,
    )
    output_rgb = RGBFrames(
        tuple(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()
            for frame in output_frames
        ),
        width,
        height,
    )
    route = route_object_preservation(
        source_rgb,
        output_rgb,
        ObjectTrackerConfig(initial_roi=object_roi),
        maximum_candidate_area_ratio=3.0,
    )
    metadata = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "sam2_hand_instance_material_edit_with_object_confidence_routing",
        "config": asdict(config),
        "object_roi": asdict(object_roi),
        "route": asdict(route),
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in inputs.items()
        },
        "output": {"path": str(destination), "sha256": _sha256(destination)},
    }
    metadata_path = destination.with_suffix(".style.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata_path
