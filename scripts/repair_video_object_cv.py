#!/usr/bin/env python3
"""Repair one colored video object with seeded GrabCut instance masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.evaluation.object_instance import (  # noqa: E402
    NormalizedROI,
    ObjectTrack,
    ObjectTrackerConfig,
    RGBFrames,
    composite_source_object,
    decode_video,
    encode_video,
    remove_duplicate_colored_objects,
    track_colored_object,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grabcut_masks(
    source: RGBFrames,
    track: ObjectTrack,
    *,
    margin: int,
    iterations: int,
) -> tuple[bytes, ...]:
    masks: list[bytes] = []
    for frame, seed, box in zip(source.frames, track.masks, track.boxes):
        if box is None:
            masks.append(seed)
            continue
        image = np.frombuffer(frame, dtype=np.uint8).reshape(
            source.height, source.width, 3
        )
        mask = np.full((source.height, source.width), cv2.GC_BGD, dtype=np.uint8)
        x0 = max(0, box[0] - margin)
        y0 = max(0, box[1] - margin)
        x1 = min(source.width, box[2] + margin)
        y1 = min(source.height, box[3] + margin)
        mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
        seed_array = np.frombuffer(seed, dtype=np.uint8).reshape(
            source.height, source.width
        )
        mask[seed_array > 0] = cv2.GC_FGD
        background = np.zeros((1, 65), dtype=np.float64)
        foreground = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            image,
            mask,
            None,
            background,
            foreground,
            iterations,
            cv2.GC_INIT_WITH_MASK,
        )
        candidate = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
        count, labels = cv2.connectedComponents(candidate, connectivity=8)
        seed_labels = {
            int(label)
            for label in labels[seed_array > 0]
            if 0 < int(label) < count
        }
        selected = np.isin(labels, tuple(seed_labels)).astype(np.uint8)
        selected = cv2.morphologyEx(
            selected,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        masks.append(bytes(selected.reshape(-1)))
    return tuple(masks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--object-roi",
        type=float,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        required=True,
    )
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-num", type=int, default=77)
    parser.add_argument("--grabcut-margin", type=int, default=16)
    parser.add_argument("--grabcut-iterations", type=int, default=3)
    parser.add_argument(
        "--color-mode",
        choices=("chromatic", "cyan"),
        default="cyan",
    )
    args = parser.parse_args()
    for path in (args.source, args.candidate):
        if not path.is_file():
            raise ValueError(f"input does not exist: {path}")
    for path in (args.output, args.mask_output, args.report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")
    if min(
        args.width,
        args.height,
        args.fps,
        args.frame_num,
        args.grabcut_iterations,
    ) <= 0:
        raise ValueError("dimensions, FPS, frame count, and iterations must be positive")
    if args.grabcut_margin < 0:
        raise ValueError("grabcut margin must be non-negative")

    ffmpeg = args.ffmpeg or Path(shutil.which("ffmpeg") or "")
    if not ffmpeg.is_file():
        try:
            import imageio_ffmpeg

            ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
        except ImportError as exc:
            raise RuntimeError("ffmpeg or imageio-ffmpeg is required") from exc

    source = RGBFrames(
        decode_video(
            args.source,
            ffmpeg,
            width=args.width,
            height=args.height,
            fps=args.fps,
            frame_num=args.frame_num,
            pixel_format="rgb24",
        ),
        args.width,
        args.height,
    )
    candidate = RGBFrames(
        decode_video(
            args.candidate,
            ffmpeg,
            width=args.width,
            height=args.height,
            fps=args.fps,
            frame_num=args.frame_num,
            pixel_format="rgb24",
        ),
        args.width,
        args.height,
    )
    config = ObjectTrackerConfig(
        initial_roi=NormalizedROI(*args.object_roi),
        initial_color_mode=args.color_mode,
    )
    track = track_colored_object(source, config)
    masks = _grabcut_masks(
        source,
        track,
        margin=args.grabcut_margin,
        iterations=args.grabcut_iterations,
    )
    cleaned, removed = remove_duplicate_colored_objects(
        source, candidate, track, config
    )
    complete_track = ObjectTrack(
        masks=masks,
        boxes=track.boxes,
        mean_colors=track.mean_colors,
        areas=tuple(sum(mask) for mask in masks),
        model=track.model,
        width=track.width,
        height=track.height,
    )
    repaired = composite_source_object(source, cleaned, complete_track)
    encode_video(
        repaired.frames,
        args.output,
        ffmpeg,
        width=args.width,
        height=args.height,
        fps=args.fps,
        pixel_format="rgb24",
    )
    encode_video(
        tuple(bytes(255 if value else 0 for value in mask) for mask in masks),
        args.mask_output,
        ffmpeg,
        width=args.width,
        height=args.height,
        fps=args.fps,
        pixel_format="gray",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "source": str(args.source.resolve()),
                "source_sha256": _sha256(args.source),
                "candidate": str(args.candidate.resolve()),
                "candidate_sha256": _sha256(args.candidate),
                "output": str(args.output.resolve()),
                "output_sha256": _sha256(args.output),
                "mask_output": str(args.mask_output.resolve()),
                "tracker_config": asdict(config),
                "grabcut_margin": args.grabcut_margin,
                "grabcut_iterations": args.grabcut_iterations,
                "high_confidence_pixels": tuple(sum(mask) for mask in track.masks),
                "grabcut_pixels": tuple(sum(mask) for mask in masks),
                "duplicate_pixels_removed": removed,
                "limitations": [
                    "GrabCut is seeded by the configured colored-object tracker.",
                    "The repair is image-space compositing, not physical contact verification.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"OUTPUT={args.output.resolve()}")
    print(f"REPORT={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
