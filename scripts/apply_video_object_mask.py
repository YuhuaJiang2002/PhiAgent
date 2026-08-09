#!/usr/bin/env python3
"""Apply a persisted object mask to repair one object in a generated video."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--object-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    args = parser.parse_args()
    for path in (args.source, args.candidate, args.object_mask):
        if not path.is_file():
            raise ValueError(f"input does not exist: {path}")
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")

    ffmpeg = args.ffmpeg or Path(shutil.which("ffmpeg") or "")
    if not ffmpeg.is_file():
        try:
            import imageio_ffmpeg

            ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
        except ImportError as exc:
            raise RuntimeError("ffmpeg or imageio-ffmpeg is required") from exc
    decode_options = {
        "ffmpeg": ffmpeg,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "frame_num": args.frame_num,
    }
    source = RGBFrames(
        decode_video(args.source, **decode_options, pixel_format="rgb24"),
        args.width,
        args.height,
    )
    candidate = RGBFrames(
        decode_video(args.candidate, **decode_options, pixel_format="rgb24"),
        args.width,
        args.height,
    )
    raw_masks = decode_video(
        args.object_mask, **decode_options, pixel_format="gray"
    )
    masks = tuple(bytes(value >= 128 for value in mask) for mask in raw_masks)
    config = ObjectTrackerConfig(
        initial_roi=NormalizedROI(*args.object_roi),
        initial_color_mode="cyan",
    )
    track = track_colored_object(source, config)
    cleaned, removed = remove_duplicate_colored_objects(
        source, candidate, track, config
    )
    mask_track = ObjectTrack(
        masks=masks,
        boxes=track.boxes,
        mean_colors=track.mean_colors,
        areas=tuple(sum(mask) for mask in masks),
        model=track.model,
        width=track.width,
        height=track.height,
    )
    repaired = composite_source_object(source, cleaned, mask_track)
    encode_video(
        repaired.frames,
        args.output,
        ffmpeg,
        width=args.width,
        height=args.height,
        fps=args.fps,
        pixel_format="rgb24",
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
                "object_mask": str(args.object_mask.resolve()),
                "object_mask_sha256": _sha256(args.object_mask),
                "output": str(args.output.resolve()),
                "output_sha256": _sha256(args.output),
                "tracker_config": asdict(config),
                "object_pixels": tuple(sum(mask) for mask in masks),
                "duplicate_pixels_removed": removed,
                "limitations": [
                    "The supplied instance mask determines restored source pixels.",
                    "The result is image-space compositing, not physical contact verification.",
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
