#!/usr/bin/env python3
"""Build one RGB-continuous transition and encode it exactly once."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _smootherstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value**3 * (value * (value * 6 - 15) + 10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--morph-frames", type=int, default=46)
    parser.add_argument("--hold-frames", type=int, default=9)
    parser.add_argument(
        "--stable-start",
        action="store_true",
        help="ignore reference morphing and hold the exact extension first frame",
    )
    args = parser.parse_args()
    if min(
        args.fps,
        args.width,
        args.height,
        args.morph_frames,
        args.hold_frames,
    ) <= 0:
        raise ValueError("video settings must be positive")
    if args.morph_frames < 2:
        raise ValueError("morph-frames must include at least two endpoints")
    reference_path = args.reference.expanduser().resolve()
    extension_path = args.extension.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if not reference_path.is_file() or not extension_path.is_file() or not ffmpeg.is_file():
        raise ValueError("reference, extension, and ffmpeg must exist")

    import numpy as np
    from PIL import Image

    reference = np.asarray(
        Image.open(reference_path)
        .convert("RGB")
        .resize((args.width, args.height), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    decoded = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(extension_path),
            "-vf",
            f"fps={args.fps},scale={args.width}:{args.height}:flags=lanczos",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    frame_size = args.width * args.height * 3
    if not decoded or len(decoded) % frame_size:
        raise ValueError("extension decoder returned an invalid RGB byte count")
    extension = np.frombuffer(decoded, dtype=np.uint8).reshape(
        -1, args.height, args.width, 3
    )
    first = extension[0].astype(np.float32)
    if args.stable_start:
        morph = [extension[0].copy() for _ in range(args.morph_frames)]
    else:
        morph = []
        for index in range(args.morph_frames):
            alpha = _smootherstep(index / (args.morph_frames - 1))
            morph.append(
                np.rint((1 - alpha) * reference + alpha * first).astype(np.uint8)
            )
    frames = np.concatenate(
        (
            np.stack(morph),
            np.repeat(extension[:1], args.hold_frames, axis=0),
            extension[1:],
        ),
        axis=0,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{args.width}x{args.height}",
            "-r",
            str(args.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.write(frames.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError("ffmpeg failed to encode zero-jump transition")
    print(f"VIDEO={output}")
    print(f"FRAME_COUNT={len(frames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
