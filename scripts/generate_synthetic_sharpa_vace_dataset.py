#!/usr/bin/env python3
"""Generate an authorized procedural Sharpa VACE dataset from a rendered robot layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.data.adaptation import (  # noqa: E402
    AdaptationArm,
    AdaptationAsset,
    AdaptationAssetKind,
    AdaptationManifest,
    AdaptationSplit,
    VaceTrainingExample,
    file_sha256,
)

SHARPA_ASSET_REVISION = "6eea427eb24189519f32b9f21674cd534d3f973c"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-layer-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--train-clips", type=int, default=12)
    parser.add_argument("--validation-clips", type=int, default=4)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--neutral-control", action="store_true")
    return parser


def _decode_video(path: Path, ffmpeg: Path, width: int, height: int, frames: int) -> object:
    import numpy as np

    completed = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"scale={width}:{height}:flags=lanczos",
            "-frames:v",
            str(frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    frame_size = width * height * 3
    if len(completed.stdout) < frame_size or len(completed.stdout) % frame_size:
        raise ValueError("robot layer decoder returned an invalid RGB byte count")
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(-1, height, width, 3)


def _encode_video(frames: object, path: Path, ffmpeg: Path, fps: int) -> None:
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
            f"{frames.shape[2]}x{frames.shape[1]}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.write(frames.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed to encode {path}")


def _make_clip(
    robot_frames: object,
    *,
    clip_index: int,
    width: int,
    height: int,
    frame_count: int,
    seed: int,
) -> tuple[object, object]:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed + clip_index * 104729)
    yy, xx = np.mgrid[0:height, 0:width]
    top = rng.integers(20, 110, size=3)
    bottom = rng.integers(80, 210, size=3)
    gradient = (
        top[None, None, :] * (1 - yy[..., None] / max(1, height - 1))
        + bottom[None, None, :] * (yy[..., None] / max(1, height - 1))
    )
    texture = rng.normal(0, 7, size=(height, width, 1))
    background = np.clip(gradient + texture, 0, 255).astype(np.uint8)
    table_y = round(height * rng.uniform(0.62, 0.76))
    background[table_y:] = np.clip(
        background[table_y:].astype(np.int16) + rng.integers(15, 55),
        0,
        255,
    ).astype(np.uint8)

    targets: list[object] = []
    controls: list[object] = []
    source_count = len(robot_frames)
    for frame_index in range(frame_count):
        source_index = round(frame_index * (source_count - 1) / max(1, frame_count - 1))
        robot = robot_frames[source_index]
        mask = robot.max(axis=2) > 24
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise ValueError("robot layer contains an empty frame")
        crop = robot[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        crop_mask = crop.max(axis=2) > 24
        angle = float(90 + rng.uniform(-22, 22) + 4 * math.sin(frame_index / 3))
        scale = float(rng.uniform(0.72, 1.04))
        image = Image.fromarray(crop)
        alpha = Image.fromarray((crop_mask * 255).astype(np.uint8))
        size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
        image = image.resize(size, Image.Resampling.BICUBIC).rotate(
            angle,
            expand=True,
            resample=Image.Resampling.BICUBIC,
        )
        alpha = alpha.resize(size, Image.Resampling.BILINEAR).rotate(
            angle,
            expand=True,
            resample=Image.Resampling.BILINEAR,
        )
        foreground = np.asarray(image)
        foreground_alpha = np.asarray(alpha).astype(np.float32) / 255

        center_x = round(width * (0.48 + 0.08 * math.sin(frame_index / 4 + clip_index)))
        center_y = round(height * (0.57 + 0.04 * math.cos(frame_index / 5 + clip_index)))
        x0 = center_x - foreground.shape[1] // 2
        y0 = center_y - foreground.shape[0] // 2
        x1 = x0 + foreground.shape[1]
        y1 = y0 + foreground.shape[0]
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1, sy1 = foreground.shape[1] - max(0, x1 - width), foreground.shape[0] - max(
            0, y1 - height
        )
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)

        target = background.copy()
        geometry_mask = np.zeros((height, width), dtype=np.uint8)
        object_center = (
            round(width * (0.57 + 0.09 * math.sin(frame_index / 4 + clip_index))),
            round(height * (0.74 - 0.10 * frame_index / max(1, frame_count - 1))),
        )
        object_color = rng.integers(45, 220, size=3, dtype=np.uint8)
        object_width = round(width * rng.uniform(0.12, 0.20))
        object_height = round(height * rng.uniform(0.05, 0.10))
        oy0 = max(0, object_center[1] - object_height // 2)
        oy1 = min(height, oy0 + object_height)
        ox0 = max(0, object_center[0] - object_width // 2)
        ox1 = min(width, ox0 + object_width)
        target[oy0:oy1, ox0:ox1] = object_color
        geometry_mask[oy0:oy1, ox0:ox1] = 255

        if dx0 < dx1 and dy0 < dy1:
            local_alpha = foreground_alpha[sy0:sy1, sx0:sx1, None]
            target[dy0:dy1, dx0:dx1] = np.rint(
                local_alpha * foreground[sy0:sy1, sx0:sx1]
                + (1 - local_alpha) * target[dy0:dy1, dx0:dx1]
            ).astype(np.uint8)
            geometry_mask[dy0:dy1, dx0:dx1] = np.maximum(
                geometry_mask[dy0:dy1, dx0:dx1],
                np.rint(local_alpha[..., 0] * 255).astype(np.uint8),
            )

        edge_x = np.abs(
            np.diff(geometry_mask.astype(np.int16), axis=1, prepend=geometry_mask[:, :1])
        )
        edge_y = np.abs(
            np.diff(geometry_mask.astype(np.int16), axis=0, prepend=geometry_mask[:1])
        )
        edges = np.clip(edge_x + edge_y, 0, 255).astype(np.uint8)
        control = np.repeat(edges[..., None], 3, axis=2)
        targets.append(target)
        controls.append(control)
    return np.stack(targets), np.stack(controls)


def main() -> int:
    args = _parser().parse_args()
    if min(
        args.train_clips,
        args.validation_clips,
        args.frames,
        args.fps,
        args.width,
        args.height,
    ) <= 0:
        raise ValueError("dataset sizes, video dimensions, and FPS must be positive")
    if (args.frames - 1) % 4:
        raise ValueError("frames must satisfy 4n+1")
    robot_video = args.robot_layer_video.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if not robot_video.is_file() or not ffmpeg.is_file():
        raise ValueError("robot layer and ffmpeg must exist")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    robot_frames = _decode_video(robot_video, ffmpeg, args.width, args.height, args.frames)

    assets: list[AdaptationAsset] = []
    training_examples: list[VaceTrainingExample] = []
    validation_records: list[dict[str, str]] = []
    total = args.train_clips + args.validation_clips
    for clip_index in range(total):
        split = (
            AdaptationSplit.TRAIN
            if clip_index < args.train_clips
            else AdaptationSplit.VALIDATION
        )
        clip_dir = output / split.value / f"clip-{clip_index:03d}"
        clip_dir.mkdir(parents=True)
        targets, controls = _make_clip(
            robot_frames,
            clip_index=clip_index,
            width=args.width,
            height=args.height,
            frame_count=args.frames,
            seed=args.seed,
        )
        if args.neutral_control:
            controls.fill(0)
        target = clip_dir / "target.mp4"
        control = clip_dir / "control.mp4"
        reference = clip_dir / "reference.png"
        _encode_video(targets, target, ffmpeg, args.fps)
        _encode_video(controls, control, ffmpeg, args.fps)
        from PIL import Image

        Image.fromarray(targets[0]).save(reference)
        prefix = f"{split.value}-{clip_index:03d}"
        clip_assets = (
            AdaptationAsset(
                f"{prefix}-target",
                str(target),
                split,
                AdaptationAssetKind.TARGET_VIDEO,
                f"local://procedural-sharpa/{prefix}/target",
                f"derived only from Apache-2.0 Sharpa revision {SHARPA_ASSET_REVISION}",
                file_sha256(target),
                target.stat().st_size,
                True,
            ),
            AdaptationAsset(
                f"{prefix}-control",
                str(control),
                split,
                AdaptationAssetKind.VACE_CONTROL_VIDEO,
                f"local://procedural-sharpa/{prefix}/control",
                f"derived only from Apache-2.0 Sharpa revision {SHARPA_ASSET_REVISION}",
                file_sha256(control),
                control.stat().st_size,
                True,
            ),
            AdaptationAsset(
                f"{prefix}-reference",
                str(reference),
                split,
                AdaptationAssetKind.VACE_REFERENCE_IMAGE,
                f"local://procedural-sharpa/{prefix}/reference",
                f"derived only from Apache-2.0 Sharpa revision {SHARPA_ASSET_REVISION}",
                file_sha256(reference),
                reference.stat().st_size,
                True,
            ),
        )
        assets.extend(clip_assets)
        if split is AdaptationSplit.TRAIN:
            training_examples.append(
                VaceTrainingExample(
                    f"clip-{clip_index:03d}",
                    clip_assets[0].asset_id,
                    clip_assets[1].asset_id,
                    clip_assets[2].asset_id,
                    "A Sharpa dexterous robot hand manipulates an object.",
                )
            )
        else:
            validation_records.append(
                {
                    "clip_id": f"clip-{clip_index:03d}",
                    "target": str(target),
                    "control": str(control),
                    "reference": str(reference),
                }
            )
    manifest = AdaptationManifest(
        experiment_id=output.name,
        arm=AdaptationArm.VACE_LORA,
        assets=tuple(assets),
        vace_examples=tuple(training_examples),
        evidence_scope="development_only",
    )
    manifest.write_json(output / "frozen" / "manifest.json")
    (output / "validation.json").write_text(
        json.dumps(validation_records, indent=2, sort_keys=True) + "\n"
    )
    provenance = {
        "schema_version": "1.0.0",
        "robot_layer": str(robot_video),
        "robot_layer_sha256": file_sha256(robot_video),
        "sharpa_asset_revision": SHARPA_ASSET_REVISION,
        "seed": args.seed,
        "train_clips": args.train_clips,
        "validation_clips": args.validation_clips,
        "frames": args.frames,
        "fps": args.fps,
        "resolution": [args.width, args.height],
        "neutral_control": args.neutral_control,
        "manifest_sha256": hashlib.sha256(
            (output / "frozen" / "manifest.json").read_bytes()
        ).hexdigest(),
        "limitations": [
            "Procedural MuJoCo-derived imagery validates geometry-conditioned training only.",
            "It does not provide photorealistic real-robot supervision.",
        ],
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(f"DATASET={output}")
    print(f"MANIFEST={output / 'frozen' / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
