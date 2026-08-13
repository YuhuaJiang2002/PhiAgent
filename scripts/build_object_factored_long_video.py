#!/usr/bin/env python3
"""Build a long robot-replacement video without regenerating known objects.

The source scene is the base layer.  The generated candidate is used only on a
tracked/dilated source-person support.  Visible source flowers are projected
back last, except where a conservative source-skin negative says that the
source hand was in front.  This makes background and flower preservation a
construction invariant rather than a diffusion-model preference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.object_factored_long_video import (  # noqa: E402
    SourceResizeCrop,
    binary_dilate_square,
    binary_erode_square,
    compose_object_factored_frame,
    remap_boolean_mask,
    resolve_flower_visibility,
    source_skin_like,
    strict_flower_seed,
    validate_visibility_partition,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--generated-video", type=Path, required=True)
    parser.add_argument("--source-person-masks", type=Path, required=True)
    parser.add_argument("--source-flower-masks", type=Path, required=True)
    parser.add_argument("--source-hand-masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("ffprobe"))
    parser.add_argument("--person-dilation", type=int, default=10)
    parser.add_argument("--skin-negative-dilation", type=int, default=2)
    parser.add_argument("--person-core-negative-erosion", type=int, default=2)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--mask-frame-name", required=True)
    parser.add_argument("--mask-source-width", type=int, required=True)
    parser.add_argument("--mask-source-height", type=int, required=True)
    parser.add_argument("--mask-scaled-width", type=int, required=True)
    parser.add_argument("--mask-scaled-height", type=int, required=True)
    parser.add_argument("--mask-crop-left", type=int, required=True)
    parser.add_argument("--mask-crop-top", type=int, required=True)
    parser.add_argument("--target-frame-name", required=True)
    parser.add_argument("--target-scaled-width", type=int, required=True)
    parser.add_argument("--target-scaled-height", type=int, required=True)
    parser.add_argument("--target-crop-left", type=int, required=True)
    parser.add_argument("--target-crop-top", type=int, required=True)
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--target-height", type=int, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(ffprobe: Path, video: Path) -> dict[str, Any]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,nb_read_frames,duration:format=duration",
        "-of",
        "json",
        str(video),
    ]
    raw = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    stream = raw["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/", 1)
    frame_value = stream.get("nb_frames")
    if frame_value in (None, "N/A"):
        frame_value = stream.get("nb_read_frames")
    if frame_value in (None, "N/A"):
        raise ValueError(f"ffprobe did not report a frame count for {video}")
    duration_value = stream.get("duration")
    if duration_value in (None, "N/A"):
        duration_value = raw.get("format", {}).get("duration")
    if duration_value in (None, "N/A"):
        duration_value = float(frame_value) / (float(numerator) / float(denominator))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
        "frames": int(frame_value),
        "duration": float(duration_value),
    }


def _load_packed(np: Any, path: Path, key: str = "packed") -> tuple[Any, dict[str, Any]]:
    payload = np.load(path, allow_pickle=False)
    if key not in payload.files:
        raise ValueError(f"packed mask file {path} has no {key!r} layer")
    height = int(payload["height"])
    width = int(payload["width"])
    bitorder = str(payload["bitorder"])
    packed = payload[key]
    # The contract is frame-contiguous: every row is the flattened bitstream
    # for one complete HxW mask.  Reject row-padded or higher-rank encodings
    # instead of accepting a reshape-compatible but spatially scrambled mask.
    if packed.ndim != 2 or packed.shape[1] * 8 < height * width:
        raise ValueError("packed mask must be a frame-contiguous frames-by-bytes array")
    unpacked = np.unpackbits(packed, axis=1, bitorder=bitorder)
    masks = unpacked[:, : height * width].reshape(len(packed), height, width).astype(bool)
    return masks, {
        "key": key,
        "frames": int(len(masks)),
        "width": width,
        "height": height,
        "bitorder": bitorder,
    }


def _read_frame(process: subprocess.Popen[bytes], byte_count: int, label: str, index: int) -> bytes:
    assert process.stdout is not None
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = process.stdout.read(remaining)
        if not chunk:
            raise RuntimeError(f"{label} decoder ended before frame {index}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _writer_command(
    ffmpeg: Path,
    *,
    output: Path,
    width: int,
    height: int,
    fps: float,
    lossless: bool,
) -> list[str]:
    base = [
        str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "-", "-an",
    ]
    if lossless:
        return [*base, "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "bgr0", str(output)]
    return [
        *base, "-c:v", "libx264", "-preset", "medium", "-crf", "10",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]


def _verify_encoded_projection(
    np: Any,
    *,
    ffmpeg: Path,
    encoded_video: Path,
    source_decode_command: list[str],
    expected_frames: int,
    width: int,
    height: int,
    mask_frame: SourceResizeCrop,
    target_frame: SourceResizeCrop,
    person_masks: Any,
    flower_masks: Any,
    hand_masks: Any,
    person_dilation: int,
    skin_negative_dilation: int,
    person_core_negative_erosion: int,
) -> dict[str, float | int]:
    """Decode an encoded result and repeat the construction-invariant gates."""

    output_command = [
        str(ffmpeg), "-v", "error", "-i", str(encoded_video), "-an", "-f",
        "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    source_decoder = subprocess.Popen(
        source_decode_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output_decoder = subprocess.Popen(
        output_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    frame_bytes = width * height * 3
    known_exact = 0
    known_values = 0
    flower_abs_error = 0
    flower_values = 0
    flower_person_core_overlap = 0
    flower_skin_overlap = 0
    temporal_abs_error = 0
    temporal_values = 0
    previous_source = None
    previous_output = None
    previous_flower = None
    for index in range(expected_frames):
        source_rgb = np.frombuffer(
            _read_frame(source_decoder, frame_bytes, "verification source", index),
            dtype=np.uint8,
        ).reshape(height, width, 3)
        output_rgb = np.frombuffer(
            _read_frame(output_decoder, frame_bytes, "verification output", index),
            dtype=np.uint8,
        ).reshape(height, width, 3)
        person = remap_boolean_mask(
            np, person_masks[index], source_frame=mask_frame, target_frame=target_frame
        )
        tracked_flower = remap_boolean_mask(
            np, flower_masks[index], source_frame=mask_frame, target_frame=target_frame
        )
        source_hand = remap_boolean_mask(
            np, hand_masks[index], source_frame=mask_frame, target_frame=target_frame
        )
        support = binary_dilate_square(np, person, person_dilation)
        strict = strict_flower_seed(np, source_rgb)
        skin = source_skin_like(np, source_rgb) & (person | source_hand)
        skin = binary_dilate_square(np, skin & support, skin_negative_dilation)
        flower = resolve_flower_visibility(
            np,
            candidates=tracked_flower | strict,
            edit_support=support,
            source_person=person,
            source_skin_negative=skin,
            person_core_erosion=person_core_negative_erosion,
        )
        person_core = binary_erode_square(np, person, person_core_negative_erosion)
        validate_visibility_partition(
            np,
            edit_support=support,
            flower_restore=flower,
            source_person_core=person_core,
            source_skin_negative=skin,
        )
        support = np.asarray(support, dtype=np.bool_).copy()
        flower = np.asarray(flower, dtype=np.bool_).copy()
        support.setflags(write=False)
        flower.setflags(write=False)
        known = ~support | flower
        exact = np.all(output_rgb == source_rgb, axis=2)
        known_exact += int(np.count_nonzero(exact & known))
        known_values += int(np.count_nonzero(known))
        if np.any(flower):
            difference = np.abs(
                output_rgb[flower].astype(np.int16) - source_rgb[flower].astype(np.int16)
            )
            flower_abs_error += int(difference.sum())
            flower_values += int(difference.size)
        flower_person_core_overlap += int(np.count_nonzero(flower & person_core))
        flower_skin_overlap += int(np.count_nonzero(flower & skin))
        if previous_source is not None:
            temporal_mask = flower & previous_flower
            if np.any(temporal_mask):
                source_delta = source_rgb.astype(np.int16) - previous_source.astype(np.int16)
                output_delta = output_rgb.astype(np.int16) - previous_output.astype(np.int16)
                residual = np.abs(output_delta[temporal_mask] - source_delta[temporal_mask])
                temporal_abs_error += int(residual.sum())
                temporal_values += int(residual.size)
        previous_source = source_rgb.copy()
        previous_output = output_rgb.copy()
        previous_flower = flower.copy()

    for label, process in (("source", source_decoder), ("output", output_decoder)):
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.wait():
            raise RuntimeError(f"encoded {label} verification failed: {stderr[-1000:]}")
    return {
        "known_source_exact_fraction": known_exact / max(1, known_values),
        "flower_mad": flower_abs_error / max(1, flower_values),
        "flower_temporal_residual_mad": temporal_abs_error / max(1, temporal_values),
        "flower_person_core_overlap_pixels": flower_person_core_overlap,
        "flower_skin_negative_overlap_pixels": flower_skin_overlap,
    }


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--short"),
    }


def main() -> int:
    args = _parser().parse_args()
    if (
        args.person_dilation < 0
        or args.skin_negative_dilation < 0
        or args.person_core_negative_erosion < 0
    ):
        raise ValueError("dilation radii must be non-negative")
    if args.expected_frames <= 0 or args.fps <= 0:
        raise ValueError("expected frames and FPS must be positive")

    import numpy as np

    source = args.source_video.expanduser().resolve()
    generated = args.generated_video.expanduser().resolve()
    person_path = args.source_person_masks.expanduser().resolve()
    flower_path = args.source_flower_masks.expanduser().resolve()
    hand_path = args.source_hand_masks.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    ffprobe = args.ffprobe.expanduser().resolve()
    for path in (source, generated, person_path, flower_path, hand_path, ffmpeg, ffprobe):
        if not path.is_file():
            raise ValueError(f"required input is missing: {path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    log_dir = output_dir / "logs"
    log_dir.mkdir()
    review_path = output_dir / "object-factored-27p5s.mp4"
    lossless_path = output_dir / "object-factored-27p5s-lossless.mkv"

    source_info = _probe(ffprobe, source)
    generated_info = _probe(ffprobe, generated)
    if source_info["frames"] != args.expected_frames or generated_info["frames"] != args.expected_frames:
        raise ValueError(
            f"expected {args.expected_frames} frames, got source={source_info['frames']} "
            f"generated={generated_info['frames']}"
        )
    if abs(source_info["fps"] - args.fps) > 1e-6 or abs(generated_info["fps"] - args.fps) > 1e-6:
        raise ValueError("source and generated videos must match the declared FPS")
    if (generated_info["width"], generated_info["height"]) != (
        args.target_width,
        args.target_height,
    ):
        raise ValueError("generated video dimensions do not match the target coordinate frame")

    mask_frame = SourceResizeCrop(
        name=args.mask_frame_name,
        source_width=args.mask_source_width,
        source_height=args.mask_source_height,
        scaled_width=args.mask_scaled_width,
        scaled_height=args.mask_scaled_height,
        crop_left=args.mask_crop_left,
        crop_top=args.mask_crop_top,
        output_width=0,
        output_height=0,
    )
    person_masks, person_info = _load_packed(np, person_path)
    flower_masks, flower_info = _load_packed(np, flower_path)
    hand_masks, hand_info = _load_packed(np, hand_path, key="hands_packed")
    if (
        person_info["frames"] != args.expected_frames
        or flower_info["frames"] != args.expected_frames
        or hand_info["frames"] != args.expected_frames
    ):
        raise ValueError("packed source masks must cover the complete declared timeline")
    if (person_info["width"], person_info["height"]) != (
        flower_info["width"],
        flower_info["height"],
    ):
        raise ValueError("person and flower masks use different pixel frames")
    if (person_info["width"], person_info["height"]) != (
        hand_info["width"],
        hand_info["height"],
    ):
        raise ValueError("person and hand masks use different pixel frames")
    mask_frame = SourceResizeCrop(
        **{
            **mask_frame.to_dict(),
            "output_width": person_info["width"],
            "output_height": person_info["height"],
        }
    )
    target_frame = SourceResizeCrop(
        name=args.target_frame_name,
        source_width=args.mask_source_width,
        source_height=args.mask_source_height,
        scaled_width=args.target_scaled_width,
        scaled_height=args.target_scaled_height,
        crop_left=args.target_crop_left,
        crop_top=args.target_crop_top,
        output_width=args.target_width,
        output_height=args.target_height,
    )
    mask_frame.validate()
    target_frame.validate()
    if (source_info["width"], source_info["height"]) != (
        target_frame.source_width,
        target_frame.source_height,
    ):
        raise ValueError("source video dimensions do not match the named camera source")

    source_filter = (
        f"scale={target_frame.scaled_width}:{target_frame.scaled_height}:flags=area,"
        f"crop={target_frame.output_width}:{target_frame.output_height}:"
        f"{target_frame.crop_left}:{target_frame.crop_top}"
    )
    source_decode_command = [
        str(ffmpeg), "-v", "error", "-i", str(source), "-vf", source_filter,
        "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    generated_decode_command = [
        str(ffmpeg), "-v", "error", "-i", str(generated), "-an", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-",
    ]
    review_write_command = _writer_command(
        ffmpeg, output=review_path, width=args.target_width, height=args.target_height,
        fps=args.fps, lossless=False,
    )
    lossless_write_command = _writer_command(
        ffmpeg, output=lossless_path, width=args.target_width, height=args.target_height,
        fps=args.fps, lossless=True,
    )

    tic = time.perf_counter()
    source_decoder = subprocess.Popen(source_decode_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    generated_decoder = subprocess.Popen(
        generated_decode_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    review_writer = subprocess.Popen(review_write_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    lossless_writer = subprocess.Popen(lossless_write_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = args.target_width * args.target_height * 3
    known_exact_pixels = 0
    total_pixels = 0
    flower_restore_pixels = 0
    flower_track_pixels = 0
    strict_seed_pixels = 0
    skin_negative_pixels = 0
    support_pixels = 0
    input_flower_abs_error = 0
    input_flower_value_count = 0
    output_flower_abs_error = 0
    output_flower_value_count = 0
    empty_flower_track_frames: list[int] = []
    per_frame: list[dict[str, Any]] = []
    previous_source = None
    previous_generated = None
    previous_output = None
    previous_flower = None
    input_flower_temporal_residual_sum = 0.0
    output_flower_temporal_residual_sum = 0.0
    temporal_value_count = 0
    try:
        assert review_writer.stdin is not None and lossless_writer.stdin is not None
        for index in range(args.expected_frames):
            source_rgb = np.frombuffer(
                _read_frame(source_decoder, frame_bytes, "source", index), dtype=np.uint8
            ).reshape(args.target_height, args.target_width, 3)
            generated_rgb = np.frombuffer(
                _read_frame(generated_decoder, frame_bytes, "generated", index), dtype=np.uint8
            ).reshape(args.target_height, args.target_width, 3)
            person = remap_boolean_mask(
                np, person_masks[index], source_frame=mask_frame, target_frame=target_frame
            )
            tracked_flower = remap_boolean_mask(
                np, flower_masks[index], source_frame=mask_frame, target_frame=target_frame
            )
            source_hand = remap_boolean_mask(
                np, hand_masks[index], source_frame=mask_frame, target_frame=target_frame
            )
            if not np.any(tracked_flower):
                empty_flower_track_frames.append(index)
            support = binary_dilate_square(np, person, args.person_dilation)
            strict = strict_flower_seed(np, source_rgb)
            # A flower-union tracker can occasionally absorb face or forearm
            # pixels.  Constrain the color negative to the measured source
            # person/hand support so warm scene objects are not rejected while
            # source-human pixels can never be projected onto the robot layer.
            source_human_occluder = person | source_hand
            skin = source_skin_like(np, source_rgb) & source_human_occluder
            skin = binary_dilate_square(np, skin & support, args.skin_negative_dilation)
            # Segmentation conflicts are resolved in favour of the generated
            # subject in the eroded source-person core.  This is the explicit
            # z-order rule that prevents a contaminated flower-union track
            # from projecting a source face or shirt onto the robot.  The
            # erosion keeps legitimate flower/person boundary pixels eligible.
            flower = resolve_flower_visibility(
                np,
                candidates=tracked_flower | strict,
                edit_support=support,
                source_person=person,
                source_skin_negative=skin,
                person_core_erosion=args.person_core_negative_erosion,
            )
            person_core = binary_erode_square(
                np, person, args.person_core_negative_erosion
            )
            validate_visibility_partition(
                np,
                edit_support=support,
                flower_restore=flower,
                source_person_core=person_core,
                source_skin_negative=skin,
            )
            support = np.asarray(support, dtype=np.bool_).copy()
            flower = np.asarray(flower, dtype=np.bool_).copy()
            support.setflags(write=False)
            flower.setflags(write=False)
            output = compose_object_factored_frame(
                np,
                source_rgb=source_rgb,
                generated_rgb=generated_rgb,
                edit_support=support,
                flower_restore=flower,
            )
            review_writer.stdin.write(output.tobytes())
            lossless_writer.stdin.write(output.tobytes())

            known = ~support | flower
            known_exact = int(np.count_nonzero(np.all(output == source_rgb, axis=2) & known))
            known_count = int(np.count_nonzero(known))
            known_exact_pixels += known_exact
            total_pixels += known_count
            flower_restore_pixels += int(np.count_nonzero(flower))
            flower_track_pixels += int(np.count_nonzero(tracked_flower & support))
            strict_seed_pixels += int(np.count_nonzero(strict & support))
            skin_negative_pixels += int(np.count_nonzero(skin & support))
            support_pixels += int(np.count_nonzero(support))
            if np.any(flower):
                input_diff = np.abs(
                    generated_rgb[flower].astype(np.int16) - source_rgb[flower].astype(np.int16)
                )
                output_diff = np.abs(
                    output[flower].astype(np.int16) - source_rgb[flower].astype(np.int16)
                )
                input_flower_abs_error += int(input_diff.sum())
                input_flower_value_count += int(input_diff.size)
                output_flower_abs_error += int(output_diff.sum())
                output_flower_value_count += int(output_diff.size)
            if previous_source is not None:
                temporal_mask = flower & previous_flower
                if np.any(temporal_mask):
                    source_delta = source_rgb.astype(np.int16) - previous_source.astype(np.int16)
                    generated_delta = generated_rgb.astype(np.int16) - previous_generated.astype(np.int16)
                    output_delta = output.astype(np.int16) - previous_output.astype(np.int16)
                    input_residual = np.abs(generated_delta[temporal_mask] - source_delta[temporal_mask])
                    output_residual = np.abs(output_delta[temporal_mask] - source_delta[temporal_mask])
                    input_flower_temporal_residual_sum += float(input_residual.sum())
                    output_flower_temporal_residual_sum += float(output_residual.sum())
                    temporal_value_count += int(input_residual.size)
            per_frame.append(
                {
                    "frame": index,
                    "edit_support_fraction": float(np.mean(support)),
                    "flower_restore_fraction": float(np.mean(flower)),
                    "known_exact_fraction": float(known_exact / max(1, known_count)),
                    "source_flower_track_empty": bool(not np.any(tracked_flower)),
                }
            )
            previous_source = source_rgb.copy()
            previous_generated = generated_rgb.copy()
            previous_output = output.copy()
            previous_flower = flower.copy()
    finally:
        if review_writer.stdin is not None:
            review_writer.stdin.close()
        if lossless_writer.stdin is not None:
            lossless_writer.stdin.close()

    processes = {
        "source_decoder": source_decoder,
        "generated_decoder": generated_decoder,
        "review_writer": review_writer,
        "lossless_writer": lossless_writer,
    }
    process_logs: dict[str, str] = {}
    for name, process in processes.items():
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        returncode = process.wait()
        log_path = log_dir / f"{name}.log"
        log_path.write_text(stderr)
        process_logs[name] = str(log_path)
        if returncode:
            raise RuntimeError(f"{name} failed with return code {returncode}: {stderr[-1000:]}")

    wall_seconds = time.perf_counter() - tic
    review_info = _probe(ffprobe, review_path)
    lossless_info = _probe(ffprobe, lossless_path)
    if review_info["frames"] != args.expected_frames or lossless_info["frames"] != args.expected_frames:
        raise RuntimeError("encoded outputs do not contain the full source timeline")
    lossless_verification = _verify_encoded_projection(
        np,
        ffmpeg=ffmpeg,
        encoded_video=lossless_path,
        source_decode_command=source_decode_command,
        expected_frames=args.expected_frames,
        width=args.target_width,
        height=args.target_height,
        mask_frame=mask_frame,
        target_frame=target_frame,
        person_masks=person_masks,
        flower_masks=flower_masks,
        hand_masks=hand_masks,
        person_dilation=args.person_dilation,
        skin_negative_dilation=args.skin_negative_dilation,
        person_core_negative_erosion=args.person_core_negative_erosion,
    )
    review_verification = _verify_encoded_projection(
        np,
        ffmpeg=ffmpeg,
        encoded_video=review_path,
        source_decode_command=source_decode_command,
        expected_frames=args.expected_frames,
        width=args.target_width,
        height=args.target_height,
        mask_frame=mask_frame,
        target_frame=target_frame,
        person_masks=person_masks,
        flower_masks=flower_masks,
        hand_masks=hand_masks,
        person_dilation=args.person_dilation,
        skin_negative_dilation=args.skin_negative_dilation,
        person_core_negative_erosion=args.person_core_negative_erosion,
    )
    verified_wall_seconds = time.perf_counter() - tic
    metrics = {
        "frames": args.expected_frames,
        "video_seconds": args.expected_frames / args.fps,
        "wall_seconds": wall_seconds,
        "projection_fps": args.expected_frames / wall_seconds,
        "projection_realtime_factor": wall_seconds / (args.expected_frames / args.fps),
        "verified_end_to_end_wall_seconds": verified_wall_seconds,
        "verified_end_to_end_fps": args.expected_frames / verified_wall_seconds,
        "known_source_exact_fraction_preencode": known_exact_pixels / max(1, total_pixels),
        "mean_edit_support_fraction": support_pixels / (args.expected_frames * args.target_width * args.target_height),
        "flower_restore_pixels": flower_restore_pixels,
        "flower_track_pixels_inside_support": flower_track_pixels,
        "strict_seed_pixels_inside_support": strict_seed_pixels,
        "skin_negative_pixels_inside_support": skin_negative_pixels,
        "input_flower_mad": input_flower_abs_error / max(1, input_flower_value_count),
        "output_flower_mad_preencode": output_flower_abs_error / max(1, output_flower_value_count),
        "input_flower_temporal_residual_mad": input_flower_temporal_residual_sum / max(1, temporal_value_count),
        "output_flower_temporal_residual_mad_preencode": output_flower_temporal_residual_sum / max(1, temporal_value_count),
        "empty_source_flower_track_frames": empty_flower_track_frames,
        "lossless_encoded_verification": lossless_verification,
        "review_encoded_verification": review_verification,
    }
    acceptance = {
        "full_660_frames": review_info["frames"] == args.expected_frames,
        "duration_at_least_20_seconds": review_info["duration"] >= 20.0,
        "known_source_exact_preencode": metrics["known_source_exact_fraction_preencode"] == 1.0,
        "flower_exact_preencode": metrics["output_flower_mad_preencode"] == 0.0,
        "flower_temporal_residual_exact_preencode": (
            metrics["output_flower_temporal_residual_mad_preencode"] == 0.0
        ),
        "lossless_known_source_exact": (
            lossless_verification["known_source_exact_fraction"] == 1.0
        ),
        "lossless_flower_exact": lossless_verification["flower_mad"] == 0.0,
        "lossless_flower_temporal_residual_exact": (
            lossless_verification["flower_temporal_residual_mad"] == 0.0
        ),
        "flower_person_core_overlap_zero": (
            lossless_verification["flower_person_core_overlap_pixels"] == 0
        ),
        "flower_skin_negative_overlap_zero": (
            lossless_verification["flower_skin_negative_overlap_pixels"] == 0
        ),
        "human_review": "pending",
    }
    automatic_pass = all(value is True for key, value in acceptance.items() if key != "human_review")
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "honest_status": (
            "PARTIAL: all automatic full-length source/object preservation gates pass; "
            "dense human review of robot geometry, contact, and source-human leakage is pending."
            if automatic_pass
            else "PARTIAL: one or more automatic full-length gates failed."
        ),
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "git": _git_state(),
        "packages": {"python": sys.version, "numpy": np.__version__},
        "gpu": {"used": False, "reason": "deterministic CPU layer projection"},
        "coordinate_frames": {
            "mask": mask_frame.to_dict(),
            "target": target_frame.to_dict(),
            "timeline": "absolute_frame_index:full_source_660",
            "occlusion": "source-visible flower mask minus dilated source-skin negative",
        },
        "inputs": {
            "source_video": {"path": str(source), "sha256": _sha256(source), "info": source_info},
            "generated_video": {"path": str(generated), "sha256": _sha256(generated), "info": generated_info},
            "source_person_masks": {"path": str(person_path), "sha256": _sha256(person_path), **person_info},
            "source_flower_masks": {"path": str(flower_path), "sha256": _sha256(flower_path), **flower_info},
            "source_hand_masks": {"path": str(hand_path), "sha256": _sha256(hand_path), **hand_info},
        },
        "method": {
            "base_layer": "source video aligned to target camera pixels",
            "generated_layer": "generated pixels only on dilated per-frame source-person support",
            "flower_layer": "tracked source flowers plus strict color core, excluding skin inside tracked source-person/hand occluders",
            "person_dilation_pixels_target_frame": args.person_dilation,
            "skin_negative_dilation_pixels_target_frame": args.skin_negative_dilation,
            "person_core_negative_erosion_pixels_target_frame": args.person_core_negative_erosion,
            "source_alignment_filter": source_filter,
        },
        "commands": {
            "argv": sys.argv,
            "source_decode": source_decode_command,
            "generated_decode": generated_decode_command,
            "review_encode": review_write_command,
            "lossless_encode": lossless_write_command,
        },
        "logs": process_logs,
        "metrics": metrics,
        "acceptance": acceptance,
        "outputs": {
            "review_video": {"path": str(review_path), "sha256": _sha256(review_path), "info": review_info},
            "lossless_video": {"path": str(lossless_path), "sha256": _sha256(lossless_path), "info": lossless_info},
        },
        "limitations": [
            "The projection proves source-visible flower/background preservation, not robot geometry or physical contact correctness.",
            "The source flower union track is object-union supervision, not immutable per-stem identity through every occlusion.",
            "Human semantic review remains required before WORKING status.",
        ],
    }
    (output_dir / "frame-metrics.json").write_text(json.dumps(per_frame, indent=2))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics, "acceptance": acceptance}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
