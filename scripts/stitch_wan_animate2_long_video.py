#!/usr/bin/env python3
"""Stitch overlapping Wan-Animate-2 windows around a proven quality anchor."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
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
    info["decoded_frames"] = len(frames)
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    return frames, info


def _overlap(
    first_start: int,
    first_count: int,
    second_start: int,
    second_count: int,
) -> tuple[int, int]:
    start = max(first_start, second_start)
    end = min(first_start + first_count, second_start + second_count)
    if end <= start:
        raise ValueError("frame intervals do not overlap")
    return start, end


def estimate_background_offset(
    np: Any,
    *,
    reference: list[Any],
    reference_start: int,
    candidate: list[Any],
    candidate_start: int,
    maximum_offset: float = 12.0,
) -> tuple[float, float, float]:
    """Estimate a bounded RGB offset from the stable left-side background."""

    start, end = _overlap(
        reference_start,
        len(reference),
        candidate_start,
        len(candidate),
    )
    per_frame = []
    for global_index in range(start, end, 2):
        first = reference[global_index - reference_start]
        second = candidate[global_index - candidate_start]
        width = first.shape[1]
        stable_width = max(1, round(width * 0.43))
        difference = (
            first[::4, :stable_width:4].astype(np.float32)
            - second[::4, :stable_width:4].astype(np.float32)
        )
        per_frame.append(np.median(difference, axis=(0, 1)))
    offset = np.median(np.asarray(per_frame), axis=0)
    offset = np.clip(offset, -maximum_offset, maximum_offset)
    return tuple(float(value) for value in offset)


def apply_color_offset(np: Any, frames: list[Any], offset: tuple[float, ...]) -> list[Any]:
    correction = np.asarray(offset, dtype=np.float32)
    return [
        np.clip(frame.astype(np.float32) + correction, 0, 255).astype(np.uint8)
        for frame in frames
    ]


def _transition_cost(np: Any, first: Any, second: Any) -> float:
    delta = np.abs(first.astype(np.float32) - second.astype(np.float32))
    width = first.shape[1]
    subject = delta[:, round(width * 0.44) : round(width * 0.94)]
    return float(np.mean(delta) + 1.5 * np.mean(subject))


def select_seam(
    np: Any,
    *,
    current: list[Any],
    current_start: int,
    following: list[Any],
    following_start: int,
    minimum_seam: int | None = None,
    maximum_seam: int | None = None,
) -> tuple[int, float]:
    """Choose the least-visible hard seam in the common timeline interval."""

    overlap_start, overlap_end = _overlap(
        current_start,
        len(current),
        following_start,
        len(following),
    )
    candidate_start = max(overlap_start + 1, minimum_seam or overlap_start + 1)
    candidate_end = min(
        overlap_end,
        maximum_seam + 1 if maximum_seam is not None else overlap_end,
    )
    candidates = range(candidate_start, candidate_end)
    scored = [
        (
            _transition_cost(
                np,
                current[seam - 1 - current_start],
                following[seam - following_start],
            ),
            seam,
        )
        for seam in candidates
    ]
    if not scored:
        raise ValueError("seam constraints leave no candidate in the overlap")
    cost, seam = min(scored)
    return seam, cost


def merge_at_best_seam(
    np: Any,
    *,
    current: list[Any],
    current_start: int,
    following: list[Any],
    following_start: int,
    blend_radius: int = 0,
    minimum_seam: int | None = None,
    maximum_seam: int | None = None,
) -> tuple[list[Any], dict[str, float | int]]:
    seam, cost = select_seam(
        np,
        current=current,
        current_start=current_start,
        following=following,
        following_start=following_start,
        minimum_seam=minimum_seam,
        maximum_seam=maximum_seam,
    )
    if blend_radius < 0:
        raise ValueError("blend_radius must be non-negative")
    overlap_start, overlap_end = _overlap(
        current_start,
        len(current),
        following_start,
        len(following),
    )
    blend_start = max(overlap_start, seam - blend_radius)
    blend_end = min(overlap_end, seam + blend_radius)
    blended = []
    blend_count = blend_end - blend_start
    for offset, global_index in enumerate(range(blend_start, blend_end)):
        progress = (offset + 1) / (blend_count + 1)
        alpha = 0.5 - 0.5 * math.cos(math.pi * progress)
        first = current[global_index - current_start].astype(np.float32)
        second = following[global_index - following_start].astype(np.float32)
        blended.append(
            np.clip(np.rint(first * (1.0 - alpha) + second * alpha), 0, 255).astype(
                np.uint8
            )
        )
    merged = (
        current[: blend_start - current_start]
        + blended
        + following[blend_end - following_start :]
    )
    return merged, {
        "current_start": current_start,
        "following_start": following_start,
        "seam_frame": seam,
        "seam_cost": cost,
        "blend_radius": blend_radius,
        "blend_start_frame": blend_start,
        "blend_end_frame_exclusive": blend_end,
    }


def merge_quality_anchor(
    np: Any,
    *,
    left: list[Any],
    left_start: int,
    anchor: list[Any],
    anchor_start: int,
    right: list[Any],
    right_start: int,
    minimum_anchor_frames: int,
    blend_radius: int = 0,
) -> tuple[list[Any], dict[str, Any]]:
    """Insert the stable core of an anchor using searched entry/exit seams."""

    if minimum_anchor_frames < 1:
        raise ValueError("minimum_anchor_frames must be positive")
    anchored, entry = merge_at_best_seam(
        np,
        current=left,
        current_start=left_start,
        following=anchor,
        following_start=anchor_start,
        blend_radius=blend_radius,
    )
    entry_frame = int(entry["seam_frame"])
    constrained_right_start = max(
        right_start,
        entry_frame + minimum_anchor_frames + 2 * blend_radius - 1,
    )
    anchor_end = anchor_start + len(anchor)
    if constrained_right_start >= anchor_end - 1:
        raise ValueError("anchor overlap cannot retain the requested minimum frames")
    if not right_start <= constrained_right_start < right_start + len(right):
        raise ValueError("right timeline does not cover the constrained anchor exit")
    right_offset = constrained_right_start - right_start
    merged, exit_seam = merge_at_best_seam(
        np,
        current=anchored,
        current_start=left_start,
        following=right[right_offset:],
        following_start=constrained_right_start,
        blend_radius=blend_radius,
    )
    exit_frame = int(exit_seam["seam_frame"])
    retained_start = entry_frame + blend_radius
    retained_end = exit_frame - blend_radius
    retained_frames = retained_end - retained_start
    if retained_frames < minimum_anchor_frames:
        raise RuntimeError("quality anchor retention constraint was violated")
    return merged, {
        "entry": {**entry, "kind": "quality_anchor_entry"},
        "exit": {**exit_seam, "kind": "quality_anchor_exit"},
        "retained_start_frame": retained_start,
        "retained_end_frame_exclusive": retained_end,
        "retained_frames": retained_frames,
        "minimum_anchor_frames": minimum_anchor_frames,
    }


def _writer(ffmpeg: Path, output: Path, width: int, height: int, fps: float) -> Any:
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            str(ffmpeg),
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


def _cosine(np: Any, first: Any, second: Any) -> float:
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-9:
        return 1.0 if float(np.linalg.norm(a - b)) < 1e-9 else 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / denominator)))


def _metrics(cv2: Any, np: Any, candidate: list[Any], source: list[Any]) -> dict[str, Any]:
    height, width = candidate[0].shape[:2]
    analysis_width = 256
    analysis_height = max(2, round(height * analysis_width / width))
    candidate_gray = [
        cv2.cvtColor(
            cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        for frame in candidate
    ]
    source_gray = [
        cv2.cvtColor(
            cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        for frame in source
    ]
    transition = []
    motion = []
    temporal = []
    for index in range(1, len(candidate_gray)):
        candidate_delta = cv2.absdiff(candidate_gray[index], candidate_gray[index - 1])
        source_delta = cv2.absdiff(source_gray[index], source_gray[index - 1])
        transition.append(float(np.mean(candidate_delta)))
        cosine = _cosine(np, candidate_delta, source_delta)
        candidate_energy = float(np.mean(candidate_delta))
        source_energy = float(np.mean(source_delta))
        energy_ratio = min(
            (candidate_energy + 1e-3) / (source_energy + 1e-3),
            (source_energy + 1e-3) / (candidate_energy + 1e-3),
        )
        motion.append(math.sqrt(max(0.0, cosine * energy_ratio)))
        residual = float(
            np.mean(
                np.abs(candidate_delta.astype(np.float32) - source_delta.astype(np.float32))
            )
        )
        temporal.append(math.exp(-residual / 32.0))
    median_transition = float(np.median(transition))
    return {
        "decoded_frames": len(candidate),
        "motion_preservation": float(np.mean(motion)),
        "temporal_consistency": float(np.mean(temporal)),
        "median_transition_energy": median_transition,
        "maximum_transition_energy": max(transition),
        "maximum_transition_ratio": max(transition) / max(median_transition, 1e-6),
        "transition_energy": transition,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--human-review", choices=("pending", "passed", "failed"), default="pending")
    parser.add_argument("--minimum-anchor-frames", type=int, default=32)
    parser.add_argument("--seam-blend-radius", type=int, default=0)
    parser.add_argument("--additional-experiment-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--allow-continuation-reference-bridges",
        action="store_true",
        help="allow additional windows conditioned on a declared source-camera frame",
    )
    parser.add_argument("--window-stable-head", type=int, default=0)
    parser.add_argument("--window-stable-tail", type=int, default=0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"stitch output already exists: {manifest_path}")
    metadata_path = experiment / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"generation metadata is missing: {metadata_path}")
    generation = json.loads(metadata_path.read_text())
    if generation.get("status") != "completed":
        raise RuntimeError(f"generation is not complete: {generation.get('status')}")
    generation_sources = [(generation, experiment, metadata_path)]
    for additional_arg in args.additional_experiment_dir:
        additional_experiment = additional_arg.expanduser().resolve()
        additional_metadata = additional_experiment / "metadata.json"
        if not additional_metadata.is_file():
            raise ValueError(
                f"additional generation metadata is missing: {additional_metadata}"
            )
        additional = json.loads(additional_metadata.read_text())
        if additional.get("status") != "completed":
            raise RuntimeError(
                f"additional generation is not complete: {additional.get('status')}"
            )
        reference_mismatch = (
            additional["reference"]["sha256"]
            != generation["reference"]["sha256"]
        )
        if reference_mismatch:
            if not args.allow_continuation_reference_bridges:
                raise ValueError("additional generation mismatch: reference.sha256")
            if additional["reference"].get("coordinate_frame") != "source_camera_frame":
                raise ValueError(
                    "continuation bridge reference must declare source_camera_frame"
                )
            reference_frame = additional["reference"].get("source_camera_frame")
            if not isinstance(reference_frame, int):
                raise ValueError(
                    "continuation bridge reference requires an integer camera frame"
                )
            if not any(
                int(item["start_frame"])
                <= reference_frame
                < int(item["start_frame"]) + int(item["expected_output_frames"])
                for item in additional["windows"]
            ):
                raise ValueError(
                    "continuation bridge reference frame is outside its generated windows"
                )
        for key_path in (
            ("source", "sha256"),
            ("quality_anchor", "sha256"),
            ("prompt_sha256",),
            ("source_commit",),
            ("model_revision",),
            ("checkpoint_hashes",),
            ("config", "clip_len"),
            ("config", "steps"),
            ("config", "guidance_scale"),
            ("config", "seed"),
        ):
            primary_value: Any = generation
            additional_value: Any = additional
            for key in key_path:
                primary_value = primary_value[key]
                additional_value = additional_value[key]
            if primary_value != additional_value:
                joined = ".".join(key_path)
                raise ValueError(f"additional generation mismatch: {joined}")
        generation_sources.append(
            (additional, additional_experiment, additional_metadata)
        )
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if not ffmpeg.is_file():
        raise ValueError(f"FFmpeg does not exist: {ffmpeg}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    source_path = Path(generation["source"]["path"])
    anchor_path = Path(generation["quality_anchor"]["path"])
    # Remote absolute paths are not valid after the experiment is copied home.
    if not source_path.is_file():
        source_path = experiment / "input" / "source-full-660.mp4"
    if not anchor_path.is_file():
        anchor_path = experiment / "input" / "quality-anchor-0236-0315.mp4"
    source, source_info = _decode(cv2, source_path)
    anchor, anchor_info = _decode(cv2, anchor_path)
    anchor_start = int(generation["quality_anchor"]["start_frame"])
    anchor_end = anchor_start + len(anchor)
    if len(source) != int(generation["source"]["info"]["frames"]):
        raise RuntimeError("local source does not match generation metadata")

    windows: dict[int, list[Any]] = {}
    window_inputs = []
    expected_shape = anchor[0].shape
    for item_generation, item_experiment, item_metadata in generation_sources:
        for item in item_generation["windows"]:
            start = int(item["start_frame"])
            if start in windows:
                raise RuntimeError(f"duplicate generated window start: {start}")
            result_path = Path(item["result"])
            if not result_path.is_file():
                result_path = (
                    item_experiment
                    / "windows"
                    / f"window-{int(item['index']):02d}-{start:04d}"
                    / "result.mp4"
                )
            frames, info = _decode(cv2, result_path)
            if len(frames) != int(item["expected_output_frames"]):
                raise RuntimeError(f"window {item['index']} decoded {len(frames)} frames")
            if frames[0].shape != expected_shape:
                raise RuntimeError(
                    f"window {item['index']} shape does not match quality anchor"
                )
            windows[start] = frames
            window_inputs.append(
                {
                    "index": int(item["index"]),
                    "start_frame": start,
                    "path": str(result_path),
                    "sha256": _sha256(result_path),
                    "info": info,
                    "generation_metadata": str(item_metadata),
                }
            )

    starts = sorted(windows)
    left_starts = [start for start in starts if start < anchor_start]
    right_starts = [start for start in starts if start < anchor_end < start + len(windows[start])]
    right_starts += [start for start in starts if start >= anchor_end]
    right_starts = sorted(set(right_starts))
    if not left_starts or not right_starts:
        raise RuntimeError("window plan does not bracket the quality anchor")

    offsets = []
    reference_frames, reference_start = anchor, anchor_start
    for start in reversed(left_starts):
        offset = estimate_background_offset(
            np,
            reference=reference_frames,
            reference_start=reference_start,
            candidate=windows[start],
            candidate_start=start,
        )
        windows[start] = apply_color_offset(np, windows[start], offset)
        offsets.append({"start_frame": start, "direction": "left", "bgr": offset})
        reference_frames, reference_start = windows[start], start

    reference_frames, reference_start = anchor, anchor_start
    for start in right_starts:
        offset = estimate_background_offset(
            np,
            reference=reference_frames,
            reference_start=reference_start,
            candidate=windows[start],
            candidate_start=start,
        )
        windows[start] = apply_color_offset(np, windows[start], offset)
        offsets.append({"start_frame": start, "direction": "right", "bgr": offset})
        reference_frames, reference_start = windows[start], start

    seams = []
    left = windows[left_starts[0]]
    left_start = left_starts[0]
    previous_start = left_start
    for start in left_starts[1:]:
        minimum_seam = start + args.window_stable_head + args.seam_blend_radius
        maximum_seam = (
            previous_start
            + len(windows[previous_start])
            - args.window_stable_tail
            - args.seam_blend_radius
        )
        left, seam = merge_at_best_seam(
            np,
            current=left,
            current_start=left_start,
            following=windows[start],
            following_start=start,
            blend_radius=args.seam_blend_radius,
            minimum_seam=minimum_seam,
            maximum_seam=maximum_seam,
        )
        seams.append(seam)
        previous_start = start
    first_right = right_starts[0]
    right = windows[first_right]
    right_start = first_right
    previous_start = first_right
    for start in right_starts[1:]:
        minimum_seam = start + args.window_stable_head + args.seam_blend_radius
        maximum_seam = (
            previous_start
            + len(windows[previous_start])
            - args.window_stable_tail
            - args.seam_blend_radius
        )
        right, seam = merge_at_best_seam(
            np,
            current=right,
            current_start=right_start,
            following=windows[start],
            following_start=start,
            blend_radius=args.seam_blend_radius,
            minimum_seam=minimum_seam,
            maximum_seam=maximum_seam,
        )
        seams.append(seam)
        previous_start = start
    final_frames, anchor_seams = merge_quality_anchor(
        np,
        left=left,
        left_start=left_start,
        anchor=anchor,
        anchor_start=anchor_start,
        right=right,
        right_start=right_start,
        minimum_anchor_frames=args.minimum_anchor_frames,
        blend_radius=args.seam_blend_radius,
    )
    seams.extend((anchor_seams["entry"], anchor_seams["exit"]))
    if len(final_frames) != len(source):
        raise RuntimeError(
            f"stitched {len(final_frames)} frames; expected source length {len(source)}"
        )
    retained_start = int(anchor_seams["retained_start_frame"])
    retained_end = int(anchor_seams["retained_end_frame_exclusive"])
    if any(
        not np.array_equal(final_frames[index], anchor[index - anchor_start])
        for index in range(retained_start, retained_end)
    ):
        raise RuntimeError("retained quality anchor core changed before encoding")

    resized_source = [
        cv2.resize(
            frame,
            (expected_shape[1], expected_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        for frame in source
    ]
    metrics = _metrics(cv2, np, final_frames, resized_source)
    boundary_indices = [retained_start, retained_end]
    metrics["anchor_boundary_transition_ratios"] = {
        str(index): metrics["transition_energy"][index - 1]
        / max(float(metrics["median_transition_energy"]), 1e-6)
        for index in boundary_indices
    }
    metrics["quality_anchor_core_exact_preencode"] = True
    metrics["quality_anchor_frames"] = int(anchor_seams["retained_frames"])
    transition_energy = metrics.pop("transition_energy")

    height, width = expected_shape[:2]
    encoded_height = height if height % 2 == 0 else height + 1
    output = output_dir / "wan-animate2-full-27s-anchor-aligned.mp4"
    writer = _writer(
        ffmpeg,
        output,
        width,
        encoded_height,
        float(source_info["fps"]),
    )
    try:
        assert writer.stdin is not None
        for frame in final_frames:
            if encoded_height != height:
                frame = cv2.copyMakeBorder(
                    frame,
                    0,
                    encoded_height - height,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            writer.stdin.write(frame.tobytes())
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
        if writer.wait():
            raise RuntimeError("FFmpeg failed to encode the stitched video")
    subprocess.run(
        [str(ffmpeg), "-v", "error", "-i", str(output), "-f", "null", "-"],
        check=True,
    )
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-i",
            str(output),
            "-vf",
            "fps=28/27.5,scale=480:-2,tile=4x7:padding=4:margin=4:color=black",
            "-frames:v",
            "1",
            str(output_dir / "dense-review.jpg"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-i",
            str(output),
            "-vf",
            "fps=16/27.5,scale=480:-2,tile=4x4:padding=4:margin=4:color=black",
            "-frames:v",
            "1",
            str(output_dir / "storyboard-16.jpg"),
        ],
        check=True,
    )
    packages = {}
    for package in ("numpy", "opencv-python", "opencv-python-headless"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    review_passed = args.human_review == "passed"
    manifest = {
        "schema_version": "1.0.0",
        "method": "wan_animate2_distilled_overlap_seam_search_quality_anchor_v2",
        "status": "accepted" if review_passed else "rejected" if args.human_review == "failed" else "review_required",
        "honest_status": "WORKING" if review_passed else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "packages": packages,
        "generation_metadata": str(metadata_path),
        "generation_metadata_sha256": _sha256(metadata_path),
        "additional_generation_metadata": [
            {"path": str(path), "sha256": _sha256(path)}
            for _, _, path in generation_sources[1:]
        ],
        "allow_continuation_reference_bridges": args.allow_continuation_reference_bridges,
        "source": {"path": str(source_path), "sha256": _sha256(source_path), "info": source_info},
        "quality_anchor": {
            "path": str(anchor_path),
            "sha256": _sha256(anchor_path),
            "info": anchor_info,
            "start_frame": anchor_start,
            "end_frame_exclusive": anchor_end,
            "retained_start_frame": retained_start,
            "retained_end_frame_exclusive": retained_end,
            "retained_frames": int(anchor_seams["retained_frames"]),
        },
        "windows": window_inputs,
        "color_offsets": offsets,
        "seams": seams,
        "metrics": metrics,
        "acceptance": {
            "full_clip_decoded": metrics["decoded_frames"] == len(source),
            "quality_anchor_core_exact_preencode": metrics["quality_anchor_core_exact_preencode"],
            "human_review": args.human_review,
            "seam_blend_radius": args.seam_blend_radius,
            "window_stable_head": args.window_stable_head,
            "window_stable_tail": args.window_stable_tail,
        },
        "outputs": {
            "video": str(output),
            "video_sha256": _sha256(output),
            "dense_review": str(output_dir / "dense-review.jpg"),
            "storyboard": str(output_dir / "storyboard-16.jpg"),
        },
        "limitations": [
            "Each 80-frame generated window is independent; bounded background color alignment and overlap seam handling do not create shared diffusion memory.",
            "A searched entry/exit seam retains an unchanged central core of the proven 80-frame quality anchor before final H.264 encoding.",
            "Motion and temporal scores are deterministic image-space proxies, not contact or robot-execution metrics.",
            "Human review is mandatory because automated metrics do not reliably detect identity drift, residual humans, or flower-contact hallucination.",
        ],
        "transition_energy_sha256": hashlib.sha256(
            json.dumps(transition_energy, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": manifest["status"],
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if review_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
