#!/usr/bin/env python3
"""Project a generated robot layer onto a coherent masked chroma state."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
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

from phiagent.rendering.chroma_state import (
    project_masked_multiscale_chroma_state,
    restore_masked_luma_carrier,
    spatial_chroma_tv,
)
from phiagent.rendering.object_factored_long_video import (
    SourceResizeCrop,
    binary_dilate_square,
)
from phiagent.rendering.temporal_occlusion import (
    evidence_ordered_flower_front,
    projected_contact_corridor,
    projected_contact_evidence_lock,
    propagate_robot_material_residual,
    reinforce_projected_contact_evidence,
    right_arm_flower_partition,
    source_owned_flower_restore_mask,
)
from scripts.audit_robot_layer_long_video import (
    _decoder_command,
    _read_frame,
    _resolve_frame_masks,
)
from scripts.stabilize_joyai_appearance_state import (
    _finish,
    _git_state,
    _packages,
    _probe,
    _sha256,
    _writer,
)
from scripts.stabilize_right_arm_flower_occlusion import (
    _load_key,
    _native_mask,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--person-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument(
        "--visibility-flower-masks",
        type=Path,
        help=(
            "Optional conservative flower observer used only for visible z-order; "
            "the primary flower mask remains the persistent object/contact state."
        ),
    )
    parser.add_argument(
        "--flower-mask-contract",
        choices=(
            "tracked_front_layer_with_human_negatives",
            "foundation_refined_front_layer",
        ),
        default="tracked_front_layer_with_human_negatives",
    )
    parser.add_argument("--pose-limb-masks", type=Path, required=True)
    parser.add_argument("--robot-limb-masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--mask-projection",
        choices=("source_native", "legacy_832x480_to_native_1280x720"),
        default="legacy_832x480_to_native_1280x720",
    )
    parser.add_argument("--kernel-sizes", default="7,15")
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--maximum-chroma-delta", type=float, default=28.0)
    parser.add_argument("--saturation-scale", type=float, default=1.0)
    parser.add_argument("--replacement-threshold", type=float, default=12.0)
    parser.add_argument("--person-dilation", type=int, default=10)
    parser.add_argument("--skin-negative-dilation", type=int, default=2)
    parser.add_argument("--person-core-negative-erosion", type=int, default=2)
    parser.add_argument("--corridor-dilation-pixels", type=int, default=31)
    parser.add_argument("--hand-dilation-pixels", type=int, default=13)
    parser.add_argument("--flower-clean-plate-padding-pixels", type=int, default=9)
    parser.add_argument("--flower-sample-footprint-pixels", type=int, default=7)
    parser.add_argument("--maximum-source-occlusion-gap", type=int, default=24)
    parser.add_argument("--contact-radius", type=int, default=3)
    parser.add_argument("--contact-codec-error-margin", type=float, default=8.0)
    parser.add_argument("--minimum-chroma-tv-reduction", type=float, default=0.25)
    parser.add_argument("--maximum-luma-mae", type=float, default=1.0)
    parser.add_argument(
        "--artifact-policy",
        choices=("review_only", "lossless_and_review"),
        default="lossless_and_review",
    )
    parser.add_argument("--review-preset", default="medium")
    parser.add_argument("--review-crf", type=int, default=8)
    parser.add_argument("--review-chroma-qp-offset", type=int, default=-12)
    parser.add_argument(
        "--review-pixel-format", choices=("yuv420p", "yuv444p"), default="yuv420p"
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260814)
    return parser


def _parse_kernels(value: str) -> tuple[int, ...]:
    kernels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not kernels or any(item < 3 or item % 2 == 0 for item in kernels):
        raise ValueError("kernel sizes must be a comma-separated list of odd values >= 3")
    return kernels


def _summary(np: Any, values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "maximum": 0.0}
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _fast_spatial_chroma_tv(cv2: Any, np: Any, frame_bgr: Any, mask: Any) -> float:
    saturation = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    return spatial_chroma_tv(np, saturation, mask)


def _review_writer(
    ffmpeg: Path,
    path: Path,
    *,
    width: int,
    height: int,
    fps: float,
    preset: str,
    crf: int,
    chroma_qp_offset: int,
    pixel_format: str,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", f"{fps:.8f}", "-i", "-", "-an", "-c:v", "libx264",
            "-preset", preset, "-crf", str(crf), "-pix_fmt", pixel_format,
            "-x264-params", f"chroma-qp-offset={chroma_qp_offset}",
            "-movflags", "+faststart", str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _process_record(
    cv2: Any,
    np: Any,
    *,
    index: int,
    candidate: Any,
    source: Any,
    person: Any,
    tracked_flower: Any,
    visibility_flower: Any,
    hands: Any,
    right_arm: Any,
    kernels: tuple[int, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Compute one frame independently so OpenCV can run frames in parallel."""

    support, tracked_front_layer = _resolve_frame_masks(
        np,
        source_rgb=cv2.cvtColor(source, cv2.COLOR_BGR2RGB),
        person=person,
        tracked_flower=tracked_flower,
        hands=hands,
        person_dilation=args.person_dilation,
        skin_negative_dilation=args.skin_negative_dilation,
        person_core_negative_erosion=args.person_core_negative_erosion,
        flower_mask_contract=args.flower_mask_contract,
    )
    _, resolved_flower = _resolve_frame_masks(
        np,
        source_rgb=cv2.cvtColor(source, cv2.COLOR_BGR2RGB),
        person=person,
        tracked_flower=visibility_flower,
        hands=hands,
        person_dilation=args.person_dilation,
        skin_negative_dilation=args.skin_negative_dilation,
        person_core_negative_erosion=args.person_core_negative_erosion,
        flower_mask_contract="resolved_visibility",
    )
    corridor_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.corridor_dilation_pixels, args.corridor_dilation_pixels),
    )
    right_arm_conflict = cv2.dilate(
        np.asarray(right_arm, dtype=np.uint8), corridor_kernel
    ) > 0
    flower_visible, incumbent_source_retained = evidence_ordered_flower_front(
        np,
        candidate=candidate,
        source=source,
        resolved_flower=resolved_flower,
        tracked_flower=tracked_front_layer,
        contested_support=right_arm_conflict,
        replacement_threshold=args.replacement_threshold,
    )
    _, flower_owner, protected_hand = right_arm_flower_partition(
        cv2,
        np,
        right_arm=right_arm,
        flower_visible=flower_visible,
        hand_support=hands,
        corridor_dilation_pixels=args.corridor_dilation_pixels,
        hand_dilation_pixels=args.hand_dilation_pixels,
    )
    flower_sampling_guard = source_owned_flower_restore_mask(
        cv2,
        np,
        flower_owner=flower_owner,
        person=person,
        hand_core=hands,
        protected_hand=protected_hand,
        clean_plate_padding_pixels=args.flower_clean_plate_padding_pixels,
        sample_footprint_pixels=args.flower_sample_footprint_pixels,
    )
    contact_lock = projected_contact_evidence_lock(
        cv2,
        np,
        candidate=candidate,
        source=source,
        hand_core=hands,
        tracked_object=tracked_flower,
        replacement_threshold=args.replacement_threshold,
        contact_radius=args.contact_radius,
        maximum_source_occlusion_gap=args.maximum_source_occlusion_gap,
    )
    contact_corridor = projected_contact_corridor(
        np,
        hand_core=hands,
        tracked_object=tracked_flower,
        contact_radius=args.contact_radius,
        maximum_source_occlusion_gap=args.maximum_source_occlusion_gap,
    )
    # The frozen contact radius also defines a compact material patch around
    # the sparse bridge, making it stable under block video encoding.
    contact_material_support = np.logical_and(
        hands,
        binary_dilate_square(np, contact_corridor, args.contact_radius),
    )
    support = np.logical_or(support, contact_material_support)
    flower_sampling_guard = np.logical_and(
        flower_sampling_guard, np.logical_not(contact_material_support)
    )
    global_flower_reconstruction = source_owned_flower_restore_mask(
        cv2,
        np,
        flower_owner=flower_visible,
        person=person,
        hand_core=hands,
        protected_hand=protected_hand,
        clean_plate_padding_pixels=args.flower_clean_plate_padding_pixels,
        sample_footprint_pixels=args.flower_sample_footprint_pixels,
    )
    global_flower_reconstruction = np.logical_and(
        global_flower_reconstruction,
        np.logical_or.reduce(
            (
                np.logical_not(right_arm_conflict),
                resolved_flower,
                incumbent_source_retained,
            )
        ),
    )
    global_flower_reconstruction = np.logical_and(
        global_flower_reconstruction,
        np.logical_not(contact_material_support),
    )
    # Every independently tracked flower sample owns its full codec
    # reconstruction footprint.  Restricting this guard to the right-arm
    # corridor lets downsampling mix neutral robot pixels back into petals.
    flower_restore = np.logical_or(
        global_flower_reconstruction, flower_sampling_guard
    )
    editable = np.logical_and.reduce(
        (
            support,
            np.logical_not(flower_restore),
            np.logical_not(contact_lock),
        )
    )
    audit_evaluation = np.logical_and(support, np.logical_not(flower_visible))
    repaired, projection = project_masked_multiscale_chroma_state(
        cv2,
        np,
        candidate,
        editable,
        kernel_sizes=kernels,
        strength=args.strength,
        maximum_chroma_delta=args.maximum_chroma_delta,
        saturation_scale=args.saturation_scale,
    )
    # The source-observed flower is an independently owned object layer, not
    # part of the robot material.  Re-compositing it from the source prevents
    # a second video encode from accumulating replacement error on the petals
    # and stems while the background outside declared alpha stays untouched.
    repaired[flower_restore] = source[flower_restore]
    reinforced_mask = np.logical_and.reduce(
        (contact_lock, support, np.logical_not(flower_restore))
    )
    repaired, contact_reinforcement = reinforce_projected_contact_evidence(
        np,
        projected=repaired,
        candidate=candidate,
        source=source,
        evidence_mask=reinforced_mask,
        replacement_threshold=args.replacement_threshold,
        codec_error_margin=args.contact_codec_error_margin,
    )
    completion_corridor = contact_material_support
    repaired, contact_completion = propagate_robot_material_residual(
        np,
        projected=repaired,
        candidate=candidate,
        source=source,
        corridor=completion_corridor,
        seed_mask=reinforced_mask,
        replacement_threshold=args.replacement_threshold,
    )
    repaired, completion_reinforcement = reinforce_projected_contact_evidence(
        np,
        projected=repaired,
        candidate=repaired,
        source=source,
        evidence_mask=completion_corridor,
        replacement_threshold=args.replacement_threshold,
        codec_error_margin=args.contact_codec_error_margin,
    )
    luma_preservation_support = np.logical_and.reduce(
        (
            audit_evaluation,
            np.logical_not(flower_restore),
            np.logical_not(contact_material_support),
        )
    )
    repaired, luma_restoration = restore_masked_luma_carrier(
        cv2,
        np,
        repaired,
        candidate,
        luma_preservation_support,
    )
    candidate_luma = cv2.cvtColor(candidate, cv2.COLOR_BGR2YCrCb)[..., 0].astype(
        np.float32
    )
    repaired_luma = cv2.cvtColor(repaired, cv2.COLOR_BGR2YCrCb)[..., 0].astype(
        np.float32
    )
    final_luma_delta = (
        float(
            np.abs(
                repaired_luma[luma_preservation_support]
                - candidate_luma[luma_preservation_support]
            ).mean()
        )
        if np.any(luma_preservation_support)
        else 0.0
    )
    final_contact_residual = np.mean(
        np.abs(repaired.astype(np.int16) - source.astype(np.int16)), axis=2
    )
    contact_completion["unresolved_source_like_pixels"] = float(
        np.count_nonzero(
            np.logical_and(
                completion_corridor,
                final_contact_residual < args.replacement_threshold,
            )
        )
    )
    chroma_tv_before = _fast_spatial_chroma_tv(
        cv2, np, candidate, audit_evaluation
    )
    chroma_tv_after = _fast_spatial_chroma_tv(cv2, np, repaired, audit_evaluation)
    background = np.logical_not(np.logical_or(support, flower_restore))
    return {
        "index": index,
        "repaired": repaired,
        "packed_editable": np.packbits(
            audit_evaluation.reshape(-1).astype(np.uint8)
        ),
        "background_exact": int(
            np.count_nonzero(np.all(repaired == candidate, axis=2)[background])
        ),
        "background_total": int(np.count_nonzero(background)),
        "flower_source_exact": int(
            np.count_nonzero(np.all(repaired == source, axis=2)[flower_restore])
        ),
        "flower_total": int(np.count_nonzero(flower_restore)),
        "luma_delta": final_luma_delta,
        "row": {
            "frame": index,
            "editable_pixels": int(np.count_nonzero(audit_evaluation)),
            "chroma_tv_before": chroma_tv_before,
            "chroma_tv_after_lossless": chroma_tv_after,
            "relative_chroma_tv_reduction_lossless": (
                1.0
                - chroma_tv_after / chroma_tv_before
                if chroma_tv_before > 0
                else 0.0
            ),
            "passes": projection["passes"],
            "contact_reinforcement": contact_reinforcement,
            "contact_completion": contact_completion,
            "contact_completion_reinforcement": completion_reinforcement,
            "luma_restoration": luma_restoration,
        },
    }


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    if args.workers < 1:
        raise ValueError("workers must be at least one")
    if not -12 <= args.review_chroma_qp_offset <= 12:
        raise ValueError("review chroma QP offset must be in [-12, 12]")
    if args.contact_radius < 0 or args.maximum_source_occlusion_gap < args.contact_radius:
        raise ValueError("invalid contact radius or maximum source occlusion gap")
    if args.contact_codec_error_margin < 0:
        raise ValueError("contact codec error margin must be non-negative")
    # The frame-level executor owns parallelism.  Disabling OpenCV's nested
    # pool avoids oversubscription and makes the recorded worker count honest.
    cv2.setNumThreads(1)
    kernels = _parse_kernels(args.kernel_sizes)
    paths = {
        "candidate": args.candidate_video.expanduser().resolve(),
        "source": args.source_video.expanduser().resolve(),
        "person_masks": args.person_masks.expanduser().resolve(),
        "flower_masks": args.flower_masks.expanduser().resolve(),
        "pose_limb_masks": args.pose_limb_masks.expanduser().resolve(),
        "robot_limb_masks": args.robot_limb_masks.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    if args.visibility_flower_masks is not None:
        paths["visibility_flower_masks"] = (
            args.visibility_flower_masks.expanduser().resolve()
        )
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{name}: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)

    required = {"frames": args.expected_frames, "width": args.width, "height": args.height}
    for name in ("candidate", "source"):
        probe = _probe(cv2, paths[name])
        if any(probe[key] != value for key, value in required.items()):
            raise ValueError(f"{name} timeline mismatch: {probe}")
        if abs(float(probe["fps"]) - args.fps) > 0.01:
            raise ValueError(f"{name} FPS mismatch: {probe}")

    person_payload = _load_key(np, paths["person_masks"], "packed")
    flower_payload = _load_key(np, paths["flower_masks"], "packed")
    visibility_flower_payload = (
        _load_key(np, paths["visibility_flower_masks"], "packed")
        if "visibility_flower_masks" in paths
        else flower_payload
    )
    hands_payload = _load_key(np, paths["pose_limb_masks"], "hands_packed")
    right_arm_payload = _load_key(np, paths["robot_limb_masks"], "right_packed")
    if any(
        len(payload[0]) != args.expected_frames
        for payload in (
            person_payload,
            flower_payload,
            visibility_flower_payload,
            hands_payload,
            right_arm_payload,
        )
    ):
        raise ValueError("packed masks must cover the complete timeline")

    lossless = output / "robot-chroma-state-lossless.mkv"
    review = output / "robot-chroma-state.mp4"
    lossless_writer = (
        _writer(
            paths["ffmpeg"], lossless, width=args.width, height=args.height,
            fps=args.fps, lossless=True,
        )
        if args.artifact_policy == "lossless_and_review"
        else None
    )
    review_writer = _review_writer(
        paths["ffmpeg"], review, width=args.width, height=args.height,
        fps=args.fps, preset=args.review_preset, crf=args.review_crf,
        chroma_qp_offset=args.review_chroma_qp_offset,
        pixel_format=args.review_pixel_format,
    )
    assert review_writer.stdin is not None
    if lossless_writer is not None:
        assert lossless_writer.stdin is not None
    audit_target_frame = SourceResizeCrop(
        name=f"camera:source_native_{args.width}x{args.height}",
        source_width=args.width,
        source_height=args.height,
        scaled_width=args.width,
        scaled_height=args.height,
        crop_left=0,
        crop_top=0,
        output_width=args.width,
        output_height=args.height,
    )
    audit_target_frame.validate()
    candidate_decoder = subprocess.Popen(
        _decoder_command(
            paths["ffmpeg"],
            paths["candidate"],
            source=False,
            target_frame=audit_target_frame,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    source_decoder = subprocess.Popen(
        _decoder_command(
            paths["ffmpeg"],
            paths["source"],
            source=True,
            target_frame=audit_target_frame,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    frame_bytes = args.width * args.height * 3
    packed_editable = []
    rows = []
    background_exact = background_total = 0
    flower_source_exact = flower_total = 0
    luma_delta = []
    unresolved_contact_pixels = 0
    started = time.perf_counter()
    batch_size = args.workers * 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for batch_start in range(0, args.expected_frames, batch_size):
            futures = []
            batch_end = min(args.expected_frames, batch_start + batch_size)
            for index in range(batch_start, batch_end):
                candidate = np.frombuffer(
                    _read_frame(
                        candidate_decoder, frame_bytes, "candidate", index
                    ),
                    dtype=np.uint8,
                ).reshape(args.height, args.width, 3)[..., ::-1].copy()
                source = np.frombuffer(
                    _read_frame(source_decoder, frame_bytes, "source", index),
                    dtype=np.uint8,
                ).reshape(args.height, args.width, 3)[..., ::-1].copy()
                person = _native_mask(cv2, np, person_payload, index, args)
                tracked_flower = _native_mask(cv2, np, flower_payload, index, args)
                visibility_flower = _native_mask(
                    cv2, np, visibility_flower_payload, index, args
                )
                hands = _native_mask(cv2, np, hands_payload, index, args)
                right_arm = _native_mask(cv2, np, right_arm_payload, index, args)
                futures.append(
                    executor.submit(
                        _process_record,
                        cv2,
                        np,
                        index=index,
                        candidate=candidate,
                        source=source,
                        person=person,
                        tracked_flower=tracked_flower,
                        visibility_flower=visibility_flower,
                        hands=hands,
                        right_arm=right_arm,
                        kernels=kernels,
                        args=args,
                    )
                )
            for future in futures:
                record = future.result()
                repaired = record["repaired"]
                if lossless_writer is not None:
                    lossless_writer.stdin.write(repaired.tobytes())
                review_writer.stdin.write(repaired.tobytes())
                packed_editable.append(record["packed_editable"])
                background_exact += record["background_exact"]
                background_total += record["background_total"]
                flower_source_exact += record["flower_source_exact"]
                flower_total += record["flower_total"]
                luma_delta.append(record["luma_delta"])
                unresolved_contact_pixels += int(
                    record["row"]["contact_completion"][
                        "unresolved_source_like_pixels"
                    ]
                )
                rows.append(record["row"])
    for decoder, name in (
        (candidate_decoder, "candidate"),
        (source_decoder, "source"),
    ):
        stderr = decoder.communicate()[1].decode(errors="replace")
        if decoder.returncode:
            raise RuntimeError(f"{name} decoder failed: {stderr}")
    lossless_log = (
        _finish(lossless_writer, "lossless")
        if lossless_writer is not None
        else None
    )
    review_log = _finish(review_writer, "review")
    generation_wall_seconds = time.perf_counter() - started

    audit_started = time.perf_counter()
    flat_pixels = args.width * args.height
    original = cv2.VideoCapture(str(paths["candidate"]))
    review_decoded = cv2.VideoCapture(str(review))
    review_before = []
    review_after = []
    decoded_frames = 0
    for index in range(args.expected_frames):
        original_ok, original_frame = original.read()
        review_ok, review_frame = review_decoded.read()
        if not original_ok or not review_ok:
            raise RuntimeError(f"post-decode audit stopped at frame {index}")
        decoded_frames += 1
        editable = np.unpackbits(packed_editable[index])[:flat_pixels].reshape(
            args.height, args.width
        ).astype(bool)
        review_before.append(_fast_spatial_chroma_tv(cv2, np, original_frame, editable))
        review_after.append(_fast_spatial_chroma_tv(cv2, np, review_frame, editable))
    original.release()
    review_decoded.release()
    review_before_mean = float(np.mean(review_before))
    review_after_mean = float(np.mean(review_after))
    review_reduction = (
        1.0 - review_after_mean / review_before_mean if review_before_mean > 0 else 0.0
    )
    audit_wall_seconds = time.perf_counter() - audit_started
    metrics = {
        "frames": decoded_frames,
        "video_seconds": args.expected_frames / args.fps,
        "wall_seconds": generation_wall_seconds,
        "generation_wall_seconds": generation_wall_seconds,
        "postdecode_audit_wall_seconds": audit_wall_seconds,
        "total_wall_seconds": generation_wall_seconds + audit_wall_seconds,
        "processing_fps": args.expected_frames / generation_wall_seconds,
        "outside_declared_alpha_and_flower_preencode_exact_fraction": (
            background_exact / max(1, background_total)
        ),
        "resolved_flower_preencode_source_exact_fraction": (
            flower_source_exact / max(1, flower_total)
        ),
        "editable_luma_mae": _summary(np, luma_delta),
        "unresolved_contact_corridor_pixels_preencode": unresolved_contact_pixels,
        "review_chroma_tv_before": _summary(np, review_before),
        "review_chroma_tv_after": _summary(np, review_after),
        "review_relative_mean_chroma_tv_reduction": review_reduction,
    }
    gates = {
        "full_timeline_decodes": decoded_frames == args.expected_frames,
        "outside_declared_alpha_and_flower_preencode_exact": (
            metrics[
                "outside_declared_alpha_and_flower_preencode_exact_fraction"
            ]
            == 1.0
        ),
        "resolved_flower_preencode_source_exact": (
            metrics["resolved_flower_preencode_source_exact_fraction"] == 1.0
        ),
        "luma_geometry_preserved": metrics["editable_luma_mae"]["p95"] <= args.maximum_luma_mae,
        "review_chroma_tv_reduced": review_reduction >= args.minimum_chroma_tv_reduction,
    }
    frame_metrics_path = output / "frame-metrics.json"
    frame_metrics_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "decision": "AWAITING_FROZEN_FULL_VIDEO_AUDITS",
        "method": (
            "parallel_single_roundtrip_masked_multiscale_hsv_state_projection_"
            "with_shared_audit_decoder"
        ),
        "physical_evidence": False,
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _packages(),
        "git": _git_state(),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items() if name != "ffmpeg"
        },
        "coordinate_frames": {
            "video": f"camera:source_native_{args.width}x{args.height}",
            "masks": f"{args.mask_projection}->{args.width}x{args.height}",
        },
        "config": {
            "kernel_sizes": kernels,
            "strength": args.strength,
            "maximum_chroma_delta": args.maximum_chroma_delta,
            "saturation_scale": args.saturation_scale,
            "replacement_threshold": args.replacement_threshold,
            "maximum_source_occlusion_gap": args.maximum_source_occlusion_gap,
            "contact_radius": args.contact_radius,
            "contact_codec_error_margin": args.contact_codec_error_margin,
            "contact_evidence_contract": "audit_aligned_chebyshev_grasp_corridor",
            "mask_resolver": (
                "audit_aligned_declared_alpha_with_source_owned_flower_footprint:"
                + args.flower_mask_contract
            ),
            "flower_mask_contract": args.flower_mask_contract,
            "artifact_policy": args.artifact_policy,
            "review_preset": args.review_preset,
            "review_crf": args.review_crf,
            "review_chroma_qp_offset": args.review_chroma_qp_offset,
            "review_pixel_format": args.review_pixel_format,
            "workers": args.workers,
            "thresholds_changed_by_run": False,
            "seed": args.seed,
        },
        "metrics": metrics,
        "gates": gates,
        "outputs": {
            "lossless": (
                {"path": str(lossless), "sha256": _sha256(lossless)}
                if lossless_writer is not None else None
            ),
            "review": {"path": str(review), "sha256": _sha256(review)},
            "frame_metrics": {
                "path": str(frame_metrics_path), "sha256": _sha256(frame_metrics_path)
            },
        },
        "encoder_logs": {"lossless": lossless_log, "review": review_log},
        "limitations": [
            "Only the generated robot layer chroma state is regularized.",
            "Resolved flower pixels are re-composited from the source-owned object layer.",
            "Luma is preserved as the image-space carrier of geometry and hand edges.",
            "This is perceptual video processing, not metric depth or contact-force evidence.",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "metrics": metrics, "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
