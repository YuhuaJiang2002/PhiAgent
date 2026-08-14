#!/usr/bin/env python3
"""Stabilize the anatomical right arm and enforce source-grounded flower z-order."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.temporal_appearance import (  # noqa: E402
    bidirectional_flow_state,
    warp_with_flow,
)
from phiagent.rendering.temporal_occlusion import (  # noqa: E402
    projected_contact_evidence_lock,
    right_arm_flower_partition,
    source_owned_flower_restore_mask,
    source_motion_residual_median_update,
)
from scripts.audit_robot_layer_long_video import _resolve_frame_masks  # noqa: E402
from scripts.compose_joyai_flower_repairs import _mask_to_native  # noqa: E402
from scripts.stabilize_joyai_appearance_state import (  # noqa: E402
    _finish,
    _git_state,
    _packages,
    _probe,
    _sha256,
    _summary,
    _writer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument(
        "--codec-reference-video",
        type=Path,
        help=(
            "Previous browser-codec decode used only by flower_codec_precompensate "
            "to close the encode/decode residual loop."
        ),
    )
    parser.add_argument("--robot-limb-masks", type=Path, required=True)
    parser.add_argument("--person-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--pose-limb-masks", type=Path, required=True)
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
    parser.add_argument("--right-arm-key", default="right_packed")
    parser.add_argument("--corridor-dilation-pixels", type=int, default=31)
    parser.add_argument("--hand-dilation-pixels", type=int, default=13)
    parser.add_argument("--flower-clean-plate-padding-pixels", type=int, default=9)
    parser.add_argument(
        "--flower-sample-footprint-pixels",
        type=int,
        default=7,
        help=(
            "Odd full-resolution footprint; 7 covers every phase of a 4x4 "
            "native-to-audit area sample containing a flower-owned pixel."
        ),
    )
    parser.add_argument("--arm-interior-erosion-pixels", type=int, default=5)
    parser.add_argument("--contact-evidence-threshold", type=float, default=12.0)
    parser.add_argument("--contact-evidence-dilation-pixels", type=int, default=7)
    parser.add_argument("--flow-scale", type=float, default=0.5)
    parser.add_argument("--minimum-confidence", type=float, default=0.2)
    parser.add_argument("--maximum-residual-delta", type=float, default=24.0)
    parser.add_argument("--person-dilation", type=int, default=10)
    parser.add_argument("--skin-negative-dilation", type=int, default=2)
    parser.add_argument("--person-core-negative-erosion", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=(
            "dual_motion_arm",
            "flower_codec_guard",
            "flower_codec_precompensate",
        ),
        default="dual_motion_arm",
        help=(
            "Run dual-flow arm stabilization plus flower ownership, or only "
            "re-establish the flower ownership band before the final review codec."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260814)
    return parser


def _load_key(np: Any, path: Path, key: str) -> tuple[Any, int, int, str]:
    payload = np.load(path, allow_pickle=False)
    if key not in payload.files:
        raise ValueError(f"{path} has no packed mask key {key!r}")
    return (
        payload[key],
        int(payload["height"]),
        int(payload["width"]),
        str(payload["bitorder"]),
    )


def _unpack(np: Any, payload: tuple[Any, int, int, str], index: int) -> Any:
    packed, height, width, bitorder = payload
    values = np.unpackbits(packed[index], bitorder=bitorder)[: height * width]
    return values.reshape(height, width).astype(np.uint8)


def _native_mask(
    cv2: Any,
    np: Any,
    payload: tuple[Any, int, int, str],
    index: int,
    args: Any,
) -> Any:
    return _mask_to_native(
        cv2,
        np,
        _unpack(np, payload, index),
        width=args.width,
        height=args.height,
        projection=args.mask_projection,
    )


def _read_record(
    cv2: Any,
    np: Any,
    *,
    index: int,
    candidate_capture: Any,
    source_capture: Any,
    codec_reference_capture: Any | None,
    right_payload: tuple[Any, int, int, str],
    person_payload: tuple[Any, int, int, str],
    flower_payload: tuple[Any, int, int, str],
    hands_payload: tuple[Any, int, int, str],
    interior_kernel: Any,
    args: Any,
) -> dict[str, Any]:
    candidate_ok, candidate = candidate_capture.read()
    source_ok, source = source_capture.read()
    if not candidate_ok or not source_ok:
        raise RuntimeError(f"video decode stopped at frame {index}")
    codec_reference = None
    if codec_reference_capture is not None:
        codec_reference_ok, codec_reference = codec_reference_capture.read()
        if not codec_reference_ok:
            raise RuntimeError(f"codec-reference decode stopped at frame {index}")
    right_arm = _native_mask(cv2, np, right_payload, index, args)
    person = _native_mask(cv2, np, person_payload, index, args)
    tracked_flower = _native_mask(cv2, np, flower_payload, index, args)
    hands = _native_mask(cv2, np, hands_payload, index, args)
    _, flower_visible = _resolve_frame_masks(
        np,
        source_rgb=cv2.cvtColor(source, cv2.COLOR_BGR2RGB),
        person=person,
        tracked_flower=tracked_flower,
        hands=hands,
        person_dilation=args.person_dilation,
        skin_negative_dilation=args.skin_negative_dilation,
        person_core_negative_erosion=args.person_core_negative_erosion,
        flower_mask_contract="resolved_visibility",
    )
    arm_editable, flower_owner, protected_hand = right_arm_flower_partition(
        cv2,
        np,
        right_arm=right_arm,
        flower_visible=flower_visible,
        hand_support=hands,
        corridor_dilation_pixels=args.corridor_dilation_pixels,
        hand_dilation_pixels=args.hand_dilation_pixels,
    )
    flower_restore = source_owned_flower_restore_mask(
        cv2,
        np,
        flower_owner=flower_owner,
        person=person,
        hand_core=hands,
        protected_hand=protected_hand,
        clean_plate_padding_pixels=args.flower_clean_plate_padding_pixels,
        sample_footprint_pixels=args.flower_sample_footprint_pixels,
    )
    contact_evidence_lock = projected_contact_evidence_lock(
        cv2,
        np,
        candidate=candidate,
        source=source,
        hand_core=hands,
        flower_owner=flower_owner,
        replacement_threshold=args.contact_evidence_threshold,
        contact_dilation_pixels=args.contact_evidence_dilation_pixels,
    )
    flower_restore = np.logical_and(
        flower_restore,
        np.logical_not(contact_evidence_lock),
    )
    arm_interior = cv2.erode(arm_editable.astype(np.uint8), interior_kernel) > 0
    return {
        "index": index,
        "candidate": candidate,
        "source": source,
        "codec_reference": codec_reference,
        "right_arm": right_arm,
        "arm_editable": arm_editable,
        "arm_interior": arm_interior,
        "flower_owner": flower_owner,
        "flower_restore": flower_restore,
        "contact_evidence_lock": contact_evidence_lock,
        "protected_hand": protected_hand,
    }


def _repair_dual_motion_arm(
    cv2: Any,
    np: Any,
    *,
    previous: dict[str, Any],
    current: dict[str, Any],
    following: dict[str, Any],
    args: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    previous_flow = bidirectional_flow_state(
        cv2,
        np,
        previous["source"],
        current["source"],
        scale=args.flow_scale,
    )
    next_flow = bidirectional_flow_state(
        cv2,
        np,
        following["source"],
        current["source"],
        scale=args.flow_scale,
    )
    previous_candidate_flow = bidirectional_flow_state(
        cv2,
        np,
        previous["candidate"],
        current["candidate"],
        scale=args.flow_scale,
    )
    next_candidate_flow = bidirectional_flow_state(
        cv2,
        np,
        following["candidate"],
        current["candidate"],
        scale=args.flow_scale,
    )
    warped_previous_arm = warp_with_flow(
        cv2, previous["arm_editable"].astype(np.uint8), previous_flow, nearest=True
    ) > 0
    warped_next_arm = warp_with_flow(
        cv2, following["arm_editable"].astype(np.uint8), next_flow, nearest=True
    ) > 0
    warped_previous_candidate_arm = warp_with_flow(
        cv2,
        previous["arm_editable"].astype(np.uint8),
        previous_candidate_flow,
        nearest=True,
    ) > 0
    warped_next_candidate_arm = warp_with_flow(
        cv2,
        following["arm_editable"].astype(np.uint8),
        next_candidate_flow,
        nearest=True,
    ) > 0
    reliable = np.logical_and.reduce(
        (
            current["arm_interior"],
            warped_previous_arm,
            warped_next_arm,
            warped_previous_candidate_arm,
            warped_next_candidate_arm,
            previous_flow.confidence >= args.minimum_confidence,
            next_flow.confidence >= args.minimum_confidence,
            previous_candidate_flow.confidence >= args.minimum_confidence,
            next_candidate_flow.confidence >= args.minimum_confidence,
        )
    )
    current_residual = (
        current["candidate"].astype(np.float32)
        - current["source"].astype(np.float32)
    )
    previous_residual = (
        previous["candidate"].astype(np.float32)
        - previous["source"].astype(np.float32)
    )
    next_residual = (
        following["candidate"].astype(np.float32)
        - following["source"].astype(np.float32)
    )
    source_repaired, metrics = source_motion_residual_median_update(
        np,
        current_candidate=current["candidate"],
        current_residual=current_residual,
        warped_previous_residual=warp_with_flow(
            cv2, previous_residual, previous_flow
        ),
        warped_next_residual=warp_with_flow(cv2, next_residual, next_flow),
        reliable=reliable,
        maximum_residual_delta=args.maximum_residual_delta,
    )
    repaired, self_metrics = source_motion_residual_median_update(
        np,
        current_candidate=source_repaired,
        current_residual=source_repaired.astype(np.float32),
        warped_previous_residual=warp_with_flow(
            cv2, previous["candidate"].astype(np.float32), previous_candidate_flow
        ),
        warped_next_residual=warp_with_flow(
            cv2, following["candidate"].astype(np.float32), next_candidate_flow
        ),
        reliable=reliable,
        maximum_residual_delta=args.maximum_residual_delta,
    )
    metrics.update({f"candidate_{key}": value for key, value in self_metrics.items()})
    return repaired, reliable, metrics


def _repair_record(
    cv2: Any,
    np: Any,
    *,
    previous: dict[str, Any],
    current: dict[str, Any],
    following: dict[str, Any],
    args: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    if args.mode == "dual_motion_arm":
        repaired, reliable, metrics = _repair_dual_motion_arm(
            cv2,
            np,
            previous=previous,
            current=current,
            following=following,
            args=args,
        )
    elif args.mode == "flower_codec_guard":
        repaired = current["candidate"].copy()
        reliable = np.zeros(current["right_arm"].shape, dtype=np.bool_)
        metrics = {}
    else:
        if current["codec_reference"] is None:
            raise RuntimeError("codec feedback mode requires a codec-reference frame")
        reliable = np.zeros(current["right_arm"].shape, dtype=np.bool_)
        codec_delta = np.clip(
            current["source"].astype(np.float32)
            - current["codec_reference"].astype(np.float32),
            -args.maximum_residual_delta,
            args.maximum_residual_delta,
        )
        repaired = np.clip(
            current["candidate"].astype(np.float32) + codec_delta,
            0,
            255,
        ).astype(np.uint8)
        codec_support = np.asarray(current["flower_restore"], dtype=np.bool_)
        codec_reference_error = np.abs(
            current["source"].astype(np.float32)
            - current["codec_reference"].astype(np.float32)
        )
        metrics = {
            "codec_reference_target_mae": (
                float(codec_reference_error[codec_support].mean())
                if np.any(codec_support)
                else 0.0
            ),
            "codec_feedback_mean_abs_correction": (
                float(np.abs(codec_delta)[codec_support].mean())
                if np.any(codec_support)
                else 0.0
            ),
            "codec_feedback_max_abs_correction": (
                float(np.abs(codec_delta)[codec_support].max())
                if np.any(codec_support)
                else 0.0
            ),
        }

    flower_owner = np.asarray(current["flower_owner"], dtype=np.bool_)
    flower_restore = np.asarray(current["flower_restore"], dtype=np.bool_)
    if args.mode != "flower_codec_precompensate":
        repaired[flower_restore] = current["source"][flower_restore]
    mutable = np.logical_or(reliable, flower_restore)
    repaired[np.logical_not(mutable)] = current["candidate"][np.logical_not(mutable)]
    metrics.update(
        {
            "frame": int(current["index"]),
            "right_arm_pixels": int(np.count_nonzero(current["right_arm"])),
            "reliable_arm_pixels": int(np.count_nonzero(reliable)),
            "flower_owner_pixels": int(np.count_nonzero(flower_owner)),
            "flower_restore_pixels": int(np.count_nonzero(flower_restore)),
            "protected_hand_pixels": int(np.count_nonzero(current["protected_hand"])),
            "contact_evidence_lock_pixels": int(
                np.count_nonzero(current["contact_evidence_lock"])
            ),
        }
    )
    return repaired, mutable, metrics


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    paths = {
        "candidate": args.candidate_video.expanduser().resolve(),
        "source": args.source_video.expanduser().resolve(),
        "robot_limb_masks": args.robot_limb_masks.expanduser().resolve(),
        "person_masks": args.person_masks.expanduser().resolve(),
        "flower_masks": args.flower_masks.expanduser().resolve(),
        "pose_limb_masks": args.pose_limb_masks.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    if args.codec_reference_video is not None:
        paths["codec_reference"] = args.codec_reference_video.expanduser().resolve()
    if args.mode == "flower_codec_precompensate" and "codec_reference" not in paths:
        raise ValueError(
            "--codec-reference-video is required by flower_codec_precompensate"
        )
    if args.mode != "flower_codec_precompensate" and "codec_reference" in paths:
        raise ValueError(
            "--codec-reference-video is only valid with flower_codec_precompensate"
        )
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{name}: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    for pixels, label in (
        (args.corridor_dilation_pixels, "corridor dilation"),
        (args.hand_dilation_pixels, "hand dilation"),
        (args.flower_clean_plate_padding_pixels, "flower clean-plate padding"),
        (args.flower_sample_footprint_pixels, "flower sample footprint"),
        (args.contact_evidence_dilation_pixels, "contact evidence dilation"),
        (args.arm_interior_erosion_pixels, "arm interior erosion"),
    ):
        if pixels < 1 or pixels % 2 == 0:
            raise ValueError(f"{label} must be a positive odd integer")
    if args.expected_frames < 3:
        raise ValueError("at least three frames are required")

    required = {
        "frames": args.expected_frames,
        "width": args.width,
        "height": args.height,
    }
    for name in ("candidate", "source", "codec_reference"):
        if name not in paths:
            continue
        probe = _probe(cv2, paths[name])
        if any(probe[key] != value for key, value in required.items()):
            raise ValueError(f"{name} video does not match the timeline: {probe}")
        if abs(float(probe["fps"]) - args.fps) > 0.01:
            raise ValueError(f"{name} FPS mismatch: {probe}")

    right_payload = _load_key(np, paths["robot_limb_masks"], args.right_arm_key)
    person_payload = _load_key(np, paths["person_masks"], "packed")
    flower_payload = _load_key(np, paths["flower_masks"], "packed")
    hands_payload = _load_key(np, paths["pose_limb_masks"], "hands_packed")
    if any(
        len(payload[0]) != args.expected_frames
        for payload in (right_payload, person_payload, flower_payload, hands_payload)
    ):
        raise ValueError("all packed masks must cover the complete timeline")

    lossless = output / "right-arm-flower-zorder-lossless.mkv"
    review = output / "right-arm-flower-zorder.mp4"
    lossless_writer = _writer(
        paths["ffmpeg"], lossless, width=args.width, height=args.height,
        fps=args.fps, lossless=True,
    )
    review_writer = _writer(
        paths["ffmpeg"], review, width=args.width, height=args.height,
        fps=args.fps, lossless=False,
    )
    assert lossless_writer.stdin is not None and review_writer.stdin is not None
    candidate_capture = cv2.VideoCapture(str(paths["candidate"]))
    source_capture = cv2.VideoCapture(str(paths["source"]))
    codec_reference_capture = (
        cv2.VideoCapture(str(paths["codec_reference"]))
        if "codec_reference" in paths
        else None
    )
    interior_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.arm_interior_erosion_pixels, args.arm_interior_erosion_pixels),
    )

    def read(index: int) -> dict[str, Any]:
        return _read_record(
            cv2,
            np,
            index=index,
            candidate_capture=candidate_capture,
            source_capture=source_capture,
            codec_reference_capture=codec_reference_capture,
            right_payload=right_payload,
            person_payload=person_payload,
            flower_payload=flower_payload,
            hands_payload=hands_payload,
            interior_kernel=interior_kernel,
            args=args,
        )

    started = time.perf_counter()
    previous = read(0)
    current = read(1)
    if args.mode == "dual_motion_arm":
        first_frame = previous["candidate"]
        first_mutable = np.zeros((args.height, args.width), dtype=np.bool_)
        first_metrics: dict[str, Any] = {"frame": 0, "first_frame_exact": True}
    else:
        first_frame, first_mutable, first_metrics = _repair_record(
            cv2,
            np,
            previous=previous,
            current=previous,
            following=current,
            args=args,
        )
    lossless_writer.stdin.write(first_frame.tobytes())
    review_writer.stdin.write(first_frame.tobytes())
    packed_mutable = [
        np.packbits(first_mutable.reshape(-1).astype(np.uint8))
    ]
    packed_flower_owner = [
        np.packbits(previous["flower_owner"].reshape(-1).astype(np.uint8))
    ]
    rows: list[dict[str, Any]] = [first_metrics]
    anteprevious = previous
    for index in range(2, args.expected_frames):
        following = read(index)
        repaired, mutable, metrics = _repair_record(
            cv2,
            np,
            previous=previous,
            current=current,
            following=following,
            args=args,
        )
        lossless_writer.stdin.write(repaired.tobytes())
        review_writer.stdin.write(repaired.tobytes())
        packed_mutable.append(np.packbits(mutable.reshape(-1).astype(np.uint8)))
        packed_flower_owner.append(
            np.packbits(current["flower_owner"].reshape(-1).astype(np.uint8))
        )
        rows.append(metrics)
        anteprevious, previous, current = previous, current, following

    repaired, mutable, metrics = _repair_record(
        cv2,
        np,
        previous=anteprevious,
        current=current,
        following=previous,
        args=args,
    )
    metrics["one_sided_tail_consensus"] = True
    lossless_writer.stdin.write(repaired.tobytes())
    review_writer.stdin.write(repaired.tobytes())
    packed_mutable.append(np.packbits(mutable.reshape(-1).astype(np.uint8)))
    packed_flower_owner.append(
        np.packbits(current["flower_owner"].reshape(-1).astype(np.uint8))
    )
    rows.append(metrics)
    candidate_capture.release()
    source_capture.release()
    if codec_reference_capture is not None:
        codec_reference_capture.release()
    lossless_log = _finish(lossless_writer, "lossless")
    review_log = _finish(review_writer, "review")
    wall_seconds = time.perf_counter() - started

    decoded = cv2.VideoCapture(str(lossless))
    review_decoded = cv2.VideoCapture(str(review))
    original = cv2.VideoCapture(str(paths["candidate"]))
    source = cv2.VideoCapture(str(paths["source"]))
    codec_reference_audit = (
        cv2.VideoCapture(str(paths["codec_reference"]))
        if "codec_reference" in paths
        else None
    )
    outside_exact = outside_total = 0
    flower_exact = flower_total = 0
    review_flower_abs_error = 0.0
    codec_reference_flower_abs_error = 0.0
    flat_pixels = args.width * args.height
    for index in range(args.expected_frames):
        decoded_ok, decoded_frame = decoded.read()
        review_ok, review_frame = review_decoded.read()
        original_ok, original_frame = original.read()
        source_ok, source_frame = source.read()
        if not decoded_ok or not review_ok or not original_ok or not source_ok:
            raise RuntimeError(f"post-decode audit stopped at frame {index}")
        codec_reference_frame = None
        if codec_reference_audit is not None:
            codec_reference_ok, codec_reference_frame = codec_reference_audit.read()
            if not codec_reference_ok:
                raise RuntimeError(
                    f"codec-reference post-decode audit stopped at frame {index}"
                )
        mutable_mask = np.unpackbits(packed_mutable[index])[:flat_pixels].reshape(
            args.height, args.width
        ).astype(bool)
        flower_mask = np.unpackbits(packed_flower_owner[index])[:flat_pixels].reshape(
            args.height, args.width
        ).astype(bool)
        outside = np.logical_not(mutable_mask)
        outside_exact += int(
            np.count_nonzero(np.all(decoded_frame == original_frame, axis=2)[outside])
        )
        outside_total += int(np.count_nonzero(outside))
        flower_exact += int(
            np.count_nonzero(np.all(decoded_frame == source_frame, axis=2)[flower_mask])
        )
        review_flower_abs_error += float(
            np.abs(
                review_frame.astype(np.float32) - source_frame.astype(np.float32)
            )[flower_mask].sum()
        )
        if codec_reference_frame is not None:
            codec_reference_flower_abs_error += float(
                np.abs(
                    codec_reference_frame.astype(np.float32)
                    - source_frame.astype(np.float32)
                )[flower_mask].sum()
            )
        flower_total += int(np.count_nonzero(flower_mask))
    decoded.release()
    review_decoded.release()
    original.release()
    source.release()
    if codec_reference_audit is not None:
        codec_reference_audit.release()

    baseline = [
        float(row["baseline_temporal_extremum_mae"])
        for row in rows if "baseline_temporal_extremum_mae" in row
    ]
    repaired_values = [
        float(row["repaired_temporal_extremum_mae"])
        for row in rows if "repaired_temporal_extremum_mae" in row
    ]
    applied = [
        float(row["mean_abs_applied_correction"])
        for row in rows if "mean_abs_applied_correction" in row
    ]
    codec_feedback = [
        float(row["codec_feedback_mean_abs_correction"])
        for row in rows if "codec_feedback_mean_abs_correction" in row
    ]
    codec_feedback_maximum = max(
        (
            float(row["codec_feedback_max_abs_correction"])
            for row in rows if "codec_feedback_max_abs_correction" in row
        ),
        default=0.0,
    )
    review_flower_target_mae = (
        review_flower_abs_error / (flower_total * 3) if flower_total else 0.0
    )
    codec_reference_flower_target_mae = (
        codec_reference_flower_abs_error / (flower_total * 3)
        if flower_total and codec_reference_audit is not None
        else None
    )
    metrics = {
        "baseline_temporal_extremum_mae": _summary(np, baseline),
        "repaired_temporal_extremum_mae": _summary(np, repaired_values),
        "relative_mean_temporal_extremum_reduction": (
            1.0 - float(np.mean(repaired_values)) / float(np.mean(baseline))
            if baseline and float(np.mean(baseline)) > 0 else None
        ),
        "mean_applied_correction": _summary(np, applied),
        "codec_feedback_mean_abs_correction": _summary(np, codec_feedback),
        "codec_feedback_max_abs_correction": codec_feedback_maximum,
        "postdecode_outside_mutable_exact_fraction": outside_exact / outside_total,
        "postdecode_flower_owner_source_exact_fraction": (
            flower_exact / flower_total if flower_total else 1.0
        ),
        "flower_owner_pixels": flower_total,
        "postdecode_review_flower_target_mae": review_flower_target_mae,
        "codec_reference_flower_target_mae": codec_reference_flower_target_mae,
        "frames": args.expected_frames,
        "video_seconds": args.expected_frames / args.fps,
        "wall_seconds": wall_seconds,
        "processing_fps": args.expected_frames / wall_seconds,
    }
    gates = {
        "full_timeline_decodes": _probe(cv2, lossless)["frames"] == args.expected_frames,
        "outside_mutable_exact": (
            metrics["postdecode_outside_mutable_exact_fraction"] == 1.0
        ),
        "flower_owner_source_exact": (
            args.mode == "flower_codec_precompensate"
            or metrics["postdecode_flower_owner_source_exact_fraction"] == 1.0
        ),
        "codec_feedback_bounded": (
            args.mode != "flower_codec_precompensate"
            or metrics["codec_feedback_max_abs_correction"]
            <= args.maximum_residual_delta
        ),
        "codec_feedback_reduces_review_target_mae": (
            args.mode != "flower_codec_precompensate"
            or (
                metrics["codec_reference_flower_target_mae"] is not None
                and metrics["postdecode_review_flower_target_mae"]
                < metrics["codec_reference_flower_target_mae"]
            )
        ),
        "temporal_extremum_non_regression": (
            args.mode in ("flower_codec_guard", "flower_codec_precompensate")
            or (
                metrics["relative_mean_temporal_extremum_reduction"] is not None
                and metrics["relative_mean_temporal_extremum_reduction"] >= 0
            )
        ),
    }
    frame_metrics = output / "frame-metrics.json"
    frame_metrics.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "decision": "AWAITING_FROZEN_RIGHT_ARM_AND_FULL_VIDEO_AUDITS",
        "method": (
            "dual_motion_arm_plus_explicit_flower_front_zorder"
            if args.mode == "dual_motion_arm"
            else (
                "codec_block_flower_front_zorder_guard"
                if args.mode == "flower_codec_guard"
                else "bounded_closed_loop_codec_flower_precompensation"
            )
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
            "timeline": f"absolute_frame_index:full_source_{args.expected_frames}",
        },
        "config": {
            "mode": args.mode,
            "right_arm_key": args.right_arm_key,
            "corridor_dilation_pixels": args.corridor_dilation_pixels,
            "hand_dilation_pixels": args.hand_dilation_pixels,
            "flower_clean_plate_padding_pixels": (
                args.flower_clean_plate_padding_pixels
            ),
            "flower_sample_footprint_pixels": args.flower_sample_footprint_pixels,
            "contact_evidence_threshold": args.contact_evidence_threshold,
            "contact_evidence_dilation_pixels": (
                args.contact_evidence_dilation_pixels
            ),
            "arm_interior_erosion_pixels": args.arm_interior_erosion_pixels,
            "flow_reference": (
                "source_and_candidate"
                if args.mode == "dual_motion_arm"
                else "not_used"
            ),
            "flow_scale": args.flow_scale,
            "minimum_confidence": args.minimum_confidence,
            "maximum_residual_delta": args.maximum_residual_delta,
            "seed": args.seed,
        },
        "metrics": metrics,
        "gates": gates,
        "outputs": {
            "lossless": {"path": str(lossless), "sha256": _sha256(lossless)},
            "review": {"path": str(review), "sha256": _sha256(review)},
            "frame_metrics": {
                "path": str(frame_metrics), "sha256": _sha256(frame_metrics)
            },
        },
        "encoder_logs": {"lossless": lossless_log, "review": review_log},
        "limitations": [
            "The source-grounded z-order is a 2-D camera-frame ownership contract.",
            "The protected hand band preserves prior projected-contact evidence.",
            "Metric depth, contact force, and force closure remain unobserved."
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "metrics": metrics, "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
