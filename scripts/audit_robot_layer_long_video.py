#!/usr/bin/env python3
"""Adversarially audit a long robot RGB-alpha-contact video layer.

The audit is object-track grounded.  It evaluates appearance only inside an
explicit robot alpha, topology on named source arm/hand supports, and projected
contact at the tracked hand/object boundary.  It also attacks each candidate
with a late magenta cast, arm erasure, and hand/object detachment and verifies
that the corresponding gate fires.

Projected contact is a two-dimensional image-space check, not evidence of
force closure, collision safety, depth correctness, or robot executability.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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

from phiagent.rendering.object_factored_long_video import (  # noqa: E402
    SourceResizeCrop,
    binary_dilate_square,
    remap_boolean_mask,
    resolve_flower_visibility,
    source_skin_like,
    strict_flower_seed,
)
from phiagent.rendering.robot_layer_contract import (  # noqa: E402
    RobotLayerContract,
    canonical_palette_histogram,
    frame_contract_metrics,
    occlusion_aware_grasp_metrics,
    robust_limit,
)


UPPER_METRICS = {
    "palette_surprisal": 0.40,
    "high_chroma_fraction": 0.03,
    "skin_like_fraction": 0.03,
    "spatial_chroma_tv": 3.0,
    "hand_edge_energy_upper_gate": 6.0,
}
LOWER_METRICS = {
    "replacement_coverage": 0.08,
    "arm_replacement_coverage": 0.12,
    "hand_replacement_coverage": 0.15,
    "grid_topology_coverage": 0.125,
    "hand_edge_energy_lower_gate": 6.0,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="NAME=VIDEO; may be supplied more than once",
    )
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--person-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument(
        "--flower-mask-contract",
        choices=("resolved_visibility", "exact_tracked"),
        default="resolved_visibility",
        help=(
            "Use exact_tracked when the candidate compositor locked the supplied "
            "flower track byte-for-byte; otherwise use conservative z-order resolution."
        ),
    )
    parser.add_argument("--limb-masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("ffprobe"))
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--reference-frame", type=int, default=276)
    parser.add_argument("--anchor-start", type=int, default=259)
    parser.add_argument("--anchor-end-exclusive", type=int, default=297)
    parser.add_argument("--late-start", type=int, default=480)
    parser.add_argument("--person-dilation", type=int, default=10)
    parser.add_argument("--skin-negative-dilation", type=int, default=2)
    parser.add_argument("--person-core-negative-erosion", type=int, default=2)
    parser.add_argument("--replacement-threshold", type=float, default=12.0)
    parser.add_argument("--contact-radius", type=int, default=3)
    parser.add_argument("--allowed-late-violation-fraction", type=float, default=0.10)
    parser.add_argument("--required-contact-recall", type=float, default=0.95)
    parser.add_argument(
        "--persistent-grasp-start",
        type=int,
        default=-1,
        help="Inclusive source-observed hold frame; negative disables the dense gate.",
    )
    parser.add_argument(
        "--persistent-grasp-end-exclusive",
        type=int,
        default=-1,
        help="Exclusive source-observed hold frame.",
    )
    parser.add_argument("--maximum-source-occlusion-gap", type=int, default=24)
    parser.add_argument("--minimum-occlusion-bridge-coverage", type=float, default=0.80)
    parser.add_argument("--required-persistent-grasp-recall", type=float, default=1.0)
    parser.add_argument("--adversarial-stride", type=int, default=12)
    parser.add_argument(
        "--frozen-limits-report",
        type=Path,
        help=(
            "Prior immutable audit report whose anchor-fitted metric limits "
            "must be reused verbatim for every challenger."
        ),
    )
    parser.add_argument(
        "--frozen-limits-candidate",
        help="Candidate name inside --frozen-limits-report; defaults to its recommendation.",
    )
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


def _git_state() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, command in {
        "head": ["git", "rev-parse", "--verify", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        result[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    return result


def _parse_candidates(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    names = set()
    for value in values:
        if "=" not in value:
            raise ValueError("candidate must have NAME=VIDEO form")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser().resolve()
        if not name or name in names:
            raise ValueError("candidate names must be non-empty and unique")
        if not path.is_file():
            raise ValueError(f"candidate video does not exist: {path}")
        names.add(name)
        result.append((name, path))
    return result


def _load_packed(np: Any, path: Path, key: str) -> tuple[Any, dict[str, Any]]:
    payload = np.load(path, allow_pickle=False)
    if key not in payload.files:
        raise ValueError(f"{path} has no packed mask key {key!r}")
    metadata = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "key": key,
        "frames": int(len(payload[key])),
        "height": int(payload["height"]),
        "width": int(payload["width"]),
        "bitorder": str(payload["bitorder"]),
    }
    return payload[key], metadata


def _unpack(np: Any, packed: Any, metadata: dict[str, Any], index: int) -> Any:
    values = np.unpackbits(
        packed[index], bitorder=metadata["bitorder"]
    )[: metadata["height"] * metadata["width"]]
    return values.reshape(metadata["height"], metadata["width"]).astype(bool)


def _decoder_command(
    ffmpeg: Path,
    path: Path,
    *,
    source: bool,
    target_frame: SourceResizeCrop,
) -> list[str]:
    command = [str(ffmpeg), "-v", "error", "-i", str(path)]
    if source:
        command.extend(
            [
                "-vf",
                (
                    f"scale={target_frame.scaled_width}:{target_frame.scaled_height}:"
                    "flags=area,"
                    f"crop={target_frame.output_width}:{target_frame.output_height}:"
                    f"{target_frame.crop_left}:{target_frame.crop_top}"
                ),
            ]
        )
    command.extend(["-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return command


def _read_frame(process: subprocess.Popen[bytes], frame_bytes: int, label: str, index: int) -> bytes:
    assert process.stdout is not None
    chunks = []
    remaining = frame_bytes
    while remaining:
        chunk = process.stdout.read(remaining)
        if not chunk:
            raise RuntimeError(f"{label} decoder ended before frame {index}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode_reference(
    np: Any,
    ffmpeg: Path,
    path: Path,
    target_frame: SourceResizeCrop,
) -> Any:
    completed = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            (
                f"scale={target_frame.scaled_width}:{target_frame.scaled_height}:flags=area,"
                f"crop={target_frame.output_width}:{target_frame.output_height}:"
                f"{target_frame.crop_left}:{target_frame.crop_top}"
            ),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    expected = target_frame.output_width * target_frame.output_height * 3
    if len(completed.stdout) != expected:
        raise ValueError("reference decoder returned an unexpected byte count")
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
        target_frame.output_height, target_frame.output_width, 3
    )


def _mapped_masks(
    np: Any,
    *,
    index: int,
    mask_frame: SourceResizeCrop,
    target_frame: SourceResizeCrop,
    packed_person: Any,
    person_meta: dict[str, Any],
    packed_flower: Any,
    flower_meta: dict[str, Any],
    packed_arms: Any,
    arm_meta: dict[str, Any],
    packed_hands: Any,
    hand_meta: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    values = []
    for packed, metadata in (
        (packed_person, person_meta),
        (packed_flower, flower_meta),
        (packed_arms, arm_meta),
        (packed_hands, hand_meta),
    ):
        values.append(
            remap_boolean_mask(
                np,
                _unpack(np, packed, metadata, index),
                source_frame=mask_frame,
                target_frame=target_frame,
            )
        )
    return tuple(values)


def _resolve_frame_masks(
    np: Any,
    *,
    source_rgb: Any,
    person: Any,
    tracked_flower: Any,
    hands: Any,
    person_dilation: int,
    skin_negative_dilation: int,
    person_core_negative_erosion: int,
    flower_mask_contract: str,
) -> tuple[Any, Any]:
    support = binary_dilate_square(np, person, person_dilation)
    skin = np.logical_and(
        source_skin_like(np, source_rgb), np.logical_or(person, hands)
    )
    skin = binary_dilate_square(
        np, np.logical_and(skin, support), skin_negative_dilation
    )
    if flower_mask_contract == "exact_tracked":
        flower = np.asarray(tracked_flower, dtype=np.bool_).copy()
    else:
        flower = resolve_flower_visibility(
            np,
            candidates=np.logical_or(tracked_flower, strict_flower_seed(np, source_rgb)),
            edit_support=support,
            source_person=person,
            source_skin_negative=skin,
            person_core_erosion=person_core_negative_erosion,
        )
    return support, flower


def _summary(
    np: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    frozen_limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    anchor = [
        row
        for row in rows
        if args.anchor_start <= row["frame"] < args.anchor_end_exclusive
    ]
    late = [row for row in rows if row["frame"] >= args.late_start]
    if not anchor or not late:
        raise ValueError("anchor and late intervals must both contain frames")
    required_limit_keys = set(UPPER_METRICS) | set(LOWER_METRICS)
    if frozen_limits is None:
        limits: dict[str, float] = {}
        for metric, margin in UPPER_METRICS.items():
            limits[metric] = robust_limit(
                np,
                [row[metric] for row in anchor],
                direction="upper",
                minimum_margin=margin,
            )
        for metric, margin in LOWER_METRICS.items():
            limits[metric] = robust_limit(
                np,
                [row[metric] for row in anchor],
                direction="lower",
                minimum_margin=margin,
            )
    else:
        missing = required_limit_keys - set(frozen_limits)
        if missing:
            raise ValueError(f"frozen limits omit metrics: {sorted(missing)}")
        limits = {
            metric: float(frozen_limits[metric]) for metric in required_limit_keys
        }
        if not all(np.isfinite(value) for value in limits.values()):
            raise ValueError("frozen limits must be finite")
    sections = {}
    for section_name, selected in {
        "anchor": anchor,
        "pre_20_seconds": [row for row in rows if row["frame"] < args.late_start],
        "at_or_after_20_seconds": late,
    }.items():
        metrics: dict[str, Any] = {}
        for metric in (*UPPER_METRICS, *LOWER_METRICS):
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            if metric in UPPER_METRICS:
                violation = values > limits[metric]
            else:
                violation = values < limits[metric]
            metrics[metric] = {
                "mean": float(values.mean()),
                "p05": float(np.quantile(values, 0.05)),
                "p95": float(np.quantile(values, 0.95)),
                "limit": limits[metric],
                "violation_fraction": float(violation.mean()),
            }
        required = [row for row in selected if row["contact_required"]]
        contact_recall = (
            sum(bool(row["contact_observed"]) for row in required) / len(required)
            if required
            else 1.0
        )
        sections[section_name] = {
            "frames": len(selected),
            "required_contact_frames": len(required),
            "projected_contact_recall": contact_recall,
            "metrics": metrics,
        }
    late_section = sections["at_or_after_20_seconds"]
    gate_results = {
        f"late_{metric}": late_section["metrics"][metric]["violation_fraction"]
        <= args.allowed_late_violation_fraction
        for metric in (*UPPER_METRICS, *LOWER_METRICS)
    }
    gate_results["late_projected_contact_recall"] = (
        late_section["projected_contact_recall"] >= args.required_contact_recall
    )
    persistent: dict[str, Any] | None = None
    if args.persistent_grasp_start >= 0:
        selected = [
            row
            for row in rows
            if args.persistent_grasp_start
            <= row["frame"]
            < args.persistent_grasp_end_exclusive
        ]
        expected = args.persistent_grasp_end_exclusive - args.persistent_grasp_start
        if len(selected) != expected:
            raise ValueError("persistent grasp interval is not fully represented")
        source_observable = sum(bool(row["source_hold_observable"]) for row in selected)
        visual_passes = sum(bool(row["visual_grasp_pass"]) for row in selected)
        baseline_positive = [row for row in selected if row["visual_grasp_pass"]]
        attack_rejections = sum(
            not bool(row["grasp_erasure_attack_pass"]) for row in baseline_positive
        )
        gaps = [
            int(row["source_hand_object_gap_pixels"])
            for row in selected
            if row["source_hand_object_gap_pixels"] is not None
        ]
        persistent = {
            "start_frame_inclusive": args.persistent_grasp_start,
            "end_frame_exclusive": args.persistent_grasp_end_exclusive,
            "frames": len(selected),
            "maximum_source_occlusion_gap_pixels": args.maximum_source_occlusion_gap,
            "minimum_occlusion_bridge_coverage": args.minimum_occlusion_bridge_coverage,
            "source_hold_observable_recall": source_observable / max(1, len(selected)),
            "visual_grasp_recall": visual_passes / max(1, len(selected)),
            "maximum_observed_source_gap_pixels": max(gaps) if gaps else None,
            "grasp_erasure_attack_positive_frames": len(baseline_positive),
            "grasp_erasure_attack_rejection_recall": (
                attack_rejections / len(baseline_positive) if baseline_positive else 0.0
            ),
        }
        persistent["gates"] = {
            "source_hold_observable_all_frames": source_observable == len(selected),
            "persistent_visual_grasp_recall": (
                persistent["visual_grasp_recall"]
                >= args.required_persistent_grasp_recall
            ),
            "grasp_erasure_attack_detected_all_frames": (
                len(baseline_positive) == len(selected)
                and attack_rejections == len(baseline_positive)
            ),
        }
        gate_results.update(
            {
                f"persistent_{name}": bool(value)
                for name, value in persistent["gates"].items()
            }
        )
    return {
        "limits_fit_only_on_anchor_frames": limits,
        "sections": sections,
        "persistent_grasp": persistent,
        "gates": gate_results,
        "image_space_contract_pass": all(gate_results.values()),
    }


def _attack_metrics(
    np: Any,
    *,
    candidate: Any,
    source: Any,
    alpha: Any,
    arms: Any,
    hands: Any,
    flower: Any,
    palette: Any,
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    kwargs = {
        "np": np,
        "source_rgb": source,
        "robot_alpha": alpha,
        "arm_support": arms,
        "hand_support": hands,
        "object_mask": flower,
        "palette": palette,
        "replacement_threshold": args.replacement_threshold,
        "contact_radius": args.contact_radius,
    }
    baseline = frame_contract_metrics(candidate_rgb=candidate, **kwargs)
    evaluation = np.logical_and(alpha, np.logical_not(flower))
    color = candidate.copy()
    color[evaluation] = np.asarray([225, 25, 185], dtype=np.uint8)
    color_metrics = frame_contract_metrics(candidate_rgb=color, **kwargs)
    topology = candidate.copy()
    arm_evaluation = np.logical_and(arms, evaluation)
    topology[arm_evaluation] = source[arm_evaluation]
    topology_metrics = frame_contract_metrics(candidate_rgb=topology, **kwargs)
    contact = candidate.copy()
    detached = np.logical_and(
        binary_dilate_square(np, hands, args.contact_radius), evaluation
    )
    contact[detached] = source[detached]
    contact_metrics = frame_contract_metrics(candidate_rgb=contact, **kwargs)
    structure = candidate.copy()
    shifted = np.roll(candidate, 4, axis=1)
    ghosted = np.rint(
        candidate.astype(np.float32) * 0.5 + shifted.astype(np.float32) * 0.5
    ).astype(np.uint8)
    hand_evaluation = np.logical_and(hands, evaluation)
    structure[hand_evaluation] = ghosted[hand_evaluation]
    structure_metrics = frame_contract_metrics(candidate_rgb=structure, **kwargs)
    return baseline, color_metrics, topology_metrics, contact_metrics, structure_metrics


def _audit_candidate(
    np: Any,
    *,
    name: str,
    path: Path,
    args: argparse.Namespace,
    mask_frame: SourceResizeCrop,
    target_frame: SourceResizeCrop,
    packed_masks: tuple[Any, dict[str, Any], Any, dict[str, Any], Any, dict[str, Any], Any, dict[str, Any]],
    palette: Any,
    frozen_limits: dict[str, float] | None,
) -> dict[str, Any]:
    packed_person, person_meta, packed_flower, flower_meta, packed_arms, arm_meta, packed_hands, hand_meta = packed_masks
    source_command = _decoder_command(
        args.ffmpeg, args.source_video.resolve(), source=True, target_frame=target_frame
    )
    candidate_command = _decoder_command(
        args.ffmpeg, path, source=False, target_frame=target_frame
    )
    source_decoder = subprocess.Popen(
        source_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    candidate_decoder = subprocess.Popen(
        candidate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    frame_bytes = target_frame.output_width * target_frame.output_height * 3
    rows = []
    attacks = []
    started = time.perf_counter()
    for index in range(args.expected_frames):
        source = np.frombuffer(
            _read_frame(source_decoder, frame_bytes, "source", index), dtype=np.uint8
        ).reshape(target_frame.output_height, target_frame.output_width, 3)
        candidate = np.frombuffer(
            _read_frame(candidate_decoder, frame_bytes, name, index), dtype=np.uint8
        ).reshape(target_frame.output_height, target_frame.output_width, 3)
        person, tracked_flower, arms, hands = _mapped_masks(
            np,
            index=index,
            mask_frame=mask_frame,
            target_frame=target_frame,
            packed_person=packed_person,
            person_meta=person_meta,
            packed_flower=packed_flower,
            flower_meta=flower_meta,
            packed_arms=packed_arms,
            arm_meta=arm_meta,
            packed_hands=packed_hands,
            hand_meta=hand_meta,
        )
        alpha, flower = _resolve_frame_masks(
            np,
            source_rgb=source,
            person=person,
            tracked_flower=tracked_flower,
            hands=hands,
            person_dilation=args.person_dilation,
            skin_negative_dilation=args.skin_negative_dilation,
            person_core_negative_erosion=args.person_core_negative_erosion,
            flower_mask_contract=args.flower_mask_contract,
        )
        metrics = frame_contract_metrics(
            np,
            candidate_rgb=candidate,
            source_rgb=source,
            robot_alpha=alpha,
            arm_support=arms,
            hand_support=hands,
            object_mask=flower,
            palette=palette,
            replacement_threshold=args.replacement_threshold,
            contact_radius=args.contact_radius,
        )
        in_persistent_interval = (
            args.persistent_grasp_start >= 0
            and args.persistent_grasp_start
            <= index
            < args.persistent_grasp_end_exclusive
        )
        if in_persistent_interval:
            grasp = occlusion_aware_grasp_metrics(
                np,
                candidate_rgb=candidate,
                source_rgb=source,
                hand_support=hands,
                # Persistent interaction state must survive occlusion.  The
                # resolved ``flower`` mask is correct for z-order/rendering
                # but deliberately removes source-person-core pixels; using
                # it here would erase the very stem segment that is held under
                # the hand.  The tracked object occupancy is the independent
                # state channel for this gate.
                object_mask=tracked_flower,
                replacement_threshold=args.replacement_threshold,
                contact_radius=args.contact_radius,
                maximum_source_occlusion_gap=args.maximum_source_occlusion_gap,
                minimum_bridge_coverage=args.minimum_occlusion_bridge_coverage,
            )
            # The adversary restores every tracked hand pixel to the source,
            # so the generated-hand support and bridge coverage are exactly
            # zero by construction.  Recomputing the same source mask distance
            # on an identical mask pair is unnecessary.
            grasp_erasure_attack_pass = False
        else:
            grasp = {
                "source_hand_object_gap_pixels": None,
                "source_hold_observable": False,
                "robot_direct_contact": False,
                "occlusion_corridor_pixels": 0,
                "occlusion_bridge_coverage": 0.0,
                "visual_grasp_pass": False,
            }
            grasp_erasure_attack_pass = False
        rows.append(
            {
                "frame": index,
                "seconds": index / args.fps,
                **metrics,
                **grasp,
                "grasp_erasure_attack_pass": grasp_erasure_attack_pass,
            }
        )
        if index % args.adversarial_stride == 0 or bool(metrics["contact_required"]):
            baseline, color, topology, contact, structure = _attack_metrics(
                np,
                candidate=candidate,
                source=source,
                alpha=alpha,
                arms=arms,
                hands=hands,
                flower=flower,
                palette=palette,
                args=args,
            )
            attacks.append(
                {
                    "frame": index,
                    "arm_attack_support_pixels": int(
                        np.count_nonzero(np.logical_and(arms, np.logical_not(flower)))
                    ),
                    "baseline": baseline,
                    "color_attack": color,
                    "topology_attack": topology,
                    "contact_attack": contact,
                    "structure_ghost_attack": structure,
                }
            )
    for label, process in (("source", source_decoder), (name, candidate_decoder)):
        if process.stdout is not None:
            process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        returncode = process.wait()
        if returncode:
            raise RuntimeError(f"{label} decoder failed with {returncode}: {stderr[-2000:]}")
    summary = _summary(np, rows, args, frozen_limits=frozen_limits)
    color_palette_delta = np.asarray(
        [
            row["color_attack"]["palette_surprisal"]
            - row["baseline"]["palette_surprisal"]
            for row in attacks
        ]
    )
    color_chroma_delta = np.asarray(
        [
            row["color_attack"]["high_chroma_fraction"]
            - row["baseline"]["high_chroma_fraction"]
            for row in attacks
        ]
    )
    topology_attacks = [
        row for row in attacks if row["arm_attack_support_pixels"] >= 16
    ]
    arm_drop = np.asarray(
        [
            row["baseline"]["arm_replacement_coverage"]
            - row["topology_attack"]["arm_replacement_coverage"]
            for row in topology_attacks
        ]
    )
    detachable = [
        row
        for row in attacks
        if row["baseline"]["contact_required"]
        and row["baseline"]["contact_observed"]
    ]
    contact_rejections = sum(
        not row["contact_attack"]["contact_observed"] for row in detachable
    )
    structure_delta = np.asarray(
        [
            abs(
                row["structure_ghost_attack"]["hand_edge_energy_lower_gate"]
                - row["baseline"]["hand_edge_energy_lower_gate"]
            )
            for row in attacks
        ]
    )
    structure_sobel_delta = np.asarray(
        [
            row["baseline"]["hand_structure_sobel_energy"]
            - row["structure_ghost_attack"]["hand_structure_sobel_energy"]
            for row in attacks
        ]
    )
    adversarial = {
        "sampled_frames": len(attacks),
        "color_attack_palette_delta_p05": float(np.quantile(color_palette_delta, 0.05)),
        "color_attack_high_chroma_delta_p05": float(np.quantile(color_chroma_delta, 0.05)),
        "topology_attack_sampled_frames": len(topology_attacks),
        "topology_attack_arm_coverage_drop_p05": (
            float(np.quantile(arm_drop, 0.05)) if len(arm_drop) else 0.0
        ),
        "baseline_contact_positive_frames": len(detachable),
        "contact_attack_rejected_frames": contact_rejections,
        "structure_ghost_attack_edge_delta_p05": float(
            np.quantile(structure_delta, 0.05)
        ),
        "structure_ghost_attack_sobel_delta_p05": float(
            np.quantile(structure_sobel_delta, 0.05)
        ),
        "gates": {
            "color_attack_detected": bool(
                np.quantile(color_palette_delta, 0.05) >= 0.35
                and np.quantile(color_chroma_delta, 0.05) >= 0.10
            ),
            "topology_attack_detected": bool(
                len(arm_drop) >= 3 and np.quantile(arm_drop, 0.05) >= 0.40
            ),
            "contact_attack_detected": bool(
                len(detachable) >= 3 and contact_rejections == len(detachable)
            ),
            "structure_ghost_attack_detected": bool(
                np.quantile(structure_sobel_delta, 0.05) >= 3.0
            ),
        },
    }
    adversarial["all_attacks_detected"] = all(adversarial["gates"].values())
    return {
        "name": name,
        "path": str(path),
        "sha256": _sha256(path),
        "wall_seconds": time.perf_counter() - started,
        "audit_fps": args.expected_frames / (time.perf_counter() - started),
        "summary": summary,
        "adversarial": adversarial,
        "frame_metrics": rows,
    }


def main() -> int:
    args = _parser().parse_args()
    import numpy as np

    if args.expected_frames <= 0 or args.fps <= 0:
        raise SystemExit("expected frames and fps must be positive")
    if not 0 <= args.allowed_late_violation_fraction <= 1:
        raise SystemExit("allowed violation fraction must be in [0, 1]")
    if not 0 <= args.required_contact_recall <= 1:
        raise SystemExit("required contact recall must be in [0, 1]")
    if not 0 <= args.required_persistent_grasp_recall <= 1:
        raise SystemExit("required persistent grasp recall must be in [0, 1]")
    persistent_disabled = (
        args.persistent_grasp_start < 0 and args.persistent_grasp_end_exclusive < 0
    )
    persistent_valid = (
        0 <= args.persistent_grasp_start < args.persistent_grasp_end_exclusive
        <= args.expected_frames
    )
    if not persistent_disabled and not persistent_valid:
        raise SystemExit("persistent grasp interval must be disabled or a valid half-open interval")
    if args.frozen_limits_candidate and not args.frozen_limits_report:
        raise SystemExit("--frozen-limits-candidate requires --frozen-limits-report")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates = _parse_candidates(args.candidate)
    mask_frame = SourceResizeCrop(
        name=args.mask_frame_name,
        source_width=args.mask_source_width,
        source_height=args.mask_source_height,
        scaled_width=args.mask_scaled_width,
        scaled_height=args.mask_scaled_height,
        crop_left=args.mask_crop_left,
        crop_top=args.mask_crop_top,
        output_width=int(np.load(args.person_masks, allow_pickle=False)["width"]),
        output_height=int(np.load(args.person_masks, allow_pickle=False)["height"]),
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
    contract = RobotLayerContract(
        camera_frame=target_frame.name,
        timeline=f"absolute_frame_index:full_source_{args.expected_frames}",
        width=target_frame.output_width,
        height=target_frame.output_height,
        contact_radius_pixels=args.contact_radius,
    )
    contract.validate()
    packed_person, person_meta = _load_packed(np, args.person_masks, "packed")
    packed_flower, flower_meta = _load_packed(np, args.flower_masks, "packed")
    packed_arms, arm_meta = _load_packed(np, args.limb_masks, "arms_packed")
    packed_hands, hand_meta = _load_packed(np, args.limb_masks, "hands_packed")
    packed_masks = (
        packed_person,
        person_meta,
        packed_flower,
        flower_meta,
        packed_arms,
        arm_meta,
        packed_hands,
        hand_meta,
    )
    reference = _decode_reference(np, args.ffmpeg, args.reference_image, target_frame)
    reference_person, reference_flower, _, reference_hands = _mapped_masks(
        np,
        index=args.reference_frame,
        mask_frame=mask_frame,
        target_frame=target_frame,
        packed_person=packed_person,
        person_meta=person_meta,
        packed_flower=packed_flower,
        flower_meta=flower_meta,
        packed_arms=packed_arms,
        arm_meta=arm_meta,
        packed_hands=packed_hands,
        hand_meta=hand_meta,
    )
    palette_mask = np.logical_and(reference_person, np.logical_not(reference_flower))
    palette = canonical_palette_histogram(np, reference, palette_mask, bins=8)
    frozen_limits: dict[str, float] | None = None
    frozen_limits_source: dict[str, Any] | None = None
    if args.frozen_limits_report is not None:
        frozen_path = args.frozen_limits_report.expanduser().resolve()
        frozen_report = json.loads(frozen_path.read_text())
        frozen_name = (
            args.frozen_limits_candidate
            or frozen_report.get("recommended_candidate")
        )
        frozen_candidate = next(
            (
                row
                for row in frozen_report.get("candidates", [])
                if row.get("name") == frozen_name
            ),
            None,
        )
        if frozen_candidate is None:
            raise ValueError(
                f"frozen report has no candidate {frozen_name!r}"
            )
        frozen_limits = {
            key: float(value)
            for key, value in frozen_candidate["summary"][
                "limits_fit_only_on_anchor_frames"
            ].items()
        }
        frozen_limits_source = {
            "path": str(frozen_path),
            "sha256": _sha256(frozen_path),
            "candidate": frozen_name,
            "limits": frozen_limits,
        }
    started = time.perf_counter()
    results = []
    for name, path in candidates:
        result = _audit_candidate(
            np,
            name=name,
            path=path,
            args=args,
            mask_frame=mask_frame,
            target_frame=target_frame,
            packed_masks=packed_masks,
            palette=palette,
            frozen_limits=frozen_limits,
        )
        frame_path = output_dir / f"{name}-frame-metrics.json"
        frame_path.write_text(
            json.dumps(result.pop("frame_metrics"), indent=2, sort_keys=True) + "\n"
        )
        result["frame_metrics_path"] = str(frame_path)
        results.append(result)
    ranking = sorted(
        results,
        key=lambda item: (
            -sum(item["summary"]["gates"].values()),
            -item["summary"]["sections"]["at_or_after_20_seconds"][
                "projected_contact_recall"
            ],
            item["summary"]["sections"]["at_or_after_20_seconds"]["metrics"][
                "palette_surprisal"
            ]["mean"],
        ),
    )
    packages = {}
    for package in ("numpy", "pillow"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "honest_scope": (
            "2D robot RGB-alpha-contact image-space audit; no depth, force, "
            "collision, or executable-robot claim"
        ),
        "contract": contract.to_dict(),
        "method": {
            "identity": "canonical reference RGB palette within tracked person support",
            "topology": "source-track-grounded replacement coverage on body, arms, and hands",
            "contact": "dilated 2D robot-hand support intersects visible source flower support",
            "persistent_grasp": (
                "tracked source-object occupancy survives source-hand occlusion; "
                "rendering and ordinary projected contact retain the resolved-visible mask"
            ),
            "threshold_fit": (
                (
                    "reused verbatim from immutable prior audit "
                    f"candidate {frozen_limits_source['candidate']}"
                    if frozen_limits_source is not None
                    else f"anchor [{args.anchor_start}, {args.anchor_end_exclusive}) only"
                )
                + "; late frames never fit their own thresholds"
            ),
        },
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "argv": sys.argv,
        },
        "inputs": {
            "source": {
                "path": str(args.source_video.resolve()),
                "sha256": _sha256(args.source_video),
            },
            "reference": {
                "path": str(args.reference_image.resolve()),
                "sha256": _sha256(args.reference_image),
                "frame": args.reference_frame,
            },
            "masks": [person_meta, flower_meta, arm_meta, hand_meta],
            "frozen_limits_source": frozen_limits_source,
        },
        "candidates": results,
        "ranking": [item["name"] for item in ranking],
        "recommended_candidate": ranking[0]["name"],
        "adversarial_audit_pass": all(
            item["adversarial"]["all_attacks_detected"] for item in results
        ),
        "wall_seconds": time.perf_counter() - started,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "seed": args.seed,
        "git": _git_state(),
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_only_audit": True,
        },
    }
    report_path = output_dir / "audit-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "recommended_candidate": report["recommended_candidate"],
                "adversarial_audit_pass": report["adversarial_audit_pass"],
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
