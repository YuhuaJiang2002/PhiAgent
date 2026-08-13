#!/usr/bin/env python3
"""Restore bounded current-frame hand detail against a frozen audit gate."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from phiagent.rendering.object_factored_long_video import (  # noqa: E402
    SourceResizeCrop,
    binary_dilate_square,
    binary_erode_square,
    remap_boolean_mask,
    resolve_flower_visibility,
    source_skin_like,
    strict_flower_seed,
)
from phiagent.rendering.robot_layer_contract import (  # noqa: E402
    project_hand_detail_to_gate,
    replacement_mask,
)
from stabilize_joyai_appearance_state import (  # noqa: E402
    _finish,
    _git_state,
    _packages,
    _probe,
    _sha256,
    _writer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--person-masks", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--limb-masks", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--frame-metrics", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--late-start", type=int, default=480)
    parser.add_argument("--repair-pre-late-failures", action="store_true")
    parser.add_argument(
        "--extra-frame",
        type=int,
        action="append",
        default=[],
        help=(
            "Additional absolute frame exposed by an encoded-output audit; "
            "may be repeated and is unioned with frozen-gate failures."
        ),
    )
    parser.add_argument("--replacement-threshold", type=float, default=12.0)
    parser.add_argument("--person-dilation", type=int, default=10)
    parser.add_argument("--skin-negative-dilation", type=int, default=2)
    parser.add_argument("--person-core-negative-erosion", type=int, default=2)
    parser.add_argument("--contact-lock-radius", type=int, default=3)
    parser.add_argument("--editable-erosion-pixels", type=int, default=3)
    parser.add_argument("--gaussian-sigma", type=float, default=0.8)
    parser.add_argument("--maximum-strength", type=float, default=3.0)
    parser.add_argument(
        "--preencode-edge-margin",
        type=float,
        default=0.0,
        help=(
            "Non-negative robustness margin above the unchanged frozen gate; "
            "used to survive measured lossy delivery encoding."
        ),
    )
    parser.add_argument("--search-iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260814)
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


def _load_mask(np: Any, path: Path, key: str) -> tuple[Any, dict[str, Any]]:
    payload = np.load(path, allow_pickle=False)
    if key not in payload.files:
        raise ValueError(f"{path} has no mask key {key!r}")
    metadata = {
        "height": int(payload["height"]),
        "width": int(payload["width"]),
        "bitorder": str(payload["bitorder"]),
    }
    return payload[key], metadata


def _unpack(np: Any, payload: tuple[Any, dict[str, Any]], index: int) -> Any:
    packed, metadata = payload
    values = np.unpackbits(packed[index], bitorder=metadata["bitorder"])
    size = metadata["height"] * metadata["width"]
    return values[:size].reshape(metadata["height"], metadata["width"]).astype(bool)


def _mapped(
    np: Any,
    payload: tuple[Any, dict[str, Any]],
    index: int,
    *,
    mask_frame: SourceResizeCrop,
    target_frame: SourceResizeCrop,
) -> Any:
    return remap_boolean_mask(
        np,
        _unpack(np, payload, index),
        source_frame=mask_frame,
        target_frame=target_frame,
    )


def main() -> int:
    args = _parser().parse_args()
    import cv2
    import numpy as np

    paths = {
        "candidate": args.candidate_video.expanduser().resolve(),
        "source": args.source_video.expanduser().resolve(),
        "person_masks": args.person_masks.expanduser().resolve(),
        "flower_masks": args.flower_masks.expanduser().resolve(),
        "limb_masks": args.limb_masks.expanduser().resolve(),
        "audit_report": args.audit_report.expanduser().resolve(),
        "frame_metrics": args.frame_metrics.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{name}: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    (output / "review").mkdir()

    audit = json.loads(paths["audit_report"].read_text())
    candidate_audit = next(
        (
            row
            for row in audit["candidates"]
            if row["name"] == args.candidate_name
        ),
        None,
    )
    if candidate_audit is None:
        raise ValueError(f"audit has no candidate {args.candidate_name!r}")
    frozen_limit = float(
        candidate_audit["summary"]["limits_fit_only_on_anchor_frames"][
            "hand_edge_energy_lower_gate"
        ]
    )
    if args.preencode_edge_margin < 0:
        raise ValueError("pre-encode edge margin must be non-negative")
    projection_target = frozen_limit + args.preencode_edge_margin
    rows = json.loads(paths["frame_metrics"].read_text())
    failing = {
        int(row["frame"])
        for row in rows
        if float(row["hand_edge_energy_lower_gate"]) < frozen_limit
        and (
            args.repair_pre_late_failures
            or int(row["frame"]) >= args.late_start
        )
    }
    extras = {int(frame) for frame in args.extra_frame}
    if any(frame < 0 or frame >= args.expected_frames for frame in extras):
        raise ValueError("extra frame lies outside the declared timeline")
    failing.update(extras)
    if not failing:
        raise ValueError("the supplied frozen audit has no selected hand-detail failures")

    mask_frame = SourceResizeCrop(
        args.mask_frame_name,
        args.mask_source_width,
        args.mask_source_height,
        args.mask_scaled_width,
        args.mask_scaled_height,
        args.mask_crop_left,
        args.mask_crop_top,
        int(np.load(paths["person_masks"], allow_pickle=False)["width"]),
        int(np.load(paths["person_masks"], allow_pickle=False)["height"]),
    )
    target_frame = SourceResizeCrop(
        args.target_frame_name,
        args.mask_source_width,
        args.mask_source_height,
        args.target_scaled_width,
        args.target_scaled_height,
        args.target_crop_left,
        args.target_crop_top,
        args.target_width,
        args.target_height,
    )
    mask_frame.validate()
    target_frame.validate()
    if (args.width, args.height) != (
        target_frame.output_width,
        target_frame.output_height,
    ):
        raise ValueError("video dimensions and named target camera frame disagree")

    probes = {name: _probe(cv2, paths[name]) for name in ("candidate", "source")}
    for name, probe in probes.items():
        if (
            probe["frames"] != args.expected_frames
            or probe["width"] != args.width
            or probe["height"] != args.height
            or abs(float(probe["fps"]) - args.fps) > 0.01
        ):
            raise ValueError(f"{name} violates the timeline contract: {probe}")

    person_payload = _load_mask(np, paths["person_masks"], "packed")
    flower_payload = _load_mask(np, paths["flower_masks"], "packed")
    hand_payload = _load_mask(np, paths["limb_masks"], "hands_packed")
    for name, payload in (
        ("person", person_payload),
        ("flower", flower_payload),
        ("hands", hand_payload),
    ):
        if len(payload[0]) != args.expected_frames:
            raise ValueError(f"{name} mask does not cover the complete timeline")

    lossless = output / "hand-detail-projected-lossless.mkv"
    review = output / "hand-detail-projected-27p5s.mp4"
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
    if args.editable_erosion_pixels < 1 or args.editable_erosion_pixels % 2 == 0:
        raise ValueError("editable erosion must be a positive odd kernel size")
    erosion_radius = (args.editable_erosion_pixels - 1) // 2
    per_frame = []
    edit_masks: dict[int, Any] = {}
    review_rows = []
    started = time.perf_counter()
    for index in range(args.expected_frames):
        candidate_ok, candidate_bgr = candidate_capture.read()
        source_ok, source_bgr = source_capture.read()
        if not candidate_ok or not source_ok:
            raise RuntimeError(f"video decode stopped at frame {index}")
        candidate = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2RGB)
        source = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        result = candidate.copy()
        record: dict[str, Any] = {"frame": index, "selected": index in failing}
        if index in failing:
            person = _mapped(
                np, person_payload, index, mask_frame=mask_frame,
                target_frame=target_frame,
            )
            tracked_flower = _mapped(
                np, flower_payload, index, mask_frame=mask_frame,
                target_frame=target_frame,
            )
            hands = _mapped(
                np, hand_payload, index, mask_frame=mask_frame,
                target_frame=target_frame,
            )
            support = binary_dilate_square(np, person, args.person_dilation)
            skin = source_skin_like(np, source) & (person | hands) & support
            skin = binary_dilate_square(
                np, skin, args.skin_negative_dilation
            )
            flower = resolve_flower_visibility(
                np,
                candidates=tracked_flower | strict_flower_seed(np, source),
                edit_support=support,
                source_person=person,
                source_skin_negative=skin,
                person_core_erosion=args.person_core_negative_erosion,
            )
            measurement = np.logical_and.reduce(
                (hands, support, np.logical_not(flower))
            )
            generated_hand = np.logical_and(measurement, replacement_mask(
                np,
                candidate,
                source,
                threshold=args.replacement_threshold,
            ))
            object_lock = binary_dilate_square(
                np, flower, args.contact_lock_radius
            )
            locked_generated_hand = np.logical_and(generated_hand, object_lock)
            editable = np.logical_and(
                generated_hand, np.logical_not(object_lock)
            )
            if int(np.count_nonzero(generated_hand)) != (
                int(np.count_nonzero(locked_generated_hand))
                + int(np.count_nonzero(editable))
            ):
                raise RuntimeError("generated-hand lock partition is not exhaustive")
            editable_pre_erosion = editable.copy()
            editable = binary_erode_square(np, editable, erosion_radius)
            before = candidate.copy()
            result, strength, achieved, passed = project_hand_detail_to_gate(
                cv2,
                np,
                frame_rgb=candidate,
                editable_mask=editable,
                measurement_mask=measurement,
                minimum_edge_energy=projection_target,
                gaussian_sigma=args.gaussian_sigma,
                maximum_strength=args.maximum_strength,
                search_iterations=args.search_iterations,
            )
            edit_masks[index] = editable
            record.update(
                {
                    "measurement_pixels": int(np.count_nonzero(measurement)),
                    "generated_hand_pixels": int(np.count_nonzero(generated_hand)),
                    "generated_hand_object_lock_overlap_pixels": int(
                        np.count_nonzero(locked_generated_hand)
                    ),
                    "editable_pixels_pre_erosion": int(
                        np.count_nonzero(editable_pre_erosion)
                    ),
                    "editable_pixels": int(np.count_nonzero(editable)),
                    "selected_strength": strength,
                    "achieved_edge_energy_preencode": achieved,
                    "frozen_edge_energy_limit": frozen_limit,
                    "preencode_edge_projection_target": projection_target,
                    "gate_pass_preencode": passed,
                    "changed_pixels": int(
                        np.count_nonzero(np.any(result != candidate, axis=2))
                    ),
                }
            )
            review_rows.append((index, before, result))
        per_frame.append(record)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        lossless_writer.stdin.write(result_bgr.tobytes())
        review_writer.stdin.write(result_bgr.tobytes())
    candidate_capture.release()
    source_capture.release()
    lossless_log = _finish(lossless_writer, "lossless")
    review_log = _finish(review_writer, "review")

    decoded = cv2.VideoCapture(str(lossless))
    original = cv2.VideoCapture(str(paths["candidate"]))
    immutable_exact = immutable_total = selected_changed = 0
    for index in range(args.expected_frames):
        decoded_ok, decoded_frame = decoded.read()
        original_ok, original_frame = original.read()
        if not decoded_ok or not original_ok:
            raise RuntimeError(f"postdecode verification stopped at frame {index}")
        exact = np.all(decoded_frame == original_frame, axis=2)
        editable = edit_masks.get(index)
        if editable is None:
            immutable_exact += int(np.count_nonzero(exact))
            immutable_total += exact.size
        else:
            immutable_exact += int(np.count_nonzero(exact[~editable]))
            immutable_total += int(np.count_nonzero(~editable))
            selected_changed += int(np.count_nonzero(~exact[editable]))
    decoded.release()
    original.release()
    wall_seconds = time.perf_counter() - started

    active_rows = [row for row in per_frame if row["selected"]]
    all_preencode = all(bool(row["gate_pass_preencode"]) for row in active_rows)
    metrics = {
        "frames": args.expected_frames,
        "video_seconds": args.expected_frames / args.fps,
        "selected_failure_frames": len(active_rows),
        "selected_frames_reaching_frozen_gate_preencode": sum(
            bool(row["gate_pass_preencode"]) for row in active_rows
        ),
        "changed_pixels_postdecode_on_selected_support": selected_changed,
        "postdecode_outside_editable_exact_fraction": (
            immutable_exact / immutable_total
        ),
        "wall_seconds": wall_seconds,
        "processing_fps": args.expected_frames / wall_seconds,
        "realtime_factor": wall_seconds / (args.expected_frames / args.fps),
        "maximum_selected_strength": max(
            float(row["selected_strength"]) for row in active_rows
        ),
    }
    gates = {
        "all_selected_frames_reach_frozen_gate_preencode": all_preencode,
        "outside_editable_lossless_exact": (
            metrics["postdecode_outside_editable_exact_fraction"] == 1.0
        ),
        "selected_support_changed": selected_changed > 0,
        "full_timeline_decodes": _probe(cv2, lossless)["frames"]
        == args.expected_frames,
    }
    frame_metrics = output / "frame-metrics.json"
    frame_metrics.write_text(json.dumps(per_frame, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "decision": "AWAITING_FULL_ENCODED_ADVERSARIAL_AUDIT",
        "method": "bounded_current_frame_high_frequency_gate_projection",
        "scope": "2D hand-detail restoration; no metric depth or force claim",
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "seed": args.seed,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
            if name != "ffmpeg"
        },
        "coordinate_frames": {
            "mask": mask_frame.to_dict(),
            "target": target_frame.to_dict(),
            "timeline": "absolute_frame_index:full_source_660",
        },
        "frozen_gate": {
            "source_audit_candidate": args.candidate_name,
            "hand_edge_energy_lower_gate": frozen_limit,
            "preencode_edge_projection_target": projection_target,
            "preencode_edge_margin": args.preencode_edge_margin,
            "selected_failed_frames": sorted(failing),
        },
        "config": {
            "repair_pre_late_failures": args.repair_pre_late_failures,
            "extra_frames": sorted(extras),
            "late_start": args.late_start,
            "replacement_threshold": args.replacement_threshold,
            "contact_lock_radius": args.contact_lock_radius,
            "editable_erosion_pixels": args.editable_erosion_pixels,
            "gaussian_sigma": args.gaussian_sigma,
            "maximum_strength": args.maximum_strength,
            "preencode_edge_margin": args.preencode_edge_margin,
            "search_iterations": args.search_iterations,
        },
        "metrics": metrics,
        "gates": gates,
        "outputs": {
            "lossless": {"path": str(lossless), "sha256": _sha256(lossless)},
            "review": {"path": str(review), "sha256": _sha256(review)},
            "frame_metrics": {"path": str(frame_metrics), "sha256": _sha256(frame_metrics)},
        },
        "runtime": {
            "python": sys.version,
            "packages": _packages(),
            "git": _git_state(),
            "encoder_logs": {"lossless": lossless_log, "review": review_log},
        },
        "limitations": [
            "The repair restores current-frame image detail only; it is not new 3D hand geometry.",
            "A full encoded long-video audit and native review remain mandatory.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"outputs": manifest["outputs"], "metrics": metrics, "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
