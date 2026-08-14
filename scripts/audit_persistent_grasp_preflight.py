#!/usr/bin/env python3
"""Run the dense grasp gate before the expensive appearance/attack audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.object_factored_long_video import (
    SourceResizeCrop,
    remap_boolean_mask,
)
from phiagent.rendering.robot_layer_contract import (
    occlusion_aware_grasp_metrics,
)
from scripts.audit_robot_layer_long_video import (
    _decoder_command,
    _load_packed,
    _read_frame,
    _sha256,
    _unpack,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--limb-masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--start", type=int, default=497)
    parser.add_argument("--end-exclusive", type=int, default=644)
    parser.add_argument("--replacement-threshold", type=float, default=12.0)
    parser.add_argument("--contact-radius", type=int, default=3)
    parser.add_argument("--maximum-source-occlusion-gap", type=int, default=24)
    parser.add_argument("--minimum-bridge-coverage", type=float, default=0.80)
    parser.add_argument("--required-recall", type=float, default=1.0)
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


def main() -> int:
    args = _parser().parse_args()
    import numpy as np

    if not 0 <= args.start < args.end_exclusive <= args.expected_frames:
        raise ValueError("persistent interval must be inside the declared timeline")
    if not 0.0 <= args.required_recall <= 1.0:
        raise ValueError("required recall must be in [0, 1]")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    mask_frame = SourceResizeCrop(
        name=args.mask_frame_name,
        source_width=args.mask_source_width,
        source_height=args.mask_source_height,
        scaled_width=args.mask_scaled_width,
        scaled_height=args.mask_scaled_height,
        crop_left=args.mask_crop_left,
        crop_top=args.mask_crop_top,
        output_width=int(np.load(args.flower_masks, allow_pickle=False)["width"]),
        output_height=int(np.load(args.flower_masks, allow_pickle=False)["height"]),
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
    packed_flower, flower_meta = _load_packed(np, args.flower_masks, "packed")
    packed_hands, hand_meta = _load_packed(np, args.limb_masks, "hands_packed")
    for name, metadata in (("flower", flower_meta), ("hands", hand_meta)):
        if metadata["frames"] != args.expected_frames:
            raise ValueError(f"{name} masks do not cover the declared timeline")

    source_path = args.source_video.expanduser().resolve()
    candidate_path = args.candidate_video.expanduser().resolve()
    source_decoder = subprocess.Popen(
        _decoder_command(args.ffmpeg, source_path, source=True, target_frame=target_frame),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    candidate_decoder = subprocess.Popen(
        _decoder_command(
            args.ffmpeg, candidate_path, source=False, target_frame=target_frame
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    frame_bytes = args.target_width * args.target_height * 3
    rows = []
    started = time.perf_counter()
    for index in range(args.expected_frames):
        source = np.frombuffer(
            _read_frame(source_decoder, frame_bytes, "source", index), dtype=np.uint8
        ).reshape(args.target_height, args.target_width, 3)
        candidate = np.frombuffer(
            _read_frame(candidate_decoder, frame_bytes, "candidate", index),
            dtype=np.uint8,
        ).reshape(args.target_height, args.target_width, 3)
        if not args.start <= index < args.end_exclusive:
            continue
        flower = remap_boolean_mask(
            np,
            _unpack(np, packed_flower, flower_meta, index),
            source_frame=mask_frame,
            target_frame=target_frame,
        )
        hands = remap_boolean_mask(
            np,
            _unpack(np, packed_hands, hand_meta, index),
            source_frame=mask_frame,
            target_frame=target_frame,
        )
        rows.append(
            {
                "frame": index,
                **occlusion_aware_grasp_metrics(
                    np,
                    candidate_rgb=candidate,
                    source_rgb=source,
                    hand_support=hands,
                    object_mask=flower,
                    replacement_threshold=args.replacement_threshold,
                    contact_radius=args.contact_radius,
                    maximum_source_occlusion_gap=args.maximum_source_occlusion_gap,
                    minimum_bridge_coverage=args.minimum_bridge_coverage,
                ),
            }
        )
    for decoder, name in ((source_decoder, "source"), (candidate_decoder, "candidate")):
        stderr = decoder.communicate()[1].decode(errors="replace")
        if decoder.returncode:
            raise RuntimeError(f"{name} decoder failed: {stderr}")
    passed = sum(bool(row["visual_grasp_pass"]) for row in rows)
    recall = passed / max(1, len(rows))
    failures = [int(row["frame"]) for row in rows if not row["visual_grasp_pass"]]
    report = {
        "schema_version": "1.0.0",
        "status": "WORKING" if recall >= args.required_recall else "PARTIAL",
        "scope": "dense 2-D persistent visual grasp preflight",
        "physical_evidence": False,
        "thresholds_frozen": True,
        "inputs": {
            "source": {"path": str(source_path), "sha256": _sha256(source_path)},
            "candidate": {
                "path": str(candidate_path),
                "sha256": _sha256(candidate_path),
            },
            "flower_masks": {
                "path": str(args.flower_masks.resolve()),
                "sha256": _sha256(args.flower_masks.resolve()),
            },
            "limb_masks": {
                "path": str(args.limb_masks.resolve()),
                "sha256": _sha256(args.limb_masks.resolve()),
            },
        },
        "interval": [args.start, args.end_exclusive],
        "frames": len(rows),
        "visual_grasp_passes": passed,
        "visual_grasp_recall": recall,
        "required_recall": args.required_recall,
        "failed_frames": failures,
        "automatic_pass": recall >= args.required_recall,
        "wall_seconds": time.perf_counter() - started,
        "rows": rows,
        "limitations": [
            "This camera-frame invariant is not metric depth or force closure.",
            "The full image-space, temporal, adversarial, and native-resolution audits remain mandatory.",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: report[key] for key in ("automatic_pass", "visual_grasp_recall", "failed_frames", "wall_seconds")}, indent=2))
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
