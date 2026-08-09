#!/usr/bin/env python3
"""Deterministic local evaluator for agentic visual-transfer candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.evaluation.video_proxy import (  # noqa: E402
    evaluate_local_videos,
    resolve_ffmpeg,
    write_evaluation_evidence,
)
from phiagent.evaluation.object_instance import NormalizedROI  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument("--maximum-seconds", type=float, default=4.0)
    parser.add_argument(
        "--object-roi",
        type=float,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        required=True,
        help="normalized first-frame ROI containing only the manipulated object",
    )
    parser.add_argument("--object-width", type=int, default=224)
    parser.add_argument("--object-height", type=int, default=128)
    return parser


def main() -> int:
    args = _parser().parse_args()
    for label, path in (
        ("source", args.source),
        ("reference", args.reference),
        ("target image", args.target_image),
        ("candidate", args.candidate),
        ("backend metadata", args.metadata),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    object_roi = NormalizedROI(*args.object_roi)
    metrics = evaluate_local_videos(
        source=args.source.resolve(),
        reference=args.reference.resolve(),
        target_image=args.target_image.resolve(),
        candidate=args.candidate.resolve(),
        ffmpeg=ffmpeg,
        width=args.width,
        height=args.height,
        sample_fps=args.sample_fps,
        maximum_seconds=args.maximum_seconds,
        object_roi=object_roi,
        object_width=args.object_width,
        object_height=args.object_height,
    )
    evidence = (
        args.evidence.expanduser().resolve()
        if args.evidence is not None
        else args.candidate.with_suffix(".evaluation.json").resolve()
    )
    write_evaluation_evidence(
        evidence,
        source=args.source,
        reference=args.reference,
        target_image=args.target_image,
        candidate=args.candidate,
        backend_metadata=args.metadata,
        ffmpeg=ffmpeg,
        metrics=metrics,
        width=args.width,
        height=args.height,
        sample_fps=args.sample_fps,
        maximum_seconds=args.maximum_seconds,
    )
    diagnoses = []
    for field, value in metrics.scorecard().items():
        if value < 0.75:
            diagnoses.append(f"{field} proxy score is below 0.75")
    for field in (
        "object_contour_similarity",
        "object_color_similarity",
        "object_temporal_deformation",
        "object_tracking_coverage",
        "object_trajectory_similarity",
        "object_lift_recall",
    ):
        value = getattr(metrics, field)
        if value < 0.75:
            diagnoses.append(f"{field} is below 0.75")
    print(
        json.dumps(
            {
                "evaluator": "phiagent-local-video-evaluator-v4-object-trajectory",
                **metrics.scorecard(),
                "diagnoses": diagnoses,
                "evidence": str(evidence),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
