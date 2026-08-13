#!/usr/bin/env python3
"""Evaluate motion/identity/temporal proxies when object tracking is unresolved."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.evaluation.video_proxy import (  # noqa: E402
    evaluate_local_core_videos,
    file_sha256,
    resolve_ffmpeg,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument("--maximum-seconds", type=float, default=4.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = (
        args.source,
        args.reference,
        args.target_image,
        args.candidate,
        args.metadata,
    )
    if any(not path.is_file() for path in paths):
        raise ValueError("one or more core evaluator inputs are missing")
    evidence = args.evidence.expanduser().resolve()
    if evidence.exists():
        raise FileExistsError(f"core evaluation evidence already exists: {evidence}")
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    metrics = evaluate_local_core_videos(
        source=args.source.resolve(),
        reference=args.reference.resolve(),
        target_image=args.target_image.resolve(),
        candidate=args.candidate.resolve(),
        ffmpeg=ffmpeg,
        width=args.width,
        height=args.height,
        sample_fps=args.sample_fps,
        maximum_seconds=args.maximum_seconds,
    )
    payload = {
        "schema_version": "1.0.0",
        "evaluator": "phiagent-local-video-core-evaluator-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            label: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for label, path in zip(
                ("source", "reference", "target_image", "candidate", "metadata"),
                paths,
            )
        },
        "decoder": {
            "ffmpeg": str(ffmpeg),
            "width": args.width,
            "height": args.height,
            "sample_fps": args.sample_fps,
            "maximum_seconds": args.maximum_seconds,
        },
        "metrics": metrics,
        "object_gate": "UNRESOLVED",
        "accepted": False,
        "claim_boundary": (
            "Core proxy evidence cannot substitute for object consistency. The candidate "
            "is rejected until a geometry/segmentation-based object evaluator passes."
        ),
    }
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

