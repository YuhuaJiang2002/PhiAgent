#!/usr/bin/env python3
"""Convert aligned hand/object teacher observations into continuous EPL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.perception.camera import PinholeIntrinsics  # noqa: E402
from phiagent.perception.extractor import (  # noqa: E402
    PhysicalStateExtractor,
    PhysicalStateExtractorConfig,
)
from phiagent.perception.schema import PerceptionSequence  # noqa: E402
from phiagent.physical_language.tokenizer import EPLTokenizer  # noqa: E402
from phiagent.physical_language.visualization import EPLVisualizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build EPL from explicit 3D teacher observations. This script does not "
            "invent missing HaMeR/FoundationPose outputs."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--visualization", type=Path)
    parser.add_argument(
        "--camera-intrinsics",
        type=Path,
        help="JSON containing fx, fy, cx, cy, width, and height",
    )
    parser.add_argument("--contact-distance-m", type=float, default=0.045)
    parser.add_argument("--moving-distance-m", type=float, default=0.005)
    args = parser.parse_args()
    if not args.video.is_file():
        raise SystemExit(f"video does not exist: {args.video}")
    observations = PerceptionSequence.from_json(args.observations)
    extractor = PhysicalStateExtractor(
        PhysicalStateExtractorConfig(
            contact_distance_m=args.contact_distance_m,
            moving_distance_m=args.moving_distance_m,
        )
    )
    epl = extractor.extract(observations, str(args.video.resolve()))
    epl.to_json(args.output)
    if args.tokens:
        args.tokens.parent.mkdir(parents=True, exist_ok=True)
        tokenizer = EPLTokenizer()
        args.tokens.write_text(
            json.dumps(
                [tokenizer.encode_chunk(chunk) for chunk in epl.chunks],
                indent=2,
            )
            + "\n"
        )
    visualization = None
    if args.visualization:
        if args.camera_intrinsics is None:
            parser.error("--visualization requires --camera-intrinsics")
        intrinsics = PinholeIntrinsics.from_dict(
            json.loads(args.camera_intrinsics.read_text())
        )
        visualization = EPLVisualizer().render(
            args.video,
            observations,
            epl,
            intrinsics,
            args.visualization,
        )
    print(
        json.dumps(
            {
                "chunks": len(epl.chunks),
                "output": str(args.output),
                "visualization": visualization,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
