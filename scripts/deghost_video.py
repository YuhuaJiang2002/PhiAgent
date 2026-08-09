#!/usr/bin/env python3
"""Apply mask-targeted temporal deghosting to a generated robot video."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.deghost import (  # noqa: E402
    MaskedDeghostConfig,
    ObjectGhostRepairConfig,
    deghost_video,
    repair_object_ghosts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--character-mask", type=Path, required=True)
    parser.add_argument("--object-mask", type=Path, required=True)
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=7.0)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.source_video is None:
        result = deghost_video(
            candidate=args.candidate,
            character_mask=args.character_mask,
            object_mask=args.object_mask,
            output=args.output,
            config=MaskedDeghostConfig(
                strength=args.strength,
                crf=args.crf,
                preset=args.preset,
            ),
            overwrite=args.overwrite,
        )
    else:
        result = repair_object_ghosts(
            candidate=args.candidate,
            source_video=args.source_video,
            character_mask=args.character_mask,
            object_prior_mask=args.object_mask,
            output=args.output,
            config=ObjectGhostRepairConfig(strength=args.strength),
            overwrite=args.overwrite,
        )
    print(json.dumps({key: str(value) for key, value in asdict(result).items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
