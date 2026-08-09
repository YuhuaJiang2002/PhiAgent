#!/usr/bin/env python3
"""Extract right-hand 3D observations with the pinned official HaMeR teacher."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.perception.hand.hamer import HamerConfig, HamerHandTracker  # noqa: E402
from phiagent.perception.schema import PerceptionSequence  # noqa: E402
from phiagent.physical_language.schema import FrameKind, FrameRef  # noqa: E402
from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hamer-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-mib", type=int, default=20000)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = HamerConfig(
        repository=args.hamer_repo,
        checkpoint=args.checkpoint,
        frame_stride=args.frame_stride,
    )
    tracker = HamerHandTracker(config)
    tracker.preflight()
    gpus, inventory, processes = query_gpus()
    gpu = select_gpu(gpus, args.minimum_free_mib, args.gpu)
    preflight = {
        "hostname": platform.node(),
        "gpu_physical_index": gpu.physical_index,
        "gpu_inventory": inventory,
        "gpu_processes": processes,
        "hamer_repository": str(args.hamer_repo.resolve()),
        "video": str(args.video.resolve()),
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite HaMeR output: {args.output}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu.physical_index)
    camera = FrameRef(FrameKind.CAMERA, args.camera_name)
    hands = tracker.track(args.video, camera)
    sequence = PerceptionSequence("0.1.0", hands, (None,) * len(hands))
    sequence.to_json(args.output)
    metadata = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata.write_text(
        json.dumps(
            {**preflight, "observations": len(hands)}, indent=2, sort_keys=True
        )
        + "\n"
    )
    print(
        json.dumps(
            {"observations": len(hands), "output": str(args.output)}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
