"""Command-line interface for visual motion transfer."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from phiagent.rendering.base import VisualTransferRequest
from phiagent.rendering.wan_animate import WanAnimateConfig, WanAnimateRenderer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer a human driving video's motion to a robot reference image "
            "with Wan2.2-Animate."
        )
    )
    parser.add_argument("--video", type=Path, required=True, help="human driving video")
    parser.add_argument("--robot-image", type=Path, required=True, help="robot reference image")
    parser.add_argument("--prompt", default="A robot performs the demonstrated manipulation.")
    parser.add_argument("--output", type=Path, required=True, help="destination .mp4")
    parser.add_argument(
        "--wan-repo",
        type=Path,
        default=os.environ.get("WAN22_REPO"),
        required="WAN22_REPO" not in os.environ,
        help="pinned Wan2.2 checkout (or WAN22_REPO)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=os.environ.get("WAN22_CHECKPOINT"),
        required="WAN22_CHECKPOINT" not in os.environ,
        help="Wan2.2-Animate-14B checkpoint (or WAN22_CHECKPOINT)",
    )
    parser.add_argument(
        "--experiment-root", type=Path, default=Path("outputs/visual_transfer")
    )
    parser.add_argument(
        "--gpu", type=int, help="physical GPU index; default selects the freest eligible GPU"
    )
    parser.add_argument("--minimum-free-gpu-gib", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-num", type=int, default=77)
    parser.add_argument("--infer-frames", type=int, default=80)
    parser.add_argument("--reference-frames", type=int, choices=(1, 5), default=1)
    parser.add_argument("--no-retarget", action="store_true")
    parser.add_argument("--use-flux", action="store_true")
    parser.add_argument("--no-offload", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate the pinned code/checkpoint and select a GPU",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = WanAnimateConfig(
        wan_repo=args.wan_repo.expanduser().resolve(),
        checkpoint_dir=args.checkpoint_dir.expanduser().resolve(),
        gpu_index=args.gpu,
        minimum_free_gpu_mib=round(args.minimum_free_gpu_gib * 1024),
        resolution_width=args.width,
        resolution_height=args.height,
        fps=args.fps,
        frame_num=args.frame_num,
        infer_frames=args.infer_frames,
        reference_frames=args.reference_frames,
        retarget=not args.no_retarget,
        use_flux=args.use_flux,
        offload_model=not args.no_offload,
    )
    renderer = WanAnimateRenderer(config)
    if args.preflight_only:
        print(json.dumps(renderer.preflight(), indent=2, sort_keys=True))
        return 0
    result = renderer.render(
        VisualTransferRequest(
            video=args.video,
            robot_image=args.robot_image,
            prompt=args.prompt,
            output=args.output,
            experiment_root=args.experiment_root,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps({key: str(value) for key, value in asdict(result).items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
