#!/usr/bin/env python3
"""Run a Wan-based agentic proxy for PhiZero hand-to-dexterous-hand transfer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.agent.visual_transfer import (  # noqa: E402
    AgenticProxyRequest,
    AgenticVisualTransferController,
    CommandProxyEvaluator,
    ProxyProposal,
    ProxyThresholds,
)
from phiagent.rendering.wan_animate import WanAnimateConfig, WanAnimateRenderer  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate Wan2.2-Animate proxy candidates. "
            "This does not execute or reproduce the unreleased PhiZero model."
        )
    )
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--reference-video", type=Path, required=True)
    parser.add_argument(
        "--target-image",
        type=Path,
        action="append",
        required=True,
        help="Sharpa first-frame candidate; repeat to create an ensemble",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="initial seed; repeat to create an ensemble (default: 42)",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "A Sharpa dexterous hand performs the exact source hand motion while "
            "preserving the object, contacts, camera, and scene."
        ),
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path("scripts/local_video_evaluator.py"),
        help="local executable that accepts the proxy evaluator CLI contract",
    )
    parser.add_argument(
        "--wan-repo",
        type=Path,
        default=os.environ.get("WAN22_REPO"),
        required="WAN22_REPO" not in os.environ,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=os.environ.get("WAN22_CHECKPOINT"),
        required="WAN22_CHECKPOINT" not in os.environ,
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/phizero-agentic-proxy"),
    )
    parser.add_argument("--maximum-rounds", type=int, default=3)
    parser.add_argument("--motion-threshold", type=float, default=0.75)
    parser.add_argument("--identity-threshold", type=float, default=0.80)
    parser.add_argument("--object-threshold", type=float, default=0.75)
    parser.add_argument("--temporal-threshold", type=float, default=0.75)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-gib", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-num", type=int, default=77)
    parser.add_argument("--infer-frames", type=int, default=80)
    parser.add_argument("--reference-frames", type=int, choices=(1, 5), default=1)
    parser.add_argument(
        "--object-roi",
        type=float,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        required=True,
        help="normalized first-frame ROI used for object tracking and mask auditing",
    )
    parser.add_argument(
        "--mode",
        choices=("replacement", "animation"),
        default="replacement",
        help=(
            "Wan replacement preserves source background/object pixels outside the "
            "character mask; animation regenerates the full frame"
        ),
    )
    parser.add_argument("--no-retarget", action="store_true")
    parser.add_argument("--use-flux", action="store_true")
    parser.add_argument("--no-relighting-lora", action="store_true")
    parser.add_argument(
        "--suppress-source-face-control",
        action="store_true",
        help=(
            "replace the cropped source-human face control with a deterministic "
            "black video in replacement mode; pose and replacement masks are retained"
        ),
    )
    parser.add_argument(
        "--t5-cpu",
        action="store_true",
        help="keep the T5 encoder on CPU to reduce peak GPU memory",
    )
    parser.add_argument("--no-offload", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.mode == "replacement" and args.use_flux:
        raise ValueError("--use-flux is only supported with --mode animation")
    evaluator = args.evaluator.expanduser().resolve()
    if not evaluator.is_file():
        raise ValueError(f"local proxy evaluator does not exist: {evaluator}")
    command = (
        str(evaluator),
        "--source",
        "{source}",
        "--reference",
        "{reference}",
        "--target-image",
        "{target_image}",
        "--candidate",
        "{candidate}",
        "--metadata",
        "{metadata}",
        "--object-roi",
        *(str(value) for value in args.object_roi),
    )
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
        mode=args.mode,
        retarget=args.mode == "animation" and not args.no_retarget,
        use_flux=args.use_flux,
        use_relighting_lora=not args.no_relighting_lora,
        suppress_source_face_control=args.suppress_source_face_control,
        offload_model=not args.no_offload,
        t5_cpu=args.t5_cpu,
        object_roi=tuple(args.object_roi),
    )
    renderer = WanAnimateRenderer(config)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "method": "agentic_proxy_not_official_phizero",
                    "evaluator": str(evaluator),
                    "wan": renderer.preflight(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    seeds = tuple(args.seed or (42,))
    proposals = tuple(
        ProxyProposal(
            backend="wan2.2-animate",
            target_image=image.expanduser().resolve(),
            prompt=args.prompt,
            seed=seed,
        )
        for image in args.target_image
        for seed in seeds
    )
    controller = AgenticVisualTransferController(
        {"wan2.2-animate": renderer},
        CommandProxyEvaluator(command),
    )
    outcome = controller.run(
        AgenticProxyRequest(
            source_video=args.source_video.expanduser().resolve(),
            reference_video=args.reference_video.expanduser().resolve(),
            initial_proposals=proposals,
            experiment_root=args.experiment_root.expanduser().resolve(),
            thresholds=ProxyThresholds(
                motion_preservation=args.motion_threshold,
                target_identity=args.identity_threshold,
                object_consistency=args.object_threshold,
                temporal_consistency=args.temporal_threshold,
            ),
            maximum_rounds=args.maximum_rounds,
        )
    )
    print(
        json.dumps(
            {
                "accepted": outcome.accepted,
                "experiment_dir": str(outcome.experiment_dir),
                "best_output": str(outcome.best_candidate.result.output),
                "best_score": outcome.best_candidate.scorecard.mean_score,
                "trace": str(outcome.trace_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if outcome.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
