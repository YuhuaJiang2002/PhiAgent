"""Command-line interface for Cosmos 3 trajectory-conditioned rendering."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from phiagent.agent.verifier import VerificationReport
from phiagent.data.schema import RigidBodyTrajectory, RobotTrajectory
from phiagent.perception.camera import PinholeIntrinsics
from phiagent.physical_language.schema import PoseSE3
from phiagent.rendering.base import TrajectoryConditionedRenderRequest
from phiagent.rendering.cosmos3 import Cosmos3Config, Cosmos3TrajectoryRenderer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a physically accepted robot rollout through pinned Cosmos 3 Nano "
            "using a deterministic simulation video as edge control."
        )
    )
    parser.add_argument("--robot-trajectory", type=Path, required=True)
    parser.add_argument(
        "--object-trajectory",
        type=Path,
        action="append",
        required=True,
        dest="object_trajectories",
    )
    parser.add_argument(
        "--control-video",
        type=Path,
        required=True,
        help="simulation render aligned frame-for-frame with the verified trajectories",
    )
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--scene-asset", type=Path, action="append", required=True)
    parser.add_argument("--verification-record", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cosmos-repo",
        type=Path,
        default=os.environ.get("COSMOS3_REPO"),
        required="COSMOS3_REPO" not in os.environ,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=os.environ.get("COSMOS3_CHECKPOINT"),
        required="COSMOS3_CHECKPOINT" not in os.environ,
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/trajectory_render"),
    )
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-gib", type=float, default=60.0)
    parser.add_argument("--resolution", type=int, choices=(256, 480, 720), default=480)
    parser.add_argument("--fps", type=int, choices=(10, 16, 24, 30), default=30)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--no-guardrails", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _verification_report(payload: dict[str, Any]) -> VerificationReport:
    return VerificationReport(
        accepted=bool(payload["accepted"]),
        collision=dict(payload["collision"]),
        contact=dict(payload["contact"]),
        reachability=dict(payload["reachability"]),
        diagnoses=tuple(str(item) for item in payload.get("diagnoses", [])),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = Cosmos3Config(
        framework_repo=args.cosmos_repo.expanduser().resolve(),
        checkpoint_dir=args.checkpoint_dir.expanduser().resolve(),
        gpu_index=args.gpu,
        minimum_free_gpu_mib=round(args.minimum_free_gpu_gib * 1024),
        resolution=args.resolution,
        fps=args.fps,
        num_steps=args.steps,
        guardrails=not args.no_guardrails,
        hf_home=args.hf_home,
        offline=not args.online,
    )
    renderer = Cosmos3TrajectoryRenderer(config)
    if args.preflight_only:
        print(json.dumps(renderer.preflight(), indent=2, sort_keys=True))
        return 0

    camera = _read_json(args.camera)
    verification_payload = _read_json(args.verification_record)
    request = TrajectoryConditionedRenderRequest(
        robot_trajectory=RobotTrajectory.from_json(args.robot_trajectory),
        object_trajectories=tuple(
            RigidBodyTrajectory.from_json(path) for path in args.object_trajectories
        ),
        control_video=args.control_video,
        prompt=args.prompt,
        camera_intrinsics=PinholeIntrinsics.from_dict(camera["intrinsics"]),
        camera_T_robot_base=PoseSE3.from_dict(camera["camera_T_robot_base"]),
        scene_assets=tuple(args.scene_asset),
        verification=_verification_report(verification_payload),
        verification_record=args.verification_record,
        output=args.output,
        experiment_root=args.experiment_root,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    result = renderer.render(request)
    print(json.dumps({key: str(value) for key, value in asdict(result).items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
