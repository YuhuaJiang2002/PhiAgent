#!/usr/bin/env python3
"""Run the measured teacher-observation -> EPL -> robot -> MuJoCo pipeline."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.perception.camera import PinholeIntrinsics  # noqa: E402
from phiagent.perception.extractor import PhysicalStateExtractor  # noqa: E402
from phiagent.perception.schema import PerceptionSequence  # noqa: E402
from phiagent.physical_language.visualization import EPLVisualizer  # noqa: E402
from phiagent.retargeting.base import (  # noqa: E402
    LinearEPLRetargeter,
    LinearRetargetingConfig,
)
from phiagent.simulation.base import ObjectPositionGoal, SimulationRequest  # noqa: E402
from phiagent.simulation.mujoco_backend import MujocoBackend  # noqa: E402


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run human-to-sim from explicit teacher observations. The script refuses "
            "to infer missing HaMeR/FoundationPose outputs."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--retarget-config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-body", action="append", default=[])
    parser.add_argument(
        "--required-contact", action="append", default=[], metavar="GEOM_A,GEOM_B"
    )
    parser.add_argument(
        "--forbidden-contact", action="append", default=[], metavar="GEOM_A,GEOM_B"
    )
    parser.add_argument(
        "--object-goal",
        action="append",
        default=[],
        metavar="BODY,X,Y,Z,TOLERANCE_M",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--camera-intrinsics", type=Path)
    parser.add_argument("--epl-visualization", action="store_true")
    args = parser.parse_args()
    for label, path in (
        ("video", args.video),
        ("observations", args.observations),
        ("retarget config", args.retarget_config),
        ("model", args.model),
    ):
        if not path.is_file():
            raise SystemExit(f"{label} does not exist: {path}")
    if args.epl_visualization and args.camera_intrinsics is None:
        parser.error("--epl-visualization requires --camera-intrinsics")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty experiment: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def contact_pairs(values: list[str]) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        for value in values:
            parts = tuple(part.strip() for part in value.split(","))
            if len(parts) != 2:
                parser.error(f"invalid contact pair {value!r}; expected GEOM_A,GEOM_B")
            pairs.append(parts)  # type: ignore[arg-type]
        return tuple(pairs)

    goals: list[ObjectPositionGoal] = []
    for value in args.object_goal:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 5:
            parser.error(
                f"invalid object goal {value!r}; expected BODY,X,Y,Z,TOLERANCE_M"
            )
        try:
            goals.append(
                ObjectPositionGoal(
                    parts[0],
                    (float(parts[1]), float(parts[2]), float(parts[3])),
                    float(parts[4]),
                )
            )
        except ValueError as exc:
            parser.error(str(exc))

    root = Path(__file__).resolve().parents[1]
    observations = PerceptionSequence.from_json(args.observations)
    epl = PhysicalStateExtractor().extract(observations, str(args.video.resolve()))
    epl_path = args.output_dir / "epl.json"
    epl.to_json(epl_path)
    config_payload = json.loads(args.retarget_config.read_text())
    config = LinearRetargetingConfig.from_dict(config_payload)
    retargeted = LinearEPLRetargeter(config).retarget(epl)
    trajectory_path = args.output_dir / "trajectory.json"
    retargeted.trajectory.to_json(trajectory_path)
    (args.output_dir / "retargeting.json").write_text(
        json.dumps(
            {"reachability_failures": retargeted.reachability_failures},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    visualization_path = None
    if args.epl_visualization:
        assert args.camera_intrinsics is not None
        intrinsics = PinholeIntrinsics.from_dict(
            json.loads(args.camera_intrinsics.read_text())
        )
        visualization_path = args.output_dir / "epl_visualization.mp4"
        EPLVisualizer().render(
            args.video,
            observations,
            epl,
            intrinsics,
            visualization_path,
        )
    rollout_path = args.output_dir / "simulation.mp4" if args.render else None
    simulation = MujocoBackend().simulate(
        SimulationRequest(
            model_xml=args.model,
            trajectory=retargeted.trajectory,
            object_body_names=tuple(args.object_body),
            required_contact_pairs=contact_pairs(args.required_contact),
            forbidden_contact_pairs=contact_pairs(args.forbidden_contact),
            object_position_goals=tuple(goals),
            render_output=rollout_path,
        )
    )
    simulation_path = args.output_dir / "simulation.json"
    simulation.to_json(simulation_path)
    packages = {}
    for package in ("mujoco", "numpy", "opencv-python"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {
        "command": [sys.executable, *sys.argv],
        "git_commit": _git_commit(root),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "packages": packages,
        "inputs": {
            "video": str(args.video.resolve()),
            "observations": str(args.observations.resolve()),
            "retarget_config": str(args.retarget_config.resolve()),
            "model": str(args.model.resolve()),
        },
        "outputs": {
            "epl": str(epl_path),
            "trajectory": str(trajectory_path),
            "simulation": str(simulation_path),
            "epl_visualization": (
                str(visualization_path) if visualization_path is not None else None
            ),
            "rollout": str(rollout_path) if rollout_path is not None else None,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    accepted = (
        not retargeted.reachability_failures
        and simulation.physically_valid
        and simulation.task_success is not False
    )
    print(
        json.dumps(
            {
                "accepted": accepted,
                "epl_chunks": len(epl.chunks),
                "trajectory_samples": len(retargeted.trajectory.timestamps_s),
                "simulation": simulation.to_dict(),
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
