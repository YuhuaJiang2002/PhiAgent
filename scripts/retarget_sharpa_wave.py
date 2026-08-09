#!/usr/bin/env python3
"""Retarget 21-point human hand observations to the official Sharpa Wave model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.perception.schema import PerceptionSequence  # noqa: E402
from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402
from phiagent.retargeting.sharpa_wave import (  # noqa: E402
    SharpaWaveRetargeter,
    load_sharpa_wave_embodiment,
)
from phiagent.simulation.base import SimulationRequest  # noqa: E402
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=1024)
    args = parser.parse_args()
    for label, path in (("observations", args.observations), ("model", args.model)):
        if not path.is_file():
            raise SystemExit(f"{label} does not exist: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty experiment: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    observations = PerceptionSequence.from_json(args.observations)
    embodiment = load_sharpa_wave_embodiment(args.model, args.side)
    result = SharpaWaveRetargeter(embodiment, args.side).retarget(observations)
    trajectory_path = args.output_dir / "trajectory.json"
    result.trajectory.to_json(trajectory_path)

    simulation_path = None
    rollout_path = None
    selected_gpu = None
    gpu_inventory = None
    gpu_processes = None
    accepted = not result.trajectory.joint_limit_violations()
    if args.render:
        gpus, gpu_inventory, gpu_processes = query_gpus()
        selected_gpu = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu.physical_index)
        rollout_path = args.output_dir / "sharpa_wave.mp4"
        simulation = MujocoBackend().simulate(
            SimulationRequest(
                model_xml=args.model,
                trajectory=result.trajectory,
                render_output=rollout_path,
            )
        )
        simulation_path = args.output_dir / "simulation.json"
        simulation.to_json(simulation_path)
        accepted = accepted and simulation.physically_valid

    packages = {}
    for package in ("mujoco", "numpy", "opencv-python"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "command": [sys.executable, *sys.argv],
        "configuration": {
            "side": args.side,
            "render": args.render,
            "seed": None,
            "requested_gpu": args.gpu,
            "minimum_free_gpu_mib": args.minimum_free_gpu_mib,
        },
        "gpu": {
            "selected": asdict(selected_gpu) if selected_gpu is not None else None,
            "inventory": gpu_inventory,
            "processes": gpu_processes,
        },
        "git_commit": _git_commit(root),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "packages": packages,
        "inputs": {
            "observations": str(args.observations.resolve()),
            "model": str(args.model.resolve()),
            "model_sha256": _sha256(args.model),
        },
        "outputs": {
            "trajectory": str(trajectory_path),
            "simulation": str(simulation_path) if simulation_path else None,
            "rollout": str(rollout_path) if rollout_path else None,
        },
        "accepted": accepted,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
