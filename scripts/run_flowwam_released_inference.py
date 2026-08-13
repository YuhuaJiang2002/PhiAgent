#!/usr/bin/env python3
"""Run one audited FlowWAM Stage-1 inference on released robot-flow data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.adapters import FlowWAMConfig, FlowWAMRenderer  # noqa: E402
from phiagent.rendering.wan_animate import (  # noqa: E402
    acquire_gpu_lease,
    query_gpus,
    select_gpu,
)

FLOWWAM_STAGE1_RUNNER_SHA256 = (
    "11d148f8d2df558981c687fd40751263c40094f0d1d15f6e8cf83e65b732018e"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_flowwam_command(
    *,
    python: Path,
    repository: Path,
    test_dataset_dir: Path,
    robot_only_dir: Path,
    embodiment_root: Path,
    output_dir: Path,
    base_model_root: Path,
    checkpoint: Path,
    episode: str,
    num_frames: int,
    num_inference_steps: int,
    fps: int,
    width: int,
    height: int,
    seed: int,
    flow_method: str,
) -> list[str]:
    return [
        str(python),
        str(repository / "inference" / "world_model_inference.py"),
        "--test_dataset_dir",
        str(test_dataset_dir),
        "--robot_only_dir",
        str(robot_only_dir),
        "--embodiment_dir",
        str(embodiment_root),
        "--output_dir",
        str(output_dir),
        "--model_name",
        "FlowWAM-stage1",
        "--full_path",
        str(checkpoint),
        "--local_model_path",
        str(base_model_root),
        "--num_output_frames",
        str(num_frames),
        "--num_inference_steps",
        str(num_inference_steps),
        "--fps",
        str(fps),
        "--size",
        str(width),
        str(height),
        "--flow_method",
        flow_method,
        "--flow_device",
        "cuda:0",
        "--flow_max_magnitude",
        "25.0",
        "--flow_resolution",
        str(width),
        str(height),
        "--robot_render_resolution",
        str(width),
        str(height),
        "--max_stride",
        "3",
        "--max_rollouts",
        "1",
        "--seed",
        str(seed),
        "--episodes",
        episode,
        "--disable_refiner",
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base-model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--test-dataset-dir", type=Path, required=True)
    parser.add_argument("--robot-only-dir", type=Path, required=True)
    parser.add_argument("--embodiment-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--episode", default="episode0")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=60 * 1024)
    parser.add_argument("--num-frames", type=int, default=57)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--flow-method", choices=("raft", "farneback"), default="raft")
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = {
        name: getattr(args, name.replace("-", "_")).expanduser().resolve()
        for name in (
            "repository",
            "base_model_root",
            "checkpoint",
            "test_dataset_dir",
            "robot_only_dir",
            "embodiment_root",
            "experiment_root",
        )
    }
    if (
        args.num_frames < 2
        or args.num_inference_steps <= 0
        or args.fps <= 0
        or args.width <= 0
        or args.height <= 0
        or args.seed < 0
    ):
        raise ValueError("FlowWAM inference settings must be positive")
    python = (
        args.python.expanduser().resolve()
        if args.python is not None
        else paths["repository"] / ".venv" / "bin" / "python"
    )
    for name in (
        "repository",
        "base_model_root",
        "test_dataset_dir",
        "robot_only_dir",
        "embodiment_root",
    ):
        if not paths[name].is_dir():
            raise ValueError(f"missing FlowWAM directory input: {paths[name]}")
    if not paths["checkpoint"].is_file() or not python.is_file():
        raise ValueError("missing FlowWAM checkpoint or Python runtime")
    stage1_runner = (
        paths["repository"] / "inference" / "world_model_inference.py"
    )
    if _sha256(stage1_runner) != FLOWWAM_STAGE1_RUNNER_SHA256:
        raise ValueError(
            "FlowWAM Stage-1 runner does not match the reviewed no-refiner overlay"
        )

    renderer = FlowWAMRenderer(
        FlowWAMConfig(
            repository=paths["repository"],
            base_model_root=paths["base_model_root"],
            checkpoint_path=paths["checkpoint"],
            python_executable=python,
            gpu_index=args.gpu,
            minimum_free_gpu_mib=args.minimum_free_gpu_mib,
            output_fps=args.fps,
        ),
        project_root=Path(__file__).resolve().parents[1],
    )
    preflight = renderer.preflight()
    selected_index = int(preflight["selected_gpu"]["physical_index"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    experiment = paths["experiment_root"] / f"{stamp}-{args.label}-{uuid.uuid4().hex[:8]}"
    experiment.mkdir(parents=True, exist_ok=False)
    output = experiment / "outputs"
    output.mkdir()
    command = build_flowwam_command(
        python=python,
        repository=paths["repository"],
        test_dataset_dir=paths["test_dataset_dir"],
        robot_only_dir=paths["robot_only_dir"],
        embodiment_root=paths["embodiment_root"],
        output_dir=output,
        base_model_root=paths["base_model_root"],
        checkpoint=paths["checkpoint"],
        episode=args.episode,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        fps=args.fps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        flow_method=args.flow_method,
    )
    config = {
        "schema_version": "1.0.0",
        "status": "STARTED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "command": command,
        "seed": args.seed,
        "episode": args.episode,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "fps": args.fps,
        "resolution": [args.width, args.height],
        "flow_method": args.flow_method,
        "refiner": "disabled",
        "preflight": preflight,
        "input_paths": {key: str(value) for key, value in paths.items()},
        "input_hashes": {
            "checkpoint": _sha256(paths["checkpoint"]),
            "stage1_runner": _sha256(stage1_runner),
            "test_action": _sha256(
                paths["test_dataset_dir"]
                / "data"
                / "fixed_scene_task"
                / f"{args.episode}.hdf5"
            ),
            "robot_only": _sha256(
                paths["robot_only_dir"] / f"{args.episode}.hdf5"
            ),
        },
    }
    _write_json(experiment / "config.json", config)
    packages = subprocess.run(
        [str(python), "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    (experiment / "packages.txt").write_text(packages.stdout)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(selected_index)
    environment["PYTHONHASHSEED"] = str(args.seed)
    lease_path, lease = acquire_gpu_lease(selected_index)
    try:
        leased_gpus, inventory, processes = query_gpus()
        selected = select_gpu(
            leased_gpus,
            selected_index,
            args.minimum_free_gpu_mib,
        )
        _write_json(
            experiment / "gpu-lease.json",
            {
                "physical_gpu": selected.physical_index,
                "lease": str(lease_path),
                "inventory_raw": inventory,
                "processes_raw": processes,
            },
        )
        started = time.monotonic()
        with (experiment / "inference.log").open("w") as log:
            completed = subprocess.run(
                command,
                cwd=paths["repository"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        wall_seconds = time.monotonic() - started
    finally:
        lease.close()

    video = output / "FlowWAM-stage1_test" / f"{args.episode}.mp4"
    stream = None
    if completed.returncode == 0 and video.is_file():
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate,nb_frames",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
    status = (
        "WORKING"
        if (
            completed.returncode == 0
            and stream is not None
            and int(stream["nb_frames"]) == args.num_frames
            and stream["avg_frame_rate"] == f"{args.fps}/1"
        )
        else "BLOCKED"
    )
    result = {
        "schema_version": "1.0.0",
        "status": status,
        "return_code": completed.returncode,
        "experiment": str(experiment),
        "video": str(video) if video.is_file() else None,
        "video_sha256": _sha256(video) if video.is_file() else None,
        "stream": stream,
        "wall_seconds": wall_seconds,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Released ALOHA/RoboTwin FlowWAM Stage-1 reproduction; "
            "not real Cobot-Magic generation or physical execution."
        ),
    }
    _write_json(experiment / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "WORKING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
