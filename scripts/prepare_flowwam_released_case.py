#!/usr/bin/env python3
"""Prepare one released FlowWAM robot-flow case with complete provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.adapters import FLOWWAM_REPOSITORY_COMMIT  # noqa: E402
from phiagent.acwm.schema import ACWMActionCondition, ActionRepresentation  # noqa: E402
from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_state(root: Path) -> dict[str, str]:
    state = {}
    for key, command in {
        "head": ["git", "rev-parse", "HEAD"],
        "status": ["git", "status", "--short"],
    }.items():
        try:
            state[key] = subprocess.run(
                command,
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            state[key] = f"unavailable: {type(exc).__name__}: {exc}"
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--test-dataset-dir", type=Path, required=True)
    parser.add_argument("--robot-only-dir", type=Path, required=True)
    parser.add_argument("--embodiment-dir", type=Path, required=True)
    parser.add_argument("--episode", default="episode0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=8 * 1024)
    parser.add_argument("--num-frames", type=int, default=57)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--flow-width", type=int, default=320)
    parser.add_argument("--flow-height", type=int, default=240)
    parser.add_argument("--flow-method", choices=("raft", "farneback"), default="raft")
    parser.add_argument("--flow-max-magnitude", type=float, default=25.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    repository = args.repository.expanduser().resolve()
    test_root = args.test_dataset_dir.expanduser().resolve()
    robot_only_root = args.robot_only_dir.expanduser().resolve()
    embodiment = args.embodiment_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"FlowWAM case already exists: {output}")
    for path, label in (
        (repository, "FlowWAM repository"),
        (test_root, "test dataset"),
        (robot_only_root, "robot-only data"),
        (embodiment, "embodiment"),
    ):
        if not path.is_dir():
            raise ValueError(f"missing {label}: {path}")
    marker = repository / ".phiagent-source-revision"
    revision = marker.read_text().strip() if marker.is_file() else None
    if revision != FLOWWAM_REPOSITORY_COMMIT:
        raise ValueError(
            f"FlowWAM source is {revision!r}, expected {FLOWWAM_REPOSITORY_COMMIT}"
        )
    if (
        args.num_frames < 2
        or args.fps <= 0
        or args.width <= 0
        or args.height <= 0
        or args.flow_width <= 0
        or args.flow_height <= 0
    ):
        raise ValueError("FlowWAM geometry and frame settings must be positive")

    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)

    sys.path.insert(0, str(repository / "inference"))
    sys.path.insert(0, str(repository))
    import imageio.v3 as iio
    import numpy as np
    from world_model_inference import RoboTwinRolloutInferenceDataset

    dataset = RoboTwinRolloutInferenceDataset(
        test_dataset_dir=str(test_root),
        robot_only_dir=str(robot_only_root),
        camera="head_camera",
        size=(args.width, args.height),
        num_frames=args.num_frames,
        flow_method=args.flow_method,
        flow_device="cuda:0",
        flow_max_magnitude=args.flow_max_magnitude,
        embodiment_dir=str(embodiment.parent),
        variant=embodiment.name,
        instruction_variant=0,
        flow_resolution=(args.flow_width, args.flow_height),
    )
    matches = [
        index
        for index, sample in enumerate(dataset.samples)
        if sample["episode_name"] == args.episode
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one released FlowWAM episode, found {len(matches)}")
    sample = dataset[matches[0]]
    flow_arrays = np.stack([np.asarray(frame.convert("RGB")) for frame in sample["flow_video"]])
    if len(flow_arrays) != args.num_frames:
        raise RuntimeError(
            f"FlowWAM producer returned {len(flow_arrays)} frames, expected {args.num_frames}"
        )

    output.mkdir(parents=True)
    inputs = output / "input"
    inputs.mkdir()
    flow_video = inputs / "robot-flow-lossless.mkv"
    iio.imwrite(
        flow_video,
        flow_arrays,
        plugin="FFMPEG",
        fps=args.fps,
        codec="ffv1",
        pixelformat="rgb24",
    )
    decoded = np.stack([frame for frame in iio.imiter(flow_video)])
    if not np.array_equal(decoded, flow_arrays):
        raise RuntimeError("FlowWAM lossless flow-video round trip changed control pixels")
    first_frame = inputs / "first-frame.png"
    np_first = np.asarray(sample["reference_image"].convert("RGB"))
    iio.imwrite(first_frame, np_first)

    source_sample = dataset.samples[matches[0]]
    action_hdf5 = Path(source_sample["action_hdf5"]).resolve()
    robot_only_hdf5 = Path(source_sample["robot_only_hdf5"]).resolve()
    urdf = (embodiment / "urdf" / "arx5_description_isaac.urdf").resolve()
    camera_config = (embodiment / "config.yml").resolve()
    instruction = Path(source_sample["instruction_path"]).resolve()
    for path, label in (
        (action_hdf5, "released action HDF5"),
        (robot_only_hdf5, "released robot-only HDF5"),
        (urdf, "released ALOHA URDF"),
        (camera_config, "released camera configuration"),
        (instruction, "released instruction"),
    ):
        if not path.is_file():
            raise ValueError(f"missing {label}: {path}")

    means = flow_arrays.astype(np.float64).mean(axis=(1, 2))
    condition = ACWMActionCondition(
        label=f"flowwam-{args.episode}",
        instruction="Follow the released ALOHA robot-only optical flow.",
        timeline=(
            f"{args.num_frames} released FlowWAM robot-flow frames at {args.fps:.6g} FPS"
        ),
        representation=ActionRepresentation.ROBOT_FLOW,
        coordinate_frame="camera:head_camera_rgb_pixels",
        timestamps_s=tuple(index / args.fps for index in range(args.num_frames)),
        channels=("encoded_flow_r_mean", "encoded_flow_g_mean", "encoded_flow_b_mean"),
        values=tuple(tuple(float(value) for value in row) for row in means),
        visual_condition=flow_video,
    )
    condition_path = inputs / "condition.json"
    condition.to_json(condition_path)
    provenance_path = inputs / "flow-provenance.json"
    provenance = {
        "schema_version": "1.0.0",
        "producer": "FlowWAM RoboTwinWorldModelInferenceDataset",
        "source_revision": revision,
        "episode": args.episode,
        "camera": "head_camera",
        "flow_method": args.flow_method,
        "flow_max_magnitude": args.flow_max_magnitude,
        "frames": args.num_frames,
        "resolution": [args.width, args.height],
        "flow_resolution": [args.flow_width, args.flow_height],
        "action_hdf5": {"path": str(action_hdf5), "sha256": _sha256(action_hdf5)},
        "robot_only_hdf5": {
            "path": str(robot_only_hdf5),
            "sha256": _sha256(robot_only_hdf5),
        },
        "urdf": {"path": str(urdf), "sha256": _sha256(urdf)},
        "camera_config": {
            "path": str(camera_config),
            "sha256": _sha256(camera_config),
        },
        "flow_video": {"path": str(flow_video), "sha256": _sha256(flow_video)},
    }
    _write_json(provenance_path, provenance)
    files = (
        flow_video,
        first_frame,
        condition_path,
        provenance_path,
        action_hdf5,
        robot_only_hdf5,
        urdf,
        camera_config,
        instruction,
    )
    manifest = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "WORKING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "command": sys.argv,
        "git": _git_state(project_root),
        "selected_gpu": asdict(selected),
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "condition": condition.to_dict(relative_to=output),
        "prompt": sample["prompt"],
        "total_action_frames": int(sample["total_action_frames"]),
        "num_rollouts": int(sample["num_rollouts"]),
        "artifacts": {
            str(path): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        },
        "claim_boundary": (
            "This validates the released ALOHA robot-only flow producer. "
            "It is not Cobot-Magic calibration or generated-video evidence."
        ),
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({"output": str(output), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
