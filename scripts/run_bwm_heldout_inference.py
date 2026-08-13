#!/usr/bin/env python3
"""Run one audited, single-GPU BWM held-out inference attempt."""

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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.wan_animate import acquire_gpu_lease, query_gpus, select_gpu  # noqa: E402
from phiagent.acwm.numeric import NumericActionStatistics  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--dataset-base-path",
        type=Path,
        default=Path("/"),
        help="base directory used to resolve relative video/action paths",
    )
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=60 * 1024)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--action-guidance-scale",
        type=float,
        default=1.0,
        help="target-vs-hold action guidance; 1.0 preserves the released BWM path",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--maximum-samples", type=int, default=1)
    parser.add_argument("--generated-video-seconds", type=float)
    args = parser.parse_args()
    if args.start_index < 0 or args.fps <= 0 or args.action_guidance_scale <= 0:
        raise ValueError("start-index must be non-negative")
    if args.maximum_samples <= 0:
        raise ValueError("maximum-samples must be positive")
    if args.generated_video_seconds is not None and args.generated_video_seconds <= 0:
        raise ValueError("generated-video-seconds must be positive")
    paths = {
        name: getattr(args, name).expanduser().resolve()
        for name in ("repository", "base_model", "checkpoint", "metadata", "action_stats")
    }
    for name, path in paths.items():
        if name in {"repository", "base_model"}:
            valid = path.is_dir()
        else:
            valid = path.is_file() and path.stat().st_size > 0
        if not valid:
            raise ValueError(f"required inference input is missing: {path}")
    guidance_manifest = None
    if args.action_guidance_scale != 1.0:
        guidance_manifest_path = (
            paths["repository"] / ".phiagent-action-guidance-patch-manifest.json"
        )
        if not guidance_manifest_path.is_file():
            raise ValueError(
                "action-guidance scale != 1 requires the verified BWM guidance patch"
            )
        guidance_manifest = json.loads(guidance_manifest_path.read_text())
        if guidance_manifest.get("source_revision") != (
            paths["repository"] / ".phiagent-source-revision"
        ).read_text().strip():
            raise ValueError("BWM action-guidance patch source revision is inconsistent")
    statistics = NumericActionStatistics.from_json(paths["action_stats"])
    metadata_rows = [
        json.loads(line) for line in paths["metadata"].read_text().splitlines() if line.strip()
    ]
    selected_rows = metadata_rows[
        args.start_index : args.start_index + args.maximum_samples
    ]
    if len(selected_rows) != args.maximum_samples:
        raise ValueError("selected BWM metadata range contains too few rows")
    for row in selected_rows:
        if row.get("coordinate_frame") != statistics.coordinate_frame:
            raise ValueError("BWM metadata frame does not match action statistics")
        channels = row.get("channels")
        if channels is not None and tuple(channels) != statistics.channels:
            raise ValueError("BWM metadata channels do not match action statistics")
        if int(row.get("length", 0)) != args.num_frames:
            raise ValueError("BWM metadata length does not match --num-frames")
    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    experiment = args.experiment_root.expanduser().resolve() / f"{timestamp}-{args.label}"
    experiment.mkdir(parents=True, exist_ok=False)
    output = experiment / "videos"
    output.mkdir()
    command = [
        str(paths["repository"] / ".venv" / "bin" / "python"),
        str(paths["repository"] / "scripts" / "infer.py"),
        "--config", str(paths["repository"] / "configs" / "infer" / "infer.yaml"),
        "--model_paths", str(paths["base_model"]),
        "--ckpt_path", str(paths["checkpoint"]),
        "--dataset_base_path", str(args.dataset_base_path.expanduser().resolve()),
        "--dataset_metadata_path", str(paths["metadata"]),
        "--action_stat_path", str(paths["action_stats"]),
        "--action_type", "eef_abs",
        "--output_path", str(output),
        "--seed", str(args.seed),
        "--num_frames", str(args.num_frames),
        "--num_inference_steps", str(args.num_inference_steps),
        "--cfg_scale", str(args.action_guidance_scale),
        "--fps", str(args.fps),
        "--start_index", str(args.start_index),
        "--max_samples", str(args.maximum_samples),
        "--disable_metrics",
    ]
    config = {
        "schema_version": "1.0.0",
        "status": "STARTED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "label": args.label,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "action_guidance_scale": args.action_guidance_scale,
        "action_guidance_baseline": (
            "initial_state_hold" if args.action_guidance_scale != 1.0 else None
        ),
        "action_guidance_patch": guidance_manifest,
        "output_fps": args.fps,
        "start_index": args.start_index,
        "maximum_samples": args.maximum_samples,
        "dataset_base_path": str(args.dataset_base_path.expanduser().resolve()),
        "selected_physical_gpu": asdict(selected),
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "inputs": {name: str(path) for name, path in paths.items()},
        "input_hashes": {
            name: _sha256(path)
            for name, path in paths.items()
            if path.is_file()
        },
        "checkpoint_bytes": paths["checkpoint"].stat().st_size,
        "action_contract": {
            "coordinate_frame": statistics.coordinate_frame,
            "channels": list(statistics.channels),
            "frames": args.num_frames,
        },
        "git": {
            "phiagent": _git_state(Path(__file__).resolve().parents[1]),
            "bwm_source_revision": (
                (paths["repository"] / ".phiagent-source-revision").read_text().strip()
                if (paths["repository"] / ".phiagent-source-revision").is_file()
                else _git_state(paths["repository"]).get("head", "unavailable")
            ),
        },
        "command": command,
    }
    _write_json(experiment / "config.json", config)
    freeze = subprocess.run(
        [str(paths["repository"] / ".venv" / "bin" / "python"), "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    (experiment / "packages.txt").write_text(freeze.stdout)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    environment["PHIAGENT_PHYSICAL_GPU_INDEX"] = str(selected.physical_index)
    environment["PYTHONHASHSEED"] = str(args.seed)
    lease_path, lease = acquire_gpu_lease(selected.physical_index)
    try:
        leased_gpus, leased_inventory, leased_processes = query_gpus()
        selected = select_gpu(
            leased_gpus, selected.physical_index, args.minimum_free_gpu_mib
        )
        _write_json(
            experiment / "gpu-lease.json",
            {
                "physical_gpu": selected.physical_index,
                "lease": str(lease_path),
                "inventory_raw": leased_inventory,
                "processes_raw": leased_processes,
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
    video_paths = sorted(output.glob("*.mp4"))
    videos = [str(path) for path in video_paths]
    status = (
        "WORKING"
        if completed.returncode == 0 and len(videos) == args.maximum_samples
        else "BLOCKED"
    )
    generated_seconds = args.generated_video_seconds
    if generated_seconds is None:
        generated_seconds = args.maximum_samples * args.num_frames / args.fps
    result = {
        "status": status,
        "return_code": completed.returncode,
        "experiment": str(experiment),
        "videos": videos,
        "video_sha256": {str(path): _sha256(path) for path in video_paths},
        "wall_seconds": wall_seconds,
        "generated_video_seconds": generated_seconds,
        "wall_seconds_per_generated_second": wall_seconds / generated_seconds,
        "samples_per_hour": 3600.0 * args.maximum_samples / wall_seconds,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "Held-out generated-video evidence; not real-robot execution.",
    }
    _write_json(experiment / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "WORKING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
