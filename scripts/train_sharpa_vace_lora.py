#!/usr/bin/env python3
"""Preflight or launch single-GPU Sharpa VACE regional LoRA training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.data.adaptation import AdaptationArm  # noqa: E402
from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402
from phiagent.training.diffsynth_animate import load_frozen_manifest  # noqa: E402
from phiagent.training.diffsynth_vace import (  # noqa: E402
    build_vace_training_command,
    write_vace_metadata,
)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_state(root: Path) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, command in {
        "head": ["git", "rev-parse", "--verify", "HEAD"],
        "status": ["git", "--no-pager", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        payload[key] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diffsynth-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/sharpa-vace-training"))
    parser.add_argument("--accelerate", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=60 * 1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dataset-repeat", type=int, default=50)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="override with rank-4, 256x448x17, one-epoch finite-loss smoke settings",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    manifest = load_frozen_manifest(args.manifest)
    if manifest.arm is not AdaptationArm.VACE_LORA:
        raise SystemExit("the VACE trainer requires a vace_lora manifest")
    accelerate = args.accelerate
    if accelerate is None:
        found = shutil.which("accelerate")
        if found is None:
            raise SystemExit("accelerate is required in the selected training environment")
        accelerate = Path(found)
    if not accelerate.is_file():
        raise SystemExit(f"accelerate does not exist: {accelerate}")

    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    settings = {
        "rank": 4 if args.smoke else args.rank,
        "learning_rate": args.learning_rate,
        "epochs": 1 if args.smoke else args.epochs,
        "dataset_repeat": 1 if args.smoke else args.dataset_repeat,
        "height": 256 if args.smoke else args.height,
        "width": 448 if args.smoke else args.width,
        "num_frames": 17 if args.smoke else args.num_frames,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = args.experiment_root.expanduser().resolve() / f"{stamp}-{uuid4().hex[:8]}"
    experiment.mkdir(parents=True)
    metadata_csv = experiment / "dataset" / "metadata.csv"
    write_vace_metadata(manifest, metadata_csv)
    command = build_vace_training_command(
        accelerate,
        args.diffsynth_repo,
        metadata_csv,
        args.checkpoint_dir,
        experiment / "checkpoints",
        **settings,
    )
    packages = {}
    for name in ("accelerate", "torch", "diffsynth", "peft", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record: dict[str, object] = {
        "schema_version": "0.1.0",
        "method": "diffsynth_wan21_vace_regional_lora_not_official_phizero",
        "status": "preflight_passed" if not args.execute else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "command_shell": shlex.join(command),
        "config": {
            **vars(args),
            "manifest": str(args.manifest),
            "diffsynth_repo": str(args.diffsynth_repo),
            "checkpoint_dir": str(args.checkpoint_dir),
            "experiment_root": str(args.experiment_root),
            "accelerate": str(accelerate),
        },
        "effective_settings": settings,
        "selected_gpu": asdict(selected),
        "gpu_inventory": [asdict(gpu) for gpu in gpus],
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "git": _git_state(project_root),
        "limitations": [
            "Development pseudo-targets validate trainability but cannot support quality claims.",
            "This VACE-1.3B student is not the unreleased PhiZero model.",
        ],
    }
    metadata_path = experiment / "metadata.json"
    _write_json(metadata_path, record)
    if not args.execute:
        print(json.dumps({"experiment": str(experiment), "status": record["status"]}))
        return 0

    log_path = experiment / "training.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=args.diffsynth_repo.expanduser().resolve(),
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record["completed_at"] = datetime.now(timezone.utc).isoformat()
    record["returncode"] = completed.returncode
    record["status"] = "completed" if completed.returncode == 0 else "failed"
    record["training_log"] = str(log_path)
    _write_json(metadata_path, record)
    if completed.returncode:
        raise SystemExit(f"VACE training failed with code {completed.returncode}; see {log_path}")
    print(json.dumps({"experiment": str(experiment), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
