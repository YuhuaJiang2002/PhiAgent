#!/usr/bin/env python3
"""Preflight or launch pinned DiffSynth Wan-Animate LoRA training."""

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

from phiagent.rendering.wan_animate import query_gpus  # noqa: E402
from phiagent.data.adaptation import AdaptationArm  # noqa: E402
from phiagent.training.diffsynth_animate import (  # noqa: E402
    build_diffsynth_training_command,
    gpu_record,
    load_frozen_manifest,
    select_training_gpus,
    verify_diffsynth_checkout,
    write_diffsynth_metadata,
)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_state(root: Path) -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "available": status.returncode == 0,
        "head": head.stdout.strip() if head.returncode == 0 else "UNBORN",
        "status": status.stdout.splitlines(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diffsynth-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/sharpa-training"))
    parser.add_argument("--accelerate", type=Path)
    parser.add_argument("--gpu", type=int, action="append", default=[])
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=75 * 1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dataset-repeat", type=int, default=100)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch the 8-GPU job; without this flag only strict preflight is run",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    manifest = load_frozen_manifest(args.manifest)
    if manifest.arm is not AdaptationArm.ANIMATE_LORA:
        raise SystemExit(
            "the reviewed DiffSynth entry point requires an animate_lora manifest; "
            "appearance_lora and replacement-mode training are not supported"
        )
    diffsynth_commit = verify_diffsynth_checkout(args.diffsynth_repo)
    accelerate = args.accelerate
    if accelerate is None:
        found = shutil.which("accelerate")
        if found is None:
            raise SystemExit("accelerate is required in the selected training environment")
        accelerate = Path(found)
    if not accelerate.is_file():
        raise SystemExit(f"accelerate does not exist: {accelerate}")

    gpus, inventory, processes = query_gpus()
    selected = select_training_gpus(
        gpus,
        args.gpu,
        minimum_free_mib=args.minimum_free_gpu_mib,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu.physical_index) for gpu in selected
    )
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = args.experiment_root.expanduser().resolve() / f"{stamp}-{uuid4().hex[:8]}"
    experiment.mkdir(parents=True)
    metadata_csv = experiment / "dataset" / "metadata.csv"
    write_diffsynth_metadata(manifest, metadata_csv)
    output_path = experiment / "checkpoints"
    command = build_diffsynth_training_command(
        accelerate,
        args.diffsynth_repo,
        metadata_csv,
        args.checkpoint_dir,
        output_path,
        rank=args.rank,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        dataset_repeat=args.dataset_repeat,
    )
    packages = {}
    for name in ("accelerate", "torch", "diffsynth"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record = {
        "schema_version": "0.1.0",
        "method": "diffsynth_wan22_animate_lora_not_official_phizero",
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
        "manifest_sha256": __import__("hashlib").sha256(
            args.manifest.read_bytes()
        ).hexdigest(),
        "diffsynth_commit": diffsynth_commit,
        "selected_gpus": gpu_record(selected),
        "gpu_inventory": [asdict(gpu) for gpu in gpus],
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "git": _git_state(project_root),
    }
    metadata_path = experiment / "metadata.json"
    _write_json(metadata_path, record)
    if not args.execute:
        print(json.dumps({"experiment": str(experiment), "status": record["status"]}))
        return 0

    environment = os.environ.copy()
    log_path = experiment / "training.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=args.diffsynth_repo.expanduser().resolve(),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record["completed_at"] = datetime.now(timezone.utc).isoformat()
    record["returncode"] = completed.returncode
    record["status"] = "completed" if completed.returncode == 0 else "failed"
    _write_json(metadata_path, record)
    if completed.returncode != 0:
        raise SystemExit(
            f"DiffSynth training failed with code {completed.returncode}; see {log_path}"
        )
    print(json.dumps({"experiment": str(experiment), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
