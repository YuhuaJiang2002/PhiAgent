#!/usr/bin/env python3
"""Train the PhiAgent DROID wrist-to-exterior VACE LoRA on one physical GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
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
from typing import Any
from uuid import uuid4


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.data.adaptation import AdaptationArm  # noqa: E402
from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402
from phiagent.training.diffsynth_animate import (  # noqa: E402
    load_frozen_manifest,
    verify_diffsynth_checkout,
)
from phiagent.training.diffsynth_vace import (  # noqa: E402
    build_vace_training_command,
    verify_vace_checkpoint,
    write_vace_metadata,
)


EXPECTED_HOLDOUT = {21, 60, 77}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--source-git-state", type=Path, required=True)
    parser.add_argument("--diffsynth-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/droid-view-lora-training"),
    )
    parser.add_argument("--accelerate", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=70 * 1024)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dataset-repeat", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--execute", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def validate_dataset_contract(
    contract: dict[str, Any],
    manifest_payload: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    if contract.get("method") != "phiagent_droid_wrist_to_exterior_vace_lora_dataset":
        raise ValueError("unexpected DROID view dataset method")
    if contract.get("status") != "WORKING":
        raise ValueError("dataset contract must be WORKING")
    leakage = contract.get("leakage_checks", {})
    if leakage != {
        "episode_disjoint": True,
        "heldout_anchors_used_for_training": False,
        "heldout_targets_used_for_training": False,
    }:
        raise ValueError(f"dataset leakage checks did not pass: {leakage}")
    split = contract.get("split", {})
    train = {int(value) for value in split.get("train", [])}
    validation = {int(value) for value in split.get("validation", [])}
    holdout = {int(value) for value in split.get("holdout", [])}
    if holdout != EXPECTED_HOLDOUT:
        raise ValueError(f"held-out episodes must be {sorted(EXPECTED_HOLDOUT)}")
    if train & validation or train & holdout or validation & holdout:
        raise ValueError("DROID episode splits overlap")
    if contract.get("adaptation_manifest_sha256") != manifest_sha256:
        raise ValueError("dataset contract does not match the adaptation manifest hash")
    assets = manifest_payload.get("assets", [])
    if not assets or any(
        not str(asset.get("path", "")).startswith("dataset/train/") for asset in assets
    ):
        raise ValueError("every adaptation asset must be under dataset/train")
    example_count = len(manifest_payload.get("vace_examples", []))
    if example_count != int(contract.get("training_example_count", -1)):
        raise ValueError("training example count differs between manifest and contract")
    condition = contract.get("conditioning_contract", {}).get("real_condition", [])
    if "one exterior-camera anchor frame at the requested target viewpoint" not in condition:
        raise ValueError("the disclosed target-view anchor condition is missing")
    holdout_records = contract.get("holdout_records", [])
    if {int(row["episode_index"]) for row in holdout_records} != EXPECTED_HOLDOUT:
        raise ValueError("held-out evaluation records are incomplete")
    if any(row.get("training_use") is not False for row in holdout_records):
        raise ValueError("held-out evaluation data cannot be marked for training")
    return {
        "train_episodes": sorted(train),
        "validation_episodes": sorted(validation),
        "holdout_episodes": sorted(holdout),
        "training_examples": example_count,
        "heldout_examples": sum(len(row.get("targets", {})) for row in holdout_records),
        "target_anchor_disclosed": True,
    }


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("accelerate", "torch", "diffsynth", "peft", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main() -> int:
    args = _parser().parse_args()
    if not math.isfinite(args.learning_rate) or min(
        args.minimum_free_gpu_mib,
        args.rank,
        args.learning_rate,
        args.epochs,
        args.dataset_repeat,
        args.height,
        args.width,
        args.num_frames,
    ) <= 0:
        raise ValueError("all numeric training settings must be finite and positive")

    manifest_path = args.manifest.expanduser().resolve()
    contract_path = args.dataset_contract.expanduser().resolve()
    git_state_path = args.source_git_state.expanduser().resolve()
    for label, path in (
        ("manifest", manifest_path),
        ("dataset contract", contract_path),
        ("source Git state", git_state_path),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    manifest_payload = json.loads(manifest_path.read_text())
    contract = json.loads(contract_path.read_text())
    contract_summary = validate_dataset_contract(
        contract,
        manifest_payload,
        _sha256(manifest_path),
    )
    manifest = load_frozen_manifest(manifest_path)
    if manifest.arm is not AdaptationArm.VACE_LORA:
        raise ValueError("DROID view training requires a vace_lora manifest")
    if manifest.evidence_scope != "claim_eligible":
        raise ValueError("DROID view training requires a claim-eligible data split")

    accelerate = args.accelerate
    if accelerate is None:
        located = shutil.which("accelerate")
        if located is None:
            raise ValueError("accelerate is required")
        accelerate = Path(located)
    accelerate = accelerate.expanduser().resolve()
    if not accelerate.is_file():
        raise ValueError(f"accelerate does not exist: {accelerate}")
    diffsynth_repo = args.diffsynth_repo.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    diffsynth_commit = verify_diffsynth_checkout(diffsynth_repo)
    checkpoint_files = verify_vace_checkpoint(checkpoint_dir)

    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = args.experiment_root.expanduser().resolve() / f"{stamp}-{uuid4().hex[:8]}"
    experiment.mkdir(parents=True)
    metadata_csv = experiment / "dataset" / "metadata.csv"
    write_vace_metadata(manifest, metadata_csv)
    command = build_vace_training_command(
        accelerate,
        diffsynth_repo,
        metadata_csv,
        checkpoint_dir,
        experiment / "checkpoints",
        rank=args.rank,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        dataset_repeat=args.dataset_repeat,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "method": "phiagent_droid_wrist_to_exterior_vace_lora_training",
        "model_label": "PhiAgent DROID View LoRA on pinned Wan2.1-VACE-1.3B base",
        "status": "running" if args.execute else "preflight_passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "command_shell": shlex.join(command),
        "config": {
            **vars(args),
            "manifest": str(manifest_path),
            "dataset_contract": str(contract_path),
            "source_git_state": str(git_state_path),
            "diffsynth_repo": str(diffsynth_repo),
            "checkpoint_dir": str(checkpoint_dir),
            "accelerate": str(accelerate),
        },
        "dataset_contract_summary": contract_summary,
        "dataset_contract_sha256": _sha256(contract_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_git_state": json.loads(git_state_path.read_text()),
        "source_git_state_sha256": _sha256(git_state_path),
        "diffsynth_commit": diffsynth_commit,
        "checkpoint_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in checkpoint_files
        ],
        "selected_gpu": asdict(selected),
        "gpu_inventory": [asdict(gpu) for gpu in gpus],
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _package_versions(),
        "limitations": [
            "The model is our trained PhiAgent LoRA adapter on a pinned open VACE base, not a from-scratch foundation model.",
            "Inference uses a disclosed real target-view anchor frame in addition to the real wrist video.",
            "Only held-out visual evaluation can support a generation-quality claim.",
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
            cwd=diffsynth_repo,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    checkpoints = sorted((experiment / "checkpoints").glob("*.safetensors"))
    record.update(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "returncode": completed.returncode,
            "status": "completed" if completed.returncode == 0 and checkpoints else "failed",
            "training_log": str(log_path),
            "training_log_sha256": _sha256(log_path),
            "trained_checkpoints": [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in checkpoints
            ],
        }
    )
    _write_json(metadata_path, record)
    if record["status"] != "completed":
        raise SystemExit(f"DROID view LoRA training failed; see {log_path}")
    print(
        json.dumps(
            {
                "experiment": str(experiment),
                "status": record["status"],
                "checkpoint": str(checkpoints[-1]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
