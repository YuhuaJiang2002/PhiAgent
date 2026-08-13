#!/usr/bin/env python3
"""Launch an auditable CP4/CP8 Cosmos Predict2 DROID LoRA experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_cosmos_predict2_droid_inference import (  # noqa: E402
    query_physical_gpus,
    validate_gpu_selection,
)
from scripts.experiment_provenance import package_inventory  # noqa: E402


EXPERIMENT_NAME = "phiagent_droid_lora_attention_14b_480p_16fps"
CUDA_ALLOCATOR_CONF = "backend:native,expandable_segments:True,max_split_size_mb:128"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--min-free-memory-mib", type=int, default=35_000)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--save-iterations", type=int, default=300)
    parser.add_argument("--lora-rank", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--train-frames", type=int, choices=(29, 45, 61, 77, 93), default=45
    )
    parser.add_argument("--master-port", type=int, default=29581)
    parser.add_argument("--git-commit")
    parser.add_argument("--git-branch")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"{label} is missing or empty: {resolved}")
    return resolved


def _require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is missing: {resolved}")
    return resolved


def validate_training_dataset(dataset: Path) -> dict[str, Any]:
    root = _require_dir(dataset, "training dataset")
    contract_path = _require_file(root.parent / "dataset-contract.json", "dataset contract")
    contract = json.loads(contract_path.read_text())
    if contract.get("leakage_checks", {}).get("final_holdout_used_for_training") is not False:
        raise ValueError("dataset contract does not attest final-holdout isolation")
    videos = sorted((root / "videos").glob("*.mp4"))
    metas = sorted((root / "metas").glob("*.txt"))
    embeddings = sorted((root / "t5_xxl").glob("*.pickle"))
    video_stems = {path.stem for path in videos if path.stat().st_size}
    meta_stems = {path.stem for path in metas if path.stat().st_size}
    embedding_stems = {path.stem for path in embeddings if path.stat().st_size}
    if not video_stems:
        raise ValueError("training dataset contains no non-empty videos")
    if video_stems != meta_stems or video_stems != embedding_stems:
        raise ValueError(
            "video/meta/T5 sample IDs differ: "
            f"videos={len(video_stems)}, metas={len(meta_stems)}, embeddings={len(embedding_stems)}"
        )
    expected = int(contract.get("split_counts", {}).get("train", -1))
    if expected != len(video_stems):
        raise ValueError(f"contract expects {expected} train samples, found {len(video_stems)}")
    if contract.get("video_contract", {}).get("training_window_frames") != 93:
        raise ValueError("dataset contract is not the 93-frame Cosmos training-window route")
    return {
        "root": str(root),
        "sample_count": len(video_stems),
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
    }


def _git_state(repo: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "branch", "--show-current"]),
        ("status", ["git", "status", "--short"]),
    ):
        completed = subprocess.run(
            command, cwd=repo, check=False, capture_output=True, text=True
        )
        result[label] = completed.stdout.strip() if completed.returncode == 0 else None
    return result


def stage_training_overlay(
    external_repo: Path, overlay: Path, output: Path
) -> tuple[Path, list[dict[str, str]]]:
    """Archive prior PhiAgent runtime copies so recursive config import stays unambiguous."""
    experiment_dir = (
        external_repo / "cosmos_predict2/configs/base/experiment"
    )
    if not experiment_dir.is_dir():
        raise ValueError(f"Cosmos experiment config directory is missing: {experiment_dir}")
    retired_dir = output / "retired-external-overlays"
    retired: list[dict[str, str]] = []
    for installed in sorted(experiment_dir.glob("phiagent_droid_lora*.py")):
        retired_dir.mkdir(exist_ok=True)
        destination = retired_dir / installed.name
        if destination.exists():
            raise FileExistsError(f"retired overlay collision: {destination}")
        digest = _sha256(installed)
        shutil.move(installed, destination)
        retired.append(
            {
                "original_path": str(installed),
                "archived_path": str(destination),
                "sha256": digest,
            }
        )
    destination = experiment_dir / "phiagent_droid_lora_attention_active.py"
    shutil.copy2(overlay, destination)
    return destination, retired


def main() -> int:
    args = _parser().parse_args()
    if args.max_iterations <= 0 or args.save_iterations <= 0:
        raise ValueError("iteration counts must be positive")
    if not 0.0 < args.learning_rate <= 1e-3:
        raise ValueError("learning-rate must be in (0, 1e-3]")
    if args.max_iterations % args.save_iterations:
        raise ValueError("max-iterations must be divisible by save-iterations")
    if len(args.physical_gpus) not in (4, 8):
        raise ValueError("this training contract requires exactly four or eight GPUs")

    external_repo = _require_dir(args.external_repo, "Cosmos Predict2 repository")
    overlay = _require_file(args.overlay, "training overlay")
    dataset = validate_training_dataset(args.dataset)
    checkpoint = _require_file(args.checkpoint, "DROID base checkpoint")
    tokenizer = _require_file(args.tokenizer, "Cosmos tokenizer")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)

    inventory = query_physical_gpus()
    selection = validate_gpu_selection(
        inventory, args.physical_gpus, args.min_free_memory_mib
    )
    overlay_destination, retired_overlays = stage_training_overlay(
        external_repo, overlay, output
    )

    (output / "packages.txt").write_text(package_inventory())
    _write_json(
        output / "gpu-selection.json",
        {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "inventory_before_launch": inventory,
            "selected_physical_gpus": selection,
            "cuda_visible_devices": ",".join(map(str, args.physical_gpus)),
            "minimum_free_memory_mib": args.min_free_memory_mib,
        },
    )
    run_name = output.name
    artifact_root = output / "training-artifacts"
    warmup_iterations = min(20, max(1, args.max_iterations // 10))
    experiment_config = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "model": "Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID",
        "training_architecture": "LoRA",
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_rank,
            "targets": "q_proj,k_proj,v_proj,output_proj",
            "profile": "attention_only_for_shared_gpu_peak_memory_safety",
            "checkpoint_policy": "adapter tensors only; immutable base referenced by SHA-256",
        },
        "optimizer": {
            "name": "FusedAdamW",
            "lr": args.learning_rate,
            "weight_decay": 0.01,
            "selection_rationale": (
                "validation-gated low-update sweep after the 4.8828125e-4 "
                "stage improved motion but degraded appearance"
            ),
        },
        "schedule": {
            "max_iterations": args.max_iterations,
            "save_iterations": args.save_iterations,
            "warmup_iterations": warmup_iterations,
            "minimum_lr_factor": 0.05,
        },
        "temporal_curriculum": {
            "train_frames": args.train_frames,
            "source_video_frames": 93,
            "sampling": "random contiguous native-16fps subwindow per access",
            "deployment_generation_frames": 93,
        },
        "parallelism": {
            "world_size": len(args.physical_gpus),
            "context_parallel": len(args.physical_gpus),
            "fsdp_shard_size": len(args.physical_gpus),
            "cuda_allocator": CUDA_ALLOCATOR_CONF,
            "allocator_rationale": (
                "PyTorch-documented last-resort max_split_size_mb mitigation for "
                "borderline OOMs with large inactive split blocks"
            ),
        },
        "conditioning": {
            "minimum_latent_conditional_frames": 1,
            "maximum_latent_conditional_frames": 1,
            "real_future_frames_passed_as_conditions": False,
        },
        "view_weighted_loss": {
            "top_left_exterior_1": 1.4,
            "top_right_exterior_2": 1.4,
            "bottom_left_wrist_ego": 0.8,
            "bottom_right_inactive_black": 0.1,
            "normalization": "divide spatial weights by their mean",
        },
        "dataset": dataset,
        "base_checkpoint": {
            "path": str(checkpoint),
            "size": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
        },
        "tokenizer": {
            "path": str(tokenizer),
            "size": tokenizer.stat().st_size,
            "sha256": _sha256(tokenizer),
        },
        "overlay": {
            "source_path": str(overlay),
            "source_sha256": _sha256(overlay),
            "runtime_path": str(overlay_destination),
            "retired_runtime_copies": retired_overlays,
        },
        "project_git": {
            "commit": args.git_commit or "unresolved",
            "branch": args.git_branch,
            "working_tree_status": "dirty",
            "launcher_sha256": _sha256(Path(__file__)),
        },
        "external_git": _git_state(external_repo),
        "seed": 20260812,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    _write_json(output / "experiment-config.json", experiment_config)

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(args.physical_gpus)}",
        f"--master_port={args.master_port}",
        "-m",
        "scripts.train",
        "--config=cosmos_predict2/configs/base/config.py",
        "--",
        f"experiment={EXPERIMENT_NAME}",
        "model.config.train_architecture=lora",
    ]
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.physical_gpus)),
            "NVTE_FUSED_ATTN": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_ALLOC_CONF": CUDA_ALLOCATOR_CONF,
            "PYTORCH_CUDA_ALLOC_CONF": CUDA_ALLOCATOR_CONF,
            "PHIAGENT_DROID_TRAIN_DATASET": dataset["root"],
            "PHIAGENT_DROID_BASE_CHECKPOINT": str(checkpoint),
            "PHIAGENT_DROID_TOKENIZER": str(tokenizer),
            "PHIAGENT_RUN_NAME": run_name,
            "PHIAGENT_MAX_ITER": str(args.max_iterations),
            "PHIAGENT_SAVE_ITER": str(args.save_iterations),
            "PHIAGENT_GPU_COUNT": str(len(args.physical_gpus)),
            "PHIAGENT_LORA_RANK": str(args.lora_rank),
            "PHIAGENT_LEARNING_RATE": str(args.learning_rate),
            "PHIAGENT_TRAIN_FRAMES": str(args.train_frames),
            "IMAGINAIRE_OUTPUT_ROOT": str(artifact_root),
        }
    )
    with (output / "run.log").open("w") as log_handle:
        completed = subprocess.run(
            command,
            cwd=external_repo,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    model_dir = (
        artifact_root
        / "phiagent/droid_multiview_lora_14b"
        / run_name
        / "checkpoints/model"
    )
    checkpoints = sorted(model_dir.glob("iter_*.pt"))
    result = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "returncode": completed.returncode,
        "status": "WORKING" if completed.returncode == 0 and checkpoints else "PARTIAL",
        "model_checkpoints": [
            {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}
            for path in checkpoints
        ],
    }
    _write_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "WORKING":
        raise RuntimeError(f"LoRA training failed; inspect {output / 'run.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
