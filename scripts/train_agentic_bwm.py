#!/usr/bin/env python3
"""Launch one evidence-recorded, single-GPU BWM adaptation attempt.

The default stage trains only the action encoder.  This is the conservative
appearance/dynamics decoupling step; joint DiT fine-tuning is an explicit later
stage and never silently replaces it.
"""

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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.adapters import (  # noqa: E402
    BWM_BASE_MODEL_REVISION,
    BWM_MODEL_REVISION,
    BWM_MODEL_BYTES,
    BWM_MODEL_SHA256,
    BWM_REPOSITORY_COMMIT,
)
from phiagent.rendering.wan_animate import (  # noqa: E402
    acquire_gpu_lease,
    query_gpus,
    select_gpu,
)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision(repository: Path) -> str:
    if (repository / ".git").is_dir():
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    marker = repository / ".phiagent-source-revision"
    if not marker.is_file():
        raise ValueError(f"BWM source revision is not recorded: {repository}")
    return marker.read_text().strip()


def _require_revision(path: Path, expected: str, label: str) -> None:
    marker = path / ".phiagent-model-revision"
    actual = marker.read_text().strip() if marker.is_file() else None
    if actual != expected:
        raise ValueError(f"{label} revision is {actual!r}, expected {expected!r}")


def build_train_command(
    *,
    repository: Path,
    base_model: Path,
    checkpoint: Path,
    dataset_root: Path,
    metadata: Path,
    action_stats: Path,
    output: Path,
    stage: str,
    seed: int,
    learning_rate: float,
    epochs: int,
    dataset_repeat: int,
    workers: int,
    gradient_accumulation: int,
    physical_gpu_index: int,
    accelerate_config: Path,
) -> list[str]:
    trainable = "action_encoder" if stage == "action-adapter" else "dit,action_encoder"
    return [
        str(repository / ".venv" / "bin" / "accelerate"),
        "launch",
        "--config_file",
        str(accelerate_config),
        "--num_processes",
        "1",
        "--gpu_ids",
        str(physical_gpu_index),
        str(repository / "scripts" / "train.py"),
        "--config",
        str(repository / "configs" / "infer" / "infer.yaml"),
        "--dataset_base_path",
        str(dataset_root),
        "--dataset_metadata_path",
        str(metadata),
        "--dataset_repeat",
        str(dataset_repeat),
        "--dataset_num_workers",
        str(workers),
        "--data_file_keys",
        "video,action",
        "--model_paths",
        str(base_model),
        "--model_config_path",
        str(repository / "configs" / "model" / "wan2_2_ti2v_5b.yaml"),
        "--text_mode",
        "none",
        "--action_mode",
        "adaln",
        "--action_type",
        "eef_abs",
        "--action_stat_path",
        str(action_stats),
        "--action_dim",
        "14",
        "--learning_rate",
        str(learning_rate),
        "--num_epochs",
        str(epochs),
        "--trainable_models",
        trainable,
        "--seed",
        str(seed),
        "--mixed_precision",
        "bf16",
        "--output_path",
        str(output),
        "--ckpt_path",
        str(checkpoint),
        "--use_gradient_checkpointing",
        "--gradient_accumulation_steps",
        str(gradient_accumulation),
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("action-adapter", "joint-finetune"), default="action-adapter")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=64 * 1024)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dataset-repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument(
        "--maximum-training-clips",
        type=int,
        default=0,
        help="deterministically truncate train metadata for a smoke test",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository = args.repository.expanduser().resolve()
    base_model = args.base_model.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    experiment_root = args.experiment_root.expanduser().resolve()
    if args.seed < 0 or args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("seed, epochs, and learning rate must be positive")
    if args.maximum_training_clips < 0:
        raise ValueError("maximum-training-clips must be non-negative")

    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    experiment = experiment_root / f"{timestamp}-{args.stage}-seed{args.seed}"
    experiment.mkdir(parents=True, exist_ok=False)
    configuration: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "STARTED",
        "stage": args.stage,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "dataset_repeat": args.dataset_repeat,
        "workers": args.workers,
        "gradient_accumulation": args.gradient_accumulation,
        "maximum_training_clips": args.maximum_training_clips,
        "repository": str(repository),
        "base_model": str(base_model),
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "experiment": str(experiment),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "selected_physical_gpu": asdict(selected),
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(experiment / "config.json", configuration)
    try:
        if _revision(repository) != BWM_REPOSITORY_COMMIT:
            raise ValueError("BWM source does not match the pinned commit")
        patch_manifest_path = repository / ".phiagent-patch-manifest.json"
        if not patch_manifest_path.is_file():
            raise ValueError("BWM packed-offset compatibility patch is not recorded")
        patch_manifest = json.loads(patch_manifest_path.read_text())
        patch_path = Path(__file__).resolve().parents[1] / "patches" / "bwm-robotwin-packed-offsets.patch"
        if (
            patch_manifest.get("source_revision") != BWM_REPOSITORY_COMMIT
            or patch_manifest.get("patch_sha256") != _sha256(patch_path)
        ):
            raise ValueError("BWM patch manifest does not match the reviewed patch")
        text_patch_manifest_path = repository / ".phiagent-text-mode-patch-manifest.json"
        if not text_patch_manifest_path.is_file():
            raise ValueError("BWM text-mode compatibility patch is not recorded")
        text_patch_manifest = json.loads(text_patch_manifest_path.read_text())
        text_patch_path = (
            Path(__file__).resolve().parents[1] / "patches" / "bwm-training-text-mode.patch"
        )
        if (
            text_patch_manifest.get("source_revision") != BWM_REPOSITORY_COMMIT
            or text_patch_manifest.get("patch_sha256") != _sha256(text_patch_path)
        ):
            raise ValueError("BWM text-mode patch evidence does not match the reviewed patch")
        builder_manifest_path = repository / ".phiagent-training-builder-patch-manifest.json"
        if not builder_manifest_path.is_file():
            raise ValueError("BWM training-builder compatibility patch is not recorded")
        builder_manifest = json.loads(builder_manifest_path.read_text())
        builder_patch_path = (
            Path(__file__).resolve().parents[1]
            / "patches"
            / "bwm-training-builder-call.patch"
        )
        if (
            builder_manifest.get("source_revision") != BWM_REPOSITORY_COMMIT
            or builder_manifest.get("patch_sha256") != _sha256(builder_patch_path)
        ):
            raise ValueError("BWM training-builder patch evidence is invalid")
        sharded_manifest_path = repository / ".phiagent-sharded-model-patch-manifest.json"
        if not sharded_manifest_path.is_file():
            raise ValueError("BWM sharded-model compatibility patch is not recorded")
        sharded_manifest = json.loads(sharded_manifest_path.read_text())
        sharded_patch_path = (
            Path(__file__).resolve().parents[1]
            / "patches"
            / "bwm-training-sharded-model-path.patch"
        )
        if (
            sharded_manifest.get("source_revision") != BWM_REPOSITORY_COMMIT
            or sharded_manifest.get("patch_sha256") != _sha256(sharded_patch_path)
        ):
            raise ValueError("BWM sharded-model patch evidence is invalid")
        dataset_contract_manifest_path = (
            repository / ".phiagent-dataset-contract-patch-manifest.json"
        )
        if not dataset_contract_manifest_path.is_file():
            raise ValueError("BWM dataset-runner compatibility patch is not recorded")
        dataset_contract_manifest = json.loads(dataset_contract_manifest_path.read_text())
        dataset_contract_patch_path = (
            Path(__file__).resolve().parents[1]
            / "patches"
            / "bwm-dataset-runner-contract.patch"
        )
        if (
            dataset_contract_manifest.get("source_revision") != BWM_REPOSITORY_COMMIT
            or dataset_contract_manifest.get("patch_sha256")
            != _sha256(dataset_contract_patch_path)
        ):
            raise ValueError("BWM dataset-contract patch evidence is invalid")
        video_tensor_manifest_path = repository / ".phiagent-video-tensor-patch-manifest.json"
        if not video_tensor_manifest_path.is_file():
            raise ValueError("BWM training video-tensor patch is not recorded")
        video_tensor_manifest = json.loads(video_tensor_manifest_path.read_text())
        video_tensor_patch_path = (
            Path(__file__).resolve().parents[1]
            / "patches"
            / "bwm-training-video-tensor-shape.patch"
        )
        if (
            video_tensor_manifest.get("source_revision") != BWM_REPOSITORY_COMMIT
            or video_tensor_manifest.get("patch_sha256")
            != _sha256(video_tensor_patch_path)
        ):
            raise ValueError("BWM video-tensor patch evidence is invalid")
        history_loss_manifest_path = repository / ".phiagent-history-loss-patch-manifest.json"
        if not history_loss_manifest_path.is_file():
            raise ValueError("BWM multi-latent history-loss patch is not recorded")
        history_loss_manifest = json.loads(history_loss_manifest_path.read_text())
        history_loss_patch_path = (
            Path(__file__).resolve().parents[1]
            / "patches"
            / "bwm-training-history-loss.patch"
        )
        if (
            history_loss_manifest.get("source_revision") != BWM_REPOSITORY_COMMIT
            or history_loss_manifest.get("patch_sha256")
            != _sha256(history_loss_patch_path)
        ):
            raise ValueError("BWM history-loss patch evidence is invalid")
        _require_revision(base_model, BWM_BASE_MODEL_REVISION, "Wan2.2 base model")
        _require_revision(checkpoint.parent, BWM_MODEL_REVISION, "BWM checkpoint")
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError(f"BWM checkpoint is missing: {checkpoint}")
        checkpoint_verification = checkpoint.parent / ".phiagent-model-verification.json"
        if not checkpoint_verification.is_file():
            raise ValueError("BWM checkpoint has no hash-verification manifest")
        verified = json.loads(checkpoint_verification.read_text())
        if (
            verified.get("sha256") != BWM_MODEL_SHA256
            or int(verified.get("bytes", -1)) != BWM_MODEL_BYTES
            or checkpoint.stat().st_size != BWM_MODEL_BYTES
        ):
            raise ValueError("BWM checkpoint hash-verification manifest is invalid")
        dataset_manifest = dataset_root / "manifest.json"
        action_stats = dataset_root / "action-stat.json"
        source_metadata = dataset_root / "train.jsonl"
        for path in (dataset_manifest, action_stats, source_metadata):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"prepared training input is missing: {path}")
        dataset_evidence = json.loads(dataset_manifest.read_text())
        dataset_status = dataset_evidence.get("status")
        dataset_honest_status = dataset_evidence.get("honest_status")
        if not (
            dataset_status == "WORKING"
            or (
                dataset_status == "completed"
                and dataset_honest_status == "WORKING"
            )
        ):
            raise ValueError("prepared dataset is not marked WORKING")

        metadata = source_metadata
        if args.maximum_training_clips:
            rows = source_metadata.read_text().splitlines()
            rows = rows[: args.maximum_training_clips]
            if not rows:
                raise ValueError("training metadata contains no clips")
            metadata = experiment / "train-subset.jsonl"
            metadata.write_text("\n".join(rows) + "\n")
        output = experiment / "checkpoints"
        output.mkdir()
        accelerate_config = experiment / "accelerate-single-gpu.yaml"
        _write_json(
            accelerate_config,
            {
                "compute_environment": "LOCAL_MACHINE",
                "distributed_type": "NO",
                "mixed_precision": "bf16",
                "use_cpu": False,
                "num_processes": 1,
                "num_machines": 1,
                "machine_rank": 0,
                "rdzv_backend": "static",
                "same_network": True,
            },
        )
        command = build_train_command(
            repository=repository,
            base_model=base_model,
            checkpoint=checkpoint,
            dataset_root=dataset_root,
            metadata=metadata,
            action_stats=action_stats,
            output=output,
            stage=args.stage,
            seed=args.seed,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            dataset_repeat=args.dataset_repeat,
            workers=args.workers,
            gradient_accumulation=args.gradient_accumulation,
            physical_gpu_index=selected.physical_index,
            accelerate_config=accelerate_config,
        )
        _write_json(
            experiment / "preflight.json",
            {
                **configuration,
                "status": "WORKING",
                "source_revision": BWM_REPOSITORY_COMMIT,
                "patch_manifest": patch_manifest,
                "text_mode_patch_manifest": text_patch_manifest,
                "training_builder_patch_manifest": builder_manifest,
                "sharded_model_patch_manifest": sharded_manifest,
                "dataset_contract_patch_manifest": dataset_contract_manifest,
                "video_tensor_patch_manifest": video_tensor_manifest,
                "history_loss_patch_manifest": history_loss_manifest,
                "base_model_revision": BWM_BASE_MODEL_REVISION,
                "checkpoint_revision": BWM_MODEL_REVISION,
                "dataset_manifest": dataset_evidence,
                "metadata": str(metadata),
                "command": command,
            },
        )
        python = repository / ".venv" / "bin" / "python"
        freeze = subprocess.run(
            [str(python), "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        (experiment / "packages.txt").write_text(freeze.stdout)
        if args.preflight_only:
            result = {
                "status": "PARTIAL",
                "reason": "GPU/model/data preflight passed; training was not requested",
                "experiment": str(experiment),
            }
            _write_json(experiment / "result.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        environment = os.environ.copy()
        inherited_pythonpath = environment.get("PYTHONPATH")
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(selected.physical_index),
                "PHIAGENT_PHYSICAL_GPU_INDEX": str(selected.physical_index),
                "PYTHONHASHSEED": str(args.seed),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONPATH": (
                    str(repository)
                    if not inherited_pythonpath
                    else os.pathsep.join((str(repository), inherited_pythonpath))
                ),
            }
        )
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
            with (experiment / "train.log").open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=repository,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        finally:
            lease.close()
        checkpoints = sorted(str(path) for path in output.rglob("*.safetensors"))
        status = "WORKING" if completed.returncode == 0 and checkpoints else "BLOCKED"
        result = {
            "status": status,
            "return_code": completed.returncode,
            "experiment": str(experiment),
            "checkpoints": checkpoints,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "claim_boundary": (
                "This proves one training stage executed, not that the model is SOTA or "
                "that it works on a real robot."
            ),
        }
        _write_json(experiment / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if status == "WORKING" else 2
    except Exception as exc:
        result = {
            "status": "BLOCKED",
            "error": f"{type(exc).__name__}: {exc}",
            "experiment": str(experiment),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(experiment / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
