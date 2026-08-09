#!/usr/bin/env python3
"""Train a lightweight EPL-conditioned repair-action policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402
from phiagent.training.epl_agent import (  # noqa: E402
    RepairAction,
    encode_example,
    feature_names,
    generate_policy_examples,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_state(project_root: Path) -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        return {"available": False, "error": status.stderr.strip(), "status": []}
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "available": True,
        "head": head.stdout.strip() if head.returncode == 0 else "UNBORN",
        "status": status.stdout.splitlines(),
    }


def _dataset_hash(features: list[list[float]], labels: list[int]) -> str:
    payload = json.dumps(
        {"features": features, "labels": labels},
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _package_versions() -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line for line in completed.stdout.splitlines() if line.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/epl-agent"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--mask-epl", action="store_true")
    parser.add_argument("--acceptance-margin", type=float, default=0.10)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.examples < 600 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("examples must be >=600 and epochs/batch-size must be positive")
    if args.hidden_dim < 4 or args.learning_rate <= 0:
        raise ValueError("hidden-dim must be >=4 and learning-rate must be positive")

    project_root = Path(__file__).resolve().parents[1]
    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("selected physical GPU did not map to exactly one CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = args.experiment_root.expanduser().resolve() / f"{stamp}-{uuid4().hex[:8]}"
    experiment.mkdir(parents=True)
    log_path = experiment / "train.jsonl"
    metadata_path = experiment / "metadata.json"

    examples = generate_policy_examples(args.examples, args.seed)
    features = [
        list(encode_example(example, include_epl=not args.mask_epl))
        for example in examples
    ]
    labels = [int(example.action) for example in examples]
    generator = random.Random(args.seed + 1)
    indices = list(range(len(examples)))
    generator.shuffle(indices)
    train_end = round(len(indices) * 0.7)
    validation_end = round(len(indices) * 0.85)
    split_indices = {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }

    device = torch.device("cuda:0")
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    model = torch.nn.Sequential(
        torch.nn.Linear(feature_tensor.shape[1], args.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_dim, args.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_dim, len(RepairAction)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    loss_function = torch.nn.CrossEntropyLoss()

    metadata = {
        "schema_version": "1.0.0",
        "status": "running",
        "method": "epl_conditioned_repair_policy",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": vars(args),
        "selected_gpu": asdict(selected),
        "gpu_inventory": [asdict(gpu) for gpu in gpus],
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "packages": _package_versions(),
        "git": _git_state(project_root),
        "dataset": {
            "sha256": _dataset_hash(features, labels),
            "examples": len(examples),
            "feature_names": feature_names(),
            "label_names": [action.name for action in RepairAction],
            "split_sizes": {key: len(value) for key, value in split_indices.items()},
            "label_counts": dict(sorted(Counter(labels).items())),
        },
    }
    metadata["config"]["experiment_root"] = str(args.experiment_root)
    _write_json(metadata_path, metadata)

    def accuracy(split: str) -> float:
        model.eval()
        selected_indices = split_indices[split]
        with torch.no_grad():
            logits = model(feature_tensor[selected_indices].to(device))
            predictions = logits.argmax(dim=1).cpu()
        return float((predictions == label_tensor[selected_indices]).float().mean())

    train_generator = torch.Generator().manual_seed(args.seed + 2)
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(split_indices["train"]), generator=train_generator)
            train_indices = torch.tensor(split_indices["train"], dtype=torch.long)[order]
            total_loss = 0.0
            batches = 0
            for start in range(0, len(train_indices), args.batch_size):
                batch_indices = train_indices[start : start + args.batch_size]
                batch_features = feature_tensor[batch_indices].to(device)
                batch_labels = label_tensor[batch_indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(batch_features), batch_labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite training loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
            record = {
                "epoch": epoch,
                "train_loss": total_loss / batches,
                "validation_accuracy": accuracy("validation"),
            }
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()

    test_indices = split_indices["test"]
    model.eval()
    with torch.no_grad():
        test_logits = model(feature_tensor[test_indices].to(device))
        test_predictions = test_logits.argmax(dim=1).cpu()
    test_labels = label_tensor[test_indices]
    test_accuracy = float((test_predictions == test_labels).float().mean())
    majority_accuracy = max(Counter(labels[index] for index in test_indices).values()) / len(
        test_indices
    )
    confusion = [[0 for _ in RepairAction] for _ in RepairAction]
    for truth, prediction in zip(test_labels.tolist(), test_predictions.tolist()):
        confusion[truth][prediction] += 1
    accepted = (
        math.isfinite(test_accuracy)
        and test_accuracy >= majority_accuracy + args.acceptance_margin
    )
    metrics = {
        "accepted": accepted,
        "test_accuracy": test_accuracy,
        "majority_accuracy": majority_accuracy,
        "acceptance_margin": args.acceptance_margin,
        "validation_accuracy": accuracy("validation"),
        "confusion_matrix": confusion,
    }
    _write_json(experiment / "metrics.json", metrics)
    checkpoint = experiment / "policy.pt"
    temporary_checkpoint = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": feature_tensor.shape[1],
            "hidden_dim": args.hidden_dim,
            "actions": [action.name for action in RepairAction],
            "include_epl": not args.mask_epl,
        },
        temporary_checkpoint,
    )
    temporary_checkpoint.replace(checkpoint)
    metadata.update(
        {
            "status": "accepted" if accepted else "rejected",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "checkpoint": str(checkpoint),
        }
    )
    _write_json(metadata_path, metadata)
    print(json.dumps({"experiment": str(experiment), **metrics}, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
