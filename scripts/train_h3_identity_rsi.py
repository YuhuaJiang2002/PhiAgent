#!/usr/bin/env python3
"""Preflight or execute one immutable native H3 identity-LoRA RSI round."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.minimax_h3 import (  # noqa: E402
    DIFFSYNTH_H3_COMMIT,
    verify_diffsynth_h3_source,
)
from phiagent.rendering.wan_animate import (  # noqa: E402
    acquire_gpu_lease,
    query_gpus,
    select_gpu,
)
from phiagent.training.h3_identity_rsi import H3IdentityRound  # noqa: E402


NF4_FILES = (
    "minimax-h3-text-encoder-nf4.safetensors",
    "minimax-h3-ref2va-nf4.safetensors",
    "video_vae_nf4.safetensors",
    "audio_vae_nf4.safetensors",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
        "error": status.stderr.strip() if status.returncode else None,
    }


def _packages() -> dict[str, str | None]:
    versions = {}
    for name in (
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "peft",
        "safetensors",
        "modelscope",
        "numpy",
        "av",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--diffsynth-repo", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=72 * 1024)
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/h3-identity-rsi"))
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--round-name", default="r0-smoke-r8")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--dataset-repeat", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=30,
        help="Persist immutable intermediate LoRA candidates for early non-regression review.",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset = args.dataset_dir.expanduser().resolve()
    diffsynth = args.diffsynth_repo.expanduser().resolve()
    model_base = args.model_base_path.expanduser().resolve()
    # Preserve a virtual-environment launcher instead of resolving its symlink
    # to the base interpreter, which would silently discard the venv context.
    python = Path(os.path.abspath(args.python.expanduser()))
    round_config = H3IdentityRound(
        name=args.round_name,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        dataset_repeat=args.dataset_repeat,
        num_epochs=args.num_epochs,
    )
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation must be positive")
    if args.save_steps < 1:
        raise ValueError("save_steps must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = (
        args.experiment_dir.expanduser().resolve()
        if args.experiment_dir
        else args.experiment_root.expanduser().resolve()
        / f"{stamp}-{round_config.name}-{uuid4().hex[:8]}"
    )
    experiment.mkdir(parents=True, exist_ok=False)
    manifest_path = experiment / "manifest.json"
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "bounded_rsi_native_minimax_h3_ref2va_identity_lora",
        "status": "preflight_started",
        "honest_status": "NOT STARTED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": _git_state(project_root),
        "seed": args.seed,
        "round": asdict(round_config),
        "execute": args.execute,
        "license": {
            "base": "MiniMax H3 Community License Agreement",
            "applicable_territory_note": "Open weights exclude EU, UK, South Korea, and USA unless separately authorized.",
            "derivative_distribution": "Must carry the upstream license, modified-file notices, NOTICE, territorial restrictions, and AUP safeguards.",
        },
    }
    _write_json(manifest_path, manifest)
    lease = None
    try:
        dataset_manifest_path = dataset / "manifest.json"
        metadata_path = dataset / "metadata.json"
        for path in (dataset_manifest_path, metadata_path, python):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"required input is missing or empty: {path}")
        dataset_manifest = json.loads(dataset_manifest_path.read_text())
        if dataset_manifest.get("status") != "completed" or not dataset_manifest.get(
            "accepted_for_training"
        ):
            raise ValueError("dataset manifest is not completed and accepted for training")
        contract = dataset_manifest["dataset_contract"]
        if not isinstance(contract, dict):
            raise ValueError("dataset manifest has no dataset_contract")
        width = int(contract["width"])
        height = int(contract["height"])
        num_frames = int(contract["num_frames"])
        if width % 32 or height % 32 or (num_frames - 5) % 17:
            raise ValueError("dataset violates MiniMax-H3 spatial/temporal grouping")
        source_revision = verify_diffsynth_h3_source(diffsynth)
        if source_revision != DIFFSYNTH_H3_COMMIT:
            raise ValueError(f"unexpected DiffSynth revision: {source_revision}")
        train_script = diffsynth / "examples/minimax_h3/model_training/train.py"
        if not train_script.is_file():
            raise ValueError(f"reviewed H3 trainer is missing: {train_script}")
        checkpoint_root = model_base / "DiffSynth-Studio/MiniMax-H3-NF4"
        model_paths = [checkpoint_root / name for name in NF4_FILES]
        processor = model_base / "MiniMax/MiniMax-H3/Ref2VA/processor"
        for path in (*model_paths, processor):
            if not path.exists():
                raise ValueError(f"H3 model input is missing: {path}")
        gpus, inventory_raw, processes_raw = query_gpus()
        selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
        output_path = experiment / "checkpoints"
        training_args = [
            "--dataset_base_path",
            str(dataset),
            "--dataset_metadata_path",
            str(metadata_path),
            "--data_file_keys",
            "video,input_audio,references",
            "--extra_inputs",
            "input_audio,references",
            "--height",
            str(height),
            "--width",
            str(width),
            "--num_frames",
            str(num_frames),
            "--dataset_repeat",
            str(round_config.dataset_repeat),
            "--model_paths",
            json.dumps([str(path) for path in model_paths]),
            "--processor_path",
            # This pinned DiffSynth trainer reconstructs processor_config from
            # model_id/origin_file_pattern and drops a direct local path.  Keep
            # the local processor preflight above, then use its supported ID
            # form; DIFFSYNTH_MODEL_BASE_PATH resolves it without downloading.
            "MiniMax/MiniMax-H3:Ref2VA/processor/",
            "--learning_rate",
            str(round_config.learning_rate),
            "--num_epochs",
            str(round_config.num_epochs),
            "--gradient_accumulation_steps",
            str(args.gradient_accumulation_steps),
            "--save_steps",
            str(args.save_steps),
            "--remove_prefix_in_ckpt",
            "pipe.dit.",
            "--output_path",
            str(output_path),
            "--lora_base_model",
            "dit",
            "--lora_target_modules",
            round_config.target_modules,
            "--lora_rank",
            str(round_config.lora_rank),
            "--use_gradient_checkpointing",
            "--find_unused_parameters",
            "--silent_on_missing_audio",
        ]
        command = [
            str(python),
            str(project_root / "scripts/run_h3_seeded_training.py"),
            "--seed",
            str(args.seed),
            "--training-script",
            str(train_script),
            "--",
            *training_args,
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
        environment["PYTHONHASHSEED"] = str(args.seed)
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(diffsynth), str(project_root), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        environment["DIFFSYNTH_MODEL_BASE_PATH"] = str(model_base)
        # Every model and processor file was already validated and hashed
        # above.  Fail closed instead of allowing a training run to mutate its
        # inputs through an implicit network download.
        environment["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
        manifest.update(
            {
                "status": "preflight_passed",
                "honest_status": "NOT STARTED" if not args.execute else "PARTIAL",
                "source_revision": source_revision,
                "dataset": {
                    "path": str(dataset),
                    "manifest_sha256": _sha256(dataset_manifest_path),
                    "metadata_sha256": _sha256(metadata_path),
                    "records": len(dataset_manifest.get("records", [])),
                    "contract": contract,
                },
                "models": [
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                    for path in model_paths
                ],
                "processor": str(processor),
                "selected_gpu": asdict(selected),
                "gpu_inventory": [asdict(gpu) for gpu in gpus],
                "gpu_inventory_raw": inventory_raw,
                "gpu_processes_raw": processes_raw,
                "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
                "diffsynth_skip_download": environment["DIFFSYNTH_SKIP_DOWNLOAD"],
                "determinism": {
                    "mode": "strict",
                    "pythonhashseed": environment["PYTHONHASHSEED"],
                    "cublas_workspace_config": environment["CUBLAS_WORKSPACE_CONFIG"],
                    "training_wrapper": str(project_root / "scripts/run_h3_seeded_training.py"),
                    "training_wrapper_sha256": _sha256(
                        project_root / "scripts/run_h3_seeded_training.py"
                    ),
                },
                "checkpoint_policy": {
                    "save_steps": args.save_steps,
                    "purpose": "early immutable non-regression and topology tournament",
                },
                "packages": _packages(),
                "training_command": command,
                "training_command_shell": shlex.join(command),
            }
        )
        _write_json(manifest_path, manifest)
        if not args.execute:
            print(json.dumps({"experiment": str(experiment), "status": "preflight_passed"}))
            return 0
        _, lease = acquire_gpu_lease(selected.physical_index)
        manifest["status"] = "running"
        manifest["started_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)
        with (experiment / "training.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=diffsynth,
                env=environment,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        checkpoints = sorted(output_path.glob("*.safetensors"))
        if completed.returncode or not checkpoints:
            tail = (experiment / "training.log").read_text(errors="replace")[-12000:]
            raise RuntimeError(
                f"H3 training exited {completed.returncode} and produced {len(checkpoints)} checkpoints; log tail:\n{tail}"
            )
        manifest.update(
            {
                "status": "completed",
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "checkpoints": [
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                    for path in checkpoints
                ],
                "acceptance": {
                    "native_lora_training_completed": True,
                    "held_out_identity_evaluated": False,
                    "promotion_contract_passed": False,
                },
                "limitations": [
                    "A completed optimization run is not an identity improvement until matched held-out inference and all promotion gates pass.",
                    "Round r0 is a low-resolution smoke/capability probe, not a publication checkpoint.",
                ],
            }
        )
        _write_json(manifest_path, manifest)
        print(json.dumps({"experiment": str(experiment), "checkpoints": len(checkpoints)}))
        return 0
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "honest_status": "PARTIAL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(manifest_path, manifest)
        raise
    finally:
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
