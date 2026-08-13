#!/usr/bin/env python3
"""Launch auditable Cosmos3-Nano 100%-I2V DROID SFT on physical GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_provenance import package_inventory  # noqa: E402
from scripts.run_cosmos3_droid_i2v import (  # noqa: E402
    query_physical_gpus,
    validate_gpu_selection,
)


OFFICIAL_TOML = "examples/toml/sft_config/vision_sft_nano.toml"
DATASET_PATH = "dataloader_train.dataloader.datasets.video.dataset"
TOKENIZER_BUNDLE_FILES = (
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.json",
    "preprocessor_config.json",
    "generation_config.json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-repo", type=Path, required=True)
    parser.add_argument("--expected-framework-commit", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-lineage-audit",
        type=Path,
        help="required WORKING all-record pixel-lineage audit for formal wrist-only SFT",
    )
    parser.add_argument(
        "--condition-mode",
        choices=("wrist_only", "anchor_multiview"),
        default="wrist_only",
    )
    parser.add_argument("--base-dcp-checkpoint", type=Path, required=True)
    parser.add_argument("--text-tokenizer-root", type=Path, required=True)
    parser.add_argument("--text-tokenizer-vocab-sha256", required=True)
    parser.add_argument("--wan-vae", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--physical-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--profile",
        choices=("smoke", "formal_lora", "formal"),
        default="formal",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--master-port", type=int, default=29641)
    parser.add_argument(
        "--enable-ema", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--enable-compile", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--project-source-revision")
    parser.add_argument("--project-source-branch")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"{label} is missing or empty: {resolved}")
    return resolved


def require_executable(path: Path, label: str) -> Path:
    """Return an absolute executable path without dereferencing a venv symlink."""
    absolute = Path(os.path.abspath(str(path.expanduser())))
    if not absolute.is_file() or not os.access(absolute, os.X_OK):
        raise ValueError(f"{label} is missing or not executable: {absolute}")
    return absolute


def _require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is missing: {resolved}")
    return resolved


def _require_nonempty_dir(path: Path, label: str) -> Path:
    resolved = _require_dir(path, label)
    if not any(item.is_file() for item in resolved.rglob("*")):
        raise ValueError(f"{label} contains no files: {resolved}")
    return resolved


def validate_text_tokenizer(
    root: Path, expected_vocab_sha256: str
) -> dict[str, Any]:
    """Bind the offline Qwen tokenizer files used by every training rank."""
    resolved = _require_dir(root, "local Cosmos3 text tokenizer")
    required = [
        resolved / "vocab.json",
        resolved / "merges.txt",
        resolved / "tokenizer.json",
        resolved / "tokenizer_config.json",
    ]
    for path in required:
        _require_file(path, f"text tokenizer file {path.name}")
    vocab_sha256 = _sha256(required[0])
    if vocab_sha256 != expected_vocab_sha256:
        raise ValueError(
            "text tokenizer vocab SHA-256 mismatch: "
            f"expected {expected_vocab_sha256}, got {vocab_sha256}"
        )
    return {
        "root": str(resolved),
        "vocab_sha256": vocab_sha256,
        "files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in required
        ],
    }


def bundle_text_tokenizer(
    root: Path, output: Path, expected_vocab_sha256: str
) -> dict[str, Any]:
    """Bundle the verified local text tokenizer beside exported HF weights."""
    validated = validate_text_tokenizer(root, expected_vocab_sha256)
    source_root = Path(validated["root"])
    output.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for name in TOKENIZER_BUNDLE_FILES:
        source = source_root / name
        if not source.is_file():
            if name in {"vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"}:
                raise ValueError(f"required tokenizer bundle file is missing: {source}")
            continue
        destination = output / name
        if destination.exists() and _sha256(destination) != _sha256(source):
            raise ValueError(f"export already contains a different tokenizer file: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        files.append(
            {
                "name": name,
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    return {**validated, "source": str(source_root), "files": files}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def project_provenance(
    explicit_revision: str | None = None,
    explicit_branch: str | None = None,
) -> dict[str, Any]:
    try:
        commit = _git(PROJECT_ROOT, "rev-parse", "HEAD")
        branch = _git(PROJECT_ROOT, "branch", "--show-current")
        status = _git(PROJECT_ROOT, "status", "--porcelain")
        git_available = True
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        commit = explicit_revision or "unavailable"
        branch = explicit_branch or "unavailable"
        status = f"Git metadata unavailable in execution copy: {exc}"
        git_available = False
    if explicit_revision:
        commit = explicit_revision
    if explicit_branch:
        branch = explicit_branch
    return {
        "commit": commit,
        "branch": branch,
        "status": status,
        "execution_copy_has_git": git_available,
        "explicit_source_revision": explicit_revision,
        "explicit_source_branch": explicit_branch,
        "launcher_sha256": _sha256(Path(__file__).resolve()),
        "i2v_launcher_sha256": _sha256(
            PROJECT_ROOT / "scripts/run_cosmos3_droid_i2v.py"
        ),
    }


def _external_packages(python: Path) -> str:
    uv = python.parent / "uv"
    command = (
        [str(uv), "pip", "freeze", "--python", str(python)]
        if uv.is_file()
        else [str(python), "-m", "pip", "freeze", "--all"]
    )
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    packages = sorted(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()),
        key=str.casefold,
    )
    if not packages:
        raise RuntimeError(f"could not inventory Cosmos packages with {python}")
    return "\n".join(packages) + "\n"


def validate_sft_dataset(
    dataset_root: Path, condition_mode: str = "wrist_only"
) -> dict[str, Any]:
    contract_path = _require_file(
        dataset_root / "dataset-contract.json", "Cosmos3 DROID dataset contract"
    )
    jsonl = _require_file(
        dataset_root / "train/video_dataset_file.jsonl", "Cosmos3 training JSONL"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_methods = {
        "wrist_only": "cosmos3_nano_droid_wrist_only_to_exterior_i2v_sft_dataset",
        "anchor_multiview": "cosmos3_nano_droid_multiview_i2v_sft_dataset",
    }
    if condition_mode not in expected_methods:
        raise ValueError(f"unknown condition mode: {condition_mode}")
    if contract.get("method") != expected_methods[condition_mode]:
        raise ValueError(
            f"dataset method {contract.get('method')!r} does not match "
            f"condition mode {condition_mode!r}"
        )
    leakage = contract.get("leakage_checks", {})
    required_false = (
        "final_holdout_used_for_training",
        "final_holdout_used_for_checkpoint_selection",
        "validation_future_frames_are_model_inputs",
    )
    if any(leakage.get(key) is not False for key in required_false):
        raise ValueError("dataset contract does not pass the heldout leakage gates")
    if condition_mode == "wrist_only":
        if leakage.get("condition_contains_exterior_pixels") is not False:
            raise ValueError("wrist-only dataset condition contains exterior pixels")
        if leakage.get("condition_contains_real_wrist_pixels_only") is not True:
            raise ValueError("wrist-only dataset does not bind real wrist pixels only")
    required_conditioning = contract.get("training", {}).get(
        "conditioning_distribution_required"
    )
    if required_conditioning != {"i2v_first_frame": 1.0}:
        raise ValueError("dataset does not require the 100%-I2V training contract")
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    expected_count = int(contract.get("split_counts", {}).get("train", -1))
    if not rows or len(rows) != expected_count:
        raise ValueError(
            f"training JSONL has {len(rows)} rows; contract expects {expected_count}"
        )
    for row in rows:
        video = (jsonl.parent / row["vision_path"]).resolve()
        _require_file(video, f"training video {row.get('uuid')}")
        if len(row.get("t2w_windows", [])) != 1:
            raise ValueError(f"sample does not contain exactly one window: {row.get('uuid')}")
        window = row["t2w_windows"][0]
        if (window.get("start_frame"), window.get("end_frame")) != (0, 96):
            raise ValueError(f"sample has wrong frame window: {row.get('uuid')}")
    return {
        "contract_path": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "jsonl_path": str(jsonl),
        "jsonl_sha256": _sha256(jsonl),
        "train_samples": len(rows),
        "total_records": len(contract.get("records", [])),
        "split_counts": contract.get("split_counts", {}),
        "method": contract["method"],
        "condition_mode": condition_mode,
        "claim_scope": contract.get("claim_scope"),
    }


def validate_dataset_lineage_audit(
    audit_path: Path, dataset: dict[str, Any]
) -> dict[str, Any]:
    path = _require_file(audit_path, "wrist-only dataset pixel-lineage audit")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("method") != "phiagent_cosmos3_droid_wrist_only_pixel_lineage_audit":
        raise ValueError("dataset audit is not the wrist-only pixel-lineage audit")
    if audit.get("status") != "WORKING" or audit.get("accepted") is not True:
        raise ValueError("wrist-only dataset pixel-lineage audit is not accepted")
    if audit.get("dataset_contract_sha256") != dataset["contract_sha256"]:
        raise ValueError("pixel-lineage audit does not bind the selected dataset contract")
    records = audit.get("records")
    expected_records = int(dataset["total_records"])
    if not isinstance(records, list) or len(records) != expected_records:
        raise ValueError(
            f"pixel-lineage audit has {len(records) if isinstance(records, list) else 0} "
            f"records; dataset contract has {expected_records}"
        )
    if any(record.get("accepted") is not True for record in records):
        raise ValueError("pixel-lineage audit contains a rejected record")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "status": audit["status"],
        "accepted": audit["accepted"],
        "records": len(records),
        "aggregate": audit.get("aggregate", {}),
        "dataset_contract_sha256": audit["dataset_contract_sha256"],
    }


def resolve_training_profile(
    *,
    profile: str,
    steps: int | None,
    save_every: int | None,
    learning_rate: float | None,
    warmup_steps: int | None,
    max_sequence_length: int | None,
    enable_ema: bool | None,
    enable_compile: bool | None,
) -> dict[str, int | float | bool | str]:
    profiles: dict[str, dict[str, int | float | bool]] = {
        "smoke": {
            "steps": 2,
            "save_every": 1,
            "learning_rate": 2.0e-5,
            "warmup_steps": 1,
            "max_sequence_length": 16_384,
            "num_video_frames": 33,
            "enable_ema": False,
            "enable_compile": False,
            "lora_enabled": False,
            "activation_checkpointing_mode": "selective",
            "context_parallel_shard_degree": 1,
        },
        # The current official FAQ explicitly supports enabling LoRA on the
        # Nano recipe to reduce optimizer memory.  These adapter settings match
        # the official vision_sft_super generation-pathway LoRA recipe while
        # retaining the Nano base and the full 93-frame DROID objective.
        "formal_lora": {
            "steps": 500,
            "save_every": 100,
            "learning_rate": 5.0e-4,
            "warmup_steps": 50,
            "max_sequence_length": 45_056,
            "num_video_frames": 93,
            "enable_ema": False,
            "enable_compile": False,
            "lora_enabled": True,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_target_modules": (
                "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
            ),
            "activation_checkpointing_mode": "full",
            "context_parallel_shard_degree": 2,
        },
        # The official Nano recipe is a 500-step generation-pathway full SFT
        # with 1e-4 LR, 50-step warmup, 45,056-token packing, and EMA.  Keep
        # compile disabled so repeated held-out comparisons remain deterministic.
        "formal": {
            "steps": 500,
            "save_every": 100,
            "learning_rate": 1.0e-4,
            "warmup_steps": 50,
            "max_sequence_length": 45_056,
            "num_video_frames": 93,
            "enable_ema": True,
            "enable_compile": False,
            "lora_enabled": False,
            "activation_checkpointing_mode": "selective",
            "context_parallel_shard_degree": 1,
        },
    }
    if profile not in profiles:
        raise ValueError(f"unknown training profile: {profile}")
    resolved = {"profile": profile, **profiles[profile]}
    explicit = {
        "steps": steps,
        "save_every": save_every,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "max_sequence_length": max_sequence_length,
        "enable_ema": enable_ema,
        "enable_compile": enable_compile,
    }
    resolved.update({key: value for key, value in explicit.items() if value is not None})
    return resolved


def validate_training_gpu_count(profile: str, gpu_count: int) -> None:
    if profile == "formal":
        if gpu_count != 8:
            raise ValueError("formal Cosmos3-Nano SFT requires exactly eight selected GPUs")
        return
    if profile == "smoke":
        if gpu_count < 2:
            raise ValueError("smoke Cosmos3-Nano SFT requires at least two selected GPUs")
        return
    if profile == "formal_lora":
        if gpu_count < 2 or gpu_count % 2:
            raise ValueError(
                "formal LoRA Cosmos3-Nano SFT requires an even GPU count of at least two"
            )
        return
    raise ValueError(f"unknown training profile: {profile}")


def build_tail_overrides(
    *,
    run_name: str,
    seed: int,
    steps: int,
    save_every: int,
    learning_rate: float,
    warmup_steps: int,
    grad_accum: int,
    max_sequence_length: int,
    num_video_frames: int,
    enable_ema: bool,
    enable_compile: bool,
    lora_enabled: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_target_modules: str = (
        "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
    ),
    activation_checkpointing_mode: str = "selective",
    context_parallel_shard_degree: int = 1,
    text_tokenizer_root: Path | None = None,
) -> list[str]:
    if steps <= 0 or save_every <= 0 or save_every > steps:
        raise ValueError("steps/save-every must be positive and save-every <= steps")
    if not 0 < learning_rate <= 1e-3:
        raise ValueError("learning-rate must be in (0, 1e-3]")
    if not 0 <= warmup_steps <= steps:
        raise ValueError("warmup-steps must be between zero and steps")
    if grad_accum <= 0 or max_sequence_length < 10_000:
        raise ValueError("grad-accum and max-sequence-length are invalid")
    if num_video_frames < 25 or (num_video_frames - 1) % 4:
        raise ValueError("num-video-frames must be at least 25 and satisfy 4n+1")
    if activation_checkpointing_mode not in {"none", "selective", "full"}:
        raise ValueError("unknown activation-checkpointing mode")
    if context_parallel_shard_degree <= 0:
        raise ValueError("context-parallel shard degree must be positive")
    overrides = [
        f"job.name={run_name}",
        f"trainer.seed={seed}",
        f"trainer.max_iter={steps}",
        f"trainer.grad_accum_iter={grad_accum}",
        f"checkpoint.save_iter={save_every}",
        f"optimizer.lr={learning_rate:.12g}",
        f"scheduler.cycle_lengths=[{steps}]",
        f"scheduler.warm_up_steps=[{warmup_steps}]",
        # Structured TOML fields under [model] are remapped by the official
        # loader to Hydra's nested ``model.config`` tree.  Extra CLI overrides
        # bypass that remapper, so they must already use the resolved paths.
        f"model.config.max_num_tokens_after_packing={max_sequence_length}",
        f"dataloader_train.max_sequence_length={max_sequence_length}",
        f"model.config.ema.enabled={str(enable_ema).lower()}",
        f"model.config.compile.enabled={str(enable_compile).lower()}",
        f"model.config.activation_checkpointing.mode={activation_checkpointing_mode}",
        "model.config.parallelism.data_parallel_shard_degree=-1",
        "model.config.parallelism.data_parallel_replicate_degree=1",
        f"model.config.parallelism.context_parallel_shard_degree={context_parallel_shard_degree}",
        "trainer.cudnn.benchmark=false",
        "trainer.cudnn.deterministic=true",
        # Hydra merges dict-valued overrides into the recipe default instead
        # of replacing them.  Keep every default key explicit and give the
        # non-I2V branches zero probability; the dataset normalizer then
        # resolves this to an exact 100% one-frame conditioning distribution.
        f"{DATASET_PATH}.conditioning_config={{0:0.0,1:1.0,2:0.0}}",
        f"{DATASET_PATH}.cfg_dropout_rate=0.0",
        f"{DATASET_PATH}.frame_selection_mode=first",
        f"{DATASET_PATH}.num_video_frames={num_video_frames}",
        f"{DATASET_PATH}.resolution='480'",
        f"{DATASET_PATH}.sample_by_window=false",
    ]
    if text_tokenizer_root is not None:
        tokenizer_root = text_tokenizer_root.expanduser().resolve()
        overrides.extend(
            [
                "model.config.vlm_config.tokenizer.config_variant=hf",
                (
                    "model.config.vlm_config.tokenizer.pretrained_model_name="
                    f"{tokenizer_root}"
                ),
            ]
        )
    if lora_enabled:
        if lora_rank <= 0 or lora_alpha <= 0 or not lora_target_modules.strip():
            raise ValueError("LoRA rank, alpha, and target modules must be non-empty")
        overrides.extend(
            [
                "model.config.lora_enabled=true",
                f"model.config.lora_rank={lora_rank}",
                f"model.config.lora_alpha={lora_alpha}",
                f"model.config.lora_target_modules='{lora_target_modules}'",
                "optimizer.keys_to_select=[lora_]",
                "checkpoint.keys_to_skip_loading=[net_ema.,lora_]",
            ]
        )
    else:
        overrides.append("model.config.lora_enabled=false")
    return overrides


def build_command(
    *,
    python: Path,
    gpu_count: int,
    master_port: int,
    toml: Path,
    overrides: Sequence[str],
) -> list[str]:
    torchrun = python.parent / "torchrun"
    if not torchrun.is_file():
        raise ValueError(f"torchrun is missing from the Cosmos environment: {torchrun}")
    return [
        str(torchrun),
        f"--nproc-per-node={gpu_count}",
        f"--master-port={master_port}",
        "-m",
        "cosmos_framework.scripts.train",
        f"--sft-toml={toml}",
        "--",
        *overrides,
    ]


def build_export_command(
    *,
    python: Path,
    checkpoint: Path,
    config: Path,
    output: Path,
) -> list[str]:
    return [
        str(python),
        "-m",
        "cosmos_framework.scripts.export_model",
        # Export is a single-process CPU-offloaded conversion.  Do not inherit
        # the training-time context-parallel degree from the resolved config:
        # CP=2 with WORLD_SIZE=1 fails before any checkpoint tensor is read.
        "--cp-size",
        "1",
        "--no-use-torch-compile",
        "--checkpoint-path",
        str(checkpoint),
        "--config-file",
        str(config),
        "--no-vit",
        "-o",
        str(output),
    ]


def write_single_process_export_config(source: Path, destination: Path) -> Path:
    """Write an export-only config whose model CP degree matches WORLD_SIZE=1."""
    text = source.read_text(encoding="utf-8")
    pattern = r"(?m)^(\s*context_parallel_shard_degree:\s*)\d+(\s*)$"
    rewritten, count = re.subn(pattern, r"\g<1>1\g<2>", text)
    if count != 1:
        raise ValueError(
            "resolved config must contain exactly one context_parallel_shard_degree"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rewritten, encoding="utf-8")
    return destination


def _new_experiment(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = resolved / f"{timestamp}-{uuid.uuid4().hex[:8]}"
    experiment.mkdir()
    return experiment


def _checkpoint_inventory(run_dir: Path) -> list[dict[str, Any]]:
    checkpoints = run_dir / "checkpoints"
    if not checkpoints.is_dir():
        return []
    inventory: list[dict[str, Any]] = []
    for checkpoint in sorted(checkpoints.glob("iter_*")):
        files = [path for path in checkpoint.rglob("*") if path.is_file()]
        inventory.append(
            {
                "path": str(checkpoint),
                "file_count": len(files),
                "bytes": sum(path.stat().st_size for path in files),
            }
        )
    return inventory


def main() -> int:
    args = _parser().parse_args()
    training_profile = resolve_training_profile(
        profile=args.profile,
        steps=args.steps,
        save_every=args.save_every,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_sequence_length=args.max_sequence_length,
        enable_ema=args.enable_ema,
        enable_compile=args.enable_compile,
    )
    framework = _require_dir(args.framework_repo, "Cosmos Framework checkout")
    if not (framework / ".git").is_dir():
        raise ValueError(f"Cosmos Framework path is not a Git checkout: {framework}")
    commit = _git(framework, "rev-parse", "HEAD")
    if commit != args.expected_framework_commit:
        raise ValueError(f"Cosmos Framework is {commit}; expected {args.expected_framework_commit}")
    official_toml = _require_file(framework / OFFICIAL_TOML, "official Nano SFT TOML")
    dataset_root = _require_dir(args.dataset_root, "Cosmos3 DROID SFT dataset")
    dataset = validate_sft_dataset(dataset_root, args.condition_mode)
    lineage_audit = None
    if args.dataset_lineage_audit:
        lineage_audit = validate_dataset_lineage_audit(
            args.dataset_lineage_audit, dataset
        )
    elif args.profile in {"formal", "formal_lora"} and args.condition_mode == "wrist_only":
        raise ValueError(
            "formal wrist-only SFT requires --dataset-lineage-audit with WORKING status"
        )
    base_checkpoint = _require_nonempty_dir(args.base_dcp_checkpoint, "base DCP checkpoint")
    text_tokenizer = validate_text_tokenizer(
        args.text_tokenizer_root, args.text_tokenizer_vocab_sha256
    )
    wan_vae = _require_file(args.wan_vae, "Wan2.2 VAE")
    python = (
        require_executable(args.python_executable, "Cosmos Python")
        if args.python_executable
        else require_executable(framework / ".venv/bin/python", "Cosmos Python")
    )
    inventory = query_physical_gpus()
    selected = validate_gpu_selection(
        inventory, args.physical_gpus, args.minimum_free_gpu_mib
    )
    validate_training_gpu_count(args.profile, len(selected))

    experiment = _new_experiment(args.experiment_root)
    inputs = experiment / "inputs"
    inputs.mkdir()
    copied_toml = inputs / "vision_sft_nano.official.toml"
    shutil.copy2(official_toml, copied_toml)
    run_name = "phiagent_droid_i2v_" + experiment.name.replace("-", "_")
    overrides = build_tail_overrides(
        run_name=run_name,
        seed=args.seed,
        steps=int(training_profile["steps"]),
        save_every=int(training_profile["save_every"]),
        learning_rate=float(training_profile["learning_rate"]),
        warmup_steps=int(training_profile["warmup_steps"]),
        grad_accum=args.grad_accum,
        max_sequence_length=int(training_profile["max_sequence_length"]),
        num_video_frames=int(training_profile["num_video_frames"]),
        enable_ema=bool(training_profile["enable_ema"]),
        enable_compile=bool(training_profile["enable_compile"]),
        lora_enabled=bool(training_profile["lora_enabled"]),
        lora_rank=int(training_profile.get("lora_rank", 16)),
        lora_alpha=int(training_profile.get("lora_alpha", 32)),
        lora_target_modules=str(
            training_profile.get(
                "lora_target_modules",
                "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen",
            )
        ),
        activation_checkpointing_mode=str(
            training_profile["activation_checkpointing_mode"]
        ),
        context_parallel_shard_degree=int(
            training_profile["context_parallel_shard_degree"]
        ),
        text_tokenizer_root=Path(text_tokenizer["root"]),
    )
    command = build_command(
        python=python,
        gpu_count=len(selected),
        master_port=args.master_port,
        toml=copied_toml,
        overrides=overrides,
    )
    training_root = experiment / "training"
    expected_run_dir = training_root / "cosmos3/sft" / run_name
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(row["physical_index"]) for row in selected
            ),
            "PYTHONHASHSEED": str(args.seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "NCCL_NET_PLUGIN": "none",
            "PYTHONPATH": str(framework),
            "PATH": str(python.parent) + os.pathsep + environment.get("PATH", ""),
            "DATASET_PATH": str(dataset_root),
            "BASE_CHECKPOINT_PATH": str(base_checkpoint),
            "WAN_VAE_PATH": str(wan_vae),
            "IMAGINAIRE_OUTPUT_ROOT": str(training_root),
            "WANDB_MODE": "disabled",
        }
    )
    if args.hf_home:
        environment["HF_HOME"] = str(args.hf_home.expanduser().resolve())
    if not args.online:
        environment["HF_HUB_OFFLINE"] = "1"
    metadata_path = experiment / "metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "preflight" if args.preflight_only else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": (
            f"cosmos3_nano_droid_{args.condition_mode}_100pct_i2v_"
            f"{args.profile}_sft"
        ),
        "framework": {
            "path": str(framework),
            "commit": commit,
            "expected_commit": args.expected_framework_commit,
            "official_toml": str(official_toml),
            "official_toml_sha256": _sha256(official_toml),
        },
        "dataset": dataset,
        "dataset_lineage_audit": lineage_audit,
        "base_dcp_checkpoint": str(base_checkpoint),
        "text_tokenizer": text_tokenizer,
        "wan_vae": str(wan_vae),
        "wan_vae_sha256": _sha256(wan_vae),
        "seed": args.seed,
        "training_profile": training_profile,
        "selected_gpus": selected,
        "gpu_inventory": inventory,
        "tail_overrides": overrides,
        "command": command,
        "command_shell": shlex.join(command),
        "expected_run_dir": str(expected_run_dir),
        "environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
                "NCCL_NET_PLUGIN",
                "DATASET_PATH",
                "BASE_CHECKPOINT_PATH",
                "WAN_VAE_PATH",
                "IMAGINAIRE_OUTPUT_ROOT",
                "HF_HOME",
                "HF_HUB_OFFLINE",
            )
            if key in environment
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "project_git": project_provenance(
            args.project_source_revision, args.project_source_branch
        ),
        "launcher_package_versions": package_inventory(),
        "cosmos_package_versions": _external_packages(python),
    }
    _write_json(metadata_path, metadata)
    (experiment / "command.txt").write_text(
        metadata["command_shell"] + "\n", encoding="utf-8"
    )
    if args.preflight_only:
        print(json.dumps({"experiment": str(experiment), "status": "preflight"}))
        return 0

    log_path = experiment / "training.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=framework,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"Cosmos3 SFT failed with exit code {completed.returncode}; inspect {log_path}"
            )
        config = _require_file(expected_run_dir / "config.yaml", "resolved training config")
        checkpoints = _checkpoint_inventory(expected_run_dir)
        if not checkpoints:
            raise RuntimeError(f"successful training wrote no checkpoint under {expected_run_dir}")
        export_metadata: dict[str, Any] = {"status": "skipped"}
        if not args.skip_export:
            pointer = _require_file(
                expected_run_dir / "checkpoints/latest_checkpoint.txt",
                "latest-checkpoint pointer",
            )
            checkpoint_name = pointer.read_text(encoding="utf-8").strip()
            latest_checkpoint = _require_nonempty_dir(
                expected_run_dir / "checkpoints" / checkpoint_name,
                "latest DCP checkpoint",
            )
            exported_model = expected_run_dir / "model"
            export_config = write_single_process_export_config(
                config, experiment / "inputs/single-process-export-config.yaml"
            )
            export_command = build_export_command(
                python=python,
                checkpoint=latest_checkpoint,
                config=export_config,
                output=exported_model,
            )
            export_log = experiment / "export.log"
            with export_log.open("w", encoding="utf-8") as log:
                exported = subprocess.run(
                    export_command,
                    cwd=framework,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if exported.returncode:
                raise RuntimeError(
                    f"Cosmos3 export failed with exit code {exported.returncode}; "
                    f"inspect {export_log}"
                )
            _require_file(exported_model / "config.json", "exported model config")
            tokenizer_bundle = bundle_text_tokenizer(
                Path(text_tokenizer["root"]),
                exported_model,
                args.text_tokenizer_vocab_sha256,
            )
            revision = (
                f"phiagent:{run_name}:{checkpoint_name}:"
                f"{_sha256(config)[:16]}"
            )
            (exported_model / ".phiagent-model-revision").write_text(
                revision + "\n", encoding="utf-8"
            )
            (exported_model / ".phiagent-model-source").write_text(
                f"dcp:{latest_checkpoint}\n", encoding="utf-8"
            )
            export_metadata = {
                "status": "succeeded",
                "command": export_command,
                "command_shell": shlex.join(export_command),
                "checkpoint": str(latest_checkpoint),
                "config": str(export_config),
                "config_sha256": _sha256(export_config),
                "text_tokenizer_bundle": tokenizer_bundle,
                "output": str(exported_model),
                "model_revision": revision,
                "revision_marker_sha256": _sha256(
                    exported_model / ".phiagent-model-revision"
                ),
            }
        metadata.update(
            {
                "status": "succeeded",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "resolved_config": str(config),
                "resolved_config_sha256": _sha256(config),
                "checkpoints": checkpoints,
                "export": export_metadata,
            }
        )
        _write_json(metadata_path, metadata)
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
            }
        )
        _write_json(metadata_path, metadata)
        raise
    print(json.dumps({"experiment": str(experiment), "run_dir": str(expected_run_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
