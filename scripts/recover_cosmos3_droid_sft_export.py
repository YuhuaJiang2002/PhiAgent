#!/usr/bin/env python3
"""Recover a completed Cosmos3 SFT checkpoint into an auditable HF export.

The official exporter is intentionally run with context parallelism one: the
resolved training config may contain CP=2, while export is a single process.
The source training attempt remains immutable, including its failed export log.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_provenance import package_inventory  # noqa: E402
from scripts.run_cosmos3_droid_i2v import (  # noqa: E402
    _external_package_inventory,
    _git_output,
    _require_dir,
    _require_file,
    _sha256,
    _write_json,
    project_provenance,
    query_physical_gpus,
    require_executable,
    validate_gpu_selection,
)
from scripts.run_cosmos3_droid_sft_training import (  # noqa: E402
    bundle_text_tokenizer,
    build_export_command,
    write_single_process_export_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-repo", type=Path, required=True)
    parser.add_argument("--expected-framework-commit", required=True)
    parser.add_argument("--training-experiment", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--text-tokenizer-root", type=Path, required=True)
    parser.add_argument("--text-tokenizer-vocab-sha256", required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=55_000)
    parser.add_argument("--project-source-revision")
    parser.add_argument("--project-source-branch")
    return parser


def _new_experiment(root: Path) -> Path:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / f"{timestamp}-{uuid.uuid4().hex[:8]}"
    output.mkdir()
    return output


def _source_training_artifacts(training_experiment: Path) -> dict[str, Path]:
    source = _require_dir(training_experiment, "source Cosmos3 training experiment")
    metadata = json.loads(
        _require_file(source / "metadata.json", "source training metadata").read_text(
            encoding="utf-8"
        )
    )
    run_dir = _require_dir(
        Path(str(metadata["expected_run_dir"])), "completed Cosmos3 training run"
    )
    training_log = _require_file(source / "training.log", "source training log")
    if "Done with training." not in training_log.read_text(encoding="utf-8"):
        raise ValueError("source training log has no successful terminal marker")
    pointer = _require_file(
        run_dir / "checkpoints/latest_checkpoint.txt", "latest checkpoint pointer"
    )
    checkpoint_name = pointer.read_text(encoding="utf-8").strip()
    if checkpoint_name != "iter_000000500":
        raise ValueError(f"expected final iteration 500, found {checkpoint_name!r}")
    return {
        "source": source,
        "metadata": source / "metadata.json",
        "training_log": training_log,
        "run_dir": run_dir,
        "config": _require_file(run_dir / "config.yaml", "resolved training config"),
        "checkpoint": _require_dir(
            run_dir / "checkpoints" / checkpoint_name, "final DCP checkpoint"
        ),
    }


def _tree_inventory(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return {
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "files": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
            if path.name in {"config.json", "checkpoint.json", "export_manifest.json"}
            or path.suffix == ".safetensors"
        ],
    }


def main() -> int:
    args = _parser().parse_args()
    framework = _require_dir(args.framework_repo, "Cosmos Framework checkout")
    commit = _git_output(framework, "rev-parse", "HEAD")
    if commit != args.expected_framework_commit:
        raise ValueError(f"framework is {commit}; expected {args.expected_framework_commit}")
    python = require_executable(args.python_executable, "Cosmos Python")
    source = _source_training_artifacts(args.training_experiment)
    inventory = query_physical_gpus()
    selected = validate_gpu_selection(
        inventory, [args.physical_gpu], args.minimum_free_gpu_mib
    )
    experiment = _new_experiment(args.experiment_root)
    inputs = experiment / "inputs"
    inputs.mkdir()
    source_config_copy = inputs / "resolved-training-config.yaml"
    shutil.copy2(source["config"], source_config_copy)
    config = write_single_process_export_config(
        source_config_copy, inputs / "single-process-export-config.yaml"
    )
    model = experiment / "model"
    command = build_export_command(
        python=python,
        checkpoint=source["checkpoint"],
        config=config,
        output=model,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
            "HF_HUB_OFFLINE": "1",
            "PYTHONHASHSEED": "20260812",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PATH": str(python.parent) + os.pathsep + environment.get("PATH", ""),
        }
    )
    source_metadata = json.loads(source["metadata"].read_text(encoding="utf-8"))
    wan_vae = source_metadata.get("environment", {}).get("WAN_VAE_PATH")
    if wan_vae:
        environment["WAN_VAE_PATH"] = str(wan_vae)
    if args.hf_home:
        environment["HF_HOME"] = str(args.hf_home.expanduser().resolve())
    metadata_path = experiment / "metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "method": "cosmos3_completed_sft_single_process_export_recovery",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cause": "training CP=2 is invalid for the official single-process exporter",
        "repair": (
            "explicit --cp-size 1 plus an export-only config copy with model CP=1; "
            "weights and source training config unchanged"
        ),
        "source_training_experiment": str(source["source"]),
        "source_training_metadata_sha256": _sha256(source["metadata"]),
        "source_training_log_sha256": _sha256(source["training_log"]),
        "source_checkpoint": str(source["checkpoint"]),
        "source_checkpoint_inventory": _tree_inventory(source["checkpoint"]),
        "source_resolved_config": str(source_config_copy),
        "source_resolved_config_sha256": _sha256(source_config_copy),
        "export_config": str(config),
        "export_config_sha256": _sha256(config),
        "command": command,
        "command_shell": shlex.join(command),
        "environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "HF_HUB_OFFLINE",
                "PYTHONHASHSEED",
                "PYTORCH_CUDA_ALLOC_CONF",
                "WAN_VAE_PATH",
            )
            if key in environment
        },
        "gpu_inventory": inventory,
        "selected_gpus": selected,
        "framework": {"path": str(framework), "commit": commit},
        "project_git": project_provenance(
            args.project_source_revision, args.project_source_branch
        ),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "launcher_package_versions": package_inventory(),
        "cosmos_package_versions": _external_package_inventory(python),
    }
    _write_json(metadata_path, metadata)
    (experiment / "command.txt").write_text(
        metadata["command_shell"] + "\n", encoding="utf-8"
    )
    log_path = experiment / "export.log"
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
                f"Cosmos3 export recovery failed with exit code {completed.returncode}; "
                f"inspect {log_path}"
            )
        _require_file(model / "config.json", "exported config")
        _require_file(model / "checkpoint.json", "exported checkpoint metadata")
        if not list(model.glob("*.safetensors")):
            raise RuntimeError("export produced no safetensors weights")
        tokenizer_bundle = bundle_text_tokenizer(
            args.text_tokenizer_root,
            model,
            args.text_tokenizer_vocab_sha256,
        )
        revision = (
            "phiagent:cosmos3-droid-wrist-lora:iter_000000500:"
            f"{_sha256(config)[:8]}:{tokenizer_bundle['vocab_sha256'][:8]}"
        )
        (model / ".phiagent-model-revision").write_text(
            revision + "\n", encoding="utf-8"
        )
        (model / ".phiagent-model-source").write_text(
            f"dcp:{source['checkpoint']}\n", encoding="utf-8"
        )
        metadata.update(
            {
                "status": "succeeded",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "model": str(model),
                "model_revision": revision,
                "text_tokenizer_bundle": tokenizer_bundle,
                "model_inventory": _tree_inventory(model),
                "export_log_sha256": _sha256(log_path),
            }
        )
        _write_json(metadata_path, metadata)
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
                "export_log_sha256": _sha256(log_path) if log_path.exists() else None,
            }
        )
        _write_json(metadata_path, metadata)
        raise
    print(json.dumps({"experiment": str(experiment), "model": str(model)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
