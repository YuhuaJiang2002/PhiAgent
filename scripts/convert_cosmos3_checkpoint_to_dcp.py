#!/usr/bin/env python3
"""Convert a verified pinned Cosmos3 Hugging Face snapshot into DCP format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-repo", type=Path, required=True)
    parser.add_argument("--expected-framework-commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument("--sound-tokenizer-sha256", required=True)
    parser.add_argument("--text-tokenizer-vocab-sha256", required=True)
    parser.add_argument("--wan-vae", type=Path, required=True)
    parser.add_argument("--wan-vae-sha256", required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--project-source-revision")
    parser.add_argument("--project-source-branch")
    parser.add_argument("--preflight-only", action="store_true")
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


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_verification(
    report: dict[str, Any], checkpoint: Path, expected_revision: str
) -> None:
    nano = report.get("nano")
    if report.get("status") != "WORKING" or not isinstance(nano, dict):
        raise ValueError("checkpoint verification is not WORKING")
    if nano.get("status") != "WORKING":
        raise ValueError("Nano checkpoint verification is not WORKING")
    if Path(str(nano.get("checkpoint", ""))).resolve() != checkpoint.resolve():
        raise ValueError("verification does not bind the supplied Nano checkpoint")
    if nano.get("revision") != expected_revision:
        raise ValueError("verification does not bind the expected Nano revision")
    indexes = nano.get("indexes")
    if not isinstance(indexes, list) or not indexes:
        raise ValueError("verification contains no indexed weight inventory")
    if any(
        row.get("actual_total_size_bytes") != row.get("expected_total_size_bytes")
        for row in indexes
    ):
        raise ValueError("verification contains an indexed byte-count mismatch")


def validate_file_sha256(path: Path, expected_sha256: str, label: str) -> str:
    resolved = _require_file(path, label)
    actual_sha256 = _sha256(resolved)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256


def build_conversion_command(python: Path, checkpoint: Path, output: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "cosmos_framework.scripts.convert_model_to_dcp",
        "--checkpoint-path",
        str(checkpoint),
        "-o",
        str(output),
    ]


def stage_local_first_checkpoint(
    checkpoint: Path,
    staging_root: Path,
    expected_vocab_sha256: str,
    wan_vae: Path,
    expected_wan_vae_sha256: str,
) -> dict[str, Any]:
    """Stage a symlink-only snapshot whose config uses bundled tokenizer files."""
    source = checkpoint.expanduser().resolve()
    staging = staging_root.expanduser().absolute()
    if staging.exists():
        raise FileExistsError(f"local-first staging directory already exists: {staging}")
    vocab = source / "vocab.json"
    vocab_sha256 = validate_file_sha256(
        vocab,
        expected_vocab_sha256,
        "pinned Cosmos3 text-tokenizer vocabulary",
    )
    vae = _require_file(wan_vae, "pinned local Wan2.2 VAE")
    wan_vae_sha256 = validate_file_sha256(
        vae,
        expected_wan_vae_sha256,
        "pinned local Wan2.2 VAE",
    )
    tokenizer_required = [
        source / "merges.txt",
        source / "tokenizer.json",
        source / "tokenizer_config.json",
    ]
    for path in tokenizer_required:
        _require_file(path, f"bundled text-tokenizer file {path.name}")
    staging.mkdir(parents=True)
    for child in sorted(source.iterdir()):
        if child.name in {".cache", "config.json"}:
            continue
        (staging / child.name).symlink_to(child, target_is_directory=child.is_dir())
    source_config = _require_file(source / "config.json", "Cosmos3 model config")
    config = json.loads(source_config.read_text(encoding="utf-8"))
    try:
        tokenizer = config["model"]["config"]["vlm_config"]["tokenizer"]
        vision_tokenizer = config["model"]["config"]["tokenizer"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Cosmos3 config has no VLM tokenizer node") from exc
    if tokenizer.get("_target") != "create_qwen2_tokenizer_with_download":
        raise ValueError("Cosmos3 config has an unsupported VLM tokenizer target")
    tokenizer["config_variant"] = "hf"
    tokenizer["pretrained_model_name"] = str(source)
    if vision_tokenizer.get("_target") != "wan2pt2_vae_interface":
        raise ValueError("Cosmos3 config has an unsupported vision tokenizer target")
    # The official Wan tokenizer prepends ``s3://{bucket_name}/`` whenever
    # bucket_name is non-empty.  Clearing the object-store settings is therefore
    # required as well as replacing vae_path; otherwise an absolute local path is
    # silently reinterpreted as an S3 key and conversion asks for GCP credentials.
    vision_tokenizer["bucket_name"] = ""
    vision_tokenizer["object_store_credential_path_pretrained"] = ""
    vision_tokenizer["vae_path"] = str(vae)
    staged_config = staging / "config.json"
    _write_json(staged_config, config)
    tokenizer_inventory = [
        {
            "path": path.relative_to(source).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in [vocab, *tokenizer_required]
    ]
    return {
        "path": str(staging),
        "method": "symlink_snapshot_with_local_text_and_vision_tokenizers",
        "source_checkpoint": str(source),
        "source_config_sha256": _sha256(source_config),
        "staged_config_sha256": _sha256(staged_config),
        "text_tokenizer_root": str(source),
        "vocab_sha256": vocab_sha256,
        "wan_vae": str(vae),
        "wan_vae_sha256": wan_vae_sha256,
        "tokenizer_files": tokenizer_inventory,
        "weight_payloads_copied": False,
    }


def _external_packages(python: Path) -> str:
    uv = python.parent / "uv"
    command = (
        [str(uv), "pip", "freeze", "--python", str(python)]
        if uv.is_file()
        else [str(python), "-m", "pip", "freeze", "--all"]
    )
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    packages = sorted(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()),
        key=str.casefold,
    )
    if not packages:
        raise RuntimeError(f"could not inventory Cosmos packages with {python}")
    return "\n".join(packages) + "\n"


def _new_experiment(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = resolved / f"{timestamp}-{uuid.uuid4().hex[:8]}"
    experiment.mkdir()
    return experiment


def _dcp_inventory(root: Path) -> list[dict[str, Any]]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            **({"sha256": _sha256(path)} if path.stat().st_size <= 64 * 1024 * 1024 else {}),
        }
        for path in files
    ]


def main() -> int:
    args = _parser().parse_args()
    framework = _require_dir(args.framework_repo, "Cosmos Framework checkout")
    commit = _git(framework, "rev-parse", "HEAD")
    if commit != args.expected_framework_commit:
        raise ValueError(f"Cosmos Framework is {commit}; expected {args.expected_framework_commit}")
    checkpoint = _require_dir(args.checkpoint, "verified Cosmos3 checkpoint")
    revision = _require_file(
        checkpoint / ".phiagent-model-revision", "model revision marker"
    ).read_text(encoding="utf-8").strip()
    if revision != args.expected_model_revision:
        raise ValueError(f"checkpoint is {revision}; expected {args.expected_model_revision}")
    sound_tokenizer = checkpoint / "sound_tokenizer/diffusion_pytorch_model.safetensors"
    sound_tokenizer_sha256 = validate_file_sha256(
        sound_tokenizer,
        args.sound_tokenizer_sha256,
        "Cosmos3 sound tokenizer required by official DCP conversion",
    )
    verification_path = _require_file(args.verification, "checkpoint verification")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    validate_verification(verification, checkpoint, revision)
    python = (
        require_executable(args.python_executable, "Cosmos Python")
        if args.python_executable
        else require_executable(framework / ".venv/bin/python", "Cosmos Python")
    )
    experiment = _new_experiment(args.experiment_root)
    staging = experiment / "local-first-hf"
    staging_metadata = stage_local_first_checkpoint(
        checkpoint,
        staging,
        args.text_tokenizer_vocab_sha256,
        args.wan_vae,
        args.wan_vae_sha256,
    )
    output = experiment / "dcp"
    command = build_conversion_command(python, staging, output)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(framework),
            "PATH": str(python.parent) + os.pathsep + environment.get("PATH", ""),
            "HF_HUB_OFFLINE": "1",
        }
    )
    if args.hf_home:
        environment["HF_HOME"] = str(args.hf_home.expanduser().resolve())
    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "preflight" if args.preflight_only else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "official_cosmos3_hf_to_dcp_conversion",
        "seed": None,
        "framework": {"path": str(framework), "commit": commit},
        "checkpoint": str(checkpoint),
        "local_first_staging": staging_metadata,
        "model_revision": revision,
        "sound_tokenizer": str(sound_tokenizer),
        "sound_tokenizer_sha256": sound_tokenizer_sha256,
        "verification": str(verification_path),
        "verification_sha256": _sha256(verification_path),
        "output": str(output),
        "command": command,
        "command_shell": shlex.join(command),
        "environment": {
            key: environment[key]
            for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "HF_HOME", "HF_HUB_OFFLINE")
            if key in environment
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "project_source_revision": args.project_source_revision,
        "project_source_branch": args.project_source_branch,
        "launcher_sha256": _sha256(Path(__file__).resolve()),
        "launcher_package_versions": package_inventory(),
        "cosmos_package_versions": _external_packages(python),
    }
    metadata_path = experiment / "metadata.json"
    _write_json(metadata_path, metadata)
    (experiment / "command.txt").write_text(metadata["command_shell"] + "\n")
    if args.preflight_only:
        print(json.dumps({"experiment": str(experiment), "status": "preflight"}))
        return 0
    log_path = experiment / "conversion.log"
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
                f"Cosmos3 DCP conversion failed with exit code {completed.returncode}; "
                f"inspect {log_path}"
            )
        inventory = _dcp_inventory(output)
        if not inventory:
            raise RuntimeError("DCP conversion returned success but wrote no files")
        metadata.update(
            {
                "status": "succeeded",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "dcp_inventory": inventory,
                "dcp_total_bytes": sum(row["size_bytes"] for row in inventory),
            }
        )
        _write_json(metadata_path, metadata)
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": repr(error),
            }
        )
        _write_json(metadata_path, metadata)
        raise
    print(json.dumps({"experiment": str(experiment), "dcp": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
