#!/usr/bin/env python3
"""Download pinned JoyAI and MiMo snapshots into an ignored checkpoint root."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import inspect
import json
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.joyai_video_edit import (  # noqa: E402
    JOYAI_MODEL_ID,
    JOYAI_MODELSCOPE_MODEL_ID,
    JOYAI_MODELSCOPE_MODEL_REVISION,
    JOYAI_MODEL_REVISION,
    JOYAI_TEXT_ENCODER_ID,
    JOYAI_TEXT_ENCODER_MODELSCOPE_REVISION,
    JOYAI_TEXT_ENCODER_REVISION,
    validate_checkpoint_layout,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers-per-model", type=int, default=8)
    parser.add_argument(
        "--model-provider",
        choices=("huggingface", "modelscope"),
        default="huggingface",
    )
    parser.add_argument(
        "--text-encoder-provider",
        choices=("huggingface", "modelscope"),
        default="huggingface",
    )
    parser.add_argument(
        "--component",
        choices=("both", "model", "text-encoder"),
        default="both",
        help="Download both snapshots or resume one component independently.",
    )
    return parser


def _snapshot_download(repo_id: str, revision: str, local_dir: Path, workers: int) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional download environment
        raise RuntimeError("huggingface-hub is required only by the checkpoint downloader") from exc
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "local_dir": str(local_dir),
        "max_workers": workers,
        "resume_download": True,
    }
    signature = inspect.signature(snapshot_download)
    if "local_dir_use_symlinks" in signature.parameters:
        kwargs["local_dir_use_symlinks"] = False
    return str(snapshot_download(**kwargs))


def _modelscope_snapshot_download(
    repo_id: str, revision: str, local_dir: Path, workers: int
) -> str:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional download environment
        raise RuntimeError(
            "modelscope is required only when a ModelScope provider is selected"
        ) from exc
    return str(
        snapshot_download(
            model_id=repo_id,
            revision=revision,
            local_dir=str(local_dir),
            max_workers=workers,
        )
    )


def _download(
    provider: str, repo_id: str, revision: str, local_dir: Path, workers: int
) -> str:
    if provider == "huggingface":
        return _snapshot_download(repo_id, revision, local_dir, workers)
    if provider == "modelscope":
        return _modelscope_snapshot_download(repo_id, revision, local_dir, workers)
    raise ValueError(f"unsupported provider: {provider}")


def _git_state() -> dict[str, Any]:
    state: dict[str, Any] = {}
    for label, command in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        state[label] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    return state


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("huggingface-hub", "modelscope"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main() -> int:
    args = _parser().parse_args()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"JoyAI download experiment already exists: {output}")
    if args.max_workers_per_model < 1:
        raise ValueError("max workers must be positive")
    output.mkdir(parents=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    model_snapshot = (
        {
            "provider": "huggingface",
            "repo_id": JOYAI_MODEL_ID,
            "revision": JOYAI_MODEL_REVISION,
        }
        if args.model_provider == "huggingface"
        else {
            "provider": "modelscope",
            "repo_id": JOYAI_MODELSCOPE_MODEL_ID,
            "revision": JOYAI_MODELSCOPE_MODEL_REVISION,
        }
    )
    text_snapshot = (
        {
            "provider": "huggingface",
            "repo_id": JOYAI_TEXT_ENCODER_ID,
            "revision": JOYAI_TEXT_ENCODER_REVISION,
        }
        if args.text_encoder_provider == "huggingface"
        else {
            "provider": "modelscope",
            "repo_id": JOYAI_TEXT_ENCODER_ID,
            "revision": JOYAI_TEXT_ENCODER_MODELSCOPE_REVISION,
        }
    )
    model_snapshot["local_dir"] = str(checkpoint_root / "JoyAI-Video-Edit")
    text_snapshot["local_dir"] = str(checkpoint_root / "MiMo-VL-7B-RL-2508")
    selected_snapshots = {
        "both": [model_snapshot, text_snapshot],
        "model": [model_snapshot],
        "text-encoder": [text_snapshot],
    }[args.component]
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "stage": "download_started",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": {"executable": sys.executable, "version": sys.version},
        "packages": _package_versions(),
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "git": _git_state(),
        "seed": None,
        "checkpoint_root": str(checkpoint_root),
        "component": args.component,
        "snapshots": selected_snapshots,
        "error": None,
    }
    write_json(output / "manifest.json", manifest)
    try:
        jobs = selected_snapshots
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    _download,
                    str(job["provider"]),
                    str(job["repo_id"]),
                    str(job["revision"]),
                    Path(str(job["local_dir"])),
                    args.max_workers_per_model,
                ): f"{job['provider']}:{job['repo_id']}"
                for job in jobs
            }
            resolved = {futures[future]: future.result() for future in concurrent.futures.as_completed(futures)}
        if args.component in {"both", "model"}:
            (checkpoint_root / "JoyAI-Video-Edit/.phiagent-model-revision").write_text(
                (
                    JOYAI_MODEL_REVISION
                    if args.model_provider == "huggingface"
                    else f"modelscope:{JOYAI_MODELSCOPE_MODEL_REVISION}"
                )
                + "\n",
                encoding="utf-8",
            )
        if args.component in {"both", "text-encoder"}:
            (checkpoint_root / "MiMo-VL-7B-RL-2508/.phiagent-model-revision").write_text(
                (
                    JOYAI_TEXT_ENCODER_REVISION
                    if args.text_encoder_provider == "huggingface"
                    else f"modelscope:{JOYAI_TEXT_ENCODER_MODELSCOPE_REVISION}"
                )
                + "\n",
                encoding="utf-8",
            )
        validation = (
            validate_checkpoint_layout(
                checkpoint_root,
                verify_large_hashes=True,
            )
            if args.component == "both"
            else None
        )
        manifest.update(
            {
                "status": "WORKING",
                "stage": (
                    "download_validated"
                    if validation is not None
                    else "component_download_completed"
                ),
                "component": args.component,
                "resolved_snapshots": resolved,
                "validation": validation,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"experiment": str(output), "status": "WORKING", "validation": validation}, indent=2))
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "PARTIAL",
                "stage": "download_or_validation_failed",
                "error": repr(exc),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
