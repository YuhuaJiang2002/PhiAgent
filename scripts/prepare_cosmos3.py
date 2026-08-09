#!/usr/bin/env python3
"""Prepare the pinned Cosmos Framework checkout without downloading it on import."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.cosmos3 import (  # noqa: E402
    COSMOS3_FRAMEWORK_COMMIT,
    COSMOS3_MODEL_REVISION,
)

FRAMEWORK_URL = "https://github.com/NVIDIA/cosmos-framework.git"
MINIMUM_UV_VERSION = (0, 11, 3)
MODELSCOPE_MODEL_ID = "nv-community/Cosmos3-Nano"
MODELSCOPE_MODEL_REVISION = "9acbac5b493bc396174ee6dac2367ad664871cbe"


def _run(command: Sequence[str], cwd: Path | None = None) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)


def _has_commit(repo: Path, commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _uv_executable() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for --install; install it from https://astral.sh/uv")
    output = subprocess.run(
        [uv, "--version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    try:
        version = tuple(int(part) for part in output.split()[1].split(".")[:3])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"could not parse uv version from {output!r}") from exc
    if version < MINIMUM_UV_VERSION:
        required = ".".join(str(part) for part in MINIMUM_UV_VERSION)
        raise RuntimeError(f"uv>={required} is required, found {output}")
    return uv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, default=Path("external"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--cuda-group", choices=("cu128-train", "cu130-train"), default="cu128-train")
    parser.add_argument("--source", choices=("huggingface", "modelscope"), default="huggingface")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--download-workers", type=int, default=4)
    args = parser.parse_args()
    if args.download_workers <= 0:
        parser.error("--download-workers must be positive")

    external_root = args.external_root.expanduser().resolve()
    external_root.mkdir(parents=True, exist_ok=True)
    repo = external_root / "cosmos-framework"
    if not repo.exists():
        _run(["git", "clone", "--filter=blob:none", FRAMEWORK_URL, str(repo)])
    if not (repo / ".git").is_dir():
        raise RuntimeError(f"existing path is not a Git checkout: {repo}")
    if not _has_commit(repo, COSMOS3_FRAMEWORK_COMMIT):
        _run(["git", "fetch", "origin", COSMOS3_FRAMEWORK_COMMIT], cwd=repo)
    _run(["git", "checkout", "--detach", COSMOS3_FRAMEWORK_COMMIT], cwd=repo)

    if args.install:
        uv = _uv_executable()
        _run([uv, "sync", "--all-extras", f"--group={args.cuda_group}"], cwd=repo)

    checkpoint = args.checkpoint_root.expanduser().resolve() / "Cosmos3-Nano"
    if args.download_model:
        checkpoint.mkdir(parents=True, exist_ok=True)
        if args.source == "huggingface":
            cli = shutil.which("hf")
            if cli is None:
                raise RuntimeError(
                    "the Hugging Face `hf` CLI is required for --source huggingface"
                )
            command = [
                cli,
                "download",
                "nvidia/Cosmos3-Nano",
                "--revision",
                COSMOS3_MODEL_REVISION,
            ]
            source_marker = f"huggingface:{COSMOS3_MODEL_REVISION}"
        else:
            cli = shutil.which("modelscope")
            if cli is None:
                raise RuntimeError(
                    "the ModelScope CLI is required for --source modelscope"
                )
            command = [
                cli,
                "download",
                MODELSCOPE_MODEL_ID,
                "--revision",
                MODELSCOPE_MODEL_REVISION,
                "--force",
            ]
            source_marker = f"modelscope:{MODELSCOPE_MODEL_REVISION}"
        command.extend(
            [
                "--local-dir",
                str(checkpoint),
                "--max-workers",
                str(args.download_workers),
            ]
        )
        _run(command)
        (checkpoint / ".phiagent-model-revision").write_text(
            COSMOS3_MODEL_REVISION + "\n"
        )
        (checkpoint / ".phiagent-model-source").write_text(source_marker + "\n")

    print(f"COSMOS3_REPO={repo}")
    print(f"COSMOS3_CHECKPOINT={checkpoint}")
    print(f"COSMOS3_MODEL_REVISION={COSMOS3_MODEL_REVISION}")
    if not args.install:
        print(
            f"Run `uv sync --all-extras --group={args.cuda_group}` in {repo} "
            "before GPU inference."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
