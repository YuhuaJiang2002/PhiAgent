#!/usr/bin/env python3
"""Prepare pinned, isolated external AC-WM repositories and checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.adapters import (  # noqa: E402
    BWM_BASE_MODEL_REVISION,
    BWM_MODEL_REVISION,
    BWM_MODEL_BYTES,
    BWM_MODEL_SHA256,
    BWM_REPOSITORY_COMMIT,
    FLOWWAM_MODEL_BYTES,
    FLOWWAM_MODEL_REVISION,
    FLOWWAM_MODEL_SHA256,
    FLOWWAM_REPOSITORY_COMMIT,
    KINEMA4D_MODEL_REVISION,
    KINEMA4D_REPOSITORY_COMMIT,
    OSCAR_COSMOS_REASON_REVISION,
    OSCAR_MODEL_REVISION,
    OSCAR_REPOSITORY_COMMIT,
    OSCAR_WAN_VAE_REVISION,
)

REPOSITORIES = {
    "oscar": (
        "git@github.com:wuzy2115/oscar-public.git",
        "oscar",
        OSCAR_REPOSITORY_COMMIT,
    ),
    "bwm": (
        "git@github.com:boundless-large-model/boundless-world-model.git",
        "boundless-world-model",
        BWM_REPOSITORY_COMMIT,
    ),
    "kinema4d": (
        "git@github.com:mutianxu/Kinema4D.git",
        "Kinema4D",
        KINEMA4D_REPOSITORY_COMMIT,
    ),
    "flowwam": (
        "git@github.com:YixiangChen515/FlowWAM_WorldArena.git",
        "FlowWAM_WorldArena",
        FLOWWAM_REPOSITORY_COMMIT,
    ),
}


def _run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_commit(repository: Path, commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _prepare_repository(backend: str, external_root: Path) -> Path:
    url, directory, commit = REPOSITORIES[backend]
    repository = external_root / directory
    if not repository.exists():
        _run(["git", "clone", "--filter=blob:none", url, str(repository)])
    if not (repository / ".git").is_dir():
        marker = repository / ".phiagent-source-revision"
        revision = marker.read_text().strip() if marker.is_file() else None
        if revision != commit:
            raise RuntimeError(
                "external source archive lacks the expected revision marker: "
                f"{repository} (found {revision!r}, expected {commit!r})"
            )
        return repository
    if not _has_commit(repository, commit):
        _run(["git", "fetch", "origin", commit], cwd=repository)
    _run(["git", "checkout", "--detach", commit], cwd=repository)
    return repository


def _uv() -> str:
    executable = shutil.which("uv")
    if executable is None:
        raise RuntimeError("uv is required for isolated AC-WM environments")
    return executable


def _install(backend: str, repository: Path) -> None:
    uv = _uv()
    environment = repository / ".venv"
    _run([uv, "venv", "--python", "3.10", str(environment)], cwd=repository)
    python = environment / "bin" / "python"
    if backend == "oscar":
        _run(
            [uv, "pip", "install", "--python", str(python), "-r", "requirements_minimal.txt"],
            cwd=repository,
        )
    elif backend == "bwm":
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
                "torch==2.8.0",
                "torchvision==0.23.0",
                "torchaudio==2.8.0",
            ],
            cwd=repository,
        )
        _run([uv, "pip", "install", "--python", str(python), "diffsynth==2.0.11"], cwd=repository)
        _run(
            [uv, "pip", "install", "--python", str(python), "-r", "requirements.txt"],
            cwd=repository,
        )
        # DiffSynth 2.0.11 has no Transformers upper bound.  On 2026-08-11 the
        # resolver selected Transformers 5.15, which requires safetensors>=0.8
        # and conflicts with BWM's exact safetensors==0.7 runtime.  Pin the
        # last project-validated Transformers stack explicitly.
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "transformers==4.51.3",
                "tokenizers==0.21.1",
                "huggingface-hub==0.36.2",
                "safetensors==0.7.0",
            ],
            cwd=repository,
        )
        _run(
            [
                str(python),
                "-c",
                "import torch, diffsynth, transformers, safetensors; "
                "assert torch.__version__.startswith('2.8.0'); "
                "assert transformers.__version__ == '4.51.3'; "
                "assert safetensors.__version__ == '0.7.0'",
            ],
            cwd=repository,
        )
    elif backend == "kinema4d":
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
                "torch==2.4.0",
                "torchvision==0.19.0",
                "torchaudio==2.4.0",
            ],
            cwd=repository,
        )
        _run(
            [uv, "pip", "install", "--python", str(python), "-r", "requirements.txt"],
            cwd=repository,
        )
    else:
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
                "torch==2.3.0",
                "torchvision==0.18.0",
            ],
            cwd=repository,
        )
        _run(
            [uv, "pip", "install", "--python", str(python), "-r", "requirements.txt"],
            cwd=repository,
        )


def _hf() -> str:
    executable = shutil.which("hf")
    if executable is None:
        raise RuntimeError("the Hugging Face `hf` CLI is required for model downloads")
    return executable


def _download_model(backend: str, checkpoint_root: Path) -> Path:
    hf = _hf()
    if backend == "oscar":
        destination = checkpoint_root / "OSCAR-2B"
        _run(
            [
                hf,
                "download",
                "zywu2115/OSCAR-2B",
                "--revision",
                OSCAR_MODEL_REVISION,
                "--local-dir",
                str(destination),
            ]
        )
        runtime = checkpoint_root / "OSCAR-runtime"
        cosmos = runtime / "Cosmos-Reason1-7B"
        _run(
            [
                hf,
                "download",
                "nvidia/Cosmos-Reason1-7B",
                "--revision",
                OSCAR_COSMOS_REASON_REVISION,
                "--local-dir",
                str(cosmos),
            ]
        )
        cosmos.mkdir(parents=True, exist_ok=True)
        (cosmos / ".phiagent-model-revision").write_text(OSCAR_COSMOS_REASON_REVISION + "\n")
        runtime.mkdir(parents=True, exist_ok=True)
        _run(
            [
                hf,
                "download",
                "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth",
                "--revision",
                OSCAR_WAN_VAE_REVISION,
                "--local-dir",
                str(runtime),
            ]
        )
        (runtime / ".phiagent-wan-vae-revision").write_text(OSCAR_WAN_VAE_REVISION + "\n")
        revision = OSCAR_MODEL_REVISION
    elif backend == "bwm":
        destination = checkpoint_root / "BWM"
        checkpoint = destination / "step-12000.safetensors"
        if not checkpoint.is_file() or checkpoint.stat().st_size != BWM_MODEL_BYTES:
            _run(
                [
                    hf,
                    "download",
                    "BLM-Lab/Boundless-World-Model",
                    "step-12000.safetensors",
                    "--revision",
                    BWM_MODEL_REVISION,
                    "--local-dir",
                    str(destination),
                ]
            )
        if checkpoint.stat().st_size != BWM_MODEL_BYTES:
            raise RuntimeError(
                f"BWM checkpoint has {checkpoint.stat().st_size} bytes, "
                f"expected {BWM_MODEL_BYTES}"
            )
        actual_sha256 = _sha256(checkpoint)
        if actual_sha256 != BWM_MODEL_SHA256:
            raise RuntimeError(
                f"BWM checkpoint hash is {actual_sha256}, expected {BWM_MODEL_SHA256}"
            )
        (destination / ".phiagent-model-verification.json").write_text(
            json.dumps(
                {
                    "file": checkpoint.name,
                    "bytes": checkpoint.stat().st_size,
                    "sha256": actual_sha256,
                    "revision": BWM_MODEL_REVISION,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        revision = BWM_MODEL_REVISION
    elif backend == "kinema4d":
        destination = checkpoint_root / "Kinema4D"
        _run(
            [
                hf,
                "download",
                "Minoday/Kinema4D",
                "kinema4d_pmcond_ckpt",
                "--revision",
                KINEMA4D_MODEL_REVISION,
                "--local-dir",
                str(destination),
            ]
        )
        revision = KINEMA4D_MODEL_REVISION
    else:
        destination = checkpoint_root / "FlowWAM"
        checkpoint = destination / "flowwam_worldarena_stage1.safetensors"
        if not checkpoint.is_file() or checkpoint.stat().st_size != FLOWWAM_MODEL_BYTES:
            _run(
                [
                    hf,
                    "download",
                    "YixiangChen/FlowWAM",
                    checkpoint.name,
                    "--revision",
                    FLOWWAM_MODEL_REVISION,
                    "--local-dir",
                    str(destination),
                ]
            )
        if checkpoint.stat().st_size != FLOWWAM_MODEL_BYTES:
            raise RuntimeError(
                f"FlowWAM checkpoint has {checkpoint.stat().st_size} bytes, "
                f"expected {FLOWWAM_MODEL_BYTES}"
            )
        actual_sha256 = _sha256(checkpoint)
        if actual_sha256 != FLOWWAM_MODEL_SHA256:
            raise RuntimeError(
                f"FlowWAM checkpoint hash is {actual_sha256}, "
                f"expected {FLOWWAM_MODEL_SHA256}"
            )
        (destination / ".phiagent-model-verification.json").write_text(
            json.dumps(
                {
                    "file": checkpoint.name,
                    "bytes": checkpoint.stat().st_size,
                    "sha256": actual_sha256,
                    "revision": FLOWWAM_MODEL_REVISION,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        revision = FLOWWAM_MODEL_REVISION
    destination.mkdir(parents=True, exist_ok=True)
    (destination / ".phiagent-model-revision").write_text(revision + "\n")
    return destination


def _download_bwm_base_model(checkpoint_root: Path) -> Path:
    hf = _hf()
    destination = checkpoint_root / "Wan2.2-TI2V-5B"
    _run(
        [
            hf,
            "download",
            "Wan-AI/Wan2.2-TI2V-5B",
            "--revision",
            BWM_BASE_MODEL_REVISION,
            "--local-dir",
            str(destination),
        ]
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / ".phiagent-model-revision").write_text(BWM_BASE_MODEL_REVISION + "\n")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("oscar", "bwm", "kinema4d", "flowwam", "all"),
        default="all",
    )
    parser.add_argument("--external-root", type=Path, default=Path("external"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument(
        "--download-base-model",
        action="store_true",
        help="also download BWM's pinned Wan2.2-TI2V-5B base model",
    )
    args = parser.parse_args()
    external_root = args.external_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    external_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    backends = tuple(REPOSITORIES) if args.backend == "all" else (args.backend,)
    for backend in backends:
        repository = _prepare_repository(backend, external_root)
        if args.install:
            _install(backend, repository)
        checkpoint = _download_model(backend, checkpoint_root) if args.download_model else None
        print(f"{backend.upper()}_REPOSITORY={repository}")
        if checkpoint is not None:
            print(f"{backend.upper()}_CHECKPOINT={checkpoint}")
            if backend == "oscar":
                runtime = checkpoint_root / "OSCAR-runtime"
                print(f"OSCAR_COSMOS_REASON={runtime / 'Cosmos-Reason1-7B'}")
                print(f"OSCAR_WAN_VAE={runtime / 'Wan2.1_VAE.pth'}")
        if args.download_base_model and backend == "bwm":
            base_model = _download_bwm_base_model(checkpoint_root)
            print(f"BWM_BASE_MODEL={base_model}")
    if args.download_base_model and "bwm" not in backends:
        parser.error("--download-base-model requires --backend bwm or all")
    if "bwm" in backends:
        print(
            "BWM also requires the external Wan2.2-TI2V-5B base model and training action statistics."
        )
    if "kinema4d" in backends:
        print("Kinema4D also requires Wan2.1-I2V-14B and a prepared RGB+pointmap episode.")
    if "flowwam" in backends:
        print(
            "FlowWAM also requires pinned Wan2.2-TI2V-5B plus the Wan2.1 "
            "UMT5 tokenizer, and a calibrated robot-only flow producer."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
