#!/usr/bin/env python3
"""Precompute DROID prompt embeddings with an audited physical GPU selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_cosmos_predict2_droid_inference import (  # noqa: E402
    load_t5_embedding,
    query_physical_gpus,
    validate_gpu_selection,
)
from scripts.experiment_provenance import package_inventory  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--negative-prompt-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--min-free-memory-mib", type=int, default=30_000)
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


def _embedding_manifest(dataset: Path) -> list[dict[str, object]]:
    videos = {path.stem for path in (dataset / "videos").glob("*.mp4")}
    metas = {path.stem for path in (dataset / "metas").glob("*.txt")}
    embeddings = {
        path.stem: path for path in (dataset / "t5_xxl").glob("*.pickle") if path.stat().st_size
    }
    if not videos or videos != metas or videos != set(embeddings):
        raise ValueError(
            "video/meta/T5 IDs differ after embedding: "
            f"videos={len(videos)}, metas={len(metas)}, embeddings={len(embeddings)}"
        )
    result = []
    for sample_id in sorted(videos):
        path = embeddings[sample_id]
        with path.open("rb") as handle:
            value = pickle.load(handle)
        if (
            not isinstance(value, list)
            or len(value) != 1
            or getattr(value[0], "ndim", None) != 2
            or value[0].shape[1] != 1024
        ):
            raise ValueError(f"invalid T5 embedding contract: {path}")
        result.append(
            {
                "sample_id": sample_id,
                "path": str(path),
                "shape": list(value[0].shape),
                "dtype": str(value[0].dtype),
                "sha256": _sha256(path),
            }
        )
    return result


def _worker(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from imaginaire.auxiliary.text_encoder import (
        CosmosT5TextEncoder,
        CosmosT5TextEncoderConfig,
    )

    metas = sorted((args.dataset / "metas").glob("*.txt"))
    if not metas:
        raise ValueError("dataset contains no prompt metadata")
    embedding_dir = args.dataset / "t5_xxl"
    embedding_dir.mkdir(exist_ok=True)
    encoder = CosmosT5TextEncoder(
        config=CosmosT5TextEncoderConfig(ckpt_path=str(args.text_encoder)),
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    def encode_to_pickle(prompt: str, destination: Path) -> None:
        if destination.is_file() and destination.stat().st_size:
            return
        if not prompt:
            raise ValueError(f"empty prompt for {destination}")
        encoded, mask = encoder.encode_prompts(
            prompt, max_length=512, return_mask=True
        )
        length = int(mask.long().sum(dim=1).cpu()[0])
        payload = [encoded[0, :length].float().cpu().numpy().astype(np.float16)]
        temporary = destination.with_suffix(".pickle.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle)
        temporary.replace(destination)

    for meta in metas:
        encode_to_pickle(meta.read_text().strip(), embedding_dir / f"{meta.stem}.pickle")
    if args.negative_prompt_file is not None:
        encode_to_pickle(
            args.negative_prompt_file.read_text().strip(),
            args.output_dir / "negative-prompt.pickle",
        )
    return 0


def _controller(args: argparse.Namespace) -> int:
    external_repo = _require_dir(args.external_repo, "Cosmos Predict2 repository")
    dataset = _require_dir(args.dataset, "DROID training dataset")
    text_encoder = _require_dir(args.text_encoder, "T5 text encoder")
    negative_prompt_file = (
        _require_file(args.negative_prompt_file, "negative prompt file")
        if args.negative_prompt_file is not None
        else None
    )
    weight = _require_file(text_encoder / "pytorch_model.bin", "T5 weights")
    for name in ("config.json", "spiece.model", "tokenizer.json"):
        _require_file(text_encoder / name, f"T5 {name}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)

    inventory = query_physical_gpus()
    selection = validate_gpu_selection(
        inventory, [args.physical_gpu], args.min_free_memory_mib
    )
    contract = _require_file(dataset.parent / "dataset-contract.json", "dataset contract")
    (output / "packages.txt").write_text(package_inventory())
    _write_json(
        output / "gpu-selection.json",
        {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "inventory_before_launch": inventory,
            "selected_physical_gpus": selection,
            "cuda_visible_devices": str(args.physical_gpu),
            "minimum_free_memory_mib": args.min_free_memory_mib,
        },
    )
    _write_json(
        output / "experiment-config.json",
        {
            "schema_version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "RUNNING",
            "method": "Cosmos T5-11B BF16 prompt embedding",
            "dataset": str(dataset),
            "dataset_contract": str(contract),
            "dataset_contract_sha256": _sha256(contract),
            "text_encoder": {
                "path": str(text_encoder),
                "weight_size": weight.stat().st_size,
                "weight_sha256": _sha256(weight),
                "runtime_precision": "bfloat16",
            },
            "negative_prompt": (
                None
                if negative_prompt_file is None
                else {
                    "path": str(negative_prompt_file),
                    "sha256": _sha256(negative_prompt_file),
                    "embedding_path": str(output / "negative-prompt.pickle"),
                }
            ),
            "git": {
                "commit": args.git_commit or "unresolved",
                "branch": args.git_branch,
                "working_tree_status": "dirty",
                "launcher_sha256": _sha256(Path(__file__)),
            },
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
        },
    )

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--external-repo",
        str(external_repo),
        "--dataset",
        str(dataset),
        "--text-encoder",
        str(text_encoder),
        "--output-dir",
        str(output),
        "--physical-gpu",
        str(args.physical_gpu),
    ]
    if negative_prompt_file is not None:
        command.extend(["--negative-prompt-file", str(negative_prompt_file)])
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTHONPATH"] = str(external_repo)
    with (output / "run.log").open("w") as log_handle:
        completed = subprocess.run(
            command,
            cwd=external_repo,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    manifest = _embedding_manifest(dataset) if completed.returncode == 0 else []
    negative_embedding = output / "negative-prompt.pickle"
    negative_manifest = None
    if completed.returncode == 0 and negative_prompt_file is not None:
        value = load_t5_embedding(negative_embedding)
        negative_manifest = {
            "path": str(negative_embedding),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _sha256(negative_embedding),
        }
    result = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "returncode": completed.returncode,
        "status": "WORKING" if completed.returncode == 0 and manifest else "PARTIAL",
        "embedding_count": len(manifest),
        "embeddings": manifest,
        "negative_prompt_embedding": negative_manifest,
    }
    _write_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "WORKING":
        raise RuntimeError(f"T5 embedding failed; inspect {output / 'run.log'}")
    return 0


def main() -> int:
    args = _parser().parse_args()
    return _worker(args) if args.worker else _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
