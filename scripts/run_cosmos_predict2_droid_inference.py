#!/usr/bin/env python3
"""Run reproducible Cosmos Predict2 DROID inference on selected physical GPUs.

The parent process validates physical GPU state, freezes an experiment record,
sets CUDA_VISIBLE_DEVICES, and launches one distributed worker per selected
GPU.  Worker imports are delayed so the phiagent package remains lightweight.
"""

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
from types import MethodType
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_provenance import package_inventory  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path)
    parser.add_argument("--lora-profile", choices=("attention", "full"), default="attention")
    parser.add_argument("--lora-rank", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path)
    parser.add_argument("--prompt-embedding", type=Path)
    parser.add_argument("--negative-prompt-embedding", type=Path)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--min-free-memory-mib", type=int, default=45_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--master-port", type=int, default=29571)
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


def query_physical_gpus() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != 7:
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        rows.append(
            {
                "physical_index": int(values[0]),
                "uuid": values[1],
                "name": values[2],
                "memory_total_mib": int(values[3]),
                "memory_used_mib": int(values[4]),
                "memory_free_mib": int(values[5]),
                "utilization_gpu_percent": int(values[6]),
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi returned no physical GPUs")
    return rows


def validate_gpu_selection(
    inventory: list[dict[str, Any]], selected: list[int], min_free_memory_mib: int
) -> list[dict[str, Any]]:
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("physical GPU selection must be non-empty and unique")
    if len(selected) not in (1, 2, 4, 8):
        raise ValueError("context parallel GPU count must be 1, 2, 4, or 8")
    by_index = {int(row["physical_index"]): row for row in inventory}
    missing = set(selected) - set(by_index)
    if missing:
        raise ValueError(f"physical GPUs do not exist: {sorted(missing)}")
    result = [by_index[index] for index in selected]
    insufficient = [
        row for row in result if int(row["memory_free_mib"]) < min_free_memory_mib
    ]
    if insufficient:
        details = [(row["physical_index"], row["memory_free_mib"]) for row in insufficient]
        raise RuntimeError(
            f"selected GPUs below {min_free_memory_mib} MiB free: {details}"
        )
    return result


def extract_lora_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
    """Strip the training-model prefix and reject non-adapter tensors."""
    return {
        key.removeprefix("net."): value
        for key, value in raw_state.items()
        if key.startswith("net.") and (".lora_A." in key or ".lora_B." in key)
    }


def scale_lora_state_dict(
    adapter_state: dict[str, Any], scale: float
) -> dict[str, Any]:
    """Scale the LoRA residual exactly once by scaling each B projection."""
    if not 0.0 < scale <= 1.0:
        raise ValueError("LoRA scale must be in (0, 1]")
    if scale == 1.0:
        return adapter_state
    return {
        key: value * scale if ".lora_B." in key else value
        for key, value in adapter_state.items()
    }


def lora_target_modules(profile: str) -> list[str]:
    if profile == "attention":
        return ["q_proj", "k_proj", "v_proj", "output_proj"]
    if profile == "full":
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "output_proj",
            "mlp.layer1",
            "mlp.layer2",
        ]
    raise ValueError(f"unknown LoRA profile: {profile}")


def install_cpu_text_encoding(pipe: Any) -> None:
    """Keep T5 weights and token IDs on CPU, then move only embeddings downstream."""
    if pipe.text_encoder is None:
        raise RuntimeError("pipeline has no text encoder")
    pipe.text_encoder.device = "cpu"

    def encode_prompt_on_cpu(
        pipeline: Any,
        prompts: str | list[str],
        max_length: int | None = None,
        return_mask: bool = False,
    ) -> Any:
        pipeline.text_encoder.device = "cpu"
        pipeline.text_encoder.to(device="cpu")
        return pipeline.text_encoder.encode_prompts(
            prompts,
            max_length=max_length,
            return_mask=return_mask,
        )

    pipe.encode_prompt = MethodType(encode_prompt_on_cpu, pipe)


def load_t5_embedding(path: Path) -> Any:
    """Load the single-array pickle contract emitted by the audited T5 runner."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or getattr(payload[0], "ndim", None) != 2
        or payload[0].shape[0] < 1
        or payload[0].shape[1] != 1024
    ):
        raise ValueError(f"invalid Cosmos T5 embedding contract: {path}")
    return payload[0]


def install_precomputed_text_encoding(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str,
    prompt_embedding: Any,
    negative_prompt_embedding: Any,
) -> None:
    """Serve audited prompt tensors without instantiating T5 in every CP rank."""
    if pipe.text_encoder is not None:
        raise RuntimeError("precomputed text mode requires text_encoder=None")

    def encode_precomputed(
        _pipeline: Any,
        prompts: str | list[str],
        max_length: int | None = None,
        return_mask: bool = False,
    ) -> Any:
        del max_length
        if return_mask:
            raise ValueError("precomputed inference embeddings do not provide a mask")
        if prompts == prompt:
            return prompt_embedding
        if prompts == negative_prompt:
            return negative_prompt_embedding
        raise ValueError("prompt does not match either pinned precomputed embedding")

    pipe.encode_prompt = MethodType(encode_precomputed, pipe)


def _probe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)["streams"][0]


def _worker(args: argparse.Namespace) -> int:
    # Heavy model imports are intentionally isolated to the GPU worker path.
    import torch
    from megatron.core import parallel_state
    from peft import LoraConfig, inject_adapter_in_model

    from cosmos_predict2.configs.base.config_video2world import (
        get_cosmos_predict2_video2world_pipeline,
    )
    from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
    from examples.video2world import _DEFAULT_NEGATIVE_PROMPT
    from examples.video2world_gr00t import process_single_generation
    from imaginaire.utils import distributed, log, misc

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    prompt = args.prompt_file.read_text().strip()
    precomputed_text = args.prompt_embedding is not None
    config = get_cosmos_predict2_video2world_pipeline(
        model_size="14B", resolution="480", fps=16
    )
    config.prompt_refiner_config.enabled = False
    config.guardrail_config.enabled = False
    config.tokenizer.vae_pth = str(args.tokenizer)
    if not precomputed_text:
        config.text_encoder.t5.ckpt_path = str(args.text_encoder)

    misc.set_random_seed(seed=args.seed, by_rank=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if len(args.physical_gpus) > 1:
        distributed.init()
        parallel_state.initialize_model_parallel(
            context_parallel_size=len(args.physical_gpus)
        )
    try:
        log.info(
            "Loading DROID 14B with BF16 CPU-offloaded T5 text encoder; "
            f"checkpoint={args.checkpoint}"
        )
        pipe = Video2WorldPipeline.from_config(
            config=config,
            dit_path=str(args.checkpoint),
            use_text_encoder=not precomputed_text,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=False,
            load_prompt_refiner=False,
            offload_text_encoder=not precomputed_text,
            downcast_text_encoder=not precomputed_text,
        )
        if precomputed_text:
            positive = torch.from_numpy(load_t5_embedding(args.prompt_embedding)).unsqueeze(0)
            negative = torch.from_numpy(
                load_t5_embedding(args.negative_prompt_embedding)
            ).unsqueeze(0)
            install_precomputed_text_encoding(
                pipe,
                prompt=prompt,
                negative_prompt=_DEFAULT_NEGATIVE_PROMPT,
                prompt_embedding=positive,
                negative_prompt_embedding=negative,
            )
            log.info("Using pinned precomputed positive and negative T5 embeddings")
        else:
            install_cpu_text_encoding(pipe)
        if args.lora_checkpoint is not None:
            pipe.dit = inject_adapter_in_model(
                LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=args.lora_rank,
                    init_lora_weights=True,
                    target_modules=lora_target_modules(args.lora_profile),
                ),
                pipe.dit,
            )
            raw_state = torch.load(args.lora_checkpoint, map_location="cpu", weights_only=True)
            adapter_state = extract_lora_state_dict(raw_state)
            if not adapter_state:
                raise RuntimeError("LoRA checkpoint contains no adapter tensors")
            adapter_state = scale_lora_state_dict(adapter_state, args.lora_scale)
            incompatible = pipe.dit.load_state_dict(adapter_state, strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f"unexpected LoRA checkpoint keys: {incompatible.unexpected_keys[:8]}"
                )
            log.info(
                f"Loaded {len(adapter_state)} LoRA adapter tensors at residual scale "
                f"{args.lora_scale} from {args.lora_checkpoint}"
            )
        process_single_generation(
            pipe=pipe,
            input_path=str(args.input_image),
            prompt=prompt,
            output_path=str(args.output_dir / "generated.mp4"),
            negative_prompt=_DEFAULT_NEGATIVE_PROMPT,
            aspect_ratio="16:9",
            num_conditional_frames=1,
            guidance=args.guidance,
            seed=args.seed,
            prompt_prefix="",
        )
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


def _controller(args: argparse.Namespace) -> int:
    if not 0.0 < args.lora_scale <= 1.0:
        raise ValueError("lora-scale must be in (0, 1]")
    if args.lora_checkpoint is None and args.lora_scale != 1.0:
        raise ValueError("non-default lora-scale requires --lora-checkpoint")
    external_repo = _require_dir(args.external_repo, "Cosmos Predict2 repository")
    checkpoint = _require_file(args.checkpoint, "DROID checkpoint")
    lora_checkpoint = (
        _require_file(args.lora_checkpoint, "LoRA checkpoint")
        if args.lora_checkpoint is not None
        else None
    )
    tokenizer = _require_file(args.tokenizer, "Cosmos tokenizer")
    embedding_args = (args.prompt_embedding, args.negative_prompt_embedding)
    if any(value is not None for value in embedding_args):
        if not all(value is not None for value in embedding_args):
            raise ValueError("both positive and negative prompt embeddings are required")
        if args.text_encoder is not None:
            raise ValueError("text encoder and precomputed embeddings are mutually exclusive")
        text_encoder = None
        prompt_embedding = _require_file(args.prompt_embedding, "positive prompt embedding")
        negative_prompt_embedding = _require_file(
            args.negative_prompt_embedding, "negative prompt embedding"
        )
        load_t5_embedding(prompt_embedding)
        load_t5_embedding(negative_prompt_embedding)
    else:
        if args.text_encoder is None:
            raise ValueError("provide a text encoder or both precomputed embeddings")
        text_encoder = _require_dir(args.text_encoder, "T5 text encoder")
        for name in ("config.json", "spiece.model", "tokenizer.json", "pytorch_model.bin"):
            _require_file(text_encoder / name, f"T5 {name}")
        prompt_embedding = None
        negative_prompt_embedding = None
    input_image = _require_file(args.input_image, "real condition image")
    prompt_file = _require_file(args.prompt_file, "prompt file")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)

    inventory = query_physical_gpus()
    selection = validate_gpu_selection(
        inventory, args.physical_gpus, args.min_free_memory_mib
    )
    prompt = prompt_file.read_text().strip()
    if not prompt:
        raise ValueError("prompt file is empty")

    (output / "packages.txt").write_text(package_inventory())
    _write_json(
        output / "gpu-selection.json",
        {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "inventory_before_launch": inventory,
            "selected_physical_gpus": selection,
            "cuda_visible_devices": ",".join(map(str, args.physical_gpus)),
            "minimum_free_memory_mib": args.min_free_memory_mib,
        },
    )
    _write_json(
        output / "experiment-config.json",
        {
            "schema_version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "RUNNING",
            "model": "Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID",
            "model_revision": "ModelScope master, artifact pinned by SHA-256",
            "checkpoint": {
                "path": str(checkpoint),
                "size": checkpoint.stat().st_size,
                "sha256": _sha256(checkpoint),
            },
            "lora_checkpoint": (
                None
                if lora_checkpoint is None
                else {
                    "path": str(lora_checkpoint),
                    "size": lora_checkpoint.stat().st_size,
                    "sha256": _sha256(lora_checkpoint),
                    "rank": args.lora_rank,
                    "alpha": args.lora_rank,
                    "residual_scale": args.lora_scale,
                    "scale_method": "multiply each lora_B tensor exactly once before load",
                    "profile": args.lora_profile,
                    "target_modules": lora_target_modules(args.lora_profile),
                }
            ),
            "tokenizer": {
                "path": str(tokenizer),
                "size": tokenizer.stat().st_size,
                "sha256": _sha256(tokenizer),
            },
            "text_conditioning": (
                {
                    "mode": "online_cpu_t5_bfloat16",
                    "text_encoder_path": str(text_encoder),
                    "weight_size": (text_encoder / "pytorch_model.bin").stat().st_size,
                    "weight_sha256": _sha256(text_encoder / "pytorch_model.bin"),
                    "only_embeddings_moved_to_gpu": True,
                }
                if text_encoder is not None
                else {
                    "mode": "precomputed_t5_float16",
                    "positive_embedding_path": str(prompt_embedding),
                    "positive_embedding_sha256": _sha256(prompt_embedding),
                    "negative_embedding_path": str(negative_prompt_embedding),
                    "negative_embedding_sha256": _sha256(negative_prompt_embedding),
                    "t5_instantiated_during_inference": False,
                }
            ),
            "conditioning": {
                "sample_id": args.sample_id,
                "real_condition_image": str(input_image),
                "real_condition_image_sha256": _sha256(input_image),
                "text_condition_file": str(prompt_file),
                "text_condition_sha256": _sha256(prompt_file),
                "generated_content": "all video frames after the real first frame",
                "real_future_frames_passed_to_model": False,
            },
            "seed": args.seed,
            "guidance": args.guidance,
            "git": {
                "project_commit": args.git_commit or "unresolved",
                "project_branch": args.git_branch,
                "working_tree_status": "dirty",
                "launcher_sha256": _sha256(Path(__file__)),
            },
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
        },
    )

    child_args = []
    for name, value in (
        ("external-repo", external_repo),
        ("checkpoint", checkpoint),
        ("tokenizer", tokenizer),
        ("input-image", input_image),
        ("prompt-file", prompt_file),
        ("sample-id", args.sample_id),
        ("output-dir", output),
        ("seed", args.seed),
        ("guidance", args.guidance),
        ("master-port", args.master_port),
    ):
        child_args.extend([f"--{name}", str(value)])
    if text_encoder is not None:
        child_args.extend(["--text-encoder", str(text_encoder)])
    else:
        child_args.extend(
            [
                "--prompt-embedding",
                str(prompt_embedding),
                "--negative-prompt-embedding",
                str(negative_prompt_embedding),
            ]
        )
    child_args.extend(["--physical-gpus", *map(str, args.physical_gpus)])
    if lora_checkpoint is not None:
        child_args.extend(
            [
                "--lora-checkpoint",
                str(lora_checkpoint),
                "--lora-profile",
                args.lora_profile,
                "--lora-rank",
                str(args.lora_rank),
                "--lora-scale",
                str(args.lora_scale),
            ]
        )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(args.physical_gpus)}",
        f"--master_port={args.master_port}",
        str(Path(__file__).resolve()),
        "--worker",
        *child_args,
    ]
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.physical_gpus))
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
    result: dict[str, Any] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "returncode": completed.returncode,
        "status": "PARTIAL",
    }
    generated = output / "generated.mp4"
    if completed.returncode == 0 and generated.is_file() and generated.stat().st_size:
        result.update(
            {
                "status": "WORKING",
                "generated_video": str(generated),
                "generated_video_sha256": _sha256(generated),
                "probe": _probe_video(generated),
            }
        )
    _write_json(output / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "WORKING":
        raise RuntimeError(f"Cosmos DROID inference failed; inspect {output / 'run.log'}")
    return 0


def main() -> int:
    args = _parser().parse_args()
    return _worker(args) if args.worker else _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
