#!/usr/bin/env python3
"""Run official Wan-Animate-2 with strict two-GPU provenance."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.wan_animate import query_gpus  # noqa: E402
from phiagent.rendering.wan_animate2 import (  # noqa: E402
    WAN_ANIMATE2_MODEL_REVISION,
    file_sha256,
    select_wan_animate2_gpus,
    verify_wan_animate2_checkpoint,
    verify_wan_animate2_source,
    wan_animate2_master_port,
    write_runtime_config,
)


def _write_json(path: Path, payload: object) -> None:
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
        "head": head.stdout.strip() if head.returncode == 0 else "UNBORN",
        "status": status.stdout.splitlines(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/wan-animate2"))
    parser.add_argument("--gpu", type=int, action="append", default=[])
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=60 * 1024)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--clip-len", type=int, default=81)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--distilled", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--method-label",
        default="wan_animate2_proxy_not_official_phizero",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Wan-Animate-2 requires the pinned Python 3.11 environment")
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        raise ValueError("width and height must be positive multiples of 16")
    if args.clip_len < 5 or (args.clip_len - 1) % 4:
        raise ValueError("clip-len must satisfy clip_len = 4n + 1")
    steps = args.steps if args.steps is not None else (10 if args.distilled else 40)
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else (1.0 if args.distilled else 3.0)
    )
    if args.fps <= 0 or steps <= 0 or guidance_scale <= 0:
        raise ValueError("fps, steps, and guidance-scale must be positive")
    if not args.method_label.strip():
        raise ValueError("method-label cannot be empty")
    source = args.source_video.expanduser().resolve()
    reference = args.reference_image.expanduser().resolve()
    for label, path in (("source video", source), ("reference image", reference)):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} does not exist or is empty: {path}")
    python = Path(os.path.abspath(args.python.expanduser()))
    if not python.is_file():
        raise ValueError(f"training Python does not exist: {python}")
    repo = args.repo.expanduser().resolve()
    checkpoint = args.checkpoint_dir.expanduser().resolve()
    source_commit = verify_wan_animate2_source(repo)
    checkpoint_hashes = verify_wan_animate2_checkpoint(
        checkpoint, distilled=args.distilled
    )

    gpus, inventory, processes = query_gpus()
    selected = select_wan_animate2_gpus(
        gpus,
        args.gpu,
        minimum_free_mib=args.minimum_free_gpu_mib,
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu.physical_index) for gpu in selected
    )
    environment["PYTHONHASHSEED"] = str(args.seed)
    environment["PYTHONPATH"] = str(repo)
    environment["MASTER_ADDR"] = "127.0.0.1"
    environment["MASTER_PORT"] = str(wan_animate2_master_port(selected))
    environment["RANK"] = "0"
    environment["WORLD_SIZE"] = "1"

    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'torch':torch.__version__,"
                "'cuda':torch.version.cuda,'available':torch.cuda.is_available(),"
                "'devices':torch.cuda.device_count()}))"
            ),
        ],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(probe.stdout.splitlines()[-1])
    if not runtime["available"] or runtime["devices"] != 2:
        raise RuntimeError("selected physical GPUs did not map to exactly two CUDA devices")
    if runtime["torch"].split("+", maxsplit=1)[0] != "2.7.1" or runtime["cuda"] != "12.6":
        raise RuntimeError(
            f"Wan-Animate-2 runtime is torch {runtime['torch']} / CUDA {runtime['cuda']}; "
            "expected torch 2.7.1 / CUDA 12.6"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = args.experiment_root.expanduser().resolve() / f"{stamp}-{uuid4().hex[:8]}"
    config = experiment / "config" / "wan_animate_2.yaml"
    write_runtime_config(repo, checkpoint, config, distilled=args.distilled)
    inductor_cache = experiment / "cache" / "torchinductor"
    triton_cache = experiment / "cache" / "triton"
    inductor_cache.mkdir(parents=True)
    triton_cache.mkdir()
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    environment["TRITON_CACHE_DIR"] = str(triton_cache)
    upstream_output = experiment / "upstream"
    command = [
        str(python),
        str(repo / "infer" / "wan_animate_2_demo.py"),
        "--prompt",
        args.prompt,
        "--refer-img-file",
        str(reference),
        "--refer-video-file",
        str(source),
        "--config",
        str(config),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--clip_len",
        str(args.clip_len),
        "--sample_guide_scale",
        str(guidance_scale),
        "--step",
        str(steps),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(upstream_output),
    ]
    packages = {}
    for name in ("torch", "torchvision", "transformers", "flash-attn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record = {
        "schema_version": "0.1.0",
        "method": args.method_label.strip(),
        "status": "preflight_passed" if args.preflight_only else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "command_shell": shlex.join(command),
        "config": {
            **vars(args),
            "source_video": str(source),
            "reference_image": str(reference),
            "repo": str(repo),
            "checkpoint_dir": str(checkpoint),
            "python": str(python),
            "experiment_root": str(args.experiment_root),
            "resolved_steps": steps,
            "resolved_guidance_scale": guidance_scale,
        },
        "inputs": {
            "source_sha256": file_sha256(source),
            "reference_sha256": file_sha256(reference),
        },
        "source_commit": source_commit,
        "model_revision": WAN_ANIMATE2_MODEL_REVISION,
        "checkpoint_revision_marker": (
            checkpoint / ".phiagent-model-revision"
        ).read_text().strip(),
        "checkpoint_hashes": checkpoint_hashes,
        "selected_gpus": [asdict(gpu) for gpu in selected],
        "gpu_inventory": [asdict(gpu) for gpu in gpus],
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        "master_addr": environment["MASTER_ADDR"],
        "master_port": environment["MASTER_PORT"],
        "machine_rank": environment["RANK"],
        "machine_world_size": environment["WORLD_SIZE"],
        "torchinductor_cache_dir": environment["TORCHINDUCTOR_CACHE_DIR"],
        "triton_cache_dir": environment["TRITON_CACHE_DIR"],
        "runtime": runtime,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "packages": packages,
        "git": _git_state(Path(__file__).resolve().parents[1]),
    }
    metadata_path = experiment / "metadata.json"
    _write_json(metadata_path, record)
    if args.preflight_only:
        print(json.dumps({"experiment": str(experiment), "status": record["status"]}))
        return 0

    log_path = experiment / "inference.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=repo / "infer",
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    candidates = list(upstream_output.glob("session_*/results.mp4"))
    record["completed_at"] = datetime.now(timezone.utc).isoformat()
    record["returncode"] = completed.returncode
    if completed.returncode != 0 or len(candidates) != 1 or not candidates[0].stat().st_size:
        record["status"] = "failed"
        _write_json(metadata_path, record)
        raise SystemExit(f"Wan-Animate-2 failed; see {log_path}")
    result = experiment / "result.mp4"
    shutil.copy2(candidates[0], result)
    record["status"] = "completed"
    record["result"] = str(result)
    record["result_sha256"] = file_sha256(result)
    _write_json(metadata_path, record)
    print(json.dumps({"experiment": str(experiment), "result": str(result)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
