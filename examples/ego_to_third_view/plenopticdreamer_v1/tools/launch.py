#!/usr/bin/env python3
"""Launch a provenance-recorded PlenopticDreamer GPU job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
COSMOS_COMMIT = "2ff49d0717af02057ae79bc75c00fbff9da1b4e7"


def run_text(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def query_gpus() -> list[dict[str, object]]:
    output = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError(f"Unexpected nvidia-smi GPU row: {line!r}")
        index, uuid, name, free, total = fields
        gpus.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "memory_free_mib": int(free),
                "memory_total_mib": int(total),
            }
        )
    if not gpus:
        raise RuntimeError("nvidia-smi did not report any physical GPU")
    return gpus


def query_compute_processes() -> list[dict[str, object]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Unable to inspect active GPU compute processes: " + result.stderr.strip())
    processes = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) == 3:
            uuid, pid, memory = fields
            processes.append(
                {
                    "gpu_uuid": uuid,
                    "pid": int(pid),
                    "used_memory_mib": int(memory) if memory.isdigit() else None,
                }
            )
    return processes


def select_gpus(
    inventory: list[dict[str, object]], selectors: str
) -> list[dict[str, object]]:
    requested = [item.strip() for item in selectors.split(",") if item.strip()]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--gpus must contain distinct physical indices or full GPU UUIDs")
    selected = []
    for selector in requested:
        matches = [
            gpu
            for gpu in inventory
            if selector in (str(gpu["index"]), str(gpu["uuid"]))
        ]
        if len(matches) != 1:
            raise ValueError(f"GPU selector {selector!r} did not match one physical GPU")
        selected.append(matches[0])
    return selected


def git_state() -> dict[str, object]:
    try:
        repository = Path(run_text(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"]))
        return {
            "repository": str(repository),
            "commit": run_text(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "branch": run_text(["git", "-C", str(ROOT), "branch", "--show-current"]),
            "status_porcelain": run_text(["git", "-C", str(ROOT), "status", "--porcelain"]),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"repository": None, "commit": None, "branch": None, "status_porcelain": None}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def strip_separator(values: list[str]) -> list[str]:
    return values[1:] if values[:1] == ["--"] else values


def option_value(values: list[str], name: str, default: object) -> object:
    for index, value in enumerate(values):
        if value == name and index + 1 < len(values):
            return values[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return default


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gpus", required=True, help="Comma-separated physical indices or full UUIDs")
    common.add_argument("--run-dir", required=True, help="New experiment directory; reuse only with --resume")
    common.add_argument("--cosmos-root", default=os.getenv("COSMOS_TRANSFER_ROOT", "cosmos-transfer2.5"))
    common.add_argument("--min-free-mib", type=int, default=60000)
    common.add_argument("--allow-busy", action="store_true")

    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="entry", required=True)

    train = commands.add_parser("train", parents=[common])
    train.add_argument("--config", required=True)
    train.add_argument("--nnodes", type=int, default=1)
    train.add_argument("--node-rank", type=int, default=0)
    train.add_argument("--master-addr", default="127.0.0.1")
    train.add_argument("--master-port", type=int, default=29670)
    checkpoint = train.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume")
    checkpoint.add_argument("--weights-from")
    train.add_argument("extra", nargs=argparse.REMAINDER)

    infer = commands.add_parser("infer", parents=[common])
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("extra", nargs=argparse.REMAINDER)

    spatial = commands.add_parser("verify-spatial", parents=[common])
    spatial.add_argument("extra", nargs=argparse.REMAINDER)

    components = commands.add_parser("verify-training", parents=[common])
    components.add_argument("--context-parallel-size", type=int, default=2)
    components.add_argument("extra", nargs=argparse.REMAINDER)
    return root


def prepare_run(args: argparse.Namespace, selected: list[dict[str, object]]) -> tuple[Path, dict[str, object]]:
    run_dir = resolve_path(args.run_dir)
    continuing = args.entry == "train" and bool(args.resume)
    manifest_path = run_dir / "run.json"
    if args.entry == "train" and args.node_rank > 0:
        deadline = time.monotonic() + 60
        while not manifest_path.is_file() and time.monotonic() < deadline:
            time.sleep(1)
        if not manifest_path.is_file():
            raise RuntimeError("Rank-zero launch did not create the shared run directory")
        manifest = json.loads(manifest_path.read_text())
    elif continuing:
        if not manifest_path.is_file():
            raise FileNotFoundError("--resume requires the existing run.json")
        manifest = json.loads(manifest_path.read_text())
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "entry": args.entry,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "root": str(ROOT),
            "git": git_state(),
        }
        if args.entry == "train":
            config = resolve_path(args.config)
            raw = config.read_bytes()
            settings = json.loads(raw)
            expected_world = args.nnodes * len(selected)
            if settings.get("world_size") != expected_world:
                raise ValueError(
                    f"Config world_size={settings.get('world_size')} but launch requests {expected_world} ranks"
                )
            manifest.update(
                {
                    "config_source": str(config),
                    "config_sha256": hashlib.sha256(raw).hexdigest(),
                    "seed": settings.get("seed"),
                    "nnodes": args.nnodes,
                    "processes_per_node": len(selected),
                }
            )
            (run_dir / "input_config.json").write_bytes(raw)
        elif args.entry == "infer":
            manifest["seed"] = int(option_value(strip_separator(args.extra), "--seed", 2026))
            manifest["checkpoint"] = str(resolve_path(args.checkpoint))
        else:
            manifest["seed"] = "fixed in verification script"
        save_json(manifest_path, manifest)
    if manifest.get("entry") != args.entry:
        raise ValueError("The existing run directory belongs to another entry point")
    if args.entry == "train":
        config = resolve_path(args.config)
        if manifest.get("config_sha256") != sha256(config):
            raise ValueError("Every node and resume must use the run's original config bytes")
        if manifest.get("nnodes") != args.nnodes or manifest.get("processes_per_node") != len(selected):
            raise ValueError("Every node and resume must use the run's original topology")
    return run_dir, manifest


def main() -> int:
    args = parser().parse_args()
    if args.min_free_mib < 0:
        raise ValueError("--min-free-mib must be non-negative")
    if args.entry == "train":
        if args.nnodes < 1 or not 0 <= args.node_rank < args.nnodes:
            raise ValueError("--node-rank must be within the requested positive node count")
        if args.nnodes > 1 and args.master_addr in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Multi-node training requires a master address reachable by every node")
    inventory = query_gpus()
    processes = query_compute_processes()
    selected = select_gpus(inventory, args.gpus)
    selected_uuids = {str(gpu["uuid"]) for gpu in selected}
    busy = [item for item in processes if str(item["gpu_uuid"]) in selected_uuids]
    if busy and not args.allow_busy:
        raise RuntimeError(f"Selected GPUs already have compute processes: {busy}")
    insufficient = [
        gpu for gpu in selected if int(gpu["memory_free_mib"]) < args.min_free_mib
    ]
    if insufficient:
        raise RuntimeError(
            f"Selected GPUs have less than {args.min_free_mib} MiB free: {insufficient}"
        )

    cosmos = resolve_path(args.cosmos_root)
    commit = run_text(["git", "-C", str(cosmos), "rev-parse", "HEAD"])
    if commit != COSMOS_COMMIT:
        raise RuntimeError(f"Cosmos source must be pinned at {COSMOS_COMMIT}; found {commit}")
    if run_text(["git", "-C", str(cosmos), "status", "--porcelain"]):
        raise RuntimeError("Pinned Cosmos source has local modifications")

    run_dir, _manifest = prepare_run(args, selected)
    node_rank = args.node_rank if args.entry == "train" else 0
    provenance = run_dir / "provenance"
    provenance.mkdir(exist_ok=True)
    node = {
        "entry": args.entry,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "node_rank": node_rank,
        "gpu_inventory": inventory,
        "selected_gpus": selected,
        "active_compute_processes_before_launch": processes,
        "allow_busy": args.allow_busy,
        "min_free_mib": args.min_free_mib,
        "cosmos_root": str(cosmos),
        "cosmos_commit": commit,
        "git": git_state(),
    }
    node_path = provenance / f"node-{node_rank:02d}.json"
    save_json(node_path, node)
    try:
        packages = run_text([sys.executable, "-m", "pip", "freeze"])
    except (OSError, subprocess.CalledProcessError) as error:
        packages = f"pip freeze failed: {type(error).__name__}: {error}"
    (provenance / f"packages-node-{node_rank:02d}.txt").write_text(packages + "\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu["uuid"]) for gpu in selected)
    env["PLENOPTIC_ROOT"] = str(ROOT)
    env["PLENOPTIC_RUN_MANIFEST"] = str(run_dir / "run.json")
    additions = [
        str(ROOT / "tools"),
        str(cosmos),
        str(cosmos / "packages/cosmos-oss"),
        str(cosmos / "packages/cosmos-cuda"),
    ]
    if env.get("PYTHONPATH"):
        additions.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(additions)

    extra = strip_separator(args.extra)
    if args.entry == "train":
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={args.nnodes}",
            f"--nproc_per_node={len(selected)}",
            f"--node_rank={args.node_rank}",
            f"--master_addr={args.master_addr}",
            f"--master_port={args.master_port}",
            str(ROOT / "tools/train_plenoptic.py"),
            "--config",
            str(resolve_path(args.config)),
            "--output",
            str(run_dir),
        ]
        if args.resume:
            command.extend(["--resume", str(resolve_path(args.resume))])
        if args.weights_from:
            command.extend(["--weights-from", str(resolve_path(args.weights_from))])
        command.extend(extra)
    elif args.entry == "infer":
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc_per_node={len(selected)}",
            "--master_addr=127.0.0.1",
            "--master_port=29671",
            str(ROOT / "tools/infer_stage1.py"),
            "--checkpoint",
            str(resolve_path(args.checkpoint)),
            "--output",
            str(run_dir),
            *extra,
        ]
    else:
        script = "verify_spatial_cp.py" if args.entry == "verify-spatial" else "verify_training_components.py"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc_per_node={len(selected)}",
            "--master_addr=127.0.0.1",
            "--master_port=29672",
            str(ROOT / "tools" / script),
            *extra,
        ]
        if args.entry == "verify-training":
            env["VERIFY_CP"] = str(args.context_parallel_size)

    save_json(
        provenance / f"command-node-{node_rank:02d}.json",
        {"argv": command, "cwd": str(ROOT), "environment": {"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]}},
    )
    log_path = run_dir / f"node-{node_rank:02d}.log"
    node["command"] = command
    node["started_at"] = datetime.now(timezone.utc).isoformat()
    save_json(node_path, node)
    with log_path.open("a", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        returncode = process.wait()
    node["finished_at"] = datetime.now(timezone.utc).isoformat()
    node["exit_code"] = returncode
    save_json(node_path, node)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
