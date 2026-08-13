#!/usr/bin/env python3
"""Wait for an authorized GPU pair, then run matched BWM baseline/candidate batches."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.wan_animate import GPUInfo, query_gpus  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def ready_gpu_indices(
    gpus: Sequence[GPUInfo],
    requested: Sequence[int],
    minimum_free_mib: int,
) -> bool:
    if (
        minimum_free_mib <= 0
        or not requested
        or len(requested) != len(set(requested))
    ):
        raise ValueError("requested GPUs and minimum memory are invalid")
    by_index = {gpu.physical_index: gpu for gpu in gpus}
    return all(
        index in by_index and by_index[index].free_mib >= minimum_free_mib
        for index in requested
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-base-path", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=61_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=57)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--maximum-wait-seconds", type=int, default=604_800)
    return parser


def _batch_command(
    args: argparse.Namespace,
    checkpoint: Path,
    root: Path,
    label: str,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_bwm_factory_batch.py"),
        "--repository",
        str(args.repository),
        "--base-model",
        str(args.base_model),
        "--checkpoint",
        str(checkpoint),
        "--metadata",
        str(args.metadata),
        "--dataset-base-path",
        str(args.dataset_base_path),
        "--action-stats",
        str(args.action_stats),
        "--experiment-root",
        str(root),
        "--label",
        label,
        "--minimum-free-gpu-mib",
        str(args.minimum_free_gpu_mib),
        "--seed",
        str(args.seed),
        "--num-frames",
        str(args.num_frames),
        "--num-inference-steps",
        str(args.num_inference_steps),
    ]
    for gpu in args.gpu:
        command.extend(("--gpu", str(gpu)))
    return command


def _wait_until_ready(
    args: argparse.Namespace,
    heartbeat: Path,
    *,
    phase: str,
    started: float,
) -> None:
    while True:
        gpus, inventory, processes = query_gpus()
        ready = ready_gpu_indices(gpus, args.gpu, args.minimum_free_gpu_mib)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "ready": ready,
            "requested_gpus": args.gpu,
            "minimum_free_gpu_mib": args.minimum_free_gpu_mib,
            "gpu_inventory_raw": inventory,
            "gpu_processes_raw": processes,
        }
        with heartbeat.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if ready:
            return
        if time.monotonic() - started >= args.maximum_wait_seconds:
            raise TimeoutError(f"timed out waiting for GPUs during {phase}")
        time.sleep(args.poll_seconds)


def main() -> int:
    args = _parser().parse_args()
    for name in (
        "repository",
        "base_model",
        "baseline_checkpoint",
        "candidate_checkpoint",
        "metadata",
        "dataset_base_path",
        "action_stats",
    ):
        path = getattr(args, name).expanduser().resolve()
        setattr(args, name, path)
        if not path.exists():
            raise ValueError(f"required matched-pair input is missing: {path}")
    if (
        args.poll_seconds <= 0
        or args.maximum_wait_seconds <= 0
        or len(args.gpu) != len(set(args.gpu))
    ):
        raise ValueError("wait settings and physical GPU list are invalid")
    root = args.experiment_root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite BWM matched-pair run: {root}")
    root.mkdir(parents=True)
    (root / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    config = {
        "schema_version": "1.0.0",
        "status": "WAITING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "successor_task": "evaluate the matched artifacts with the frozen 20-episode audit",
    }
    _write_json(root / "config.json", config)
    heartbeat = root / "heartbeat.jsonl"
    started = time.monotonic()
    runs = []
    try:
        for method, checkpoint in (
            ("official-bwm", args.baseline_checkpoint),
            ("promoted-adapter", args.candidate_checkpoint),
        ):
            _wait_until_ready(
                args,
                heartbeat,
                phase=f"waiting-{method}",
                started=started,
            )
            run_root = root / "runs" / method
            run_root.mkdir(parents=True)
            command = _batch_command(
                args,
                checkpoint,
                run_root,
                f"{method}-wipe20-seed{args.seed}",
            )
            with (root / f"{method}.log").open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            runs.append(
                {
                    "method": method,
                    "checkpoint": str(checkpoint),
                    "command": command,
                    "return_code": completed.returncode,
                    "run_root": str(run_root),
                }
            )
            if completed.returncode != 0:
                result = {
                    "status": "BLOCKED",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "blocker": f"{method}_batch_failed",
                    "runs": runs,
                }
                _write_json(root / "result.json", result)
                return 2
    except TimeoutError as error:
        result = {
            "status": "BLOCKED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "blocker": "timed_out_waiting_for_requested_gpu_pair",
            "error": str(error),
            "wall_seconds": time.monotonic() - started,
            "runs": runs,
        }
        _write_json(root / "result.json", result)
        return 2
    result: dict[str, Any] = {
        "status": "WORKING",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - started,
        "runs": runs,
        "claim_boundary": (
            "Matched generated-video artifacts only; counterfactual physical ground "
            "truth, task success, and SOTA remain separate gates."
        ),
    }
    _write_json(root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
