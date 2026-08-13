#!/usr/bin/env python3
"""Run a sharded, audited BWM production batch across physical GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-base-path", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=60 * 1024)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--num-frames", type=int, default=57)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--maximum-samples", type=int, default=0)
    parser.add_argument("--fps", type=float, default=24.0)
    return parser


def plan_shards(
    *, total: int, physical_gpus: list[int], start_index: int
) -> tuple[dict[str, int], ...]:
    """Plan balanced contiguous metadata shards without duplicate samples."""

    if total <= 0 or start_index < 0:
        raise ValueError("total must be positive and start_index non-negative")
    if not physical_gpus or len(physical_gpus) != len(set(physical_gpus)):
        raise ValueError("physical GPU indices must be non-empty and unique")
    worker_count = min(len(physical_gpus), total)
    base_count, extra = divmod(total, worker_count)
    cursor = start_index
    shards = []
    for worker_index, gpu in enumerate(physical_gpus[:worker_count]):
        count = base_count + (1 if worker_index < extra else 0)
        shards.append(
            {
                "worker_index": worker_index,
                "physical_gpu": gpu,
                "start_index": cursor,
                "samples": count,
            }
        )
        cursor += count
    return tuple(shards)


def main() -> int:
    args = _parser().parse_args()
    repository = args.repository.expanduser().resolve()
    base_model = args.base_model.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    dataset_base = args.dataset_base_path.expanduser().resolve()
    action_stats = args.action_stats.expanduser().resolve()
    root = args.experiment_root.expanduser().resolve()
    paths = (repository, base_model, checkpoint, metadata, dataset_base, action_stats)
    if any(not path.exists() for path in paths):
        raise ValueError("one or more required batch inputs are missing")
    if len(args.gpu) != len(set(args.gpu)) or any(gpu < 0 for gpu in args.gpu):
        raise ValueError("physical GPU indices must be unique and non-negative")
    if args.start_index < 0 or args.maximum_samples < 0:
        raise ValueError("sample indices and counts must be non-negative")
    if args.num_frames <= 0 or args.num_inference_steps <= 0 or args.fps <= 0:
        raise ValueError("frame, inference-step, and fps settings must be positive")
    rows = [line for line in metadata.read_text().splitlines() if line.strip()]
    available = len(rows) - args.start_index
    total = available if args.maximum_samples == 0 else min(args.maximum_samples, available)
    if total <= 0:
        raise ValueError("selected metadata range contains no samples")
    planned_shards = plan_shards(
        total=total, physical_gpus=args.gpu, start_index=args.start_index
    )
    worker_count = len(planned_shards)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    campaign = root / f"{timestamp}-{args.label}"
    campaign.mkdir(parents=True, exist_ok=False)
    generated_seconds = total * args.num_frames / args.fps
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "STARTED",
        "honest_status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "label": args.label,
        "inputs": {
            "repository": str(repository),
            "base_model": str(base_model),
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": _sha256(checkpoint),
            "metadata": str(metadata),
            "metadata_sha256": _sha256(metadata),
            "dataset_base_path": str(dataset_base),
            "action_stats": str(action_stats),
            "action_stats_sha256": _sha256(action_stats),
        },
        "config": {
            "physical_gpus": args.gpu[:worker_count],
            "minimum_free_gpu_mib": args.minimum_free_gpu_mib,
            "seed": args.seed,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "fps": args.fps,
            "start_index": args.start_index,
            "samples": total,
            "generated_video_seconds": generated_seconds,
        },
        "shards": [],
    }
    _write_json(campaign / "manifest.json", manifest)
    try:
        shard_specs = []
        for planned in planned_shards:
            worker_index = planned["worker_index"]
            gpu = planned["physical_gpu"]
            count = planned["samples"]
            cursor = planned["start_index"]
            shard_root = campaign / "shards" / f"gpu-{gpu}"
            command = [
                sys.executable,
                str(Path(__file__).resolve().parent / "run_bwm_heldout_inference.py"),
                "--repository",
                str(repository),
                "--base-model",
                str(base_model),
                "--checkpoint",
                str(checkpoint),
                "--metadata",
                str(metadata),
                "--dataset-base-path",
                str(dataset_base),
                "--action-stats",
                str(action_stats),
                "--experiment-root",
                str(shard_root),
                "--label",
                f"shard-{worker_index}",
                "--gpu",
                str(gpu),
                "--minimum-free-gpu-mib",
                str(args.minimum_free_gpu_mib),
                "--seed",
                str(args.seed),
                "--num-frames",
                str(args.num_frames),
                "--num-inference-steps",
                str(args.num_inference_steps),
                "--start-index",
                str(cursor),
                "--maximum-samples",
                str(count),
                "--generated-video-seconds",
                str(count * args.num_frames / args.fps),
            ]
            shard_specs.append(
                {
                    "worker_index": worker_index,
                    "physical_gpu": gpu,
                    "start_index": cursor,
                    "samples": count,
                    "command": command,
                    "stdout": str(campaign / f"shard-{worker_index}.stdout.log"),
                }
            )
        manifest["shards"] = shard_specs
        _write_json(campaign / "manifest.json", manifest)
        processes = []
        started = time.monotonic()
        for spec in shard_specs:
            log = Path(str(spec["stdout"])).open("w")
            process = subprocess.Popen(
                spec["command"],  # type: ignore[arg-type]
                cwd=Path(__file__).resolve().parents[1],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            processes.append((process, log, spec))
        shard_results = []
        for process, log, spec in processes:
            return_code = process.wait()
            log.close()
            shard_root = campaign / "shards" / f"gpu-{spec['physical_gpu']}"
            result_paths = sorted(shard_root.glob("*/result.json"))
            result = (
                json.loads(result_paths[-1].read_text()) if len(result_paths) == 1 else None
            )
            shard_results.append(
                {**spec, "return_code": return_code, "result": result}
            )
        wall_seconds = time.monotonic() - started
        videos_root = campaign / "videos"
        videos_root.mkdir()
        video_records = []
        for shard in shard_results:
            result = shard["result"]
            if not isinstance(result, dict):
                continue
            for source_name in result.get("videos", []):
                source = Path(str(source_name))
                destination = videos_root / source.name
                if destination.exists():
                    raise ValueError(f"duplicate batch output: {destination.name}")
                os.link(source, destination)
                video_records.append(
                    {
                        "path": str(destination),
                        "bytes": destination.stat().st_size,
                        "sha256": _sha256(destination),
                        "physical_gpu": shard["physical_gpu"],
                    }
                )
        complete = (
            len(video_records) == total
            and all(shard["return_code"] == 0 for shard in shard_results)
            and all(
                isinstance(shard["result"], dict)
                and shard["result"].get("status") == "WORKING"
                for shard in shard_results
            )
        )
        gpu_seconds = sum(
            float(shard["result"]["wall_seconds"])
            for shard in shard_results
            if isinstance(shard["result"], dict)
        )
        result = {
            "schema_version": "1.0.0",
            "status": "WORKING" if complete else "BLOCKED",
            "honest_status": "WORKING" if complete else "BLOCKED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "campaign": str(campaign),
            "samples_requested": total,
            "samples_completed": len(video_records),
            "generated_video_seconds": generated_seconds,
            "wall_seconds": wall_seconds,
            "gpu_seconds": gpu_seconds,
            "samples_per_hour_wall": 3600.0 * len(video_records) / wall_seconds,
            "generated_seconds_per_hour_wall": (
                3600.0 * generated_seconds / wall_seconds
            ),
            "wall_seconds_per_sample": wall_seconds / max(len(video_records), 1),
            "gpu_seconds_per_sample": gpu_seconds / max(len(video_records), 1),
            "wall_seconds_per_generated_second": wall_seconds / generated_seconds,
            "gpu_seconds_per_generated_second": gpu_seconds / generated_seconds,
            "worker_count": worker_count,
            "shards": shard_results,
            "videos": video_records,
            "claim_boundary": (
                "End-to-end generated-video throughput on this host; excludes upstream "
                "scenario authoring and does not establish real-robot validity."
            ),
        }
        _write_json(campaign / "result.json", result)
        manifest.update(
            {
                "status": "completed" if complete else "failed",
                "honest_status": result["honest_status"],
                "completed_at": result["completed_at"],
                "result": str(campaign / "result.json"),
            }
        )
        _write_json(campaign / "manifest.json", manifest)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if complete else 2
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "honest_status": "BLOCKED",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(campaign / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
