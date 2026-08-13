#!/usr/bin/env python3
"""Run multiple Wan-Animate-2 windows while loading the model only once.

This is an internal GPU entry point used by ``run_wan_animate2_long_video.py``.
It validates the physical-to-logical GPU mapping, records a fresh GPU snapshot,
and emits an append-only JSONL event stream for recovery and exact timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _gpu_snapshot(physical_gpus: tuple[int, ...]) -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    selected = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and int(fields[0]) in physical_gpus:
            selected.append(line)
    if len(selected) != len(physical_gpus):
        raise RuntimeError("nvidia-smi did not report every requested physical GPU")
    return "\n".join(selected)


def _extract_continuation_reference(
    ffmpeg: Path,
    *,
    source: Path,
    frame_index: int,
    output: Path,
) -> list[str]:
    if frame_index < 0:
        raise ValueError("continuation reference frame must be non-negative")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-frames:v",
        "1",
        str(output),
    ]
    subprocess.run(command, check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"failed to extract continuation reference frame {frame_index}")
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-file", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, action="append", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo = args.repo.expanduser().resolve()
    config = args.config.expanduser().resolve()
    job_file = args.job_file.expanduser().resolve()
    status_file = args.status_file.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    physical_gpus = tuple(args.physical_gpu)
    if len(physical_gpus) != 2 or len(set(physical_gpus)) != 2:
        raise ValueError("persistent Wan-Animate-2 batches require two physical GPUs")
    expected_visible = ",".join(str(index) for index in physical_gpus)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES does not match the validated physical GPU pair: "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r} vs {expected_visible!r}"
        )
    master_port = os.environ.get("MASTER_PORT")
    if master_port is None or not master_port.isdigit():
        raise RuntimeError("persistent batch requires an explicit MASTER_PORT")
    if os.environ.get("RANK") != "0" or os.environ.get("WORLD_SIZE") != "1":
        raise RuntimeError("persistent batch requires RANK=0 and WORLD_SIZE=1")
    if status_file.exists():
        raise FileExistsError(f"batch status already exists: {status_file}")
    for label, path in (
        ("Wan-Animate-2 repo", repo),
        ("runtime config", config),
        ("job file", job_file),
        ("FFmpeg", ffmpeg),
    ):
        if not path.exists():
            raise ValueError(f"{label} does not exist: {path}")

    jobs_payload = json.loads(job_file.read_text())
    jobs = jobs_payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("job file must contain a non-empty jobs list")
    temporal_anchor_mode = jobs_payload.get("temporal_anchor_mode", "fixed")
    if temporal_anchor_mode not in {"fixed", "rolling"}:
        raise ValueError("temporal_anchor_mode must be fixed or rolling")

    # Validate the inherited logical mapping before any model allocation.
    runtime_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,torch; print(json.dumps({"
                "'available':torch.cuda.is_available(),"
                "'devices':torch.cuda.device_count(),"
                "'names':[torch.cuda.get_device_name(i) "
                "for i in range(torch.cuda.device_count())]}))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(runtime_probe.stdout.splitlines()[-1])
    if not runtime["available"] or runtime["devices"] != 2:
        raise RuntimeError("persistent batch did not map to exactly two CUDA devices")
    _append_event(
        status_file,
        {
            "event": "batch_started",
            "at": _now(),
            "physical_gpus": physical_gpus,
            "cuda_visible_devices": expected_visible,
            "master_addr": os.environ.get("MASTER_ADDR"),
            "master_port": int(master_port),
            "rank": 0,
            "world_size": 1,
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "gpu_snapshot": _gpu_snapshot(physical_gpus),
            "runtime": runtime,
            "job_count": len(jobs),
            "temporal_anchor_mode": temporal_anchor_mode,
        },
    )

    sys.path.insert(0, str(repo / "infer"))
    load_started = time.perf_counter()
    from core import build_object_from_config_file  # type: ignore  # noqa: PLC0415

    pipeline = build_object_from_config_file(str(config))
    load_seconds = time.perf_counter() - load_started
    _append_event(
        status_file,
        {"event": "model_loaded", "at": _now(), "load_seconds": load_seconds},
    )

    previous_result: Path | None = None
    for position, job in enumerate(jobs):
        index = int(job["index"])
        output_dir = Path(job["output_dir"]).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        canonical_reference = Path(job["canonical_reference"]).expanduser().resolve()
        reference = canonical_reference
        reference_kind = "canonical"
        extraction_command: list[str] | None = None
        if temporal_anchor_mode == "rolling" and previous_result is not None:
            local_frame = int(job["previous_window_local_frame"])
            reference = output_dir / "continuation-reference.png"
            extraction_command = _extract_continuation_reference(
                ffmpeg,
                source=previous_result,
                frame_index=local_frame,
                output=reference,
            )
            reference_kind = "previous_window_camera_frame"

        _append_event(
            status_file,
            {
                "event": "window_started",
                "at": _now(),
                "index": index,
                "position": position,
                "reference": str(reference),
                "reference_kind": reference_kind,
                "reference_sha256": _sha256(reference),
                "reference_extraction_command": extraction_command,
                "gpu_snapshot": _gpu_snapshot(physical_gpus),
            },
        )
        started = time.perf_counter()
        result_value = pipeline(
            refer_img_path=str(reference),
            tpl_video_path=str(Path(job["input"]).expanduser().resolve()),
            output_path=str(output_dir),
            width=int(job["width"]),
            height=int(job["height"]),
            fps=int(job["fps"]),
            seed=int(job["seed"]),
            clip_len=int(job["clip_len"]),
            sample_guide_scale=float(job["guidance_scale"]),
            step=int(job["steps"]),
            prompt=str(job["prompt"]),
            prompt_ref=str(job.get("prompt_ref", "机器人动作的参考视频")),
        )
        elapsed = time.perf_counter() - started
        candidates = [
            candidate
            for candidate in (
                Path(str(result_value)).expanduser(),
                output_dir / "results.mp4",
            )
            if candidate.is_file() and candidate.stat().st_size > 0
        ]
        if not candidates:
            candidates = sorted(output_dir.glob("**/results.mp4"))
        if not candidates:
            raise RuntimeError(f"persistent window {index} produced no results.mp4")
        previous_result = candidates[0].resolve()
        _append_event(
            status_file,
            {
                "event": "window_completed",
                "at": _now(),
                "index": index,
                "elapsed_seconds": elapsed,
                "result": str(previous_result),
                "result_sha256": _sha256(previous_result),
            },
        )

    _append_event(
        status_file,
        {
            "event": "batch_completed",
            "at": _now(),
            "load_seconds": load_seconds,
            "completed_windows": len(jobs),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
