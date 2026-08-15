#!/usr/bin/env python3
"""Persist a reproducible JoyAI replacement-video capacity estimate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.joyai_video_edit import write_json  # noqa: E402
from phiagent.rendering.replacement_capacity import (  # noqa: E402
    estimate_joyai_replacement_capacity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-hours", type=float, default=100.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--average-clip-seconds", type=float, default=27.5)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--gpu-utilization", type=float, default=0.85)
    parser.add_argument("--postprocess-workers", type=int, default=1)
    parser.add_argument("--postprocess-utilization", type=float, default=0.85)
    parser.add_argument("--session-overhead-seconds", type=float, default=0.0)
    parser.add_argument("--review-bitrate-mbps", type=float, default=50.0)
    parser.add_argument("--protocol-jpeg-kib", type=float, default=200.0)
    parser.add_argument(
        "--gpu-sweep",
        default="1,2,4,8,16,32",
        help="Comma-separated A800 counts; each row uses balanced CPU postprocessing.",
    )
    return parser


def _git_state() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, command in {
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
        result[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    return result


def _packages(output: Path) -> dict[str, Any]:
    rows = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in metadata.distributions()
        if distribution.metadata["Name"]
    )
    path = output / "packages.txt"
    payload = "\n".join(rows) + "\n"
    path.write_text(payload, encoding="utf-8")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "count": len(rows),
        "python": sys.version,
        "executable": sys.executable,
    }


def _gpu_counts(raw: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError("gpu sweep must contain comma-separated integers") from exc
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("gpu sweep counts must be positive")
    if len(set(counts)) != len(counts):
        raise ValueError("gpu sweep counts must be unique")
    return counts


def _estimate(args: argparse.Namespace, gpu_count: int, workers: int) -> dict[str, Any]:
    average_clip_frames = round(args.average_clip_seconds * args.fps)
    if not math.isclose(
        average_clip_frames / args.fps,
        args.average_clip_seconds,
        abs_tol=1e-9,
    ):
        raise ValueError("average clip seconds must resolve to an integer frame count")
    return estimate_joyai_replacement_capacity(
        video_hours=args.video_hours,
        fps=args.fps,
        average_clip_frames=average_clip_frames,
        gpu_count=gpu_count,
        gpu_utilization=args.gpu_utilization,
        postprocess_workers=workers,
        postprocess_utilization=args.postprocess_utilization,
        session_overhead_seconds=args.session_overhead_seconds,
        review_bitrate_mbps=args.review_bitrate_mbps,
        protocol_jpeg_kib=args.protocol_jpeg_kib,
    )


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"capacity experiment already exists: {output}")
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()

    estimate = _estimate(args, args.gpu_count, args.postprocess_workers)
    sweep = []
    for gpu_count in _gpu_counts(args.gpu_sweep):
        provisional = _estimate(args, gpu_count, 1)
        workers = provisional["recommendation"]["balanced_postprocess_workers"]
        row = _estimate(args, gpu_count, workers)
        sweep.append(
            {
                "gpu_count": gpu_count,
                "postprocess_workers": workers,
                **row["calendar"],
            }
        )

    command = [sys.executable, *sys.argv]
    created_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "stage": "joyai_100h_capacity_estimate",
        "created_at": created_at,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "command": command,
        "command_shell": shlex.join(command),
        "random_seed": None,
        "random_seed_reason": "deterministic analytic estimate",
        "estimate": estimate,
        "balanced_gpu_sweep": sweep,
    }
    report_path = output / "capacity.json"
    write_json(report_path, report)
    log_payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (logs / "estimate.log").write_text(log_payload, encoding="utf-8")

    manifest = {
        **report,
        "git": _git_state(),
        "packages": _packages(output),
        "outputs": {
            "capacity": {
                "path": str(report_path),
                "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            },
            "log": {
                "path": str(logs / "estimate.log"),
                "sha256": hashlib.sha256((logs / "estimate.log").read_bytes()).hexdigest(),
            },
        },
        "acceptance": {
            "analytic_estimate_complete": True,
            "optimized_client_cpu_tests_pending": True,
            "optimized_client_a800_benchmark_pending": True,
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
    raise SystemExit(main())
