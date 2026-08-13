#!/usr/bin/env python3
"""Build a reproducible before/after Wan long-video infrastructure report."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elapsed(start: str, end: str) -> float:
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def legacy_throughput(metadata: dict[str, Any]) -> dict[str, float]:
    completed = [item for item in metadata["windows"] if item["status"] == "completed"]
    if len(completed) != len(metadata["windows"]):
        raise ValueError("legacy baseline must have every planned window completed")
    source_frames = int(metadata["source"]["info"]["frames"])
    fps = float(metadata["source"]["info"]["fps"])
    generation_wall = _elapsed(completed[0]["started_at"], completed[-1]["completed_at"])
    end_to_end = _elapsed(metadata["created_at"], metadata["completed_at"])
    gpu_count = len(metadata["selected_gpus"])
    video_seconds = source_frames / fps
    return {
        "generation_wall_seconds": generation_wall,
        "end_to_end_wall_seconds": end_to_end,
        "effective_generation_fps": source_frames / generation_wall,
        "effective_end_to_end_fps": source_frames / end_to_end,
        "generation_realtime_factor": generation_wall / video_seconds,
        "end_to_end_realtime_factor": end_to_end / video_seconds,
        "a800_gpu_hours": generation_wall * gpu_count / 3600.0,
    }


def compare(
    baseline: dict[str, Any], optimized: dict[str, Any]
) -> dict[str, Any]:
    for key_path in (
        ("source", "sha256"),
        ("reference", "sha256"),
        ("prompt_sha256",),
        ("source_commit",),
        ("model_revision",),
        ("checkpoint_hashes",),
        ("config", "width"),
        ("config", "height"),
        ("config", "fps"),
        ("config", "clip_len"),
        ("config", "steps"),
        ("config", "guidance_scale"),
        ("config", "seed"),
    ):
        first: Any = baseline
        second: Any = optimized
        for key in key_path:
            first = first[key]
            second = second[key]
        if first != second:
            raise ValueError(f"benchmark mismatch at {'.'.join(key_path)}")
    if baseline.get("status") != "completed" or optimized.get("status") != "completed":
        raise ValueError("both benchmark runs must be completed")
    old = legacy_throughput(baseline)
    new = optimized.get("throughput") or optimized.get(
        "recovered_generation_throughput"
    )
    if not isinstance(new, dict):
        raise ValueError("optimized run has no measured or recovered throughput")
    improvement = {
        "generation_wall_speedup": old["generation_wall_seconds"]
        / float(new["generation_wall_seconds"]),
        "effective_generation_fps_gain": float(new["effective_generation_fps"])
        / old["effective_generation_fps"],
        "a800_gpu_hour_reduction_fraction": 1.0
        - float(new["a800_gpu_hours"]) / old["a800_gpu_hours"],
    }
    if new.get("end_to_end_wall_seconds") is not None:
        improvement["end_to_end_wall_speedup"] = old["end_to_end_wall_seconds"] / float(
            new["end_to_end_wall_seconds"]
        )
    return {
        "baseline": old,
        "optimized": new,
        "improvement": improvement,
        "long_horizon_continuity": optimized.get("long_horizon_continuity"),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metadata", type=Path, required=True)
    parser.add_argument("--optimized-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    baseline_path = args.baseline_metadata.expanduser().resolve()
    optimized_path = args.optimized_metadata.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"benchmark report already exists: {output}")
    baseline = json.loads(baseline_path.read_text())
    optimized = json.loads(optimized_path.read_text())
    report = {
        "schema_version": "1.0.0",
        "baseline_metadata": {
            "path": str(baseline_path),
            "sha256": _sha256(baseline_path),
        },
        "optimized_metadata": {
            "path": str(optimized_path),
            "sha256": _sha256(optimized_path),
        },
        **compare(baseline, optimized),
    }
    _write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
