#!/usr/bin/env python3
"""Promote a held-out-accepted BWM adapter into an immutable factory model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _mean_metric(evaluation: dict[str, object], side: str, field: str) -> float:
    section = evaluation.get(side)
    if not isinstance(section, dict):
        raise ValueError(f"evaluation lacks {side}")
    aggregate = section.get("aggregate")
    if not isinstance(aggregate, dict) or not isinstance(aggregate.get(field), dict):
        raise ValueError(f"evaluation lacks {side}.{field}")
    return float(aggregate[field]["mean"])  # type: ignore[index]


def _evaluation_summary(evaluation: dict[str, object]) -> dict[str, object]:
    fields = (
        "future_ssim",
        "endpoint_ssim",
        "background_mad_0_1",
        "temporal_gradient_mae_0_1",
        "flow_endpoint_error_px_at_224x168",
    )
    candidate = {field: _mean_metric(evaluation, "candidate", field) for field in fields}
    baseline = {field: _mean_metric(evaluation, "baseline", field) for field in fields}
    return {
        "accepted": evaluation.get("accepted"),
        "gates": evaluation.get("gates"),
        "sample_future_ssim_win_fraction": evaluation.get(
            "candidate_sample_future_ssim_win_fraction"
        ),
        "candidate_mean": candidate,
        "baseline_mean": baseline,
        "delta": {field: candidate[field] - baseline[field] for field in fields},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-verification", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--merged-checkpoint", type=Path, required=True)
    parser.add_argument("--merge-manifest", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--validation-evaluation", type=Path, required=True)
    parser.add_argument("--test-evaluation", type=Path, required=True)
    parser.add_argument("--benchmark-result", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"promoted model already exists: {output}")
    inputs = {
        name: getattr(args, name).expanduser().resolve()
        for name in (
            "base_checkpoint",
            "base_verification",
            "adapter",
            "merged_checkpoint",
            "merge_manifest",
            "training_run",
            "dataset_manifest",
            "validation_evaluation",
            "test_evaluation",
            "benchmark_result",
            "base_model",
        )
    }
    for name, path in inputs.items():
        valid = path.is_dir() if name in {"training_run", "base_model"} else path.is_file()
        if not valid:
            raise ValueError(f"promotion input is missing: {path}")
    base_verification = _json(inputs["base_verification"])
    merge = _json(inputs["merge_manifest"])
    dataset = _json(inputs["dataset_manifest"])
    validation = _json(inputs["validation_evaluation"])
    test = _json(inputs["test_evaluation"])
    benchmark = _json(inputs["benchmark_result"])
    training_config = _json(inputs["training_run"] / "config.json")
    training_result = _json(inputs["training_run"] / "result.json")
    if validation.get("accepted") is not True or test.get("accepted") is not True:
        raise ValueError("validation and test evaluations must both be accepted")
    if benchmark.get("status") != "WORKING" or (
        benchmark.get("samples_completed") != benchmark.get("samples_requested")
    ):
        raise ValueError("production benchmark is not complete and WORKING")
    if training_result.get("status") != "WORKING":
        raise ValueError("training run is not WORKING")
    if not (
        dataset.get("status") == "completed"
        and dataset.get("honest_status") == "WORKING"
    ):
        raise ValueError("dataset manifest is not completed and WORKING")
    base_hash = _sha256(inputs["base_checkpoint"])
    adapter_hash = _sha256(inputs["adapter"])
    merged_hash = _sha256(inputs["merged_checkpoint"])
    merge_adapter = merge.get("adapter")
    merge_output = merge.get("output")
    if not isinstance(merge_adapter, dict) or not isinstance(merge_output, dict):
        raise ValueError("merge manifest lacks adapter or output evidence")
    if (
        base_verification.get("sha256") != base_hash
        or int(base_verification.get("bytes", -1)) != inputs["base_checkpoint"].stat().st_size
        or merge_adapter.get("sha256") != adapter_hash
        or merge_output.get("sha256") != merged_hash
    ):
        raise ValueError("checkpoint hashes do not match verification evidence")
    parameter_count = int(merge_adapter["parameter_count"])
    training_started = datetime.fromisoformat(str(training_config["started_at"]))
    training_completed = datetime.fromisoformat(str(training_result["completed_at"]))
    training_wall_seconds = (training_completed - training_started).total_seconds()
    worker_count = int(benchmark["worker_count"])
    samples_per_hour = float(benchmark["samples_per_hour_wall"])
    gpu_seconds_per_sample = float(benchmark["gpu_seconds_per_sample"])
    wall_seconds_per_sample = float(benchmark["wall_seconds_per_sample"])
    output.mkdir(parents=True)
    weights = output / "weights"
    weights.mkdir()
    promoted_adapter = weights / "action-adapter.safetensors"
    promoted_merged = weights / "merged-checkpoint.safetensors"
    os.link(inputs["adapter"], promoted_adapter)
    os.link(inputs["merged_checkpoint"], promoted_merged)
    model = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "honest_status": "WORKING",
        "model_name": args.model_name,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "architecture": (
            "Wan2.2-TI2V-5B + Boundless World Model step-12000 + "
            "WorldArena2 action-encoder adapter"
        ),
        "action_contract": {
            "type": "eef_abs",
            "dimensions": 14,
            "coordinate_frame": "robot_base:worldarena2-cobot-magic-max-end-pose",
            "history_frames": 9,
            "rollout_frames": 57,
        },
        "weights": {
            "base_model": str(inputs["base_model"]),
            "base_checkpoint": {
                "source": str(inputs["base_checkpoint"]),
                "bytes": inputs["base_checkpoint"].stat().st_size,
                "sha256": base_hash,
            },
            "adapter": {
                "path": str(promoted_adapter),
                "bytes": promoted_adapter.stat().st_size,
                "sha256": adapter_hash,
                "parameter_count": parameter_count,
            },
            "merged_checkpoint": {
                "path": str(promoted_merged),
                "bytes": promoted_merged.stat().st_size,
                "sha256": merged_hash,
                "storage": "hard link to the audited merge; no duplicate payload bytes",
            },
        },
        "training": {
            "run": str(inputs["training_run"]),
            "stage": training_config.get("stage"),
            "learning_rate": training_config.get("learning_rate"),
            "epochs": training_config.get("epochs"),
            "seed": training_config.get("seed"),
            "wall_seconds": training_wall_seconds,
            "trainable_parameter_count": parameter_count,
            "dataset": dataset.get("source"),
            "split_contract": dataset.get("config"),
        },
        "evaluation": {
            "validation": {
                "evidence": str(inputs["validation_evaluation"]),
                **_evaluation_summary(validation),
            },
            "task_disjoint_test": {
                "evidence": str(inputs["test_evaluation"]),
                **_evaluation_summary(test),
            },
        },
        "production_benchmark": {
            "evidence": str(inputs["benchmark_result"]),
            "physical_gpu_type": "NVIDIA A800-SXM4-80GB",
            "worker_count": worker_count,
            "samples": benchmark.get("samples_completed"),
            "generated_video_seconds": benchmark.get("generated_video_seconds"),
            "wall_seconds": benchmark.get("wall_seconds"),
            "gpu_seconds": benchmark.get("gpu_seconds"),
            "samples_per_hour": samples_per_hour,
            "wall_seconds_per_sample": wall_seconds_per_sample,
            "gpu_seconds_per_sample": gpu_seconds_per_sample,
            "gpu_hours_per_1000_samples": gpu_seconds_per_sample * 1000 / 3600,
            "wall_hours_per_1000_samples_at_measured_worker_count": 1000
            / samples_per_hour,
            "cost_formula": "GPU cost = sample_count * gpu_seconds_per_sample / 3600 * price_per_GPU_hour",
        },
        "source_license": dataset.get("source"),
        "limitations": [
            "Evaluation uses generated video versus lossy transfer-cache references, not physical robot execution.",
            "Pixel and optical-flow gates do not prove 3-D contact, force, collision safety, or task completion.",
            "The measured throughput applies to 57-frame, 20-step inference on A800 80GB GPUs.",
            "WorldArena2 end_pose is dataset-declared robot-base EEF data without independent calibration.",
        ],
    }
    _write_path = output / "model.json"
    _write_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    production = {
        "schema_version": "1.0.0",
        "model_manifest": str(_write_path),
        "batch_entrypoint": str(Path(__file__).resolve().parent / "run_bwm_factory_batch.py"),
        "checkpoint": str(promoted_merged),
        "base_model": str(inputs["base_model"]),
        "defaults": {
            "num_frames": 57,
            "num_inference_steps": 20,
            "fps": 24,
            "minimum_free_gpu_mib": 61000,
            "seed": 20260811,
        },
    }
    (output / "production.json").write_text(
        json.dumps(production, indent=2, sort_keys=True) + "\n"
    )
    readme = f"""# {args.model_name}

Status: WORKING for audited generated-video production; not real-robot execution.

This package combines the pinned official BWM checkpoint with a {parameter_count:,}-parameter
WorldArena2 action-encoder adapter. Validation and task-disjoint test gates both passed.

Measured on {worker_count} NVIDIA A800 80GB GPUs: {samples_per_hour:.2f} samples/hour,
{wall_seconds_per_sample:.3f} wall seconds/sample, and
{gpu_seconds_per_sample * 1000 / 3600:.3f} GPU-hours per 1,000 samples.

Use `production.json` with `scripts/run_bwm_factory_batch.py`. Exact hashes,
evaluation evidence, limitations, and the parameterized monetary-cost formula are in
`model.json`.
"""
    (output / "README.md").write_text(readme)
    print(json.dumps(model, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
