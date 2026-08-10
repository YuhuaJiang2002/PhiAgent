#!/usr/bin/env python3
"""Isolated GPU entry point for OSCAR, BWM, and Kinema4D.

This script is launched only after the PhiAgent adapter has inspected all
physical GPUs, selected one, and set ``CUDA_VISIBLE_DEVICES``.  Heavy model
imports stay inside the selected backend function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.acwm.schema import ACWMActionCondition, ActionRepresentation  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_requests(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.expanduser().resolve().read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("request manifest must contain a non-empty JSON array")
    required = {
        "case_id",
        "first_frame",
        "source_video",
        "condition",
        "prompt",
        "output",
        "seed",
        "num_inference_steps",
        "guidance_scale",
    }
    for item in payload:
        if not isinstance(item, dict) or required - item.keys():
            raise ValueError("every backend request must contain the complete contract")
    return payload


def _runtime_gpu() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be set by the PhiAgent preflight")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"selected physical GPU {visible} is not available to PyTorch")
    return {
        "cuda_visible_devices": visible,
        "logical_device": 0,
        "device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def _metadata(
    *, backend: str, item: dict[str, Any], output: Path, runtime: dict[str, Any]
) -> Path:
    path = output.with_suffix(".metadata.json")
    _write_json(
        path,
        {
            "schema_version": "1.0.0",
            "status": "generated_pending_evaluation_and_human_review",
            "backend": backend,
            "case_id": item["case_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "first_frame": item["first_frame"],
            "first_frame_sha256": _sha256(Path(item["first_frame"])),
            "source_video": item["source_video"],
            "source_video_sha256": _sha256(Path(item["source_video"])),
            "condition": item["condition"],
            "condition_sha256": _sha256(Path(item["condition"])),
            "prompt": item["prompt"],
            "seed": item["seed"],
            "num_inference_steps": item["num_inference_steps"],
            "guidance_scale": item["guidance_scale"],
            "output": str(output),
            "output_sha256": _sha256(output),
            "runtime": runtime,
        },
    )
    return path


def _run_oscar(args: argparse.Namespace, requests: list[dict[str, Any]]) -> list[dict[str, str]]:
    runtime = _runtime_gpu()
    # OSCAR initializes a one-rank NCCL group even through its library wrapper.
    # torchrun normally supplies these variables; the isolated batch runner
    # provides the equivalent explicit single-process contract.
    distributed_defaults = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(29500 + os.getpid() % 1000),
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
    }
    for name, value in distributed_defaults.items():
        os.environ.setdefault(name, value)
    runtime["distributed"] = {
        name.lower(): os.environ[name] for name in distributed_defaults
    }
    repository = args.repository.expanduser().resolve()
    sys.path.insert(0, str(repository))
    from oscar_diffsynth import OSCARDiffSynthPipeline

    import imageio.v3 as iio
    import numpy as np

    pipe = OSCARDiffSynthPipeline.from_dcp(str(args.checkpoint.expanduser().resolve()))
    results: list[dict[str, str]] = []
    for item in requests:
        condition = ACWMActionCondition.from_json(Path(item["condition"]))
        if condition.representation is not ActionRepresentation.KINEMATIC_SKELETON_2D:
            raise ValueError("OSCAR requires kinematic_skeleton_2d")
        assert condition.visual_condition is not None
        generated = pipe(
            first_frame=item["first_frame"],
            skeleton_video=condition.visual_condition,
            prompt=item["prompt"],
            num_inference_steps=int(item["num_inference_steps"]),
            guidance_scale=float(item["guidance_scale"]),
            seed=int(item["seed"]),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=args.fps,
        )
        output = Path(item["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        frames = np.stack(generated.frames)
        iio.imwrite(
            output,
            frames,
            plugin="FFMPEG",
            fps=args.fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        )
        metadata = _metadata(backend="oscar", item=item, output=output, runtime=runtime)
        results.append(
            {"case_id": str(item["case_id"]), "output": str(output), "metadata": str(metadata)}
        )
    return results


def _bwm_action_type(representation: ActionRepresentation) -> tuple[str, str]:
    mapping = {
        ActionRepresentation.EEF_ABSOLUTE: ("eef_abs", "observation.state"),
        ActionRepresentation.EEF_DELTA: ("eef_delta", "action"),
        ActionRepresentation.JOINT_ABSOLUTE: ("joint_abs", "observation.state"),
        ActionRepresentation.JOINT_DELTA: ("joint_delta", "action"),
    }
    try:
        return mapping[representation]
    except KeyError as exc:
        raise ValueError("BWM requires robot-base EEF or joint actions") from exc


def _run_bwm(args: argparse.Namespace, requests: list[dict[str, Any]]) -> list[dict[str, str]]:
    runtime = _runtime_gpu()
    import pyarrow as pa
    import pyarrow.parquet as pq

    experiment = args.result_manifest.parent / "bwm-staging"
    data = experiment / "data"
    raw_outputs = experiment / "raw-outputs"
    data.mkdir(parents=True, exist_ok=False)
    raw_outputs.mkdir(parents=True)
    metadata_path = experiment / "episodes.jsonl"
    metadata_rows: list[dict[str, Any]] = []
    action_type: str | None = None
    for index, item in enumerate(requests):
        condition = ACWMActionCondition.from_json(Path(item["condition"]))
        current_type, column = _bwm_action_type(condition.representation)
        if action_type is not None and current_type != action_type:
            raise ValueError("one BWM batch must use one action representation")
        action_type = current_type
        if len(condition.channels) != 14:
            raise ValueError("the public BWM checkpoint requires 14 action channels")
        image_name = f"{item['case_id']}.png"
        action_name = f"{item['case_id']}.parquet"
        shutil.copy2(item["first_frame"], data / image_name)
        rows = [list(row) for row in condition.values]
        table = pa.table({column: pa.array(rows, type=pa.list_(pa.float32()))})
        pq.write_table(table, data / action_name)
        metadata_rows.append(
            {
                "episode_index": index,
                "length": len(rows),
                "start_frame": 0,
                "end_frame": len(rows) - 1,
                "video": image_name,
                "action": action_name,
                "prompt": item["prompt"],
                "task": item["case_id"],
            }
        )
    metadata_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata_rows)
    )
    if len({int(item["seed"]) for item in requests}) != 1:
        raise ValueError("one BWM batch must use one matched seed")
    command = [
        sys.executable,
        str(args.repository / "scripts" / "infer.py"),
        "--config",
        str(args.config),
        "--model_paths",
        str(args.base_model),
        "--ckpt_path",
        str(args.checkpoint),
        "--dataset_base_path",
        str(data),
        "--dataset_metadata_path",
        str(metadata_path),
        "--action_stat_path",
        str(args.action_stats),
        "--action_type",
        str(action_type),
        "--output_path",
        str(raw_outputs),
        "--max_samples",
        str(len(requests)),
        "--seed",
        str(requests[0]["seed"]),
    ]
    subprocess.run(command, cwd=args.repository, check=True)
    results: list[dict[str, str]] = []
    for index, item in enumerate(requests):
        raw = raw_outputs / f"episode{index}.mp4"
        if not raw.is_file():
            raise RuntimeError(f"BWM did not produce {raw}")
        output = Path(item["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw, output)
        metadata = _metadata(backend="bwm", item=item, output=output, runtime=runtime)
        results.append(
            {"case_id": str(item["case_id"]), "output": str(output), "metadata": str(metadata)}
        )
    return results


def _run_kinema4d(
    args: argparse.Namespace, requests: list[dict[str, Any]]
) -> list[dict[str, str]]:
    runtime = _runtime_gpu()
    experiment = args.result_manifest.parent / "kinema4d-staging"
    data = experiment / "dataset"
    video_dir = data / "videos"
    frame_dir = data / "first_frames"
    raw_outputs = experiment / "raw-outputs"
    video_dir.mkdir(parents=True, exist_ok=False)
    frame_dir.mkdir(parents=True)
    raw_outputs.mkdir(parents=True)
    episode_list = experiment / "episodes.txt"
    lines: list[str] = []
    for item in requests:
        condition = ACWMActionCondition.from_json(Path(item["condition"]))
        if condition.representation is not ActionRepresentation.ROBOT_POINTMAP:
            raise ValueError("Kinema4D requires robot_pointmap")
        if condition.visual_condition is None:
            raise ValueError("Kinema4D requires a robot RGB+pointmap condition video")
        name = str(item["case_id"])
        shutil.copy2(condition.visual_condition, video_dir / f"{name}.mp4")
        shutil.copy2(item["first_frame"], frame_dir / f"{name}.png")
        lines.append(f"videos/{name}.mp4")
    episode_list.write_text("\n".join(lines) + "\n")
    command = [
        sys.executable,
        str(args.repository / "inference.py"),
        "--data_path",
        str(data) + os.sep,
        "--video",
        str(episode_list),
        "--out",
        str(raw_outputs),
        "--sft_path",
        str(args.base_model),
        "--lora_path",
        str(args.checkpoint),
        "--lora_rank",
        "64",
        "--type",
        "i2vwbw-demb-samerope-act",
        "--mode",
        args.kinema_mode,
    ]
    subprocess.run(command, cwd=args.repository, check=True)
    results: list[dict[str, str]] = []
    for item in requests:
        raw = raw_outputs / "videos" / f"{item['case_id']}.mp4"
        if not raw.is_file():
            raise RuntimeError(f"Kinema4D did not produce {raw}")
        output = Path(item["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw, output)
        metadata = _metadata(backend="kinema4d", item=item, output=output, runtime=runtime)
        results.append(
            {"case_id": str(item["case_id"]), "output": str(output), "metadata": str(metadata)}
        )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("oscar", "bwm", "kinema4d"))
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--result-manifest", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--action-stats", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--episode-list", type=Path)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--kinema-mode", default="xyzrgb")
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.request_manifest = args.request_manifest.expanduser().resolve()
    args.result_manifest = args.result_manifest.expanduser().resolve()
    args.repository = args.repository.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    for name in ("base_model", "action_stats", "config", "dataset_root", "episode_list"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    requests = _load_requests(args.request_manifest)
    if args.backend == "oscar":
        results = _run_oscar(args, requests)
    elif args.backend == "bwm":
        if args.base_model is None or args.action_stats is None or args.config is None:
            raise ValueError("BWM requires --base-model, --action-stats, and --config")
        results = _run_bwm(args, requests)
    else:
        if args.base_model is None:
            raise ValueError("Kinema4D requires --base-model")
        results = _run_kinema4d(args, requests)
    _write_json(args.result_manifest, results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
