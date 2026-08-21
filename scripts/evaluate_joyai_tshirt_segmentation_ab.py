#!/usr/bin/env python3
"""Run frozen SAM2 and shadow SAM3.1 T-shirt tracking concurrently."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
from typing import Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.evaluation.segmentation_ab import (  # noqa: E402
    AB_SCHEMA_VERSION,
    SAM2_MODEL_ID,
    SAM31_MODEL_ID,
    capture_git_state,
    compare_tracker_results,
    load_json_object,
    parse_tracker_spec,
    validate_sam31_config,
    validate_result_mask_artifact,
    validate_task_config,
)
from phiagent.evaluation.video_proxy import file_sha256  # noqa: E402
from phiagent.rendering.wan_animate import (  # noqa: E402
    GPUInfo,
    PreflightError,
    acquire_gpu_lease,
    query_gpus,
    select_gpu,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--sam31-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam2-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--sam31-python", type=Path, required=True)
    parser.add_argument("--sam2-gpu", type=int)
    parser.add_argument("--sam31-gpu", type=int)
    parser.add_argument("--sam2-minimum-free-gpu-mib", type=int, default=12 * 1024)
    parser.add_argument("--sam31-minimum-free-gpu-mib", type=int, default=32 * 1024)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_python(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    absolute = Path(os.path.abspath(expanded))
    if not absolute.is_file() or not os.access(absolute, os.X_OK):
        raise FileNotFoundError(f"{label} is not an executable file: {absolute}")
    return absolute


def select_parallel_gpus(
    gpus: Sequence[GPUInfo],
    *,
    sam2_requested: int | None,
    sam31_requested: int | None,
    sam2_minimum_free_mib: int,
    sam31_minimum_free_mib: int,
) -> tuple[GPUInfo, GPUInfo]:
    """Select distinct physical GPUs, reserving explicit requests first."""

    if sam2_minimum_free_mib < 1 or sam31_minimum_free_mib < 1:
        raise ValueError("minimum free GPU memory must be positive")
    if sam2_requested is not None and sam2_requested == sam31_requested:
        raise ValueError("parallel A/B requires two distinct physical GPUs")

    if sam31_requested is None:
        candidates = [gpu for gpu in gpus if gpu.physical_index != sam2_requested]
        sam31 = select_gpu(candidates, None, sam31_minimum_free_mib)
    else:
        sam31 = select_gpu(gpus, sam31_requested, sam31_minimum_free_mib)

    if sam2_requested is None:
        candidates = [gpu for gpu in gpus if gpu.physical_index != sam31.physical_index]
        sam2 = select_gpu(candidates, None, sam2_minimum_free_mib)
    else:
        sam2 = select_gpu(gpus, sam2_requested, sam2_minimum_free_mib)
    if sam2.physical_index == sam31.physical_index:
        raise ValueError("parallel A/B selected the same physical GPU twice")
    return sam2, sam31


@contextmanager
def hold_parallel_gpu_leases(
    physical_indices: Sequence[int],
) -> Iterator[dict[int, str]]:
    """Hold distinct physical-GPU leases in deterministic index order."""

    if len(physical_indices) != 2 or len(set(physical_indices)) != 2:
        raise ValueError("parallel A/B requires exactly two distinct GPU leases")
    handles = []
    paths: dict[int, str] = {}
    try:
        for physical_index in sorted(physical_indices):
            path, handle = acquire_gpu_lease(physical_index)
            handles.append(handle)
            paths[physical_index] = str(path)
        yield paths
    finally:
        for handle in reversed(handles):
            handle.close()


def _prepare_inputs(
    *,
    video: Path,
    task_config_path: Path,
    task_config: Mapping[str, object],
    prepared_dir: Path,
) -> Path:
    import cv2
    import numpy as np

    prepared_dir.mkdir()
    frame_dir = prepared_dir / "frames"
    frame_dir.mkdir()
    scoring_frame_dir = prepared_dir / "scoring-frames"
    scoring_frame_dir.mkdir()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    frame_records = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index = len(frames)
        frame_path = frame_dir / f"{frame_index:05d}.jpg"
        scoring_frame_path = scoring_frame_dir / f"{frame_index:05d}.png"
        if not cv2.imwrite(str(frame_path), frame, (cv2.IMWRITE_JPEG_QUALITY, 98)):
            raise RuntimeError(f"failed to write shared frame: {frame_path}")
        if not cv2.imwrite(str(scoring_frame_path), frame):
            raise RuntimeError(f"failed to write lossless scoring frame: {scoring_frame_path}")
        frames.append(frame)
        frame_records.append(
            {
                "index": frame_index,
                "path": str(frame_path),
                "sha256": file_sha256(frame_path),
                "scoring_path": str(scoring_frame_path),
                "scoring_sha256": file_sha256(scoring_frame_path),
            }
        )
    capture.release()

    expected_count = int(task_config["frame_count"])
    expected_width, expected_height = task_config.get("frame_size", [1248, 720])
    expected_fps = float(task_config.get("fps", 24.0))
    if len(frames) != expected_count:
        raise ValueError(f"video has {len(frames)} frames; expected {expected_count}")
    if (width, height) != (expected_width, expected_height):
        raise ValueError(
            f"video size {(width, height)} differs from {(expected_width, expected_height)}"
        )
    if not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError(f"video FPS {fps} differs from frozen {expected_fps}")

    refinement = task_config["initial_mask_refinement"]
    initial_frame_index = int(task_config.get("initial_frame_index", 0))
    first_gray = cv2.cvtColor(frames[initial_frame_index], cv2.COLOR_BGR2GRAY)
    closing_kernel_pixels = int(refinement["closing_kernel_pixels"])
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (closing_kernel_pixels, closing_kernel_pixels),
    )
    masks_payload = {}
    object_records = []
    for name, raw in task_config["objects"].items():
        object_id = int(raw["object_id"])
        maximum_luma = int(raw.get("maximum_luma", refinement["maximum_luma"]))
        if not 0 <= maximum_luma <= 255:
            raise ValueError(f"{name} maximum_luma must be in [0, 255]")
        polygon = np.asarray(raw["initial_polygon_xy"], dtype=np.int32)
        polygon_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon], 1)
        mask = ((polygon_mask > 0) & (first_gray <= maximum_luma)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            raise ValueError(f"{name} initial mask has no connected component")
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
        area = int(np.count_nonzero(mask))
        if area < int(refinement["minimum_area_pixels"]):
            raise ValueError(f"{name} refined initial mask is too small")
        key = f"object_{object_id}"
        masks_payload[key] = mask
        object_records.append(
            {
                "name": name,
                "object_id": object_id,
                "mask_key": key,
                "initial_area_pixels": area,
                "maximum_luma": maximum_luma,
            }
        )
    initial_masks_path = prepared_dir / "initial-masks.npz"
    np.savez_compressed(initial_masks_path, **masks_payload)
    manifest = {
        "schema_version": AB_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": str(video),
        "video_sha256": file_sha256(video),
        "task_config": str(task_config_path),
        "task_config_sha256": file_sha256(task_config_path),
        "frame_count": len(frames),
        "frame_size": [width, height],
        "fps": fps,
        "jpeg_quality": 98,
        "frame_dir": str(frame_dir),
        "scoring_frame_dir": str(scoring_frame_dir),
        "frames": frame_records,
        "initial_frame_index": initial_frame_index,
        "initial_masks": str(initial_masks_path),
        "initial_masks_sha256": file_sha256(initial_masks_path),
        "objects": sorted(object_records, key=lambda item: item["object_id"]),
        "packages": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    }
    manifest_path = prepared_dir / "input-manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def build_worker_command(
    *,
    python: Path,
    model_id: str,
    prepared_dir: Path,
    task_config: Path,
    model_config: Path,
    output_dir: Path,
    gpu: GPUInfo,
    minimum_free_gpu_mib: int,
    seed: int,
) -> list[str]:
    """Build one shell-free model worker command."""

    if model_id not in {SAM2_MODEL_ID, SAM31_MODEL_ID}:
        raise ValueError(f"unsupported worker model: {model_id}")
    return [
        str(python),
        str(PROJECT_ROOT / "scripts" / "evaluate_joyai_tshirt_segmentation_worker.py"),
        "--model",
        model_id,
        "--prepared-dir",
        str(prepared_dir),
        "--task-config",
        str(task_config),
        "--model-config",
        str(model_config),
        "--output-dir",
        str(output_dir),
        "--gpu",
        str(gpu.physical_index),
        "--minimum-free-gpu-mib",
        str(minimum_free_gpu_mib),
        "--seed",
        str(seed),
    ]


def _launch_workers(
    commands: Mapping[str, Sequence[str]],
    logs_dir: Path,
) -> tuple[dict[str, int], dict[str, dict[str, str]], float]:
    processes: dict[str, subprocess.Popen[str]] = {}
    streams = {}
    return_codes: dict[str, int] = {}
    launch_errors: dict[str, dict[str, str]] = {}
    started = time.perf_counter()
    for model_id, command in commands.items():
        stdout = None
        stderr = None
        try:
            stdout = (logs_dir / f"{model_id}.stdout.log").open("w", encoding="utf-8")
            stderr = (logs_dir / f"{model_id}.stderr.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
        except OSError as error:
            launch_errors[model_id] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            return_codes[model_id] = 127
            if stderr is not None:
                stderr.write(f"{type(error).__name__}: {error}\n")
            for stream in (stdout, stderr):
                if stream is not None:
                    stream.close()
            continue
        streams[model_id] = (stdout, stderr)
        processes[model_id] = process

    try:
        for model_id, process in processes.items():
            try:
                return_codes[model_id] = process.wait()
            except OSError as error:
                launch_errors[model_id] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                return_codes[model_id] = 126
                if process.poll() is None:
                    process.terminate()
                    process.wait()
    except KeyboardInterrupt:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            if process.poll() is None:
                process.wait()
        raise
    finally:
        for stdout, stderr in streams.values():
            stdout.close()
            stderr.close()
    return return_codes, launch_errors, time.perf_counter() - started


def classify_worker_outcome(return_codes: Mapping[str, int]) -> str:
    """Classify completion without giving the shadow model veto power."""

    if set(return_codes) != {SAM2_MODEL_ID, SAM31_MODEL_ID}:
        raise ValueError("worker return codes must contain SAM2 and SAM3.1")
    if return_codes[SAM2_MODEL_ID] != 0:
        return "authoritative_failed"
    if return_codes[SAM31_MODEL_ID] != 0:
        return "shadow_failed"
    return "complete"


def _available_log_records(
    logs_dir: Path,
    model_ids: Sequence[str],
) -> dict[str, object]:
    records = {}
    for model_id in model_ids:
        model_logs = {}
        for stream in ("stdout", "stderr"):
            path = logs_dir / f"{model_id}.{stream}.log"
            if path.is_file():
                model_logs[stream] = {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
        records[model_id] = model_logs
    return records


def _record_failure(
    *,
    run_path: Path,
    run_record: dict[str, object],
    phase: str,
    error: BaseException | str,
    status: str,
    hard_gates_passed: object,
) -> None:
    run_record.update(
        {
            "status": status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "hard_gates_passed": hard_gates_passed,
            "promotion_eligible": False,
            "failure": {
                "phase": phase,
                "type": type(error).__name__ if isinstance(error, BaseException) else None,
                "message": str(error),
            },
        }
    )
    _write_json(run_path, run_record)


def main() -> int:
    args = _parser().parse_args()
    video = args.video.expanduser().resolve()
    task_config_path = args.task_config.expanduser().resolve()
    sam31_config_path = args.sam31_config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if not task_config_path.is_file():
        raise FileNotFoundError(task_config_path)
    if not sam31_config_path.is_file():
        raise FileNotFoundError(sam31_config_path)
    if output.exists():
        raise FileExistsError(f"refusing to reuse A/B run directory: {output}")
    sam2_python = _require_python(args.sam2_python, "SAM2 Python")
    sam31_python = _require_python(args.sam31_python, "SAM3.1 Python")
    task_config = load_json_object(task_config_path, "task config")
    sam31_config = load_json_object(sam31_config_path, "SAM3.1 config")
    validate_task_config(task_config)
    validate_sam31_config(sam31_config)
    parse_tracker_spec(
        task_config["sam2"],
        project_root=PROJECT_ROOT,
        model_id=SAM2_MODEL_ID,
    )
    parse_tracker_spec(
        sam31_config,
        project_root=PROJECT_ROOT,
        model_id=SAM31_MODEL_ID,
    )

    output.mkdir(parents=True)
    configs_dir = output / "configs"
    configs_dir.mkdir()
    copied_task_config = configs_dir / "task.json"
    copied_sam31_config = configs_dir / "sam31.json"
    shutil.copy2(task_config_path, copied_task_config)
    shutil.copy2(sam31_config_path, copied_sam31_config)
    logs_dir = output / "logs"
    logs_dir.mkdir()
    prepared_dir = output / "shared-input"
    prepared_manifest = _prepare_inputs(
        video=video,
        task_config_path=copied_task_config,
        task_config=task_config,
        prepared_dir=prepared_dir,
    )

    gpus, inventory, gpu_processes = query_gpus()
    sam2_gpu, sam31_gpu = select_parallel_gpus(
        gpus,
        sam2_requested=args.sam2_gpu,
        sam31_requested=args.sam31_gpu,
        sam2_minimum_free_mib=args.sam2_minimum_free_gpu_mib,
        sam31_minimum_free_mib=args.sam31_minimum_free_gpu_mib,
    )
    sam2_output = output / SAM2_MODEL_ID
    sam31_output = output / SAM31_MODEL_ID
    commands = {
        SAM2_MODEL_ID: build_worker_command(
            python=sam2_python,
            model_id=SAM2_MODEL_ID,
            prepared_dir=prepared_dir,
            task_config=copied_task_config,
            model_config=copied_task_config,
            output_dir=sam2_output,
            gpu=sam2_gpu,
            minimum_free_gpu_mib=args.sam2_minimum_free_gpu_mib,
            seed=args.seed,
        ),
        SAM31_MODEL_ID: build_worker_command(
            python=sam31_python,
            model_id=SAM31_MODEL_ID,
            prepared_dir=prepared_dir,
            task_config=copied_task_config,
            model_config=copied_sam31_config,
            output_dir=sam31_output,
            gpu=sam31_gpu,
            minimum_free_gpu_mib=args.sam31_minimum_free_gpu_mib,
            seed=args.seed,
        ),
    }
    started_at = datetime.now(timezone.utc)
    run_record = {
        "schema_version": AB_SCHEMA_VERSION,
        "status": "WAITING_FOR_GPU_LEASE",
        "started_at": started_at.isoformat(),
        "command": list(sys.argv),
        "video": str(video),
        "video_sha256": file_sha256(video),
        "prepared_input": str(prepared_manifest),
        "prepared_input_sha256": file_sha256(prepared_manifest),
        "task_config": str(copied_task_config),
        "task_config_sha256": file_sha256(copied_task_config),
        "sam31_config": str(copied_sam31_config),
        "sam31_config_sha256": file_sha256(copied_sam31_config),
        "commands": commands,
        "selected_gpus": {
            SAM2_MODEL_ID: asdict(sam2_gpu),
            SAM31_MODEL_ID: asdict(sam31_gpu),
        },
        "gpu_inventory": inventory,
        "gpu_processes": gpu_processes,
        "git": capture_git_state(PROJECT_ROOT),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "seed": args.seed,
        "decision_policy": "sam2_authoritative_sam31_shadow",
    }
    run_path = output / "run.json"
    _write_json(run_path, run_record)

    try:
        with hold_parallel_gpu_leases(
            (sam2_gpu.physical_index, sam31_gpu.physical_index)
        ) as gpu_leases:
            leased_gpus, leased_inventory, leased_processes = query_gpus()
            sam2_gpu = select_gpu(
                leased_gpus,
                sam2_gpu.physical_index,
                args.sam2_minimum_free_gpu_mib,
            )
            sam31_gpu = select_gpu(
                leased_gpus,
                sam31_gpu.physical_index,
                args.sam31_minimum_free_gpu_mib,
            )
            run_record.update(
                {
                    "status": "RUNNING",
                    "selected_gpus": {
                        SAM2_MODEL_ID: asdict(sam2_gpu),
                        SAM31_MODEL_ID: asdict(sam31_gpu),
                    },
                    "gpu_leases": gpu_leases,
                    "post_lease_gpu_inventory": leased_inventory,
                    "post_lease_gpu_processes": leased_processes,
                }
            )
            _write_json(run_path, run_record)
            return_codes, launch_errors, parallel_elapsed = _launch_workers(commands, logs_dir)
    except (OSError, PreflightError, KeyboardInterrupt) as error:
        run_record["logs"] = _available_log_records(logs_dir, tuple(commands))
        _record_failure(
            run_path=run_path,
            run_record=run_record,
            phase="gpu_lease_revalidation_or_worker_wait",
            error=error,
            status="BLOCKED",
            hard_gates_passed=None,
        )
        raise
    completed_at = datetime.now(timezone.utc)
    log_records = _available_log_records(logs_dir, tuple(commands))
    run_record.update(
        {
            "completed_at": completed_at.isoformat(),
            "parallel_elapsed_seconds": parallel_elapsed,
            "worker_return_codes": return_codes,
            "worker_launch_errors": launch_errors,
            "logs": log_records,
        }
    )
    outcome = classify_worker_outcome(return_codes)
    if outcome == "authoritative_failed":
        _record_failure(
            run_path=run_path,
            run_record=run_record,
            phase="sam2_worker",
            error="the authoritative SAM2 worker failed; inspect bound logs",
            status="BLOCKED",
            hard_gates_passed=None,
        )
        print(json.dumps(run_record, indent=2, sort_keys=True))
        return 2

    sam2_result_path = sam2_output / "result.json"
    try:
        sam2_result = load_json_object(sam2_result_path, "SAM2 result")
        authoritative_masks_path, authoritative_masks_sha256 = validate_result_mask_artifact(
            sam2_result,
            expected_model_id=SAM2_MODEL_ID,
        )
        if sam2_result.get("decision_eligible") is not True:
            raise ValueError("SAM2 result is not marked decision eligible")
        if sam2_result.get("prepared_input_sha256") != run_record["prepared_input_sha256"]:
            raise ValueError("SAM2 result binds a different prepared input")
        if sam2_result.get("task_config_sha256") != run_record["task_config_sha256"]:
            raise ValueError("SAM2 result binds a different task config")
        authoritative_sha256 = file_sha256(sam2_result_path)
        authoritative_hard_gates = sam2_result["hard_gates_passed"]
        if type(authoritative_hard_gates) is not bool:
            raise ValueError("SAM2 hard_gates_passed must be boolean")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        _record_failure(
            run_path=run_path,
            run_record=run_record,
            phase="sam2_result_validation",
            error=error,
            status="BLOCKED",
            hard_gates_passed=None,
        )
        print(json.dumps(run_record, indent=2, sort_keys=True))
        return 2
    run_record.update(
        {
            "hard_gates_passed": authoritative_hard_gates,
            "authoritative_result": str(sam2_result_path),
            "authoritative_result_sha256": authoritative_sha256,
            "authoritative_masks": str(authoritative_masks_path),
            "authoritative_masks_sha256": authoritative_masks_sha256,
        }
    )
    if outcome == "shadow_failed":
        _record_failure(
            run_path=run_path,
            run_record=run_record,
            phase="sam31_worker",
            error=(
                "the SAM3.1 shadow worker failed; the SAM2 decision remains "
                "available and authoritative"
            ),
            status="PARTIAL",
            hard_gates_passed=authoritative_hard_gates,
        )
        run_record.update(
            {
                "shadow_result": None,
                "comparison": None,
                "shadow_status": "FAILED",
            }
        )
        _write_json(run_path, run_record)
        print(json.dumps(run_record, indent=2, sort_keys=True))
        return 3

    sam31_result_path = sam31_output / "result.json"
    try:
        sam31_result = load_json_object(sam31_result_path, "SAM3.1 result")
        shadow_sha256 = file_sha256(sam31_result_path)
        comparison = compare_tracker_results(sam2_result, sam31_result)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        _record_failure(
            run_path=run_path,
            run_record=run_record,
            phase="sam31_result_or_comparison_validation",
            error=error,
            status="PARTIAL",
            hard_gates_passed=authoritative_hard_gates,
        )
        run_record.update(
            {
                "shadow_result": (str(sam31_result_path) if sam31_result_path.exists() else None),
                "comparison": None,
                "shadow_status": "INVALID",
            }
        )
        _write_json(run_path, run_record)
        print(json.dumps(run_record, indent=2, sort_keys=True))
        return 3
    comparison.update(
        {
            "created_at": completed_at.isoformat(),
            "sam2_result": str(sam2_result_path),
            "sam2_result_sha256": authoritative_sha256,
            "sam31_result": str(sam31_result_path),
            "sam31_result_sha256": shadow_sha256,
        }
    )
    comparison_path = output / "comparison.json"
    _write_json(comparison_path, comparison)
    run_record.update(
        {
            "status": "PARTIAL",
            "shadow_result": str(sam31_result_path),
            "shadow_result_sha256": shadow_sha256,
            "shadow_status": "COMPLETED",
            "comparison": str(comparison_path),
            "comparison_sha256": file_sha256(comparison_path),
            "promotion_eligible": False,
            "evidence_boundary": (
                "SAM2 remains the only decision-bearing evaluator. SAM3.1 "
                "reports agreement and diagnostics only until labeled-keyframe "
                "validation and a separate evaluator epoch register its thresholds."
            ),
        }
    )
    _write_json(run_path, run_record)
    print(json.dumps(run_record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
