#!/usr/bin/env python3
"""Estimate metric depth and camera state from a video using pinned DA3 Nested."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_flower_robot_demo import _gpu_inventory, _select_gpu  # noqa: E402


DA3_REVISION = "41736238f5bced4debf3f2a12375d2466874866d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *command], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_optional(command: list[str], cwd: Path) -> str:
    try:
        return _git(command, cwd)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "NOT_AVAILABLE_ON_EXECUTION_HOST"


def _versions(names: tuple[str, ...]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--da3-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=50000)
    parser.add_argument("--sample-hz", type=float, default=2.0)
    parser.add_argument("--maximum-frames", type=int, default=12)
    parser.add_argument(
        "--frame-indices",
        default="",
        help="Optional comma-separated exact source-frame indices; overrides sampling",
    )
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--use-ray-pose", action="store_true")
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def _extract(
    cv2: Any,
    source: Path,
    sample_hz: float,
    maximum_frames: int,
    requested_indices: tuple[int, ...],
) -> tuple[list[Any], list[int], dict[str, Any]]:
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not capture.isOpened() or fps <= 0 or total <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("source video metadata is invalid")
    interval = max(1, int(round(fps / sample_hz)))
    candidates = list(requested_indices) if requested_indices else list(range(0, total, interval))
    if any(index < 0 or index >= total for index in candidates):
        raise ValueError("an exact DA3 source frame index lies outside the video")
    if candidates != sorted(set(candidates)):
        raise ValueError("exact DA3 source frame indices must be unique and increasing")
    if not requested_indices and maximum_frames and len(candidates) > maximum_frames:
        positions = [
            round(index * (len(candidates) - 1) / (maximum_frames - 1))
            for index in range(maximum_frames)
        ] if maximum_frames > 1 else [0]
        selected_indices = {candidates[position] for position in positions}
    else:
        selected_indices = set(candidates)
    frames: list[Any] = []
    indices: list[int] = []
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if index in selected_indices:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                indices.append(index)
            index += 1
    finally:
        capture.release()
    return frames, indices, {
        "fps": fps,
        "frames": total,
        "width": width,
        "height": height,
        "sample_interval_frames": interval,
        "effective_sample_hz": fps / interval,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.sample_hz <= 0 or args.maximum_frames < 0 or args.process_res <= 0:
        raise ValueError("sampling and processing parameters are invalid")
    exact_indices = tuple(
        int(value) for value in args.frame_indices.split(",") if value.strip()
    )
    source = args.source.expanduser().resolve()
    repository = args.da3_repo.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source.is_file() or not (repository / "src" / "depth_anything_3").is_dir():
        raise ValueError("source video or DA3 repository is missing")
    if not (checkpoint / "model.safetensors").is_file():
        raise ValueError(f"DA3 checkpoint is incomplete: {checkpoint}")
    revision = _git(["rev-parse", "HEAD"], repository)
    if revision != DA3_REVISION:
        raise RuntimeError(f"DA3 revision {revision} != pinned {DA3_REVISION}")
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite experiment directory: {output_dir}")
    output_dir.mkdir(parents=True)

    inventory = _gpu_inventory()
    selected = _select_gpu(inventory, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    sys.path.insert(0, str(repository / "src"))

    import cv2
    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3

    if not torch.cuda.is_available():
        raise RuntimeError("DA3 GPU entrypoint requires CUDA after device selection")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    extraction_started = time.perf_counter()
    frames, frame_indices, video = _extract(
        cv2, source, args.sample_hz, args.maximum_frames, exact_indices
    )
    extraction_seconds = time.perf_counter() - extraction_started
    if len(frames) < 2:
        raise RuntimeError("DA3 camera estimation requires at least two sampled frames")

    load_started = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(checkpoint)).to(device=device)
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats(device)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        prediction = model.inference(
            frames,
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
            use_ray_pose=args.use_ray_pose,
            ref_view_strategy="middle",
        )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    peak_memory_mib = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    depth = np.asarray(prediction.depth, dtype=np.float32)
    confidence = np.asarray(prediction.conf, dtype=np.float32)
    intrinsics = np.asarray(prediction.intrinsics, dtype=np.float32)
    camera_from_world = np.asarray(prediction.extrinsics, dtype=np.float32)
    processed = np.asarray(prediction.processed_images, dtype=np.uint8)
    if camera_from_world.shape == (len(frames), 3, 4):
        bottom = np.repeat(
            np.asarray([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32),
            len(frames),
            axis=0,
        )
        camera_from_world_h = np.concatenate((camera_from_world, bottom), axis=1)
    elif camera_from_world.shape == (len(frames), 4, 4):
        camera_from_world_h = camera_from_world
    else:
        raise RuntimeError(f"DA3 extrinsics have invalid shape {camera_from_world.shape}")
    world_from_camera = np.linalg.inv(camera_from_world_h).astype(np.float32)
    expected_depth_shape = (len(frames), processed.shape[1], processed.shape[2])
    if depth.shape != expected_depth_shape or confidence.shape != expected_depth_shape:
        raise RuntimeError(
            f"DA3 depth/confidence shapes {depth.shape}/{confidence.shape} != {expected_depth_shape}"
        )
    if intrinsics.shape != (len(frames), 3, 3):
        raise RuntimeError(f"DA3 intrinsics have invalid shape {intrinsics.shape}")

    artifact = output_dir / "da3-metric-camera-samples.npz"
    np.savez_compressed(
        artifact,
        source_frame_indices=np.asarray(frame_indices, dtype=np.int32),
        processed_images_rgb=processed,
        depth_m=depth.astype(np.float16),
        confidence=confidence.astype(np.float16),
        intrinsics_px=intrinsics,
        camera_from_world=camera_from_world_h.astype(np.float32),
        world_from_camera=world_from_camera,
    )
    valid = np.isfinite(depth) & (depth > 0)
    rotations = world_from_camera[:, :3, :3].astype(np.float64)
    orthogonality = np.linalg.norm(
        np.swapaxes(rotations, 1, 2) @ rotations - np.eye(3), axis=(1, 2)
    )
    checkpoint_file = checkpoint / "model.safetensors"
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "status_reason": (
            "DA3 Nested supplies learned metric depth and camera state, but the scale "
            "uncertainty still requires an independent calibration or cross-model bound"
        ),
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": _versions(
            ("depth-anything-3", "numpy", "opencv-python", "torch", "torchvision", "xformers")
        ),
        "seed": args.seed,
        "git": {
            "project_head": _git_optional(["rev-parse", "HEAD"], PROJECT_ROOT),
            "project_status": _git_optional(["status", "--short"], PROJECT_ROOT),
            "da3_status": _git(["status", "--short"], repository),
        },
        "gpu": {
            "physical_index": args.gpu,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "inventory_before": inventory,
            "selected": selected,
            "peak_allocated_memory_mib": peak_memory_mib,
        },
        "model": {
            "name": "DA3NESTED-GIANT-LARGE-1.1",
            "repository": "https://github.com/ByteDance-Seed/Depth-Anything-3",
            "revision": revision,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint_file),
            "license": "CC-BY-NC-4.0",
            "evidence_class": "foundation_model_estimate",
        },
        "input": {"path": str(source), "sha256": _sha256(source), **video},
        "sampling": {
            "requested_hz": args.sample_hz,
            "sampled_frames": len(frames),
            "source_frame_indices": frame_indices,
            "maximum_frames": args.maximum_frames,
            "exact_frame_indices_requested": list(exact_indices),
            "process_res": args.process_res,
            "processed_height": int(processed.shape[1]),
            "processed_width": int(processed.shape[2]),
            "use_ray_pose": args.use_ray_pose,
            "reference_view_strategy": "middle",
        },
        "coordinate_contract": {
            "depth": "camera:da3_processed_metric",
            "depth_unit": "m",
            "intrinsics": "camera:da3_processed_pixels",
            "extrinsics": "camera_from_world OpenCV convention",
            "poses": "world_from_camera",
            "world": "world:da3_learned_metric",
            "timeline": "frame:source_video",
            "metric_scale_source": "DA3 Nested metric submodel",
            "learned_scale_uncertainty_fraction": None,
        },
        "statistics": {
            "valid_positive_depth_fraction": float(np.mean(valid)),
            "depth_m_p01": float(np.percentile(depth[valid], 1)),
            "depth_m_median": float(np.median(depth[valid])),
            "depth_m_p99": float(np.percentile(depth[valid], 99)),
            "confidence_median": float(np.median(confidence)),
            "rotation_orthogonality_error_max": float(np.max(orthogonality)),
            "intrinsics_px_median": np.median(intrinsics, axis=0).tolist(),
        },
        "performance": {
            "frame_extraction_seconds": extraction_seconds,
            "model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "sampled_frames_per_inference_second": len(frames)
            / max(inference_seconds, 1e-12),
            "source_video_seconds_per_inference_second": (
                (frame_indices[-1] - frame_indices[0] + 1) / video["fps"]
            )
            / max(inference_seconds, 1e-12),
        },
        "outputs": {"samples": {"path": str(artifact), "sha256": _sha256(artifact)}},
        "limitations": [
            "DA3 Nested metric scale is learned and must be uncertainty-bounded before force closure is accepted.",
            "Sparse temporal sampling is a geometry preflight, not a frame-complete 20-second reconstruction.",
            "Static-scene camera/depth geometry does not by itself model deformable stem correspondence.",
            "DA3 does not estimate contact forces.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
