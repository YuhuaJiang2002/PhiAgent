#!/usr/bin/env python3
"""Run pinned V-DPM in overlapping windows and persist dynamic point maps."""

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


VDPM_REVISION = "5e2a57cf6007dfb0511a8b396a0805089b9edcc4"
VDPM_CHECKPOINT_URL = "https://huggingface.co/edgarsucar/vdpm/resolve/main/model.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *command], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


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
    parser.add_argument("--vdpm-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=30000)
    parser.add_argument("--sample-hz", type=float, default=1.0)
    parser.add_argument("--window-frames", type=int, default=6)
    parser.add_argument("--overlap-frames", type=int, default=2)
    parser.add_argument("--maximum-windows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def _extract_frames(cv2: Any, np: Any, source: Path, sample_hz: float) -> tuple[list[Any], list[int], dict[str, Any]]:
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not capture.isOpened() or fps <= 0 or total <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("source video metadata is invalid")
    interval = max(1, int(round(fps / sample_hz)))
    frames: list[Any] = []
    indices: list[int] = []
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if index % interval == 0:
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


def _preprocess(cv2: Any, np: Any, torch: Any, frames: list[Any]) -> Any:
    target_width = 518
    tensors = []
    shapes = set()
    for frame in frames:
        height, width = frame.shape[:2]
        target_height = round(height * (target_width / width) / 14) * 14
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)
        shapes.add(tuple(tensor.shape[1:]))
    if len(shapes) != 1:
        raise RuntimeError(f"V-DPM preprocessed frames disagree in shape: {shapes}")
    return torch.stack(tensors)


def _window_starts(total: int, size: int, overlap: int) -> list[int]:
    if total < 2:
        raise ValueError("V-DPM requires at least two sampled frames")
    step = size - overlap
    starts = list(range(0, max(1, total - 1), step))
    starts = [start for start in starts if total - start >= 2]
    return starts


def main() -> int:
    args = _parser().parse_args()
    if args.sample_hz <= 0:
        raise ValueError("sample Hz must be positive")
    if args.window_frames < 3 or not 1 <= args.overlap_frames < args.window_frames:
        raise ValueError("window must have at least three frames and a smaller positive overlap")
    if args.maximum_windows < 0:
        raise ValueError("maximum windows must be non-negative")
    source = args.source.expanduser().resolve()
    repository = args.vdpm_repo.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if (
        not source.is_file()
        or not (repository / "dpm").is_dir()
        or not checkpoint.is_file()
    ):
        raise ValueError(
            "source video, V-DPM repository, or local checkpoint is missing"
        )
    revision = _git(["rev-parse", "HEAD"], repository)
    if revision != VDPM_REVISION:
        raise RuntimeError(f"V-DPM revision {revision} != pinned {VDPM_REVISION}")
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite experiment directory: {output_dir}")
    output_dir.mkdir(parents=True)

    inventory = _gpu_inventory()
    selected = _select_gpu(inventory, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    sys.path.insert(0, str(repository))

    import cv2
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    from dpm.model import VDPM

    if not torch.cuda.is_available():
        raise RuntimeError("V-DPM GPU entrypoint requires CUDA after device selection")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")

    extract_started = time.perf_counter()
    rgb_frames, source_indices, video = _extract_frames(
        cv2, np, source, args.sample_hz
    )
    images = _preprocess(cv2, np, torch, rgb_frames)
    extraction_seconds = time.perf_counter() - extract_started
    starts = _window_starts(len(rgb_frames), args.window_frames, args.overlap_frames)
    if args.maximum_windows:
        starts = starts[: args.maximum_windows]

    load_started = time.perf_counter()
    with initialize_config_dir(version_base=None, config_dir=str(repository / "configs")):
        cfg = compose(config_name="visualise")
    model = VDPM(cfg).to(device)
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    windows = []
    total_inference_seconds = 0.0
    for window_index, start in enumerate(starts):
        end = min(start + args.window_frames, len(rgb_frames))
        window_images = images[start:end].to(device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model.inference(None, images=window_images.unsqueeze(0))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        total_inference_seconds += elapsed
        points = np.concatenate(
            [row["pts3d"].detach().float().cpu().numpy() for row in prediction["pointmaps"]],
            axis=0,
        )
        confidence = np.concatenate(
            [row["conf"].detach().float().cpu().numpy() for row in prediction["pointmaps"]],
            axis=0,
        )
        pose_encoding = prediction["pose_enc"]
        image_hw = prediction["pointmaps"][0]["pts3d"].shape[2:4]
        camera_from_world, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding, image_hw
        )
        camera_from_world = camera_from_world[0].detach().float().cpu().numpy()
        intrinsics = intrinsics[0].detach().float().cpu().numpy()
        homogeneous = np.concatenate(
            (
                camera_from_world,
                np.repeat(np.asarray([[[0.0, 0.0, 0.0, 1.0]]]), end - start, axis=0),
            ),
            axis=1,
        )
        world_from_camera = np.linalg.inv(homogeneous)
        path = output_dir / f"window-{window_index:03d}.npz"
        np.savez_compressed(
            path,
            source_frame_indices=np.asarray(source_indices[start:end], dtype=np.int32),
            images_rgb=np.rint(
                window_images.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            ).astype(np.uint8),
            dynamic_pointmaps=points.astype(np.float16),
            confidence=confidence.astype(np.float16),
            world_from_camera=world_from_camera.astype(np.float32),
            intrinsics_px=intrinsics.astype(np.float32),
        )
        windows.append(
            {
                "window": window_index,
                "sample_start": start,
                "sample_end_exclusive": end,
                "source_frame_indices": source_indices[start:end],
                "frames": end - start,
                "inference_seconds": elapsed,
                "path": str(path),
                "sha256": _sha256(path),
                "pointmap_shape": list(points.shape),
            }
        )
        del prediction, points, confidence, window_images
        torch.cuda.empty_cache()

    sampled_count = sum(row["frames"] for row in windows)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "status_reason": (
            "overlapping relative dynamic point maps are estimated; global window alignment, "
            "metric scale fusion, and per-stem centerline extraction remain downstream gates"
        ),
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": _versions(("numpy", "opencv-python", "torch", "torchvision", "hydra-core")),
        "seed": args.seed,
        "git": {
            "project_head": _git(["rev-parse", "HEAD"], PROJECT_ROOT),
            "project_status": _git(["status", "--short"], PROJECT_ROOT),
        },
        "gpu": {
            "physical_index": args.gpu,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "inventory_before": inventory,
            "selected": selected,
        },
        "model": {
            "name": "V-DPM",
            "repository": "https://github.com/eldar/vdpm",
            "revision": revision,
            "checkpoint": VDPM_CHECKPOINT_URL,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "license": "MIT code; VGGT model license for inherited weights",
            "evidence_class": "foundation_model_estimate",
        },
        "input": {"path": str(source), "sha256": _sha256(source), **video},
        "sampling": {
            "requested_hz": args.sample_hz,
            "sampled_source_frame_indices": source_indices,
            "window_frames": args.window_frames,
            "overlap_frames": args.overlap_frames,
            "windows_executed": len(windows),
            "maximum_windows": args.maximum_windows,
        },
        "coordinate_contract": {
            "pointmaps": "world:vdpm_window_relative",
            "camera_poses": "world_from_camera OpenCV convention, per window",
            "intrinsics": "camera:vdpm_resized_pixels",
            "scale": "relative; MUST be aligned to bounded metric evidence",
            "timeline": "frame:source_video",
        },
        "performance": {
            "frame_extraction_and_preprocess_seconds": extraction_seconds,
            "model_load_seconds": load_seconds,
            "inference_seconds": total_inference_seconds,
            "window_frame_evaluations": sampled_count,
            "window_frame_evaluations_per_second": sampled_count
            / max(total_inference_seconds, 1e-12),
            "unique_sampled_frames": len(set(index for row in windows for index in row["source_frame_indices"])),
        },
        "windows": windows,
        "limitations": [
            "V-DPM coordinates have learned relative scale and are not metric until fused and uncertainty-bounded.",
            "Window overlap does not itself prove successful global alignment.",
            "Point maps are geometry proposals, not force measurements or force estimates.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
