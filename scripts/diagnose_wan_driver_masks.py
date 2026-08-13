#!/usr/bin/env python3
"""Locate empty Wan replacement masks in an action driver without patching Wan."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.wan_animate import (  # noqa: E402
    acquire_gpu_lease,
    query_gpus,
    select_gpu,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=60 * 1024)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    wan_repo = args.wan_repo.expanduser().resolve()
    checkpoints = args.checkpoint_dir.expanduser().resolve()
    experiment = args.experiment_dir.expanduser().resolve()
    report_path = experiment / "mask-diagnosis.json"
    if report_path.exists():
        raise FileExistsError(f"diagnosis already exists: {report_path}")
    required = (
        video,
        wan_repo / "wan/modules/animate/preprocess/process_pipepline.py",
        checkpoints / "det/yolov10m.onnx",
        checkpoints / "pose2d/vitpose_h_wholebody.onnx",
        checkpoints / "sam2/sam2_hiera_large.pt",
    )
    for path in required:
        missing = not path.exists()
        empty_file = path.is_file() and path.stat().st_size == 0
        empty_directory = path.is_dir() and not any(path.iterdir())
        if missing or empty_file or empty_directory:
            raise ValueError(f"required diagnosis input is missing: {path}")
    experiment.mkdir(parents=True, exist_ok=True)
    gpus, inventory_raw, processes_raw = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    preprocess_dir = wan_repo / "wan/modules/animate/preprocess"
    sys.path.insert(0, str(preprocess_dir))
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "seed": args.seed,
        "selected_gpu": asdict(selected),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_inventory_raw": inventory_raw,
        "gpu_processes_raw": processes_raw,
        "video": str(video),
        "wan_repo": str(wan_repo),
        "checkpoint_dir": str(checkpoints),
        "packages": {},
    }
    for name in ("torch", "opencv-python", "numpy", "decord", "sam-2"):
        try:
            payload["packages"][name] = importlib.metadata.version(name)  # type: ignore[index]
        except importlib.metadata.PackageNotFoundError:
            payload["packages"][name] = None  # type: ignore[index]
    _write_json(report_path, payload)

    lease_path, lease = acquire_gpu_lease(selected.physical_index)
    try:
        from decord import VideoReader
        from process_pipepline import ProcessPipeline
        from utils import get_frame_indices, resize_by_area

        reader = VideoReader(str(video))
        video_fps = float(reader.get_avg_fps())
        target_num = int(len(reader) / video_fps * args.fps)
        indices = get_frame_indices(len(reader), video_fps, target_num, args.fps)
        frames = reader.get_batch(indices).asnumpy()
        frames = [
            resize_by_area(frame, args.width * args.height, divisor=16)
            for frame in frames
        ]
        pipeline = ProcessPipeline(
            det_checkpoint_path=str(checkpoints / "det/yolov10m.onnx"),
            pose2d_checkpoint_path=str(checkpoints / "pose2d/vitpose_h_wholebody.onnx"),
            sam_checkpoint_path=str(checkpoints / "sam2/sam2_hiera_large.pt"),
            flux_kontext_path=None,
        )
        pose_metas = pipeline.pose2d(frames)
        masks = pipeline.get_mask(frames, 400, pose_metas)
        nonzero_pixels = [int(mask.sum()) for mask in masks]
        empty = [index for index, value in enumerate(nonzero_pixels) if value == 0]
        payload.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "gpu_lease": str(lease_path),
                "decoded_frames": len(frames),
                "mask_frames": len(masks),
                "empty_mask_frames": empty,
                "nonzero_pixels": nonzero_pixels,
                "acceptance": {"no_empty_masks": not empty},
            }
        )
        _write_json(report_path, payload)
        print(json.dumps({"report": str(report_path), "empty_mask_frames": empty}, indent=2))
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
            }
        )
        _write_json(report_path, payload)
        raise
    finally:
        import fcntl

        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        lease.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
