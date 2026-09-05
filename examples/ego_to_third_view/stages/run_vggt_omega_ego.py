#!/usr/bin/env python3
"""Run VGGT-Omega on an ego frame sequence and export compact geometry state."""

from __future__ import annotations

from runtime import require_launcher
require_launcher()

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    return p.parse_args()


def strip_batch(x: torch.Tensor) -> np.ndarray:
    a = x.detach().float().cpu().numpy()
    return a[0] if a.shape[0] == 1 else a


def camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]
    return np.asarray([-R[i].T @ t[i] for i in range(len(R))])


def main() -> None:
    cfg = parse_args()
    names = sorted(cfg.frames_dir.glob("*.jpg"))
    if not names or not cfg.checkpoint.is_file():
        raise SystemExit("frames/checkpoint missing")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "start", "frames": len(names), "checkpoint": str(cfg.checkpoint),
                      "cuda_visible_device": 0, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = VGGTOmega().eval()
    state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    del state
    model = model.to("cuda")
    t_model = time.perf_counter()
    images = load_and_preprocess_images([str(p) for p in names], image_resolution=cfg.resolution).to("cuda")
    print(json.dumps({"stage": "preprocessed", "shape": list(images.shape), "seconds": time.perf_counter() - t_model}), flush=True)
    torch.cuda.synchronize()
    t_forward = time.perf_counter()
    with torch.inference_mode():
        pred = model(images)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - t_forward
    extrinsic_t, intrinsic_t = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
    extrinsic = strip_batch(extrinsic_t)
    intrinsic = strip_batch(intrinsic_t)
    depth = strip_batch(pred["depth"]).astype(np.float16)
    depth_conf = strip_batch(pred["depth_conf"]).astype(np.float16)
    processed_images = strip_batch(pred["images"]).astype(np.float16)
    centers = camera_centers(extrinsic)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    out_npz = cfg.output_dir / "vggt_omega_predictions_compact.npz"
    np.savez_compressed(
        out_npz,
        extrinsic_camera_from_omega_world=extrinsic.astype(np.float32),
        intrinsic_processed_pixels=intrinsic.astype(np.float32),
        depth=depth,
        depth_confidence=depth_conf,
        processed_images=processed_images,
        source_frame_names=np.asarray([p.name for p in names]),
        source_image_wh=np.asarray([cv2.imread(str(names[0])).shape[1], cv2.imread(str(names[0])).shape[0]], np.int32),
        processed_image_hw=np.asarray(depth.shape[1:3], np.int32),
    )
    report = {
        "schema_version": "phiagent-vggt-omega-ego/0.1",
        "frames": len(names),
        "checkpoint": str(cfg.checkpoint.resolve()),
        "resolution": cfg.resolution,
        "preprocessed_shape": list(images.shape),
        "prediction_shapes": {"extrinsic": list(extrinsic.shape), "intrinsic": list(intrinsic.shape), "depth": list(depth.shape), "depth_confidence": list(depth_conf.shape)},
        "runtime_seconds": {"model_load": t_model - t0, "forward": forward_seconds, "total": time.perf_counter() - t0},
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "raw_camera_diagnostics": {
            "center_min": centers.min(axis=0).tolist(), "center_max": centers.max(axis=0).tolist(),
            "step_median_model_units": float(np.median(steps)), "step_p99_model_units": float(np.percentile(steps, 99)), "step_max_model_units": float(np.max(steps)),
        },
        "authority": "VGGT-Omega prediction in its arbitrary world gauge; not metric until table similarity alignment",
        "output": str(out_npz.resolve()),
    }
    (cfg.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": "complete", **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
