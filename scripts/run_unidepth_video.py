#!/usr/bin/env python3
"""Run pinned UniDepthV2 on sampled video frames with full GPU provenance."""

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


UNIDEPTH_REVISION = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
DEFAULT_MODEL = "lpiccinelli/unidepth-v2-vits14"


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


def _source_revision(repository: Path) -> tuple[str, str]:
    if (repository / ".git").exists():
        return _git(["rev-parse", "HEAD"], repository), "git_checkout"
    marker = repository / ".phiagent-source-revision"
    if marker.is_file():
        return marker.read_text().strip(), "verified_source_archive_marker"
    raise RuntimeError("UniDepth source has neither Git metadata nor a revision marker")


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--unidepth-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=12000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--frame-stride", type=int, default=12)
    parser.add_argument("--maximum-frames", type=int, default=0)
    parser.add_argument("--resolution-level", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.frame_stride < 1 or args.maximum_frames < 0:
        raise ValueError("frame stride must be positive and maximum frames non-negative")
    source = args.source.expanduser().resolve()
    repository = args.unidepth_repo.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"source video does not exist: {source}")
    if not (repository / "unidepth").is_dir():
        raise ValueError(f"UniDepth repository is incomplete: {repository}")
    revision, revision_evidence = _source_revision(repository)
    if revision != UNIDEPTH_REVISION:
        raise RuntimeError(f"UniDepth revision {revision} != pinned {UNIDEPTH_REVISION}")
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
    from unidepth.models import UniDepthV2

    if not torch.cuda.is_available():
        raise RuntimeError("UniDepth GPU entrypoint requires CUDA after device selection")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not capture.isOpened() or fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("source video metadata is invalid")

    load_started = time.perf_counter()
    model = UniDepthV2.from_pretrained(args.model)
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"
    model = model.to(device).eval()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    frame_indices: list[int] = []
    depths: list[Any] = []
    confidences: list[Any] = []
    intrinsics: list[Any] = []
    inference_seconds = 0.0
    decode_started = time.perf_counter()
    frame_index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            should_sample = frame_index % args.frame_stride == 0
            if should_sample and (
                args.maximum_frames == 0 or len(frame_indices) < args.maximum_frames
            ):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).to(device)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    prediction = model.infer(tensor)
                torch.cuda.synchronize()
                inference_seconds += time.perf_counter() - started
                depth = prediction["depth"].squeeze().detach().float().cpu().numpy()
                confidence = (
                    prediction["confidence"].squeeze().detach().float().cpu().numpy()
                )
                intrinsic = (
                    prediction["intrinsics"].squeeze().detach().float().cpu().numpy()
                )
                if depth.shape != (height, width) or confidence.shape != (height, width):
                    raise RuntimeError(
                        f"UniDepth returned {depth.shape}/{confidence.shape}, expected {(height, width)}"
                    )
                if intrinsic.shape != (3, 3):
                    raise RuntimeError(f"UniDepth intrinsics have invalid shape {intrinsic.shape}")
                frame_indices.append(frame_index)
                depths.append(depth.astype(np.float16))
                confidences.append(confidence.astype(np.float16))
                intrinsics.append(intrinsic.astype(np.float32))
            frame_index += 1
            if args.maximum_frames and len(frame_indices) >= args.maximum_frames:
                break
    finally:
        capture.release()
    end_to_end_seconds = time.perf_counter() - decode_started
    if not frame_indices:
        raise RuntimeError("no frames were sampled for UniDepth")

    artifact = output_dir / "unidepth-samples.npz"
    np.savez_compressed(
        artifact,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        depth_m=np.stack(depths),
        confidence=np.stack(confidences),
        intrinsics_px=np.stack(intrinsics),
    )
    intrinsics_array = np.stack(intrinsics).astype(np.float64)
    depth_array = np.stack(depths).astype(np.float32)
    confidence_array = np.stack(confidences).astype(np.float32)
    finite_positive = np.isfinite(depth_array) & (depth_array > 0)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "status_reason": (
            "metric depth and intrinsics estimated; camera extrinsics and independently "
            "calibrated scale uncertainty are not provided by this stage"
        ),
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": _package_versions(
            ("numpy", "opencv-python", "torch", "torchvision", "unidepth")
        ),
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
            "name": args.model,
            "repository": "https://github.com/lpiccinelli-eth/UniDepth",
            "revision": revision,
            "revision_evidence": revision_evidence,
            "license": "CC-BY-NC-4.0",
            "evidence_class": "foundation_model_estimate",
        },
        "input": {
            "path": str(source),
            "sha256": _sha256(source),
            "fps": fps,
            "frames": total_frames,
            "width": width,
            "height": height,
        },
        "sampling": {
            "frame_stride": args.frame_stride,
            "sampled_frames": len(frame_indices),
            "frame_indices": frame_indices,
            "maximum_frames": args.maximum_frames,
            "resolution_level": args.resolution_level,
        },
        "coordinate_contract": {
            "depth": "camera:source_metric",
            "intrinsics": "camera:source_pixels",
            "timeline": "frame:source_video",
            "depth_unit": "m",
            "camera_extrinsics": "NOT_ESTIMATED",
            "metric_scale_source": "UniDepthV2 learned metric prior",
            "learned_scale_uncertainty_fraction": None,
        },
        "statistics": {
            "valid_positive_depth_fraction": float(np.mean(finite_positive)),
            "depth_m_p01": float(np.percentile(depth_array[finite_positive], 1)),
            "depth_m_median": float(np.median(depth_array[finite_positive])),
            "depth_m_p99": float(np.percentile(depth_array[finite_positive], 99)),
            "confidence_median": float(np.median(confidence_array)),
            "intrinsics_px_median": np.median(intrinsics_array, axis=0).tolist(),
        },
        "performance": {
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
            "end_to_end_sampling_seconds": end_to_end_seconds,
            "sampled_frames_per_inference_second": len(frame_indices)
            / max(inference_seconds, 1e-12),
            "sampled_frames_per_end_to_end_second": len(frame_indices)
            / max(end_to_end_seconds, 1e-12),
        },
        "outputs": {
            "samples": {"path": str(artifact), "sha256": _sha256(artifact)}
        },
        "limitations": [
            "Monocular metric scale is a learned estimate, not an RGB-D or fiducial calibration.",
            "This sampled stage cannot alone supply per-frame deformable dynamics.",
            "No force is inferred by UniDepth and downstream code must not treat depth confidence as force confidence.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
