#!/usr/bin/env python3
"""DINOv2 verifier for native robot-hand appearance and temporal stability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.rendering.wan_animate import query_gpus, select_gpu


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"video has no frames: {path}")
    return frames


def _masked_crop(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    padding_fraction: float = 0.12,
) -> np.ndarray | None:
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    points = cv2.findNonZero((mask >= 128).astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    padding = round(padding_fraction * max(width, height))
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1 = min(image.shape[1], x + width + padding)
    y1 = min(image.shape[0], y + height + padding)
    crop = image[y0:y1, x0:x1].copy()
    selected = mask[y0:y1, x0:x1] >= 128
    crop[~selected] = 0
    return crop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hand-mask-video", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--reference-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=8 * 1024)
    parser.add_argument("--sample-stride", type=int, default=4)
    args = parser.parse_args()
    for path in (
        args.video,
        args.hand_mask_video,
        args.reference_image,
        args.reference_mask,
        args.dino_repo,
        args.dino_checkpoint,
    ):
        if not path.exists():
            raise ValueError(f"required input does not exist: {path}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.minimum_free_gpu_mib <= 0 or args.sample_stride <= 0:
        raise ValueError("GPU memory and sample stride must be positive")

    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)

    import torch
    import torch.nn.functional as functional

    model = torch.hub.load(
        str(args.dino_repo.resolve()),
        "dinov2_vitb14_reg",
        source="local",
        pretrained=False,
    )
    state = torch.load(
        args.dino_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model = model.eval().cuda()
    mean = torch.tensor((0.485, 0.456, 0.406), device="cuda")[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device="cuda")[:, None, None]

    def feature(crop: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().cuda() / 255.0
        tensor = functional.interpolate(
            tensor[None],
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        with torch.inference_mode():
            return functional.normalize(model((tensor - mean) / std), dim=-1)[0]

    reference = cv2.imread(str(args.reference_image), cv2.IMREAD_COLOR)
    reference_mask = cv2.imread(str(args.reference_mask), cv2.IMREAD_GRAYSCALE)
    if reference is None or reference_mask is None:
        raise ValueError("reference image or mask is invalid")
    reference_crop = _masked_crop(reference, reference_mask)
    if reference_crop is None:
        raise ValueError("reference hand mask is empty")
    reference_feature = feature(reference_crop)

    frames = _read_video(args.video)
    masks = _read_video(args.hand_mask_video)
    features: list[torch.Tensor] = []
    sampled = 0
    for index in range(0, min(len(frames), len(masks)), args.sample_stride):
        sampled += 1
        mask = cv2.cvtColor(masks[index], cv2.COLOR_BGR2GRAY)
        crop = _masked_crop(frames[index], mask)
        if crop is not None:
            features.append(feature(crop))
    if not features:
        raise ValueError("no candidate hand crops were available")
    similarities = [
        float(torch.dot(reference_feature, candidate).item())
        for candidate in features
    ]
    temporal = [
        float(torch.dot(left, right).item())
        for left, right in zip(features, features[1:])
    ]
    metrics = {
        "sampled_frames": sampled,
        "valid_hand_crops": len(features),
        "tracking_coverage": len(features) / sampled,
        "mean_native_hand_similarity": sum(similarities) / len(similarities),
        "minimum_native_hand_similarity": min(similarities),
        "mean_temporal_hand_similarity": (
            sum(temporal) / len(temporal) if temporal else 1.0
        ),
        "minimum_temporal_hand_similarity": min(temporal) if temporal else 1.0,
    }
    gates = {
        "hand_track_complete": metrics["tracking_coverage"] >= 0.80,
        "native_hand_mean": metrics["mean_native_hand_similarity"] >= 0.55,
        "native_hand_worst_frame": metrics["minimum_native_hand_similarity"] >= 0.35,
        "hand_temporal_mean": metrics["mean_temporal_hand_similarity"] >= 0.80,
        "hand_temporal_worst_frame": metrics["minimum_temporal_hand_similarity"] >= 0.55,
    }
    payload = {
        "schema_version": "1.0.0",
        "method": "DINOv2 native Sunday hand similarity",
        "video": str(args.video.resolve()),
        "video_sha256": _sha256(args.video),
        "hand_mask_video": str(args.hand_mask_video.resolve()),
        "reference_image": str(args.reference_image.resolve()),
        "reference_mask": str(args.reference_mask.resolve()),
        "dino_checkpoint": str(args.dino_checkpoint.resolve()),
        "dino_checkpoint_sha256": _sha256(args.dino_checkpoint),
        "selected_gpu": asdict(selected),
        "gpu_inventory_raw": inventory,
        "gpu_processes_raw": processes,
        "metrics": metrics,
        "gates": gates,
        "accepted": all(gates.values()),
        "limitations": [
            "DINO similarity verifies appearance, not kinematic feasibility.",
            "SAM2 hand masks define the evaluated hand region.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
