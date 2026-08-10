#!/usr/bin/env python3
"""Extract one topology-stable canonical robot hand with pinned SAM2."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.minimax_h3 import file_sha256  # noqa: E402
from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402

SAM2_COMMIT = "0e78a118995e66bb27d78518c4bd9a3e95b4e266"


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _point(value: str) -> tuple[float, float]:
    fields = value.split(",")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("point must be X,Y")
    return float(fields[0]), float(fields[1])


def _box(value: str) -> tuple[float, float, float, float]:
    fields = value.split(",")
    if len(fields) != 4:
        raise argparse.ArgumentTypeError("box must be LEFT,TOP,RIGHT,BOTTOM")
    box = tuple(float(field) for field in fields)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise argparse.ArgumentTypeError("box must have positive width and height")
    return box


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", default="sam2_hiera_l.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive-point", type=_point, action="append", required=True)
    parser.add_argument("--negative-point", type=_point, action="append", default=[])
    parser.add_argument("--box", type=_box, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=12 * 1024)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


def main() -> int:
    args = _parser().parse_args()
    image_path = args.image.expanduser().resolve()
    repository = args.sam2_repo.expanduser().resolve()
    checkpoint = args.sam2_checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path, label in (
        (image_path, "canonical image"),
        (repository, "SAM2 repository"),
        (checkpoint, "SAM2 checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")
    if output.exists():
        raise FileExistsError(f"experiment already exists: {output}")
    output.mkdir(parents=True)
    (output / "provenance").mkdir()
    frozen_source = output / "provenance" / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), frozen_source)

    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_revision != SAM2_COMMIT:
        raise RuntimeError(f"SAM2 source is {source_revision}, expected {SAM2_COMMIT}")
    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    sys.path.insert(0, str(repository))

    import cv2
    import numpy as np
    import torch
    import sam2.modeling.sam.transformer as sam_transformer
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.manual_seed(args.seed)
    # Some shared GPU environments disable every SDPA backend while importing
    # other video pipelines. SAM2's image predictor needs at least the stable
    # math implementation for its small float32 prompt-attention tensors.
    torch.backends.cuda.enable_math_sdp(True)
    sam_transformer.MATH_KERNEL_ON = True
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {image_path}")
    height, width = image.shape[:2]
    points = np.asarray(args.positive_point + args.negative_point, dtype=np.float32)
    labels = np.asarray(
        [1] * len(args.positive_point) + [0] * len(args.negative_point),
        dtype=np.int32,
    )
    model = build_sam2(args.sam2_config, str(checkpoint), device="cuda")
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=np.asarray(args.box, dtype=np.float32),
        multimask_output=True,
    )

    candidates = []
    positive_count = len(args.positive_point)
    for index, (mask, score) in enumerate(zip(masks, scores)):
        binary = mask.astype(np.uint8)
        positive_recall = float(
            np.mean([binary[round(y), round(x)] for x, y in args.positive_point])
        )
        negative_rejection = float(
            np.mean([not binary[round(y), round(x)] for x, y in args.negative_point])
            if args.negative_point
            else 1.0
        )
        area_fraction = float(np.mean(binary))
        plausible = (
            0.005 <= area_fraction <= 0.10 and positive_recall >= 0.66 and negative_rejection == 1.0
        )
        rank = (
            int(plausible),
            negative_rejection,
            positive_recall,
            float(score),
            -abs(area_fraction - 0.035),
        )
        candidates.append(
            {
                "index": index,
                "mask": binary,
                "sam_score": float(score),
                "positive_recall": positive_recall,
                "negative_rejection": negative_rejection,
                "area_fraction": area_fraction,
                "plausible": plausible,
                "rank": rank,
            }
        )
    selected_candidate = max(candidates, key=lambda item: item["rank"])
    mask = selected_candidate["mask"] * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(mask)
    anchor_x, anchor_y = map(round, args.positive_point[min(positive_count - 1, 3)])
    anchor_component = int(component_labels[anchor_y, anchor_x])
    if anchor_component == 0:
        anchor_component = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
    canonical_mask = np.where(component_labels == anchor_component, 255, 0).astype(np.uint8)
    final_positive_recall = float(
        np.mean([canonical_mask[round(y), round(x)] > 0 for x, y in args.positive_point])
    )
    if final_positive_recall < 0.66:
        raise RuntimeError("selected SAM2 component misses the reviewed hand core")
    final_components = cv2.connectedComponents(canonical_mask)[0] - 1
    if final_components != 1:
        raise RuntimeError("canonical robot hand must be one connected component")

    alpha = cv2.GaussianBlur(canonical_mask, (5, 5), 0)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    cv2.imwrite(str(output / "canonical-hand-mask.png"), canonical_mask)
    cv2.imwrite(str(output / "canonical-hand-rgba.png"), rgba)
    preview = image.copy()
    preview[canonical_mask == 0] = (preview[canonical_mask == 0] * 0.16).astype(np.uint8)
    cv2.imwrite(str(output / "canonical-hand-preview.jpg"), preview)
    packages = {}
    for package in ("torch", "torchvision", "sam-2"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted_canonical_hand_mask",
        "honest_status": "WORKING",
        "method": "sam2_prompted_single_component_robot_hand",
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "source_revision": source_revision,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "input": {
            "path": str(image_path),
            "sha256": file_sha256(image_path),
            "width": width,
            "height": height,
        },
        "prompts": {
            "positive_points_xy": args.positive_point,
            "negative_points_xy": args.negative_point,
            "box_xyxy": args.box,
        },
        "gpu": {
            "selected": asdict(selected),
            "inventory": [asdict(gpu) for gpu in gpus],
            "inventory_raw": inventory,
            "processes_raw": processes,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "selection": {
            "selected_candidate": selected_candidate["index"],
            "connected_components": final_components,
            "area_pixels": int(np.count_nonzero(canonical_mask)),
            "area_fraction": float(np.mean(canonical_mask > 0)),
            "final_positive_recall": final_positive_recall,
            "candidates": [
                {key: value for key, value in candidate.items() if key not in {"mask", "rank"}}
                for candidate in candidates
            ],
        },
        "outputs": {
            name: {"path": name, "sha256": file_sha256(output / name)}
            for name in (
                "canonical-hand-mask.png",
                "canonical-hand-rgba.png",
                "canonical-hand-preview.jpg",
            )
        },
        "limitations": [
            "SAM2 extracts one reviewed 2-D canonical hand; it does not infer robot joints or 3-D pose.",
            "The mask is suitable for topology-locked image compositing, not robot control or physics validation.",
        ],
        "execution_source": {
            "path": str(frozen_source),
            "sha256": file_sha256(frozen_source),
        },
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
