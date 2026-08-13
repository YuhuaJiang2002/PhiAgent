#!/usr/bin/env python3
"""Lift persistent stem masks into metric 3-D centerlines using DA3 geometry."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.perception.foundation_contact import (  # noqa: E402
    EvidenceClass,
    StemCenterlineContract,
    validate_stem_centerlines,
)
from phiagent.rendering.object_factored_long_video import (  # noqa: E402
    SourceResizeCrop,
    remap_boolean_mask,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    result: dict[str, object] = {}
    for label, command in (
        ("head", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--short"]),
    ):
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
        )
        result[label] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--da3-samples", type=Path, required=True)
    parser.add_argument("--stem-masks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nodes-per-stem", type=int, default=12)
    parser.add_argument("--source-width", type=int, default=1280)
    parser.add_argument("--source-height", type=int, default=720)
    parser.add_argument("--mask-scaled-width", type=int, default=854)
    parser.add_argument("--mask-scaled-height", type=int, default=480)
    parser.add_argument("--mask-crop-left", type=int, default=11)
    parser.add_argument("--mask-crop-top", type=int, default=0)
    parser.add_argument("--target-width", type=int, default=624)
    parser.add_argument("--target-height", type=int, default=352)
    parser.add_argument("--target-scaled-width", type=int, default=625)
    parser.add_argument("--target-scaled-height", type=int, default=352)
    parser.add_argument("--target-crop-left", type=int, default=0)
    parser.add_argument("--target-crop-top", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def _largest_component(np: Any, pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
    remaining = set(pixels)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    neighbor = (y + dy, x + dx)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
        components.append(component)
    return max(components, key=len) if components else set()


def _dijkstra_farthest(
    pixels: set[tuple[int, int]], start: tuple[int, int]
) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], float]]:
    distances = {start: 0.0}
    predecessors: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, start)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances.get(current):
            continue
        y, x = current
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = (y + dy, x + dx)
                if neighbor not in pixels:
                    continue
                candidate = distance + (2.0**0.5 if dx and dy else 1.0)
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    predecessors[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
    farthest = max(distances, key=distances.get)
    return farthest, predecessors, distances


def _morphological_skeleton(cv2: Any, np: Any, mask: Any) -> Any:
    image = np.asarray(mask, dtype=np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    iterations = 0
    while cv2.countNonZero(image) and iterations < max(image.shape):
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = cv2.erode(image, element)
        iterations += 1
    return skeleton > 0


def extract_centerline_pixels(cv2: Any, np: Any, mask: Any, nodes: int) -> Any:
    """Return root-to-tip (x, y) samples along the skeleton geodesic diameter."""

    if nodes < 3:
        raise ValueError("centerline requires at least three nodes")
    skeleton = _morphological_skeleton(cv2, np, mask)
    ys, xs = np.nonzero(skeleton)
    component = _largest_component(np, set(zip(ys.tolist(), xs.tolist())))
    if len(component) < nodes:
        raise ValueError("stem skeleton has fewer pixels than requested nodes")
    endpoint_a, _, _ = _dijkstra_farthest(component, next(iter(component)))
    endpoint_b, predecessors, distances = _dijkstra_farthest(component, endpoint_a)
    path = [endpoint_b]
    while path[-1] != endpoint_a:
        if path[-1] not in predecessors:
            raise RuntimeError("stem skeleton diameter predecessor chain is broken")
        path.append(predecessors[path[-1]])
    path_array = np.asarray([(x, y) for y, x in path[::-1]], dtype=np.float64)
    if path_array[0, 1] < path_array[-1, 1]:
        path_array = path_array[::-1]
    deltas = np.linalg.norm(np.diff(path_array, axis=0), axis=1)
    arc = np.concatenate((np.zeros(1), np.cumsum(deltas)))
    sample_arc = np.linspace(0.0, float(arc[-1]), nodes)
    sampled = np.stack(
        (
            np.interp(sample_arc, arc, path_array[:, 0]),
            np.interp(sample_arc, arc, path_array[:, 1]),
        ),
        axis=1,
    )
    return sampled


def _lift_node(
    np: Any,
    *,
    xy: Any,
    depth: Any,
    confidence: Any,
    intrinsic: Any,
    world_from_camera: Any,
    mask: Any,
) -> tuple[Any, Any, float, float]:
    x, y = (float(xy[0]), float(xy[1]))
    column = int(np.clip(round(x), 0, depth.shape[1] - 1))
    row = int(np.clip(round(y), 0, depth.shape[0] - 1))
    y0, y1 = max(0, row - 2), min(depth.shape[0], row + 3)
    x0, x1 = max(0, column - 2), min(depth.shape[1], column + 3)
    local_valid = mask[y0:y1, x0:x1] & np.isfinite(depth[y0:y1, x0:x1]) & (depth[y0:y1, x0:x1] > 0)
    local_depth = depth[y0:y1, x0:x1][local_valid]
    local_confidence = confidence[y0:y1, x0:x1][local_valid]
    if len(local_depth) == 0:
        raise ValueError("stem node has no valid metric depth support")
    z = float(np.median(local_depth))
    raw_confidence = float(np.median(local_confidence))
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    camera = np.asarray(((x - cx) * z / fx, (y - cy) * z / fy, z), dtype=np.float64)
    world = (world_from_camera @ np.concatenate((camera, np.ones(1))))[:3]
    median_absolute_deviation = float(np.median(np.abs(local_depth - z)))
    sigma_z = max(
        1.4826 * median_absolute_deviation,
        z * 0.01 / max(1.0, raw_confidence),
    )
    pixel_sigma_m = z * 0.5 * ((1.0 / fx) ** 2 + (1.0 / fy) ** 2) ** 0.5
    position_sigma = float((sigma_z**2 + pixel_sigma_m**2) ** 0.5)
    return camera, world, raw_confidence, position_sigma


def main() -> int:
    args = _parser().parse_args()
    if args.nodes_per_stem < 3:
        raise ValueError("nodes per stem must be at least three")
    da3_path = args.da3_samples.expanduser().resolve()
    masks_path = args.stem_masks.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not da3_path.is_file() or not masks_path.is_file():
        raise ValueError("DA3 samples or stem masks are missing")
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite experiment directory: {output_dir}")
    output_dir.mkdir(parents=True)

    import cv2
    import numpy as np

    started = time.perf_counter()
    da3 = np.load(da3_path, allow_pickle=False)
    masks_payload = np.load(masks_path, allow_pickle=False)
    frame_indices = da3["source_frame_indices"].astype(np.int64)
    mask_indices = masks_payload["source_frame_indices"].astype(np.int64)
    if tuple(frame_indices) != tuple(mask_indices):
        raise ValueError("DA3 frames and persistent stem-mask frames must align exactly")
    masks_packed = masks_payload["masks_packed"]
    instance_ids = tuple(str(value) for value in masks_payload["instance_ids"])
    mask_height = int(masks_payload["height"])
    mask_width = int(masks_payload["width"])
    bitorder = str(masks_payload["bitorder"])
    if masks_packed.shape[:2] != (len(instance_ids), len(frame_indices)):
        raise ValueError("packed stem masks do not align with IDs and frames")

    processed_images = da3["processed_images_rgb"]
    depth = da3["depth_m"].astype(np.float32)
    confidence = da3["confidence"].astype(np.float32)
    intrinsics = da3["intrinsics_px"].astype(np.float64)
    world_from_camera = da3["world_from_camera"].astype(np.float64)
    processed_height, processed_width = depth.shape[1:]
    if processed_images.shape[:3] != depth.shape:
        raise ValueError("DA3 images and depth are not aligned")

    mask_frame = SourceResizeCrop(
        name="camera:source_flower_mask_pixels",
        source_width=args.source_width,
        source_height=args.source_height,
        scaled_width=args.mask_scaled_width,
        scaled_height=args.mask_scaled_height,
        crop_left=args.mask_crop_left,
        crop_top=args.mask_crop_top,
        output_width=mask_width,
        output_height=mask_height,
    )
    target_frame = SourceResizeCrop(
        name="camera:candidate_pixels",
        source_width=args.source_width,
        source_height=args.source_height,
        scaled_width=args.target_scaled_width,
        scaled_height=args.target_scaled_height,
        crop_left=args.target_crop_left,
        crop_top=args.target_crop_top,
        output_width=args.target_width,
        output_height=args.target_height,
    )

    shape = (len(frame_indices), len(instance_ids), args.nodes_per_stem)
    pixels = np.full((*shape, 2), np.nan, dtype=np.float32)
    camera_points = np.full((*shape, 3), np.nan, dtype=np.float32)
    world_points = np.full((*shape, 3), np.nan, dtype=np.float32)
    node_confidence = np.zeros(shape, dtype=np.float32)
    position_sigma = np.full(shape, np.inf, dtype=np.float32)
    overlay_frames = []
    extraction_errors = []
    for frame_slot in range(len(frame_indices)):
        overlay = processed_images[frame_slot].copy()
        for instance_slot, instance_id in enumerate(instance_ids):
            packed = masks_packed[instance_slot, frame_slot]
            count = mask_height * mask_width
            unpacked = np.unpackbits(packed, bitorder=bitorder)[:count].reshape(mask_height, mask_width).astype(bool)
            mapped = remap_boolean_mask(
                np, unpacked, source_frame=mask_frame, target_frame=target_frame
            )
            processed_mask = cv2.resize(
                mapped.astype(np.uint8),
                (processed_width, processed_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            try:
                centerline = extract_centerline_pixels(
                    cv2, np, processed_mask, args.nodes_per_stem
                )
                for node_slot, xy in enumerate(centerline):
                    camera, world, raw_confidence, sigma = _lift_node(
                        np,
                        xy=xy,
                        depth=depth[frame_slot],
                        confidence=confidence[frame_slot],
                        intrinsic=intrinsics[frame_slot],
                        world_from_camera=world_from_camera[frame_slot],
                        mask=processed_mask,
                    )
                    pixels[frame_slot, instance_slot, node_slot] = xy
                    camera_points[frame_slot, instance_slot, node_slot] = camera
                    world_points[frame_slot, instance_slot, node_slot] = world
                    node_confidence[frame_slot, instance_slot, node_slot] = max(
                        raw_confidence, 1e-8
                    )
                    position_sigma[frame_slot, instance_slot, node_slot] = sigma
                path = np.rint(centerline).astype(np.int32)
                cv2.polylines(overlay, [path], False, (0, 255, 255), 2, cv2.LINE_AA)
                for node_slot, (x, y) in enumerate(path):
                    color = (255, 64, 64) if node_slot == 0 else (64, 255, 64)
                    cv2.circle(overlay, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
            except (ValueError, RuntimeError) as exc:
                extraction_errors.append(
                    {
                        "source_frame": int(frame_indices[frame_slot]),
                        "instance_id": instance_id,
                        "error": str(exc),
                    }
                )
        overlay_frames.append(overlay)

    contract = StemCenterlineContract(
        instance_ids=instance_ids,
        coordinate_frame="world:da3_learned_metric",
        timeline="frame:source_video",
        nodes_per_stem=args.nodes_per_stem,
        geometry_evidence=EvidenceClass.FOUNDATION_MODEL_ESTIMATE,
    )
    validation = validate_stem_centerlines(
        np,
        contract=contract,
        frame_indices=frame_indices,
        centerlines_m=world_points,
        confidence=node_confidence,
    )
    artifact = output_dir / "metric-stem-centerlines.npz"
    np.savez_compressed(
        artifact,
        source_frame_indices=frame_indices.astype(np.int32),
        instance_ids=np.asarray(instance_ids),
        centerline_pixels_da3=pixels,
        centerlines_camera_m=camera_points,
        centerlines_world_m=world_points,
        confidence=node_confidence,
        position_sigma_m=position_sigma,
        coordinate_frame=np.asarray("world:da3_learned_metric"),
        evidence_class=np.asarray("foundation_model_estimate"),
    )
    overlay_path = output_dir / "centerline-overlay.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (processed_width, processed_height),
    )
    if not writer.isOpened():
        raise RuntimeError("could not open centerline overlay writer")
    for frame in overlay_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    elapsed = time.perf_counter() - started
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "status_reason": (
            "centerlines are metric foundation-model estimates; learned scale uncertainty and "
            "frame-complete dynamic correspondence remain uncalibrated"
        ),
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "opencv-python")
        },
        "seed": args.seed,
        "git": _git_state(),
        "inputs": {
            "da3_samples": {"path": str(da3_path), "sha256": _sha256(da3_path)},
            "stem_masks": {"path": str(masks_path), "sha256": _sha256(masks_path)},
        },
        "coordinate_frames": {
            "mask": mask_frame.to_dict(),
            "candidate": target_frame.to_dict(),
            "da3_pixels": {
                "width": processed_width,
                "height": processed_height,
                "transform": "whole-image resize from candidate; no crop",
            },
            "camera_metric": "camera:da3_processed_metric",
            "world_metric": "world:da3_learned_metric",
            "timeline": "frame:source_video",
        },
        "centerlines": {
            "instances": list(instance_ids),
            "frames": len(frame_indices),
            "nodes_per_stem": args.nodes_per_stem,
            "source_frame_indices": frame_indices.tolist(),
            "extraction_errors": extraction_errors,
            "uncertainty": (
                "local depth MAD plus half-pixel deprojection term; heuristic, not sensor calibration"
            ),
        },
        "validation": validation,
        "performance": {
            "wall_seconds": elapsed,
            "stem_frames_per_second": len(frame_indices) * len(instance_ids)
            / max(elapsed, 1e-12),
        },
        "outputs": {
            "centerlines": {"path": str(artifact), "sha256": _sha256(artifact)},
            "overlay": {"path": str(overlay_path), "sha256": _sha256(overlay_path)},
        },
        "limitations": [
            "SAM2 instance masks are visual observations and may include flower-head pixels.",
            "Skeleton geodesics impose root-to-tip topology but do not measure botanical branching.",
            "DA3 learned metric scale is not an independently calibrated sensor scale.",
            "No contact force is inferred in this stage.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if validation["passed"] and not extraction_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
