#!/usr/bin/env python3
"""Track explicitly prompted real flower stems through one critical window.

This is intentionally narrower than the older flower-union tracker.  Each
object has an immutable ID, a manually reviewed anchor box, positive stem/head
points, and negative hand/neighbor points in a named camera-pixel frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402


SAM2_COMMIT = "0e78a118995e66bb27d78518c4bd9a3e95b4e266"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--seed-spec", type=Path, required=True)
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", default="sam2_hiera_l.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-basename",
        default="flower-stem-instances-packed.npz",
        help="NPZ filename for the prompted instance tracks",
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_seed_spec(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0.0":
        raise ValueError("seed spec must be a schema_version 1.0.0 object")
    if raw.get("coordinate_frame") != "camera:source_video_pixels":
        raise ValueError("seed coordinates must use camera:source_video_pixels")
    raw = dict(raw)
    indices = raw.get("source_frame_indices")
    frame_range = raw.get("source_frame_range")
    if indices is None and frame_range is not None:
        if (
            not isinstance(frame_range, list)
            or len(frame_range) != 3
            or any(type(value) is not int for value in frame_range)  # noqa: E721
            or frame_range[0] < 0
            or frame_range[1] <= frame_range[0]
            or frame_range[2] <= 0
        ):
            raise ValueError("source_frame_range must be [start, end_exclusive, step]")
        indices = list(range(*frame_range))
        raw["source_frame_indices"] = indices
    if (
        not isinstance(indices, list)
        or len(indices) < 2
        or any(type(value) is not int or value < 0 for value in indices)
        or indices != sorted(set(indices))
    ):
        raise ValueError("source_frame_indices must be unique increasing integers")
    instances = raw.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("seed spec must contain at least one instance")
    ids: set[str] = set()
    object_ids: set[int] = set()

    def validate_prompt(
        prompt: Any,
        *,
        instance_id: str,
        label: str,
        minimum_area: int,
        maximum_area: int,
    ) -> None:
        if not isinstance(prompt, dict):
            raise ValueError(f"{instance_id}.{label} must be an object")
        if prompt.get("anchor_source_frame") not in indices:
            raise ValueError(f"{instance_id} {label} anchor must be one selected source frame")
        box = prompt.get("box_xyxy")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(not isinstance(value, (int, float)) for value in box)
            or not (box[0] < box[2] and box[1] < box[3])
        ):
            raise ValueError(f"{instance_id}.{label}.box_xyxy is invalid")
        for key in ("positive_points_xy", "negative_points_xy"):
            points = prompt.get(key)
            if not isinstance(points, list) or (key == "positive_points_xy" and not points):
                raise ValueError(f"{instance_id}.{label}.{key} must be a point list")
            if any(
                not isinstance(point, list)
                or len(point) != 2
                or any(not isinstance(value, (int, float)) for value in point)
                for point in points
            ):
                raise ValueError(f"{instance_id}.{label}.{key} contains an invalid point")
        prompt_minimum = prompt.get("minimum_area_pixels", minimum_area)
        prompt_maximum = prompt.get("maximum_area_pixels", maximum_area)
        if (
            type(prompt_minimum) is not int
            or type(prompt_maximum) is not int
            or prompt_minimum <= 0
            or prompt_maximum <= prompt_minimum
        ):
            raise ValueError(f"{instance_id}.{label} area bounds are invalid")

    for row in instances:
        if not isinstance(row, dict):
            raise ValueError("each instance must be an object")
        instance_id = row.get("instance_id")
        object_id = row.get("object_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("instance_id must be a non-empty string")
        if type(object_id) is not int or object_id <= 0:
            raise ValueError("object_id must be a positive integer")
        if instance_id in ids or object_id in object_ids:
            raise ValueError("instance_id and object_id must be unique")
        ids.add(instance_id)
        object_ids.add(object_id)
        minimum = row.get("minimum_area_pixels")
        maximum = row.get("maximum_area_pixels")
        if (
            type(minimum) is not int
            or type(maximum) is not int
            or minimum <= 0
            or maximum <= minimum
        ):
            raise ValueError(f"{instance_id} area bounds are invalid")
        validate_prompt(
            row,
            instance_id=instance_id,
            label="primary_prompt",
            minimum_area=minimum,
            maximum_area=maximum,
        )
        corrections = row.get("correction_prompts", [])
        if not isinstance(corrections, list):
            raise ValueError(f"{instance_id}.correction_prompts must be a list")
        correction_frames: set[int] = set()
        for correction_index, correction in enumerate(corrections):
            validate_prompt(
                correction,
                instance_id=instance_id,
                label=f"correction_prompts[{correction_index}]",
                minimum_area=minimum,
                maximum_area=maximum,
            )
            correction_frame = correction["anchor_source_frame"]
            if correction_frame == row["anchor_source_frame"] or correction_frame in correction_frames:
                raise ValueError(f"{instance_id} prompt frames must be unique")
            correction_frames.add(correction_frame)
    return raw


def candidate_statistics(
    np: Any,
    mask: Any,
    *,
    positive_points: Any,
    negative_points: Any,
    box: Any,
    score: float,
    minimum_area: int,
    maximum_area: int,
) -> dict[str, float | int | bool]:
    mask = mask.astype(bool)
    height, width = mask.shape

    def hits(points: Any) -> int:
        if len(points) == 0:
            return 0
        rounded = np.rint(points).astype(np.int32)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
        return int(sum(bool(mask[y, x]) for x, y in rounded))

    positive_hits = hits(positive_points)
    negative_hits = hits(negative_points)
    area = int(np.count_nonzero(mask))
    box_mask = np.zeros_like(mask)
    left, top, right, bottom = np.rint(box).astype(np.int32)
    box_mask[max(0, top) : min(height, bottom + 1), max(0, left) : min(width, right + 1)] = True
    outside_fraction = float(np.count_nonzero(mask & ~box_mask) / max(1, area))
    positive_fraction = float(positive_hits / max(1, len(positive_points)))
    negative_fraction = float(negative_hits / max(1, len(negative_points)))
    plausible = (
        minimum_area <= area <= maximum_area
        and positive_fraction >= 2.0 / 3.0
        and negative_fraction <= 0.25
        and outside_fraction <= 0.45
    )
    rank = (
        4.0 * positive_fraction
        - 5.0 * negative_fraction
        - 2.0 * outside_fraction
        + float(score)
    )
    return {
        "area_pixels": area,
        "positive_hits": positive_hits,
        "positive_fraction": positive_fraction,
        "negative_hits": negative_hits,
        "negative_fraction": negative_fraction,
        "outside_box_fraction": outside_fraction,
        "sam_score": float(score),
        "plausible": plausible,
        "rank": rank,
    }


def select_anchor_candidate(
    np: Any,
    masks: Any,
    scores: Any,
    *,
    positive_points: Any,
    negative_points: Any,
    box: Any,
    minimum_area: int,
    maximum_area: int,
) -> tuple[int, list[dict[str, float | int | bool]]]:
    rows = [
        candidate_statistics(
            np,
            mask,
            positive_points=positive_points,
            negative_points=negative_points,
            box=box,
            score=float(score),
            minimum_area=minimum_area,
            maximum_area=maximum_area,
        )
        for mask, score in zip(masks, scores)
    ]
    plausible = [index for index, row in enumerate(rows) if row["plausible"]]
    if not plausible:
        raise RuntimeError(f"SAM2 produced no plausible flower instance: {rows}")
    selected = max(plausible, key=lambda index: float(rows[index]["rank"]))
    return selected, rows


def _largest_component(cv2: Any, np: Any, mask: Any) -> Any:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count <= 1:
        return mask.astype(bool)
    component = max(
        range(1, count), key=lambda index: int(stats[index, cv2.CC_STAT_AREA])
    )
    return labels == component


def merge_directional_track_mask(np: Any, existing: Any | None, candidate: Any) -> Any:
    """Keep valid forward evidence and let reverse propagation fill only holes."""
    candidate = candidate.astype(bool)
    if existing is None:
        return candidate
    if np.any(existing) or not np.any(candidate):
        return existing
    return candidate


def _contact_sheet(
    cv2: Any, np: Any, frames: list[Any], masks: list[Any], label: str
) -> Any:
    cells = []
    review_indices = np.unique(
        np.rint(np.linspace(0, len(frames) - 1, min(40, len(frames)))).astype(np.int32)
    )
    for index in review_indices:
        frame, mask = frames[int(index)], masks[int(index)]
        rendered = frame.copy()
        rendered[mask] = (
            0.28 * rendered[mask] + 0.72 * np.asarray([30, 40, 250])
        ).astype("uint8")
        contours, _ = cv2.findContours(
            mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(rendered, contours, -1, (255, 255, 255), 1)
        cv2.putText(
            rendered,
            f"{label}  local={index}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cells.append(cv2.resize(rendered, (416, 240), interpolation=cv2.INTER_AREA))
    black = np.zeros_like(cells[0])
    while len(cells) % 4:
        cells.append(black.copy())
    rows = [cv2.hconcat(cells[index : index + 4]) for index in range(0, len(cells), 4)]
    return cv2.vconcat(rows)


def main() -> int:
    args = _parser().parse_args()
    if Path(args.output_basename).name != args.output_basename or not args.output_basename.endswith(
        ".npz"
    ):
        raise ValueError("output-basename must be one local .npz filename")
    paths = {
        "source_video": args.source_video.expanduser().resolve(),
        "seed_spec": args.seed_spec.expanduser().resolve(),
        "sam2_repo": args.sam2_repo.expanduser().resolve(),
        "sam2_checkpoint": args.sam2_checkpoint.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {name}: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    spec = validate_seed_spec(json.loads(paths["seed_spec"].read_text()))

    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=paths["sam2_repo"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != SAM2_COMMIT:
        raise RuntimeError(f"SAM2 commit is {commit}, expected {SAM2_COMMIT}")

    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, str(paths["sam2_repo"]))
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    output.mkdir(parents=True)
    frames_dir = output / "input/frames"
    review_dir = output / "review"
    provenance_dir = output / "provenance/execution-sources"
    frames_dir.mkdir(parents=True)
    review_dir.mkdir()
    provenance_dir.mkdir(parents=True)
    frozen_source = provenance_dir / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), frozen_source)

    capture = cv2.VideoCapture(str(paths["source_video"]))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode source video: {paths['source_video']}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    for local_index, source_index in enumerate(spec["source_frame_indices"]):
        if source_index >= total:
            raise ValueError(f"source frame {source_index} exceeds video length {total}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"decode failed at source frame {source_index}")
        frames.append(frame)
        if not cv2.imwrite(str(frames_dir / f"{local_index:05d}.jpg"), frame):
            raise RuntimeError(f"could not write local frame {local_index}")
    capture.release()

    torch.manual_seed(args.seed)
    predictor = build_sam2_video_predictor(
        args.sam2_config, str(paths["sam2_checkpoint"]), device="cuda"
    )
    image_predictor = SAM2ImagePredictor(predictor)
    prompt_masks: dict[tuple[int, int], Any] = {}
    seed_rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for instance in spec["instances"]:
            prompts = [("primary", instance)] + [
                (f"correction-{index + 1}", correction)
                for index, correction in enumerate(instance.get("correction_prompts", []))
            ]
            for prompt_kind, prompt in prompts:
                anchor_local = spec["source_frame_indices"].index(
                    prompt["anchor_source_frame"]
                )
                image_predictor.set_image(
                    cv2.cvtColor(frames[anchor_local], cv2.COLOR_BGR2RGB)
                )
                positive = np.asarray(prompt["positive_points_xy"], dtype=np.float32)
                negative = np.asarray(prompt["negative_points_xy"], dtype=np.float32).reshape(-1, 2)
                points = np.concatenate((positive, negative), axis=0)
                labels = np.concatenate(
                    (
                        np.ones(len(positive), dtype=np.int32),
                        np.zeros(len(negative), dtype=np.int32),
                    )
                )
                box = np.asarray(prompt["box_xyxy"], dtype=np.float32)
                candidates, scores, _ = image_predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=box,
                    multimask_output=True,
                )
                selected_index, rows = select_anchor_candidate(
                    np,
                    candidates,
                    scores,
                    positive_points=positive,
                    negative_points=negative,
                    box=box,
                    minimum_area=prompt.get(
                        "minimum_area_pixels", instance["minimum_area_pixels"]
                    ),
                    maximum_area=prompt.get(
                        "maximum_area_pixels", instance["maximum_area_pixels"]
                    ),
                )
                object_id = instance["object_id"]
                prompt_masks[(object_id, anchor_local)] = candidates[
                    selected_index
                ].astype(bool)
                seed_rows.append(
                    {
                        "instance_id": instance["instance_id"],
                        "object_id": object_id,
                        "prompt_kind": prompt_kind,
                        "anchor_local_frame": anchor_local,
                        "anchor_source_frame": prompt["anchor_source_frame"],
                        "selected_candidate": selected_index,
                        "candidates": [
                            {key: value for key, value in row.items() if key != "rank"}
                            for row in rows
                        ],
                    }
                )

        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )
        for (object_id, anchor_local), mask in sorted(
            prompt_masks.items(), key=lambda row: (row[0][1], row[0][0])
        ):
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=anchor_local,
                obj_id=object_id,
                mask=mask,
            )
        tracked: dict[int, list[Any | None]] = {
            instance["object_id"]: [None] * len(frames) for instance in spec["instances"]
        }
        anchor_min = min(anchor_local for _, anchor_local in prompt_masks)
        anchor_max = max(anchor_local for _, anchor_local in prompt_masks)
        for reverse in (False, True):
            for frame_index, object_ids, logits in predictor.propagate_in_video(
                state,
                start_frame_idx=anchor_max if reverse else anchor_min,
                reverse=reverse,
            ):
                for object_index, object_id in enumerate(object_ids):
                    object_id = int(object_id)
                    tracked[object_id][frame_index] = merge_directional_track_mask(
                        np,
                        tracked[object_id][frame_index],
                        logits[object_index, 0].detach().cpu().numpy() > 0.0,
                    )

    final_tracks = []
    metrics = {}
    for instance in spec["instances"]:
        object_id = instance["object_id"]
        missing = [index for index, mask in enumerate(tracked[object_id]) if mask is None]
        if missing:
            raise RuntimeError(f"{instance['instance_id']} missing local frames {missing}")
        masks = [
            _largest_component(cv2, np, mask) for mask in tracked[object_id]
        ]
        areas = np.asarray([np.count_nonzero(mask) for mask in masks], dtype=np.float64)
        centroids = []
        for mask in masks:
            ys, xs = np.where(mask)
            centroids.append((float(np.mean(xs)), float(np.mean(ys))))
        steps = [
            float(np.linalg.norm(np.asarray(second) - np.asarray(first)))
            for first, second in zip(centroids, centroids[1:])
        ]
        adjacent_ious = []
        for first, second in zip(masks, masks[1:]):
            adjacent_ious.append(
                float(np.count_nonzero(first & second) / max(1, np.count_nonzero(first | second)))
            )
        metrics[instance["instance_id"]] = {
            "empty_frames": int(np.count_nonzero(areas == 0)),
            "area_pixels_min": int(np.min(areas)),
            "area_pixels_max": int(np.max(areas)),
            "area_ratio_max_min": float(np.max(areas) / max(1.0, np.min(areas))),
            "centroid_step_max_pixels": max(steps),
            "adjacent_iou_min": min(adjacent_ious),
            "adjacent_iou_mean": float(np.mean(adjacent_ious)),
            "centroids_xy": centroids,
        }
        review = _contact_sheet(cv2, np, frames, masks, instance["instance_id"])
        cv2.imwrite(str(review_dir / f"{instance['instance_id']}.jpg"), review)
        final_tracks.append(np.stack(masks))
    final = np.stack(final_tracks)
    packed_path = output / args.output_basename
    np.savez_compressed(
        packed_path,
        masks_packed=np.packbits(
            final.reshape(final.shape[0], final.shape[1], -1), axis=2, bitorder="little"
        ),
        instance_ids=np.asarray([row["instance_id"] for row in spec["instances"]]),
        object_ids=np.asarray([row["object_id"] for row in spec["instances"]], dtype=np.int32),
        source_frame_indices=np.asarray(spec["source_frame_indices"], dtype=np.int32),
        height=np.asarray(height, dtype=np.int32),
        width=np.asarray(width, dtype=np.int32),
        bitorder=np.asarray("little"),
    )
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "method": "explicit_prompted_sam2_bidirectional_instance_tracking",
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "coordinate_frame": spec["coordinate_frame"],
        "source_frame_indices": spec["source_frame_indices"],
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path) if path.is_file() else None}
            for name, path in paths.items()
        },
        "execution_source": {"path": str(frozen_source), "sha256": _sha256(frozen_source)},
        "sam2_commit": commit,
        "gpu": {
            "selected": selected.physical_index,
            "name": selected.name,
            "free_mib": selected.free_mib,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "inventory_raw": inventory,
            "processes_raw": processes,
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "video": {"frames": total, "width": width, "height": height},
        "seed_selection": seed_rows,
        "metrics": metrics,
        "outputs": {"packed_masks": {"path": str(packed_path), "sha256": _sha256(packed_path)}},
        "limitations": [
            "Primary and optional correction prompts are manually selected and require storyboard review.",
            "SAM2 identity consistency is visual/temporal evidence, not physical identity instrumentation.",
            "This run covers only the frame sequence declared by its seed specification."
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
