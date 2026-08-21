#!/usr/bin/env python3
"""Run one pinned video segmentation backend on shared prepared inputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
import time
from statistics import median
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.evaluation.segmentation_ab import (  # noqa: E402
    AB_SCHEMA_VERSION,
    SAM2_MODEL_ID,
    SAM31_MODEL_ID,
    MaskGeometryThresholds,
    capture_git_state,
    effective_component_area_threshold,
    load_json_object,
    parse_tracker_spec,
    save_packed_masks,
    score_attachment_distance,
    score_centroid_continuity,
    score_mask_geometry,
    validate_sam31_config,
    validate_task_config,
)
from phiagent.evaluation.video_proxy import file_sha256  # noqa: E402
from phiagent.rendering.wan_animate import query_gpus, select_gpu  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(SAM2_MODEL_ID, SAM31_MODEL_ID), required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if match is None:
        raise ValueError(f"could not parse version: {value!r}")
    return tuple(int(part) for part in match.groups(default="0"))


def _validate_prepared_input(
    prepared_dir: Path,
    task_config_path: Path,
) -> tuple[dict[str, object], str]:
    manifest_path = prepared_dir / "input-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_json_object(manifest_path, "prepared input manifest")
    if manifest.get("schema_version") != AB_SCHEMA_VERSION:
        raise ValueError("prepared input manifest schema is unsupported")
    if manifest.get("task_config_sha256") != file_sha256(task_config_path):
        raise RuntimeError("prepared input binds a different task config")
    frame_dir = Path(str(manifest["frame_dir"]))
    if not frame_dir.is_dir():
        raise FileNotFoundError(frame_dir)
    scoring_frame_dir = Path(str(manifest["scoring_frame_dir"]))
    if not scoring_frame_dir.is_dir():
        raise FileNotFoundError(scoring_frame_dir)
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != int(manifest["frame_count"]):
        raise ValueError("prepared frame manifest is incomplete")
    for expected_index, raw in enumerate(frames):
        if not isinstance(raw, dict) or raw.get("index") != expected_index:
            raise ValueError("prepared frame indices are not contiguous")
        frame_path = Path(str(raw.get("path")))
        if not frame_path.is_file():
            raise FileNotFoundError(frame_path)
        if file_sha256(frame_path) != raw.get("sha256"):
            raise RuntimeError(f"prepared frame hash differs: {frame_path}")
        scoring_frame_path = Path(str(raw.get("scoring_path")))
        if not scoring_frame_path.is_file():
            raise FileNotFoundError(scoring_frame_path)
        if file_sha256(scoring_frame_path) != raw.get("scoring_sha256"):
            raise RuntimeError(f"prepared scoring-frame hash differs: {scoring_frame_path}")
    initial_masks = Path(str(manifest["initial_masks"]))
    if not initial_masks.is_file():
        raise FileNotFoundError(initial_masks)
    if file_sha256(initial_masks) != manifest.get("initial_masks_sha256"):
        raise RuntimeError("prepared initial-mask hash differs")
    return manifest, file_sha256(manifest_path)


def _load_initial_masks(
    manifest: Mapping[str, object],
    np: Any,
) -> tuple[list[int], dict[int, Any], dict[int, str]]:
    objects = manifest.get("objects")
    if not isinstance(objects, list):
        raise ValueError("prepared input requires object metadata")
    object_ids = []
    names: dict[int, str] = {}
    masks: dict[int, Any] = {}
    expected_shape = (
        int(manifest["frame_size"][1]),
        int(manifest["frame_size"][0]),
    )
    with np.load(Path(str(manifest["initial_masks"])), allow_pickle=False) as archive:
        for raw in objects:
            if not isinstance(raw, dict):
                raise ValueError("prepared object metadata must be an object")
            object_id = int(raw["object_id"])
            key = str(raw["mask_key"])
            if key not in archive:
                raise ValueError(f"prepared masks are missing {key}")
            mask = archive[key].astype(bool)
            if mask.shape != expected_shape:
                raise ValueError(
                    f"prepared mask {key} has shape {mask.shape}; expected {expected_shape}"
                )
            object_ids.append(object_id)
            names[object_id] = str(raw["name"])
            masks[object_id] = mask
    return object_ids, masks, names


def _track_sam2(
    *,
    spec: Any,
    manifest: Mapping[str, object],
    object_ids: list[int],
    initial_masks: Mapping[int, Any],
    torch: Any,
) -> dict[int, list[Any | None]]:
    sys.path.insert(0, str(spec.repository))
    import sam2.modeling.sam.transformer as sam_transformer
    from sam2.build_sam import build_sam2_video_predictor

    torch.backends.cuda.enable_math_sdp(True)
    sam_transformer.MATH_KERNEL_ON = True
    predictor = build_sam2_video_predictor(
        spec.model_config,
        str(spec.checkpoint),
        device="cuda",
    )
    state = predictor.init_state(video_path=str(manifest["frame_dir"]))
    frame_count = int(manifest["frame_count"])
    initial_frame = int(manifest["initial_frame_index"])
    masks_by_id: dict[int, list[Any | None]] = {
        object_id: [None] * frame_count for object_id in object_ids
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for object_id in object_ids:
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=initial_frame,
                obj_id=object_id,
                mask=initial_masks[object_id],
            )
        directions = [(initial_frame, None, False)]
        if initial_frame > 0:
            directions.append((initial_frame, initial_frame, True))
        for start, maximum, reverse in directions:
            for frame_index, output_ids, logits in predictor.propagate_in_video(
                state,
                start_frame_idx=start,
                max_frame_num_to_track=maximum,
                reverse=reverse,
            ):
                for index, object_id_raw in enumerate(output_ids):
                    object_id = int(object_id_raw)
                    if object_id in masks_by_id:
                        masks_by_id[object_id][int(frame_index)] = (
                            logits[index].squeeze().detach().cpu().numpy() > 0
                        )
    predictor.reset_state(state)
    return masks_by_id


def _track_sam31(
    *,
    spec: Any,
    manifest: Mapping[str, object],
    object_ids: list[int],
    initial_masks: Mapping[int, Any],
    torch: Any,
    np: Any,
) -> dict[int, list[Any | None]]:
    if sys.version_info < (3, 12):
        raise RuntimeError("official SAM3.1 runtime requires Python 3.12 or newer")
    if _version_tuple(torch.__version__) < (2, 7, 0):
        raise RuntimeError("official SAM3.1 runtime requires PyTorch 2.7 or newer")
    if torch.version.cuda is None or _version_tuple(torch.version.cuda) < (12, 6, 0):
        raise RuntimeError("official SAM3.1 runtime requires CUDA 12.6 or newer")
    sys.path.insert(0, str(spec.repository))
    from sam3.model_builder import build_sam3_multiplex_video_model
    from sam3.model.video_tracking_multiplex_demo import (
        VideoTrackingMultiplexDemo,
    )

    predictor = build_sam3_multiplex_video_model(
        checkpoint_path=str(spec.checkpoint),
        load_from_HF=False,
        multiplex_count=spec.multiplex_count,
        device="cuda",
        compile=spec.compile_model,
    )
    predictor.eval()
    # The builder returns the cached-feature subclass used by the detector
    # wrapper. Mask-only VOS needs the parent loader to retain source frames.
    state = VideoTrackingMultiplexDemo.init_state(
        predictor,
        video_path=str(manifest["frame_dir"]),
        offload_video_to_cpu=spec.offload_video_to_cpu,
        offload_state_to_cpu=spec.offload_state_to_cpu,
    )
    frame_count = int(manifest["frame_count"])
    initial_frame = int(manifest["initial_frame_index"])
    masks_by_id: dict[int, list[Any | None]] = {
        object_id: [None] * frame_count for object_id in object_ids
    }
    stacked_masks = torch.from_numpy(
        np.stack([initial_masks[object_id] for object_id in object_ids])
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.add_new_masks(
            inference_state=state,
            frame_idx=initial_frame,
            obj_ids=object_ids,
            masks=stacked_masks,
        )
        predictor.propagate_in_video_preflight(state)
        directions = [(initial_frame, None, False)]
        if initial_frame > 0:
            directions.append((initial_frame, initial_frame, True))
        for start, maximum, reverse in directions:
            outputs = predictor.propagate_in_video(
                state,
                start_frame_idx=start,
                max_frame_num_to_track=maximum,
                reverse=reverse,
            )
            for frame_index, output_ids, _, video_res_masks, _ in outputs:
                for index, object_id_raw in enumerate(output_ids):
                    object_id = int(object_id_raw)
                    if object_id in masks_by_id:
                        masks_by_id[object_id][int(frame_index)] = (
                            video_res_masks[index].squeeze().detach().cpu().numpy() > 0
                        )
    del state
    del predictor
    return masks_by_id


def _major_axis(cv2: Any, np: Any, mask: Any) -> float:
    points = np.column_stack(np.nonzero(mask))[:, ::-1].astype("float32")
    if len(points) < 5:
        return 0.0
    (_, _), (width, height), _ = cv2.minAreaRect(points)
    return float(max(width, height))


def _meaningful_component_count(
    cv2: Any,
    mask: Any,
    *,
    minimum_area_pixels: int,
) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"))
    return sum(
        int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area_pixels for index in range(1, count)
    )


def _minimum_mask_distance(
    cv2: Any,
    np: Any,
    first: Any,
    second: Any,
) -> float:
    if not np.any(first) or not np.any(second):
        return math.hypot(*first.shape[:2])
    distance_to_second = cv2.distanceTransform(
        (second == 0).astype("uint8"),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    return float(np.min(distance_to_second[first.astype(bool)]))


def _score_masks(
    *,
    task_config: Mapping[str, object],
    thresholds: MaskGeometryThresholds,
    masks_by_name: Mapping[str, Any],
    frames: list[Any],
    cv2: Any,
    np: Any,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    object_results = {}
    for name, mask_array in masks_by_name.items():
        areas = [int(np.count_nonzero(mask)) for mask in mask_array]
        baseline_area = float(median(areas[: thresholds.baseline_hold_frames]))
        component_area_threshold = effective_component_area_threshold(
            baseline_area,
            thresholds,
        )
        components = [
            _meaningful_component_count(
                cv2,
                mask,
                minimum_area_pixels=component_area_threshold,
            )
            for mask in mask_array
        ]
        major_axes = [_major_axis(cv2, np, mask) for mask in mask_array]
        centroids = []
        for mask in mask_array:
            rows, columns = np.nonzero(mask)
            centroids.append(
                (0.0, 0.0) if not len(columns) else (float(columns.mean()), float(rows.mean()))
            )
        score = score_mask_geometry(
            areas,
            major_axes,
            components,
            thresholds=thresholds,
        )
        centroid_continuity = score_centroid_continuity(
            centroids,
            maximum_step_pixels=thresholds.maximum_centroid_step_pixels,
        )
        score["gate_results"]["centroid_step"] = centroid_continuity["passed"]
        score["passed"] = bool(score["passed"] and centroid_continuity["passed"])
        object_results[name] = {
            **score,
            "area_pixels": areas,
            "major_axis_pixels": major_axes,
            "connected_components": components,
            "effective_minimum_component_area_pixels": component_area_threshold,
            "centroid_continuity": centroid_continuity,
        }

    relationship_results = {}
    relationships = task_config.get("relationships", [])
    if not isinstance(relationships, list):
        raise ValueError("relationships must be a list")
    for raw in relationships:
        if not isinstance(raw, dict):
            raise ValueError("relationship must be an object")
        name = str(raw["name"])
        if name in relationship_results:
            raise ValueError(f"duplicate relationship name: {name}")
        first_name = str(raw["first_object"])
        second_name = str(raw["second_object"])
        if first_name not in masks_by_name or second_name not in masks_by_name:
            raise ValueError(f"{name} references an unknown object")
        distances = [
            _minimum_mask_distance(cv2, np, first, second)
            for first, second in zip(
                masks_by_name[first_name],
                masks_by_name[second_name],
            )
        ]
        relationship_results[name] = {
            "first_object": first_name,
            "second_object": second_name,
            **score_attachment_distance(
                distances,
                baseline_hold_frames=int(raw["baseline_hold_frames"]),
                maximum_distance_increase_pixels=float(raw["maximum_distance_increase_pixels"]),
            ),
        }

    color_attachment_results = {}
    color_attachments = task_config.get("color_attachments", [])
    if not isinstance(color_attachments, list):
        raise ValueError("color_attachments must be a list")
    for raw in color_attachments:
        if not isinstance(raw, dict):
            raise ValueError("color attachment must be an object")
        name = str(raw["name"])
        if name in color_attachment_results:
            raise ValueError(f"duplicate color attachment name: {name}")
        object_name = str(raw["object"])
        if object_name not in masks_by_name:
            raise ValueError(f"{name} references an unknown object")
        lower = np.asarray(raw["hsv_lower"], dtype=np.int32)
        upper = np.asarray(raw["hsv_upper"], dtype=np.int32)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or np.any(lower < 0)
            or np.any(upper > np.asarray([179, 255, 255]))
            or np.any(lower > upper)
        ):
            raise ValueError(f"{name} has invalid OpenCV HSV bounds")
        distances = []
        color_pixels = []
        for frame, mask in zip(frames, masks_by_name[object_name]):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(
                hsv,
                lower.astype(np.uint8),
                upper.astype(np.uint8),
            )
            color_pixels.append(int(np.count_nonzero(color_mask)))
            distances.append(_minimum_mask_distance(cv2, np, mask, color_mask > 0))
        color_attachment_results[name] = {
            "object": object_name,
            "hsv_lower": lower.tolist(),
            "hsv_upper": upper.tolist(),
            "minimum_color_pixels": min(color_pixels),
            **score_attachment_distance(
                distances,
                baseline_hold_frames=int(raw["baseline_hold_frames"]),
                maximum_distance_increase_pixels=float(raw["maximum_distance_increase_pixels"]),
            ),
        }
    return object_results, relationship_results, color_attachment_results


def _write_storyboard(
    *,
    path: Path,
    frames: list[Any],
    masks_by_name: Mapping[str, Any],
    cv2: Any,
    np: Any,
) -> None:
    indices = tuple(round(index * (len(frames) - 1) / 11) for index in range(12))
    colors = ((0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0))
    cells = []
    for frame_index in indices:
        cell = cv2.resize(frames[frame_index], (416, 240), interpolation=cv2.INTER_AREA)
        for object_index, (_, masks) in enumerate(sorted(masks_by_name.items())):
            mask = cv2.resize(
                masks[frame_index].astype(np.uint8),
                (416, 240),
                interpolation=cv2.INTER_NEAREST,
            )
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(
                cell,
                contours,
                -1,
                colors[object_index % len(colors)],
                2,
            )
        cv2.putText(
            cell,
            f"frame {frame_index}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cells.append(cell)
    storyboard = np.vstack([np.hstack(cells[row * 4 : (row + 1) * 4]) for row in range(3)])
    if not cv2.imwrite(str(path), storyboard):
        raise RuntimeError(f"failed to write storyboard: {path}")


def _package_versions(model_id: str) -> dict[str, str | None]:
    names = ["torch", "torchvision", "numpy", "opencv-python"]
    names.append("sam-2" if model_id == SAM2_MODEL_ID else "sam3")
    versions = {}
    for package in names:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main() -> int:
    args = _parser().parse_args()
    prepared_dir = args.prepared_dir.expanduser().resolve()
    task_config_path = args.task_config.expanduser().resolve()
    model_config_path = args.model_config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse model run directory: {output}")
    task_config = load_json_object(task_config_path, "task config")
    thresholds = validate_task_config(task_config)
    if args.model == SAM2_MODEL_ID:
        model_config = task_config["sam2"]
    else:
        model_config_document = load_json_object(model_config_path, "SAM3.1 config")
        validate_sam31_config(model_config_document)
        model_config = model_config_document
    spec = parse_tracker_spec(
        model_config,
        project_root=PROJECT_ROOT,
        model_id=args.model,
    )
    manifest, prepared_input_sha256 = _validate_prepared_input(prepared_dir, task_config_path)

    output.mkdir(parents=True)
    gpus, inventory, processes = query_gpus()
    selected = select_gpu(gpus, args.gpu, args.minimum_free_gpu_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    preflight = {
        "schema_version": AB_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model,
        "tracker": spec.public_dict(),
        "prepared_input_sha256": prepared_input_sha256,
        "selected_gpu": asdict(selected),
        "gpu_inventory": inventory,
        "gpu_processes": processes,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "command": list(sys.argv),
        "git": capture_git_state(PROJECT_ROOT),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    preflight_path = output / "preflight.json"
    _write_json(preflight_path, preflight)

    import cv2
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA after GPU selection")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("model worker must see exactly one CUDA device after selection")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    object_ids, initial_masks, object_names = _load_initial_masks(manifest, np)
    torch.cuda.synchronize()
    started = time.perf_counter()
    if args.model == SAM2_MODEL_ID:
        masks_by_id = _track_sam2(
            spec=spec,
            manifest=manifest,
            object_ids=object_ids,
            initial_masks=initial_masks,
            torch=torch,
        )
    else:
        masks_by_id = _track_sam31(
            spec=spec,
            manifest=manifest,
            object_ids=object_ids,
            initial_masks=initial_masks,
            torch=torch,
            np=np,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    missing = [
        (object_id, frame_index)
        for object_id, masks in masks_by_id.items()
        for frame_index, mask in enumerate(masks)
        if mask is None
    ]
    if missing:
        raise RuntimeError(
            f"{args.model} propagation missed {len(missing)} object frames; "
            f"first missing entry is {missing[0]}"
        )
    masks_by_name = {
        object_names[object_id]: np.stack(masks).astype(np.uint8)
        for object_id, masks in masks_by_id.items()
    }
    scoring_frame_dir = Path(str(manifest["scoring_frame_dir"]))
    frames = [
        cv2.imread(
            str(scoring_frame_dir / f"{index:05d}.png"),
            cv2.IMREAD_COLOR,
        )
        for index in range(int(manifest["frame_count"]))
    ]
    if any(frame is None for frame in frames):
        raise RuntimeError("failed to reload one or more prepared frames")
    object_results, relationship_results, color_attachment_results = _score_masks(
        task_config=task_config,
        thresholds=thresholds,
        masks_by_name=masks_by_name,
        frames=frames,
        cv2=cv2,
        np=np,
    )
    masks_path = output / "masks.npz"
    mask_keys = save_packed_masks(
        masks_path,
        masks_by_name,
        frame_count=int(manifest["frame_count"]),
        height=int(manifest["frame_size"][1]),
        width=int(manifest["frame_size"][0]),
    )
    storyboard_path = output / "mask-storyboard.jpg"
    _write_storyboard(
        path=storyboard_path,
        frames=frames,
        masks_by_name=masks_by_name,
        cv2=cv2,
        np=np,
    )
    incumbent_diagnostic_pass = (
        all(bool(value["passed"]) for value in object_results.values())
        and all(bool(value["passed"]) for value in relationship_results.values())
        and all(bool(value["passed"]) for value in color_attachment_results.values())
    )
    result = {
        "schema_version": AB_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL",
        "model_id": args.model,
        "role": spec.role,
        "decision_eligible": args.model == SAM2_MODEL_ID,
        "hard_gates_passed": (incumbent_diagnostic_pass if args.model == SAM2_MODEL_ID else None),
        "incumbent_threshold_diagnostic_passed": incumbent_diagnostic_pass,
        "incumbent_threshold_diagnostics": object_results,
        "incumbent_relationship_diagnostics": relationship_results,
        "incumbent_color_attachment_diagnostics": color_attachment_results,
        "threshold_provenance": {
            "evaluator_epoch_id": task_config["sam2"]["evaluator_epoch_id"],
            "source_model": SAM2_MODEL_ID,
            "decision_bearing": args.model == SAM2_MODEL_ID,
            "warning": (
                None
                if args.model == SAM2_MODEL_ID
                else "SAM2 thresholds are diagnostic only for the SAM3.1 shadow output"
            ),
        },
        "coordinate_frame": task_config["coordinate_frame"],
        "prepared_input": str(prepared_dir / "input-manifest.json"),
        "prepared_input_sha256": prepared_input_sha256,
        "task_config": str(task_config_path),
        "task_config_sha256": file_sha256(task_config_path),
        "model_config": str(model_config_path),
        "model_config_sha256": file_sha256(model_config_path),
        "preflight": str(preflight_path),
        "preflight_sha256": file_sha256(preflight_path),
        "tracker": spec.public_dict(),
        "masks": str(masks_path),
        "masks_sha256": file_sha256(masks_path),
        "mask_keys": mask_keys,
        "storyboard": str(storyboard_path),
        "storyboard_sha256": file_sha256(storyboard_path),
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_mib": (torch.cuda.max_memory_allocated() / (1024 * 1024)),
            "selected_gpu": asdict(selected),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "seed": args.seed,
        "command": list(sys.argv),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": _package_versions(args.model),
        "git": capture_git_state(PROJECT_ROOT),
        "evidence_boundary": (
            task_config["evidence_boundary"]
            if args.model == SAM2_MODEL_ID
            else (
                "SAM3.1 shadow camera-frame masks and agreement diagnostics only. "
                "They cannot alter the authoritative SAM2 decision in this epoch."
            )
        ),
    }
    _write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
