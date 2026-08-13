#!/usr/bin/env python3
"""Remove the generated gray halo around the H3 robot with tracked matting."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.minimax_h3 import file_sha256  # noqa: E402
from phiagent.rendering.wan_animate import query_gpus  # noqa: E402

SAM2_COMMIT = "0e78a118995e66bb27d78518c4bd9a3e95b4e266"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state(project_root: Path) -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "available": status.returncode == 0,
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
        "error": status.stderr.strip() if status.returncode != 0 else None,
    }


def _flower_mask(
    cv2: Any,
    np: Any,
    frame: Any,
    *,
    dilation: int,
    exclude_skin_like: bool,
) -> Any:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue >= 26) & (hue <= 96) & (saturation >= 44) & (value >= 25)
    pink = (hue >= 145) & (hue <= 179) & (saturation >= 85) & (value >= 45)
    yellow = (hue >= 12) & (hue <= 35) & (saturation >= 80) & (value >= 65)
    height, width = frame.shape[:2]
    scene = np.zeros((height, width), dtype=bool)
    scene[round(height * 0.27) :, round(width * 0.31) :] = True
    scene[round(height * 0.61) :, :] = True
    scene[round(height * 0.36) :, round(width * 0.72) :] = True
    selected = (green | pink | yellow) & scene
    if exclude_skin_like:
        selected &= ~_skin_like_mask(cv2, np, frame)
    mask = selected.astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    if dilation:
        size = dilation * 2 + 1
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
    return mask


def _video_info(cv2: Any, path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    info: dict[str, int | float] = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "reported_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    info["decoded_frames"] = decoded
    return info


def _largest_component(cv2: Any, np: Any, mask: Any) -> Any:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _fill_mask_holes(cv2: Any, np: Any, mask: Any) -> Any:
    """Fill enclosed gaps without expanding the mask's exterior silhouette."""

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    filled = np.zeros(mask.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled.astype(bool)


def _align_mask(cv2: Any, mask: Any, width: int, height: int) -> Any:
    scale = max(width / mask.shape[1], height / mask.shape[0])
    scaled_width = round(mask.shape[1] * scale)
    scaled_height = round(mask.shape[0] * scale)
    resized = cv2.resize(
        mask,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_NEAREST,
    )
    left = max(0, (scaled_width - width) // 2)
    top = max(0, (scaled_height - height) // 2)
    return resized[top : top + height, left : left + width]


def _skin_like_mask(cv2: Any, np: Any, frame: Any) -> Any:
    _, cr, cb = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb))
    blue, green, red = cv2.split(frame.astype(np.float32))
    return (
        (cr >= 132)
        & (cr <= 180)
        & (cb >= 75)
        & (cb <= 135)
        & (red > green * 1.03)
        & (green > blue * 0.90)
    )


def compose_shadow_free_frame(
    cv2: Any,
    np: Any,
    *,
    source: Any,
    generated: Any,
    clean_plate: Any,
    robot_mask: Any,
    source_person_mask: Any,
    flower_mask: Any,
    dilation_pixels: int,
    source_human_residual_mask: Any | None = None,
    person_background_feather_sigma: float = 0.0,
) -> tuple[Any, dict[str, float]]:
    """Keep the tracked robot and replace its gray exterior with clean scene."""

    if dilation_pixels < 0:
        raise ValueError("dilation_pixels must be non-negative")
    core = robot_mask.astype(bool)
    keep = core.astype(np.uint8) * 255
    if dilation_pixels:
        size = dilation_pixels * 2 + 1
        keep = cv2.dilate(
            keep,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
    alpha = cv2.GaussianBlur(keep, (5, 5), 0.8).astype(np.float32) / 255.0
    alpha[core] = 1.0
    background = source.copy()
    person = source_person_mask.astype(bool)
    person_alpha = person.astype(np.float32)
    if person_background_feather_sigma < 0:
        raise ValueError("person_background_feather_sigma must be non-negative")
    if person_background_feather_sigma:
        person_alpha = cv2.GaussianBlur(
            person.astype(np.float32),
            (0, 0),
            person_background_feather_sigma,
        )
        person_alpha[person] = 1.0
        background = np.rint(
            clean_plate.astype(np.float32) * person_alpha[..., None]
            + source.astype(np.float32) * (1.0 - person_alpha[..., None])
        ).astype(np.uint8)
    else:
        background[person] = clean_plate[person]
    result = np.rint(
        generated.astype(np.float32) * alpha[..., None]
        + background.astype(np.float32) * (1.0 - alpha[..., None])
    ).astype(np.uint8)
    flowers = flower_mask.astype(bool)
    result[flowers] = source[flowers]
    not_robot_halo = np.logical_not(
        cv2.dilate(
            core.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        )
        > 0
    )
    not_flowers = np.logical_not(flowers)
    halo_region = np.logical_and(person, not_robot_halo)
    halo_region = np.logical_and(halo_region, not_flowers)
    halo_mae = float(
        np.mean(
            np.abs(
                result[halo_region].astype(np.float32)
                - background[halo_region].astype(np.float32)
            )
        )
    ) if np.count_nonzero(halo_region) else 0.0
    baseline_halo_mae = float(
        np.mean(
            np.abs(
                generated[halo_region].astype(np.float32)
                - background[halo_region].astype(np.float32)
            )
        )
    ) if np.count_nonzero(halo_region) else 0.0
    halo_remaining_fraction = (
        halo_mae / baseline_halo_mae if baseline_halo_mae > 1e-6 else 0.0
    )
    robot_evaluation = np.logical_and(core, not_flowers)
    core_exact = float(
        np.mean(
            np.all(
                result[robot_evaluation] == generated[robot_evaluation],
                axis=1,
            )
        )
    ) if np.count_nonzero(robot_evaluation) else 0.0
    flowers_exact = float(
        np.mean(np.all(result[flowers] == source[flowers], axis=1))
    ) if np.count_nonzero(flowers) else 1.0
    edit_support = np.logical_or(person_alpha > 0.0, alpha > 0.0)
    edit_support = np.logical_or(edit_support, flowers)
    exterior = np.logical_not(edit_support)
    exterior_exact = float(
        np.mean(np.all(result[exterior] == source[exterior], axis=1))
    ) if np.count_nonzero(exterior) else 1.0
    if source_human_residual_mask is None:
        human_residual = np.zeros(core.shape, dtype=bool)
    else:
        human_residual = np.logical_and(
            source_human_residual_mask.astype(bool), not_flowers
        )
    human_residual_retained = float(
        np.mean(np.all(result[human_residual] == source[human_residual], axis=1))
    ) if np.count_nonzero(human_residual) else 0.0
    return result, {
        "halo_background_mae": halo_mae,
        "baseline_halo_background_mae": baseline_halo_mae,
        "halo_remaining_fraction": halo_remaining_fraction,
        "robot_core_exact_fraction": core_exact,
        "flower_exact_fraction": flowers_exact,
        "protected_exterior_exact_fraction": exterior_exact,
        "source_human_residual_retained_fraction": human_residual_retained,
        "robot_mask_fraction": float(np.mean(core)),
        "source_person_mask_fraction": float(np.mean(person)),
    }


def _writer(
    ffmpeg: Path,
    output: Path,
    width: int,
    height: int,
    fps: float,
) -> Any:
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _track_segmentation_masks(
    cv2: Any,
    np: Any,
    mp: Any,
    *,
    video: Path,
    model: Path,
    fps: float,
    threshold: float,
) -> tuple[list[Any | None], dict[str, object]]:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.20,
        min_pose_presence_confidence=0.20,
        min_tracking_confidence=0.20,
        output_segmentation_masks=True,
    )
    capture = cv2.VideoCapture(str(video))
    masks: list[Any | None] = []
    detected = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect_for_video(
                image, round(index * 1000.0 / fps)
            )
            if result.segmentation_masks:
                values = result.segmentation_masks[0].numpy_view()
                if values.ndim == 3:
                    values = values[..., 0]
                mask = _largest_component(cv2, np, values >= threshold)
                area = float(np.mean(mask))
                if 0.01 <= area <= 0.25:
                    masks.append(mask)
                    detected += 1
                else:
                    masks.append(None)
            else:
                masks.append(None)
            index += 1
    capture.release()
    return masks, {
        "decoded_frames": len(masks),
        "detected_frames": detected,
        "missing_frames": len(masks) - detected,
        "threshold": threshold,
    }


def _nearest_seed_indices(
    masks: list[Any | None],
    *,
    spacing: int,
    maximum_distance: int,
) -> list[int]:
    available = [index for index, mask in enumerate(masks) if mask is not None]
    if not available:
        raise RuntimeError("MediaPipe did not produce any robot segmentation seeds")
    targets = list(range(0, len(masks), spacing))
    if targets[-1] != len(masks) - 1:
        targets.append(len(masks) - 1)
    selected = []
    for target in targets:
        nearest = min(available, key=lambda index: abs(index - target))
        if abs(nearest - target) <= maximum_distance:
            selected.append(nearest)
    selected.append(available[0])
    selected.append(available[-1])
    return sorted(set(selected))


def _track_pose_prompt_points(
    cv2: Any,
    np: Any,
    mp: Any,
    *,
    video: Path,
    model: Path,
    fps: float,
    requested_indices: list[int],
) -> tuple[dict[int, Any], dict[str, object]]:
    """Read source wrists for bounded no-box SAM2 robot prompts."""

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.30,
        min_pose_presence_confidence=0.30,
        min_tracking_confidence=0.30,
        output_segmentation_masks=False,
    )
    requested = set(requested_indices)
    points: dict[int, Any] = {}
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded in requested:
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                result = landmarker.detect_for_video(
                    image, round(decoded * 1000.0 / fps)
                )
                if not result.pose_landmarks:
                    raise RuntimeError(f"source pose missing on seed frame {decoded}")
                landmarks = result.pose_landmarks[0]
                points[decoded] = np.asarray(
                    [
                        (
                            landmarks[index].x * frame.shape[1],
                            landmarks[index].y * frame.shape[0],
                        )
                        for index in (15, 16)
                    ],
                    dtype=np.float32,
                )
            decoded += 1
    capture.release()
    missing = sorted(requested - set(points))
    if missing:
        raise RuntimeError(f"source wrist prompts missing for frames {missing}")
    return points, {
        "method": "MediaPipe source-pose wrist landmarks 15 and 16",
        "coordinate_frame": "camera:source_pixels aligned to H3 output pixels",
        "decoded_frames": decoded,
        "prompt_frames": len(points),
    }


def _fill_missing_masks(
    np: Any,
    masks: list[Any | None],
) -> tuple[list[Any], int]:
    valid = [index for index, mask in enumerate(masks) if mask is not None]
    if not valid:
        raise RuntimeError("no valid source-person segmentation masks")
    filled = []
    for index, mask in enumerate(masks):
        if mask is not None:
            filled.append(mask)
            continue
        nearest = min(valid, key=lambda value: abs(value - index))
        filled.append(masks[nearest])
    return [np.asarray(mask, dtype=bool) for mask in filled], len(masks) - len(valid)


def _sam2_track_masks(
    cv2: Any,
    np: Any,
    torch: Any,
    *,
    predictor: Any,
    frames_dir: Path,
    seed_masks: list[Any | None],
    seed_indices: list[int],
) -> list[Any]:
    tracked: list[Any | None] = [None] * len(seed_masks)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )
        for index in seed_indices:
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=index,
                obj_id=1,
                mask=seed_masks[index],
            )
        for frame_index, object_ids, logits in predictor.propagate_in_video(state):
            object_index = list(object_ids).index(1)
            mask = logits[object_index, 0].detach().cpu().numpy() > 0.0
            tracked[frame_index] = _largest_component(cv2, np, mask)
    if any(mask is None for mask in tracked):
        missing = [index for index, mask in enumerate(tracked) if mask is None]
        raise RuntimeError(f"SAM2 did not return all frames: {missing[:10]}")
    del state
    return [np.asarray(mask, dtype=bool) for mask in tracked]


def _sam2_box_point_seeds(
    cv2: Any,
    np: Any,
    torch: Any,
    *,
    predictor: Any,
    frames_dir: Path,
    seed_indices: list[int],
    safety_mask: Any,
    use_box: bool = True,
    pose_prompt_points: dict[int, Any] | None = None,
) -> tuple[list[Any | None], dict[str, object]]:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ys, xs = np.where(safety_mask.astype(bool))
    if not len(xs):
        raise RuntimeError("source safety mask is empty")
    height, width = safety_mask.shape
    left = max(0, int(xs.min()) - 16)
    top = max(0, int(ys.min()) - 18)
    right = min(width - 1, int(xs.max()) + 18)
    bottom = min(height - 1, int(ys.max()) + 16)
    box = np.asarray([left, top, right, bottom], dtype=np.float32)
    box_width = right - left
    box_height = bottom - top
    points = np.asarray(
        [
            [left + 0.583 * box_width, top + 0.242 * box_height],
            [left + 0.535 * box_width, top + 0.459 * box_height],
            [left + 0.758 * box_width, top + 0.434 * box_height],
            [left + 0.583 * box_width, top + 0.689 * box_height],
        ],
        dtype=np.float32,
    )
    image_predictor = SAM2ImagePredictor(predictor)
    masks: list[Any | None] = [None] * 660
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in seed_indices:
            frame = cv2.imread(str(frames_dir / f"{index:05d}.jpg"))
            if frame is None:
                raise RuntimeError(f"cannot decode extracted frame {index}")
            frame_points = points
            if pose_prompt_points is not None:
                frame_points = np.concatenate((points, pose_prompt_points[index]))
            labels = np.ones(len(frame_points), dtype=np.int32)
            image_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            candidates, scores, _ = image_predictor.predict(
                point_coords=frame_points,
                point_labels=labels,
                box=box if use_box else None,
                multimask_output=True,
            )
            candidate_rows = []
            for candidate_index, (candidate, score) in enumerate(
                zip(candidates, scores)
            ):
                candidate = candidate.astype(bool)
                fraction = float(np.mean(candidate))
                contains_points = float(
                    np.mean(
                        [
                            candidate[
                                min(height - 1, max(0, round(y))),
                                min(width - 1, max(0, round(x))),
                            ]
                            for x, y in frame_points
                        ]
                    )
                )
                valid_area = (0.05 if use_box else 0.035) <= fraction <= 0.22
                rank = (
                    int(valid_area),
                    contains_points,
                    float(score) - abs(fraction - 0.11),
                )
                candidate_rows.append(
                    {
                        "candidate": candidate_index,
                        "mask": candidate,
                        "score": float(score),
                        "fraction": fraction,
                        "contains_points_fraction": contains_points,
                        "valid_area": valid_area,
                        "rank": rank,
                    }
                )
            chosen = max(candidate_rows, key=lambda item: item["rank"])
            if not chosen["valid_area"]:
                raise RuntimeError(
                    f"no valid SAM2 box-point robot seed on frame {index}: "
                    f"{[(row['fraction'], row['score']) for row in candidate_rows]}"
                )
            masks[index] = chosen["mask"]
            rows.append(
                {
                    "frame": index,
                    "chosen_candidate": chosen["candidate"],
                    "chosen_fraction": chosen["fraction"],
                    "chosen_score": chosen["score"],
                    "positive_points_xy": frame_points.tolist(),
                    "candidates": [
                        {
                            key: value
                            for key, value in row.items()
                            if key not in {"mask", "rank"}
                        }
                        for row in candidate_rows
                    ],
                }
            )
    return masks, {
        "method": (
            "SAM2 image predictor with four reviewed robot-positive points on "
            "head, torso, right shoulder, and lower torso; "
            + ("with safety box" if use_box else "without box")
            + ("; plus source-pose wrist points" if pose_prompt_points else "")
        ),
        "box_xyxy": box.tolist() if use_box else None,
        "positive_points_xy": points.tolist(),
        "frames": rows,
    }


def _clean_plate(
    cv2: Any,
    np: Any,
    *,
    source_video: Path,
    person_masks: list[Any],
    sample_stride: int,
) -> tuple[Any, dict[str, object]]:
    capture = cv2.VideoCapture(str(source_video))
    samples = []
    sample_masks = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % sample_stride == 0 or index == len(person_masks) - 1:
            samples.append(frame.astype(np.float32))
            sample_masks.append(person_masks[index])
        index += 1
    capture.release()
    stack = np.stack(samples, axis=0)
    blocked = np.stack(sample_masks, axis=0)
    stack[blocked] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        plate = np.nanmedian(stack, axis=0)
    missing = np.any(~np.isfinite(plate), axis=2)
    plate = np.nan_to_num(plate, nan=0.0).astype(np.uint8)
    if np.count_nonzero(missing):
        plate = cv2.inpaint(
            plate,
            missing.astype(np.uint8) * 255,
            9.0,
            cv2.INPAINT_TELEA,
        )
    return plate, {
        "sample_stride": sample_stride,
        "sample_count": len(samples),
        "inpainted_fraction": float(np.mean(missing)),
        "method": "masked temporal median plus Telea fill for never-visible pixels",
    }


def _load_packed_masks(
    np: Any,
    path: Path,
    *,
    expected_frames: int,
    expected_height: int,
    expected_width: int,
) -> list[Any]:
    payload = np.load(path)
    height = int(payload["height"])
    width = int(payload["width"])
    packed = payload["packed"]
    if (height, width) != (expected_height, expected_width):
        raise RuntimeError("precomputed mask geometry does not match the video")
    if packed.shape[0] != expected_frames:
        raise RuntimeError("precomputed mask frame count does not match the video")
    unpacked = np.unpackbits(packed, axis=1, bitorder="little")
    return [
        row[: height * width].reshape(height, width).astype(bool)
        for row in unpacked
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-video", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--source-stitch-manifest", type=Path, required=True)
    parser.add_argument("--source-safety-mask", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--sam2-repo", type=Path)
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument("--sam2-config", default="sam2_hiera_l.yaml")
    parser.add_argument("--precomputed-source-person-masks", type=Path)
    parser.add_argument("--precomputed-clean-plate", type=Path)
    parser.add_argument("--precomputed-robot-masks", type=Path)
    parser.add_argument("--force-robot-object-presence", action="store_true")
    parser.add_argument(
        "--fill-source-person-mask-holes",
        action="store_true",
        help=(
            "Fill enclosed gaps in source-person masks so apron and hand pixels "
            "cannot leak through as background."
        ),
    )
    parser.add_argument(
        "--union-source-safety-mask",
        action="store_true",
        help=(
            "Union the reviewed source-person safety corridor into every source "
            "mask to remove disconnected apron and hand residuals."
        ),
    )
    parser.add_argument(
        "--robot-seed-method",
        choices=(
            "mediapipe",
            "sam2_box_points",
            "sam2_no_box_points",
            "sam2_no_box_pose_wrists",
        ),
        default="mediapipe",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=49152)
    parser.add_argument("--seed-spacing", type=int, default=72)
    parser.add_argument("--seed-maximum-distance", type=int, default=36)
    parser.add_argument("--sample-stride", type=int, default=12)
    parser.add_argument(
        "--flower-restoration-mode",
        choices=("source", "none"),
        default="source",
        help=(
            "Restore source flower pixels, or disable restoration to prevent "
            "source hands adjacent to flowers from leaking into the result."
        ),
    )
    parser.add_argument(
        "--person-background-feather-sigma",
        type=float,
        default=0.0,
        help="Outward clean-plate blend width in pixels; person pixels remain exact.",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--human-review", choices=("pending", "passed", "failed"), default="pending")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    use_gpu = args.precomputed_robot_masks is None
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"experiment already exists: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_source = output_dir / "provenance" / "execution-sources" / Path(__file__).name
    frozen_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), frozen_source)
    paths = {
        "generated_video": args.generated_video.expanduser().resolve(),
        "source_video": args.source_video.expanduser().resolve(),
        "source_stitch_manifest": args.source_stitch_manifest.expanduser().resolve(),
        "source_safety_mask": args.source_safety_mask.expanduser().resolve(),
        "pose_model": args.pose_model.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    if use_gpu:
        if args.sam2_repo is None or args.sam2_checkpoint is None or args.gpu is None:
            raise ValueError(
                "SAM2 repository, checkpoint, and GPU are required without "
                "precomputed robot masks"
            )
        paths["sam2_repo"] = args.sam2_repo.expanduser().resolve()
        paths["sam2_checkpoint"] = args.sam2_checkpoint.expanduser().resolve()
    else:
        paths["precomputed_robot_masks"] = (
            args.precomputed_robot_masks.expanduser().resolve()
        )
    if bool(args.precomputed_source_person_masks) != bool(
        args.precomputed_clean_plate
    ):
        raise ValueError(
            "precomputed source-person masks and clean plate must be supplied together"
        )
    if args.precomputed_source_person_masks:
        paths["precomputed_source_person_masks"] = (
            args.precomputed_source_person_masks.expanduser().resolve()
        )
        paths["precomputed_clean_plate"] = (
            args.precomputed_clean_plate.expanduser().resolve()
        )
    if not use_gpu and not args.precomputed_source_person_masks:
        raise ValueError(
            "CPU recomposition requires precomputed source-person masks and clean plate"
        )
    for name, path in paths.items():
        expected = path.is_dir() if name == "sam2_repo" else path.is_file()
        if not expected:
            raise ValueError(f"{name} is missing: {path}")
    command = [sys.executable, *sys.argv]
    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "sam2_robot_matte_clean_plate_h3_shadow_removal",
        "status": "preflight_started",
        "honest_status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "command_shell": shlex.join(command),
        "seed": args.seed,
        "execution_source": {
            "path": str(frozen_source),
            "sha256": file_sha256(frozen_source),
        },
    }
    _write_json(manifest_path, record)

    import cv2
    import numpy as np

    mp = None
    if use_gpu:
        import mediapipe as mp

    generated_info = _video_info(cv2, paths["generated_video"])
    source_info = _video_info(cv2, paths["source_video"])
    if generated_info != source_info:
        raise RuntimeError("generated and source videos do not have identical geometry/timing")
    if int(generated_info["decoded_frames"]) != 660:
        raise RuntimeError("expected the full 660-frame flower video")
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    sam2_commit = None
    if use_gpu:
        gpus, inventory_raw, processes_raw = query_gpus()
        selected = next(
            (item for item in gpus if item.physical_index == args.gpu), None
        )
        if selected is None:
            raise RuntimeError(f"physical GPU {args.gpu} is unavailable")
        if selected.free_mib < args.minimum_free_gpu_mib:
            raise RuntimeError(
                f"GPU {args.gpu} has {selected.free_mib} MiB free; "
                f"need {args.minimum_free_gpu_mib} MiB"
            )
        sam2_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths["sam2_repo"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if sam2_commit != SAM2_COMMIT:
            raise RuntimeError(
                f"SAM2 commit is {sam2_commit}, expected {SAM2_COMMIT}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        sys.path.insert(0, str(paths["sam2_repo"]))
        device_record: dict[str, object] = {
            "used": True,
            "physical_index": selected.physical_index,
            "name": selected.name,
            "free_mib": selected.free_mib,
            "total_mib": selected.total_mib,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "inventory_raw": inventory_raw,
            "processes_raw": processes_raw,
        }
    else:
        device_record = {
            "used": False,
            "mode": "CPU recomposition from pinned precomputed masks",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    record.update(
        {
            "status": "running",
            "inputs": {
                name: {
                    "path": str(path),
                    "sha256": file_sha256(path) if path.is_file() else None,
                }
                for name, path in paths.items()
            },
            "video_info": generated_info,
            "gpu": device_record,
            "sam2_commit": sam2_commit,
            "git": _git_state(PROJECT_ROOT),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        }
    )
    _write_json(manifest_path, record)

    fps = float(generated_info["fps"])
    height = int(generated_info["height"])
    width = int(generated_info["width"])
    robot_seeds: list[Any] = []
    seed_indices: list[int] = []
    robot_seed_tracking: dict[str, object] = {
        "method": "not run; reused precomputed robot masks"
    }
    source_person_raw: list[Any] = []
    source_person_filled: list[Any] = []
    source_seed_indices: list[int] = []
    source_mask_fills: list[int] = []
    source_person_tracking: dict[str, object] = {
        "method": "not run; reused precomputed source-person masks"
    }
    aligned_safety = None
    if use_gpu:
        assert mp is not None
        robot_seeds, robot_seed_tracking = _track_segmentation_masks(
            cv2,
            np,
            mp,
            video=paths["generated_video"],
            model=paths["pose_model"],
            fps=fps,
            threshold=0.50,
        )
        seed_indices = _nearest_seed_indices(
            robot_seeds,
            spacing=args.seed_spacing,
            maximum_distance=args.seed_maximum_distance,
        )
    if use_gpu or args.union_source_safety_mask:
        safety_raw = cv2.imread(
            str(paths["source_safety_mask"]), cv2.IMREAD_GRAYSCALE
        )
        if safety_raw is None:
            raise RuntimeError("cannot decode source safety mask")
        aligned_safety = _align_mask(cv2, safety_raw, width, height) >= 127
    if not args.precomputed_source_person_masks:
        assert use_gpu and mp is not None
        source_person_raw, source_person_tracking = _track_segmentation_masks(
            cv2,
            np,
            mp,
            video=paths["source_video"],
            model=paths["pose_model"],
            fps=fps,
            threshold=0.20,
        )
        source_person_filled, source_mask_fills = _fill_missing_masks(
            np, source_person_raw
        )
        source_seed_indices = _nearest_seed_indices(
            source_person_raw,
            spacing=args.seed_spacing,
            maximum_distance=args.seed_maximum_distance,
        )
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for index in seed_indices:
        cv2.imwrite(
            str(assets / f"robot-seed-{index:04d}.png"),
            robot_seeds[index].astype(np.uint8) * 255,
        )
    for index in source_seed_indices:
        cv2.imwrite(
            str(assets / f"source-person-seed-{index:04d}.png"),
            source_person_raw[index].astype(np.uint8) * 255,
        )

    frames_dir = output_dir / "input" / "frames"
    source_frames_dir = output_dir / "input" / "source-frames"
    if use_gpu:
        frames_dir.mkdir(parents=True, exist_ok=True)
    if not args.precomputed_source_person_masks:
        source_frames_dir.mkdir(parents=True, exist_ok=True)
    extract_commands = []
    extraction_inputs = []
    if use_gpu:
        extraction_inputs.append((paths["generated_video"], frames_dir))
    if use_gpu and not args.precomputed_source_person_masks:
        extraction_inputs.append((paths["source_video"], source_frames_dir))
    for video, directory in extraction_inputs:
        command_item = [
            str(paths["ffmpeg"]),
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(directory / "%05d.jpg"),
        ]
        subprocess.run(command_item, check=True)
        extract_commands.append(command_item)

    if use_gpu:
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        torch.manual_seed(args.seed)
        predictor = build_sam2_video_predictor(
            args.sam2_config,
            str(paths["sam2_checkpoint"]),
            device="cuda",
        )
        default_pred_obj_scores: bool | None = bool(predictor.pred_obj_scores)
        robot_seed_generation: dict[str, object] = {
            "method": "MediaPipe pose segmentation",
        }
        pose_prompt_tracking: dict[str, object] | None = None
        if args.robot_seed_method in {
            "sam2_box_points",
            "sam2_no_box_points",
            "sam2_no_box_pose_wrists",
        }:
            assert aligned_safety is not None
            pose_prompt_points = None
            if args.robot_seed_method == "sam2_no_box_pose_wrists":
                assert mp is not None
                pose_prompt_points, pose_prompt_tracking = _track_pose_prompt_points(
                    cv2,
                    np,
                    mp,
                    video=paths["source_video"],
                    model=paths["pose_model"],
                    fps=fps,
                    requested_indices=seed_indices,
                )
            robot_seeds, robot_seed_generation = _sam2_box_point_seeds(
                cv2,
                np,
                torch,
                predictor=predictor,
                frames_dir=frames_dir,
                seed_indices=seed_indices,
                safety_mask=aligned_safety,
                use_box=args.robot_seed_method == "sam2_box_points",
                pose_prompt_points=pose_prompt_points,
            )
            for index in seed_indices:
                cv2.imwrite(
                    str(assets / f"sam2-box-robot-seed-{index:04d}.png"),
                    robot_seeds[index].astype(np.uint8) * 255,
                )
        if args.force_robot_object_presence:
            predictor.pred_obj_scores = False
        robot_masks = _sam2_track_masks(
            cv2,
            np,
            torch,
            predictor=predictor,
            frames_dir=frames_dir,
            seed_masks=robot_seeds,
            seed_indices=seed_indices,
        )
        torch.cuda.empty_cache()
    else:
        default_pred_obj_scores = None
        pose_prompt_tracking = None
        robot_masks = _load_packed_masks(
            np,
            paths["precomputed_robot_masks"],
            expected_frames=660,
            expected_height=height,
            expected_width=width,
        )
        robot_seed_generation = {
            "method": "reused pinned SAM2 box-point robot masks",
            "source": str(paths["precomputed_robot_masks"]),
        }
    if args.precomputed_source_person_masks:
        source_person = _load_packed_masks(
            np,
            paths["precomputed_source_person_masks"],
            expected_frames=660,
            expected_height=height,
            expected_width=width,
        )
        source_sam_masks = source_person
        if args.fill_source_person_mask_holes:
            source_person = [
                _fill_mask_holes(cv2, np, mask) for mask in source_person
            ]
        if args.union_source_safety_mask:
            assert aligned_safety is not None
            source_person = [
                np.logical_or(mask, aligned_safety) for mask in source_person
            ]
        skin_candidate_pixels = -1
        plate = cv2.imread(str(paths["precomputed_clean_plate"]), cv2.IMREAD_COLOR)
        if plate is None or plate.shape[:2] != (height, width):
            raise RuntimeError("cannot decode the precomputed clean plate")
        plate_record = {
            "method": "reused pinned precomputed clean plate",
            "source_masks": str(paths["precomputed_source_person_masks"]),
            "source_clean_plate": str(paths["precomputed_clean_plate"]),
        }
    else:
        assert use_gpu
        assert aligned_safety is not None
        predictor.pred_obj_scores = bool(default_pred_obj_scores)
        source_sam_masks = _sam2_track_masks(
            cv2,
            np,
            torch,
            predictor=predictor,
            frames_dir=source_frames_dir,
            seed_masks=source_person_raw,
            seed_indices=source_seed_indices,
        )
        safety = aligned_safety
        source_capture = cv2.VideoCapture(str(paths["source_video"]))
        source_person = []
        skin_candidate_pixels = 0
        source_person_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (13, 13)
        )
        skin_corridor_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (41, 41)
        )
        for frame_index in range(660):
            ok, source_frame = source_capture.read()
            if not ok:
                raise RuntimeError(f"cannot decode source frame {frame_index}")
            combined = (
                source_sam_masks[frame_index]
                | source_person_filled[frame_index]
            )
            corridor = (
                cv2.dilate(
                    combined.astype(np.uint8) * 255,
                    skin_corridor_kernel,
                )
                > 0
            ) & safety
            skin = _skin_like_mask(cv2, np, source_frame) & corridor
            skin_candidate_pixels += int(np.count_nonzero(skin & ~combined))
            combined |= skin
            source_person.append(
                cv2.dilate(
                    combined.astype(np.uint8) * 255,
                    source_person_kernel,
                )
                > 0
            )
        source_capture.release()
        plate, plate_record = _clean_plate(
            cv2,
            np,
            source_video=paths["source_video"],
            person_masks=source_person,
            sample_stride=args.sample_stride,
        )
    cv2.imwrite(str(assets / "clean-plate.png"), plate)
    packed_robot = np.stack(
        [np.packbits(mask.reshape(-1), bitorder="little") for mask in robot_masks]
    )
    np.savez_compressed(
        assets / "sam2-robot-masks-packed.npz",
        packed=packed_robot,
        height=height,
        width=width,
        bitorder="little",
    )
    packed_source = np.stack(
        [np.packbits(mask.reshape(-1), bitorder="little") for mask in source_person]
    )
    np.savez_compressed(
        assets / "sam2-source-person-masks-packed.npz",
        packed=packed_source,
        height=height,
        width=width,
        bitorder="little",
    )
    if use_gpu:
        del predictor
        torch.cuda.empty_cache()

    robot_mask_fractions = [float(np.mean(mask)) for mask in robot_masks]
    robot_tracking_summary = {
        "minimum_fraction": min(robot_mask_fractions),
        "median_fraction": float(np.median(robot_mask_fractions)),
        "maximum_fraction": max(robot_mask_fractions),
        "empty_frames": [
            index for index, fraction in enumerate(robot_mask_fractions)
            if fraction == 0.0
        ],
        "force_object_presence": args.force_robot_object_presence,
        "default_pred_obj_scores": default_pred_obj_scores,
    }
    record.update({"robot_tracking_summary": robot_tracking_summary})
    record.update({"robot_seed_generation": robot_seed_generation})
    _write_json(manifest_path, record)

    rounds = []
    round_settings = (6, 3, 1)
    for round_index, dilation in enumerate(round_settings):
        output = output_dir / "rounds" / f"round-{round_index:02d}-dilate-{dilation}.mp4"
        writer = _writer(paths["ffmpeg"], output, width, height, fps)
        source_capture = cv2.VideoCapture(str(paths["source_video"]))
        generated_capture = cv2.VideoCapture(str(paths["generated_video"]))
        metric_rows = []
        frame_index = 0
        try:
            assert writer.stdin is not None
            while True:
                source_ok, source_frame = source_capture.read()
                generated_ok, generated_frame = generated_capture.read()
                if not source_ok or not generated_ok:
                    break
                if args.flower_restoration_mode == "source":
                    flowers = _flower_mask(
                        cv2,
                        np,
                        source_frame,
                        dilation=2,
                        exclude_skin_like=True,
                    )
                else:
                    flowers = np.zeros((height, width), dtype=bool)
                human_residual = (
                    _skin_like_mask(cv2, np, source_frame)
                    & source_person[frame_index]
                    & (
                        np.mean(
                            np.abs(
                                source_frame.astype(np.float32)
                                - plate.astype(np.float32)
                            ),
                            axis=2,
                        )
                        >= 12.0
                    )
                )
                candidate, metrics = compose_shadow_free_frame(
                    cv2,
                    np,
                    source=source_frame,
                    generated=generated_frame,
                    clean_plate=plate,
                    robot_mask=robot_masks[frame_index],
                    source_person_mask=source_person[frame_index],
                    flower_mask=flowers,
                    dilation_pixels=dilation,
                    source_human_residual_mask=human_residual,
                    person_background_feather_sigma=(
                        args.person_background_feather_sigma
                    ),
                )
                writer.stdin.write(candidate.tobytes())
                metric_rows.append(metrics)
                frame_index += 1
        finally:
            source_capture.release()
            generated_capture.release()
            if writer.stdin is not None:
                writer.stdin.close()
            if writer.wait():
                raise RuntimeError(f"FFmpeg failed for {output}")
        if frame_index != 660:
            raise RuntimeError(f"round {round_index} wrote {frame_index} frames")
        means = {
            key: float(np.mean([row[key] for row in metric_rows]))
            for key in metric_rows[0]
        }
        rounds.append(
            {
                "round": round_index,
                "dilation_pixels": dilation,
                "metrics": means,
                "output": str(output),
                "sha256": file_sha256(output),
            }
        )
    record.update({"rounds": rounds})
    _write_json(manifest_path, record)
    eligible = [
        item
        for item in rounds
        if item["metrics"]["robot_core_exact_fraction"] >= 0.999
        and item["metrics"]["flower_exact_fraction"] >= 0.999
        and robot_tracking_summary["minimum_fraction"] >= 0.005
        and robot_tracking_summary["maximum_fraction"] <= 0.25
    ]
    if not eligible:
        raise RuntimeError("no shadow-removal round preserved robot core and flowers")
    selected = min(
        eligible,
        key=lambda item: (
            item["metrics"]["halo_background_mae"],
            item["dilation_pixels"],
        ),
    )
    final = output_dir / "minimax-h3-epl-full-27s-shadow-removed.mp4"
    shutil.copy2(selected["output"], final)
    final_info = _video_info(cv2, final)
    comparison = output_dir / "human-vs-h3-epl-shadow-removed.mp4"
    subprocess.run(
        [
            str(paths["ffmpeg"]),
            "-y",
            "-v",
            "error",
            "-i",
            str(paths["source_video"]),
            "-i",
            str(final),
            "-filter_complex",
            "[0:v][1:v]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "15",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(comparison),
        ],
        check=True,
    )
    dense_review = output_dir / "dense-review.jpg"
    subprocess.run(
        [
            str(paths["ffmpeg"]),
            "-y",
            "-v",
            "error",
            "-i",
            str(final),
            "-vf",
            "fps=28/27.5,scale=416:-2,tile=4x7:padding=4:margin=4:color=black",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(dense_review),
        ],
        check=True,
    )
    review_passed = args.human_review == "passed"
    automatic = {
        "full_clip_decoded": int(final_info["decoded_frames"]) == 660,
        "robot_core_preserved": selected["metrics"]["robot_core_exact_fraction"] >= 0.999,
        "flower_policy_applied": (
            args.flower_restoration_mode == "none"
            or selected["metrics"]["flower_exact_fraction"] >= 0.999
        ),
        "source_flower_restoration_disabled": (
            args.flower_restoration_mode == "none"
        ),
        "protected_exterior_preserved": selected["metrics"]["protected_exterior_exact_fraction"] >= 0.999,
        "halo_background_mae_passed": selected["metrics"]["halo_background_mae"] <= 1.0,
        "halo_reduction_passed": (
            selected["metrics"]["baseline_halo_background_mae"] <= 1.0
            or selected["metrics"]["halo_remaining_fraction"] <= 0.25
        ),
        "source_human_residual_passed": (
            selected["metrics"]["source_human_residual_retained_fraction"]
            <= 0.01
        ),
        "sam2_full_coverage": len(robot_masks) == 660,
        "sam2_source_person_full_coverage": len(source_sam_masks) == 660,
        "robot_track_area_valid": (
            robot_tracking_summary["minimum_fraction"] >= 0.005
            and robot_tracking_summary["maximum_fraction"] <= 0.25
        ),
    }
    accepted = all(automatic.values()) and review_passed
    status = "accepted" if accepted else "review_required" if args.human_review == "pending" else "rejected"
    record.update(
        {
            "status": status,
            "honest_status": "WORKING" if accepted else "PARTIAL",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "packages": {
                name: _package_version(name)
                for name in ("numpy", "opencv-contrib-python", "mediapipe", "torch")
            },
            "config": {
                "used_gpu": use_gpu,
                "sam2_config": args.sam2_config,
                "seed_spacing": args.seed_spacing,
                "seed_maximum_distance": args.seed_maximum_distance,
                "sample_stride": args.sample_stride,
                "force_robot_object_presence": args.force_robot_object_presence,
                "robot_seed_method": args.robot_seed_method,
                "fill_source_person_mask_holes": (
                    args.fill_source_person_mask_holes
                ),
                "union_source_safety_mask": args.union_source_safety_mask,
                "flower_restoration_mode": args.flower_restoration_mode,
                "person_background_feather_sigma": (
                    args.person_background_feather_sigma
                ),
                "round_dilation_pixels": list(round_settings),
                "coordinate_frame": "camera:H3_output_pixels",
            },
            "robot_seed_tracking": robot_seed_tracking,
            "robot_pose_prompt_tracking": pose_prompt_tracking,
            "robot_seed_indices": seed_indices,
            "source_person_tracking": source_person_tracking,
            "source_person_seed_indices": source_seed_indices,
            "source_person_nearest_fills": source_mask_fills,
            "source_skin_residual_candidate_pixels": skin_candidate_pixels,
            "clean_plate": {
                **plate_record,
                "path": str(assets / "clean-plate.png"),
                "sha256": file_sha256(assets / "clean-plate.png"),
            },
            "frame_extraction_commands": extract_commands,
            "sam2_masks": {
                "robot": {
                    "path": str(assets / "sam2-robot-masks-packed.npz"),
                    "sha256": file_sha256(assets / "sam2-robot-masks-packed.npz"),
                    "frame_count": len(robot_masks),
                },
                "source_person": {
                    "path": str(assets / "sam2-source-person-masks-packed.npz"),
                    "sha256": file_sha256(assets / "sam2-source-person-masks-packed.npz"),
                    "frame_count": len(source_sam_masks),
                    "postprocess": (
                        "union MediaPipe; add skin-like residuals inside safety "
                        "corridor; dilate 6 pixels"
                        + (
                            "; fill enclosed mask holes"
                            if args.fill_source_person_mask_holes
                            else ""
                        )
                        + (
                            "; union reviewed source-person safety mask"
                            if args.union_source_safety_mask
                            else ""
                        )
                    ),
                },
            },
            "rounds": rounds,
            "selected_round": selected["round"],
            "acceptance": {**automatic, "human_review": args.human_review},
            "outputs": {
                "video": str(final),
                "video_sha256": file_sha256(final),
                "video_info": final_info,
                "comparison": str(comparison),
                "comparison_sha256": file_sha256(comparison),
                "dense_review": str(dense_review),
            },
            "limitations": [
                "SAM2 tracks the generated robot silhouette; it is not a 3D geometry mask.",
                "Pixels never visible behind the source person require a synthesized clean plate.",
                "This repairs visual halo contamination only and does not improve physical task validity.",
                (
                    "Source flower-pixel restoration is disabled to prevent adjacent "
                    "human hands from leaking back into the composite."
                    if args.flower_restoration_mode == "none"
                    else "Source flower-pixel restoration can include adjacent foreground details."
                ),
            ],
        }
    )
    _write_json(manifest_path, record)
    print(
        json.dumps(
            {
                "status": status,
                "honest_status": record["honest_status"],
                "selected_round": selected["round"],
                "metrics": selected["metrics"],
                "output": str(final),
                "comparison": str(comparison),
                "acceptance": record["acceptance"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if accepted else 2


def _entrypoint() -> int:
    try:
        return main()
    except Exception as exc:
        try:
            args = _parser().parse_args()
            manifest_path = args.output_dir.expanduser().resolve() / "manifest.json"
            if manifest_path.is_file():
                record = json.loads(manifest_path.read_text())
                if record.get("status") in {"preflight_started", "running"}:
                    record.update(
                        {
                            "status": "failed",
                            "honest_status": "BLOCKED",
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    _write_json(manifest_path, record)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
