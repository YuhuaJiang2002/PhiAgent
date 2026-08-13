#!/usr/bin/env python3
"""Track complete task-flower instances while rejecting source human limbs."""

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


def _git_state() -> dict[str, object]:
    status = subprocess.run(
        ["git", "--no-pager", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=PROJECT_ROOT,
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


def _video_info(cv2: Any, path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    result: dict[str, int | float] = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    return result


def _align_mask(cv2: Any, mask: Any, width: int, height: int) -> Any:
    scale = max(width / mask.shape[1], height / mask.shape[0])
    resized = cv2.resize(
        mask,
        (round(mask.shape[1] * scale), round(mask.shape[0] * scale)),
        interpolation=cv2.INTER_NEAREST,
    )
    left = max(0, (resized.shape[1] - width) // 2)
    top = max(0, (resized.shape[0] - height) // 2)
    return resized[top : top + height, left : left + width]


def _load_packed_layer(np: Any, path: Path, key: str) -> Any:
    payload = np.load(path)
    height, width = int(payload["height"]), int(payload["width"])
    packed = payload[key]
    unpacked = np.unpackbits(packed, axis=1, bitorder="little")
    return unpacked[:, : height * width].reshape(len(packed), height, width).astype(bool)


def _strict_flower_seed(cv2: Any, np: Any, frame: Any) -> Any:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue >= 28) & (hue <= 91) & (saturation >= 67) & (value >= 28)
    pink = (hue >= 145) & (hue <= 179) & (saturation >= 105) & (value >= 55)
    yellow = (hue >= 12) & (hue <= 35) & (saturation >= 105) & (value >= 70)
    mask = (green | pink | yellow).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def _skin_like(cv2: Any, np: Any, frame: Any) -> Any:
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


def _sample_component_points(cv2: Any, np: Any, mask: Any, maximum: int) -> Any:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    components = sorted(
        range(1, count),
        key=lambda item: int(stats[item, cv2.CC_STAT_AREA]),
        reverse=True,
    )
    points = []
    for component in components[:maximum]:
        component_mask = (labels == component).astype(np.uint8)
        distance = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        points.append((float(x), float(y)))
    if not points:
        ys, xs = np.where(mask)
        if len(xs):
            points.append((float(np.median(xs)), float(np.median(ys))))
    return np.asarray(points, dtype=np.float32)


def _sample_negative_points(np: Any, mask: Any, maximum: int) -> Any:
    ys, xs = np.where(mask)
    if not len(xs):
        return np.empty((0, 2), dtype=np.float32)
    chosen = np.linspace(0, len(xs) - 1, min(maximum, len(xs)), dtype=np.int32)
    return np.stack((xs[chosen], ys[chosen]), axis=1).astype(np.float32)


def _seed_instance_mask(
    cv2: Any,
    np: Any,
    *,
    image_predictor: Any,
    frame: Any,
    hands: Any,
    arms: Any,
    safety: Any,
    interaction_radius: int,
    support_radius: int,
    box_padding: int,
) -> tuple[Any, dict[str, object]]:
    height, width = frame.shape[:2]
    strict = _strict_flower_seed(cv2, np, frame)
    interaction = cv2.dilate(
        hands.astype(np.uint8) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (interaction_radius * 2 + 1, interaction_radius * 2 + 1),
        ),
    ) > 0
    safety_neighborhood = cv2.dilate(
        safety.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)),
    ) > 0
    strict &= interaction & safety_neighborhood
    clustered = cv2.dilate(
        strict.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(clustered)
    skin = _skin_like(cv2, np, frame) & arms
    image_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    union = np.zeros((height, width), dtype=bool)
    rows = []
    for component in range(1, count):
        component_area = int(stats[component, cv2.CC_STAT_AREA])
        component_support = labels == component
        component_seed = strict & component_support
        seed_pixels = int(np.count_nonzero(component_seed))
        if seed_pixels < 8 or component_area > round(height * width * 0.12):
            continue
        ys, xs = np.where(component_seed)
        left = max(0, int(xs.min()) - box_padding)
        top = max(0, int(ys.min()) - box_padding)
        right = min(width - 1, int(xs.max()) + box_padding)
        bottom = min(height - 1, int(ys.max()) + box_padding)
        box = np.asarray([left, top, right, bottom], dtype=np.float32)
        positive = _sample_component_points(cv2, np, component_seed, 6)
        negative_region = np.zeros_like(skin)
        negative_region[top : bottom + 1, left : right + 1] = skin[
            top : bottom + 1, left : right + 1
        ]
        negative = _sample_negative_points(np, negative_region, 6)
        points = np.concatenate((positive, negative), axis=0)
        labels_points = np.concatenate(
            (
                np.ones(len(positive), dtype=np.int32),
                np.zeros(len(negative), dtype=np.int32),
            )
        )
        candidates, scores, _ = image_predictor.predict(
            point_coords=points,
            point_labels=labels_points,
            box=box,
            multimask_output=True,
        )
        candidate_rows = []
        for candidate_index, (candidate, sam_score) in enumerate(
            zip(candidates, scores)
        ):
            candidate = candidate.astype(bool)
            area = int(np.count_nonzero(candidate))
            recall = float(np.count_nonzero(candidate & component_seed) / seed_pixels)
            skin_fraction = float(np.count_nonzero(candidate & skin) / max(1, area))
            expansion = float(area / seed_pixels)
            plausible = (
                recall >= 0.55
                and area >= seed_pixels
                and area <= round(height * width * 0.045)
                and expansion <= 45.0
                and skin_fraction <= 0.20
            )
            rank = (
                int(plausible),
                recall - 2.5 * skin_fraction - max(0.0, expansion - 18.0) / 50.0,
                float(sam_score),
            )
            candidate_rows.append(
                {
                    "candidate": candidate_index,
                    "mask": candidate,
                    "sam_score": float(sam_score),
                    "area": area,
                    "seed_recall": recall,
                    "skin_fraction": skin_fraction,
                    "expansion": expansion,
                    "plausible": plausible,
                    "rank": rank,
                }
            )
        selected = max(candidate_rows, key=lambda item: item["rank"])
        if selected["plausible"]:
            union |= selected["mask"]
        rows.append(
            {
                "box_xyxy": box.tolist(),
                "seed_pixels": seed_pixels,
                "positive_points_xy": positive.tolist(),
                "negative_points_xy": negative.tolist(),
                "selected_candidate": selected["candidate"],
                "accepted": selected["plausible"],
                "candidates": [
                    {key: value for key, value in row.items() if key not in {"mask", "rank"}}
                    for row in candidate_rows
                ],
            }
        )
    support = cv2.dilate(
        strict.astype(np.uint8) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (support_radius * 2 + 1, support_radius * 2 + 1),
        ),
    ) > 0
    union &= support
    union &= ~(skin & (cv2.dilate(hands.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0))
    union = cv2.morphologyEx(
        union.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    return union, {
        "strict_seed_pixels": int(np.count_nonzero(strict)),
        "mask_pixels": int(np.count_nonzero(union)),
        "clusters_considered": len(rows),
        "clusters_accepted": sum(bool(row["accepted"]) for row in rows),
        "clusters": rows,
    }


def _largest_components(cv2: Any, np: Any, mask: Any, minimum: int) -> Any:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    result = np.zeros(mask.shape, dtype=bool)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= minimum:
            result |= labels == component
    return result


def _contact_sheet(cv2: Any, np: Any, rows: list[list[Any]], labels: list[str]) -> Any:
    rendered = []
    for frame_index, row in enumerate(rows):
        cells = []
        for label, cell in zip(labels, row):
            item = cell.copy()
            cv2.putText(
                item,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if label == "source":
                cv2.putText(
                    item,
                    f"sample {frame_index}",
                    (12, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cells.append(item)
        rendered.append(cv2.hconcat(cells))
    return cv2.vconcat(rendered)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--source-safety-mask", type=Path, required=True)
    parser.add_argument("--source-limb-masks", type=Path, required=True)
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", default="sam2_hiera_l.yaml")
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=60000)
    parser.add_argument("--seed-spacing", type=int, default=24)
    parser.add_argument("--interaction-radius", type=int, default=100)
    parser.add_argument("--support-radius", type=int, default=30)
    parser.add_argument("--box-padding", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"experiment already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    for relative in ("assets", "input/frames", "logs", "review", "provenance/execution-sources"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    frozen_source = output_dir / "provenance/execution-sources" / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), frozen_source)
    paths = {
        "source_video": args.source_video.expanduser().resolve(),
        "source_safety_mask": args.source_safety_mask.expanduser().resolve(),
        "source_limb_masks": args.source_limb_masks.expanduser().resolve(),
        "sam2_repo": args.sam2_repo.expanduser().resolve(),
        "sam2_checkpoint": args.sam2_checkpoint.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    if any(not path.exists() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.exists()]
        raise FileNotFoundError(f"missing inputs: {missing}")
    command = [sys.executable, *sys.argv]
    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "preflight_started",
        "honest_status": "PARTIAL",
        "scope": "source task-flower instance tracking with human-limb negatives",
        "command": command,
        "command_shell": shlex.join(command),
        "seed": args.seed,
        "execution_source": {"path": str(frozen_source), "sha256": file_sha256(frozen_source)},
    }
    _write_json(output_dir / "manifest.json", record)

    import cv2
    import numpy as np

    info = _video_info(cv2, paths["source_video"])
    if int(info["frames"]) != 660:
        raise RuntimeError(f"expected 660 frames, got {info['frames']}")
    height, width = int(info["height"]), int(info["width"])
    hands = _load_packed_layer(np, paths["source_limb_masks"], "hands_packed")
    arms = _load_packed_layer(np, paths["source_limb_masks"], "arms_packed")
    if hands.shape != (660, height, width) or arms.shape != hands.shape:
        raise RuntimeError("source limb mask geometry does not match source video")
    safety_raw = cv2.imread(str(paths["source_safety_mask"]), cv2.IMREAD_GRAYSCALE)
    if safety_raw is None:
        raise RuntimeError("cannot decode source safety mask")
    safety = _align_mask(cv2, safety_raw, width, height) >= 127

    gpus, inventory_raw, processes_raw = query_gpus()
    selected = next((item for item in gpus if item.physical_index == args.gpu), None)
    if selected is None:
        raise RuntimeError(f"physical GPU {args.gpu} is unavailable")
    if selected.free_mib < args.minimum_free_gpu_mib:
        raise RuntimeError(
            f"GPU {args.gpu} has {selected.free_mib} MiB free; need {args.minimum_free_gpu_mib} MiB"
        )
    sam2_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=paths["sam2_repo"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if sam2_commit != SAM2_COMMIT:
        raise RuntimeError(f"SAM2 commit is {sam2_commit}, expected {SAM2_COMMIT}")
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(paths["sam2_repo"]))
    record.update(
        {
            "status": "running",
            "inputs": {
                name: {"path": str(path), "sha256": file_sha256(path) if path.is_file() else None}
                for name, path in paths.items()
            },
            "video": info,
            "gpu": {
                "physical_index": selected.physical_index,
                "name": selected.name,
                "free_mib": selected.free_mib,
                "total_mib": selected.total_mib,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "inventory_raw": inventory_raw,
                "processes_raw": processes_raw,
            },
            "sam2_commit": sam2_commit,
            "git": _git_state(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        }
    )
    _write_json(output_dir / "manifest.json", record)

    frames_dir = output_dir / "input/frames"
    extract_command = [
        str(paths["ffmpeg"]), "-y", "-v", "error", "-i", str(paths["source_video"]),
        "-q:v", "2", "-start_number", "0", str(frames_dir / "%05d.jpg"),
    ]
    subprocess.run(extract_command, check=True)
    seed_indices = list(range(0, 660, args.seed_spacing))
    if seed_indices[-1] != 659:
        seed_indices.append(659)

    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.manual_seed(args.seed)
    predictor = build_sam2_video_predictor(
        args.sam2_config,
        str(paths["sam2_checkpoint"]),
        device="cuda",
    )
    image_predictor = SAM2ImagePredictor(predictor)
    seed_masks: list[Any | None] = [None] * 660
    seed_rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in seed_indices:
            frame = cv2.imread(str(frames_dir / f"{index:05d}.jpg"))
            if frame is None:
                raise RuntimeError(f"cannot decode extracted frame {index}")
            mask, row = _seed_instance_mask(
                cv2,
                np,
                image_predictor=image_predictor,
                frame=frame,
                hands=hands[index],
                arms=arms[index],
                safety=safety,
                interaction_radius=args.interaction_radius,
                support_radius=args.support_radius,
                box_padding=args.box_padding,
            )
            if not np.any(mask):
                raise RuntimeError(f"flower seed mask is empty on frame {index}")
            seed_masks[index] = mask
            cv2.imwrite(
                str(output_dir / "assets" / f"flower-seed-{index:04d}.png"),
                mask.astype(np.uint8) * 255,
            )
            seed_rows.append({"frame": index, **row})

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
        tracked: list[Any | None] = [None] * 660
        for frame_index, object_ids, logits in predictor.propagate_in_video(state):
            object_index = list(object_ids).index(1)
            tracked[frame_index] = logits[object_index, 0].detach().cpu().numpy() > 0.0
    if any(mask is None for mask in tracked):
        missing = [index for index, mask in enumerate(tracked) if mask is None]
        raise RuntimeError(f"SAM2 flower tracking missed frames: {missing[:12]}")

    final_masks = []
    strict_recalls = []
    hand_overlaps = []
    areas = []
    for index, tracked_mask in enumerate(tracked):
        frame = cv2.imread(str(frames_dir / f"{index:05d}.jpg"))
        strict = _strict_flower_seed(cv2, np, frame)
        interaction = cv2.dilate(
            hands[index].astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (args.interaction_radius * 2 + 1, args.interaction_radius * 2 + 1),
            ),
        ) > 0
        strict &= interaction
        support = cv2.dilate(
            strict.astype(np.uint8) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (args.support_radius * 2 + 1, args.support_radius * 2 + 1),
            ),
        ) > 0
        skin = _skin_like(cv2, np, frame) & arms[index]
        tracked_supported = np.logical_and(tracked_mask.astype(bool), support)
        mask = np.logical_or(tracked_supported, strict)
        mask = np.logical_and(mask, np.logical_not(skin.copy()))
        mask = _largest_components(cv2, np, mask, minimum=12)
        final_masks.append(mask)
        seed_pixels = int(np.count_nonzero(strict))
        strict_recalls.append(
            float(np.count_nonzero(mask & strict) / seed_pixels) if seed_pixels else 1.0
        )
        hand_overlaps.append(
            float(np.count_nonzero(mask & hands[index]) / max(1, np.count_nonzero(mask)))
        )
        areas.append(float(np.mean(mask)))
    final = np.stack(final_masks, axis=0)
    packed_path = output_dir / "assets/source-flower-instance-masks-packed.npz"
    np.savez_compressed(
        packed_path,
        packed=np.packbits(final.reshape(660, -1), axis=1, bitorder="little"),
        height=np.asarray(height, dtype=np.int32),
        width=np.asarray(width, dtype=np.int32),
        bitorder=np.asarray("little"),
    )

    review_indices = np.linspace(0, 659, 28, dtype=np.int32)
    review_rows = []
    for index in review_indices:
        frame = cv2.imread(str(frames_dir / f"{int(index):05d}.jpg"))
        strict = _strict_flower_seed(cv2, np, frame)
        strict_view = frame.copy()
        strict_view[strict] = np.asarray([40, 220, 255], dtype=np.uint8)
        overlay = frame.copy()
        overlay[final[index]] = np.rint(
            0.30 * overlay[final[index]] + 0.70 * np.asarray([40, 40, 255])
        ).astype(np.uint8)
        cutout = np.zeros_like(frame)
        cutout[final[index]] = frame[final[index]]
        review_rows.append([frame, strict_view, overlay, cutout])
    review_path = output_dir / "review/flower-instance-track-review.jpg"
    cv2.imwrite(
        str(review_path),
        _contact_sheet(cv2, np, review_rows, ["source", "color seeds", "instance mask", "cutout"]),
        [cv2.IMWRITE_JPEG_QUALITY, 93],
    )

    metrics = {
        "frames": 660,
        "seed_frames": len(seed_indices),
        "flower_fraction_mean": float(np.mean(areas)),
        "flower_fraction_min": float(np.min(areas)),
        "flower_fraction_max": float(np.max(areas)),
        "strict_seed_recall_mean": float(np.mean(strict_recalls)),
        "strict_seed_recall_min": float(np.min(strict_recalls)),
        "hand_overlap_mean": float(np.mean(hand_overlaps)),
        "hand_overlap_max": float(np.max(hand_overlaps)),
        "empty_frames": int(np.count_nonzero(np.asarray(areas) == 0.0)),
    }
    record.update(
        {
            "status": "PARTIAL",
            "honest_status": (
                "PARTIAL: complete-flower candidates were tracked and automatic recall/hand "
                "overlap metrics recorded; end-to-end compositing and human review remain pending."
            ),
            "method": {
                "seed_spacing": args.seed_spacing,
                "interaction_radius": args.interaction_radius,
                "support_radius": args.support_radius,
                "box_padding": args.box_padding,
                "seed_strategy": "strict flower colors as SAM2 positive seeds",
                "negative_strategy": "skin pixels inside per-frame MediaPipe arm corridors",
                "temporal_strategy": "one flower-union object with repeated SAM2 mask corrections",
            },
            "extract_command": extract_command,
            "seed_generation": seed_rows,
            "metrics": metrics,
            "packages": {
                name: _package_version(name)
                for name in ("numpy", "opencv-python-headless", "torch")
            },
            "outputs": {
                "packed_masks": {"path": str(packed_path), "sha256": file_sha256(packed_path)},
                "review": {"path": str(review_path), "sha256": file_sha256(review_path)},
            },
        }
    )
    _write_json(output_dir / "manifest.json", record)
    (output_dir / "logs/run.log").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), **metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
