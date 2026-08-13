#!/usr/bin/env python3
"""Track the H3 robot's left and right limbs as independent SAM2 objects."""

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
        ["git", "--no-pager", "status", "--short"], cwd=PROJECT_ROOT,
        check=False, capture_output=True, text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=PROJECT_ROOT,
        check=False, capture_output=True, text=True,
    )
    return {
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def _load_packed(np: Any, path: Path, key: str = "packed") -> Any:
    payload = np.load(path)
    height, width = int(payload["height"]), int(payload["width"])
    packed = payload[key]
    unpacked = np.unpackbits(packed, axis=1, bitorder="little")
    return unpacked[:, : height * width].reshape(len(packed), height, width).astype(bool)


def _pose_tracks(np: Any, path: Path) -> tuple[Any, dict[int, int]]:
    payload = np.load(path)
    landmarks = payload["landmarks_xy"].astype(np.float32)
    ids = [int(item) for item in payload["landmark_ids"]]
    return landmarks, {landmark_id: index for index, landmark_id in enumerate(ids)}


def _corridor(
    cv2: Any,
    np: Any,
    points: Any,
    mapping: dict[int, int],
    side: str,
    radius: int,
    shape: tuple[int, int],
) -> Any:
    joints = (11, 13, 15, 17, 19, 21) if side == "left" else (12, 14, 16, 18, 20, 22)
    shoulder, elbow, wrist, pinky, index_tip, thumb = [
        points[mapping[item]] for item in joints
    ]
    mask = np.zeros(shape, dtype=np.uint8)
    pairs = (
        (shoulder, elbow), (elbow, wrist), (wrist, pinky),
        (wrist, index_tip), (wrist, thumb),
    )
    for start, end in pairs:
        a = tuple(np.rint(start).astype(int))
        b = tuple(np.rint(end).astype(int))
        cv2.line(mask, a, b, 255, radius * 2, cv2.LINE_AA)
        cv2.circle(mask, a, radius, 255, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(mask, b, radius, 255, cv2.FILLED, cv2.LINE_AA)
    return mask > 0


def _snap_points(np: Any, points: Any, reference: Any, maximum_distance: float) -> Any:
    ys, xs = np.where(reference)
    if not len(xs):
        return points.astype(np.float32)
    candidates = np.stack((xs, ys), axis=1).astype(np.float32)
    snapped = []
    for point in points:
        distances = np.sum((candidates - point[None, :]) ** 2, axis=1)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= maximum_distance * maximum_distance:
            snapped.append(candidates[nearest])
        else:
            snapped.append(point)
    return np.asarray(snapped, dtype=np.float32)


def _arm_prompt_points(np: Any, pose: Any, mapping: dict[int, int], side: str) -> Any:
    elbow_id, wrist_id, hand_ids = (
        (13, 15, (17, 19, 21)) if side == "left" else (14, 16, (18, 20, 22))
    )
    elbow = pose[mapping[elbow_id]]
    wrist = pose[mapping[wrist_id]]
    hand = np.mean([pose[mapping[item]] for item in hand_ids], axis=0)
    return np.asarray([elbow, (elbow + wrist) / 2.0, wrist, hand], dtype=np.float32)


def _seed_mask(
    cv2: Any,
    np: Any,
    *,
    image_predictor: Any,
    frame: Any,
    pose: Any,
    mapping: dict[int, int],
    side: str,
    body: Any,
    wrist_reference: Any,
    corridor_radius: int,
    box_padding: int,
) -> tuple[Any, dict[str, object]]:
    height, width = frame.shape[:2]
    corridor = _corridor(
        cv2, np, pose, mapping, side, corridor_radius, (height, width)
    )
    reference = np.logical_or(body, wrist_reference)
    local_reference = np.logical_and(reference, corridor)
    points = _arm_prompt_points(np, pose, mapping, side)
    points = _snap_points(np, points, local_reference, maximum_distance=44.0)
    pose_ids = (11, 13, 15, 17, 19, 21) if side == "left" else (12, 14, 16, 18, 20, 22)
    box_points = np.asarray([pose[mapping[item]] for item in pose_ids])
    left = max(0, round(float(np.min(box_points[:, 0]))) - box_padding)
    top = max(0, round(float(np.min(box_points[:, 1]))) - box_padding)
    right = min(width - 1, round(float(np.max(box_points[:, 0]))) + box_padding)
    bottom = min(height - 1, round(float(np.max(box_points[:, 1]))) + box_padding)
    box = np.asarray([left, top, right, bottom], dtype=np.float32)
    image_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    candidates, scores, _ = image_predictor.predict(
        point_coords=points,
        point_labels=np.ones(len(points), dtype=np.int32),
        box=box,
        multimask_output=True,
    )
    reference_pixels = max(1, int(np.count_nonzero(local_reference)))
    rows = []
    for candidate_index, (candidate, sam_score) in enumerate(zip(candidates, scores)):
        candidate = candidate.astype(bool)
        candidate = np.logical_and(candidate, corridor)
        area = int(np.count_nonzero(candidate))
        reference_recall = float(
            np.count_nonzero(np.logical_and(candidate, local_reference))
            / reference_pixels
        )
        contains_points = float(
            np.mean(
                [
                    candidate[
                        min(height - 1, max(0, round(float(y)))),
                        min(width - 1, max(0, round(float(x)))),
                    ]
                    for x, y in points
                ]
            )
        )
        fraction = float(area / (height * width))
        plausible = 0.002 <= fraction <= 0.055 and contains_points >= 0.50
        rank = (
            int(plausible),
            0.55 * contains_points + 0.35 * reference_recall + 0.10 * float(sam_score),
            -abs(fraction - 0.018),
        )
        rows.append(
            {
                "candidate": candidate_index,
                "mask": candidate,
                "sam_score": float(sam_score),
                "fraction": fraction,
                "reference_recall": reference_recall,
                "contains_points": contains_points,
                "plausible": plausible,
                "rank": rank,
            }
        )
    selected = max(rows, key=lambda item: item["rank"])
    if not selected["plausible"]:
        fallback = cv2.morphologyEx(
            local_reference.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        ) > 0
        selected_mask = fallback
        fallback_used = True
    else:
        selected_mask = selected["mask"]
        fallback_used = False
    return selected_mask, {
        "side": side,
        "box_xyxy": box.tolist(),
        "positive_points_xy": points.tolist(),
        "fallback_used": fallback_used,
        "selected_candidate": selected["candidate"],
        "selected_fraction": float(np.mean(selected_mask)),
        "candidates": [
            {key: value for key, value in row.items() if key not in {"mask", "rank"}}
            for row in rows
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-video", type=Path, required=True)
    parser.add_argument("--source-limb-masks", type=Path, required=True)
    parser.add_argument("--robot-body-masks", type=Path, required=True)
    parser.add_argument("--robot-wrist-masks", type=Path, required=True)
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", default="sam2_hiera_l.yaml")
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=60000)
    parser.add_argument("--seed-spacing", type=int, default=24)
    parser.add_argument("--corridor-radius", type=int, default=34)
    parser.add_argument("--box-padding", type=int, default=42)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"experiment already exists: {output_dir}")
    for relative in ("assets", "input/frames", "logs", "review", "provenance/execution-sources"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    frozen_source = output_dir / "provenance/execution-sources" / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), frozen_source)
    paths = {
        "generated_video": args.generated_video.expanduser().resolve(),
        "source_limb_masks": args.source_limb_masks.expanduser().resolve(),
        "robot_body_masks": args.robot_body_masks.expanduser().resolve(),
        "robot_wrist_masks": args.robot_wrist_masks.expanduser().resolve(),
        "sam2_repo": args.sam2_repo.expanduser().resolve(),
        "sam2_checkpoint": args.sam2_checkpoint.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    command = [sys.executable, *sys.argv]
    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "preflight_started",
        "honest_status": "PARTIAL",
        "scope": "independent left/right H3 mechanical-limb SAM2 tracking",
        "command": command,
        "command_shell": shlex.join(command),
        "seed": args.seed,
        "execution_source": {"path": str(frozen_source), "sha256": file_sha256(frozen_source)},
    }
    _write_json(output_dir / "manifest.json", record)

    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(paths["generated_video"]))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if (frame_count, width, height) != (660, 832, 480):
        raise RuntimeError("expected the aligned 660-frame 832x480 H3 video")
    body = _load_packed(np, paths["robot_body_masks"])
    wrist = _load_packed(np, paths["robot_wrist_masks"])
    pose, mapping = _pose_tracks(np, paths["source_limb_masks"])
    if body.shape != (660, height, width) or wrist.shape != body.shape:
        raise RuntimeError("robot reference masks do not match the video")

    gpus, inventory_raw, processes_raw = query_gpus()
    selected = next((item for item in gpus if item.physical_index == args.gpu), None)
    if selected is None or selected.free_mib < args.minimum_free_gpu_mib:
        raise RuntimeError("requested physical GPU is missing or lacks required free memory")
    sam2_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=paths["sam2_repo"], check=True,
        capture_output=True, text=True,
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
            "video": {"frames": frame_count, "width": width, "height": height, "fps": fps},
            "gpu": {
                "physical_index": selected.physical_index, "name": selected.name,
                "free_mib": selected.free_mib, "total_mib": selected.total_mib,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "inventory_raw": inventory_raw, "processes_raw": processes_raw,
            },
            "sam2_commit": sam2_commit,
            "git": _git_state(), "hostname": socket.gethostname(),
            "platform": platform.platform(),
        }
    )
    _write_json(output_dir / "manifest.json", record)

    frames_dir = output_dir / "input/frames"
    extract_command = [
        str(paths["ffmpeg"]), "-y", "-v", "error", "-i", str(paths["generated_video"]),
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
        args.sam2_config, str(paths["sam2_checkpoint"]), device="cuda"
    )
    image_predictor = SAM2ImagePredictor(predictor)
    seeds = {1: [None] * 660, 2: [None] * 660}
    seed_rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in seed_indices:
            frame = cv2.imread(str(frames_dir / f"{index:05d}.jpg"))
            for object_id, side in ((1, "left"), (2, "right")):
                mask, row = _seed_mask(
                    cv2, np, image_predictor=image_predictor, frame=frame,
                    pose=pose[index], mapping=mapping, side=side, body=body[index],
                    wrist_reference=wrist[index], corridor_radius=args.corridor_radius,
                    box_padding=args.box_padding,
                )
                if not np.any(mask):
                    raise RuntimeError(f"empty {side} limb seed on frame {index}")
                seeds[object_id][index] = mask
                seed_rows.append({"frame": index, **row})
                cv2.imwrite(
                    str(output_dir / "assets" / f"{side}-seed-{index:04d}.png"),
                    mask.astype(np.uint8) * 255,
                )
        state = predictor.init_state(
            video_path=str(frames_dir), offload_video_to_cpu=True,
            offload_state_to_cpu=True, async_loading_frames=False,
        )
        for index in seed_indices:
            for object_id in (1, 2):
                predictor.add_new_mask(
                    inference_state=state, frame_idx=index, obj_id=object_id,
                    mask=seeds[object_id][index],
                )
        tracked = {1: [None] * 660, 2: [None] * 660}
        for frame_index, object_ids, logits in predictor.propagate_in_video(state):
            for object_id in (1, 2):
                object_index = list(object_ids).index(object_id)
                tracked[object_id][frame_index] = (
                    logits[object_index, 0].detach().cpu().numpy() > 0.0
                )
    if any(mask is None for masks in tracked.values() for mask in masks):
        raise RuntimeError("SAM2 limb tracking did not return every object/frame")

    final_sides = {1: [], 2: []}
    for index in range(660):
        for object_id, side in ((1, "left"), (2, "right")):
            support = _corridor(
                cv2, np, pose[index], mapping, side,
                args.corridor_radius + 12, (height, width),
            )
            mask = np.logical_and(tracked[object_id][index], support)
            final_sides[object_id].append(mask)
    left_masks = np.stack(final_sides[1])
    right_masks = np.stack(final_sides[2])
    union_masks = np.logical_or(left_masks, right_masks)
    packed_path = output_dir / "assets/sam2-robot-limb-masks-packed.npz"
    np.savez_compressed(
        packed_path,
        packed=np.packbits(union_masks.reshape(660, -1), axis=1, bitorder="little"),
        left_packed=np.packbits(left_masks.reshape(660, -1), axis=1, bitorder="little"),
        right_packed=np.packbits(right_masks.reshape(660, -1), axis=1, bitorder="little"),
        height=np.asarray(height, dtype=np.int32), width=np.asarray(width, dtype=np.int32),
        bitorder=np.asarray("little"),
    )

    rows = []
    for index in np.linspace(0, 659, 28, dtype=np.int32):
        frame = cv2.imread(str(frames_dir / f"{int(index):05d}.jpg"))
        body_cutout = np.zeros_like(frame)
        body_cutout[body[index]] = frame[body[index]]
        wrist_cutout = np.zeros_like(frame)
        wrist_cutout[wrist[index]] = frame[wrist[index]]
        limb_cutout = np.zeros_like(frame)
        limb_cutout[union_masks[index]] = frame[union_masks[index]]
        combined = np.zeros_like(frame)
        combined[np.logical_or(body[index], union_masks[index])] = frame[
            np.logical_or(body[index], union_masks[index])
        ]
        cells = []
        for label, item in (
            (f"generated {int(index)}", frame), ("body v19", body_cutout),
            ("wrist v18", wrist_cutout), ("new limbs", limb_cutout),
            ("body + new limbs", combined),
        ):
            cell = item.copy()
            cv2.putText(cell, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(cell)
        rows.append(cv2.hconcat(cells))
    review_path = output_dir / "review/robot-limb-track-review.jpg"
    cv2.imwrite(str(review_path), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 93])
    areas = np.mean(union_masks, axis=(1, 2))
    metrics = {
        "frames": 660,
        "seed_frames": len(seed_indices),
        "empty_frames": int(np.count_nonzero(areas == 0.0)),
        "limb_fraction_mean": float(np.mean(areas)),
        "limb_fraction_min": float(np.min(areas)),
        "limb_fraction_max": float(np.max(areas)),
        "fallback_seed_count": sum(bool(row["fallback_used"]) for row in seed_rows),
    }
    record.update(
        {
            "status": "PARTIAL",
            "honest_status": (
                "PARTIAL: independent mechanical-limb tracks were generated; "
                "end-to-end composition and human review remain pending."
            ),
            "method": {
                "objects": ["left mechanical arm/hand", "right mechanical arm/hand"],
                "seed_spacing": args.seed_spacing,
                "corridor_radius": args.corridor_radius,
                "box_padding": args.box_padding,
                "coordinate_frame": "camera:generated_video_pixels with aligned source pose",
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
