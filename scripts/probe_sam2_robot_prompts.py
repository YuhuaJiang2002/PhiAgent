#!/usr/bin/env python3
"""Compare bounded SAM2 robot prompts before a full 660-frame rerun."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.minimax_h3 import file_sha256  # noqa: E402
from phiagent.rendering.wan_animate import query_gpus  # noqa: E402

SAM2_COMMIT = "0e78a118995e66bb27d78518c4bd9a3e95b4e266"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
        "head": head.stdout.strip() if head.returncode == 0 else None,
        "status": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--pose-model", type=Path)
    parser.add_argument("--sam2-config", default="sam2_hiera_l.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", default="72,144,284,432,576,659")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=75000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"experiment already exists: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = output_dir / "provenance" / "execution-sources" / Path(__file__).name
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), frozen)
    paths = {
        "frames_dir": args.frames_dir.expanduser().resolve(),
        "sam2_repo": args.sam2_repo.expanduser().resolve(),
        "sam2_checkpoint": args.sam2_checkpoint.expanduser().resolve(),
    }
    if bool(args.source_video) != bool(args.pose_model):
        raise ValueError("source-video and pose-model must be supplied together")
    if args.source_video:
        paths["source_video"] = args.source_video.expanduser().resolve()
        paths["pose_model"] = args.pose_model.expanduser().resolve()
    if not paths["frames_dir"].is_dir() or not paths["sam2_repo"].is_dir():
        raise ValueError("frames-dir and sam2-repo must exist")
    if not paths["sam2_checkpoint"].is_file():
        raise ValueError("SAM2 checkpoint does not exist")
    for optional in ("source_video", "pose_model"):
        if optional in paths and not paths[optional].is_file():
            raise ValueError(f"{optional} does not exist")
    frame_indices = [int(item) for item in args.frames.split(",")]
    frame_paths = [paths["frames_dir"] / f"{index:05d}.jpg" for index in frame_indices]
    if not all(path.is_file() for path in frame_paths):
        raise ValueError("one or more requested frames do not exist")

    gpus, inventory_raw, processes_raw = query_gpus()
    selected = next((item for item in gpus if item.physical_index == args.gpu), None)
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
        raise RuntimeError(f"SAM2 commit is {sam2_commit}, expected {SAM2_COMMIT}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected.physical_index)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    sys.path.insert(0, str(paths["sam2_repo"]))

    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "sam2_image_predictor_robot_prompt_probe",
        "status": "running",
        "honest_status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "git": _git_state(),
        "sam2_commit": sam2_commit,
        "gpu": {
            "physical_index": selected.physical_index,
            "name": selected.name,
            "free_mib": selected.free_mib,
            "total_mib": selected.total_mib,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "inventory_raw": inventory_raw,
            "processes_raw": processes_raw,
        },
        "inputs": {
            "frames": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in frame_paths
            ],
            "sam2_checkpoint": {
                "path": str(paths["sam2_checkpoint"]),
                "sha256": file_sha256(paths["sam2_checkpoint"]),
            },
        },
        "execution_source": {"path": str(frozen), "sha256": file_sha256(frozen)},
    }
    _write_json(manifest_path, record)

    import cv2
    import mediapipe as mp
    import numpy as np
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
    pose_points: dict[int, np.ndarray] = {}
    if "source_video" in paths:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(paths["pose_model"])),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.30,
            min_pose_presence_confidence=0.30,
            min_tracking_confidence=0.30,
            output_segmentation_masks=False,
        )
        capture = cv2.VideoCapture(str(paths["source_video"]))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        requested = set(frame_indices)
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                ok, source_frame = capture.read()
                if not ok:
                    break
                if frame_index in requested:
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB),
                    )
                    result = landmarker.detect_for_video(
                        image, round(frame_index * 1000.0 / fps)
                    )
                    if not result.pose_landmarks:
                        raise RuntimeError(f"source pose missing on frame {frame_index}")
                    landmarks = result.pose_landmarks[0]
                    pose_points[frame_index] = np.asarray(
                        [
                            (landmarks[index].x * source_frame.shape[1],
                             landmarks[index].y * source_frame.shape[0])
                            for index in (13, 14, 15, 16)
                        ],
                        dtype=np.float32,
                    )
                frame_index += 1
        capture.release()
    box = np.asarray((407.0, 35.0, 721.0, 427.0), dtype=np.float32)
    baseline = np.asarray(
        ((526.32, 113.40), (545.16, 215.32), (557.72, 293.72)),
        dtype=np.float32,
    )
    corrected = np.asarray(
        ((590.0, 130.0), (575.0, 215.0), (645.0, 205.0), (590.0, 305.0)),
        dtype=np.float32,
    )
    negative_left = np.asarray(((430.0, 280.0),), dtype=np.float32)
    negative_left_pair = np.asarray(((430.0, 280.0), (465.0, 300.0)), dtype=np.float32)
    rows = []
    results = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for frame_index, frame_path in zip(frame_indices, frame_paths, strict=True):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"cannot decode {frame_path}")
            image_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if frame_index in pose_points:
                elbows_wrists = pose_points[frame_index]
                wrists = elbows_wrists[2:]
                configurations = [
                    ("baseline_box", baseline, np.ones(len(baseline), np.int32), box),
                    ("corrected_box", corrected, np.ones(len(corrected), np.int32), box),
                    ("corrected_no_box", corrected, np.ones(len(corrected), np.int32), None),
                    (
                        "corrected_no_box_wrists",
                        np.concatenate((corrected, wrists)),
                        np.ones(len(corrected) + len(wrists), np.int32),
                        None,
                    ),
                    (
                        "corrected_no_box_elbows_wrists",
                        np.concatenate((corrected, elbows_wrists)),
                        np.ones(len(corrected) + len(elbows_wrists), np.int32),
                        None,
                    ),
                ]
            else:
                configurations = [
                    ("baseline_box", baseline, np.ones(len(baseline), np.int32), box),
                    ("corrected_box", corrected, np.ones(len(corrected), np.int32), box),
                    ("corrected_no_box", corrected, np.ones(len(corrected), np.int32), None),
                    (
                        "corrected_box_negative_left",
                        np.concatenate((corrected, negative_left)),
                        np.concatenate(
                            (np.ones(len(corrected)), np.zeros(len(negative_left)))
                        ).astype(np.int32),
                        box,
                    ),
                    (
                        "corrected_box_negative_left_pair",
                        np.concatenate((corrected, negative_left_pair)),
                        np.concatenate(
                            (np.ones(len(corrected)), np.zeros(len(negative_left_pair)))
                        ).astype(np.int32),
                        box,
                    ),
                ]
            row = []
            for name, points, labels, prompt_box in configurations:
                masks, scores, _ = image_predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=prompt_box,
                    multimask_output=True,
                )
                candidates = []
                for candidate, (mask, score) in enumerate(zip(masks, scores, strict=True)):
                    mask = mask.astype(bool)
                    positive = points[labels == 1]
                    negative = points[labels == 0]
                    pos_fraction = float(
                        np.mean([mask[round(y), round(x)] for x, y in positive])
                    )
                    neg_fraction = float(
                        np.mean([mask[round(y), round(x)] for x, y in negative])
                    ) if len(negative) else 0.0
                    fraction = float(np.mean(mask))
                    rank = (
                        pos_fraction == 1.0,
                        neg_fraction == 0.0,
                        0.025 <= fraction <= 0.15,
                        float(score),
                        -abs(fraction - 0.07),
                    )
                    candidates.append(
                        {
                            "candidate": candidate,
                            "score": float(score),
                            "fraction": fraction,
                            "positive_fraction": pos_fraction,
                            "negative_fraction": neg_fraction,
                            "rank": rank,
                        }
                    )
                chosen = max(candidates, key=lambda item: item["rank"])
                mask = masks[int(chosen["candidate"])].astype(bool)
                overlay = frame.copy()
                overlay[mask] = np.rint(
                    overlay[mask].astype(np.float32) * 0.48
                    + np.asarray((20, 35, 240), dtype=np.float32) * 0.52
                ).astype(np.uint8)
                for point, label in zip(points, labels, strict=True):
                    color = (50, 240, 50) if label else (40, 220, 255)
                    cv2.drawMarker(
                        overlay,
                        tuple(np.rint(point).astype(int)),
                        color,
                        cv2.MARKER_CROSS,
                        13,
                        2,
                    )
                cv2.putText(
                    overlay,
                    f"{frame_index} {name} A={chosen['fraction']:.3f}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                row.append(cv2.resize(overlay, (416, 240), interpolation=cv2.INTER_AREA))
                results.append(
                    {
                        "frame": frame_index,
                        "configuration": name,
                        "points_xy": points.tolist(),
                        "point_labels": labels.tolist(),
                        "box_xyxy": prompt_box.tolist() if prompt_box is not None else None,
                        "chosen": chosen,
                        "candidates": candidates,
                    }
                )
            rows.append(np.hstack(row))
    contact = np.vstack(rows)
    contact_path = output_dir / "sam2-prompt-comparison.jpg"
    cv2.imwrite(str(contact_path), contact)
    record.update(
        {
            "status": "completed_review_required",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "results": results,
            "output": {"path": str(contact_path), "sha256": file_sha256(contact_path)},
        }
    )
    _write_json(manifest_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
