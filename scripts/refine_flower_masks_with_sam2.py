#!/usr/bin/env python3
"""Refine a drifting flower track with per-frame SAM2 image evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

SAM2_COMMIT = "0e78a118995e66bb27d78518c4bd9a3e95b4e266"


def _gpu_inventory() -> tuple[list[dict[str, Any]], str, str]:
    inventory_command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    inventory = subprocess.run(
        inventory_command, check=True, capture_output=True, text=True
    )
    rows = []
    for raw in inventory.stdout.splitlines():
        index, name, total, free, used, utilization = [
            value.strip() for value in raw.split(",", 5)
        ]
        rows.append(
            {
                "physical_index": int(index),
                "name": name,
                "total_mib": int(total),
                "free_mib": int(free),
                "used_mib": int(used),
                "utilization_percent": int(utilization),
            }
        )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return rows, inventory.stdout, processes.stdout


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
) -> tuple[Any, dict[str, Any]]:
    height, width = frame.shape[:2]
    strict = _strict_flower_seed(cv2, np, frame)
    interaction = cv2.dilate(
        hands.astype(np.uint8) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (interaction_radius * 2 + 1, interaction_radius * 2 + 1),
        ),
    ) > 0
    strict &= interaction & safety
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
        component_support = labels == component
        component_seed = strict & component_support
        seed_pixels = int(np.count_nonzero(component_seed))
        component_area = int(stats[component, cv2.CC_STAT_AREA])
        if seed_pixels < 8 or component_area > round(height * width * 0.12):
            continue
        ys, xs = np.where(component_seed)
        box = np.asarray(
            [
                max(0, int(xs.min()) - box_padding),
                max(0, int(ys.min()) - box_padding),
                min(width - 1, int(xs.max()) + box_padding),
                min(height - 1, int(ys.max()) + box_padding),
            ],
            dtype=np.float32,
        )
        positive = _sample_component_points(cv2, np, component_seed, 6)
        left, top, right, bottom = box.astype(int)
        negative_region = np.zeros_like(skin)
        negative_region[top : bottom + 1, left : right + 1] = skin[
            top : bottom + 1, left : right + 1
        ]
        negative = _sample_negative_points(np, negative_region, 6)
        points = np.concatenate((positive, negative), axis=0)
        point_labels = np.concatenate(
            (
                np.ones(len(positive), dtype=np.int32),
                np.zeros(len(negative), dtype=np.int32),
            )
        )
        candidates, scores, _ = image_predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        ranked = []
        for candidate_index, (candidate, score) in enumerate(zip(candidates, scores)):
            candidate = candidate.astype(bool)
            area = int(np.count_nonzero(candidate))
            recall = float(np.count_nonzero(candidate & component_seed) / seed_pixels)
            skin_fraction = float(np.count_nonzero(candidate & skin) / max(1, area))
            expansion = float(area / seed_pixels)
            plausible = (
                recall >= 0.55
                and seed_pixels <= area <= round(height * width * 0.045)
                and expansion <= 45.0
                and skin_fraction <= 0.20
            )
            rank = (
                int(plausible),
                recall - 2.5 * skin_fraction - max(0.0, expansion - 18.0) / 50.0,
                float(score),
            )
            ranked.append(
                {
                    "mask": candidate,
                    "candidate": candidate_index,
                    "sam_score": float(score),
                    "area": area,
                    "seed_recall": recall,
                    "skin_fraction": skin_fraction,
                    "expansion": expansion,
                    "plausible": plausible,
                    "rank": rank,
                }
            )
        selected = max(ranked, key=lambda item: item["rank"])
        if selected["plausible"]:
            union |= selected["mask"]
        rows.append(
            {
                key: value
                for key, value in selected.items()
                if key not in {"mask", "rank"}
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
    return union, {
        "strict_seed_pixels": int(np.count_nonzero(strict)),
        "mask_pixels": int(np.count_nonzero(union)),
        "clusters_considered": len(rows),
        "clusters_accepted": sum(bool(row["plausible"]) for row in rows),
        "clusters": rows,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_state() -> dict[str, Any]:
    result = {}
    for name, command in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        result[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    return result


def _load_packed(np: Any, path: Path, key: str) -> tuple[Any, int, int, str]:
    payload = np.load(path, allow_pickle=False)
    return (
        payload[key],
        int(payload["height"]),
        int(payload["width"]),
        str(payload["bitorder"]),
    )


def _unpack(np: Any, payload: tuple[Any, int, int, str], index: int) -> Any:
    packed, height, width, bitorder = payload
    return np.unpackbits(packed[index], bitorder=bitorder)[: height * width].reshape(
        height, width
    ).astype(bool)


def compose_refined_track(
    np: Any,
    *,
    tracked: Any,
    person: Any,
    hands: Any,
    sam2_object: Any,
    appearance_core: Any,
) -> Any:
    """Fuse persistent identity and an independent per-frame object mask.

    The persistent track remains authoritative outside the person.  Within the
    person, SAM2 or a conservative flower appearance core must independently
    confirm ownership.  Source-hand pixels always remain available for robot
    replacement and contact evidence.
    """

    values = [
        np.asarray(item, dtype=np.bool_)
        for item in (tracked, person, hands, sam2_object, appearance_core)
    ]
    if any(item.ndim != 2 or item.shape != values[0].shape for item in values):
        raise ValueError("all refinement masks must share one 2-D image plane")
    track, human, hand, model_object, appearance = values
    outside_person = np.logical_and(track, np.logical_not(human))
    confirmed_inside = np.logical_and.reduce(
        (track, human, np.logical_or(model_object, appearance))
    )
    return np.logical_and(
        np.logical_or(outside_person, confirmed_inside), np.logical_not(hand)
    )


def _aligned_frame(cv2: Any, frame: Any, *, width: int, height: int) -> Any:
    scale = max(width / frame.shape[1], height / frame.shape[0])
    scaled_width = round(frame.shape[1] * scale)
    scaled_height = round(frame.shape[0] * scale)
    resized = cv2.resize(
        frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LANCZOS4
    )
    left = max(0, (scaled_width - width) // 2)
    top = max(0, (scaled_height - height) // 2)
    return resized[top : top + height, left : left + width]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--flower-masks", type=Path, required=True)
    parser.add_argument("--person-masks", type=Path, required=True)
    parser.add_argument("--pose-limb-masks", type=Path, required=True)
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", default="sam2_hiera_l.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=20000)
    parser.add_argument("--expected-frames", type=int, default=660)
    parser.add_argument("--interaction-radius", type=int, default=100)
    parser.add_argument("--support-radius", type=int, default=12)
    parser.add_argument("--box-padding", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    (output / "assets").mkdir()
    (output / "review").mkdir()
    paths = {
        "source": args.source_video.expanduser().resolve(),
        "flower_masks": args.flower_masks.expanduser().resolve(),
        "person_masks": args.person_masks.expanduser().resolve(),
        "pose_limb_masks": args.pose_limb_masks.expanduser().resolve(),
        "sam2_repo": args.sam2_repo.expanduser().resolve(),
        "sam2_checkpoint": args.sam2_checkpoint.expanduser().resolve(),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    gpus, inventory_raw, processes_raw = _gpu_inventory()
    selected = next(
        (row for row in gpus if row["physical_index"] == args.physical_gpu), None
    )
    if selected is None:
        raise RuntimeError(f"physical GPU {args.physical_gpu} is unavailable")
    if selected["free_mib"] < args.minimum_free_gpu_mib:
        raise RuntimeError(
            f"GPU {args.physical_gpu} has {selected['free_mib']} MiB free; "
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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected["physical_index"])
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(paths["sam2_repo"]))

    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cv2.setNumThreads(1)
    torch.manual_seed(args.seed)
    flower_payload = _load_packed(np, paths["flower_masks"], "packed")
    person_payload = _load_packed(np, paths["person_masks"], "packed")
    hands_payload = _load_packed(np, paths["pose_limb_masks"], "hands_packed")
    arms_payload = _load_packed(np, paths["pose_limb_masks"], "arms_packed")
    _, height, width, _ = flower_payload
    if any(len(payload[0]) != args.expected_frames for payload in (
        flower_payload, person_payload, hands_payload, arms_payload
    )):
        raise ValueError("all packed masks must cover the full timeline")

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "honest_status": "PARTIAL",
        "scope": "per-frame SAM2 flower observation fused with a persistent track",
        "physical_evidence": False,
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "git": _git_state(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": {
            "physical_index": selected["physical_index"],
            "name": selected["name"],
            "free_mib_at_start": selected["free_mib"],
            "total_mib": selected["total_mib"],
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "inventory_raw": inventory_raw,
            "processes_raw": processes_raw,
        },
        "sam2": {
            "commit": sam2_commit,
            "config": args.sam2_config,
            "checkpoint": {
                "path": str(paths["sam2_checkpoint"]),
                "sha256": _sha256(paths["sam2_checkpoint"]),
            },
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path) if path.is_file() else None}
            for name, path in paths.items()
        },
        "coordinate_frame": f"camera:source_aligned_{width}x{height}",
        "config": {
            "expected_frames": args.expected_frames,
            "interaction_radius": args.interaction_radius,
            "support_radius": args.support_radius,
            "box_padding": args.box_padding,
        },
    }
    _write_json(output / "manifest.json", manifest)

    model = build_sam2(
        args.sam2_config, str(paths["sam2_checkpoint"]), device="cuda"
    )
    predictor = SAM2ImagePredictor(model)
    capture = cv2.VideoCapture(str(paths["source"]))
    refined_frames = []
    review_rows = []
    frame_rows = []
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(args.expected_frames):
            ok, native = capture.read()
            if not ok:
                raise RuntimeError(f"source decode stopped at frame {index}")
            frame = _aligned_frame(cv2, native, width=width, height=height)
            tracked = _unpack(np, flower_payload, index)
            person = _unpack(np, person_payload, index)
            hands = _unpack(np, hands_payload, index)
            arms = _unpack(np, arms_payload, index)
            sam2_object, sam_row = _seed_instance_mask(
                cv2,
                np,
                image_predictor=predictor,
                frame=frame,
                hands=hands,
                arms=arms,
                safety=np.ones_like(tracked),
                interaction_radius=args.interaction_radius,
                support_radius=args.support_radius,
                box_padding=args.box_padding,
            )
            appearance = _strict_flower_seed(cv2, np, frame)
            refined = compose_refined_track(
                np,
                tracked=tracked,
                person=person,
                hands=hands,
                sam2_object=sam2_object,
                appearance_core=appearance,
            )
            refined_frames.append(refined)
            input_person = int(np.count_nonzero(tracked & person))
            output_person = int(np.count_nonzero(refined & person))
            frame_rows.append(
                {
                    "frame": index,
                    "input_pixels": int(np.count_nonzero(tracked)),
                    "refined_pixels": int(np.count_nonzero(refined)),
                    "input_person_overlap_pixels": input_person,
                    "refined_person_overlap_pixels": output_person,
                    "person_overlap_retained_fraction": (
                        output_person / input_person if input_person else 1.0
                    ),
                    "sam2": sam_row,
                }
            )
            if index in {0, 120, 240, 360, 480, 514, 558, 600, 659}:
                overlay = frame.copy()
                overlay[refined] = np.rint(
                    0.35 * overlay[refined] + 0.65 * np.asarray([40, 220, 40])
                ).astype(np.uint8)
                review_rows.append(cv2.resize(overlay, (416, 240)))
    capture.release()
    elapsed = time.perf_counter() - started
    refined_array = np.stack(refined_frames)
    packed_path = output / "assets" / "sam2-refined-flower-masks-packed.npz"
    np.savez_compressed(
        packed_path,
        packed=np.packbits(
            refined_array.reshape(args.expected_frames, -1), axis=1, bitorder="little"
        ),
        height=np.asarray(height, dtype=np.int32),
        width=np.asarray(width, dtype=np.int32),
        bitorder=np.asarray("little"),
    )
    rows_path = output / "frame-metrics.json"
    rows_path.write_text(json.dumps(frame_rows, indent=2, sort_keys=True) + "\n")
    review = np.vstack(
        [np.hstack(review_rows[start : start + 3]) for start in range(0, 9, 3)]
    )
    review_path = output / "review" / "sam2-refined-flower-track.jpg"
    cv2.imwrite(str(review_path), review)
    retained = [row["person_overlap_retained_fraction"] for row in frame_rows]
    manifest.update(
        {
            "status": "PARTIAL",
            "decision": "AWAITING_DOWNSTREAM_FROZEN_VIDEO_AUDITS",
            "metrics": {
                "frames": args.expected_frames,
                "wall_seconds": elapsed,
                "processing_fps": args.expected_frames / elapsed,
                "mean_person_overlap_retained_fraction": float(np.mean(retained)),
                "p95_person_overlap_retained_fraction": float(np.quantile(retained, 0.95)),
            },
            "outputs": {
                "packed_masks": {"path": str(packed_path), "sha256": _sha256(packed_path)},
                "frame_metrics": {"path": str(rows_path), "sha256": _sha256(rows_path)},
                "review": {"path": str(review_path), "sha256": _sha256(review_path)},
            },
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "hydra-core", "iopath", "omegaconf")
            },
            "limitations": [
                "SAM2 masks are camera-frame semantic observations, not metric depth.",
                "Downstream frozen temporal, contact, adversarial, and human review gates remain mandatory.",
            ],
        }
    )
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output / "manifest.json"), **manifest["metrics"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
