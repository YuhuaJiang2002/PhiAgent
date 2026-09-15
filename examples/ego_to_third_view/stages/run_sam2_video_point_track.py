#!/usr/bin/env python3
"""Propagate one non-GT point prompt through a rectified Aria video sequence."""

from __future__ import annotations

from runtime import require_launcher
require_launcher()

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"), required=True)
    parser.add_argument(
        "--prompt-frame-index",
        type=int,
        default=0,
        help="Frame on which the operator supplies the non-GT point prompt.",
    )
    parser.add_argument("--label", default="keyboard")
    parser.add_argument(
        "--negative-point",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        action="append",
        default=[],
        help="Optional operator-supplied background click on the prompt frame; repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    cfg = parse_args()
    if not cfg.frames_dir.is_dir() or not cfg.checkpoint.is_file():
        raise SystemExit("SAM2 sequence directory/checkpoint is missing")
    # SAM2's video loader consumes JPEG image sequences.  The rectification
    # stage therefore writes .jpg frames deliberately instead of keeping the
    # larger PNG intermediates.
    frame_files = sorted(cfg.frames_dir.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(cfg.frames_dir.glob("*.jpeg"))
    if not frame_files:
        raise SystemExit("frames directory has no JPEG frames (.jpg or .jpeg)")
    if not 0 <= cfg.prompt_frame_index < len(frame_files):
        raise SystemExit(f"prompt frame must be in [0, {len(frame_files) - 1}]")
    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(cfg.config, str(cfg.checkpoint), device=cfg.device)
    inference_state = predictor.init_state(video_path=str(cfg.frames_dir))
    prompt_points = np.asarray([cfg.point, *cfg.negative_point], dtype=np.float32)
    prompt_labels = np.asarray([1, *([0] * len(cfg.negative_point))], dtype=np.int32)
    _, object_ids, mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=cfg.prompt_frame_index,
        obj_id=1,
        points=prompt_points,
        labels=prompt_labels,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = cfg.output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    areas: dict[str, int] = {}
    def save(frame_idx: int, ids: object, logits: object) -> None:
        for obj_id, mask_logit in zip(ids, logits):
            if int(obj_id) != 1:
                continue
            mask = (mask_logit[0] > 0.0).detach().cpu().numpy().astype(np.uint8)
            frame_id = frame_files[frame_idx].stem
            cv2.imwrite(str(masks_dir / f"{frame_id}.png"), mask * 255)
            areas[frame_id] = int(mask.sum())
    save(cfg.prompt_frame_index, object_ids, mask_logits)
    with torch.inference_mode():
        # Propagate in both temporal directions so a clear middle-frame prompt
        # is not forced to begin at a potentially occluded first frame.
        for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=cfg.prompt_frame_index
        ):
            save(int(frame_idx), object_ids, mask_logits)
        for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=cfg.prompt_frame_index, reverse=True
        ):
            save(int(frame_idx), object_ids, mask_logits)
    report = {
        "schema_version": "phiagent-sam2-video-point-track/1.0",
        "frames_dir": str(cfg.frames_dir.resolve()),
        "frames": len(frame_files),
        "masks": str(masks_dir.resolve()),
        "task_label": cfg.label,
        "prompt": {"type": "positive_and_optional_negative_points", "frame_index": cfg.prompt_frame_index, "positive_xy_px": list(cfg.point), "negative_xy_px": cfg.negative_point, "source": "runtime user/task interaction, not HOT3D annotation"},
        "mask_area_px": areas,
        "coverage": {"frames_with_nonempty_mask": sum(area > 0 for area in areas.values()), "median_area_px": float(np.median(list(areas.values()))) if areas else 0.0},
        "authority": {"uses_hot3d_hand_gt": False, "uses_hot3d_object_gt": False, "state": "predicted_temporal_instance_masks"},
    }
    (cfg.output_dir / "track_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"frames": report["frames"], "coverage": report["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
