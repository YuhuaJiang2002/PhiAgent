#!/usr/bin/env python3
"""Build an authentic-background-first clean plate for the flower demo."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import socket
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remove_h3_robot_shadow import (
    PROJECT_ROOT,
    _git_state,
    _load_packed_masks,
    _package_version,
    _video_info,
    _write_json,
    file_sha256,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--source-person-masks", type=Path, required=True)
    parser.add_argument("--generative-clean-plate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=12)
    parser.add_argument("--person-mask-dilation", type=int, default=8)
    parser.add_argument("--minimum-visible-samples", type=int, default=3)
    parser.add_argument("--transition-sigma", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists():
        raise FileExistsError(f"experiment directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    command = [sys.executable, *sys.argv]
    paths = {
        "source_video": args.source_video.expanduser().resolve(),
        "source_person_masks": args.source_person_masks.expanduser().resolve(),
        "generative_clean_plate": args.generative_clean_plate.expanduser().resolve(),
    }
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "method": "masked_temporal_median_with_generative_never_visible_fill",
        "status": "preflight_started",
        "honest_status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "command_shell": shlex.join(command),
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "git": _git_state(PROJECT_ROOT),
        "gpu": {"used": False, "mode": "CPU clean-plate construction"},
        "config": {
            "sample_stride": args.sample_stride,
            "person_mask_dilation": args.person_mask_dilation,
            "minimum_visible_samples": args.minimum_visible_samples,
            "transition_sigma": args.transition_sigma,
            "coordinate_frame": "camera:source_video_pixels",
        },
    }
    _write_json(manifest_path, record)
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{name} is missing: {path}")

    import cv2
    import numpy as np

    info = _video_info(cv2, paths["source_video"])
    frame_count = int(info["decoded_frames"])
    height = int(info["height"])
    width = int(info["width"])
    masks = _load_packed_masks(
        np,
        paths["source_person_masks"],
        expected_frames=frame_count,
        expected_height=height,
        expected_width=width,
    )
    generative = cv2.imread(str(paths["generative_clean_plate"]), cv2.IMREAD_COLOR)
    if generative is None or generative.shape[:2] != (height, width):
        raise RuntimeError("generative clean plate geometry does not match video")

    sample_indices = list(range(0, frame_count, args.sample_stride))
    if sample_indices[-1] != frame_count - 1:
        sample_indices.append(frame_count - 1)
    wanted = set(sample_indices)
    kernel_size = args.person_mask_dilation * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    capture = cv2.VideoCapture(str(paths["source_video"]))
    samples = []
    blocked = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            mask = masks[index].astype(np.uint8)
            if args.person_mask_dilation:
                mask = cv2.dilate(mask, kernel)
            samples.append(frame)
            blocked.append(mask.astype(bool))
        index += 1
    capture.release()
    if index != frame_count or len(samples) != len(sample_indices):
        raise RuntimeError("could not decode all requested source samples")

    stack = np.stack(samples).astype(np.float32)
    blocked_stack = np.stack(blocked)
    visible_count = np.sum(~blocked_stack, axis=0).astype(np.uint16)
    stack[blocked_stack] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        temporal = np.nanmedian(stack, axis=0)
    low_coverage = visible_count < args.minimum_visible_samples
    temporal = np.where(
        np.isfinite(temporal), temporal, generative.astype(np.float32)
    )
    alpha = low_coverage.astype(np.float32)
    if args.transition_sigma:
        alpha = cv2.GaussianBlur(alpha, (0, 0), args.transition_sigma)
        alpha[low_coverage] = 1.0
    hybrid = np.rint(
        generative.astype(np.float32) * alpha[..., None]
        + temporal * (1.0 - alpha[..., None])
    ).clip(0, 255).astype(np.uint8)

    clean_plate = output_dir / "hybrid-clean-plate.png"
    coverage = output_dir / "visible-sample-count.png"
    review = output_dir / "clean-plate-review.jpg"
    cv2.imwrite(str(clean_plate), hybrid)
    coverage_vis = np.rint(
        visible_count.astype(np.float32) / max(1, len(samples)) * 255.0
    ).astype(np.uint8)
    cv2.imwrite(str(coverage), coverage_vis)
    coverage_color = cv2.applyColorMap(coverage_vis, cv2.COLORMAP_VIRIDIS)
    panels = []
    for label, image in (
        ("generative fill", generative),
        ("authentic temporal median + fill", hybrid),
        ("source visibility", coverage_color),
    ):
        panel = image.copy()
        cv2.putText(
            panel,
            label,
            (14, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            label,
            (14, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(cv2.resize(panel, (416, 240)))
    cv2.imwrite(str(review), cv2.hconcat(panels), [cv2.IMWRITE_JPEG_QUALITY, 95])

    record.update(
        {
            "status": "completed",
            "honest_status": "WORKING",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in paths.items()
            },
            "video_info": info,
            "packages": {
                name: _package_version(name)
                for name in ("numpy", "opencv-contrib-python")
            },
            "sample_indices": sample_indices,
            "metrics": {
                "sample_count": len(samples),
                "zero_visibility_fraction": float(np.mean(visible_count == 0)),
                "low_coverage_fraction": float(np.mean(low_coverage)),
                "median_visible_samples": float(np.median(visible_count)),
            },
            "outputs": {
                "clean_plate": str(clean_plate),
                "clean_plate_sha256": file_sha256(clean_plate),
                "coverage": str(coverage),
                "coverage_sha256": file_sha256(coverage),
                "review": str(review),
                "review_sha256": file_sha256(review),
            },
            "limitations": [
                "Never-visible pixels still depend on the pinned generative clean plate.",
                "Temporal medians can suppress moving flowers inside the replacement region.",
            ],
        }
    )
    _write_json(manifest_path, record)
    print(json.dumps(record["metrics"], indent=2, sort_keys=True))
    print(clean_plate)
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except Exception as exc:
        try:
            args = _parser().parse_args()
            manifest_path = args.output_dir.expanduser().resolve() / "manifest.json"
            if manifest_path.is_file():
                record = json.loads(manifest_path.read_text())
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
