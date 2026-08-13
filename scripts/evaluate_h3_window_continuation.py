#!/usr/bin/env python3
"""Compare an independent H3 window with a continuation-conditioned repair."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.rendering.h3_long_video import overlap_continuity_metrics  # noqa: E402
from scripts.build_multi_anchor_robot_replacement import _git_state  # noqa: E402
from scripts.stitch_minimax_h3_long_flower import _align_mask, _decode  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--baseline-following", type=Path, required=True)
    parser.add_argument("--conditioned-following", type=Path, required=True)
    parser.add_argument("--safety-mask", type=Path, required=True)
    parser.add_argument("--previous-start", type=int, default=0)
    parser.add_argument("--following-start", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--human-review", choices=("pending", "passed", "failed"), default="pending")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"evaluation already exists: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "previous": args.previous.expanduser().resolve(),
        "baseline_following": args.baseline_following.expanduser().resolve(),
        "conditioned_following": args.conditioned_following.expanduser().resolve(),
        "safety_mask": args.safety_mask.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")

    import cv2
    import numpy as np

    previous, previous_info = _decode(cv2, paths["previous"])
    baseline, baseline_info = _decode(cv2, paths["baseline_following"])
    conditioned, conditioned_info = _decode(cv2, paths["conditioned_following"])
    if not (
        previous[0].shape == baseline[0].shape == conditioned[0].shape
        and len(previous) == len(baseline) == len(conditioned)
    ):
        raise RuntimeError("continuation ablation videos are not shape/frame aligned")
    height, width = previous[0].shape[:2]
    mask_raw = cv2.imread(str(paths["safety_mask"]), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise RuntimeError("cannot decode safety mask")
    mask = (_align_mask(cv2, mask_raw, width, height) >= 127).astype(np.uint8) * 255
    baseline_metrics = overlap_continuity_metrics(
        np,
        previous=previous,
        previous_start=args.previous_start,
        following=baseline,
        following_start=args.following_start,
        subject_mask=mask,
    )
    conditioned_metrics = overlap_continuity_metrics(
        np,
        previous=previous,
        previous_start=args.previous_start,
        following=conditioned,
        following_start=args.following_start,
        subject_mask=mask,
    )
    same_time_ratio = (
        conditioned_metrics["mean_same_time_subject_mad"]
        / baseline_metrics["mean_same_time_subject_mad"]
    )
    seam_ratio = (
        conditioned_metrics["best_seam_subject_mad"]
        / baseline_metrics["best_seam_subject_mad"]
    )
    overlap_start = int(conditioned_metrics["overlap_start"])
    overlap_end = int(conditioned_metrics["overlap_end_exclusive"])
    if overlap_end - overlap_start < 8:
        raise RuntimeError("overlap is too short for a three-frame review storyboard")
    selected_frames = (
        overlap_start + 4,
        (overlap_start + overlap_end) // 2,
        overlap_end - 4,
    )
    rows = []
    for absolute in selected_frames:
        items = (
            previous[absolute - args.previous_start],
            baseline[absolute - args.following_start],
            conditioned[absolute - args.following_start],
        )
        rows.append(np.hstack(items))
    storyboard = np.vstack(rows)
    cv2.imwrite(str(output_dir / "overlap-storyboard.jpg"), storyboard)
    automatic = {
        "mean_same_time_improved": same_time_ratio < 1.0,
        "best_seam_improved": seam_ratio < 1.0,
        "material_improvement": min(same_time_ratio, seam_ratio) <= 0.90,
    }
    review_passed = args.human_review == "passed"
    accepted = all(automatic.values()) and review_passed
    packages = {}
    for name in ("numpy", "opencv-contrib-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema_version": "1.0.0",
        "method": "h3_picture2_recursive_window_continuation_ablation",
        "status": "accepted" if accepted else "rejected" if args.human_review == "failed" else "review_required",
        "honest_status": "WORKING" if accepted else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "packages": packages,
        "gpu": {"used": False, "cuda_visible_devices": None, "reason": "CPU overlap evaluation"},
        "git": _git_state(PROJECT_ROOT),
        "execution_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in paths.items()
        },
        "video_info": {
            "previous": previous_info,
            "baseline_following": baseline_info,
            "conditioned_following": conditioned_info,
        },
        "coordinate_frame": "camera:H3_output_pixels in shared absolute source frame indices",
        "baseline": baseline_metrics,
        "conditioned": conditioned_metrics,
        "improvement_ratios": {
            "mean_same_time": same_time_ratio,
            "best_seam": seam_ratio,
        },
        "acceptance": {**automatic, "human_review": args.human_review},
        "outputs": {"storyboard": str(output_dir / "overlap-storyboard.jpg")},
        "limitations": [
            "This ablation measures only the first overlap and does not prove full-video continuity.",
            "The conservative static person-union mask includes some nearby workspace pixels.",
            "Pixel MAD and three-frame review do not replace dense full-video human review.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
