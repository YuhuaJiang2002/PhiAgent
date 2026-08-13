#!/usr/bin/env python3
"""Validate and promote an immutable pose-rig candidate after human review."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_multi_anchor_robot_replacement import (  # noqa: E402
    _git_state,
    _sha256,
    _write_json,
)


def _transition_statistics(np: Any, energies: list[float], threshold: float) -> dict[str, Any]:
    values = np.asarray(energies, dtype=np.float64)
    median = float(np.median(values))
    ratios = values / max(median, 1e-9)
    maximum_index = int(np.argmax(ratios)) + 1
    return {
        "median_energy": median,
        "maximum_ratio": float(np.max(ratios)),
        "maximum_ratio_transition_to_frame": maximum_index,
        "threshold": threshold,
        "outlier_count": int(np.count_nonzero(ratios > threshold)),
        "outlier_transition_to_frames": (np.flatnonzero(ratios > threshold) + 1).tolist(),
    }


def _measure_video(cv2: Any, np: Any, video: Path, mask_path: Path) -> dict[str, Any]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"cannot decode ROI mask: {mask_path}")
    mask = cv2.resize(mask, (256, 144), interpolation=cv2.INTER_NEAREST) > 0
    capture = cv2.VideoCapture(str(video))
    previous = None
    full_energy: list[float] = []
    roi_energy: list[float] = []
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (256, 144),
                interpolation=cv2.INTER_AREA,
            )
            if previous is not None:
                difference = cv2.absdiff(gray, previous)
                full_energy.append(float(np.mean(difference)))
                roi_energy.append(float(np.mean(difference[mask])))
            previous = gray
            frames += 1
    finally:
        capture.release()
    return {
        "decoded_frames": frames,
        "full_frame": _transition_statistics(np, full_energy, 4.0),
        "person_roi": _transition_statistics(np, roi_energy, 4.0),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--human-review", choices=("passed", "failed"), required=True)
    parser.add_argument("--review-notes", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    validation_config = json.loads(config_path.read_text())
    source_experiment = args.source_experiment.expanduser().resolve()
    experiment = args.experiment_dir.expanduser().resolve()
    source_manifest_path = source_experiment / "final" / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_video = Path(source_manifest["outputs"]["video"])
    source_video_hash = _sha256(source_video)
    if source_video_hash != source_manifest["outputs"]["video_sha256"]:
        raise RuntimeError("source video hash no longer matches its manifest")
    source_config = source_manifest["resolved_config"]
    mask_path = Path(source_config["safety_mask"])
    peak_review = source_experiment / "final" / "peak-transition-consecutive-review.jpg"
    if not peak_review.is_file():
        raise ValueError(f"missing peak transition review: {peak_review}")

    experiment.mkdir(parents=True, exist_ok=True)
    final = experiment / "final"
    final.mkdir(exist_ok=True)
    accepted_video = final / "robot-motion-replacement-rigged.mp4"
    shutil.copy2(source_video, accepted_video)
    copied_peak_review = final / peak_review.name
    shutil.copy2(peak_review, copied_peak_review)
    review_names = (
        "early-consecutive-review.jpg",
        "dense-review.jpg",
        "storyboard-16.jpg",
        "early-hand-orientation-review.jpg",
        "roi-peak-consecutive-review.jpg",
        "hand-angle-late-review.jpg",
    )
    for name in review_names:
        source_review = source_experiment / "final" / name
        if source_review.is_file():
            shutil.copy2(source_review, final / name)

    import cv2
    import numpy as np

    measurement = _measure_video(cv2, np, accepted_video, mask_path)
    subprocess.run(
        [str(args.ffmpeg), "-v", "error", "-i", str(accepted_video), "-f", "null", "-"],
        check=True,
    )
    inherited_acceptance = {
        key: value
        for key, value in source_manifest["acceptance"].items()
        if key not in ("human_review_passed", "hand_orientation_step_bounded")
    }
    source_hand_orientation = source_manifest["metrics"].get(
        "hand_orientation", {}
    )
    configured_hand_angle_limit = float(
        source_manifest["resolved_config"].get(
            "maximum_hand_angle_step_degrees", float("inf")
        )
    )
    hand_angle_tolerance = float(
        validation_config.get("hand_angle_float_tolerance_degrees", 1e-9)
    )
    hand_orientation_step_bounded = bool(source_hand_orientation) and all(
        float(record["maximum_hand_angle_step_degrees"])
        <= configured_hand_angle_limit + hand_angle_tolerance
        for record in source_hand_orientation.values()
    )
    acceptance = {
        **inherited_acceptance,
        "hand_orientation_step_bounded": hand_orientation_step_bounded,
        "source_automated_acceptance_passed": all(inherited_acceptance.values())
        and hand_orientation_step_bounded,
        "video_hash_preserved": _sha256(accepted_video) == source_video_hash,
        "frame_count_preserved": measurement["decoded_frames"]
        == source_manifest["source_video"]["frames"],
        "zero_full_frame_transition_outliers": measurement["full_frame"]["outlier_count"]
        == 0,
        "zero_person_roi_transition_outliers": measurement["person_roi"]["outlier_count"]
        == 0,
        "peak_transition_review_present": copied_peak_review.is_file(),
        "human_review_passed": args.human_review == "passed",
    }
    accepted = all(acceptance.values())
    manifest = {
        "schema_version": "1.0.0",
        "status": "accepted" if accepted else "rejected",
        "honest_status": "WORKING" if accepted else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "immutable_candidate_promotion_after_transition_and_human_review",
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            "numpy": importlib.metadata.version("numpy"),
            "opencv-contrib-python": importlib.metadata.version("opencv-contrib-python"),
        },
        "gpu": {"used": False, "reason": "CPU-only immutable video validation"},
        "git": _git_state(PROJECT_ROOT),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "resolved_config": validation_config,
        "source_experiment": str(source_experiment),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_video_sha256": source_video_hash,
        "review_notes": args.review_notes,
        "measurement": measurement,
        "acceptance": acceptance,
        "outputs": {
            "video": str(accepted_video),
            "video_sha256": _sha256(accepted_video),
            "early_consecutive_review": str(final / "early-consecutive-review.jpg"),
            "dense_review": str(final / "dense-review.jpg"),
            "peak_transition_review": str(copied_peak_review),
            "early_hand_orientation_review": str(
                final / "early-hand-orientation-review.jpg"
            ),
            "roi_peak_review": str(final / "roi-peak-consecutive-review.jpg"),
            "late_hand_angle_review": str(final / "hand-angle-late-review.jpg"),
        },
        "scope": {
            "working": "Zero detected artificial discontinuities, exact same-frame 2D shoulder/elbow/wrist retargeting, exact hand-root placement, and morphology-locked hand rendering under the declared tests.",
            "not_claimed": "A universal perceptual 100% guarantee, 3D depth, finger articulation, contact force, or real-robot execution.",
        },
    }
    _write_json(experiment / "trace.json", manifest)
    _write_json(final / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
