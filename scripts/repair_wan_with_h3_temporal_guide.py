#!/usr/bin/env python3
"""Use MiniMax-H3 motion as a guide for bounded Wan temporal repair."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.repair_video_transition_spikes import (  # noqa: E402
    _decode,
    _sha256,
    _transition_energy,
    _write_json,
    _writer,
    group_transition_frames,
    intervals_overlap,
)


def local_transition_ratios(np: Any, energy: list[float], radius: int = 6) -> list[float]:
    """Normalize every transition against neighboring transitions."""

    if radius < 1:
        raise ValueError("local ratio radius must be positive")
    values = np.asarray(energy, dtype=np.float64)
    result = []
    for index, value in enumerate(values):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        neighbors = np.concatenate((values[left:index], values[index + 1 : right]))
        baseline = float(np.median(neighbors)) if len(neighbors) else float(value)
        result.append(float(value) / max(baseline, 0.1))
    return result


def detect_guided_anomalies(
    candidate_energy: list[float],
    guide_energy: list[float],
    source_energy: list[float],
    *,
    analysis_end_frame: int,
    repair_radius: int,
    minimum_guide_score: float,
    minimum_candidate_ratio: float,
    minimum_candidate_energy: float,
) -> tuple[list[int], list[dict[str, float | int]]]:
    """Find Wan transition outliers that H3 and the source do not support."""

    import numpy as np

    if not (len(candidate_energy) == len(guide_energy) == len(source_energy)):
        raise ValueError("transition-energy arrays must have equal lengths")
    candidate_ratio = local_transition_ratios(np, candidate_energy)
    guide_ratio = local_transition_ratios(np, guide_energy)
    source_ratio = local_transition_ratios(np, source_energy)
    selected = []
    diagnostics = []
    maximum = min(analysis_end_frame, len(candidate_energy))
    for transition_frame in range(repair_radius, maximum + 1):
        index = transition_frame - 1
        expected_ratio = max(0.5, (guide_ratio[index] + source_ratio[index]) / 2.0)
        guide_score = candidate_ratio[index] / expected_ratio
        item = {
            "transition_frame": transition_frame,
            "candidate_energy": float(candidate_energy[index]),
            "guide_energy": float(guide_energy[index]),
            "source_energy": float(source_energy[index]),
            "candidate_local_ratio": float(candidate_ratio[index]),
            "guide_local_ratio": float(guide_ratio[index]),
            "source_local_ratio": float(source_ratio[index]),
            "guide_score": float(guide_score),
        }
        if (
            guide_score >= minimum_guide_score
            and candidate_ratio[index] >= minimum_candidate_ratio
            and candidate_energy[index] >= minimum_candidate_energy
        ):
            selected.append(transition_frame)
            diagnostics.append(item)
    return selected, diagnostics


def merge_overlapping_repair_groups(
    groups: list[tuple[int, int]], radius: int
) -> list[tuple[int, int]]:
    """Merge groups only when their replaced frame intervals overlap."""

    if radius < 1:
        raise ValueError("repair radius must be positive")
    merged: list[tuple[int, int]] = []
    for start, end in groups:
        if not merged:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        previous_right_endpoint = previous_end + radius
        current_left_endpoint = start - radius
        if current_left_endpoint < previous_right_endpoint:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def guided_progress(
    np: Any,
    guide_ratios: list[float],
    source_ratios: list[float],
) -> list[float]:
    """Convert H3/source motion weights into monotonic interpolation progress."""

    if len(guide_ratios) != len(source_ratios) or len(guide_ratios) < 2:
        raise ValueError("guide and source weights must have equal length >= 2")
    weights = np.maximum(
        0.05,
        0.5 * np.asarray(guide_ratios, dtype=np.float64)
        + 0.5 * np.asarray(source_ratios, dtype=np.float64),
    )
    cumulative = np.cumsum(weights)
    return [float(value / cumulative[-1]) for value in cumulative[:-1]]


def guided_crossfade(
    np: Any,
    first: Any,
    second: Any,
    progress: list[float],
) -> list[Any]:
    """Crossfade Wan endpoints according to H3/source motion timing."""

    if first.shape != second.shape:
        raise ValueError("bridge endpoints must have equal shapes")
    result = []
    previous = 0.0
    for value in progress:
        if not previous < value < 1.0:
            raise ValueError("guided progress must be strictly increasing in (0, 1)")
        previous = value
        alpha = 0.5 - 0.5 * math.cos(math.pi * value)
        result.append(
            np.clip(
                np.rint(
                    first.astype(np.float32) * (1.0 - alpha)
                    + second.astype(np.float32) * alpha
                ),
                0,
                255,
            ).astype(np.uint8)
        )
    return result


def _subject_transition_energy(
    cv2: Any,
    np: Any,
    frames: list[Any],
) -> list[float]:
    gray = [
        cv2.cvtColor(
            cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        for frame in frames
    ]
    left, right = round(256 * 0.44), round(256 * 0.94)
    return [
        float(np.mean(cv2.absdiff(gray[index], gray[index - 1])[:, left:right]))
        for index in range(1, len(gray))
    ]


def _summary(np: Any, energy: list[float], end_frame: int) -> dict[str, float]:
    selected = energy[:end_frame]
    median = float(np.median(selected))
    maximum = max(selected)
    return {
        "median": median,
        "maximum": maximum,
        "maximum_ratio": maximum / max(median, 1e-6),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--h3-guide", type=Path, required=True)
    parser.add_argument("--h3-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-end-frame", type=int, default=236)
    parser.add_argument("--repair-radius", type=int, default=2)
    parser.add_argument("--minimum-guide-score", type=float, default=2.0)
    parser.add_argument("--minimum-candidate-ratio", type=float, default=1.8)
    parser.add_argument("--minimum-candidate-energy", type=float, default=1.0)
    parser.add_argument("--protected-start-frame", type=int, default=259)
    parser.add_argument("--protected-end-frame-exclusive", type=int, default=297)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument(
        "--human-review",
        choices=("pending", "passed", "failed"),
        default="pending",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "candidate": args.candidate.expanduser().resolve(),
        "h3_guide": args.h3_guide.expanduser().resolve(),
        "h3_manifest": args.h3_manifest.expanduser().resolve(),
        "source": args.source.expanduser().resolve(),
        "ffmpeg": args.ffmpeg.expanduser().resolve(),
    }
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"hybrid output already exists: {manifest_path}")
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} is missing or empty: {path}")
    if args.repair_radius < 1:
        raise ValueError("repair radius must be positive")
    if not 0 < args.analysis_end_frame < args.protected_start_frame:
        raise ValueError("analysis range must end before protected frames")
    if not (
        0
        <= args.protected_start_frame
        < args.protected_end_frame_exclusive
    ):
        raise ValueError("protected frame range is invalid")
    output_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    candidate, candidate_info = _decode(cv2, paths["candidate"])
    h3_guide, h3_info = _decode(cv2, paths["h3_guide"])
    source, source_info = _decode(cv2, paths["source"])
    if not len(candidate) == len(h3_guide) == len(source):
        raise RuntimeError("candidate, H3 guide, and source frame counts differ")
    if args.analysis_end_frame >= len(candidate):
        raise ValueError("analysis end frame falls outside the video")

    candidate_subject = _subject_transition_energy(cv2, np, candidate)
    h3_subject = _subject_transition_energy(cv2, np, h3_guide)
    source_subject = _subject_transition_energy(cv2, np, source)
    selected, diagnostics = detect_guided_anomalies(
        candidate_subject,
        h3_subject,
        source_subject,
        analysis_end_frame=args.analysis_end_frame,
        repair_radius=args.repair_radius,
        minimum_guide_score=args.minimum_guide_score,
        minimum_candidate_ratio=args.minimum_candidate_ratio,
        minimum_candidate_energy=args.minimum_candidate_energy,
    )
    if not selected:
        raise RuntimeError("H3/source consensus selected no early Wan anomalies")
    groups = merge_overlapping_repair_groups(
        group_transition_frames(selected), args.repair_radius
    )

    h3_ratio = local_transition_ratios(np, h3_subject)
    source_ratio = local_transition_ratios(np, source_subject)
    repaired = [frame.copy() for frame in candidate]
    repairs = []
    for transition_start, transition_end in groups:
        left = transition_start - args.repair_radius
        right = transition_end + args.repair_radius
        if left < 0 or right >= len(repaired):
            raise ValueError("repair interval falls outside the video")
        if intervals_overlap(
            left + 1,
            right,
            args.protected_start_frame,
            args.protected_end_frame_exclusive,
        ):
            raise ValueError("repair interval overlaps the protected anchor")
        progress = guided_progress(
            np,
            h3_ratio[left:right],
            source_ratio[left:right],
        )
        intermediate = guided_crossfade(
            np,
            candidate[left],
            candidate[right],
            progress,
        )
        repaired[left + 1 : right] = intermediate
        repairs.append(
            {
                "transition_start_frame": transition_start,
                "transition_end_frame": transition_end,
                "left_endpoint_frame": left,
                "right_endpoint_frame": right,
                "replaced_start_frame": left + 1,
                "replaced_end_frame_exclusive": right,
                "guided_progress": progress,
            }
        )

    before_full = _transition_energy(cv2, np, candidate)
    after_full = _transition_energy(cv2, np, repaired)
    after_subject = _subject_transition_energy(cv2, np, repaired)
    output = output_dir / "wan-h3-temporal-guide-early-repaired.mp4"
    writer = _writer(
        paths["ffmpeg"],
        output,
        int(candidate_info["width"]),
        int(candidate_info["height"]),
        float(candidate_info["fps"]),
    )
    try:
        assert writer.stdin is not None
        for frame in repaired:
            writer.stdin.write(frame.tobytes())
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
        if writer.wait():
            raise RuntimeError("FFmpeg failed to encode H3-guided repair")
    subprocess.run(
        [str(paths["ffmpeg"]), "-v", "error", "-i", str(output), "-f", "null", "-"],
        check=True,
    )
    encoded, encoded_info = _decode(cv2, output)
    if len(encoded) != len(candidate):
        raise RuntimeError("encoded output changed the frame count")
    encoded_full = _transition_energy(cv2, np, encoded)
    encoded_subject = _subject_transition_energy(cv2, np, encoded)

    selected_set = sorted(selected)
    metrics = {
        "analysis_end_frame": args.analysis_end_frame,
        "full_frame_before": _summary(np, before_full, len(before_full)),
        "full_frame_preencode_after": _summary(np, after_full, len(after_full)),
        "full_frame_encoded_after": _summary(np, encoded_full, len(encoded_full)),
        "early_subject_before": _summary(
            np, candidate_subject, args.analysis_end_frame
        ),
        "early_subject_preencode_after": _summary(
            np, after_subject, args.analysis_end_frame
        ),
        "early_subject_encoded_after": _summary(
            np, encoded_subject, args.analysis_end_frame
        ),
        "selected_subject_energy_before": {
            str(frame): candidate_subject[frame - 1] for frame in selected_set
        },
        "selected_subject_energy_encoded_after": {
            str(frame): encoded_subject[frame - 1] for frame in selected_set
        },
    }
    h3_manifest = json.loads(paths["h3_manifest"].read_text())
    packages = {}
    for package in ("numpy", "opencv-python", "opencv-python-headless"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    review_passed = args.human_review == "passed"
    manifest = {
        "schema_version": "1.0.0",
        "method": "wan_pixels_h3_source_consensus_timed_local_bridge_v1",
        "status": (
            "accepted"
            if review_passed
            else "rejected"
            if args.human_review == "failed"
            else "review_required"
        ),
        "honest_status": "WORKING" if review_passed else "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "packages": packages,
        "inputs": {
            "candidate": {
                "path": str(paths["candidate"]),
                "sha256": _sha256(paths["candidate"]),
                "info": candidate_info,
            },
            "h3_guide": {
                "path": str(paths["h3_guide"]),
                "sha256": _sha256(paths["h3_guide"]),
                "info": h3_info,
            },
            "h3_manifest": {
                "path": str(paths["h3_manifest"]),
                "sha256": _sha256(paths["h3_manifest"]),
                "acceptance": h3_manifest.get("acceptance"),
                "selected_scorecard": h3_manifest.get("selected_scorecard"),
            },
            "source": {
                "path": str(paths["source"]),
                "sha256": _sha256(paths["source"]),
                "info": source_info,
            },
        },
        "detector": {
            "subject_x_fraction": [0.44, 0.94],
            "local_ratio_radius": 6,
            "analysis_end_frame": args.analysis_end_frame,
            "repair_radius": args.repair_radius,
            "minimum_guide_score": args.minimum_guide_score,
            "minimum_candidate_ratio": args.minimum_candidate_ratio,
            "minimum_candidate_energy": args.minimum_candidate_energy,
            "selected_transition_frames": selected_set,
            "selected_diagnostics": diagnostics,
        },
        "repairs": repairs,
        "protected_frame_range": [
            args.protected_start_frame,
            args.protected_end_frame_exclusive,
        ],
        "protected_range_unchanged_preencode": all(
            np.array_equal(candidate[index], repaired[index])
            for index in range(
                args.protected_start_frame,
                args.protected_end_frame_exclusive,
            )
        ),
        "metrics": metrics,
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
            "info": encoded_info,
        },
        "human_review": args.human_review,
        "limitations": [
            "MiniMax-H3 is used only as a temporal anomaly and timing guide; no H3 pixels are copied into the Wan output.",
            "The reviewed H3 guide failed its robot-identity gate and is not itself an accepted replacement video.",
            "Endpoint crossfades can ghost fast motion and require consecutive-frame human review at every repaired interval.",
            "This repair does not establish exact contact, physics, kinematics, or robot execution.",
        ],
    }
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "selected": selected_set,
                "groups": groups,
                "metrics": metrics,
                "status": manifest["status"],
            },
            indent=2,
        )
    )
    return 0 if review_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
