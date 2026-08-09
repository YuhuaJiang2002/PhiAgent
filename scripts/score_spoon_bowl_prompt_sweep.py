#!/usr/bin/env python3
"""Score spoon-bowl prompt candidates before manual physical review."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.evaluation.object_instance import (  # noqa: E402
    NormalizedROI,
    ObjectTrackerConfig,
    RGBFrames,
    decode_video,
    track_colored_object,
)
from phiagent.evaluation.video_proxy import (  # noqa: E402
    _temporal_jerk,
    decode_grayscale,
)


def _ffmpeg_metric(
    ffmpeg: Path,
    first: Path,
    second: Path,
    filter_graph: str,
    pattern: str,
    *,
    loop_first: bool = False,
) -> float:
    command = [str(ffmpeg), "-v", "info"]
    if loop_first:
        command.extend(["-loop", "1", "-framerate", "12"])
    command.extend(
        [
            "-i",
            str(first),
            "-i",
            str(second),
            "-frames:v",
            "49",
            "-lavfi",
            filter_graph,
            "-f",
            "null",
            "-",
        ]
    )
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    matches = re.findall(pattern, completed.stderr)
    if not matches:
        raise ValueError(f"ffmpeg metric not found for {second}")
    return float(matches[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "VIDEO"))
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.candidate:
        raise ValueError("at least one candidate is required")
    reference = args.reference.expanduser().resolve()
    control = args.control.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if not reference.is_file() or not control.is_file() or not ffmpeg.is_file():
        raise ValueError("reference, control, and ffmpeg must exist")

    results: dict[str, object] = {}
    for name, raw_path in args.candidate:
        candidate = Path(raw_path).expanduser().resolve()
        scene_ssim = _ffmpeg_metric(
            ffmpeg,
            reference,
            candidate,
            "ssim",
            r"SSIM .* All:([0-9.]+)",
            loop_first=True,
        )
        control_edge_ssim = _ffmpeg_metric(
            ffmpeg,
            control,
            candidate,
            "[0:v]format=gray[c];[1:v]format=gray,edgedetect[o];[c][o]ssim",
            r"SSIM .* All:([0-9.]+)",
        )
        decoded_gray = decode_grayscale(
            candidate,
            ffmpeg,
            width=112,
            height=64,
            sample_fps=12,
            maximum_seconds=5,
        )
        jerk = _temporal_jerk(decoded_gray)
        temporal_score = math.exp(-32 * jerk)
        frames = RGBFrames(
            decode_video(
                candidate,
                ffmpeg,
                width=224,
                height=128,
                fps=12,
                frame_num=49,
                pixel_format="rgb24",
            ),
            224,
            128,
        )
        try:
            track = track_colored_object(
                frames,
                ObjectTrackerConfig(
                    NormalizedROI(0.08, 0.32, 0.84, 0.64),
                    initial_color_mode="cyan",
                    minimum_chroma=10,
                    chroma_tolerance=42,
                    brightness_tolerance=120,
                    search_margin=0.25,
                    minimum_component_pixels=10,
                ),
            )
            tracked_frames = sum(box is not None for box in track.boxes)
            coverage = tracked_frames / len(track.boxes)
            positive_areas = [area for area in track.areas if area]
            area_ratio = max(positive_areas) / min(positive_areas)
            shape_stability = min(1.0, 3.0 / area_ratio)
        except ValueError:
            tracked_frames = 0
            coverage = 0.0
            area_ratio = None
            shape_stability = 0.0
        automatic_score = (
            0.30 * scene_ssim
            + 0.20 * control_edge_ssim
            + 0.25 * temporal_score
            + 0.15 * coverage
            + 0.10 * shape_stability
        )
        results[name] = {
            "path": str(candidate),
            "scene_ssim": scene_ssim,
            "control_edge_ssim": control_edge_ssim,
            "temporal_jerk": jerk,
            "temporal_score": temporal_score,
            "spoon_tracked_frames": tracked_frames,
            "spoon_tracking_coverage": coverage,
            "spoon_area_ratio": area_ratio,
            "spoon_shape_stability": shape_stability,
            "automatic_score": automatic_score,
        }
    ranking = sorted(
        results,
        key=lambda name: float(results[name]["automatic_score"]),
        reverse=True,
    )
    payload = {
        "schema_version": "1.0.0",
        "ranking": ranking,
        "results": results,
        "selection_rule": (
            "Automatic score is diagnostic only; candidates must also pass manual "
            "same-robot-arm, bowl-table-support, locked-grasp, single-object, and "
            "final-containment gates."
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
