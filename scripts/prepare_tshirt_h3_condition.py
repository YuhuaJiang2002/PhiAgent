#!/usr/bin/env python3
"""Prepare a static-scene H3 condition for language-planned T-shirt folding."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from phiagent.acwm.schema import (  # noqa: E402
    ACWMActionCondition,
    ActionRepresentation,
)
from phiagent.harness.provenance import capture_provenance, write_json_atomic  # noqa: E402
from phiagent.harness.task_reasoning import (  # noqa: E402
    TSHIRT_FOLD_TASK,
    TaskReasoningPlan,
)
from phiagent.rendering.minimax_h3 import file_sha256  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-frame", type=Path, required=True)
    parser.add_argument("--task-reasoning-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--fps", type=float, default=24.0)
    return parser


def _phase_progress(timestamp: float, start: float, end: float) -> float:
    if timestamp <= start:
        return 0.0
    if timestamp >= end:
        return 1.0
    return (timestamp - start) / (end - start)


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    first_frame = args.first_frame.expanduser().resolve()
    plan_path = args.task_reasoning_plan.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"condition output already exists: {output_dir}")
    if args.width <= 0 or args.height <= 0 or args.frames < 2 or args.fps <= 0:
        raise ValueError("width, height, frames, and fps must be positive")
    with Image.open(first_frame) as image:
        if image.size != (args.width, args.height):
            raise ValueError(
                f"first frame is {image.size}, expected {(args.width, args.height)}"
            )
    plan_payload = json.loads(plan_path.read_text())
    if not isinstance(plan_payload, dict):
        raise ValueError("task reasoning plan must contain one JSON object")
    plan = TaskReasoningPlan.from_dict(plan_payload)
    if plan.task_type != TSHIRT_FOLD_TASK:
        raise ValueError("T-shirt condition requires a T-shirt reasoning plan")
    if plan.coordinate_frame != "camera:tshirt_fold_832x480_pixels":
        raise ValueError("unexpected T-shirt camera frame")
    expected_duration = args.frames / args.fps
    if abs(plan.duration_seconds - expected_duration) > 1e-5:
        raise ValueError("reasoning plan duration does not match the H3 frame contract")

    control_dir = output_dir / "control"
    input_dir = output_dir / "input"
    control_dir.mkdir(parents=True)
    input_dir.mkdir()
    frozen_frame = control_dir / "00-source.png"
    shutil.copy2(first_frame, frozen_frame)
    hold_video = control_dir / "static-scene-reference.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to build the static H3 reference video")
    ffmpeg_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(args.fps),
        "-i",
        str(frozen_frame),
        "-frames:v",
        str(args.frames),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "12",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(hold_video),
    ]
    subprocess.run(ffmpeg_command, check=True)
    timestamps = tuple(index / args.fps for index in range(args.frames))
    channels = tuple(f"phase_progress:{phase.phase_id}" for phase in plan.phases)
    values = tuple(
        tuple(
            _phase_progress(timestamp, phase.start_seconds, phase.end_seconds)
            for phase in plan.phases
        )
        for timestamp in timestamps
    )
    condition_path = control_dir / "action-condition.json"
    condition = ACWMActionCondition(
        label="fold-shirt-left-right-pack-aside-physical-v1",
        instruction=(
            "Execute the hash-bound physical task plan continuously: viewer-left sleeve, "
            "viewer-right sleeve, body fold, then move the completed bundle viewer-left."
        ),
        timeline="task-reasoning-plan phase progress at 24 FPS",
        representation=ActionRepresentation.CAMERA_PIXEL_CONTROL_VIDEO,
        coordinate_frame=plan.coordinate_frame,
        timestamps_s=timestamps,
        channels=channels,
        values=values,
        visual_condition=hold_video,
    )
    condition.to_json(condition_path)
    trajectory_path = control_dir / "trajectory.json"
    write_json_atomic(
        trajectory_path,
        {
            "schema_version": "1.0.0",
            "coordinate_frame": plan.coordinate_frame,
            "representation": "language_phase_progress",
            "timestamps_s": list(timestamps),
            "channels": list(channels),
            "values": [list(row) for row in values],
            "task_reasoning_plan": str(plan_path),
            "task_reasoning_plan_sha256": file_sha256(plan_path),
            "plan_sha256": plan.plan_sha256,
        },
    )
    prompt = """[multi-reference robot manipulation video generation]
<Picture 1> is the exact two-white-robot embodiment and gripper appearance reference.
<Picture 2> is the exact real camera, table, T-shirt, background, lighting, and initial-state reference.
<Video 1> is a static scene/camera identity reference only. It deliberately contains no target motion and no edited target states. Do not copy its stillness; synthesize the new continuous manipulation from the appended hash-bound task plan.

Generate one uninterrupted physically plausible shot. The same two robot arms establish visible cloth contact, fold the viewer-left sleeve continuously, settle it, fold the viewer-right sleeve continuously, settle it, fold the lower body upward into one compact layered rectangle, then move that completed bundle to the viewer-left side and hold. Preserve one shirt and all material identities. Never use a cut, dissolve, crossfade, teleport, object swap, sleeve shrink, sleeve growth, or keyframe morph. Do not render text, guide marks, colored overlays, humans, or extra limbs."""
    prompt_path = control_dir / "prompt.txt"
    prompt_path.write_text(prompt)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        **capture_provenance(
            project_root,
            [sys.executable, *sys.argv],
            args.seed,
        ),
        "status": "condition_compiled",
        "honest_status": "NOT STARTED",
        "method": "static_scene_reference_plus_hash_bound_physical_language_plan",
        "representation": "camera_pixel_control_video",
        "claim_boundary": plan.claim_boundary,
        "first_frame": str(frozen_frame.relative_to(output_dir)),
        "first_frame_sha256": file_sha256(frozen_frame),
        "source_video": str(hold_video.relative_to(output_dir)),
        "source_video_sha256": file_sha256(hold_video),
        "auxiliary_inputs": {
            "embodiment_reference": str(frozen_frame.relative_to(output_dir))
        },
        "auxiliary_input_sha256": {
            "embodiment_reference": file_sha256(frozen_frame)
        },
        "task_reasoning_plan": str(plan_path),
        "task_reasoning_plan_sha256": file_sha256(plan_path),
        "plan_sha256": plan.plan_sha256,
        "ffmpeg_command": ffmpeg_command,
        "variants": [
            {
                "label": condition.label,
                "instruction": condition.instruction,
                "condition": str(condition_path.relative_to(output_dir)),
                "condition_sha256": file_sha256(condition_path),
                "prompt": prompt,
                "prompt_file": str(prompt_path.relative_to(output_dir)),
                "prompt_sha256": file_sha256(prompt_path),
                "control_video": str(hold_video.relative_to(output_dir)),
                "control_video_sha256": file_sha256(hold_video),
                "trajectory": str(trajectory_path.relative_to(output_dir)),
                "trajectory_sha256": file_sha256(trajectory_path),
                "auxiliary_inputs": {},
            }
        ],
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "plan_sha256": plan.plan_sha256}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
