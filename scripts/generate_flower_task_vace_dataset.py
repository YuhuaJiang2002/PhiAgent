#!/usr/bin/env python3
"""Generate paired synthetic bimanual flower-contact clips for a VACE task LoRA.

The target clips are derived from the pinned robot render, not from the real
evaluation video.  One hand continuously holds a bouquet while the other
approaches, grasps, moves, releases, and retracts from one explicit flower.
Every clip persists its object trajectory and contact error so the dataset tests
causal contact rather than only robot appearance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.agent.flower_task_adaptation import HandPhase  # noqa: E402
from phiagent.data.adaptation import (  # noqa: E402
    AdaptationArm,
    AdaptationAsset,
    AdaptationAssetKind,
    AdaptationManifest,
    AdaptationSplit,
    VaceTrainingExample,
    file_sha256,
)


SHARPA_ASSET_REVISION = "6eea427eb24189519f32b9f21674cd534d3f973c"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-layer-video", type=Path, required=True)
    parser.add_argument("--wrist-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--train-clips", type=int, default=12)
    parser.add_argument("--validation-clips", type=int, default=4)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--source-frame-step", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser


def canonical_contact_phases(frame_count: int) -> tuple[HandPhase, ...]:
    """Return a complete approach/grasp/manipulate/release/retract schedule."""

    if frame_count < 9 or (frame_count - 1) % 4:
        raise ValueError("task clips require frame_count = 4n+1 and at least 9 frames")
    approach_end = max(1, round(frame_count * 0.24))
    grasp_end = max(approach_end + 1, round(frame_count * 0.35))
    manipulate_end = max(grasp_end + 1, round(frame_count * 0.71))
    release_end = max(manipulate_end + 1, round(frame_count * 0.82))
    release_end = min(release_end, frame_count - 1)
    result = []
    for frame in range(frame_count):
        if frame < approach_end:
            result.append(HandPhase.APPROACH)
        elif frame < grasp_end:
            result.append(HandPhase.GRASP)
        elif frame < manipulate_end:
            result.append(HandPhase.MANIPULATE)
        elif frame < release_end:
            result.append(HandPhase.RELEASE)
        else:
            result.append(HandPhase.RETRACT)
    return tuple(result)


def causal_flower_grip_trajectory(np: Any, hand_xy: Any, phases: tuple[HandPhase, ...]) -> Any:
    """Keep the flower fixed when free and exactly attached during contact."""

    if hand_xy.shape != (len(phases), 2):
        raise ValueError("hand trajectory must have shape [frames, 2]")
    contact_indices = [
        index
        for index, phase in enumerate(phases)
        if phase in {HandPhase.GRASP, HandPhase.MANIPULATE, HandPhase.RELEASE}
    ]
    if not contact_indices:
        raise ValueError("contact schedule contains no contact frames")
    first, last = contact_indices[0], contact_indices[-1]
    result = np.empty_like(hand_xy, dtype=np.float32)
    result[:first] = hand_xy[first]
    result[first : last + 1] = hand_xy[first : last + 1]
    result[last + 1 :] = hand_xy[last]
    return result


def _decode_video(path: Path, ffmpeg: Path, width: int, height: int) -> Any:
    import numpy as np

    completed = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"scale={width}:{height}:flags=lanczos",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    frame_size = width * height * 3
    if len(completed.stdout) < frame_size or len(completed.stdout) % frame_size:
        raise ValueError("robot layer decoder returned an invalid RGB byte count")
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(-1, height, width, 3)


def _encode_video(frames: Any, path: Path, ffmpeg: Path, fps: int) -> None:
    process = subprocess.Popen(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{frames.shape[2]}x{frames.shape[1]}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.write(frames.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed to encode {path}")


def _background(np: Any, rng: Any, width: int, height: int) -> Any:
    yy, _ = np.mgrid[0:height, 0:width]
    top = rng.integers(155, 220, size=3)
    bottom = rng.integers(85, 155, size=3)
    gradient = (
        top[None, None, :] * (1 - yy[..., None] / max(1, height - 1))
        + bottom[None, None, :] * (yy[..., None] / max(1, height - 1))
    )
    texture = rng.normal(0, 4, size=(height, width, 1))
    result = np.clip(gradient + texture, 0, 255).astype(np.uint8)
    table_y = round(height * 0.72)
    result[table_y:] = np.clip(result[table_y:].astype(np.int16) - 35, 0, 255)
    return result.astype(np.uint8)


def _draw_flower(
    cv2: Any,
    canvas: Any,
    grip_xy: Any,
    *,
    length: float,
    angle: float,
    stem_color: tuple[int, int, int],
    petal_color: tuple[int, int, int],
    thickness: int = 3,
) -> tuple[tuple[int, int], tuple[int, int]]:
    grip = (round(float(grip_xy[0])), round(float(grip_xy[1])))
    direction = (math.sin(angle), -math.cos(angle))
    head = (
        round(grip[0] + direction[0] * length * 0.58),
        round(grip[1] + direction[1] * length * 0.58),
    )
    base = (
        round(grip[0] - direction[0] * length * 0.42),
        round(grip[1] - direction[1] * length * 0.42),
    )
    cv2.line(canvas, base, head, stem_color, thickness, cv2.LINE_AA)
    for petal_index in range(6):
        theta = petal_index * math.tau / 6
        center = (
            round(head[0] + math.cos(theta) * 4),
            round(head[1] + math.sin(theta) * 4),
        )
        cv2.circle(canvas, center, 4, petal_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, head, 3, (245, 190, 45), -1, cv2.LINE_AA)
    return base, head


def _make_clip(
    np: Any,
    cv2: Any,
    robot_frames: Any,
    left_hand_xy: Any,
    right_hand_xy: Any,
    *,
    clip_index: int,
    seed: int,
) -> tuple[Any, Any, dict[str, object]]:
    frame_count, height, width = robot_frames.shape[:3]
    phases = canonical_contact_phases(frame_count)
    active_grip = causal_flower_grip_trajectory(np, right_hand_xy, phases)
    rng = np.random.default_rng(seed + clip_index * 104729)
    background = _background(np, rng, width, height)
    targets, controls = [], []
    trajectory_rows = []
    active_color = tuple(int(value) for value in rng.integers(130, 245, size=3))
    bouquet_colors = [
        (235, 110, 150),
        (245, 205, 70),
        (220, 80, 100),
        (245, 180, 195),
        (245, 135, 70),
    ]
    for frame_index in range(frame_count):
        target = background.copy()
        geometry = np.zeros((height, width), dtype=np.uint8)
        left = left_hand_xy[frame_index]
        right = right_hand_xy[frame_index]
        vase_center = (
            round(float(left[0])),
            min(height - 10, round(float(left[1] + 48))),
        )
        cv2.ellipse(target, vase_center, (16, 23), 0, 0, 360, (90, 135, 165), -1, cv2.LINE_AA)
        cv2.ellipse(geometry, vase_center, (16, 23), 0, 0, 360, 255, -1, cv2.LINE_AA)
        for flower_index, offset in enumerate((-22, -11, 0, 11, 22)):
            bouquet_grip = left + np.asarray([offset * 0.18, offset * 0.06], dtype=np.float32)
            _draw_flower(
                cv2,
                target,
                bouquet_grip,
                length=48 + flower_index * 3,
                angle=-0.24 + flower_index * 0.12,
                stem_color=(55, 135, 70),
                petal_color=bouquet_colors[flower_index],
            )
            _draw_flower(
                cv2,
                geometry,
                bouquet_grip,
                length=48 + flower_index * 3,
                angle=-0.24 + flower_index * 0.12,
                stem_color=(255, 255, 255),
                petal_color=(255, 255, 255),
            )
        active_base, active_head = _draw_flower(
            cv2,
            target,
            active_grip[frame_index],
            length=62,
            angle=0.18,
            stem_color=(45, 125, 60),
            petal_color=active_color,
            thickness=3,
        )
        _draw_flower(
            cv2,
            geometry,
            active_grip[frame_index],
            length=62,
            angle=0.18,
            stem_color=(255, 255, 255),
            petal_color=(255, 255, 255),
            thickness=4,
        )
        robot = robot_frames[frame_index]
        robot_mask = robot.max(axis=2) > 24
        target[robot_mask] = robot[robot_mask]
        geometry[robot_mask] = 255
        edges = cv2.Canny(geometry, 60, 140)
        control = np.repeat(edges[..., None], 3, axis=2)
        contact = phases[frame_index] in {
            HandPhase.GRASP,
            HandPhase.MANIPULATE,
            HandPhase.RELEASE,
        }
        cv2.circle(
            control,
            (round(float(left[0])), round(float(left[1]))),
            5,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )
        if contact:
            cv2.circle(
                control,
                (round(float(right[0])), round(float(right[1]))),
                5,
                (255, 0, 0),
                -1,
                cv2.LINE_AA,
            )
        targets.append(target)
        controls.append(control)
        trajectory_rows.append(
            {
                "frame": frame_index,
                "phase": phases[frame_index].value,
                "left_hand_xy": left.tolist(),
                "bouquet_grip_xy": left.tolist(),
                "right_hand_xy": right.tolist(),
                "active_flower_grip_xy": active_grip[frame_index].tolist(),
                "active_flower_base_xy": list(active_base),
                "active_flower_head_xy": list(active_head),
                "right_contact_required": contact,
                "right_contact_error_pixels": (
                    float(np.linalg.norm(active_grip[frame_index] - right)) if contact else None
                ),
                "occlusion_order": "flower_behind_robot_hand_at_grip",
            }
        )
    targets_array = np.stack(targets)
    controls_array = np.stack(controls)
    contact_errors = [
        row["right_contact_error_pixels"]
        for row in trajectory_rows
        if row["right_contact_error_pixels"] is not None
    ]
    first_contact = next(
        index for index, phase in enumerate(phases) if phase is HandPhase.GRASP
    )
    last_contact = max(
        index
        for index, phase in enumerate(phases)
        if phase in {HandPhase.GRASP, HandPhase.MANIPULATE, HandPhase.RELEASE}
    )
    pre_contact_step = float(
        np.max(np.linalg.norm(np.diff(active_grip[: first_contact + 1], axis=0), axis=1))
    )
    post_contact_step = float(
        np.max(np.linalg.norm(np.diff(active_grip[last_contact:], axis=0), axis=1))
        if last_contact < frame_count - 1
        else 0.0
    )
    metrics = {
        "left_bouquet_attachment_error_max_pixels": 0.0,
        "right_contact_attachment_error_max_pixels": max(contact_errors),
        "active_flower_pre_contact_step_max_pixels": pre_contact_step,
        "active_flower_post_release_step_max_pixels": post_contact_step,
    }
    if any(float(value) > 1e-6 for value in metrics.values()):
        raise RuntimeError(f"synthetic contact invariants failed: {metrics}")
    return targets_array, controls_array, {
        "coordinate_frame": "camera:synthetic_pixels",
        "flower_frame": "object:flower",
        "robot_frame": "robot:base projected into camera:synthetic_pixels",
        "frames": trajectory_rows,
        "metrics": metrics,
    }


def _asset(
    asset_id: str,
    path: Path,
    split: AdaptationSplit,
    kind: AdaptationAssetKind,
) -> AdaptationAsset:
    return AdaptationAsset(
        asset_id,
        str(path),
        split,
        kind,
        f"local://procedural-flower-task/{asset_id}",
        f"derived only from Apache-2.0 Sharpa revision {SHARPA_ASSET_REVISION}",
        file_sha256(path),
        path.stat().st_size,
        True,
    )


def main() -> int:
    args = _parser().parse_args()
    if min(
        args.train_clips,
        args.validation_clips,
        args.frames,
        args.fps,
        args.width,
        args.height,
        args.source_frame_step,
    ) <= 0:
        raise ValueError("dataset sizes, dimensions, FPS, and source step must be positive")
    canonical_contact_phases(args.frames)
    robot_path = args.robot_layer_video.expanduser().resolve()
    trace_path = args.wrist_trace.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    for label, path in (
        ("robot layer", robot_path),
        ("wrist trace", trace_path),
        ("ffmpeg", ffmpeg),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{label} does not exist or is empty: {path}")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {output}")
    output.mkdir(parents=True)

    import cv2
    import numpy as np
    from PIL import Image

    robot_frames = _decode_video(robot_path, ffmpeg, args.width, args.height)
    trace = json.loads(trace_path.read_text())
    if len(robot_frames) != len(trace) or len(trace) != 660:
        raise RuntimeError("task adaptation requires the aligned 660-frame robot and wrist traces")
    if any(
        row.get("robot_frame") != "robot:base"
        or row.get("source_frame") != "camera:source_pixels"
        for row in trace
    ):
        raise ValueError("wrist trace coordinate frames are missing or inconsistent")
    source_width, source_height = 640.0, 480.0
    wrists = np.asarray(
        [
            [
                row["rendered_hand_centroids"]["left"],
                row["rendered_hand_centroids"]["right"],
            ]
            for row in trace
        ],
        dtype=np.float32,
    )
    wrists[..., 0] *= args.width / source_width
    wrists[..., 1] *= args.height / source_height
    if not np.isfinite(wrists).all():
        raise ValueError("wrist trace contains missing or non-finite hand centroids")
    maximum_start = len(robot_frames) - 1 - (args.frames - 1) * args.source_frame_step
    if maximum_start < 0:
        raise ValueError("source trace is too short for the requested task clips")
    rng = np.random.default_rng(args.seed)
    total = args.train_clips + args.validation_clips
    starts = rng.choice(maximum_start + 1, size=total, replace=total > maximum_start + 1)
    assets: list[AdaptationAsset] = []
    training_examples: list[VaceTrainingExample] = []
    validation_records = []
    all_metrics = []
    for clip_index, start in enumerate(starts.tolist()):
        split = (
            AdaptationSplit.TRAIN
            if clip_index < args.train_clips
            else AdaptationSplit.VALIDATION
        )
        indices = start + np.arange(args.frames) * args.source_frame_step
        target_frames, control_frames, trajectory = _make_clip(
            np,
            cv2,
            robot_frames[indices],
            wrists[indices, 0],
            wrists[indices, 1],
            clip_index=clip_index,
            seed=args.seed,
        )
        clip_dir = output / split.value / f"clip-{clip_index:03d}"
        clip_dir.mkdir(parents=True)
        target_path = clip_dir / "target.mp4"
        control_path = clip_dir / "control.mp4"
        reference_path = clip_dir / "reference.png"
        trajectory_path = clip_dir / "contact-trajectory.json"
        _encode_video(target_frames, target_path, ffmpeg, args.fps)
        _encode_video(control_frames, control_path, ffmpeg, args.fps)
        Image.fromarray(target_frames[0]).save(reference_path)
        trajectory.update(
            {
                "source_start_frame": start,
                "source_frame_step": args.source_frame_step,
                "source_frame_indices": indices.tolist(),
            }
        )
        trajectory_path.write_text(json.dumps(trajectory, indent=2, sort_keys=True) + "\n")
        all_metrics.append(trajectory["metrics"])
        prefix = f"{split.value}-{clip_index:03d}"
        clip_assets = (
            _asset(prefix + "-target", target_path, split, AdaptationAssetKind.TARGET_VIDEO),
            _asset(
                prefix + "-control",
                control_path,
                split,
                AdaptationAssetKind.VACE_CONTROL_VIDEO,
            ),
            _asset(
                prefix + "-reference",
                reference_path,
                split,
                AdaptationAssetKind.VACE_REFERENCE_IMAGE,
            ),
        )
        assets.extend(clip_assets)
        prompt = (
            "A silver humanoid robot uses both articulated hands to arrange flowers. "
            "The left hand continuously holds one bouquet while the right hand approaches, "
            "grasps, moves, releases, and retracts from one flower with causal stem contact."
        )
        if split is AdaptationSplit.TRAIN:
            training_examples.append(
                VaceTrainingExample(
                    f"flower-task-{clip_index:03d}",
                    clip_assets[0].asset_id,
                    clip_assets[1].asset_id,
                    clip_assets[2].asset_id,
                    prompt,
                )
            )
        else:
            validation_records.append(
                {
                    "clip_id": f"flower-task-{clip_index:03d}",
                    "target": str(target_path),
                    "control": str(control_path),
                    "reference": str(reference_path),
                    "contact_trajectory": str(trajectory_path),
                }
            )
    manifest = AdaptationManifest(
        experiment_id=output.name,
        arm=AdaptationArm.VACE_LORA,
        assets=tuple(assets),
        vace_examples=tuple(training_examples),
        evidence_scope="development_only",
    )
    manifest.write_json(output / "frozen" / "manifest.json")
    _write = lambda path, payload: path.write_text(  # noqa: E731
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _write(output / "validation.json", validation_records)
    packages = {}
    for name in ("numpy", "opencv-python", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    provenance = {
        "schema_version": "1.0.0",
        "method": "paired_synthetic_bimanual_flower_contact_vace_task_adapter_data",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "honest_status": "PARTIAL",
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "gpu": {"used": False, "reason": "procedural CPU dataset generation"},
        "robot_layer": {"path": str(robot_path), "sha256": file_sha256(robot_path)},
        "wrist_trace": {"path": str(trace_path), "sha256": file_sha256(trace_path)},
        "sharpa_asset_revision": SHARPA_ASSET_REVISION,
        "seed": args.seed,
        "train_clips": args.train_clips,
        "validation_clips": args.validation_clips,
        "frames": args.frames,
        "fps": args.fps,
        "resolution": [args.width, args.height],
        "source_frame_step": args.source_frame_step,
        "manifest_sha256": hashlib.sha256(
            (output / "frozen" / "manifest.json").read_bytes()
        ).hexdigest(),
        "contact_metrics": all_metrics,
        "limitations": [
            "The paired data is procedural synthetic supervision, not real robot flower footage.",
            "The explicit stems are rigid 2-D objects and do not model petal deformation or force.",
            "A successful training smoke test proves trainability only; held-out real-video gates remain required.",
        ],
    }
    _write(output / "provenance.json", provenance)
    print(json.dumps({"dataset": str(output), "manifest": str(output / "frozen" / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
