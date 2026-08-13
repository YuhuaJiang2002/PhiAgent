#!/usr/bin/env python3
"""Export calibrated flower-simulation windows as leakage-safe VACE training data."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phiagent.data.adaptation import (  # noqa: E402
    AdaptationArm,
    AdaptationAsset,
    AdaptationAssetKind,
    AdaptationManifest,
    AdaptationSplit,
    VaceTrainingExample,
    file_sha256,
)
from phiagent.rendering.metric_flower_simulation import project_world_points  # noqa: E402


PROMPT = (
    "A silver Unitree G1 humanoid with bilateral articulated Sharpa Wave hands "
    "approaches, force-closes around one named flower stem, bends the persistent "
    "metric stem through causal contact, releases it, and retracts. Preserve exact "
    "robot topology, flower identity, calibrated depth order, and the vase."
)
RIGHTS_BASIS = (
    "Procedural PhiAgent simulation derived from BSD-3-Clause MuJoCo Menagerie "
    "Unitree G1 and Apache-2.0 Sharpa Wave assets; no Pexels frames are included."
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--source-frame-step", type=int, default=3)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _git_state() -> dict[str, object]:
    state = {}
    for name, command in (
        ("head", ("git", "rev-parse", "HEAD")),
        ("status", ("git", "status", "--short")),
    ):
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        state[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return state


def selected_clip_starts(
    *,
    source_frames: int,
    clip_frames: int,
    source_frame_step: int,
) -> dict[str, tuple[int, ...]]:
    """Return four phase-disjoint validation windows and non-overlapping train groups."""

    if (source_frames, clip_frames, source_frame_step) != (660, 17, 3):
        raise ValueError("the frozen v1 split requires 660 source frames and 17x3 windows")
    train = (0, 30, 60, 180, 210, 216, 330, 360, 366, 480, 510, 540)
    validation = (120, 270, 420, 600)
    window_span = (clip_frames - 1) * source_frame_step
    train_sets = {
        frame
        for start in train
        for frame in range(start, start + window_span + 1)
    }
    validation_sets = {
        frame
        for start in validation
        for frame in range(start, start + window_span + 1)
    }
    if train_sets.intersection(validation_sets):
        raise RuntimeError("frozen metric flower train and validation windows overlap")
    if max((*train, *validation)) + window_span >= source_frames:
        raise RuntimeError("frozen metric flower window exceeds the source timeline")
    return {"train": train, "validation": validation}


def _decode_video(cv2: Any, path: Path) -> Any:
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode simulation video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError("simulation video decoded zero frames")
    return np.stack(frames)


def _presentation_frame(cv2: Any, frame: Any, width: int, height: int) -> Any:
    source_height, source_width = frame.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized_width = max(width, round(source_width * scale))
    resized_height = max(height, round(source_height * scale))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    return resized[top : top + height, left : left + width]


def _presentation_xy(
    np: Any,
    xy: Any,
    *,
    source_width: int,
    source_height: int,
    width: int,
    height: int,
) -> Any:
    scale = max(width / source_width, height / source_height)
    resized_width = max(width, round(source_width * scale))
    resized_height = max(height, round(source_height * scale))
    offset = np.asarray(
        ((resized_width - width) * 0.5, (resized_height - height) * 0.5),
        dtype=np.float64,
    )
    return np.asarray(xy, dtype=np.float64) * scale - offset


def _encode_video(
    ffmpeg: Path,
    frames: Any,
    path: Path,
    *,
    fps: int,
) -> None:
    height, width = frames.shape[1:3]
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
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
        input=frames.tobytes(),
        check=True,
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create {path}")


def _asset(
    asset_id: str,
    path: Path,
    split: AdaptationSplit,
    kind: AdaptationAssetKind,
) -> AdaptationAsset:
    return AdaptationAsset(
        asset_id=asset_id,
        path=str(path),
        split=split,
        kind=kind,
        source_uri=f"local://metric-flower-simulation/{asset_id}",
        rights_basis=RIGHTS_BASIS,
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        training_authorized=True,
    )


def main() -> int:
    args = _parser().parse_args()
    simulation_dir = args.simulation_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite dataset directory: {output_dir}")
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"ffmpeg is missing: {ffmpeg}")
    inputs = {
        "video": simulation_dir / "metric-flower-simulation.mp4",
        "report": simulation_dir / "simulation-report.json",
        "camera": simulation_dir / "metric-camera-samples.npz",
        "stems": simulation_dir / "metric-stem-centerlines.npz",
        "identity": simulation_dir / "robot-identity-reference.png",
    }
    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"simulation dataset inputs are missing: {missing}")
    report = json.loads(inputs["report"].read_text())
    if (
        report.get("machine_acceptance_passed") is not True
        or report.get("simulated_physical_bundle_status") != "WORKING"
    ):
        raise ValueError("source simulation has not passed its physical acceptance contract")
    declared = report.get("artifacts", {})
    for name, artifact_name in (
        ("video", "video"),
        ("camera", "metric_camera_samples"),
        ("stems", "stem_centerlines"),
        ("identity", "robot_identity_reference"),
    ):
        if declared.get(artifact_name, {}).get("sha256") != _sha256(inputs[name]):
            raise ValueError(f"source simulation hash mismatch for {name}")
    if min(
        args.frames,
        args.source_frame_step,
        args.fps,
        args.width,
        args.height,
    ) <= 0:
        raise ValueError("clip dimensions, frames, step, and FPS must be positive")

    import cv2
    import numpy as np

    video_frames = _decode_video(cv2, inputs["video"])
    if video_frames.shape[0] != 660:
        raise RuntimeError(f"expected 660 source frames, decoded {video_frames.shape[0]}")
    camera = np.load(inputs["camera"], allow_pickle=False)
    stems = np.load(inputs["stems"], allow_pickle=False)
    if (
        camera["depth_m"].shape[0] != 660
        or stems["centerlines_world_m"].shape[:2] != (660, 1)
    ):
        raise RuntimeError("metric camera and named-stem timelines must cover 660 frames")
    split_starts = selected_clip_starts(
        source_frames=660,
        clip_frames=args.frames,
        source_frame_step=args.source_frame_step,
    )
    approach_end = int(report["timeline"]["approach_end_frame"])
    release_frame = int(report["timeline"]["release_frame"])
    source_height, source_width = video_frames.shape[1:3]
    output_dir.mkdir(parents=True)
    identity = cv2.imread(str(inputs["identity"]), cv2.IMREAD_COLOR)
    if identity is None:
        raise RuntimeError("cannot decode the independent robot identity reference")
    identity = _presentation_frame(
        cv2,
        identity,
        args.width,
        args.height,
    )
    identity_exact_target_matches = 0
    identity_corner_mad = []
    for source_frame in video_frames:
        presented_source = _presentation_frame(
            cv2,
            source_frame,
            args.width,
            args.height,
        )
        identity_exact_target_matches += int(
            np.array_equal(identity, presented_source)
        )
        identity_corner_mad.append(
            float(
                np.mean(
                    np.abs(
                        identity[:16, :16].astype(np.float32)
                        - presented_source[:16, :16].astype(np.float32)
                    )
                )
            )
        )
    identity_reference_independent = (
        identity_exact_target_matches == 0
        and min(identity_corner_mad) >= 16.0
    )
    if not identity_reference_independent:
        raise RuntimeError(
            "robot identity reference is not independent of target backgrounds"
        )
    identity_dir = output_dir / "identity"
    identity_dir.mkdir()
    identity_path = identity_dir / "robot-reference.png"
    if not cv2.imwrite(str(identity_path), identity):
        raise RuntimeError("failed to write the independent robot identity reference")
    identity_asset_id = "train-robot-identity-reference"
    assets = [
        _asset(
            identity_asset_id,
            identity_path,
            AdaptationSplit.TRAIN,
            AdaptationAssetKind.VACE_REFERENCE_IMAGE,
        )
    ]
    examples = []
    clip_records = []
    for split_name, starts in split_starts.items():
        split = AdaptationSplit(split_name)
        for split_index, start in enumerate(starts):
            source_indices = start + np.arange(args.frames) * args.source_frame_step
            clip_dir = output_dir / split_name / f"clip-{split_index:03d}"
            clip_dir.mkdir(parents=True)
            target_frames = np.stack(
                [
                    _presentation_frame(
                        cv2,
                        video_frames[index],
                        args.width,
                        args.height,
                    )
                    for index in source_indices
                ]
            )
            selected_depth = camera["depth_m"][source_indices].astype(np.float32)
            inverse_depth = 1.0 / np.maximum(selected_depth, 1e-6)
            low, high = np.percentile(inverse_depth, (1.0, 99.0))
            depth_normalized = np.clip(
                (inverse_depth - low) / max(1e-6, high - low),
                0.0,
                1.0,
            )
            controls = []
            trajectory = []
            for local_index, source_index in enumerate(source_indices):
                depth_frame = _presentation_frame(
                    cv2,
                    np.rint(depth_normalized[local_index] * 255.0)
                    .astype(np.uint8)[..., None]
                    .repeat(3, axis=2),
                    args.width,
                    args.height,
                )[..., 0]
                depth_edges = cv2.Canny(depth_frame, 12, 36)
                control = np.stack(
                    (depth_edges, depth_frame, depth_edges),
                    axis=2,
                )
                pixels, _ = project_world_points(
                    np,
                    points_world_m=stems["centerlines_world_m"][source_index, 0],
                    intrinsics_px=camera["intrinsics_px"][source_index],
                    world_from_camera=camera["world_from_camera"][source_index],
                )
                pixels = _presentation_xy(
                    np,
                    pixels,
                    source_width=source_width,
                    source_height=source_height,
                    width=args.width,
                    height=args.height,
                )
                polyline = np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(control, [polyline], False, (0, 255, 0), 2, cv2.LINE_AA)
                contact_active = approach_end <= source_index < release_frame
                contact_xy = tuple(int(value) for value in np.rint(pixels[7]))
                if contact_active:
                    cv2.circle(control, contact_xy, 5, (0, 0, 255), -1, cv2.LINE_AA)
                controls.append(control)
                trajectory.append(
                    {
                        "local_frame_index": local_index,
                        "source_frame_index": int(source_index),
                        "stem_instance_id": "active-stem-01",
                        "contact_active": bool(contact_active),
                        "contact_node": 7,
                        "contact_xy": list(contact_xy),
                    }
                )
            control_frames = np.stack(controls)
            target_path = clip_dir / "target.mp4"
            control_path = clip_dir / "control.mp4"
            trajectory_path = clip_dir / "trajectory.json"
            _encode_video(ffmpeg, target_frames, target_path, fps=args.fps)
            _encode_video(ffmpeg, control_frames, control_path, fps=args.fps)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "coordinate_frames": {
                            "control": "camera:vace_control_pixels",
                            "metric": str(stems["coordinate_frame"]),
                            "timeline": "frame:source_video",
                        },
                        "frames": trajectory,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            prefix = f"{split_name}-{split_index:03d}"
            target_id = f"{prefix}-target"
            control_id = f"{prefix}-control"
            assets.extend(
                (
                    _asset(
                        target_id,
                        target_path,
                        split,
                        AdaptationAssetKind.TARGET_VIDEO,
                    ),
                    _asset(
                        control_id,
                        control_path,
                        split,
                        AdaptationAssetKind.VACE_CONTROL_VIDEO,
                    ),
                )
            )
            if split is AdaptationSplit.TRAIN:
                examples.append(
                    VaceTrainingExample(
                        example_id=f"metric-flower-{split_index:03d}",
                        target_video_asset_id=target_id,
                        control_video_asset_id=control_id,
                        reference_image_asset_id=identity_asset_id,
                        prompt=PROMPT,
                    )
                )
            clip_records.append(
                {
                    "split": split.value,
                    "clip_index": split_index,
                    "source_start": start,
                    "source_indices": [int(value) for value in source_indices],
                    "target_sha256": file_sha256(target_path),
                    "control_sha256": file_sha256(control_path),
                    "reference_sha256": file_sha256(identity_path),
                    "trajectory_sha256": file_sha256(trajectory_path),
                }
            )

    manifest = AdaptationManifest(
        experiment_id=output_dir.name,
        arm=AdaptationArm.VACE_LORA,
        assets=tuple(assets),
        vace_examples=tuple(examples),
        evidence_scope="development_only",
    )
    manifest.write_json(output_dir / "frozen" / "manifest.json")
    train_source_frames = {
        frame
        for row in clip_records
        if row["split"] == "train"
        for frame in row["source_indices"]
    }
    validation_source_frames = {
        frame
        for row in clip_records
        if row["split"] == "validation"
        for frame in row["source_indices"]
    }
    validation = {
        "passed": not train_source_frames.intersection(validation_source_frames),
        "source_frames": int(video_frames.shape[0]),
        "train_clips": len(split_starts["train"]),
        "validation_clips": len(split_starts["validation"]),
        "train_validation_frame_overlap": len(
            train_source_frames.intersection(validation_source_frames)
        ),
        "metric_camera_bound": True,
        "persistent_stem_id_bound": True,
        "solver_contact_state_bound": True,
        "target_pixels_excluded_from_controls": True,
        "identity_reference_independent_of_target_clips": (
            identity_reference_independent
        ),
        "identity_exact_target_frame_matches": (
            identity_exact_target_matches
        ),
        "identity_target_corner_mad_min": min(identity_corner_mad),
        "clip_records": clip_records,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    packages = {}
    for package in ("numpy", "opencv-python"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    provenance = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "seed": args.seed,
        "git": _git_state(),
        "source_artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "rights_basis": RIGHTS_BASIS,
        "limitations": [
            "All clips share one simulated scene and cannot establish scene generalization.",
            "Validation is source-frame-disjoint but not an independent physical rollout.",
            "No Pexels evaluation frame is included in training.",
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "dataset": str(output_dir),
                "manifest": str(output_dir / "frozen" / "manifest.json"),
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
