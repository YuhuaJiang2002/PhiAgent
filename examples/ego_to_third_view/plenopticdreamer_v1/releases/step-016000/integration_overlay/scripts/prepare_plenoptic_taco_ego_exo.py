#!/usr/bin/env python3
"""Prepare one calibrated TACO head-mounted-to-fixed-external Plenoptic case."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable

import cv2
import numpy as np


FRAMES = 81
WIDTH = 768
HEIGHT = 432
FPS = 15


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ego-video", type=Path, required=True)
    result.add_argument("--ego-extrinsics", type=Path, required=True)
    result.add_argument("--ego-intrinsics", type=Path, required=True)
    result.add_argument("--external-video", type=Path, required=True)
    result.add_argument("--external-calibration", type=Path, required=True)
    result.add_argument("--external-camera-id", required=True)
    result.add_argument("--triplet", required=True)
    result.add_argument("--sequence", required=True)
    result.add_argument("--generation-input", type=Path, required=True)
    result.add_argument("--heldout-output", type=Path, required=True)
    result.add_argument("--start-frame", type=int, default=0)
    result.add_argument("--stride", type=int, default=2)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
    return resolved


def encode_selected(
    source: Path,
    output: Path,
    indices: list[int],
    transform: Callable[[np.ndarray], np.ndarray],
) -> None:
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "4",
        str(output.with_suffix(".tmp.mp4")),
    ]
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    selected = set(indices)
    last = indices[-1]
    emitted = 0
    try:
        for index in range(last + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"video ended before frame {index}: {source}")
            if index not in selected:
                continue
            normalized = np.ascontiguousarray(transform(frame), dtype=np.uint8)
            if normalized.shape != (HEIGHT, WIDTH, 3):
                raise ValueError(f"normalized frame has invalid shape: {normalized.shape}")
            process.stdin.write(normalized.tobytes())
            emitted += 1
    finally:
        capture.release()
        process.stdin.close()
    returncode = process.wait()
    if returncode or emitted != FRAMES:
        raise RuntimeError(f"encoding failed: returncode={returncode}, frames={emitted}")
    output.with_suffix(".tmp.mp4").replace(output)


def adjusted_intrinsics(
    intrinsic: np.ndarray,
    *,
    crop_x: int = 0,
    crop_y: int = 0,
    crop_width: int,
    crop_height: int,
) -> np.ndarray:
    result = intrinsic.astype(np.float64).copy()
    result[0, 2] -= crop_x
    result[1, 2] -= crop_y
    result[0, :] *= WIDTH / crop_width
    result[1, :] *= HEIGHT / crop_height
    return result


def video_probe(path: Path) -> dict[str, object]:
    value = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ], text=True))["streams"][0]
    expected = (WIDTH, HEIGHT, f"{FPS}/1", str(FRAMES))
    actual = (int(value["width"]), int(value["height"]),
              value["r_frame_rate"], value["nb_read_frames"])
    if actual != expected:
        raise ValueError(f"unexpected encoded video: {actual}, expected {expected}")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.start_frame < 0 or args.stride < 1:
        raise ValueError("start-frame must be non-negative and stride must be positive")
    indices = [args.start_frame + args.stride * index for index in range(FRAMES)]
    generation = args.generation_input.resolve()
    heldout = args.heldout_output.resolve()
    if generation == heldout or generation in heldout.parents or heldout in generation.parents:
        raise ValueError("generation inputs and held-out target must use disjoint directories")
    if generation.exists() or heldout.exists():
        raise FileExistsError("generation-input and heldout-output must both be fresh")

    ego_video = require_file(args.ego_video, "egocentric video")
    ego_extrinsics_path = require_file(args.ego_extrinsics, "egocentric extrinsics")
    ego_intrinsics_path = require_file(args.ego_intrinsics, "egocentric intrinsics")
    external_video = require_file(args.external_video, "external video")
    external_calibration_path = require_file(
        args.external_calibration, "external calibration"
    )
    ego_w2c = np.load(ego_extrinsics_path, allow_pickle=False)
    ego_k = np.loadtxt(ego_intrinsics_path)
    if ego_w2c.shape[0] <= indices[-1] or ego_w2c.shape[1:] != (4, 4):
        raise ValueError(f"egocentric extrinsics do not cover selected frames: {ego_w2c.shape}")
    if ego_k.shape != (3, 3):
        raise ValueError(f"egocentric intrinsics have invalid shape: {ego_k.shape}")

    calibration = json.loads(external_calibration_path.read_text(encoding="utf-8"))
    if args.external_camera_id not in calibration:
        raise KeyError(f"external camera is absent from calibration: {args.external_camera_id}")
    camera = calibration[args.external_camera_id]
    external_k = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    external_r = np.asarray(camera["R"], dtype=np.float64).reshape(3, 3)
    external_t = np.asarray(camera["T"], dtype=np.float64)
    distortion = np.asarray(camera["distCoeff"], dtype=np.float64)
    external_size = tuple(int(value) for value in camera["imgSize"])
    if external_size != (2048, 1500):
        raise ValueError(f"this audited crop expects a 2048x1500 camera: {external_size}")
    if not np.allclose(np.linalg.det(ego_w2c[:, :3, :3]), 1.0, atol=2e-4):
        raise ValueError("egocentric extrinsics contain improper rotations")
    if not np.isclose(np.linalg.det(external_r), 1.0, atol=2e-4):
        raise ValueError("external extrinsic contains an improper rotation")
    if distortion.shape != (5,) or not np.isfinite(distortion).all():
        raise ValueError("external distortion metadata is invalid")

    generation.mkdir(parents=True)
    heldout.mkdir(parents=True)
    source_output = generation / "ego.mp4"
    target_output = heldout / "external.mp4"
    encode_selected(
        ego_video,
        source_output,
        indices,
        lambda frame: cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA),
    )

    crop_y, crop_height = 174, 1152

    def external_transform(frame: np.ndarray) -> np.ndarray:
        # TACO's released visualization projects into the published allocentric
        # RGB with K/R/T directly and does not apply distCoeff again.  Applying
        # it here double-rectifies this stream and creates a circular warp.
        cropped = frame[crop_y:crop_y + crop_height, :2048]
        return cv2.resize(cropped, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

    encode_selected(external_video, target_output, indices, external_transform)
    source_k = adjusted_intrinsics(
        ego_k, crop_width=1920, crop_height=1080
    )
    target_k = adjusted_intrinsics(
        external_k, crop_y=crop_y, crop_width=2048, crop_height=crop_height
    )
    external_w2c = np.eye(4, dtype=np.float64)
    external_w2c[:3, :3] = external_r
    external_w2c[:3, 3] = external_t
    source_c2w = np.linalg.inv(ego_w2c[indices].astype(np.float64))
    target_c2w = np.repeat(np.linalg.inv(external_w2c)[None], FRAMES, axis=0)
    c2w = np.stack((source_c2w, target_c2w)).astype(np.float32)
    intrinsics = np.stack((
        np.repeat(source_k[None], FRAMES, axis=0),
        np.repeat(target_k[None], FRAMES, axis=0),
    )).astype(np.float32)
    camera_output = generation / "ego-to-external-cameras.npz"
    np.savez_compressed(camera_output, c2w=c2w, intrinsics=intrinsics)
    baseline = np.linalg.norm(c2w[1, :, :3, 3] - c2w[0, :, :3, 3], axis=-1)
    provenance = {
        "schema": 1,
        "calibrated": True,
        "dataset": "TACO-v1",
        "license": "CC-BY-4.0",
        "triplet": args.triplet,
        "sequence": args.sequence,
        "source_camera": "head_mounted_egocentric",
        "target_camera": f"fixed_allocentric_{args.external_camera_id}",
        "frame": "TACO_capture_world",
        "pose_convention": "opencv_c2w",
        "source_pose_conversion": "inverse of published per-frame world_to_camera",
        "target_pose_conversion": "inverse of published fixed R,T world_to_camera",
        "unit": "m",
        "per_frame_aligned": True,
        "images_undistorted": True,
        "source_image_transform": "1920x1080 resized to 768x432",
        "target_image_transform": (
            "Published rectified allocentric RGB used without reapplying distCoeff, "
            "matching the official K/R/T projector; center crop 2048x1152 at y=174; "
            "resize to 768x432"
        ),
        "published_rectify_alpha": float(camera.get("rectifyAlpha", 0.0)),
        "published_distortion_coefficients_retained_as_metadata": True,
        "selected_original_frames": {
            "start": args.start_frame,
            "stride": args.stride,
            "count": FRAMES,
            "last": indices[-1],
        },
        "source_target_baseline_min_m": float(baseline.min()),
        "source_target_baseline_max_m": float(baseline.max()),
        "target_rgb_present_in_generation_directory": False,
    }
    write_json(generation / "camera-provenance.json", provenance)
    (generation / "prompt.txt").write_text(
        "A fixed wide third-person view of a person using a blue roller to dust "
        "a black pan on a round teal table. Preserve both hands, the tool-object "
        "interaction, the seated person's identity, and the static room layout.\n",
        encoding="utf-8",
    )
    generation_manifest = {
        "schema": 1,
        "status": "prepared_target_free",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_video": str(source_output),
        "camera_npz": str(camera_output),
        "camera_provenance": str(generation / "camera-provenance.json"),
        "target_rgb_present": False,
        "raw_input_sha256": {
            "ego_video": sha256_file(ego_video),
            "ego_extrinsics": sha256_file(ego_extrinsics_path),
            "ego_intrinsics": sha256_file(ego_intrinsics_path),
            "external_calibration": sha256_file(external_calibration_path),
        },
        "prepared_sha256": {
            "ego.mp4": sha256_file(source_output),
            "ego-to-external-cameras.npz": sha256_file(camera_output),
        },
        "probe": video_probe(source_output),
    }
    write_json(generation / "preparation.json", generation_manifest)
    heldout_manifest = {
        "schema": 1,
        "status": "prepared_heldout",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_external_video": str(external_video),
        "selected_original_frames": indices,
        "external_video_sha256": sha256_file(external_video),
        "prepared_target_sha256": sha256_file(target_output),
        "probe": video_probe(target_output),
    }
    write_json(heldout / "heldout.json", heldout_manifest)
    print(json.dumps({
        "generation_input": str(generation),
        "heldout_output": str(heldout),
        "baseline_min_m": float(baseline.min()),
        "baseline_max_m": float(baseline.max()),
        "source_sha256": sha256_file(source_output),
        "target_sha256": sha256_file(target_output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
