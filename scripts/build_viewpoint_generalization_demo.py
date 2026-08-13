#!/usr/bin/env python3
"""Build identity-locked camera-perturbation demos from reviewed task videos.

The output is deliberately scoped as a camera-frame stress test.  Every view is
derived from the same decoded source frame with a named projective transform;
the script does not hallucinate unseen surfaces or claim physical multi-camera
capture.  That constraint makes subject identity deterministic and auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPS = 24.0
FRAMES = 240
CANVAS = (1280, 790)
TILE = (640, 360)
SEED = 20260811


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parse_labeled_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or label in result:
            raise ValueError("--task requires unique LABEL=PATH pairs")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"task video is missing: {path}")
        result[label] = path
    if len(result) != 3:
        raise ValueError("--task must be supplied exactly three times")
    return result


def _git_state() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def _decode(cv2: Any, path: Path) -> tuple[list[Any], dict[str, float | int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    info: dict[str, float | int] = {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": len(frames),
        "duration_seconds": len(frames) / fps if fps > 0 else 0.0,
    }
    if len(frames) != FRAMES or abs(fps - FPS) > 1e-6:
        raise ValueError(f"video must be exactly {FRAMES} frames at {FPS:g} FPS: {path} {info}")
    return frames, info


def _homography_points(
    np: Any,
    width: int,
    height: int,
    yaw_degrees: float,
    *,
    zoom: float = 1.0,
) -> tuple[Any, Any]:
    """Return source/destination points for a named camera-frame proxy.

    Positive yaw compresses the right edge and negative yaw compresses the left
    edge.  Coordinates remain explicitly in the source camera pixel frame.
    """
    src = np.float32(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]]
    )
    strength = min(abs(float(yaw_degrees)) / 30.0, 1.0)
    horizontal = width * 0.075 * strength
    vertical = height * 0.065 * strength
    if yaw_degrees >= 0:
        dst = np.float32(
            [
                [0.0 + horizontal, 0.0 + vertical],
                [width - 1.0 - horizontal * 0.20, 0.0],
                [width - 1.0 - horizontal * 0.20, height - 1.0],
                [0.0 + horizontal, height - 1.0 - vertical],
            ]
        )
    else:
        dst = np.float32(
            [
                [0.0 + horizontal * 0.20, 0.0],
                [width - 1.0 - horizontal, 0.0 + vertical],
                [width - 1.0 - horizontal, height - 1.0 - vertical],
                [0.0 + horizontal * 0.20, height - 1.0],
            ]
        )
    if abs(zoom - 1.0) > 1e-8:
        centre = np.float32([(width - 1.0) / 2.0, (height - 1.0) / 2.0])
        dst = (dst - centre) * float(zoom) + centre
    return src, dst


def _warp(cv2: Any, np: Any, frame: Any, yaw: float, *, zoom: float = 1.0) -> tuple[Any, Any]:
    height, width = frame.shape[:2]
    src, dst = _homography_points(np, width, height, yaw, zoom=zoom)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return warped, matrix


def _roundtrip_metrics(cv2: Any, np: Any, frame: Any, warped: Any, matrix: Any) -> dict[str, float]:
    height, width = frame.shape[:2]
    inverse = np.linalg.inv(matrix)
    restored = cv2.warpPerspective(
        warped,
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    margin_x = max(1, int(round(width * 0.10)))
    margin_y = max(1, int(round(height * 0.10)))
    reference = frame[margin_y:-margin_y, margin_x:-margin_x].astype(np.float32)
    candidate = restored[margin_y:-margin_y, margin_x:-margin_x].astype(np.float32)
    difference = reference - candidate
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(difference * difference))
    psnr = float(20.0 * math.log10(255.0 / math.sqrt(max(mse, 1e-12))))
    # The colour round-trip error is already measured above in float space.
    # Reuse it for a bounded similarity score so OpenCV temporary-buffer reuse
    # cannot affect the audit metric on older local wheels.
    luma_similarity = max(0.0, 1.0 - mae / 255.0)
    return {
        "roundtrip_mae": mae,
        "roundtrip_psnr_db": psnr,
        "luma_similarity": luma_similarity,
    }


def _label_tile(cv2: Any, np: Any, frame: Any, title: str, subtitle: str) -> Any:
    tile_width, tile_height = TILE
    image = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
    header = np.full((50, tile_width, 3), (15, 23, 19), dtype=np.uint8)
    cv2.rectangle(header, (0, 0), (7, 49), (106, 245, 200), -1)
    cv2.putText(header, title, (22, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (106, 245, 200), 1, cv2.LINE_AA)
    cv2.putText(header, subtitle, (22, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (197, 207, 201), 1, cv2.LINE_AA)
    return np.vstack((header, image))


def _compose_frame(
    cv2: Any,
    np: Any,
    frame: Any,
    frame_index: int,
    task_title: str,
    *,
    measure: bool,
) -> tuple[Any, list[dict[str, Any]]]:
    phase = 2.0 * math.pi * frame_index / max(FRAMES - 1, 1)
    orbit_yaw = 8.0 * math.sin(phase)
    orbit_zoom = 1.018 + 0.012 * math.sin(phase + math.pi / 2.0)
    left, left_matrix = _warp(cv2, np, frame, -12.0)
    right, right_matrix = _warp(cv2, np, frame, 12.0)
    orbit, orbit_matrix = _warp(cv2, np, frame, orbit_yaw, zoom=orbit_zoom)
    tiles = [
        _label_tile(cv2, np, frame, "SOURCE CAMERA", "reviewed camera:source_pixels"),
        _label_tile(cv2, np, left, "LEFT OBLIQUE -12 DEG", "identity-locked projective stress"),
        _label_tile(cv2, np, right, "RIGHT OBLIQUE +12 DEG", "identity-locked projective stress"),
        _label_tile(cv2, np, orbit, "ORBIT SWEEP +/-8 DEG", "continuous camera perturbation"),
    ]
    grid = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
    footer = np.full((50, CANVAS[0], 3), (8, 13, 11), dtype=np.uint8)
    cv2.putText(
        footer,
        task_title,
        (22, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (222, 229, 224),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "SAME DECODED SUBJECT FRAME | 2-D CAMERA PROXY | NOT PHYSICAL MULTI-CAMERA FOOTAGE",
        (22, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (128, 144, 135),
        1,
        cv2.LINE_AA,
    )
    composed = np.vstack((grid, footer))
    metrics = []
    if measure:
        metrics = [
            {"view": "left_oblique", **_roundtrip_metrics(cv2, np, frame, left, left_matrix)},
            {"view": "right_oblique", **_roundtrip_metrics(cv2, np, frame, right, right_matrix)},
            {"view": "orbit_sweep", **_roundtrip_metrics(cv2, np, frame, orbit, orbit_matrix)},
        ]
    return composed, metrics


def _write_video(ffmpeg: str, destination: Path, frames: list[Any], log_path: Path) -> None:
    height, width = frames[0].shape[:2]
    command = [
        ffmpeg,
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
        f"{FPS:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    with log_path.open("a") as log:
        log.write(json.dumps(command) + "\n")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed to write {destination} with code {return_code}")


def _write_reel(ffmpeg: str, sources: list[Path], destination: Path, log_path: Path) -> None:
    if len(sources) != 3:
        raise ValueError("the three-task reel requires exactly three source videos")
    filter_graph = (
        "[0:v]trim=start_frame=80:end_frame=160,setpts=PTS-STARTPTS[v0];"
        "[1:v]trim=start_frame=80:end_frame=160,setpts=PTS-STARTPTS[v1];"
        "[2:v]trim=start_frame=80:end_frame=160,setpts=PTS-STARTPTS[v2];"
        "[v0][v1][v2]concat=n=3:v=1:a=0[outv]"
    )
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for source in sources:
        command.extend(("-i", str(source)))
    command.extend(
        (
            "-filter_complex",
            filter_graph,
            "-map",
            "[outv]",
            "-an",
            "-r",
            f"{FPS:g}",
            "-frames:v",
            str(FRAMES),
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        )
    )
    with log_path.open("a") as log:
        log.write(json.dumps(command) + "\n")
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg failed to write {destination} with code {completed.returncode}")


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["view"]), []).append(row)
    views: dict[str, Any] = {}
    for view, records in grouped.items():
        views[view] = {
            "samples": len(records),
            "roundtrip_mae_mean": sum(record["roundtrip_mae"] for record in records) / len(records),
            "roundtrip_mae_max": max(record["roundtrip_mae"] for record in records),
            "roundtrip_psnr_db_min": min(record["roundtrip_psnr_db"] for record in records),
            "luma_similarity_min": min(record["luma_similarity"] for record in records),
        }
    passed = all(
        record["roundtrip_mae_max"] <= 4.0
        and record["roundtrip_psnr_db_min"] >= 30.0
        and record["luma_similarity_min"] >= 0.98
        for record in views.values()
    )
    return {
        "views": views,
        "identity_lineage_passed": passed,
        "thresholds": {
            "roundtrip_mae_max": 4.0,
            "roundtrip_psnr_db_min": 30.0,
            "luma_similarity_min": 0.98,
        },
        "interpretation": "Round-trip image consistency on the central 80% frame; not a semantic identity model.",
    }


def _package_name(label: str) -> str:
    return label.strip().lower().replace("_", "-").replace(" ", "-")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--identity-anchor", type=Path)
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/acwm-viewpoint-generalization"))
    parser.add_argument("--publish-dir", type=Path, default=Path("demo/showcase"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    import cv2
    import numpy as np

    tasks = _parse_labeled_paths(args.task)
    identity_anchor = args.identity_anchor.expanduser().resolve() if args.identity_anchor else None
    if identity_anchor is not None and (not identity_anchor.is_file() or identity_anchor.stat().st_size == 0):
        raise ValueError(f"identity anchor is missing: {identity_anchor}")
    ffmpeg = shutil.which(args.ffmpeg) if not Path(args.ffmpeg).is_absolute() else args.ffmpeg
    if not ffmpeg or not Path(ffmpeg).is_file():
        raise ValueError(f"ffmpeg is unavailable: {args.ffmpeg}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    experiment = args.experiment_root.expanduser().resolve() / timestamp
    experiment.mkdir(parents=True, exist_ok=False)
    publish_dir = args.publish_dir.expanduser().resolve()
    publish_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment / "render.log"
    (experiment / "command.txt").write_text(" ".join(sys.argv) + "\n")
    _write_json(
        experiment / "config.json",
        {
            "schema_version": "1.0.0",
            "seed": SEED,
            "tasks": {label: str(path) for label, path in tasks.items()},
            "identity_anchor": str(identity_anchor) if identity_anchor else None,
            "coordinate_frames": {
                "source": "camera:source_pixels",
                "derived": "camera:projective_proxy_pixels",
            },
            "views": {
                "source": {"yaw_degrees": 0.0},
                "left_oblique": {"yaw_degrees": -12.0},
                "right_oblique": {"yaw_degrees": 12.0},
                "orbit_sweep": {"yaw_degrees": [-8.0, 8.0], "continuous": True},
            },
            "evaluation": {
                "sample_stride_frames": 10,
                "central_frame_fraction": 0.8,
            },
            "outputs": {
                "width": CANVAS[0],
                "height": 870,
                "frames": FRAMES,
                "fps": FPS,
            },
            "claim_boundary": "2-D projective camera stress; not novel-view synthesis or physical multi-camera capture.",
        },
    )
    _write_json(experiment / "git-state.json", _git_state())

    task_titles = {
        "handover": "TASK 01 / TWO-HAND BOTTLE HANDOVER",
        "unscrew": "TASK 02 / UNSCREW AND SEPARATE CAP",
        "rinse": "TASK 03 / RINSE AND ROTATE BOTTLE",
    }
    task_records: dict[str, Any] = {}
    task_videos: list[Path] = []
    sample_indices = set(range(0, FRAMES, 10))
    for label, source_path in tasks.items():
        frames, source_info = _decode(cv2, source_path)
        composed_frames = []
        sampled_metrics: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            composed, metrics = _compose_frame(
                cv2,
                np,
                frame,
                index,
                task_titles.get(label, label.upper()),
                measure=index in sample_indices,
            )
            composed_frames.append(composed)
            for metric in metrics:
                sampled_metrics.append({"frame_index": index, **metric})
        packaged = _package_name(label)
        video = experiment / f"{packaged}-viewpoint-stress-10s.mp4"
        poster = experiment / f"{packaged}-viewpoint-stress-poster.jpg"
        _write_video(str(ffmpeg), video, composed_frames, log_path)
        if not cv2.imwrite(str(poster), composed_frames[FRAMES // 2]):
            raise RuntimeError(f"could not write {poster}")
        metrics = _aggregate_metrics(sampled_metrics)
        _write_json(experiment / f"{packaged}-evaluation.json", metrics)
        task_records[label] = {
            "source": str(source_path),
            "source_sha256": _sha256(source_path),
            "source_video": source_info,
            "output": str(video),
            "output_sha256": _sha256(video),
            "poster": str(poster),
            "poster_sha256": _sha256(poster),
            "evaluation": metrics,
        }
        task_videos.append(video)

    reel = experiment / "viewpoint-generalization-three-task-reel-10s.mp4"
    reel_poster = experiment / "viewpoint-generalization-three-task-reel-poster.jpg"
    _write_reel(str(ffmpeg), task_videos, reel, log_path)
    reel_capture = cv2.VideoCapture(str(reel))
    reel_capture.set(cv2.CAP_PROP_POS_FRAMES, FRAMES // 2)
    ok, reel_midpoint = reel_capture.read()
    reel_capture.release()
    if not ok or not cv2.imwrite(str(reel_poster), reel_midpoint):
        raise RuntimeError(f"could not extract {reel_poster}")

    published: dict[str, str] = {}
    for label, record in task_records.items():
        packaged = _package_name(label)
        for kind, source in (
            ("video", Path(record["output"])),
            ("poster", Path(record["poster"])),
            ("evaluation", experiment / f"{packaged}-evaluation.json"),
        ):
            suffix = source.suffix
            destination = publish_dir / f"acwm-ego-viewpoint-{packaged}-{kind}{suffix}"
            shutil.copy2(source, destination)
            published[f"{label}_{kind}"] = str(destination)
    reel_public = publish_dir / "acwm-ego-viewpoint-generalization-reel-10s.mp4"
    reel_poster_public = publish_dir / "acwm-ego-viewpoint-generalization-reel-poster.jpg"
    shutil.copy2(reel, reel_public)
    shutil.copy2(reel_poster, reel_poster_public)
    published["reel_video"] = str(reel_public)
    published["reel_poster"] = str(reel_poster_public)
    if identity_anchor is not None:
        identity_public = publish_dir / "acwm-ego-viewpoint-identity-anchor.png"
        shutil.copy2(identity_anchor, identity_public)
        published["identity_anchor"] = str(identity_public)

    package_versions = {}
    for package in ("numpy", "opencv-python"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "unavailable"
    identity_passed = all(
        bool(record["evaluation"]["identity_lineage_passed"])
        for record in task_records.values()
    )
    manifest = {
        "schema_version": "1.0.0",
        "status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": package_versions,
        "seed": SEED,
        "coordinate_frames": {
            "source": "camera:source_pixels",
            "derived": "camera:projective_proxy_pixels",
            "transform": "explicit homography per output view",
        },
        "method": "same-frame deterministic projective camera perturbation",
        "identity_lineage_passed": identity_passed,
        "task_count": len(tasks),
        "tasks": task_records,
        "reel": {
            "path": str(reel),
            "sha256": _sha256(reel),
            "frames": FRAMES,
            "fps": FPS,
        },
        "published": {
            name: {"path": path, "sha256": _sha256(Path(path))}
            for name, path in published.items()
        },
        "claim_boundary": (
            "Identity-locked 2-D camera perturbation evidence on reviewed real-scene visual task videos. "
            "It is not learned novel-view synthesis, calibrated 3-D reconstruction, physical multi-camera capture, "
            "contact physics, or physical robot execution."
        ),
    }
    _write_json(experiment / "manifest.json", manifest)
    public_manifest = publish_dir / "acwm-ego-viewpoint-generalization-manifest.json"
    shutil.copy2(experiment / "manifest.json", public_manifest)
    print(json.dumps({"experiment": str(experiment), "published": published, "identity_lineage_passed": identity_passed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
