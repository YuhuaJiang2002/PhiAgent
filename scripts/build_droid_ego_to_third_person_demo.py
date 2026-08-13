#!/usr/bin/env python3
"""Build auditable first-person to third-person demos from synchronized DROID views.

The wrist stream is the first-person input.  The two exterior streams are
measured third-person targets from the same physical episode and timestamp.
They are deliberately labelled as paired ground truth, not model-generated
novel views.  This script packages the pairs for training/evaluation and for a
public demo while preserving explicit camera-frame names and source lineage.
"""

from __future__ import annotations

import argparse
import hashlib
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
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
PANEL_WIDTH = 400
PANEL_HEIGHT = 225
FPS = 15.0
REEL_FRAMES_PER_TASK = 50
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


def _slug(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in cleaned.split("-") if part)


def _git_state() -> dict[str, str]:
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


def _require_file(raw_path: str | Path, *, label: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    return path


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("config must contain one JSON object")
    if float(payload.get("fps", 0.0)) != FPS:
        raise ValueError(f"config fps must be {FPS:g}")
    streams = payload.get("streams")
    required_streams = {"first_person", "third_person_a", "third_person_b"}
    if not isinstance(streams, dict) or set(streams) != required_streams:
        raise ValueError(f"streams must be exactly {sorted(required_streams)}")
    expected_frames = {
        "first_person": "camera:wrist_image_left_pixels",
        "third_person_a": "camera:exterior_image_1_left_pixels",
        "third_person_b": "camera:exterior_image_2_left_pixels",
    }
    for name, expected_frame in expected_frames.items():
        stream = streams[name]
        if stream.get("coordinate_frame") != expected_frame:
            raise ValueError(f"{name} coordinate_frame must be {expected_frame}")
        _require_file(stream.get("path", ""), label=name)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 3:
        raise ValueError("config must contain exactly three episodes")
    labels: set[str] = set()
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("every episode must be an object")
        required = {"episode_index", "label", "task", "start_frame", "frame_count", "timestamps"}
        if not required.issubset(episode):
            raise ValueError(f"episode is missing fields: {sorted(required - set(episode))}")
        if int(episode["start_frame"]) < 0 or int(episode["frame_count"]) <= 0:
            raise ValueError("episode frame interval must be positive")
        label = str(episode["label"])
        if label in labels:
            raise ValueError(f"episode labels must be unique: {label}")
        labels.add(label)
        timestamps = episode["timestamps"]
        if not isinstance(timestamps, dict) or set(timestamps) != required_streams:
            raise ValueError(f"episode timestamps must cover {sorted(required_streams)}")
        ranges = {
            (float(interval["from_seconds"]), float(interval["to_seconds"]))
            for interval in timestamps.values()
        }
        if len(ranges) != 1:
            raise ValueError(f"episode {episode['episode_index']} camera timestamps are not synchronized")
    return payload


def _probe_video(ffprobe: str, path: Path) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = (int(part) for part in stream["avg_frame_rate"].split("/"))
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": numerator / denominator,
        "frames": int(stream["nb_frames"]),
        "duration_seconds": float(stream["duration"]),
    }


def _stream_contract(probes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    required = {"first_person", "third_person_a", "third_person_b"}
    if set(probes) != required:
        raise ValueError(f"probe set must be exactly {sorted(required)}")
    dimensions = {(probe["width"], probe["height"]) for probe in probes.values()}
    frame_counts = {probe["frames"] for probe in probes.values()}
    frame_rates = {round(float(probe["fps"]), 9) for probe in probes.values()}
    durations = {round(float(probe["duration_seconds"]), 6) for probe in probes.values()}
    passed = len(dimensions) == len(frame_counts) == len(frame_rates) == len(durations) == 1
    return {
        "passed": passed,
        "dimensions_equal": len(dimensions) == 1,
        "frame_counts_equal": len(frame_counts) == 1,
        "frame_rates_equal": len(frame_rates) == 1,
        "durations_equal": len(durations) == 1,
        "probes": probes,
    }


def _decode_slice(
    ffmpeg: str,
    np: Any,
    path: Path,
    *,
    start_frame: int,
    frame_count: int,
    width: int,
    height: int,
) -> Any:
    start_seconds = start_frame / FPS
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.9f}",
        "-i",
        str(path),
        "-frames:v",
        str(frame_count),
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    expected_bytes = frame_count * width * height * 3
    if len(completed.stdout) != expected_bytes:
        raise RuntimeError(
            f"decoded byte count mismatch for {path}: expected {expected_bytes}, got {len(completed.stdout)}"
        )
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(frame_count, height, width, 3)


def _put_text(
    cv2: Any,
    image: Any,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def _panel(
    cv2: Any,
    np: Any,
    frame: Any,
    *,
    title: str,
    subtitle: str,
    accent: tuple[int, int, int],
) -> Any:
    panel = np.full((277, PANEL_WIDTH, 3), (12, 17, 15), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH - 1, 276), accent, 2)
    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH - 1, 51), (17, 25, 21), -1)
    cv2.rectangle(panel, (0, 0), (7, 51), accent, -1)
    _put_text(cv2, panel, title, (20, 22), scale=0.50, colour=accent, thickness=1)
    _put_text(cv2, panel, subtitle, (20, 42), scale=0.32, colour=(175, 190, 181), thickness=1)
    resized = cv2.resize(frame, (PANEL_WIDTH - 4, PANEL_HEIGHT - 2), interpolation=cv2.INTER_CUBIC)
    panel[53:276, 2:398] = resized
    return panel


def _compose_frame(
    cv2: Any,
    np: Any,
    first_person: Any,
    third_person_a: Any,
    third_person_b: Any,
    *,
    episode_index: int,
    task: str,
    frame_index: int,
    frame_count: int,
) -> Any:
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), (6, 10, 8), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH - 1, 92), (13, 20, 17), -1)
    cv2.rectangle(canvas, (0, 0), (10, 92), (93, 230, 180), -1)
    _put_text(
        cv2,
        canvas,
        "REAL CONDITION  >  REAL THIRD-PERSON TARGETS",
        (28, 37),
        scale=0.84,
        colour=(108, 245, 198),
        thickness=2,
    )
    _put_text(cv2, canvas, task.upper(), (29, 70), scale=0.53, colour=(224, 232, 227), thickness=1)
    _put_text(
        cv2,
        canvas,
        f"OUR COMPOSED DEMO  /  EP {episode_index:02d}  /  {frame_index / FPS:05.2f} S",
        (870, 51),
        scale=0.38,
        colour=(154, 169, 160),
        thickness=1,
    )
    panels = [
        _panel(
            cv2,
            np,
            first_person,
            title="REAL CONDITION / FIRST-PERSON",
            subtitle="camera:wrist_image_left_pixels",
            accent=(255, 190, 78),
        ),
        _panel(
            cv2,
            np,
            third_person_a,
            title="REAL TARGET A / THIRD-PERSON",
            subtitle="camera:exterior_image_1_left_pixels",
            accent=(108, 245, 198),
        ),
        _panel(
            cv2,
            np,
            third_person_b,
            title="REAL TARGET B / THIRD-PERSON",
            subtitle="camera:exterior_image_2_left_pixels",
            accent=(108, 245, 198),
        ),
    ]
    for x, panel in zip((20, 440, 860), panels, strict=True):
        canvas[112:389, x : x + PANEL_WIDTH] = panel

    cv2.rectangle(canvas, (20, 421), (1260, 620), (11, 17, 14), -1)
    cv2.rectangle(canvas, (20, 421), (1260, 620), (45, 64, 54), 1)
    _put_text(cv2, canvas, "IDENTITY BASIS", (42, 458), scale=0.42, colour=(129, 149, 137))
    _put_text(
        cv2,
        canvas,
        "SAME PHYSICAL EPISODE / SAME TIMESTAMP / SAME ROBOT + OBJECT + ACTION",
        (42, 492),
        scale=0.55,
        colour=(228, 236, 231),
        thickness=1,
    )
    _put_text(cv2, canvas, "EVIDENCE ROLE", (42, 538), scale=0.42, colour=(129, 149, 137))
    _put_text(
        cv2,
        canvas,
        "REAL CONDITION + REAL MEASURED TARGETS",
        (42, 572),
        scale=0.55,
        colour=(108, 245, 198),
        thickness=1,
    )
    _put_text(
        cv2,
        canvas,
        "OUR GENERATED VIDEO: NOT AVAILABLE / NOT STARTED",
        (692, 572),
        scale=0.44,
        colour=(95, 165, 255),
        thickness=1,
    )

    progress = (frame_index + 1) / frame_count
    cv2.rectangle(canvas, (20, 660), (1260, 668), (35, 47, 40), -1)
    cv2.rectangle(canvas, (20, 660), (20 + round(1240 * progress), 668), (93, 230, 180), -1)
    phase = "EARLY" if progress < 0.34 else "MID" if progress < 0.67 else "LATE"
    _put_text(
        cv2,
        canvas,
        f"SYNCHRONIZED FRAME {frame_index + 1:03d} / {frame_count:03d}  |  {phase} ACTION PHASE",
        (20, 700),
        scale=0.45,
        colour=(149, 165, 155),
    )
    return canvas


def _write_video(
    ffmpeg: str,
    destination: Path,
    frames: Any,
    *,
    log_path: Path,
) -> None:
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
        f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}",
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
        raise ValueError("reel requires exactly three task videos")
    filter_parts = []
    labels = []
    for index, source in enumerate(sources):
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=nw=1:nk=1",
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        total = int(probe.stdout.strip())
        start = max(0, (total - REEL_FRAMES_PER_TASK) // 2)
        end = start + REEL_FRAMES_PER_TASK
        filter_parts.append(f"[{index}:v]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS[v{index}]")
        labels.append(f"[v{index}]")
    filter_parts.append("".join(labels) + f"concat=n={len(sources)}:v=1:a=0[outv]")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for source in sources:
        command.extend(("-i", str(source)))
    command.extend(
        (
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[outv]",
            "-an",
            "-r",
            f"{FPS:g}",
            "-frames:v",
            str(REEL_FRAMES_PER_TASK * len(sources)),
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
        completed = subprocess.run(command, check=False, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg failed to write reel with code {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/droid-ego-third-person"))
    parser.add_argument("--publish-dir", type=Path, default=Path("demo/showcase"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    import cv2
    import numpy as np

    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    ffmpeg = shutil.which(args.ffmpeg) if not Path(args.ffmpeg).is_absolute() else args.ffmpeg
    ffprobe = shutil.which(args.ffprobe) if not Path(args.ffprobe).is_absolute() else args.ffprobe
    if not ffmpeg or not Path(ffmpeg).is_file():
        raise ValueError(f"ffmpeg is unavailable: {args.ffmpeg}")
    if not ffprobe or not Path(ffprobe).is_file():
        raise ValueError(f"ffprobe is unavailable: {args.ffprobe}")

    if args.experiment_dir:
        experiment = args.experiment_dir.expanduser().resolve()
        experiment.mkdir(parents=True, exist_ok=True)
        if (experiment / "manifest.json").exists():
            raise ValueError(f"experiment already concluded: {experiment}")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        experiment = args.experiment_root.expanduser().resolve() / timestamp
        experiment.mkdir(parents=True, exist_ok=False)
    outputs = experiment / "final"
    outputs.mkdir(parents=True, exist_ok=False)
    publish_dir = args.publish_dir.expanduser().resolve()
    publish_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment / "render.log"
    (experiment / "command.txt").write_text(" ".join(sys.argv) + "\n")
    shutil.copy2(config_path, experiment / "input-config.json")
    _write_json(experiment / "git-state.json", _git_state())

    stream_paths = {
        name: _require_file(stream["path"], label=name)
        for name, stream in config["streams"].items()
    }
    probes = {name: _probe_video(str(ffprobe), path) for name, path in stream_paths.items()}
    contract = _stream_contract(probes)
    if not contract["passed"]:
        raise ValueError(f"input streams do not share one frame contract: {contract}")
    if next(iter(probes.values()))["fps"] != FPS:
        raise ValueError(f"input streams must be exactly {FPS:g} FPS")
    width = next(iter(probes.values()))["width"]
    height = next(iter(probes.values()))["height"]

    task_records: dict[str, Any] = {}
    task_videos: list[Path] = []
    for episode in config["episodes"]:
        label = str(episode["label"])
        start_frame = int(episode["start_frame"])
        frame_count = int(episode["frame_count"])
        decoded = {
            name: _decode_slice(
                str(ffmpeg),
                np,
                path,
                start_frame=start_frame,
                frame_count=frame_count,
                width=width,
                height=height,
            )
            for name, path in stream_paths.items()
        }
        if {array.shape[0] for array in decoded.values()} != {frame_count}:
            raise RuntimeError(f"decoded view lengths differ for {label}")
        frames = (
            _compose_frame(
                cv2,
                np,
                decoded["first_person"][index],
                decoded["third_person_a"][index],
                decoded["third_person_b"][index],
                episode_index=int(episode["episode_index"]),
                task=str(episode["task"]),
                frame_index=index,
                frame_count=frame_count,
            )
            for index in range(frame_count)
        )
        video = outputs / f"{_slug(label)}-ego-to-third-person.mp4"
        _write_video(str(ffmpeg), video, frames, log_path=log_path)

        midpoint = frame_count // 2
        poster_frame = _compose_frame(
            cv2,
            np,
            decoded["first_person"][midpoint],
            decoded["third_person_a"][midpoint],
            decoded["third_person_b"][midpoint],
            episode_index=int(episode["episode_index"]),
            task=str(episode["task"]),
            frame_index=midpoint,
            frame_count=frame_count,
        )
        poster = outputs / f"{_slug(label)}-ego-to-third-person-poster.jpg"
        if not cv2.imwrite(str(poster), poster_frame, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"could not write {poster}")
        ranges = list(episode["timestamps"].values())
        maximum_timestamp_delta = max(
            max(abs(float(a[key]) - float(b[key])) for key in ("from_seconds", "to_seconds"))
            for a in ranges
            for b in ranges
        )
        task_records[label] = {
            "episode_index": int(episode["episode_index"]),
            "task": str(episode["task"]),
            "dataset_frame_interval": [start_frame, start_frame + frame_count],
            "frame_count": frame_count,
            "fps": FPS,
            "duration_seconds": frame_count / FPS,
            "timestamps": episode["timestamps"],
            "maximum_cross_camera_timestamp_delta_seconds": maximum_timestamp_delta,
            "trajectory_lineage_passed": maximum_timestamp_delta == 0.0,
            "identity_consistency_basis": (
                "All panels are measured cameras from the same physical DROID episode and timestamp; "
                "no subject is regenerated."
            ),
            "video": str(video),
            "video_sha256": _sha256(video),
            "poster": str(poster),
            "poster_sha256": _sha256(poster),
        }
        task_videos.append(video)

    reel = outputs / "ego-to-third-person-three-task-reel-10s.mp4"
    _write_reel(str(ffmpeg), task_videos, reel, log_path)
    reel_capture = cv2.VideoCapture(str(reel))
    reel_capture.set(cv2.CAP_PROP_POS_FRAMES, REEL_FRAMES_PER_TASK + REEL_FRAMES_PER_TASK // 2)
    ok, reel_midpoint = reel_capture.read()
    reel_capture.release()
    reel_poster = outputs / "ego-to-third-person-three-task-reel-poster.jpg"
    if not ok or not cv2.imwrite(str(reel_poster), reel_midpoint, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"could not write {reel_poster}")

    published: dict[str, str] = {}
    for label, record in task_records.items():
        for kind in ("video", "poster"):
            source = Path(record[kind])
            destination = publish_dir / f"acwm-ego-third-person-{_slug(label)}-{kind}{source.suffix}"
            shutil.copy2(source, destination)
            published[f"{label}_{kind}"] = str(destination)
    reel_public = publish_dir / "acwm-ego-third-person-reel-10s.mp4"
    reel_poster_public = publish_dir / "acwm-ego-third-person-reel-poster.jpg"
    shutil.copy2(reel, reel_public)
    shutil.copy2(reel_poster, reel_poster_public)
    published["reel_video"] = str(reel_public)
    published["reel_poster"] = str(reel_poster_public)

    package_versions: dict[str, str] = {}
    for package in ("numpy", "opencv-python"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "unavailable"
    manifest = {
        "schema_version": "1.0.0",
        "status": "WORKING",
        "capability_status": "NOT STARTED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": package_versions,
        "seed": SEED,
        "dataset": config["dataset"],
        "coordinate_frames": {
            name: stream["coordinate_frame"] for name, stream in config["streams"].items()
        },
        "source_streams": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "probe": probes[name],
            }
            for name, path in stream_paths.items()
        },
        "stream_contract": contract,
        "tasks": task_records,
        "task_count": len(task_records),
        "all_trajectory_lineage_passed": all(
            bool(record["trajectory_lineage_passed"]) for record in task_records.values()
        ),
        "reel": {
            "path": str(reel),
            "sha256": _sha256(reel),
            "frames": REEL_FRAMES_PER_TASK * len(task_records),
            "fps": FPS,
            "duration_seconds": REEL_FRAMES_PER_TASK * len(task_records) / FPS,
        },
        "published": {
            name: {"path": path, "sha256": _sha256(Path(path))}
            for name, path in published.items()
        },
        "claim_boundary": (
            "WORKING synchronized real multi-camera first-person/third-person paired demo. "
            "The exterior views are measured ground-truth targets, not model-generated novel views. "
            "A learned ego-to-third-person generation capability remains NOT STARTED."
        ),
    }
    _write_json(experiment / "manifest.json", manifest)
    public_manifest = publish_dir / "acwm-ego-third-person-manifest.json"
    shutil.copy2(experiment / "manifest.json", public_manifest)
    print(
        json.dumps(
            {
                "experiment": str(experiment),
                "published": published,
                "stream_contract_passed": contract["passed"],
                "trajectory_lineage_passed": manifest["all_trajectory_lineage_passed"],
                "capability_status": manifest["capability_status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
