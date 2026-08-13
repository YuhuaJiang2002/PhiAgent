#!/usr/bin/env python3
"""Materialize compact, pinned WorldArena2 factory clips for offline GPU hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from prepare_worldarena2_bwm_factory_data import (
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    motion_windows,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_index(source: Path) -> dict[str, dict[str, object]]:
    manifest_path = source / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    expected = {
        "status": "completed",
        "dataset_id": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("source cache manifest identity or status is invalid")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("source cache manifest lacks files")
    return {str(item["path"]): item for item in files if isinstance(item, dict)}


def _verified_path(
    source: Path, index: dict[str, dict[str, object]], relative: Path
) -> tuple[Path, dict[str, object]]:
    item = index.get(str(relative))
    path = source / relative
    if not isinstance(item, dict) or not path.is_file():
        raise ValueError(f"source cache is missing {relative}")
    if item.get("sha256") != _sha256(path):
        raise ValueError(f"source cache hash mismatch for {relative}")
    return path, item


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--clips-per-episode", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=57)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--crf", type=int, default=27)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.source_cache.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"clip cache experiment already exists: {output}")
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "preparing",
        "honest_status": "PARTIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "motion_selected_lossy_transfer_cache",
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": {"used": False, "reason": "deterministic CPU clip materialization"},
        "dataset_id": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
        "source_cache": str(source),
        "tasks": list(args.task),
        "config": {
            "episodes_per_task": args.episodes_per_task,
            "clips_per_episode": args.clips_per_episode,
            "num_frames": args.num_frames,
            "width": args.width,
            "height": args.height,
            "crf": args.crf,
        },
    }
    _write_json(manifest_path, manifest)
    try:
        import h5py
        import numpy as np

        if len(args.task) != len(set(args.task)):
            raise ValueError("tasks must be unique")
        if args.episodes_per_task <= 0 or args.clips_per_episode <= 0:
            raise ValueError("episode and clip counts must be positive")
        if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
            raise ValueError("width and height must be positive multiples of 16")
        if not 0 <= args.crf <= 51:
            raise ValueError("crf must be between 0 and 51")
        ffmpeg = args.ffmpeg.expanduser().resolve()
        if not ffmpeg.is_file():
            raise ValueError(f"ffmpeg does not exist: {ffmpeg}")
        index = _source_index(source)
        reports = []
        files = []
        total_source_video_bytes = 0
        for task in args.task:
            episode_roots = sorted(
                (path for path in (source / task).glob("episode_*") if path.is_dir()),
                key=lambda path: int(path.name[8:]),
            )[: args.episodes_per_task]
            if len(episode_roots) != args.episodes_per_task:
                raise ValueError(f"source cache task {task} lacks requested episodes")
            derived_episode = 0
            for episode_root in episode_roots:
                source_hdf5, hdf5_item = _verified_path(
                    source, index, Path(task) / episode_root.name / "episode.hdf5"
                )
                source_video, video_item = _verified_path(
                    source, index, Path(task) / episode_root.name / "cam_high.mp4"
                )
                source_meta, meta_item = _verified_path(
                    source, index, Path(task) / episode_root.name / "meta.json"
                )
                total_source_video_bytes += source_video.stat().st_size
                with h5py.File(source_hdf5, "r") as handle:
                    end_pose = np.asarray(handle["observations/end_pose"], dtype=np.float32)
                    action = np.asarray(handle["action"], dtype=np.float32)
                if end_pose.shape != action.shape:
                    raise ValueError(f"shape mismatch in {task}/{episode_root.name}")
                starts = motion_windows(
                    end_pose, num_frames=args.num_frames, count=args.clips_per_episode
                )
                source_metadata = json.loads(source_meta.read_text())
                for source_clip, start in enumerate(starts):
                    relative_root = Path(task) / f"episode_{derived_episode}"
                    target_root = output / relative_root
                    target_root.mkdir(parents=True)
                    target_hdf5 = target_root / "episode.hdf5"
                    target_video = target_root / "cam_high.mp4"
                    target_meta = target_root / "meta.json"
                    stop = start + args.num_frames
                    with h5py.File(target_hdf5, "w") as handle:
                        observations = handle.create_group("observations")
                        observations.create_dataset("end_pose", data=end_pose[start:stop])
                        handle.create_dataset("action", data=action[start:stop])
                    metadata = dict(source_metadata)
                    metadata.update(
                        {
                            "source_episode": f"{task}/{episode_root.name}",
                            "source_clip_index": source_clip,
                            "source_start_frame": start,
                            "source_end_frame": stop - 1,
                            "transfer_cache": "lossy_h264_crf",
                        }
                    )
                    _write_json(target_meta, metadata)
                    subprocess.run(
                        [
                            str(ffmpeg),
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            str(source_video),
                            "-vf",
                            (
                                f"trim=start_frame={start}:end_frame={stop},"
                                f"setpts=PTS-STARTPTS,scale={args.width}:{args.height}:"
                                "force_original_aspect_ratio=decrease,"
                                f"pad={args.width}:{args.height}:(ow-iw)/2:(oh-ih)/2"
                            ),
                            "-frames:v",
                            str(args.num_frames),
                            "-an",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "fast",
                            "-crf",
                            str(args.crf),
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            str(target_video),
                        ],
                        check=True,
                    )
                    for filename, source_item in (
                        ("meta.json", meta_item),
                        ("episode.hdf5", hdf5_item),
                        ("cam_high.mp4", video_item),
                    ):
                        path = target_root / filename
                        files.append(
                            {
                                "path": str(relative_root / filename),
                                "bytes": path.stat().st_size,
                                "sha256": _sha256(path),
                                "source_uri": source_item.get("source_uri"),
                            }
                        )
                    reports.append(
                        {
                            "task": task,
                            "derived_episode": f"episode_{derived_episode}",
                            "source_episode": episode_root.name,
                            "source_clip_index": source_clip,
                            "source_start_frame": start,
                            "source_end_frame": stop - 1,
                            "video_bytes": target_video.stat().st_size,
                        }
                    )
                    derived_episode += 1
        total_bytes = sum(int(item["bytes"]) for item in files)
        manifest.update(
            {
                "status": "completed",
                "honest_status": "WORKING",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "files": files,
                "file_count": len(files),
                "total_bytes": total_bytes,
                "total_source_video_bytes": total_source_video_bytes,
                "transfer_reduction_ratio": (
                    total_source_video_bytes / total_bytes if total_bytes else None
                ),
                "clips": reports,
                "limitations": [
                    "Video was resized and encoded with lossy H.264 for transfer efficiency.",
                    "The cache is for model training/evaluation, not pixel-fidelity benchmarking.",
                    "Original pinned source URIs and frame ranges remain in the manifest.",
                ],
            }
        )
        _write_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "honest_status": "PARTIAL",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
