#!/usr/bin/env python3
"""Compile pinned WorldArena2.0 real-robot episodes for BWM factory runs.

H5Py, NumPy, and PyArrow remain optional and are imported only by this entry
point. Importing ``phiagent`` does not require them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import shutil
import subprocess
import sys
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.worldarena import WORLD_ARENA_EEF_QUATERNION_CHANNELS  # noqa: E402


DATASET_ID = "WorldArena/WorldArena2.0"
DATASET_REVISION = "af1ac34d3881f84096345542c631fbb1b9540d50"
DATASET_LICENSE = "apache-2.0"
HF_ROOT = "https://huggingface.co/datasets/WorldArena/WorldArena2.0"


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


def _request_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "PhiAgent/0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"download destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "PhiAgent/0"})
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)
    if partial.stat().st_size == 0:
        raise ValueError(f"download is empty: {url}")
    partial.replace(destination)


def _video_info(ffprobe: Path, video: Path) -> dict[str, int | float | str]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,codec_name",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    numerator, denominator = str(stream["r_frame_rate"]).split("/", maxsplit=1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(numerator) / int(denominator),
        "frames": int(stream["nb_frames"]),
        "duration_seconds": float(payload["format"]["duration"]),
        "codec": str(stream["codec_name"]),
    }


def motion_windows(
    values: Any,
    *,
    num_frames: int,
    count: int,
    stride: int = 4,
) -> tuple[int, ...]:
    """Select high-motion non-overlapping windows deterministically."""

    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 14:
        raise ValueError(f"end_pose must be N x 14, found {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("end_pose contains non-finite values")
    if num_frames < 5 or (num_frames - 1) % 4:
        raise ValueError("num_frames must equal 4n+1")
    if count <= 0 or stride <= 0 or len(array) < num_frames:
        raise ValueError("invalid motion-window selection parameters")
    deltas = np.linalg.norm(np.diff(array, axis=0), axis=1)
    candidates = []
    for start in range(0, len(array) - num_frames + 1, stride):
        score = float(np.sum(deltas[start : start + num_frames - 1]))
        candidates.append((score, start))
    selected: list[int] = []
    for _, start in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if all(abs(start - previous) >= num_frames for previous in selected):
            selected.append(start)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"episode cannot provide {count} non-overlapping {num_frames}-frame windows"
        )
    return tuple(sorted(selected))


def terminal_video_frames_excluded(video_frames: int, action_frames: int) -> int:
    """Accept exact alignment or one dataset terminal-frame sentinel."""

    if action_frames <= 0 or video_frames not in {action_frames, action_frames + 1}:
        raise ValueError(
            "video/action alignment requires equal counts or one extra terminal video frame"
        )
    return video_frames - action_frames


def _task_episodes(task: str, count: int) -> tuple[str, ...]:
    payload = _request_json(
        f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/{DATASET_REVISION}/{task}"
        "?expand=false&limit=1000"
    )
    if not isinstance(payload, list):
        raise ValueError(f"dataset API returned an invalid task listing for {task}")
    episodes = []
    for item in payload:
        if not isinstance(item, dict) or item.get("type") != "directory":
            continue
        name = str(item.get("path", "")).split("/")[-1]
        if name.startswith("episode_") and name[8:].isdigit():
            episodes.append(name)
    episodes.sort(key=lambda name: int(name[8:]))
    if len(episodes) < count:
        raise ValueError(f"task {task} has only {len(episodes)} numbered episodes")
    return tuple(episodes[:count])


def _cached_task_episodes(cache: Path, task: str, count: int) -> tuple[str, ...]:
    episodes = sorted(
        (
            path.name
            for path in (cache / task).glob("episode_*")
            if path.is_dir() and path.name[8:].isdigit()
        ),
        key=lambda name: int(name[8:]),
    )
    if len(episodes) < count:
        raise ValueError(f"source cache task {task} has only {len(episodes)} episodes")
    return tuple(episodes[:count])


def _source_cache_index(cache: Path) -> dict[str, dict[str, object]]:
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"source cache lacks manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise ValueError("source cache manifest is not completed")
    expected = {
        "dataset_id": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("source cache identity, revision, or license does not match")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("source cache manifest lacks a files array")
    index = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("source cache file record is invalid")
        index[str(item["path"])] = item
    return index


def _stats(values: Iterable[Any], np: Any) -> dict[str, object]:
    rows = [np.asarray(value, dtype=np.float64) for value in values]
    combined = np.concatenate(rows, axis=0)
    if combined.ndim != 2 or combined.shape[1] != 14:
        raise ValueError("training action statistics require N x 14 arrays")
    return {
        "state_pose": {
            "shape": [14],
            "min": combined.min(axis=0).tolist(),
            "max": combined.max(axis=0).tolist(),
            "p01": np.percentile(combined, 1, axis=0).tolist(),
            "p99": np.percentile(combined, 99, axis=0).tolist(),
            "mean": combined.mean(axis=0).tolist(),
            "std": combined.std(axis=0).tolist(),
            "frame_count": int(len(combined)),
            "coordinate_frame": "robot_base:worldarena2-cobot-magic-max-end-pose",
            "channels": list(WORLD_ARENA_EEF_QUATERNION_CHANNELS),
        }
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-task", action="append", required=True)
    parser.add_argument("--validation-task", action="append", required=True)
    parser.add_argument("--test-task", action="append", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--clips-per-episode", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=57)
    parser.add_argument("--history-frames", type=int, default=9)
    parser.add_argument("--source-revision", default=DATASET_REVISION)
    parser.add_argument(
        "--source-cache",
        type=Path,
        help="offline pinned source cache from download_worldarena2_factory_source.py",
    )
    parser.add_argument("--ffprobe", type=Path, default=Path("/usr/bin/ffprobe"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"dataset experiment already exists: {output}")
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    splits = {
        "train": tuple(args.train_task),
        "validation": tuple(args.validation_task),
        "test": tuple(args.test_task),
    }
    started = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "preparing",
        "honest_status": "PARTIAL",
        "created_at": started,
        "method": "pinned_worldarena2_real_robot_to_bwm_factory_clips",
        "command": [sys.executable, *sys.argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": {"used": False, "reason": "dataset download and deterministic CPU compilation"},
        "source": {
            "dataset_id": DATASET_ID,
            "revision": args.source_revision,
            "license": DATASET_LICENSE,
            "cache": None
            if args.source_cache is None
            else str(args.source_cache.expanduser().resolve()),
        },
        "config": {
            "splits": {key: list(value) for key, value in splits.items()},
            "episodes_per_task": args.episodes_per_task,
            "clips_per_episode": args.clips_per_episode,
            "num_frames": args.num_frames,
            "history_frames": args.history_frames,
        },
        "episodes": [],
    }
    _write_json(manifest_path, manifest)
    try:
        if args.source_revision != DATASET_REVISION:
            raise ValueError(f"source revision must equal pinned {DATASET_REVISION}")
        if args.episodes_per_task <= 0 or args.clips_per_episode <= 0:
            raise ValueError("episode and clip counts must be positive")
        if not 1 <= args.history_frames < args.num_frames:
            raise ValueError("history_frames must be inside the generated clip")
        all_tasks = [task for tasks in splits.values() for task in tasks]
        if len(all_tasks) != len(set(all_tasks)):
            raise ValueError("train, validation, and test tasks must be disjoint")
        try:
            import h5py
            import numpy as np
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError(
                "WorldArena2 compilation requires optional h5py, numpy, and pyarrow"
            ) from error
        ffprobe = args.ffprobe.expanduser().resolve()
        if not ffprobe.is_file():
            raise ValueError(f"ffprobe does not exist: {ffprobe}")
        records: dict[str, list[dict[str, object]]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        training_values = []
        episode_reports = []
        sample_index = 0
        source_cache = (
            None
            if args.source_cache is None
            else args.source_cache.expanduser().resolve()
        )
        cache_index = None if source_cache is None else _source_cache_index(source_cache)
        for split, tasks in splits.items():
            for task in tasks:
                episode_names = (
                    _task_episodes(task, args.episodes_per_task)
                    if source_cache is None
                    else _cached_task_episodes(source_cache, task, args.episodes_per_task)
                )
                for episode_name in episode_names:
                    relative_root = Path("assets") / task / episode_name
                    episode_root = output / relative_root
                    urls = {}
                    paths = {}
                    for source_name, destination_name in (
                        ("meta.json", "meta.json"),
                        ("episode.hdf5", "episode.hdf5"),
                        ("cam_high.mp4", "cam_high.mp4"),
                    ):
                        url = (
                            f"{HF_ROOT}/resolve/{args.source_revision}/{task}/"
                            f"{episode_name}/{source_name}"
                        )
                        destination = episode_root / destination_name
                        if source_cache is None:
                            _download(url, destination)
                        else:
                            relative_cache_path = str(Path(task) / episode_name / source_name)
                            item = cache_index.get(relative_cache_path)  # type: ignore[union-attr]
                            cached = source_cache / relative_cache_path
                            if not isinstance(item, dict) or not cached.is_file():
                                raise ValueError(f"source cache is missing {relative_cache_path}")
                            if item.get("sha256") != _sha256(cached):
                                raise ValueError(
                                    f"source cache hash mismatch for {relative_cache_path}"
                                )
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(cached, destination)
                        urls[source_name] = url
                        paths[source_name] = destination
                    meta = json.loads(paths["meta.json"].read_text())
                    with h5py.File(paths["episode.hdf5"], "r") as handle:
                        end_pose = np.asarray(handle["observations/end_pose"], dtype=np.float32)
                        action = np.asarray(handle["action"], dtype=np.float32)
                    if end_pose.shape != action.shape or end_pose.ndim != 2 or end_pose.shape[1] != 14:
                        raise ValueError(
                            f"{task}/{episode_name} action/end_pose shapes are incompatible"
                        )
                    video_info = _video_info(ffprobe, paths["cam_high.mp4"])
                    excluded_terminal_frames = terminal_video_frames_excluded(
                        int(video_info["frames"]), len(end_pose)
                    )
                    starts = motion_windows(
                        end_pose,
                        num_frames=args.num_frames,
                        count=args.clips_per_episode,
                    )
                    action_relative = Path("actions") / task / episode_name / "end-pose.parquet"
                    action_path = output / action_relative
                    action_path.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(
                        pa.table(
                            {
                                "observation.state": pa.array(
                                    end_pose.tolist(), type=pa.list_(pa.float32(), 14)
                                ),
                                "action": pa.array(
                                    action.tolist(), type=pa.list_(pa.float32(), 14)
                                ),
                            }
                        ),
                        action_path,
                        compression="zstd",
                    )
                    if split == "train":
                        training_values.extend(
                            end_pose[start : start + args.num_frames] for start in starts
                        )
                    for clip_number, start in enumerate(starts):
                        end = start + args.num_frames - 1
                        records[split].append(
                            {
                                "episode_index": sample_index,
                                "source_episode": f"{task}/{episode_name}",
                                "group_id": f"{task}-{episode_name}",
                                "task": task,
                                "split": split,
                                "clip_index": clip_number,
                                "length": args.num_frames,
                                "start_frame": start,
                                "end_frame": end,
                                "history_frames": args.history_frames,
                                "video": {
                                    "data": str(relative_root / "cam_high.mp4"),
                                    "start_frame": start,
                                    "end_frame": end,
                                },
                                "action": {
                                    "data": str(action_relative),
                                    "start_frame": start,
                                    "end_frame": end,
                                },
                                "prompt": str(meta.get("instruction") or meta.get("task_name") or task),
                                "coordinate_frame": (
                                    "robot_base:worldarena2-cobot-magic-max-end-pose"
                                ),
                                "license_id": DATASET_LICENSE,
                                "source_uri": urls["cam_high.mp4"],
                            }
                        )
                        sample_index += 1
                    episode_reports.append(
                        {
                            "task": task,
                            "episode": episode_name,
                            "split": split,
                            "meta": meta,
                            "selected_window_starts": list(starts),
                            "video": {
                                "path": str(paths["cam_high.mp4"]),
                                "sha256": _sha256(paths["cam_high.mp4"]),
                                "probe": video_info,
                                "source_uri": urls["cam_high.mp4"],
                                "terminal_frames_excluded": excluded_terminal_frames,
                            },
                            "hdf5": {
                                "path": str(paths["episode.hdf5"]),
                                "sha256": _sha256(paths["episode.hdf5"]),
                                "source_uri": urls["episode.hdf5"],
                                "end_pose_shape": list(end_pose.shape),
                            },
                            "compiled_action": {
                                "path": str(action_path),
                                "sha256": _sha256(action_path),
                            },
                        }
                    )
        for split, rows in records.items():
            (output / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )
        all_rows = [row for split in ("train", "validation", "test") for row in records[split]]
        (output / "all.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows)
        )
        _write_json(output / "action-stat.json", _stats(training_values, np))
        manifest.update(
            {
                "status": "completed",
                "honest_status": "WORKING",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "episodes": episode_reports,
                "records": {split: len(rows) for split, rows in records.items()},
                "artifacts": {
                    name: {
                        "path": str(output / name),
                        "sha256": _sha256(output / name),
                    }
                    for name in (
                        "train.jsonl",
                        "validation.jsonl",
                        "test.jsonl",
                        "all.jsonl",
                        "action-stat.json",
                    )
                },
                "limitations": [
                    "WorldArena2.0 end_pose is treated as dataset-declared robot-base EEF data; no independent calibration was performed.",
                    "These real-robot videos are model inputs and frozen references, not evidence that generated rollouts are physically executable.",
                    "Automatic motion-window selection favors high EEF motion and does not prove task success or contact quality.",
                    "A single terminal video frame beyond the HDF5 action length is excluded when present; all selected windows remain inside the action-aligned prefix.",
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
