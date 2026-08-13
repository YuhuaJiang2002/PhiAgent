#!/usr/bin/env python3
"""Compile a pinned LeRobot RoboTwin export into frame-explicit BWM metadata.

PyArrow and NumPy are optional runtime dependencies and are imported only after
the source revision and output-directory contracts have been validated.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.robotwin import (  # noqa: E402
    BWM_EEF_CHANNELS,
    ROBOTWIN_FPS,
    ROBOTWIN_LEROBOT_REPOSITORY,
    ROBOTWIN_LEROBOT_REVISION,
    RoboTwinEpisode,
    bwm_clip_record,
    grouped_split,
    overlapping_clip_starts,
    parse_robotwin_task,
)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {"head": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-revision",
        default=ROBOTWIN_LEROBOT_REVISION,
        help="must equal the pinned dataset revision",
    )
    parser.add_argument(
        "--view",
        choices=("head", "front", "left_wrist", "right_wrist"),
        default="head",
    )
    parser.add_argument("--split-seed", type=int, default=20260811)
    parser.add_argument("--num-frames", type=int, default=57)
    parser.add_argument("--history-frames", type=int, default=9)
    parser.add_argument(
        "--max-episodes-per-group",
        type=int,
        default=0,
        help="zero keeps every episode; otherwise caps each embodiment/task group",
    )
    return parser


def _load_source_rows(source_root: Path, view: str, pq: Any) -> list[dict[str, Any]]:
    paths = sorted((source_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise ValueError(f"no LeRobot episode metadata found under {source_root}")
    required = {
        "episode_index",
        "tasks",
        "length",
        "dataset_from_index",
        "data/chunk_index",
        "data/file_index",
        f"videos/observation.images.{view}/chunk_index",
        f"videos/observation.images.{view}/file_index",
        f"videos/observation.images.{view}/from_timestamp",
    }
    rows: list[dict[str, Any]] = []
    for path in paths:
        table = pq.read_table(path)
        missing = required - set(table.column_names)
        if missing:
            raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
        rows.extend(table.select(sorted(required)).to_pylist())
    return rows


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    split_seed: int,
    maximum_per_group: int,
) -> tuple[list[dict[str, Any]], dict[int, tuple[str, str, str, str]]]:
    if maximum_per_group < 0:
        raise ValueError("max-episodes-per-group must be non-negative")
    selected: list[dict[str, Any]] = []
    identities: dict[int, tuple[str, str, str, str]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    for row in sorted(rows, key=lambda item: int(item["episode_index"])):
        tasks = row["tasks"]
        if not isinstance(tasks, list) or len(tasks) != 1:
            raise ValueError(f"episode {row['episode_index']} must have exactly one task")
        embodiment, task, instruction = parse_robotwin_task(str(tasks[0]))
        group = (embodiment, task)
        if maximum_per_group and counts[group] >= maximum_per_group:
            continue
        counts[group] += 1
        split = grouped_split(embodiment, task, seed=split_seed)
        episode_index = int(row["episode_index"])
        if episode_index in identities:
            raise ValueError(f"duplicate episode index: {episode_index}")
        identities[episode_index] = (embodiment, task, instruction, split)
        selected.append(row)
    if not selected:
        raise ValueError("episode selection is empty")
    return selected, identities


def _convert_eef_array(source: Any, np: Any) -> Any:
    """Vectorized ``xyzw -> Euler XYZ`` conversion for an N x 16 array."""

    values = np.asarray(source, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError(f"expected an N x 16 EE array, found {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("source EE array contains non-finite values")
    result = np.empty((len(values), 14), dtype=np.float32)
    for source_offset, target_offset in ((0, 0), (8, 7)):
        result[:, target_offset : target_offset + 3] = values[
            :, source_offset : source_offset + 3
        ]
        quaternion = values[:, source_offset + 3 : source_offset + 7].astype(
            np.float64, copy=True
        )
        norms = np.linalg.norm(quaternion, axis=1)
        if np.any(norms < 1e-8):
            raise ValueError("source EE array contains a zero-norm quaternion")
        quaternion /= norms[:, None]
        x, y, z, w = quaternion.T
        result[:, target_offset + 3] = np.arctan2(
            2 * (w * x + y * z), 1 - 2 * (x * x + y * y)
        )
        result[:, target_offset + 4] = np.arcsin(
            np.clip(2 * (w * y - z * x), -1.0, 1.0)
        )
        result[:, target_offset + 5] = np.arctan2(
            2 * (w * z + x * y), 1 - 2 * (y * y + z * z)
        )
        gripper = values[:, source_offset + 7]
        if np.any(gripper < -1e-6) or np.any(gripper > 1 + 1e-6):
            raise ValueError("source EE array contains a gripper value outside [0, 1]")
        result[:, target_offset + 6] = np.clip(gripper, 0, 1)
    return result


class _RunningStats:
    def __init__(self, np: Any) -> None:
        self.np = np
        self.count = 0
        self.minimum = np.full(14, np.inf, dtype=np.float64)
        self.maximum = np.full(14, -np.inf, dtype=np.float64)
        self.total = np.zeros(14, dtype=np.float64)
        self.total_square = np.zeros(14, dtype=np.float64)

    def update(self, values: Any) -> None:
        if len(values) == 0:
            return
        as_float = values.astype(self.np.float64, copy=False)
        self.count += len(as_float)
        self.minimum = self.np.minimum(self.minimum, as_float.min(axis=0))
        self.maximum = self.np.maximum(self.maximum, as_float.max(axis=0))
        self.total += as_float.sum(axis=0)
        self.total_square += self.np.square(as_float).sum(axis=0)

    def payload(self) -> dict[str, object]:
        if self.count == 0:
            raise ValueError("training split contains no action frames")
        mean = self.total / self.count
        variance = self.np.maximum(self.total_square / self.count - mean * mean, 0)
        return {
            "state_pose": {
                "shape": [14],
                "min": self.minimum.tolist(),
                "max": self.maximum.tolist(),
                "mean": mean.tolist(),
                "std": self.np.sqrt(variance).tolist(),
                "frame_count": self.count,
                "coordinate_frame_policy": "robot_base:robotwin2-<embodiment>",
                "channels": list(BWM_EEF_CHANNELS),
            }
        }


def main() -> int:
    args = _parser().parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    project_root = Path(__file__).resolve().parents[1]
    if output_root.exists():
        raise FileExistsError(f"dataset attempt directory already exists: {output_root}")
    output_root.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    config = {
        "schema_version": "1.0.0",
        "source_repository": ROBOTWIN_LEROBOT_REPOSITORY,
        "source_revision": args.source_revision,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "view": args.view,
        "split_seed": args.split_seed,
        "num_frames": args.num_frames,
        "history_frames": args.history_frames,
        "max_episodes_per_group": args.max_episodes_per_group,
        "started_at": started,
        "hostname": socket.gethostname(),
        "git": _git_state(project_root),
        "command": sys.argv,
    }
    _write_json(output_root / "config.json", config)
    try:
        if args.source_revision != ROBOTWIN_LEROBOT_REVISION:
            raise ValueError(
                f"source revision {args.source_revision} is not pinned revision "
                f"{ROBOTWIN_LEROBOT_REVISION}"
            )
        marker = source_root / ".phiagent-dataset-revision"
        if not marker.is_file() or marker.read_text().strip() != args.source_revision:
            raise ValueError(
                f"source dataset lacks exact revision marker {marker}; download it with "
                "the pinned preparation command"
            )
        try:
            import numpy as np
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "RoboTwin conversion requires optional numpy and pyarrow packages"
            ) from exc

        rows = _load_source_rows(source_root, args.view, pq)
        selected, identities = _select_rows(
            rows,
            split_seed=args.split_seed,
            maximum_per_group=args.max_episodes_per_group,
        )
        rows_by_data_file: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
            rows_by_data_file[key].append(row)

        starts: dict[int, int] = {}
        action_paths: dict[int, Path] = {}
        stats = _RunningStats(np)
        for (chunk_index, file_index), file_rows in sorted(rows_by_data_file.items()):
            relative = Path("data") / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
            source_path = source_root / relative
            if not source_path.is_file():
                raise ValueError(f"missing source action file: {source_path}")
            table = pq.read_table(
                source_path,
                columns=["episode_index", "frame_index", "index", "observation.state_ee"],
            )
            episode_ids = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
            converted = _convert_eef_array(table["observation.state_ee"].to_pylist(), np)
            selected_ids = {int(row["episode_index"]) for row in file_rows}
            keep = np.isin(episode_ids, list(selected_ids))
            # Preserve packed-file offsets: non-selected rows remain in the converted file.
            destination = output_root / "actions" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            converted_table = pa.table(
                {
                    "episode_index": table["episode_index"],
                    "frame_index": table["frame_index"],
                    "index": table["index"],
                    "observation.state": pa.array(
                        converted.tolist(), type=pa.list_(pa.float32(), 14)
                    ),
                }
            )
            pq.write_table(converted_table, destination, compression="zstd")
            for episode_index in selected_ids:
                positions = np.flatnonzero(episode_ids == episode_index)
                expected = int(next(
                    row["length"]
                    for row in file_rows
                    if int(row["episode_index"]) == episode_index
                ))
                if len(positions) != expected or np.any(np.diff(positions) != 1):
                    raise ValueError(
                        f"episode {episode_index} is not one contiguous {expected}-frame slice "
                        f"of {source_path}"
                    )
                starts[episode_index] = int(positions[0])
                action_paths[episode_index] = destination
            training_ids = {
                episode_index
                for episode_index in selected_ids
                if identities[episode_index][3] == "train"
            }
            stats.update(converted[keep & np.isin(episode_ids, list(training_ids))])

        records: dict[str, list[dict[str, object]]] = defaultdict(list)
        group_splits: dict[str, str] = {}
        episode_counts: Counter[str] = Counter()
        clip_counts: Counter[str] = Counter()
        for row in selected:
            episode_index = int(row["episode_index"])
            embodiment, task, instruction, split = identities[episode_index]
            video_chunk = int(row[f"videos/observation.images.{args.view}/chunk_index"])
            video_file = int(row[f"videos/observation.images.{args.view}/file_index"])
            video_path = (
                source_root
                / "videos"
                / f"observation.images.{args.view}"
                / f"chunk-{video_chunk:03d}"
                / f"file-{video_file:03d}.mp4"
            )
            if not video_path.is_file():
                raise ValueError(f"missing source video file: {video_path}")
            episode = RoboTwinEpisode(
                episode_index=episode_index,
                embodiment=embodiment,
                task=task,
                instruction=instruction,
                length=int(row["length"]),
                data_path=str(action_paths[episode_index]),
                video_path=str(video_path),
                data_start_frame=starts[episode_index],
                video_start_frame=round(
                    float(row[f"videos/observation.images.{args.view}/from_timestamp"])
                    * ROBOTWIN_FPS
                ),
                coordinate_frame=f"robot_base:robotwin2-{embodiment}",
            )
            group_splits[episode.group] = split
            episode_counts[split] += 1
            for clip_start in overlapping_clip_starts(
                episode.length,
                num_frames=args.num_frames,
                history=args.history_frames,
            ):
                record = bwm_clip_record(
                    episode, clip_start=clip_start, num_frames=args.num_frames
                )
                record["split"] = split
                records[split].append(record)
                clip_counts[split] += 1

        for split in ("train", "validation", "test"):
            metadata = output_root / f"{split}.jsonl"
            metadata.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in records[split])
            )
        _write_json(output_root / "action-stat.json", stats.payload())
        result = {
            **config,
            "status": "WORKING",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "selected_episodes": dict(episode_counts),
            "compiled_clips": dict(clip_counts),
            "group_splits": group_splits,
            "action_statistics": str(output_root / "action-stat.json"),
            "limitations": [
                "This artifact prepares data; it is not evidence that BWM training converges.",
                "RoboTwin2.0 is simulated data and is not a real-robot demonstration.",
                "Euler angles have wrap discontinuities at plus/minus pi.",
            ],
        }
        _write_json(output_root / "manifest.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        _write_json(
            output_root / "failure.json",
            {
                **config,
                "status": "BLOCKED",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
