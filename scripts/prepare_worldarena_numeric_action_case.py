#!/usr/bin/env python3
"""Prepare one lineage-safe WorldArena numeric BWM action case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.numeric import NumericActionStatistics  # noqa: E402
from phiagent.acwm.robotwin import BWM_EEF_CHANNELS  # noqa: E402
from phiagent.acwm.schema import ACWMActionCondition, ActionRepresentation  # noqa: E402
from phiagent.acwm.worldarena import (  # noqa: E402
    WORLD_ARENA_EEF_QUATERNION_CHANNELS,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def quaternion_norm_bounds(
    rows: Sequence[Sequence[float]],
) -> dict[str, tuple[float, float]]:
    if not rows or any(len(row) != 14 for row in rows):
        raise ValueError("WorldArena end_pose must contain non-empty 14-D rows")
    result = {}
    for arm, start in (("left", 3), ("right", 10)):
        norms = [
            math.sqrt(sum(float(row[index]) ** 2 for index in range(start, start + 4)))
            for row in rows
        ]
        result[arm] = min(norms), max(norms)
    return result


def correct_worldarena_action_stats(
    payload: Mapping[str, Any],
    *,
    quaternion_bounds: Mapping[str, tuple[float, float]],
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    copied = json.loads(json.dumps(payload))
    entry = copied.get("state_pose")
    if not isinstance(entry, dict):
        raise ValueError("WorldArena action statistics require state_pose")
    original = tuple(str(value) for value in entry.get("channels", ()))
    if original not in {BWM_EEF_CHANNELS, WORLD_ARENA_EEF_QUATERNION_CHANNELS}:
        raise ValueError("WorldArena statistics have an unknown 14-D channel contract")
    for arm in ("left", "right"):
        low, high = quaternion_bounds[arm]
        if abs(low - 1.0) > tolerance or abs(high - 1.0) > tolerance:
            raise ValueError(f"{arm} WorldArena orientation is not unit quaternion data")
    entry["channels"] = list(WORLD_ARENA_EEF_QUATERNION_CHANNELS)
    copied["semantic_correction"] = {
        "source_channels": list(original),
        "corrected_channels": list(WORLD_ARENA_EEF_QUATERNION_CHANNELS),
        "evidence": {
            arm: {"minimum_norm": bounds[0], "maximum_norm": bounds[1]}
            for arm, bounds in quaternion_bounds.items()
        },
        "reason": (
            "WorldArena observations/end_pose stores dual-arm XYZ + quaternion XYZW; "
            "the legacy compiler labels were Euler + gripper although the numeric arrays "
            "were preserved exactly."
        ),
    }
    return copied


def _probe(ffprobe: Path, video: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(numerator) / int(denominator),
        "frames": int(stream["nb_frames"]),
    }


def _capture(command: list[str], *, cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--source-episode", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("ffprobe"))
    parser.add_argument(
        "--prompt",
        default=(
            "A real bimanual cobot follows the supplied right-arm end-effector "
            "trajectory to wipe the tabletop while the left arm remains fixed."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    stats_path = args.action_stats.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"numeric action case already exists: {output}")
    for path, label in (
        (dataset_root, "dataset root"),
        (metadata_path, "metadata"),
        (stats_path, "action statistics"),
    ):
        valid = path.is_dir() if label == "dataset root" else path.is_file()
        if not valid:
            raise ValueError(f"missing {label}: {path}")

    rows = [
        json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()
    ]
    matches = [row for row in rows if row.get("source_episode") == args.source_episode]
    if len(matches) != 1:
        raise ValueError(
            f"expected one metadata row for {args.source_episode!r}, found {len(matches)}"
        )
    row = matches[0]
    if int(row.get("length", 0)) != 57:
        raise ValueError("numeric BWM case requires exactly 57 aligned frames")
    coordinate_frame = str(row.get("coordinate_frame", ""))
    if not coordinate_frame.startswith("robot_base:"):
        raise ValueError("WorldArena numeric action requires a named robot_base frame")

    action_spec = row["action"]
    video_spec = row["video"]
    action_source = dataset_root / str(action_spec["data"])
    video_source = dataset_root / str(video_spec["data"])
    if not action_source.is_file() or not video_source.is_file():
        raise ValueError("WorldArena metadata refers to missing action or video data")

    import pyarrow.parquet as pq

    table = pq.read_table(action_source, columns=["observation.state"])
    all_rows = table["observation.state"].to_pylist()
    start = int(action_spec["start_frame"])
    end = int(action_spec["end_frame"])
    values = tuple(tuple(float(value) for value in row) for row in all_rows[start : end + 1])
    if len(values) != 57:
        raise ValueError("WorldArena action slice did not produce 57 rows")
    quaternion_bounds = quaternion_norm_bounds(values)
    corrected_stats = correct_worldarena_action_stats(
        json.loads(stats_path.read_text()),
        quaternion_bounds=quaternion_bounds,
    )

    source_probe = _probe(args.ffprobe, video_source)
    if end >= int(source_probe["frames"]):
        raise ValueError("WorldArena action window lies outside the source video")
    action_sample_hz = float(source_probe["fps"])
    output.mkdir(parents=True)
    inputs = output / "input"
    inputs.mkdir()
    copied_video = inputs / "source-video.mp4"
    copied_action = inputs / "end-pose.parquet"
    copied_stats = inputs / "action-stat.json"
    shutil.copy2(video_source, copied_video)
    shutil.copy2(action_source, copied_action)
    _write_json(copied_stats, corrected_stats)

    first_frame = inputs / "first-frame.png"
    reference_clip = inputs / "reference-clip.mp4"
    subprocess.run(
        [
            str(args.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(copied_video),
            "-vf",
            f"select=eq(n\\,{start})",
            "-frames:v",
            "1",
            str(first_frame),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(args.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(copied_video),
            "-vf",
            f"select=between(n\\,{start}\\,{end}),setpts=N/{action_sample_hz:.12g}/TB",
            "-frames:v",
            "57",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(reference_clip),
        ],
        check=True,
    )
    reference_probe = _probe(args.ffprobe, reference_clip)
    if reference_probe["frames"] != 57:
        raise RuntimeError("prepared WorldArena reference does not contain 57 frames")

    condition = ACWMActionCondition(
        label="worldarena-wipe-table-episode-0",
        instruction="Execute the measured WorldArena wipe-table end-effector action.",
        timeline=(
            f"Exact source frames {start}-{end} sampled at {action_sample_hz:.6g} Hz; "
            "dual-arm XYZ + quaternion XYZW values preserved without conversion."
        ),
        representation=ActionRepresentation.EEF_ABSOLUTE,
        coordinate_frame=coordinate_frame,
        timestamps_s=tuple(index / action_sample_hz for index in range(57)),
        channels=WORLD_ARENA_EEF_QUATERNION_CHANNELS,
        values=values,
    )
    condition_path = inputs / "condition.json"
    condition.to_json(condition_path)
    stats = NumericActionStatistics.from_json(copied_stats)
    stats_summary = stats.validate(condition)

    compact_row = {
        **row,
        "episode_index": 0,
        "prompt": args.prompt,
        "start_frame": start,
        "end_frame": end,
        "video": {
            "data": str(copied_video),
            "start_frame": start,
            "end_frame": end,
        },
        "action": {
            "data": str(copied_action),
            "start_frame": start,
            "end_frame": end,
        },
        "channels": list(WORLD_ARENA_EEF_QUATERNION_CHANNELS),
        "action_sample_hz": action_sample_hz,
    }
    compact_metadata = inputs / "metadata.jsonl"
    compact_metadata.write_text(json.dumps(compact_row, sort_keys=True) + "\n")

    git = {}
    for key, command in {
        "head": ["git", "rev-parse", "HEAD"],
        "status": ["git", "status", "--short"],
    }.items():
        try:
            git[key] = _capture(command, cwd=project_root)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            git[key] = f"unavailable: {type(exc).__name__}: {exc}"
    artifacts = {
        path.name: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in (
            copied_video,
            copied_action,
            copied_stats,
            first_frame,
            reference_clip,
            condition_path,
            compact_metadata,
        )
    }
    manifest = {
        "schema_version": "1.0.0",
        "status": "prepared",
        "honest_status": "WORKING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": sorted(
            f"{name}=={distribution.version}"
            for distribution in distributions()
            if (name := distribution.metadata.get("Name"))
        ),
        "command": sys.argv,
        "git": git,
        "source": {
            "dataset_root": str(dataset_root),
            "metadata": str(metadata_path),
            "metadata_sha256": _sha256(metadata_path),
            "action_stats": str(stats_path),
            "action_stats_sha256": _sha256(stats_path),
            "source_episode": args.source_episode,
            "source_uri": row.get("source_uri"),
            "split": row.get("split"),
            "license_id": row.get("license_id"),
            "start_frame": start,
            "end_frame": end,
            "action_sample_hz": action_sample_hz,
        },
        "action": {
            "representation": condition.representation.value,
            "coordinate_frame": coordinate_frame,
            "channels": list(condition.channels),
            "quaternion_norm_bounds": quaternion_bounds,
            "statistics": stats_summary,
        },
        "video": {
            "source_probe": source_probe,
            "reference_probe": reference_probe,
        },
        "artifacts": artifacts,
        "claim_boundary": (
            "This is a synchronized real-robot observation/action input case. "
            "It is not generated-video quality or physical execution evidence."
        ),
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({"output": str(output), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
