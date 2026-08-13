#!/usr/bin/env python3
"""Build synchronized three-view DROID clips for Cosmos multiview adaptation.

The public conditioning contract is deliberately explicit: the first frame of
all three cameras and the task text are real conditions.  Every later frame is
predicted.  DROID-100 does not publish camera calibration in this LeRobot
conversion, so this builder never invents intrinsics or extrinsics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FPS = 15.0
FPS = 8
FRAMES = 17
WIDTH = 448
HEIGHT = 256
ENCODE_CRF = 24

# The legacy development set has already been inspected repeatedly and is not
# eligible as a clean final test set.  The final set below was previously held
# out as validation and had not been used for checkpoint selection.
LEGACY_DEV_EPISODES = (21, 60, 77)
VALIDATION_EPISODES = (12, 20)
FINAL_HOLDOUT_EPISODES = (54, 66, 91)

CAMERAS = {
    "exterior_1": "observation.images.exterior_image_1_left",
    "wrist": "observation.images.wrist_image_left",
    "exterior_2": "observation.images.exterior_image_2_left",
}
COSMOS_ROLE = {
    "exterior_1": "head",
    "wrist": "hand_0",
    "exterior_2": "hand_1",
}
COORDINATE_FRAMES = {
    name: f"camera:{feature.removeprefix('observation.images.')}_pixels"
    for name, feature in CAMERAS.items()
}
DATASET_CARD = "https://huggingface.co/datasets/lerobot/droid_100"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--episodes-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clips-per-train-episode", type=int, default=1)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path(shutil.which("ffprobe") or "ffprobe"))
    parser.add_argument("--git-commit")
    parser.add_argument("--git-branch")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"{label} is missing or empty: {resolved}")
    return resolved


def _parse_ids(values: Iterable[int]) -> set[int]:
    result = {int(value) for value in values}
    if any(value < 0 for value in result):
        raise ValueError("episode indices must be non-negative")
    return result


def split_episode_ids(
    episodes: list[dict[str, Any]],
    *,
    legacy_dev: Iterable[int] = LEGACY_DEV_EPISODES,
    validation: Iterable[int] = VALIDATION_EPISODES,
    final_holdout: Iterable[int] = FINAL_HOLDOUT_EPISODES,
) -> dict[str, tuple[int, ...]]:
    """Create disjoint train/dev/validation/final episode splits."""
    available = tuple(sorted(int(row["episode_index"]) for row in episodes))
    if len(available) != len(set(available)):
        raise ValueError("episode metadata contains duplicate episode indices")
    split_sets = {
        "legacy_dev": _parse_ids(legacy_dev),
        "validation": _parse_ids(validation),
        "final_holdout": _parse_ids(final_holdout),
    }
    names = tuple(split_sets)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if split_sets[name] & split_sets[other]:
                raise ValueError(f"{name} and {other} must be disjoint")
    reserved = set().union(*split_sets.values())
    missing = reserved - set(available)
    if missing:
        raise ValueError(f"reserved episodes are absent: {sorted(missing)}")
    return {
        "train": tuple(value for value in available if value not in reserved),
        **{name: tuple(sorted(values)) for name, values in split_sets.items()},
    }


def plan_clip_starts(
    start: float,
    end: float,
    clips: int,
    *,
    episode_index: int,
    seed: int,
    clip_seconds: float = FRAMES / FPS,
) -> tuple[float, ...]:
    """Choose deterministic synchronized windows strictly inside an episode."""
    if start < 0 or end <= start or clips <= 0 or clip_seconds <= 0:
        raise ValueError("invalid clip-planning interval")
    latest = end - clip_seconds - 1.0 / SOURCE_FPS
    if latest < start:
        raise ValueError(f"episode {episode_index} is too short")
    if clips == 1:
        rng = random.Random(seed + episode_index * 1_000_003)
        return (start + (latest - start) * (0.2 + 0.6 * rng.random()),)
    return tuple(start + (latest - start) * index / (clips - 1) for index in range(clips))


def _probe(ffprobe: Path, path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = (int(part) for part in stream["avg_frame_rate"].split("/"))
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": numerator / denominator,
        "frames": int(stream["nb_read_frames"]),
    }


def _extract_clip(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    start: float,
    *,
    frames: int,
    fps: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.9f}",
            "-i",
            str(source),
            "-vf",
            f"fps={fps},scale=456:{HEIGHT}:flags=lanczos,crop={WIDTH}:{HEIGHT}",
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(ENCODE_CRF),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"ffmpeg clip extraction failed: {completed.stderr}")


def _extract_anchor(ffmpeg: Path, video: Path, destination: Path) -> None:
    completed = subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"ffmpeg anchor extraction failed: {completed.stderr}")


def _git_state(commit: str | None, branch: str | None) -> dict[str, Any]:
    if (commit is None) != (branch is None):
        raise ValueError("git-commit and git-branch must be supplied together")
    return {
        "commit": commit or "unresolved",
        "branch": branch,
        "working_tree_status": "dirty",
        "task_file_sha256": {
            "scripts/build_droid_cosmos_multiview_dataset.py": _sha256(Path(__file__)),
        },
    }


def main() -> int:
    args = _parser().parse_args()
    if args.clips_per_train_episode <= 0:
        raise ValueError("clips-per-train-episode must be positive")
    if args.frames <= 1 or args.fps <= 0:
        raise ValueError("frames must exceed one and fps must be positive")
    source_root = args.source_root.expanduser().resolve()
    episodes_path = _require_file(args.episodes_json, "episodes JSON")
    ffmpeg = _require_file(args.ffmpeg, "ffmpeg")
    ffprobe = _require_file(args.ffprobe, "ffprobe")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)

    episode_payload = json.loads(episodes_path.read_text())
    episodes = episode_payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("episodes JSON must contain a non-empty episodes list")
    if episode_payload.get("synchronized") is not True:
        raise ValueError("episodes JSON must attest synchronized camera timestamps")
    minimum_duration = args.frames / args.fps + 1 / SOURCE_FPS
    excluded_short = {
        int(row["episode_index"])
        for row in episodes
        if float(row["to_timestamp"]) - float(row["from_timestamp"]) < minimum_duration
    }
    selection_reserved = set(VALIDATION_EPISODES + FINAL_HOLDOUT_EPISODES)
    if excluded_short & selection_reserved:
        raise ValueError(
            "validation or final-holdout episodes are too short: "
            f"{sorted(excluded_short & selection_reserved)}"
        )
    usable = [row for row in episodes if int(row["episode_index"]) not in excluded_short]
    compatible_legacy_dev = tuple(
        episode for episode in LEGACY_DEV_EPISODES if episode not in excluded_short
    )
    splits = split_episode_ids(usable, legacy_dev=compatible_legacy_dev)
    rows = {int(row["episode_index"]): row for row in usable}
    sources = {
        name: _require_file(
            source_root / "videos" / feature / "chunk-000" / "file-000.mp4",
            name,
        )
        for name, feature in CAMERAS.items()
    }

    command = [sys.executable, *sys.argv]
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    _write_json(output / "git-state.json", _git_state(args.git_commit, args.git_branch))
    _write_json(
        output / "input-config.json",
        {
            **vars(args),
            "source_root": str(source_root),
            "episodes_json": str(episodes_path),
            "output_dir": str(output),
        },
    )

    records: list[dict[str, Any]] = []
    for split_name, episode_ids in splits.items():
        clips = args.clips_per_train_episode if split_name == "train" else 1
        for episode_id in episode_ids:
            row = rows[episode_id]
            starts = plan_clip_starts(
                float(row["from_timestamp"]),
                float(row["to_timestamp"]),
                clips,
                episode_index=episode_id,
                seed=args.seed,
                clip_seconds=args.frames / args.fps,
            )
            task = str(row["tasks"][0]) if row.get("tasks") else "Perform the manipulation task"
            for clip_index, start in enumerate(starts):
                sample_id = f"ep{episode_id:03d}-clip{clip_index:02d}"
                sample_dir = output / "dataset" / split_name / sample_id
                cameras: dict[str, Any] = {}
                common_probe: dict[str, Any] | None = None
                for camera_name, source in sources.items():
                    video = sample_dir / f"{camera_name}.mp4"
                    anchor = sample_dir / f"{camera_name}-real-condition.png"
                    _extract_clip(
                        ffmpeg,
                        source,
                        video,
                        start,
                        frames=args.frames,
                        fps=args.fps,
                    )
                    _extract_anchor(ffmpeg, video, anchor)
                    probe = _probe(ffprobe, video)
                    expected = {
                        "width": WIDTH,
                        "height": HEIGHT,
                        "fps": float(args.fps),
                        "frames": args.frames,
                    }
                    if any(probe[key] != value for key, value in expected.items()):
                        raise RuntimeError(f"invalid video contract for {video}: {probe}")
                    if common_probe is not None and probe != common_probe:
                        raise RuntimeError(f"synchronized camera probes differ for {sample_id}")
                    common_probe = probe
                    cameras[camera_name] = {
                        "cosmos_role": COSMOS_ROLE[camera_name],
                        "coordinate_frame": COORDINATE_FRAMES[camera_name],
                        "video": str(video.relative_to(output)),
                        "video_sha256": _sha256(video),
                        "real_first_frame_condition": str(anchor.relative_to(output)),
                        "real_first_frame_condition_sha256": _sha256(anchor),
                    }
                caption_rows = [
                    {"caption": task, "view": camera_name, "tag": "long"}
                    for camera_name in CAMERAS
                ]
                caption = sample_dir / "caption.jsonl"
                caption.write_text("".join(json.dumps(item) + "\n" for item in caption_rows))
                records.append(
                    {
                        "sample_id": sample_id,
                        "episode_index": episode_id,
                        "clip_index": clip_index,
                        "split": split_name,
                        "training_use": split_name == "train",
                        "task_text_real_condition": task,
                        "start_seconds": start,
                        "end_seconds": start + args.frames / args.fps,
                        "cameras": cameras,
                        "caption": str(caption.relative_to(output)),
                    }
                )

    contract = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "phiagent_droid_cosmos_predict2_three_view_dataset",
        "status": "WORKING",
        "model_route": "Cosmos-Predict2-14B GR00T-Dreams-DROID native 2x2 multiview adaptation",
        "source": {
            "dataset": "LeRobot DROID-100",
            "dataset_card": DATASET_CARD,
            "streams": {
                name: {
                    "feature": CAMERAS[name],
                    "coordinate_frame": COORDINATE_FRAMES[name],
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for name, path in sources.items()
            },
        },
        "split": {
            **{name: list(values) for name, values in splits.items()},
            "excluded_too_short": sorted(excluded_short),
            "legacy_dev_excluded_too_short": sorted(
                set(LEGACY_DEV_EPISODES) & excluded_short
            ),
            "legacy_dev_policy": "previously inspected; never report as clean final holdout",
            "final_holdout_policy": "never used for training or checkpoint selection",
        },
        "leakage_checks": {
            "all_episode_splits_disjoint": len({value for values in splits.values() for value in values})
            == sum(len(values) for values in splits.values()),
            "final_holdout_used_for_training": False,
            "real_future_frames_passed_at_inference": False,
        },
        "conditioning_contract": {
            "real_conditions": [
                "wrist-camera first frame",
                "exterior-camera-1 first frame",
                "exterior-camera-2 first frame",
                "task text",
            ],
            "our_generated_video": "all three camera streams after their real first frames",
            "disclosure_required": "Demo must label the three first frames and task text as REAL CONDITIONS, and all predicted continuations as OUR GENERATED VIDEO.",
        },
        "calibration_contract": {
            "status": "unavailable",
            "intrinsics": None,
            "extrinsics": None,
            "reason": "LeRobot DROID-100 conversion used here does not expose camera calibration fields.",
            "proxy_allowed_for_baseline": True,
            "proxy_disclosure_required": True,
        },
        "video_contract": {
            "frames": args.frames,
            "fps": args.fps,
            "width": WIDTH,
            "height": HEIGHT,
            "frames_satisfy_4n_plus_1": (args.frames - 1) % 4 == 0,
        },
        "camera_to_cosmos_role": COSMOS_ROLE,
        "records": records,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "seed": args.seed,
        "command": command,
    }
    _write_json(output / "dataset-contract.json", contract)
    print(
        json.dumps(
            {
                "output": str(output),
                "train_samples": sum(row["split"] == "train" for row in records),
                "legacy_dev_samples": sum(row["split"] == "legacy_dev" for row in records),
                "validation_samples": sum(row["split"] == "validation" for row in records),
                "final_holdout_samples": sum(row["split"] == "final_holdout" for row in records),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
