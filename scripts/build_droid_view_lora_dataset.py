#!/usr/bin/env python3
"""Build leakage-safe DROID wrist-to-exterior VACE LoRA clips.

The learned mapping is explicit: a real wrist-camera video and one real
exterior-camera anchor frame condition generation of the remaining exterior
video.  Entire held-out episodes are excluded from every training asset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
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
FPS = 8
FRAMES = 17
WIDTH = 448
HEIGHT = 256
SOURCE_FPS = 15.0
ENCODE_CRF = 24
HOLDOUT_EPISODES = (21, 60, 77)
VALIDATION_EPISODES = (12, 20, 54, 66, 91)
FEATURES = {
    "condition": "observation.images.wrist_image_left",
    "target_a": "observation.images.exterior_image_1_left",
    "target_b": "observation.images.exterior_image_2_left",
}
COORDINATE_FRAMES = {
    "condition": "camera:wrist_image_left_pixels",
    "target_a": "camera:exterior_image_1_left_pixels",
    "target_b": "camera:exterior_image_2_left_pixels",
}
DATASET_CARD = "https://huggingface.co/datasets/lerobot/droid_100"
RIGHTS_BASIS = "MIT as listed by the LeRobot DROID-100 dataset card"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--episodes-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path(shutil.which("ffprobe") or "ffprobe"))
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--clips-per-episode", type=int, default=1)
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("target_a", "target_b"),
        default=("target_a", "target_b"),
    )
    parser.add_argument("--seed", type=int, default=20260811)
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def _git_state(commit_override: str | None = None, branch_override: str | None = None) -> dict[str, Any]:
    git_dir = PROJECT_ROOT / ".git"
    if (commit_override is None) != (branch_override is None):
        raise ValueError("git-commit and git-branch must be provided together")
    if commit_override is not None:
        if len(commit_override) != 40 or any(
            character not in "0123456789abcdef" for character in commit_override
        ):
            raise ValueError("git-commit must be a lowercase 40-character SHA-1")
        head = f"ref: refs/heads/{branch_override}"
        branch = branch_override
        commit = commit_override
        resolution = "explicit CLI snapshot supplied after local .git reads blocked"
    else:
        head = (git_dir / "HEAD").read_text().strip()
        branch = None
        commit = head
        if head.startswith("ref: "):
            reference = head.removeprefix("ref: ")
            branch = reference.removeprefix("refs/heads/")
            loose_ref = git_dir / reference
            if loose_ref.is_file():
                commit = loose_ref.read_text().strip()
            else:
                packed = (git_dir / "packed-refs").read_text().splitlines()
                match = next(
                    (line.split(" ", 1)[0] for line in packed if line.endswith(f" {reference}")),
                    None,
                )
                if match is None:
                    raise RuntimeError(f"could not resolve Git reference {reference}")
                commit = match
        resolution = "resolved from local .git files"
    task_files = [
        PROJECT_ROOT / "scripts/build_droid_view_lora_dataset.py",
        PROJECT_ROOT / "tests/test_build_droid_view_lora_dataset.py",
    ]
    return {
        "commit": commit,
        "branch": branch,
        "head_raw": head,
        "resolution": resolution,
        "task_file_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in task_files
        },
        "working_tree_status": "dirty",
        "status_scope": (
            "The pre-existing workspace is dirty. Git subprocesses block on its large state, "
            "so HEAD is resolved directly and task file hashes are captured."
        ),
    }


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"{label} is missing or empty: {resolved}")
    return resolved


def _source_video(source_root: Path, feature: str) -> Path:
    return _require_file(
        source_root / "videos" / feature / "chunk-000" / "file-000.mp4",
        feature,
    )


def _parse_episode_ids(raw: Iterable[int]) -> set[int]:
    result = {int(value) for value in raw}
    if any(value < 0 for value in result):
        raise ValueError("episode indices must be non-negative")
    return result


def choose_evenly(values: list[int], count: int) -> tuple[int, ...]:
    """Choose a deterministic spread without duplicating indices."""
    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(values):
        raise ValueError(f"requested {count} episodes but only {len(values)} are eligible")
    if count == 1:
        return (values[len(values) // 2],)
    if count == len(values):
        return tuple(values)
    positions = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return tuple(values[position] for position in positions)


def plan_clip_starts(
    start: float,
    end: float,
    clips: int,
    *,
    episode_index: int,
    seed: int,
    clip_seconds: float = FRAMES / FPS,
) -> tuple[float, ...]:
    """Plan reproducible starts strictly inside one episode."""
    if start < 0 or end <= start or clips <= 0 or clip_seconds <= 0:
        raise ValueError("invalid clip-planning interval")
    latest = end - clip_seconds - 1.0 / SOURCE_FPS
    if latest < start:
        raise ValueError(
            f"episode {episode_index} is too short for {FRAMES} frames at {FPS} FPS"
        )
    if clips == 1:
        rng = random.Random(seed + episode_index * 1_000_003)
        return (start + (latest - start) * (0.2 + 0.6 * rng.random()),)
    return tuple(start + (latest - start) * index / (clips - 1) for index in range(clips))


def split_episode_ids(
    episodes: list[dict[str, Any]],
    train_count: int,
    *,
    holdout: Iterable[int] = HOLDOUT_EPISODES,
    validation: Iterable[int] = VALIDATION_EPISODES,
) -> dict[str, tuple[int, ...]]:
    available = sorted(int(row["episode_index"]) for row in episodes)
    if len(available) != len(set(available)):
        raise ValueError("episode metadata contains duplicate episode_index values")
    holdout_ids = _parse_episode_ids(holdout)
    validation_ids = _parse_episode_ids(validation)
    if holdout_ids & validation_ids:
        raise ValueError("holdout and validation episodes must be disjoint")
    missing = (holdout_ids | validation_ids) - set(available)
    if missing:
        raise ValueError(f"requested split episodes are absent: {sorted(missing)}")
    eligible = [value for value in available if value not in holdout_ids | validation_ids]
    training = choose_evenly(eligible, train_count)
    return {
        "train": training,
        "validation": tuple(sorted(validation_ids)),
        "holdout": tuple(sorted(holdout_ids)),
    }


def _probe(ffprobe: Path, path: Path, *, count_frames: bool = True) -> dict[str, Any]:
    count_arguments = ["-count_frames"] if count_frames else []
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            *count_arguments,
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
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
    frame_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if not frame_value or frame_value == "N/A":
        raise RuntimeError(f"ffprobe did not report a frame count for {path}")
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": numerator / denominator,
        "frames": int(frame_value),
        "duration_seconds": float(stream["duration"]),
    }


def _extract_clip(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    start_seconds: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.9f}",
        "-i",
        str(source),
        "-vf",
        f"fps={FPS},scale=456:{HEIGHT}:flags=lanczos,crop={WIDTH}:{HEIGHT}",
        "-frames:v",
        str(FRAMES),
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
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg clip extraction failed: {completed.stderr}")


def _extract_reference(ffmpeg: Path, video: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError(f"ffmpeg reference extraction failed: {completed.stderr}")


def _asset(
    asset_id: str,
    path: Path,
    kind: str,
    feature: str,
    manifest_dir: Path,
    split: str = "train",
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "path": str(path.resolve().relative_to(manifest_dir.resolve())),
        "split": split,
        "kind": kind,
        "source_uri": f"{DATASET_CARD}/tree/main/{feature}",
        "rights_basis": RIGHTS_BASIS,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "training_authorized": split == "train",
    }


def main() -> int:
    args = _parser().parse_args()
    if args.train_episodes <= 0 or args.clips_per_episode <= 0:
        raise ValueError("training episode and clip counts must be positive")
    source_root = args.source_root.expanduser().resolve()
    episodes_path = _require_file(args.episodes_json, "episode metadata JSON")
    ffmpeg = _require_file(args.ffmpeg, "ffmpeg")
    ffprobe = _require_file(args.ffprobe, "ffprobe")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset experiment: {output}")
    output.mkdir(parents=True)
    log_path = output / "build.log"
    _log(log_path, "dataset build started")

    episode_payload = json.loads(episodes_path.read_text())
    episodes = episode_payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("episodes JSON must contain a non-empty episodes list")
    if not episode_payload.get("synchronized"):
        raise ValueError("episodes JSON must attest synchronized camera timestamps")
    reserved_ids = set(HOLDOUT_EPISODES) | set(VALIDATION_EPISODES)
    minimum_duration = FRAMES / FPS + 1.0 / SOURCE_FPS
    excluded_short = sorted(
        int(row["episode_index"])
        for row in episodes
        if float(row["to_timestamp"]) - float(row["from_timestamp"]) < minimum_duration
    )
    invalid_reserved = reserved_ids & set(excluded_short)
    if invalid_reserved:
        raise ValueError(
            f"held-out or validation episodes are too short: {sorted(invalid_reserved)}"
        )
    usable_episodes = [
        row for row in episodes if int(row["episode_index"]) not in set(excluded_short)
    ]
    splits = split_episode_ids(usable_episodes, args.train_episodes)
    rows_by_id = {int(row["episode_index"]): row for row in episodes}
    sources = {name: _source_video(source_root, feature) for name, feature in FEATURES.items()}
    source_probes = {
        name: _probe(ffprobe, path, count_frames=False) for name, path in sources.items()
    }
    if any(
        (probe["width"], probe["height"], probe["fps"], probe["frames"])
        != (320, 180, SOURCE_FPS, 32212)
        for probe in source_probes.values()
    ):
        raise RuntimeError(f"unexpected DROID source stream contract: {source_probes}")

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
            "ffmpeg": str(ffmpeg),
            "ffprobe": str(ffprobe),
        },
    )

    assets: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    clip_records: list[dict[str, Any]] = []
    holdout_records: list[dict[str, Any]] = []
    dataset_root = output / "dataset"
    for episode_id in splits["train"]:
        row = rows_by_id[episode_id]
        starts = plan_clip_starts(
            float(row["from_timestamp"]),
            float(row["to_timestamp"]),
            args.clips_per_episode,
            episode_index=episode_id,
            seed=args.seed,
        )
        task = str(row["tasks"][0]) if row.get("tasks") else "Perform the manipulation task"
        for clip_index, start in enumerate(starts):
            stem = f"ep{episode_id:03d}-clip{clip_index:02d}"
            control = dataset_root / "train" / stem / "real-wrist-condition.mp4"
            _extract_clip(ffmpeg, sources["condition"], control, start)
            control_probe = _probe(ffprobe, control)
            if (
                control_probe["frames"],
                control_probe["width"],
                control_probe["height"],
                control_probe["fps"],
            ) != (FRAMES, WIDTH, HEIGHT, FPS):
                raise RuntimeError(f"invalid control clip contract: {control_probe}")
            control_id = f"{stem}-real-wrist-condition"
            assets.append(
                _asset(
                    control_id,
                    control,
                    "vace_control_video",
                    FEATURES["condition"],
                    output,
                )
            )
            record = {
                "episode_index": episode_id,
                "clip_index": clip_index,
                "task": task,
                "start_seconds": start,
                "end_seconds": start + FRAMES / FPS,
                "condition_coordinate_frame": COORDINATE_FRAMES["condition"],
                "targets": {},
            }
            for view_name in args.views:
                target = dataset_root / "train" / stem / f"real-{view_name}-target.mp4"
                reference = dataset_root / "train" / stem / f"real-{view_name}-anchor.png"
                _extract_clip(ffmpeg, sources[view_name], target, start)
                _extract_reference(ffmpeg, target, reference)
                target_probe = _probe(ffprobe, target)
                if target_probe != control_probe:
                    raise RuntimeError(
                        f"paired clip contract differs for {stem} {view_name}: "
                        f"{control_probe} vs {target_probe}"
                    )
                target_id = f"{stem}-real-{view_name}-target"
                reference_id = f"{stem}-real-{view_name}-anchor"
                assets.extend(
                    [
                        _asset(
                            target_id,
                            target,
                            "target_video",
                            FEATURES[view_name],
                            output,
                        ),
                        _asset(
                            reference_id,
                            reference,
                            "vace_reference_image",
                            FEATURES[view_name],
                            output,
                        ),
                    ]
                )
                examples.append(
                    {
                        "example_id": f"{stem}-{view_name}",
                        "target_video_asset_id": target_id,
                        "control_video_asset_id": control_id,
                        "reference_image_asset_id": reference_id,
                        "prompt": (
                            f"Third-person DROID robot view {view_name[-1].upper()}: {task}. "
                            "Preserve the robot, manipulated object, scene, and temporal action."
                        ),
                    }
                )
                record["targets"][view_name] = {
                    "coordinate_frame": COORDINATE_FRAMES[view_name],
                    "target": str(target.relative_to(output)),
                    "anchor": str(reference.relative_to(output)),
                }
            clip_records.append(record)
            _log(
                log_path,
                f"built episode={episode_id} clip={clip_index} views={','.join(args.views)}",
            )

    for episode_id in splits["holdout"]:
        row = rows_by_id[episode_id]
        start = plan_clip_starts(
            float(row["from_timestamp"]),
            float(row["to_timestamp"]),
            1,
            episode_index=episode_id,
            seed=args.seed,
        )[0]
        task = str(row["tasks"][0]) if row.get("tasks") else "Perform the manipulation task"
        stem = f"ep{episode_id:03d}-heldout"
        control = dataset_root / "holdout" / stem / "real-wrist-condition.mp4"
        _extract_clip(ffmpeg, sources["condition"], control, start)
        control_probe = _probe(ffprobe, control)
        record = {
            "episode_index": episode_id,
            "task": task,
            "start_seconds": start,
            "end_seconds": start + FRAMES / FPS,
            "condition": {
                "path": str(control.relative_to(output)),
                "sha256": _sha256(control),
                "coordinate_frame": COORDINATE_FRAMES["condition"],
                "probe": control_probe,
            },
            "targets": {},
            "training_use": False,
        }
        for view_name in args.views:
            target = dataset_root / "holdout" / stem / f"real-{view_name}-target.mp4"
            reference = dataset_root / "holdout" / stem / f"real-{view_name}-anchor.png"
            _extract_clip(ffmpeg, sources[view_name], target, start)
            _extract_reference(ffmpeg, target, reference)
            target_probe = _probe(ffprobe, target)
            if target_probe != control_probe:
                raise RuntimeError(
                    f"held-out pair contract differs for {stem} {view_name}: "
                    f"{control_probe} vs {target_probe}"
                )
            record["targets"][view_name] = {
                "coordinate_frame": COORDINATE_FRAMES[view_name],
                "target": str(target.relative_to(output)),
                "target_sha256": _sha256(target),
                "anchor": str(reference.relative_to(output)),
                "anchor_sha256": _sha256(reference),
                "prompt": (
                    f"Third-person DROID robot view {view_name[-1].upper()}: {task}. "
                    "Preserve the robot, manipulated object, scene, and temporal action."
                ),
            }
        holdout_records.append(record)
        _log(log_path, f"built heldout episode={episode_id} views={','.join(args.views)}")

    manifest = {
        "schema_version": "0.1.0",
        "method": "sharpa_lightweight_adaptation_not_official_phizero",
        "experiment_id": f"droid-wrist-to-exterior-vace-lora-{output.name}",
        "arm": "vace_lora",
        "evidence_scope": "claim_eligible",
        "assets": assets,
        "animate_examples": [],
        "vace_examples": examples,
    }
    manifest_path = output / "adaptation-manifest.json"
    _write_json(manifest_path, manifest)
    contract = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "phiagent_droid_wrist_to_exterior_vace_lora_dataset",
        "status": "WORKING",
        "command": command,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            distribution: (
                importlib.metadata.version(distribution)
                if importlib.util.find_spec(module) is not None
                else None
            )
            for distribution, module in (
                ("numpy", "numpy"),
                ("opencv-python", "cv2"),
                ("pyarrow", "pyarrow"),
            )
        },
        "seed": args.seed,
        "source": {
            "dataset": "LeRobot DROID-100",
            "dataset_card": DATASET_CARD,
            "rights_basis": RIGHTS_BASIS,
            "episodes_metadata": str(episodes_path),
            "episodes_metadata_sha256": _sha256(episodes_path),
            "streams": {
                name: {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "probe": source_probes[name],
                    "coordinate_frame": COORDINATE_FRAMES[name],
                }
                for name, path in sources.items()
            },
        },
        "split": {
            **{name: list(values) for name, values in splits.items()},
            "excluded_too_short": excluded_short,
        },
        "leakage_checks": {
            "episode_disjoint": not (
                set(splits["train"]) & set(splits["validation"])
                or set(splits["train"]) & set(splits["holdout"])
                or set(splits["validation"]) & set(splits["holdout"])
            ),
            "heldout_targets_used_for_training": False,
            "heldout_anchors_used_for_training": False,
        },
        "conditioning_contract": {
            "real_condition": [
                "full wrist-camera clip",
                "one exterior-camera anchor frame at the requested target viewpoint",
                "task text",
            ],
            "our_generated_video": "all predicted exterior-view video frames",
            "real_target": "synchronized exterior-camera clip used only for training or held-out evaluation according to episode split",
            "disclosure_required": (
                "Public demos must label both the real wrist clip and the real exterior "
                "anchor frame as conditions."
            ),
        },
        "video_contract": {
            "frames": FRAMES,
            "fps": FPS,
            "width": WIDTH,
            "height": HEIGHT,
            "clip_duration_seconds": FRAMES / FPS,
            "h264_crf": ENCODE_CRF,
        },
        "training_episode_count": len(splits["train"]),
        "training_example_count": len(examples),
        "clip_records": clip_records,
        "heldout_example_count": sum(len(row["targets"]) for row in holdout_records),
        "holdout_records": holdout_records,
        "adaptation_manifest": str(manifest_path.relative_to(output)),
        "adaptation_manifest_sha256": _sha256(manifest_path),
        "limitations": [
            "The target-view anchor is a disclosed real condition, so this is not wrist-only generation.",
            "DROID-100 is small and heterogeneous; held-out visual quality must be measured before promotion.",
            "The training target is captured imagery, not physical execution by a newly controlled robot.",
        ],
    }
    _write_json(output / "dataset-contract.json", contract)
    _log(
        log_path,
        f"dataset build completed training_examples={len(examples)} "
        f"heldout_episodes={list(splits['holdout'])}",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "training_episodes": len(splits["train"]),
                "training_examples": len(examples),
                "heldout_episodes": list(splits["holdout"]),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
