#!/usr/bin/env python3
"""Recover raw DROID paths and verify them against LeRobot video pixels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_PREFIX = PurePosixPath(
    "/nfs/kun2/datasets/r2d2/r2d2-data-full"
)
STREAMS = (
    "wrist_image_left",
    "exterior_image_1_left",
    "exterior_image_2_left",
)
DEFAULT_EPISODES = (21, 60, 77)
DEFAULT_MIN_P05_PSNR_DB = 25.0
DEFAULT_MAX_P95_DHASH_HAMMING = 8


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--episodes-json", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path(shutil.which("ffmpeg") or "ffmpeg"),
    )
    parser.add_argument("--source-fps", type=float, default=15.0)
    parser.add_argument("--episodes", type=int, nargs="+", default=DEFAULT_EPISODES)
    parser.add_argument(
        "--min-p05-psnr-db",
        type=float,
        default=DEFAULT_MIN_P05_PSNR_DB,
    )
    parser.add_argument(
        "--max-p95-dhash-hamming",
        type=int,
        default=DEFAULT_MAX_P95_DHASH_HAMMING,
    )
    parser.add_argument("--seed", type=int, default=20260812)
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


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values or not 0 < percentile <= 1:
        raise ValueError("percentile requires non-empty values and 0 < percentile <= 1")
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def derive_raw_paths(file_path: str, recording_folderpath: str) -> dict[str, str]:
    trajectory = PurePosixPath(file_path)
    recording = PurePosixPath(recording_folderpath)
    if trajectory.name != "trajectory.h5":
        raise ValueError(f"unexpected DROID trajectory path: {file_path}")
    episode_root = trajectory.parent
    expected_recording = episode_root / "recordings" / "MP4"
    if recording != expected_recording:
        raise ValueError(
            "DROID recording_folderpath does not share the trajectory episode root"
        )
    try:
        relative = episode_root.relative_to(RAW_DATASET_PREFIX)
    except ValueError as error:
        raise ValueError(
            f"DROID path is outside the pinned raw dataset prefix: {file_path}"
        ) from error
    return {
        "file_path": str(trajectory),
        "recording_folderpath": str(recording),
        "raw_episode_relative_path": str(relative),
        "raw_gcs_prefix": f"1.0.1/{relative}",
    }


def dhash_hamming(first: int, second: int) -> int:
    if first < 0 or second < 0:
        raise ValueError("dHash values must be non-negative")
    return (first ^ second).bit_count()


def _dhash64(cv2: Any, np: Any, frame: Any) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    packed = np.packbits(bits.reshape(-1))
    return int.from_bytes(packed.tobytes(), "big")


def _pixel_psnr_db(np: Any, first: Any, second: Any) -> float:
    error = float(
        np.mean(
            (
                first.astype(np.float32)
                - second.astype(np.float32)
            )
            ** 2
        )
    )
    if error == 0:
        return 100.0
    return 10.0 * math.log10((255.0**2) / error)


def _single_bytes_feature(example: Any, name: str) -> bytes:
    feature = example.context.feature.get(name)
    if feature is None or feature.WhichOneof("kind") != "bytes_list":
        raise ValueError(f"missing bytes-list SequenceExample feature: {name}")
    values = feature.bytes_list.value
    if len(values) != 1:
        raise ValueError(f"expected exactly one value for SequenceExample feature: {name}")
    return bytes(values[0])


def _bytes_feature(example: Any, name: str) -> list[bytes]:
    feature = example.context.feature.get(name)
    if feature is None or feature.WhichOneof("kind") != "bytes_list":
        raise ValueError(f"missing bytes-list SequenceExample feature: {name}")
    return [bytes(value) for value in feature.bytes_list.value]


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("numpy", "opencv-python", "tensorflow", "protobuf"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _git_state(
    commit_override: str | None = None,
    branch_override: str | None = None,
) -> dict[str, object]:
    if (commit_override is None) != (branch_override is None):
        raise ValueError("git-commit and git-branch must be provided together")
    if commit_override is not None:
        if len(commit_override) != 40 or any(
            character not in "0123456789abcdef" for character in commit_override
        ):
            raise ValueError("git-commit must be a lowercase 40-character SHA-1")
        return {
            "commit": commit_override,
            "branch": branch_override,
            "dirty": None,
            "status_porcelain": None,
            "resolution": "explicit source-worktree snapshot",
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
        }

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = run("status", "--porcelain=v1")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "resolution": "local Git worktree",
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
    }


def _decode_example(sequence_example_type: Any, path: Path) -> Any:
    return sequence_example_type.FromString(path.read_bytes())


def _decode_video_segment(
    *,
    ffmpeg: Path,
    np: Any,
    video_path: Path,
    start_frame: int,
    frame_count: int,
    source_fps: float,
    height: int,
    width: int,
) -> Any:
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-c:v",
        "libdav1d",
        "-ss",
        f"{start_frame / source_fps:.9f}",
        "-i",
        str(video_path),
        "-frames:v",
        str(frame_count),
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode != 0:
        diagnostic = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg AV1 segment decode failed: {diagnostic}")
    expected_bytes = frame_count * height * width * 3
    if len(result.stdout) != expected_bytes:
        raise ValueError(
            "ffmpeg decoded byte count mismatch: "
            f"{len(result.stdout)} != {expected_bytes}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        frame_count, height, width, 3
    )


def _episode_index(episodes_payload: object) -> dict[int, dict[str, Any]]:
    if not isinstance(episodes_payload, dict):
        raise ValueError("episodes JSON must contain an object")
    rows = episodes_payload.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("episodes JSON must contain an episodes list")
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("episode_index"), int):
            raise ValueError("each episode row must contain an integer episode_index")
        indexed[row["episode_index"]] = row
    return indexed


def _decode_sequence_frames(cv2: Any, np: Any, encoded_frames: list[bytes]) -> list[Any]:
    if not encoded_frames:
        raise ValueError("SequenceExample image feature is empty")
    decoded_sequence = []
    for relative_index, encoded in enumerate(encoded_frames):
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(
                f"could not decode SequenceExample JPEG at index {relative_index}"
            )
        decoded_sequence.append(decoded)
    shapes = {tuple(frame.shape) for frame in decoded_sequence}
    if len(shapes) != 1:
        raise ValueError(f"SequenceExample frames have inconsistent shapes: {shapes}")
    height, width, channels = decoded_sequence[0].shape
    if channels != 3:
        raise ValueError(f"expected three decoded image channels, received {channels}")
    return decoded_sequence


def _alignment_metrics(
    *,
    cv2: Any,
    np: Any,
    decoded_sequence: list[Any],
    lerobot_frames: Any,
) -> dict[str, object]:
    if len(decoded_sequence) != len(lerobot_frames):
        raise ValueError("SequenceExample and LeRobot frame counts differ")
    psnr_values: list[float] = []
    hamming_values: list[float] = []
    mae_values: list[float] = []
    exact_pixel_matches = 0
    sampled_hashes = []
    sample_indices = {
        0,
        len(decoded_sequence) // 2,
        len(decoded_sequence) - 1,
    }
    for relative_index, decoded in enumerate(decoded_sequence):
        lerobot = lerobot_frames[relative_index]
        decoded_sha = hashlib.sha256(decoded.tobytes()).hexdigest()
        lerobot_sha = hashlib.sha256(lerobot.tobytes()).hexdigest()
        exact_pixel_matches += decoded_sha == lerobot_sha
        psnr_values.append(_pixel_psnr_db(np, decoded, lerobot))
        mae_values.append(
            float(
                np.mean(
                    np.abs(
                        decoded.astype(np.float32)
                        - lerobot.astype(np.float32)
                    )
                )
            )
        )
        decoded_dhash = _dhash64(cv2, np, decoded)
        lerobot_dhash = _dhash64(cv2, np, lerobot)
        hamming_values.append(float(dhash_hamming(decoded_dhash, lerobot_dhash)))
        if relative_index in sample_indices:
            sampled_hashes.append(
                {
                    "relative_frame": relative_index,
                    "sequence_decoded_pixel_sha256": decoded_sha,
                    "lerobot_decoded_pixel_sha256": lerobot_sha,
                    "sequence_dhash64": f"{decoded_dhash:016x}",
                    "lerobot_dhash64": f"{lerobot_dhash:016x}",
                }
            )

    return {
        "frames_compared": len(decoded_sequence),
        "exact_decoded_pixel_matches": exact_pixel_matches,
        "psnr_db_min": min(psnr_values),
        "psnr_db_median": median(psnr_values),
        "psnr_db_p05": percentile_nearest_rank(psnr_values, 0.05),
        "pixel_mae_0_255_median": median(mae_values),
        "dhash_hamming_median": median(hamming_values),
        "dhash_hamming_p95": percentile_nearest_rank(hamming_values, 0.95),
        "dhash_hamming_max": max(hamming_values),
        "sampled_decoded_hashes": sampled_hashes,
    }


def select_exterior_assignment(
    pair_metrics: dict[tuple[str, str], dict[str, object]],
) -> tuple[str, dict[str, str], dict[str, float]]:
    first = "exterior_image_1_left"
    second = "exterior_image_2_left"
    scores = {
        "identity": float(pair_metrics[(first, first)]["psnr_db_median"])
        + float(pair_metrics[(second, second)]["psnr_db_median"]),
        "swapped": float(pair_metrics[(first, second)]["psnr_db_median"])
        + float(pair_metrics[(second, first)]["psnr_db_median"]),
    }
    selected = "identity" if scores["identity"] >= scores["swapped"] else "swapped"
    mapping = (
        {first: first, second: second}
        if selected == "identity"
        else {first: second, second: first}
    )
    return selected, mapping, scores


def main() -> int:
    args = _parser().parse_args()
    mapping_root = args.mapping_root.expanduser().resolve()
    episodes_json = args.episodes_json.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DROID lineage audit: {output}")
    if not mapping_root.is_dir() or not episodes_json.is_file() or not video_root.is_dir():
        raise ValueError("mapping root, episodes JSON, and video root must exist")
    if not ffmpeg.is_file():
        raise ValueError(f"ffmpeg executable must exist: {ffmpeg}")
    if len(set(args.episodes)) != len(args.episodes):
        raise ValueError("episode indices must be unique")
    if (
        args.min_p05_psnr_db <= 0
        or args.max_p95_dhash_hamming < 0
        or args.source_fps <= 0
    ):
        raise ValueError("alignment thresholds must be positive/non-negative")

    output.mkdir(parents=True)
    (output / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n"
    )
    _write_json(
        output / "config.json",
        {
            "mapping_root": str(mapping_root),
            "episodes_json": str(episodes_json),
            "video_root": str(video_root),
            "ffmpeg": str(ffmpeg),
            "ffmpeg_sha256": _sha256(ffmpeg),
            "source_fps": args.source_fps,
            "episodes": args.episodes,
            "min_p05_psnr_db": args.min_p05_psnr_db,
            "max_p95_dhash_hamming": args.max_p95_dhash_hamming,
            "seed": args.seed,
            "seed_use": "recorded for experiment reproducibility; audit is deterministic",
            "git_commit": args.git_commit,
            "git_branch": args.git_branch,
        },
    )
    _write_json(
        output / "git-state.json",
        _git_state(args.git_commit, args.git_branch),
    )

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        from tensorflow.train import SequenceExample
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow protobuf bindings are required only for this offline decoder"
        ) from error
    import cv2
    import numpy as np

    _write_json(output / "package-versions.json", _package_versions())
    log_path = output / "audit.log"
    log_path.write_text(
        f"{datetime.now(timezone.utc).isoformat()} starting real DROID lineage audit\n"
    )

    mapping_manifest_path = mapping_root / "manifest.json"
    if not mapping_manifest_path.is_file():
        raise ValueError("mapping root must contain manifest.json")
    mapping_manifest = json.loads(mapping_manifest_path.read_text())
    mapping_rows = mapping_manifest.get("mappings")
    if not isinstance(mapping_rows, dict):
        raise ValueError("mapping manifest must contain a mappings object")
    episodes = _episode_index(json.loads(episodes_json.read_text()))

    source_videos = {}
    for stream in STREAMS:
        path = video_root / f"observation.images.{stream}" / "chunk-000" / "file-000.mp4"
        if not path.is_file():
            raise ValueError(f"missing LeRobot source video: {path}")
        source_videos[stream] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    episode_results = []
    all_stream_gates = []
    for episode in args.episodes:
        if episode not in episodes:
            raise ValueError(f"episode {episode} is absent from episodes JSON")
        mapping = mapping_rows.get(str(episode))
        if not isinstance(mapping, dict):
            raise ValueError(f"episode {episode} is absent from mapping manifest")
        payload_path = mapping_root / f"episode-{episode:03d}-sequence-example.bin"
        if not payload_path.is_file():
            raise ValueError(f"missing SequenceExample payload: {payload_path}")
        payload_sha256 = _sha256(payload_path)
        if payload_sha256 != mapping.get("payload_sha256"):
            raise ValueError(f"SequenceExample SHA-256 mismatch for episode {episode}")

        example = _decode_example(SequenceExample, payload_path)
        raw_paths = derive_raw_paths(
            _single_bytes_feature(example, "episode_metadata/file_path").decode("utf-8"),
            _single_bytes_feature(
                example, "episode_metadata/recording_folderpath"
            ).decode("utf-8"),
        )
        episode_row = episodes[episode]
        length = episode_row.get("length")
        start_frame = episode_row.get("dataset_from_index")
        end_frame = episode_row.get("dataset_to_index")
        tasks = episode_row.get("tasks")
        if (
            not isinstance(length, int)
            or not isinstance(start_frame, int)
            or not isinstance(end_frame, int)
            or end_frame - start_frame != length
        ):
            raise ValueError(f"invalid frame interval for episode {episode}")
        if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], str):
            raise ValueError(f"episode {episode} must have exactly one string task")
        instructions = _bytes_feature(example, "steps/language_instruction")
        instruction_values = {value.decode("utf-8") for value in instructions}
        task_match = instruction_values == {tasks[0]} and len(instructions) == length

        sequence_frames = {}
        for stream in STREAMS:
            encoded_frames = _bytes_feature(
                example, f"steps/observation/{stream}"
            )
            if len(encoded_frames) != length:
                raise ValueError(
                    f"SequenceExample frame count mismatch for episode {episode}, "
                    f"stream {stream}: {len(encoded_frames)} != {length}"
                )
            sequence_frames[stream] = _decode_sequence_frames(
                cv2, np, encoded_frames
            )
        height, width, _ = sequence_frames[STREAMS[0]][0].shape
        lerobot_frames = {}
        for stream in STREAMS:
            lerobot_frames[stream] = _decode_video_segment(
                ffmpeg=ffmpeg,
                np=np,
                video_path=Path(source_videos[stream]["path"]),
                start_frame=start_frame,
                frame_count=length,
                source_fps=args.source_fps,
                height=height,
                width=width,
            )

        pair_metrics = {}
        pair_metrics[("wrist_image_left", "wrist_image_left")] = _alignment_metrics(
            cv2=cv2,
            np=np,
            decoded_sequence=sequence_frames["wrist_image_left"],
            lerobot_frames=lerobot_frames["wrist_image_left"],
        )
        exterior_streams = STREAMS[1:]
        for sequence_stream in exterior_streams:
            for lerobot_stream in exterior_streams:
                pair_metrics[(sequence_stream, lerobot_stream)] = _alignment_metrics(
                    cv2=cv2,
                    np=np,
                    decoded_sequence=sequence_frames[sequence_stream],
                    lerobot_frames=lerobot_frames[lerobot_stream],
                )
        assignment, exterior_mapping, assignment_scores = select_exterior_assignment(
            pair_metrics
        )
        selected_mapping = {
            "wrist_image_left": "wrist_image_left",
            **exterior_mapping,
        }
        stream_results = {}
        for sequence_stream, lerobot_stream in selected_mapping.items():
            metrics = dict(pair_metrics[(sequence_stream, lerobot_stream)])
            alignment_gate = (
                metrics["psnr_db_p05"] >= args.min_p05_psnr_db
                and metrics["dhash_hamming_p95"] <= args.max_p95_dhash_hamming
            )
            metrics.update(
                {
                    "sequence_stream": sequence_stream,
                    "lerobot_stream": lerobot_stream,
                    "alignment_gate": alignment_gate,
                }
            )
            all_stream_gates.append(alignment_gate)
            stream_results[sequence_stream] = metrics

        pairwise_summary = {}
        for (sequence_stream, lerobot_stream), metrics in pair_metrics.items():
            pairwise_summary[f"{sequence_stream}->{lerobot_stream}"] = {
                "psnr_db_p05": metrics["psnr_db_p05"],
                "psnr_db_median": metrics["psnr_db_median"],
                "dhash_hamming_p95": metrics["dhash_hamming_p95"],
            }

        episode_results.append(
            {
                "episode_index": episode,
                "payload": str(payload_path),
                "payload_sha256": payload_sha256,
                "shard": mapping.get("shard"),
                "record_offset": mapping.get("record_offset"),
                "task": tasks[0],
                "task_match": task_match,
                "frame_interval": {
                    "dataset_from_index": start_frame,
                    "dataset_to_index": end_frame,
                    "length": length,
                },
                "frame_count_match": True,
                "camera_assignment": {
                    "policy": (
                        "wrist fixed; exterior one-to-one permutation maximizing "
                        "summed median decoded-frame PSNR"
                    ),
                    "selected": assignment,
                    "sequence_to_lerobot_stream": selected_mapping,
                    "median_psnr_sum_db": assignment_scores,
                },
                "pairwise_alignment_summary": pairwise_summary,
                **raw_paths,
                "streams": stream_results,
            }
        )

    gates = {
        "all_requested_episodes_decoded": len(episode_results) == len(args.episodes),
        "raw_paths_structurally_valid": len(episode_results) == len(args.episodes),
        "tasks_match": all(row["task_match"] for row in episode_results),
        "frame_counts_match": all(row["frame_count_match"] for row in episode_results),
        "all_three_camera_streams_pixel_aligned": all(all_stream_gates),
    }
    accepted = all(gates.values())
    completed_at = datetime.now(timezone.utc).isoformat()
    result = {
        "schema_version": "1.0.0",
        "completed_at": completed_at,
        "status": "WORKING" if accepted else "PARTIAL",
        "accepted": accepted,
        "method": "structured_sequenceexample_path_and_pixel_lineage_audit",
        "mapping_manifest": str(mapping_manifest_path),
        "mapping_manifest_sha256": _sha256(mapping_manifest_path),
        "episodes_json": str(episodes_json),
        "episodes_json_sha256": _sha256(episodes_json),
        "thresholds": {
            "minimum_frame_psnr_p05_db": args.min_p05_psnr_db,
            "maximum_frame_dhash_hamming_p95": args.max_p95_dhash_hamming,
        },
        "camera_assignment_policy": (
            "Wrist is fixed by name. The two exterior streams use the better of "
            "the only two one-to-one assignments by summed median PSNR; all "
            "selected streams must still pass the frozen PSNR and dHash gates."
        ),
        "gates": gates,
        "source_videos": source_videos,
        "episodes": episode_results,
        "rights_boundary": (
            "Pixel lineage is an internal technical audit. The unresolved official "
            "raw DROID license still blocks claim-eligible raw-data training and "
            "redistribution."
        ),
        "claim_boundary": (
            "The audit can establish raw-path-to-LeRobot lineage for episodes "
            "21/60/77. It does not establish novel-view generation quality."
        ),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "seed": args.seed,
    }
    _write_json(output / "manifest.json", result)
    with log_path.open("a") as handle:
        handle.write(f"{completed_at} accepted={accepted}\n")
    print(json.dumps({"output": str(output), "accepted": accepted, "gates": gates}))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
