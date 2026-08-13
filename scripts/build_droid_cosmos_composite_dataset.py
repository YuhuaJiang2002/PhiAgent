#!/usr/bin/env python3
"""Convert synchronized DROID views to Cosmos Predict2's native DROID layout.

The GR00T-Dreams-DROID checkpoint was trained on a 2x2 composite: exterior
left, exterior right, wrist/ego, and an inactive black tile.  This builder
keeps that convention exact and records the disclosure boundary: the composite
first frame and text are REAL CONDITIONS; continuation frames are targets only
during training/evaluation and must be generated at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TILE_WIDTH = 384
TILE_HEIGHT = 216
WIDTH = TILE_WIDTH * 2
HEIGHT = TILE_HEIGHT * 2
FPS = 16
FRAMES = 97
TRAINING_WINDOW_FRAMES = 93
ENCODE_CRF = 20
VIEW_LAYOUT = {
    "top_left": "exterior_1",
    "top_right": "exterior_2",
    "bottom_left": "wrist",
    "bottom_right": "inactive_black",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def normalize_task(task: str) -> str:
    """Normalize task text without changing its semantics."""
    normalized = " ".join(task.strip().split()).rstrip(".")
    if not normalized:
        return "perform the manipulation task"
    return normalized[0].lower() + normalized[1:]


def task_condition_kind(task: str) -> str:
    """Disclose whether text came from DROID or a fixed missing-label fallback."""
    return "REAL DATASET ANNOTATION" if task.strip().rstrip(".") else "FIXED NEUTRAL FALLBACK"


def validate_source_video_contract(contract: dict[str, Any]) -> None:
    """Reject temporally upsampled source clips that duplicate adjacent frames."""
    source_video_contract = contract.get("video_contract", {})
    source_fps = int(source_video_contract.get("fps", 0))
    source_frames = int(source_video_contract.get("frames", 0))
    if source_fps != FPS or source_frames < FRAMES:
        raise ValueError(
            "source dataset must be sampled directly at 16 fps with at least 97 frames; "
            f"got fps={source_fps}, frames={source_frames}"
        )


def droid_multiview_prompt(task: str) -> str:
    """Return the prompt form used by the official DROID checkpoint example."""
    action = normalize_task(task)
    return (
        f"A multi-view video shows that a robot {action}. The video is split into four views: "
        "The top-left view shows the robotic arm from the left side, the top-right view shows "
        "it from the right side, the bottom-left view shows a first-person perspective from "
        "the robot's end-effector (gripper), and the bottom-right view is a black screen "
        f"(inactive view). The robot {action}"
    )


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


def _compose(ffmpeg: Path, sources: dict[str, Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    duration = FRAMES / FPS
    filters = []
    for index in range(3):
        filters.append(
            f"[{index}:v]fps={FPS},scale={TILE_WIDTH}:{TILE_HEIGHT}:flags=lanczos,"
            f"setsar=1[v{index}]"
        )
    filters.extend(
        [
            f"color=c=black:s={TILE_WIDTH}x{TILE_HEIGHT}:r={FPS}:d={duration}[v3]",
            "[v0][v1][v2][v3]xstack=inputs=4:layout="
            f"0_0|{TILE_WIDTH}_0|0_{TILE_HEIGHT}|{TILE_WIDTH}_{TILE_HEIGHT},"
            "format=yuv420p[out]",
        ]
    )
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y"]
    for name in ("exterior_1", "exterior_2", "wrist"):
        command.extend(["-i", str(sources[name])])
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
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
    )
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg composition failed: {completed.stderr}")


def _extract_first_frame(ffmpeg: Path, video: Path, destination: Path) -> None:
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
        raise RuntimeError(f"first-frame extraction failed: {completed.stderr}")


def main() -> int:
    args = _parser().parse_args()
    source_contract = _require_file(args.source_contract, "source contract")
    ffmpeg = _require_file(args.ffmpeg, "ffmpeg")
    ffprobe = _require_file(args.ffprobe, "ffprobe")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)

    contract = json.loads(source_contract.read_text())
    source_root = source_contract.parent
    validate_source_video_contract(contract)
    source_records = contract.get("records")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("source contract must contain records")
    if contract.get("leakage_checks", {}).get("final_holdout_used_for_training") is not False:
        raise ValueError("source contract does not attest final-holdout isolation")

    command = [sys.executable, *sys.argv]
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    _write_json(
        output / "input-config.json",
        {
            **vars(args),
            "source_contract": str(source_contract),
            "source_contract_sha256": _sha256(source_contract),
            "output_dir": str(output),
        },
    )

    records: list[dict[str, Any]] = []
    expected_probe = {
        "width": WIDTH,
        "height": HEIGHT,
        "fps": float(FPS),
        "frames": FRAMES,
    }
    for source_record in source_records:
        sample_id = str(source_record["sample_id"])
        split = str(source_record["split"])
        cameras = source_record["cameras"]
        sources = {
            name: _require_file(source_root / cameras[name]["video"], f"{sample_id} {name}")
            for name in ("exterior_1", "exterior_2", "wrist")
        }
        video = output / split / "videos" / f"{sample_id}.mp4"
        condition = output / split / "conditions" / f"{sample_id}-real-condition.png"
        metadata = output / split / "metas" / f"{sample_id}.txt"
        raw_task = str(source_record["task_text_real_condition"])
        prompt = droid_multiview_prompt(raw_task)
        _compose(ffmpeg, sources, video)
        _extract_first_frame(ffmpeg, video, condition)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(prompt + "\n")
        probe = _probe(ffprobe, video)
        if any(probe[key] != value for key, value in expected_probe.items()):
            raise RuntimeError(f"invalid composite contract for {video}: {probe}")
        records.append(
            {
                "sample_id": sample_id,
                "episode_index": int(source_record["episode_index"]),
                "split": split,
                "training_use": split == "train",
                "prompt": prompt,
                "raw_task_text": raw_task,
                "task_text_condition_kind": task_condition_kind(raw_task),
                "real_composite_first_frame_condition": str(condition.relative_to(output)),
                "real_composite_first_frame_condition_sha256": _sha256(condition),
                "real_multiview_target_video": str(video.relative_to(output)),
                "real_multiview_target_video_sha256": _sha256(video),
                "metadata": str(metadata.relative_to(output)),
            }
        )

    split_counts = {
        split: sum(row["split"] == split for row in records)
        for split in ("train", "legacy_dev", "validation", "final_holdout")
    }
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING",
        "method": "cosmos_predict2_gr00t_dreams_droid_native_2x2_layout",
        "model": "Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID",
        "source_contract": str(source_contract),
        "source_contract_sha256": _sha256(source_contract),
        "layout": VIEW_LAYOUT,
        "video_contract": {
            "width": WIDTH,
            "height": HEIGHT,
            "tile_width": TILE_WIDTH,
            "tile_height": TILE_HEIGHT,
            "fps": FPS,
            "frames": FRAMES,
            "training_window_frames": TRAINING_WINDOW_FRAMES,
            "time_resampling": (
                "15 fps source decoded and nearest-time sampled once to 16 fps before composition; "
                "no second-stage 8-to-16 frame duplication"
            ),
        },
        "conditioning_contract": {
            "real_conditions": ["2x2 composite first frame", "task text when supplied by DROID metadata"],
            "missing_task_policy": "Use the fixed neutral prompt 'perform the manipulation task' and label it FIXED NEUTRAL FALLBACK, never REAL CONDITION.",
            "our_generated_video": "all 2x2 continuation frames after the real first frame",
            "training_only_real_target": "frames 2-97 of synchronized real clips; training samples a 93-frame window",
            "inference_leakage_forbidden": "real target continuation frames must not be passed to the model",
            "demo_disclosure": "Label first frame and text REAL CONDITIONS; label every predicted continuation OUR GENERATED VIDEO.",
        },
        "split_counts": split_counts,
        "leakage_checks": {
            "final_holdout_used_for_training": False,
            "final_holdout_used_for_checkpoint_selection": False,
            "legacy_dev_previously_inspected": True,
        },
        "coordinate_frames": contract.get("source", {}).get("streams", {}),
        "camera_calibration": contract.get("calibration_contract"),
        "records": records,
        "git_state": {
            "commit": args.git_commit or "unresolved",
            "branch": args.git_branch,
            "working_tree_status": "dirty",
            "script_sha256": _sha256(Path(__file__)),
        },
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "command": command,
    }
    _write_json(output / "dataset-contract.json", result)
    print(json.dumps({"output": str(output), "split_counts": split_counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
