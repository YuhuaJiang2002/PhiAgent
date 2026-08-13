#!/usr/bin/env python3
"""Build true wrist-only to third-person Cosmos3 I2V SFT sequences.

Each derived 97-frame sequence has exactly one real wrist-camera frame at
frame 0 and synchronized real frames from one named exterior camera at frames
1--96.  No exterior pixel is present in the visual condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


WIDTH = 768
HEIGHT = 432
TILE_WIDTH = 384
TILE_HEIGHT = 216
FPS = 16
FRAMES = 97
TARGET_VIEWS = {
    "exterior_1": (0, 0, "left-side third-person exterior camera"),
    "exterior_2": (384, 0, "right-side third-person exterior camera"),
}
WRIST_TILE = (0, 216)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path(shutil.which("ffprobe") or "ffprobe"))
    parser.add_argument("--target-view", action="append", choices=sorted(TARGET_VIEWS))
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--git-commit")
    parser.add_argument("--git-branch")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"{label} is missing or empty: {resolved}")
    return resolved


def validate_source_contract(contract: dict[str, Any]) -> None:
    video = contract.get("video_contract", {})
    expected = {"width": WIDTH, "height": HEIGHT, "fps": FPS, "frames": FRAMES}
    actual = {key: int(video.get(key, -1)) for key in expected}
    if actual != expected:
        raise ValueError(f"unexpected DROID composite video contract: {actual}")
    layout = contract.get("layout", {})
    if layout != {
        "top_left": "exterior_1",
        "top_right": "exterior_2",
        "bottom_left": "wrist",
        "bottom_right": "inactive_black",
    }:
        raise ValueError(f"unexpected or unnamed composite layout: {layout}")
    leakage = contract.get("leakage_checks", {})
    if leakage.get("final_holdout_used_for_training") is not False:
        raise ValueError("source contract does not isolate final holdout")
    records = contract.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("source contract contains no records")
    train_episodes = {
        int(row["episode_index"]) for row in records if row.get("split") == "train"
    }
    heldout_episodes = {
        int(row["episode_index"])
        for row in records
        if row.get("split") in {"validation", "final_holdout"}
    }
    if train_episodes & heldout_episodes:
        raise ValueError("episode leakage across training and heldout splits")


def view_switch_filter(target_view: str) -> str:
    if target_view not in TARGET_VIEWS:
        raise ValueError(f"unsupported target view: {target_view}")
    target_x, target_y, _ = TARGET_VIEWS[target_view]
    wrist_x, wrist_y = WRIST_TILE
    return (
        f"[0:v]crop={TILE_WIDTH}:{TILE_HEIGHT}:{wrist_x}:{wrist_y},"
        f"scale={WIDTH}:{HEIGHT}:flags=lanczos,select='eq(n\\,0)',"
        f"setpts=N/({FPS}*TB)[wrist];"
        f"[0:v]crop={TILE_WIDTH}:{TILE_HEIGHT}:{target_x}:{target_y},"
        f"scale={WIDTH}:{HEIGHT}:flags=lanczos,select='gte(n\\,1)',"
        f"setpts=N/({FPS}*TB)[third];"
        "[wrist][third]concat=n=2:v=1:a=0,format=yuv420p[outv]"
    )


def build_video_command(
    ffmpeg: Path, source: Path, output: Path, target_view: str, crf: int
) -> list[str]:
    if not 0 <= crf <= 30:
        raise ValueError("CRF must be between 0 and 30")
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-filter_complex",
        view_switch_filter(target_view),
        "-map",
        "[outv]",
        "-an",
        "-r",
        str(FPS),
        "-frames:v",
        str(FRAMES),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def _probe(ffprobe: Path, video: Path) -> dict[str, Any]:
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
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected exactly one video stream: {video}")
    return streams[0]


def validate_probe(probe: dict[str, Any]) -> None:
    actual = {
        "width": int(probe.get("width", -1)),
        "height": int(probe.get("height", -1)),
        "fps": str(probe.get("avg_frame_rate")),
        "frames": int(probe.get("nb_read_frames", -1)),
    }
    expected = {"width": WIDTH, "height": HEIGHT, "fps": "16/1", "frames": FRAMES}
    if actual != expected:
        raise ValueError(f"derived wrist-to-third video contract mismatch: {actual}")


def _extract_first_frame(ffmpeg: Path, video: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-y",
            str(output),
        ],
        check=True,
    )


def structured_caption(record: dict[str, Any], target_view: str) -> dict[str, Any]:
    if target_view not in TARGET_VIEWS:
        raise ValueError(f"unsupported target view: {target_view}")
    _, _, view_description = TARGET_VIEWS[target_view]
    task = " ".join(str(record["raw_task_text"]).strip().split()).rstrip(".")
    if not task:
        task = "perform the manipulation task"
    return {
        "subjects": [
            {
                "description": (
                    "The same DROID robot, gripper, and manipulated object seen in "
                    "the real first-person wrist-camera condition"
                ),
                "action": task,
                "state_changes": (
                    "Preserve exact robot and object identity while continuing the "
                    f"action from the synchronized {view_description}"
                ),
            }
        ],
        "background_setting": (
            "The same real DROID workspace and fixtures, reconstructed from the named "
            "third-person camera without scene cuts, new objects, or removed objects"
        ),
        "cinematography": {
            "camera_motion": "Static",
            "framing": f"Full-frame {view_description}",
            "camera_angle": target_view,
            "view_transition": (
                "Frame 1 is the real wrist first-person condition only; every future "
                f"frame is the requested {target_view} third-person view"
            ),
            "focus": "Sharp robot embodiment, gripper, and manipulated object",
        },
        "actions": [{"time": "0:00-0:06", "description": task}],
        "temporal_caption": (
            f"From the real wrist-camera first-person condition, generate {target_view} "
            f"third-person video of the same robot as it performs: {task}."
        ),
        "style_medium": "Photoreal live-action robot manipulation footage",
        "resolution": {"W": WIDTH, "H": HEIGHT},
        "aspect_ratio": "16,9",
        "duration": "6.0625s",
        "fps": FPS,
    }


def sft_record(
    source_record: dict[str, Any], sample_id: str, video_relative: str, target_view: str
) -> dict[str, Any]:
    caption = structured_caption(source_record, target_view)
    return {
        "uuid": sample_id,
        "duration": FRAMES / FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "vision_path": video_relative,
        "t2w_windows": [
            {
                "start_frame": 0,
                "end_frame": FRAMES - 1,
                "temporal_interval": 1,
                "caption_json": caption,
                "caption": caption["temporal_caption"],
            }
        ],
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = _parser().parse_args()
    contract_path = _require_file(args.composite_contract, "composite dataset contract")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite wrist-only SFT dataset: {output}")
    output.mkdir(parents=True)
    ffmpeg = _require_file(args.ffmpeg, "ffmpeg")
    ffprobe = _require_file(args.ffprobe, "ffprobe")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_source_contract(contract)
    source_root = contract_path.parent
    target_views = tuple(args.target_view or sorted(TARGET_VIEWS))
    if len(set(target_views)) != len(target_views):
        raise ValueError("target views must not contain duplicates")

    commands_log = output / "ffmpeg-commands.txt"
    train_rows: list[dict[str, Any]] = []
    validation_samples: list[dict[str, Any]] = []
    derived_records: list[dict[str, Any]] = []
    split_counts = {"train": 0, "validation": 0}
    with commands_log.open("w", encoding="utf-8") as command_handle:
        for record in contract["records"]:
            split = str(record["split"])
            if split not in split_counts:
                continue
            source_id = str(record["sample_id"])
            source_video = _require_file(
                source_root / record["real_multiview_target_video"],
                f"{source_id} real multiview source",
            )
            if _sha256(source_video) != record["real_multiview_target_video_sha256"]:
                raise ValueError(f"source-video hash mismatch: {source_id}")
            for target_view in target_views:
                sample_id = f"{source_id}-wrist-to-{target_view.replace('_', '-')}"
                if split == "train":
                    video = output / "train/videos" / f"{sample_id}.mp4"
                else:
                    video = output / "val/targets" / f"{sample_id}.mp4"
                video.parent.mkdir(parents=True, exist_ok=True)
                command = build_video_command(
                    ffmpeg, source_video, video, target_view, args.crf
                )
                command_handle.write(shlex.join(command) + "\n")
                subprocess.run(command, check=True)
                probe = _probe(ffprobe, video)
                validate_probe(probe)
                condition = (
                    output / "train/conditions" / f"{sample_id}.png"
                    if split == "train"
                    else output / "val/images" / f"{sample_id}.png"
                )
                _extract_first_frame(ffmpeg, video, condition)
                video_sha = _sha256(video)
                condition_sha = _sha256(condition)
                caption = structured_caption(record, target_view)
                derived = {
                    "sample_id": sample_id,
                    "source_sample_id": source_id,
                    "episode_index": int(record["episode_index"]),
                    "split": split,
                    "training_use": split == "train",
                    "target_view": target_view,
                    "condition": str(condition.relative_to(output)),
                    "condition_sha256": condition_sha,
                    "condition_label": "REAL CONDITION / FIRST-PERSON WRIST ONLY",
                    "condition_pixel_lineage": (
                        "decoded real bottom-left wrist tile at source frame 1, resized "
                        "to 768x432; no exterior pixel is present"
                    ),
                    "target": str(video.relative_to(output)),
                    "target_sha256": video_sha,
                    "target_label": "REAL TRAINING TARGET" if split == "train" else "WITHHELD REAL TARGET / THIRD-PERSON EVALUATION ONLY",
                    "frame_semantics": {
                        "frame_0": "real first-person wrist condition",
                        "frames_1_96": f"synchronized real {target_view} third-person future",
                    },
                    "source_composite": str(source_video),
                    "source_composite_sha256": record["real_multiview_target_video_sha256"],
                    "probe": probe,
                }
                derived_records.append(derived)
                if split == "train":
                    train_rows.append(
                        sft_record(
                            record,
                            sample_id,
                            f"videos/{sample_id}.mp4",
                            target_view,
                        )
                    )
                else:
                    prompt = output / "val/prompts" / f"{sample_id}.json.txt"
                    prompt.parent.mkdir(parents=True, exist_ok=True)
                    prompt.write_text(json.dumps(caption, sort_keys=True) + "\n")
                    spec = output / "val/inference_prompt_i2v" / f"{sample_id}.json"
                    _write_json(
                        spec,
                        {
                            "name": sample_id,
                            "model_mode": "image2video",
                            "prompt": json.dumps(caption, sort_keys=True),
                            "vision_path": f"../images/{sample_id}.png",
                            "resolution": "480",
                            "aspect_ratio": "16,9",
                            "num_frames": 93,
                            "fps": FPS,
                            "num_steps": 35,
                            "guidance": 6.0,
                            "shift": 10.0,
                            "seed": int(contract.get("seed", 20260812)),
                            "enable_sound": False,
                        },
                    )
                    validation_samples.append(
                        {
                            "sample_id": sample_id,
                            "source_sample_id": source_id,
                            "episode_index": int(record["episode_index"]),
                            "target_view": target_view,
                            "condition": str(condition.relative_to(output)),
                            "condition_sha256": condition_sha,
                            "condition_label": "REAL CONDITION / FIRST-PERSON WRIST ONLY",
                            "prompt": str(prompt.relative_to(output)),
                            "prompt_label": record["task_text_condition_kind"],
                            "withheld_target": str(video.relative_to(output)),
                            "withheld_target_sha256": video_sha,
                            "withheld_target_label": "WITHHELD REAL TARGET / THIRD-PERSON EVALUATION ONLY",
                            "inference_spec": str(spec.relative_to(output)),
                            "generated_continuation_label": "OUR GENERATED VIDEO / THIRD-PERSON",
                        }
                    )
                split_counts[split] += 1

    train_jsonl = output / "train/video_dataset_file.jsonl"
    _write_jsonl(train_jsonl, train_rows)
    summary = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "WORKING",
        "method": "cosmos3_nano_droid_wrist_only_to_exterior_i2v_sft_dataset",
        "claim_scope": "true wrist-only first-person condition to named third-person exterior continuation",
        "source_contract": str(contract_path),
        "source_contract_sha256": _sha256(contract_path),
        "target_views": list(target_views),
        "split_counts": split_counts,
        "training": {
            "jsonl": str(train_jsonl.relative_to(output)),
            "jsonl_sha256": _sha256(train_jsonl),
            "conditioning_distribution_required": {"i2v_first_frame": 1.0},
            "condition": "one resized real wrist-camera frame only",
            "target_frames": "synchronized real named exterior-camera frames 2-97",
        },
        "validation_samples": validation_samples,
        "records": derived_records,
        "exclusions": {
            "final_holdout": "excluded from training, validation, and checkpoint selection",
            "legacy_dev": "excluded from training and validation",
        },
        "leakage_checks": {
            "episode_disjoint_train_validation": True,
            "final_holdout_used_for_training": False,
            "final_holdout_used_for_checkpoint_selection": False,
            "validation_future_frames_are_model_inputs": False,
            "condition_contains_exterior_pixels": False,
            "condition_contains_real_wrist_pixels_only": True,
        },
        "transform": {
            "source_frame": "canonical_droid_2x2_composite_pixel_frame",
            "condition_frame": "resized_wrist_camera_pixel_frame",
            "target_frame": "resized_named_exterior_camera_pixel_frame",
            "wrist_crop_xywh": [0, 216, 384, 216],
            "exterior_1_crop_xywh": [0, 0, 384, 216],
            "exterior_2_crop_xywh": [384, 0, 384, 216],
            "output_wh": [WIDTH, HEIGHT],
            "resize_filter": "lanczos",
        },
        "encoding": {"codec": "libx264", "crf": args.crf, "pixel_format": "yuv420p"},
        "command": [sys.executable, *sys.argv],
        "command_shell": shlex.join([sys.executable, *sys.argv]),
        "ffmpeg_commands": str(commands_log.relative_to(output)),
        "ffmpeg_sha256": _sha256(ffmpeg),
        "ffprobe_sha256": _sha256(ffprobe),
        "git": {"commit": args.git_commit or "unresolved", "branch": args.git_branch},
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
    }
    _write_json(output / "dataset-contract.json", summary)
    (output / "command.txt").write_text(summary["command_shell"] + "\n")
    print(json.dumps({"output": str(output), "split_counts": split_counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
